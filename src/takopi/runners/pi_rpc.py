"""Long-lived Pi RPC transport, separate from the one-shot PiRunner."""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio
import msgspec

from ..model import ResumeToken, StartedEvent, TakopiEvent
from ..schemas import pi as pi_schema
from .pi import ENGINE, PiStreamState, translate_pi_event

_PI_COMMAND = "pi"
_OWNERS: dict[Path, PiRpcClient] = {}
_OWNER_LOCK = asyncio.Lock()
_END = object()


class PiRpcClient:
    def __init__(self, session_path: Path, cwd: Path, args: list[str]) -> None:
        self.session_path = session_path.resolve()
        self.cwd = cwd
        self.args = list(args)
        self._proc: Any | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._events: asyncio.Queue[dict[str, Any] | object] = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._closed = False
        self._error: RuntimeError | None = None

    async def start(self) -> None:
        if self._proc is not None:
            return
        if self._closed:
            raise RuntimeError("Pi RPC client is closed")
        async with _OWNER_LOCK:
            if self.session_path in _OWNERS:
                raise RuntimeError("Pi RPC session already has an owner")
            _OWNERS[self.session_path] = self
        try:
            cmd = [
                _PI_COMMAND,
                *self.args,
                "--mode",
                "rpc",
                "--session",
                str(self.session_path),
            ]
            env = dict(os.environ)
            env.setdefault("NO_COLOR", "1")
            env.setdefault("CI", "1")
            self._proc = await anyio.open_process(
                cmd,
                cwd=self.cwd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
            assert self._proc.stdout is not None and self._proc.stderr is not None
            self._tasks = [
                asyncio.create_task(self._read_stdout()),
                asyncio.create_task(self._drain_stderr()),
            ]
        except BaseException:
            _OWNERS.pop(self.session_path, None)
            raise

    async def request(
        self, type: str, *, timeout: float = 30, **fields: object
    ) -> dict[str, Any]:
        await self.start()
        if self._error is not None:
            raise self._error
        assert self._proc is not None and self._proc.stdin is not None
        ident = str(uuid4())
        future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[ident] = future
        try:
            payload = msgspec.json.encode({"id": ident, "type": type, **fields}) + b"\n"
            async with self._write_lock:
                await self._proc.stdin.send(payload)
            response = await asyncio.wait_for(future, timeout)
            if not response.get("success"):
                raise RuntimeError(str(response.get("error", "Pi RPC request failed")))
            return response
        finally:
            self._pending.pop(ident, None)

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._events.get()
            if item is _END:
                # Allow subsequent consumers to observe termination too.
                self._events.put_nowait(_END)
                if self._error is not None:
                    raise self._error
                return
            if isinstance(item, dict):
                yield item

    async def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        buffer = bytearray()
        try:
            async for chunk in self._proc.stdout:
                buffer.extend(chunk)
                while (index := buffer.find(b"\n")) >= 0:
                    line = bytes(buffer[:index]).removesuffix(b"\r")
                    del buffer[: index + 1]
                    record = msgspec.json.decode(line)
                    if not isinstance(record, dict):
                        raise RuntimeError("Pi RPC emitted a non-object record")
                    if record.get("type") == "response":
                        waiter = self._pending.get(record.get("id"))
                        if waiter is not None and not waiter.done():
                            waiter.set_result(record)
                    else:
                        self._events.put_nowait(record)
            if buffer:
                raise RuntimeError("Pi RPC exited with an incomplete record")
            rc = await self._proc.wait()
            self._fail(RuntimeError(f"Pi RPC process exited (rc={rc})"))
        except asyncio.CancelledError:
            raise
        except (RuntimeError, OSError, ValueError, msgspec.DecodeError) as exc:
            self._fail(RuntimeError(f"Pi RPC stream failed: {exc}"))

    async def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        try:
            async for _ in self._proc.stderr:
                pass  # Diagnostics are never protocol records or exposed to another topic.
        except (OSError, anyio.ClosedResourceError):
            pass

    def _fail(self, error: RuntimeError) -> None:
        if self._error is None:
            self._error = error
            for waiter in self._pending.values():
                if not waiter.done():
                    waiter.set_exception(error)
            self._events.put_nowait(_END)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._fail(RuntimeError("Pi RPC client closed"))
        proc = self._proc
        try:
            if proc is not None:
                if proc.stdin is not None:
                    await proc.stdin.aclose()
                try:
                    with anyio.fail_after(2):
                        await proc.wait()
                except TimeoutError:
                    proc.terminate()
                    try:
                        with anyio.fail_after(2):
                            await proc.wait()
                    except TimeoutError:
                        proc.kill()
                        await proc.wait()
        finally:
            for task in self._tasks:
                task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            _OWNERS.pop(self.session_path, None)


class PiRpcRun:
    """One active run on an exclusively owned RPC session; no implicit abort."""

    def __init__(self, client: PiRpcClient) -> None:
        self.client = client
        self.session_key: str | None = None
        self._active = False

    async def run(
        self, prompt: str, resume: ResumeToken | None
    ) -> AsyncIterator[TakopiEvent]:
        if self._active:
            raise RuntimeError("Pi RPC run already active")
        self._active = True
        try:
            await self.client.start()
            response = await self.client.request("get_state")
            data = response.get("data")
            if (
                not isinstance(data, dict)
                or not isinstance(data.get("sessionId"), str)
                or not isinstance(data.get("sessionFile"), str)
            ):
                raise RuntimeError(
                    "Pi RPC get_state returned no persistent session identity"
                )
            self.session_key = data["sessionId"]
            session_file = str(Path(data["sessionFile"]).resolve())
            if resume is not None:
                if resume.engine != ENGINE:
                    raise RuntimeError("Pi RPC resume engine mismatch")
                if resume.value not in {session_file, self.session_key}:
                    raise RuntimeError(
                        "Pi RPC session mismatch: resolved session differs from resume"
                    )
            # Use the actual resolved file rather than truncating a session ID.
            token = ResumeToken(engine=ENGINE, value=session_file)
            state = PiStreamState(resume=token, has_modern_agent_end=True)
            result = await self.client.request("prompt", message=prompt)
            disposition = result.get("data", {}).get("disposition")
            if disposition != "started":
                raise RuntimeError(f"Pi RPC prompt did not start a run: {disposition}")
            async for record in self.client.events():
                kind = record.get("type")
                if kind == "agent_start" and not state.started:
                    state.started = True
                    yield StartedEvent(
                        engine=ENGINE,
                        resume=token,
                        title="pi",
                        meta={"cwd": str(self.client.cwd)},
                    )
                try:
                    event = pi_schema.decode_event(msgspec.json.encode(record))
                except msgspec.DecodeError:
                    continue  # RPC also emits queue updates and extension records.
                for translated in translate_pi_event(
                    event, title="pi", meta={"cwd": str(self.client.cwd)}, state=state
                ):
                    yield translated
                if kind == "agent_settled":
                    break
            else:
                raise RuntimeError("Pi RPC stopped before agent_settled")
        finally:
            self._active = False

    async def steer(self, text: str) -> str:
        if not self._active:
            raise RuntimeError("Pi RPC run is not active")
        response = await self.client.request("steer", message=text)
        data = response.get("data")
        if not isinstance(data, dict) or data.get("disposition") not in {
            "queued",
            "handled",
        }:
            raise RuntimeError("Pi RPC steer returned invalid disposition")
        return str(data["disposition"])

    async def interrupt(self) -> bool:
        if not self._active:
            return False
        await self.client.request("abort")
        return True
