"""Long-lived Pi RPC transport, separate from the one-shot PiRunner."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio
import msgspec

from ..model import ResumeToken, StartedEvent, TakopiEvent
from ..schemas import pi as pi_schema
from .pi import ENGINE, PiStreamState, translate_pi_event

_PI_COMMAND = "pi"
_OWNERS: dict[Path, object] = {}
_OWNER_LOCK = asyncio.Lock()
_LIVE_CLAIM_USERS = 0
_END = object()


@contextmanager
def enable_live_claims() -> Iterator[None]:
    """Enable shared ownership only while a live-enabled Telegram loop is running."""
    global _LIVE_CLAIM_USERS
    _LIVE_CLAIM_USERS += 1
    try:
        yield
    finally:
        _LIVE_CLAIM_USERS -= 1


def live_claims_enabled() -> bool:
    return _LIVE_CLAIM_USERS > 0


async def _claim_owner(path: Path, owner: object) -> None:
    async with _OWNER_LOCK:
        if path in _OWNERS:
            raise RuntimeError("Pi session already has an owner")
        _OWNERS[path] = owner


async def _release_owner(path: Path, owner: object) -> None:
    async with _OWNER_LOCK:
        if _OWNERS.get(path) is owner:
            del _OWNERS[path]


@asynccontextmanager
async def claim_one_shot(path: Path) -> AsyncIterator[None]:
    token = object()
    canonical = path.resolve()
    await _claim_owner(canonical, token)
    try:
        yield
    finally:
        with anyio.CancelScope(shield=True):
            await _release_owner(canonical, token)


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
        self._lifecycle_lock = asyncio.Lock()
        self._closed = False
        self._error: RuntimeError | None = None

    async def start(self) -> None:
        # Serialize creation with close: shutdown must never miss a child in flight.
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Pi RPC client is closed")
            if self._proc is not None:
                return
            await _claim_owner(self.session_path, self)
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
                with anyio.CancelScope(shield=True):
                    self._proc = await anyio.open_process(
                        cmd,
                        cwd=self.cwd,
                        env=env,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=os.name == "posix",
                    )
                    assert (
                        self._proc.stdout is not None and self._proc.stderr is not None
                    )
                    self._tasks = [
                        asyncio.create_task(self._read_stdout()),
                        asyncio.create_task(self._drain_stderr()),
                        asyncio.create_task(self._watch_parent()),
                    ]
            except BaseException:
                if self._proc is not None:
                    with anyio.CancelScope(shield=True):
                        await self._close_claimed_child()
                else:
                    with anyio.CancelScope(shield=True):
                        await _release_owner(self.session_path, self)
                raise

    async def send_request(
        self, type: str, **fields: object
    ) -> tuple[str, asyncio.Future[dict[str, Any]]]:
        """Write an ordered command without waiting for its independent response."""
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
            return ident, future
        except BaseException:
            self._pending.pop(ident, None)
            future.cancel()
            raise

    async def wait_response(
        self, ident: str, future: asyncio.Future[dict[str, Any]]
    ) -> dict[str, Any]:
        try:
            response = await future
            if not response.get("success"):
                raise RuntimeError(str(response.get("error", "Pi RPC request failed")))
            return response
        finally:
            self._pending.pop(ident, None)
            future.cancel()

    async def request(
        self, type: str, *, timeout: float = 30, **fields: object
    ) -> dict[str, Any]:
        # The budget covers process creation, lock contention, stdin and response.
        async with asyncio.timeout(timeout):
            ident, future = await self.send_request(type, **fields)
            return await self.wait_response(ident, future)

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

    async def _watch_parent(self) -> None:
        # Tools can inherit stdout and keep the pipe open after Pi itself dies.
        # Observing only EOF would strand response waiters and event consumers.
        assert self._proc is not None
        while self._proc.returncode is None:
            # asyncio.Process.wait() can itself wait for inherited pipes to
            # close even after its process has exited; returncode does not.
            await asyncio.sleep(0.1)
        self._fail(RuntimeError(f"Pi RPC process exited (rc={self._proc.returncode})"))

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

    async def _close_claimed_child(self) -> None:
        self._fail(RuntimeError("Pi RPC client closed"))
        proc = self._proc
        group_safe = os.name != "posix" or proc is None
        try:
            if proc is not None:
                # Reap the whole dedicated Pi group before waiting on a parent
                # whose inherited pipes may be held by a TERM-ignoring tool.
                if os.name == "posix":
                    with suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGTERM)
                    await anyio.sleep(0.1)
                    with suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    group_safe = True
                if proc.stdin is not None:
                    with suppress(OSError, anyio.ClosedResourceError):
                        await proc.stdin.aclose()
                try:
                    with anyio.fail_after(2):
                        await proc.wait()
                except TimeoutError:
                    # Do not await wait() indefinitely if an escaped child
                    # retained the pipe: the ownership claim must stay held.
                    group_safe = False
                    if proc.returncode is None:
                        proc.kill()
                        with anyio.fail_after(2):
                            await proc.wait()
        finally:
            for task in self._tasks:
                task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            # If reaping fails, retain ownership and permit a second close.
            if group_safe and (proc is None or proc.returncode is not None):
                self._closed = True
                await _release_owner(self.session_path, self)

    async def close(self) -> None:
        # Bridge shutdown runs under AnyIO task-group cancellation.
        with anyio.CancelScope(shield=True):
            async with self._lifecycle_lock:
                if not self._closed:
                    await self._close_claimed_child()


class PiRpcRun:
    """One active run on an exclusively owned RPC session; no implicit abort."""

    def __init__(self, client: PiRpcClient) -> None:
        self.client = client
        self.session_key: str | None = None
        self.expected_session_id: str | None = None
        self._active = False
        self._prompt_started = False
        self._unsettled = False
        self._interrupting = False
        self._command_lock = asyncio.Lock()

    async def _reconcile(self) -> None:
        # The stream observer can disappear while Pi continues its original run.
        # Never submit a new prompt until the old run's boundary is consumed.
        async with asyncio.timeout(30):
            async for record in self.client.events():
                if record.get("type") == "agent_settled":
                    self._unsettled = False
                    return
            raise RuntimeError("Pi RPC stopped before agent_settled")

    async def run(
        self, prompt: str, resume: ResumeToken | None
    ) -> AsyncGenerator[TakopiEvent]:
        if self._active:
            raise RuntimeError("Pi RPC run already active")
        self._active = True
        self._prompt_started = False
        try:
            if self._unsettled:
                await self._reconcile()
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
            session_file = str(Path(data["sessionFile"]).resolve())
            if session_file != str(self.client.session_path.resolve()) or (
                self.expected_session_id is not None
                and data["sessionId"] != self.expected_session_id
            ):
                raise RuntimeError("Pi RPC session identity mismatch before prompt")
            self.session_key = data["sessionId"]
            self.expected_session_id = self.session_key
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
            self._interrupting = False
            self._unsettled = True  # A timed-out prompt may still have reached Pi.
            result = await self.client.request("prompt", message=prompt)
            disposition = result.get("data", {}).get("disposition")
            if disposition != "started":
                raise RuntimeError(f"Pi RPC prompt did not start a run: {disposition}")
            self._prompt_started = True
            meta = {"cwd": str(self.client.cwd), "control": self}
            async for record in self.client.events():
                kind = record.get("type")
                if kind == "agent_start" and not state.started:
                    state.started = True
                    yield StartedEvent(
                        engine=ENGINE,
                        resume=token,
                        title="pi",
                        meta=meta,
                    )
                try:
                    event = pi_schema.decode_event(msgspec.json.encode(record))
                except msgspec.DecodeError:
                    continue  # RPC also emits queue updates and extension records.
                for translated in translate_pi_event(
                    event, title="pi", meta=meta, state=state
                ):
                    yield translated
                if kind == "agent_settled":
                    self._unsettled = False
                    break
            else:
                raise RuntimeError("Pi RPC stopped before agent_settled")
        finally:
            self._prompt_started = False
            self._active = False

    async def steer(self, text: str) -> str:
        # Serialize command writes with interrupt, but never hold the control
        # lock while Pi's response waits behind a long-running tool call.
        async with asyncio.timeout(30):
            async with self._command_lock:
                if self._interrupting:
                    raise RuntimeError("Pi RPC run is being interrupted")
                if not self._active or not self._prompt_started:
                    raise RuntimeError("Pi RPC prompt has not started")
                ident, future = await self.client.send_request("steer", message=text)
            response = await self.client.wait_response(ident, future)
            data = response.get("data")
            if not isinstance(data, dict) or data.get("disposition") not in {
                "queued",
                "handled",
            }:
                raise RuntimeError("Pi RPC steer returned invalid disposition")
            return str(data["disposition"])

    async def interrupt(self) -> bool:
        async with self._command_lock:
            if not self._unsettled:
                return False
            if self._interrupting:
                return True
            self._interrupting = True
            # Preserve stdin command order, but a stalled clear_queue response
            # must not prevent abort from being sent. Do not claim a clean
            # cancellation when the clear acknowledgement was never observed.
            clear_error: Exception | None = None
            try:
                async with asyncio.timeout(0.2):
                    ident, future = await self.client.send_request("clear_queue")
                    await self.client.wait_response(ident, future)
            except (TimeoutError, RuntimeError) as exc:
                clear_error = exc
            await self.client.request("abort")
            if clear_error is not None:
                raise RuntimeError(
                    "Pi RPC queue clear unconfirmed; abort requested"
                ) from clear_error
            return True
