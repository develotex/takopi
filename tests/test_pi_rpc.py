"""Pi RPC transport against a byte-framed fake child process."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest

from takopi.model import CompletedEvent, ResumeToken, StartedEvent
from takopi.runners.pi import PiRunner
from takopi.runners.pi_rpc import PiRpcClient, PiRpcRun
from takopi.runners.run_options import EngineRunOptions, apply_run_options


@pytest.fixture
def fake_pi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "fake_pi.py"
    script.write_text("""import json, sys, threading, time
lock = threading.Lock()
commands = []
def emit(obj):
    with lock:
        sys.stdout.buffer.write(json.dumps(obj, ensure_ascii=False).encode() + b"\\n")
        sys.stdout.buffer.flush()
def settle(answer):
    time.sleep(.08)
    emit({"type":"agent_end","messages":[{"role":"assistant","content":[{"type":"text","text":"old"}],"stopReason":"error","errorMessage":"retry"}],"willRetry":True})
    time.sleep(.08)
    emit({"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":answer}],"stopReason":"stop"}})
    emit({"type":"agent_end","messages":[],"willRetry":False})
    emit({"type":"agent_settled"})
for line in sys.stdin.buffer:
    cmd = json.loads(line)
    typ = cmd["type"]
    commands.append(typ)
    if typ == "get_commands":
        emit({"type":"response","id":cmd["id"],"command":typ,"success":True,"data":{"commands":commands[:]}})
    elif typ in ("abort", "clear_queue"):
        emit({"type":"response","id":cmd["id"],"command":typ,"success":True})
    elif typ == "get_state":
        emit({"type":"response","id":cmd["id"],"command":typ,"success":True,"data":{"sessionId":"abc12345-uuid","sessionFile":sys.argv[-1],"isStreaming":False}})
    elif typ == "get_args":
        emit({"type":"response","id":cmd["id"],"command":typ,"success":True,"data":{"args":sys.argv[1:]}})
    elif typ == "slow":
        def respond(c):
            time.sleep(.15)
            emit({"type":"response","id":c["id"],"command":"slow","success":True,"data":{"value":"slow"}})
        threading.Thread(target=respond,args=(cmd,),daemon=True).start()
    elif typ == "fast":
        emit({"type":"response","id":cmd["id"],"command":typ,"success":True,"data":{"value":"fast\\u2028value"}})
    elif typ == "hang":
        pass
    elif typ == "fail":
        emit({"type":"response","id":cmd["id"],"command":typ,"success":False,"error":"rejected"})
    elif typ == "exit":
        sys.exit(3)
    elif typ in ("prompt", "steer"):
        emit({"type":"response","id":cmd["id"],"command":typ,"success":True,"data":{"disposition":"started" if typ == "prompt" else "queued"}})
        if typ == "prompt":
            emit({"type":"agent_start"})
            answer = cmd["message"] if cmd["message"] in ("FIRST", "SECOND") else "final\\u2028answer"
            threading.Thread(target=settle,args=(answer,),daemon=True).start()
    sys.stderr.write("diagnostic " * 2000 + "\\n")
    sys.stderr.flush()
""")
    monkeypatch.setattr("takopi.runners.pi_rpc._PI_COMMAND", sys.executable)
    return script


def client(fake_pi: Path, tmp_path: Path) -> PiRpcClient:
    return PiRpcClient(tmp_path / "session.jsonl", tmp_path, ["-u", str(fake_pi)])


@pytest.mark.anyio
async def test_out_of_order_responses_and_unicode_separator(
    fake_pi: Path, tmp_path: Path
) -> None:
    rpc = client(fake_pi, tmp_path)
    try:
        await rpc.start()
        slow = asyncio.create_task(rpc.request("slow"))
        fast = asyncio.create_task(rpc.request("fast"))
        assert (await fast)["data"]["value"] == "fast\u2028value"
        assert (await slow)["data"]["value"] == "slow"
    finally:
        await rpc.close()


@pytest.mark.anyio
async def test_failure_timeout_and_exit_release_waiters(
    fake_pi: Path, tmp_path: Path
) -> None:
    rpc = client(fake_pi, tmp_path)
    try:
        await rpc.start()
        with pytest.raises(RuntimeError, match="rejected"):
            await rpc.request("fail")
        with pytest.raises(TimeoutError):
            await rpc.request("hang", timeout=0.05)
        waiting = asyncio.create_task(rpc.request("hang", timeout=5))
        with pytest.raises(RuntimeError, match="exited"):
            await rpc.request("exit")
        with pytest.raises(RuntimeError, match="exited"):
            await waiting
    finally:
        await rpc.close()


@pytest.mark.anyio
async def test_run_starts_from_state_and_only_completes_at_settled(
    fake_pi: Path, tmp_path: Path
) -> None:
    rpc = client(fake_pi, tmp_path)
    run = PiRpcRun(rpc)
    try:
        stream = run.run("hello", None)
        started = await anext(stream)
        assert isinstance(started, StartedEvent)
        assert started.resume == ResumeToken("pi", str(tmp_path / "session.jsonl"))
        assert run.session_key == "abc12345-uuid"
        assert await run.steer("new constraint") == "queued"
        remaining = [event async for event in stream]
        completed = [event for event in remaining if isinstance(event, CompletedEvent)]
        assert len(completed) == 1
        assert completed[0].answer == "final\u2028answer"
        assert completed[0].resume == started.resume
    finally:
        await rpc.close()


@pytest.mark.anyio
async def test_run_rejects_different_resumed_session(
    fake_pi: Path, tmp_path: Path
) -> None:
    rpc = client(fake_pi, tmp_path)
    run = PiRpcRun(rpc)
    try:
        with pytest.raises(RuntimeError, match="session mismatch"):
            _ = [
                event
                async for event in run.run(
                    "hello", ResumeToken("pi", str(tmp_path / "other.jsonl"))
                )
            ]
    finally:
        await rpc.close()


@pytest.mark.anyio
async def test_rpc_factory_preserves_runner_and_per_run_options(
    fake_pi: Path, tmp_path: Path
) -> None:
    runner = PiRunner(
        extra_args=["-u", str(fake_pi), "--no-extensions"],
        model="default",
        provider="openai",
    )
    with apply_run_options(EngineRunOptions(model="override", reasoning="high")):
        run = runner.rpc_run(tmp_path / "model.jsonl", cwd=tmp_path)
    try:
        args = (await run.client.request("get_args"))["data"]["args"]
        assert args == [
            "--no-extensions",
            "--provider",
            "openai",
            "--model",
            "override",
            "--thinking",
            "high",
            "--mode",
            "rpc",
            "--session",
            str(tmp_path / "model.jsonl"),
        ]
    finally:
        await run.client.close()


@pytest.mark.anyio
async def test_close_terminates_child_and_releases_owner(
    fake_pi: Path, tmp_path: Path
) -> None:
    rpc = client(fake_pi, tmp_path)
    other = client(fake_pi, tmp_path)
    await rpc.start()
    with pytest.raises(RuntimeError, match="owner"):
        await other.start()
    process = rpc._proc
    await rpc.close()
    assert process is not None and process.returncode is not None
    await other.start()
    await other.close()


@pytest.mark.anyio
async def test_rejected_client_close_cannot_release_live_owner(
    fake_pi: Path, tmp_path: Path
) -> None:
    first, rejected, third = (client(fake_pi, tmp_path) for _ in range(3))
    try:
        await first.start()
        with pytest.raises(RuntimeError, match="owner"):
            await rejected.start()
        await rejected.close()
        with pytest.raises(RuntimeError, match="owner"):
            await third.start()
    finally:
        await third.close()
        await first.close()


@pytest.mark.anyio
async def test_close_waits_for_in_progress_start(
    fake_pi: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import takopi.runners.pi_rpc as module

    real_open = module.anyio.open_process
    entered, release = asyncio.Event(), asyncio.Event()

    async def gated_open(command: Any, **kwargs: Any) -> Any:
        entered.set()
        await release.wait()
        return await real_open(command, **kwargs)

    monkeypatch.setattr(module.anyio, "open_process", gated_open)
    rpc = client(fake_pi, tmp_path)
    starting = asyncio.create_task(rpc.start())
    await entered.wait()
    closing = asyncio.create_task(rpc.close())
    try:
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        await starting
        await closing
        assert rpc._proc is not None and rpc._proc.returncode is not None
        replacement = client(fake_pi, tmp_path)
        await replacement.start()
        await replacement.close()
    finally:
        release.set()
        await rpc.close()


@pytest.mark.anyio
async def test_close_reaps_child_inside_cancelled_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "ignore_eof.py"
    script.write_text("import time\ntime.sleep(30)\n")
    rpc = PiRpcClient(tmp_path / "session.jsonl", tmp_path, [str(script)])
    # Actual executable is Python; unlike the normal fake, this process never reads stdin.
    monkeypatch.setattr("takopi.runners.pi_rpc._PI_COMMAND", sys.executable)
    try:
        await rpc.start()
        proc = rpc._proc
        with anyio.CancelScope() as scope:
            scope.cancel()
            await rpc.close()
        assert proc is not None and proc.returncode is not None
        replacement = PiRpcClient(rpc.session_path, tmp_path, [str(script)])
        await replacement.start()
        await replacement.close()
    finally:
        await rpc.close()


@pytest.mark.anyio
async def test_abandoned_observer_does_not_reuse_previous_run_events(
    fake_pi: Path, tmp_path: Path
) -> None:
    rpc = client(fake_pi, tmp_path)
    run = PiRpcRun(rpc)
    try:
        first = run.run("FIRST", None)
        assert isinstance(await anext(first), StartedEvent)
        await first.aclose()
        second = [event async for event in run.run("SECOND", None)]
        completed = [event for event in second if isinstance(event, CompletedEvent)]
        assert len(completed) == 1
        assert completed[0].answer == "SECOND"
    finally:
        await rpc.close()


@pytest.mark.anyio
async def test_interrupt_clears_pending_steer_before_abort(
    fake_pi: Path, tmp_path: Path
) -> None:
    rpc = client(fake_pi, tmp_path)
    run = PiRpcRun(rpc)
    try:
        stream = run.run("hello", None)
        await anext(stream)
        assert await run.steer("queued update") == "queued"
        assert await run.interrupt()
        commands = (await rpc.request("get_commands"))["data"]["commands"]
        assert commands[-3:-1] == ["clear_queue", "abort"]
        with pytest.raises(RuntimeError, match="interrupt"):
            await run.steer("must not queue after cancel")
        await stream.aclose()
    finally:
        await rpc.close()


@pytest.mark.anyio
async def test_interrupt_does_not_wait_for_stalled_clear_queue_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rpc = PiRpcClient(tmp_path / "session.jsonl", tmp_path, [])
    run = PiRpcRun(rpc)
    run._unsettled = True
    writes: list[str] = []
    clear_future: asyncio.Future[dict[str, Any]] = asyncio.Future()

    async def send_request(typ: str, **_fields: object):
        writes.append(typ)
        future = asyncio.Future()
        if typ == "clear_queue":
            future = clear_future
        else:
            future.set_result({"success": True})
        return typ, future

    async def wait_response(ident: str, future: asyncio.Future[dict[str, Any]]):
        return await future

    monkeypatch.setattr(rpc, "send_request", send_request)
    monkeypatch.setattr(rpc, "wait_response", wait_response)
    async with asyncio.timeout(1):
        with pytest.raises(RuntimeError, match="queue clear unconfirmed"):
            await run.interrupt()
    assert writes == ["clear_queue", "abort"]


@pytest.mark.anyio
@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").exists(), reason="Linux process groups"
)
async def test_close_reaps_tool_descendants_before_releasing_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_pid = tmp_path / "child.pid"
    script = tmp_path / "parent.py"
    script.write_text(
        "import pathlib, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))\n"
        "for line in sys.stdin.buffer: pass\n"
    )
    monkeypatch.setattr("takopi.runners.pi_rpc._PI_COMMAND", sys.executable)
    rpc = PiRpcClient(tmp_path / "session.jsonl", tmp_path, ["-u", str(script)])
    await rpc.start()
    try:
        async with asyncio.timeout(2):
            while not child_pid.exists():
                await asyncio.sleep(0.01)
        pid = int(child_pid.read_text())
    finally:
        await rpc.close()
    # Zombie descendants cannot execute or write the session; live descendants can.
    try:
        async with asyncio.timeout(1):
            while (
                Path(f"/proc/{pid}/stat").exists()
                and Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
            ):
                await asyncio.sleep(0.02)
        assert rpc._closed
    finally:
        if (
            Path(f"/proc/{pid}/stat").exists()
            and Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
        ):
            os.kill(pid, signal.SIGKILL)  # Only the synthetic test child.


@pytest.mark.anyio
async def test_started_exposes_active_control(fake_pi: Path, tmp_path: Path) -> None:
    rpc = client(fake_pi, tmp_path)
    run = PiRpcRun(rpc)
    try:
        stream = run.run("hello", None)
        started = await anext(stream)
        assert isinstance(started, StartedEvent)
        assert started.meta is not None and started.meta["control"] is run
        assert await started.meta["control"].interrupt()
        await stream.aclose()
    finally:
        await rpc.close()


@pytest.mark.anyio
async def test_request_timeout_bounds_blocked_stdin_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "blocked.py"
    script.write_text("import time\ntime.sleep(30)\n")
    monkeypatch.setattr("takopi.runners.pi_rpc._PI_COMMAND", sys.executable)
    rpc = PiRpcClient(tmp_path / "session.jsonl", tmp_path, [str(script)])
    try:
        with anyio.fail_after(1):
            with pytest.raises(TimeoutError):
                await rpc.request("blocked", timeout=0.02, message="x" * 2_000_000)
    finally:
        await rpc.close()
