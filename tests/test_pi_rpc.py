"""Pi RPC transport against a byte-framed fake child process."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

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
def emit(obj):
    with lock:
        sys.stdout.buffer.write(json.dumps(obj, ensure_ascii=False).encode() + b"\\n")
        sys.stdout.buffer.flush()
def settle():
    time.sleep(.08)
    emit({"type":"agent_end","messages":[{"role":"assistant","content":[{"type":"text","text":"old"}],"stopReason":"error","errorMessage":"retry"}],"willRetry":True})
    time.sleep(.08)
    emit({"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"final\\u2028answer"}],"stopReason":"stop"}})
    emit({"type":"agent_end","messages":[],"willRetry":False})
    emit({"type":"agent_settled"})
for line in sys.stdin.buffer:
    cmd = json.loads(line)
    typ = cmd["type"]
    if typ == "get_state":
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
            threading.Thread(target=settle,daemon=True).start()
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
