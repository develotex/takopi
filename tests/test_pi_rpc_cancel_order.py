"""Cancel must clear Pi's queue promptly while steer response remains delayed."""

import asyncio
import sys
from pathlib import Path

import anyio
import pytest

from takopi.runners.pi_rpc import PiRpcClient, PiRpcRun


@pytest.mark.anyio
async def test_cancel_clears_queue_before_abort_without_waiting_for_steer_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import takopi.runners.pi_rpc as pi_rpc

    script = tmp_path / "held_steer.py"
    log = tmp_path / "commands.txt"
    script.write_text("""import json, sys, threading, time
log = sys.argv[1]
lock = threading.Lock()
def respond(cmd):
    with lock:
        sys.stdout.write(json.dumps({'type':'response','id':cmd['id'],'command':cmd['type'],'success':True,'data':{'disposition':'queued'}}) + '\\n')
        sys.stdout.flush()
def delayed(cmd):
    time.sleep(1)
    respond(cmd)
for line in sys.stdin:
    cmd = json.loads(line)
    with open(log, 'a') as out:
        out.write(cmd['type'] + '\\n')
    if cmd['type'] == 'steer':
        threading.Thread(target=delayed, args=(cmd,), daemon=True).start()
    else:
        respond(cmd)
""")
    monkeypatch.setattr(pi_rpc, "_PI_COMMAND", sys.executable)
    client = PiRpcClient(
        tmp_path / "session.jsonl", tmp_path, ["-u", str(script), str(log)]
    )
    run = PiRpcRun(client)
    steer: asyncio.Task[str] | None = None
    try:
        await client.start()
        run._active = True
        run._prompt_started = True  # Simulate an acknowledged initial prompt.
        run._unsettled = True
        steer = asyncio.create_task(run.steer("queued instruction"))
        with anyio.fail_after(1):
            while not log.exists() or "steer" not in log.read_text():
                await anyio.sleep(0.01)
        with anyio.fail_after(0.25):
            assert await run.interrupt()
        with pytest.raises(RuntimeError, match="interrupted"):
            await run.steer("must not be written after abort")
        assert log.read_text().splitlines() == ["steer", "clear_queue", "abort"]
        assert await steer == "queued"
    finally:
        if steer is not None and not steer.done():
            steer.cancel()
        await client.close()
