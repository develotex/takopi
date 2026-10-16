"""Unsupported extension dialogs must be declined, including before prompt acceptance."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from takopi.model import CompletedEvent
from takopi.runners.pi_rpc import PiRpcClient, PiRpcRun


@pytest.mark.anyio
@pytest.mark.parametrize("method", ["confirm", "select", "input", "editor"])
@pytest.mark.parametrize("before_acceptance", [True, False])
async def test_rpc_declines_dialogs_without_approval_or_response_waiter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    before_acceptance: bool,
) -> None:
    # A prompt awaiting an extension dialog cannot finish until its matching
    # cancellation arrives. Moving handling into PiRpcRun would deadlock the
    # pre-acceptance variant, because that run still awaits the prompt response.
    child = tmp_path / "dialog_pi.py"
    child.write_text(
        "import json, sys\n"
        f"method = {method!r}\nbefore = {before_acceptance!r}\n"
        "def emit(x):\n"
        " print(json.dumps(x), flush=True)\n"
        "prompt = None\n"
        "for line in sys.stdin:\n"
        " c=json.loads(line)\n"
        " if c['type']=='get_state':\n"
        "  emit({'type':'response','id':c['id'],'success':True,'data':{'sessionId':'dialog-test','sessionFile':sys.argv[-1]}})\n"
        " elif c['type']=='prompt':\n"
        "  prompt=c\n"
        "  if not before:\n"
        "   emit({'type':'response','id':c['id'],'success':True,'data':{'disposition':'started'}})\n"
        "   emit({'type':'agent_start'})\n"
        "  emit({'type':'extension_ui_request','id':'dialog-1','method':method,'title':'PRIVATE CONTENT','message':'PRIVATE CONTENT','options':['approve']})\n"
        " elif c['type']=='extension_ui_response':\n"
        "  assert c == {'type':'extension_ui_response','id':'dialog-1','cancelled':True}, c\n"
        "  if before:\n"
        "   emit({'type':'response','id':prompt['id'],'success':True,'data':{'disposition':'started'}})\n"
        "   emit({'type':'agent_start'})\n"
        "  emit({'type':'message_end','message':{'role':'assistant','content':[{'type':'text','text':'declined'}],'stopReason':'stop'}})\n"
        "  emit({'type':'agent_end','messages':[],'willRetry':False})\n"
        "  emit({'type':'agent_settled'})\n"
    )
    monkeypatch.setattr("takopi.runners.pi_rpc._PI_COMMAND", sys.executable)
    rpc = PiRpcClient(tmp_path / "session.jsonl", tmp_path, ["-u", str(child)])
    try:
        async with asyncio.timeout(2):
            events = [event async for event in PiRpcRun(rpc).run("test", None)]
        final = [event for event in events if isinstance(event, CompletedEvent)]
        assert len(final) == 1
        assert final[0].ok
        assert final[0].answer == "declined"
        assert rpc._pending == {}
    finally:
        await rpc.close()


@pytest.mark.anyio
async def test_dialog_reply_pipe_failure_releases_prompt_waiter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = tmp_path / "closed_input_pi.py"
    child.write_text(
        "import json, os, sys, time\n"
        "c=json.loads(sys.stdin.readline())\n"
        "os.close(0)\n"
        "print(json.dumps({'type':'extension_ui_request','id':'dialog','method':'confirm'}),flush=True)\n"
        "time.sleep(10)\n"
    )
    monkeypatch.setattr("takopi.runners.pi_rpc._PI_COMMAND", sys.executable)
    rpc = PiRpcClient(tmp_path / "closed.jsonl", tmp_path, ["-u", str(child)])
    try:
        with pytest.raises(RuntimeError, match="stream failed"):
            await rpc.request("first", timeout=1)
        assert rpc._pending == {}
    finally:
        await rpc.close()


@pytest.mark.anyio
async def test_rpc_does_not_respond_to_notifications_and_preserves_state_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = tmp_path / "notify_pi.py"
    child.write_text(
        "import json, sys\n"
        "seen=[]\n"
        "def emit(x): print(json.dumps(x), flush=True)\n"
        "for line in sys.stdin:\n"
        " c=json.loads(line); seen.append(c)\n"
        " if c['type']=='first':\n"
        "  emit({'type':'extension_ui_request','id':'note','method':'notify','message':'private'})\n"
        "  emit({'type':'session_info_changed','name':'Generated title'})\n"
        " emit({'type':'response','id':c.get('id'),'success':True,'data':{'commands':seen[:]}})\n"
    )
    monkeypatch.setattr("takopi.runners.pi_rpc._PI_COMMAND", sys.executable)
    rpc = PiRpcClient(tmp_path / "notify.jsonl", tmp_path, ["-u", str(child)])
    try:
        await rpc.request("first")
        response = await rpc.request("inspect")
        assert [c["type"] for c in response["data"]["commands"]] == [
            "first",
            "inspect",
        ]
        async with asyncio.timeout(1):
            records = rpc.events()
            record = await anext(records)
            if record["type"] == "extension_ui_request":
                record = await anext(records)
            assert record == {
                "type": "session_info_changed",
                "name": "Generated title",
            }
    finally:
        await rpc.close()
