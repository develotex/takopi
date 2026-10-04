from __future__ import annotations

import pytest

from takopi.telegram.live_conversation import (
    LiveConversationService,
    LiveOwner,
    LiveRunner,
)
from takopi.telegram.live_inbox import LiveInbox
from pathlib import Path


class FakePi:
    engine = "pi"

    def format_resume(self, token):
        return token.value

    def is_resume_line(self, line):
        return False

    def extract_resume(self, text):
        return None


class FakeRun:
    def __init__(self):
        self.client = self
        self._active = True
        self.sent: list[str] = []
        self.entries: list[dict] = []

    async def steer(self, text):
        self.sent.append(text)
        return "queued"

    async def request(self, typ):
        assert typ == "get_messages"
        return {"data": {"messages": self.entries}}

    async def run(self, prompt, resume):
        from takopi.model import CompletedEvent

        self.sent.append(prompt)
        yield CompletedEvent(engine="pi", ok=True, answer="done")


@pytest.mark.anyio
async def test_adapter_uses_same_rpc_owner_and_observed_messages(tmp_path: Path):
    rpc = FakeRun()
    runner = LiveRunner(FakePi(), rpc)

    async def answer(*_):
        return "answer"

    async def reply(*_):
        return None

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    owner = LiveOwner(1, 10, str(tmp_path / "s.jsonl"), runner, "task")
    svc.register(owner)
    assert await svc.handle(1, 10, 33, owner.session_key, "/update use green")
    assert rpc.sent[0].endswith("use green")
    receipt = (await svc.inbox.pending(owner.session_key))[0]
    assert receipt.state == "submitted"
    rpc.entries = [
        {"role": "user", "content": await svc.inbox.delivery_text(receipt.id)}
    ]
    await svc.reconcile(owner)
    assert (await svc.inbox.get(receipt.id)).state == "delivered"


@pytest.mark.anyio
async def test_settlement_race_uses_same_owner_followup_once(tmp_path: Path):
    rpc = FakeRun()
    rpc._active = False
    runner = LiveRunner(FakePi(), rpc)
    replies = []

    async def answer(*_):
        return "answer"

    async def reply(*args):
        replies.append(args)

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    owner = LiveOwner(1, 10, str(tmp_path / "s.jsonl"), runner, "task")
    svc.register(owner)
    receipt = await svc.inbox.receive(1, 10, 33, owner.session_key, "use green")
    await svc.flush(owner)
    assert len(rpc.sent) == 1
    assert rpc.sent[0].endswith("use green")
    await svc.submit(owner, receipt)
    assert len(rpc.sent) == 1
    assert (await svc.inbox.get(receipt.id)).state == "submitted"


@pytest.mark.anyio
async def test_main_prompt_teaches_explicit_ack_without_quick_transcript():
    rpc = FakeRun()
    runner = LiveRunner(FakePi(), rpc)
    _ = [event async for event in runner.run("build feature", None)]
    assert "[takopi-considered:" in rpc.sent[0]
    assert "[takopi-deferred:" in rpc.sent[0]
    assert "build feature" in rpc.sent[0]
