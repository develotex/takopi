from typing import cast

import pytest

from takopi.runners.pi_rpc import PiRpcRun
from takopi.telegram.live_conversation import (
    LiveConversationService,
    LiveOwner,
    LiveRunner,
)
from takopi.telegram.live_inbox import LiveInbox
from tests.test_telegram_live_integration import FakePi, FakeRun


class BufferedRun(FakeRun):
    def __init__(self):
        super().__init__()
        self.assistant_outputs: list[tuple[str, bool]] = []
        self.output_generation = 0


@pytest.mark.anyio
@pytest.mark.parametrize("ack_before_user", [False, True])
async def test_fresh_rpc_owner_recovers_only_valid_historical_receipt_evidence(
    tmp_path, ack_before_user
):
    inbox = LiveInbox(tmp_path / "i.json")
    r = await inbox.receive(1, 10, 7, "a", "Use 20")
    rpc = BufferedRun()
    user = {"role": "user", "content": await inbox.delivery_text(r.id)}
    ack = {
        "role": "assistant",
        "content": f"[takopi-considered:{r.marker}] Limit is 20.",
    }
    rpc.entries = [ack, user] if ack_before_user else [user, ack]
    rpc.entries.insert(0, {"role": "assistant", "content": "OLD UNRELATED TASK OUTPUT"})
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))
    sent = []

    async def reply(*args):
        sent.append(args[-1])

    svc = LiveConversationService(inbox, None, reply)
    owner = LiveOwner(1, 10, "a", runner, "recovery")
    await svc.reconcile(owner)
    await svc.reconcile(owner)
    assert sent == ([] if ack_before_user else ["Limit is 20."])
    assert (await inbox.get(r.id)).state == (
        "delivered" if ack_before_user else "considered"
    )


@pytest.mark.anyio
async def test_mixed_completed_event_preserves_prefix_marker_suffix_order(tmp_path):
    inbox = LiveInbox(tmp_path / "i.json")
    r = await inbox.receive(1, 10, 7, "a", "Use 20")
    text = f"Earlier result: limit=10\n[takopi-considered:{r.marker}] Corrected result: limit=20\nFinal details: limit=20"
    rpc = BufferedRun()
    rpc.assistant_outputs = [(text, True)]
    rpc.output_generation = 1
    rpc.entries = [
        {"role": "user", "content": await inbox.delivery_text(r.id)},
        {"role": "assistant", "content": text},
    ]
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))
    sent = []

    async def reply(*args):
        sent.append(args[-1])

    svc = LiveConversationService(inbox, None, reply)
    owner = LiveOwner(1, 10, "a", runner, "task")
    runner.on_intermediate_answer = lambda body: svc.queue_output(1, 10, 1, "a", body)
    await svc.reconcile(owner)
    await svc.reconcile(owner)
    assert sent == [
        "Earlier result: limit=10",
        "Corrected result: limit=20",
        "Final details: limit=20",
    ]
