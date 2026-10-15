from pathlib import Path

import pytest

from takopi.telegram.live_conversation import LiveConversationService, LiveOwner
from takopi.telegram.live_inbox import LiveInbox
from tests.test_telegram_live_conversation import Main, no_surrogate


@pytest.mark.anyio
async def test_answer_retried_after_transport_failure_and_restart_without_resteer(
    tmp_path: Path,
):
    inbox = LiveInbox(tmp_path / "inbox.json")
    main = Main()
    sent = []

    async def broken(*args):
        raise ConnectionError("Telegram unavailable")

    svc = LiveConversationService(inbox, no_surrogate, broken)
    owner = LiveOwner(1, 10, "a", main, "task")
    svc.register(owner)
    await svc.handle(1, 10, 7, "a", "Tests passed?")
    receipt = (await inbox.pending("a"))[0]
    main.entries = [
        {"role": "user", "content": main.messages[0]},
        {
            "role": "assistant",
            "content": f"[takopi-considered:{receipt.marker}] 65 tests passed.",
        },
    ]
    await svc.reconcile(owner)
    assert (await inbox.get(receipt.id)).state == "considered"

    async def reply(*args):
        sent.append(args)

    restarted = LiveConversationService(
        LiveInbox(tmp_path / "inbox.json"), no_surrogate, reply
    )
    await restarted.notify_recovery()
    await restarted.notify_recovery()
    assert sent == [(1, 10, 7, "65 tests passed.")]
    assert len(main.messages) == 1


def test_only_delivered_marker_answers_are_removed_from_final():
    from takopi.telegram.live_conversation import LiveRunner
    from tests.test_telegram_live_integration import FakePi, FakeRun

    runner = LiveRunner(FakePi(), FakeRun())
    marker = "a" * 32
    text = f"Task result.\n[takopi-considered:{marker}] 65 tests passed.\nMore result."
    assert runner.public_answer(text) == text
    runner.delivered_markers.add(marker)
    assert runner.public_answer(text) == "Task result.\n\nMore result."
