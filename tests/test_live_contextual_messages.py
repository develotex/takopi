"""Conversation uses the actual task owner, never a phase/counts-only surrogate."""

from pathlib import Path

import anyio
import pytest

from takopi.telegram.live_conversation import LiveConversationService, LiveOwner
from takopi.telegram.live_inbox import LiveInbox
from tests.test_telegram_live_conversation import Main


@pytest.mark.anyio
@pytest.mark.parametrize(
    "text",
    [
        "Какой статус?",
        "И что это значит блять?",
        "Ты че еблан? Я тебе большую задачку отдал написал вперед и что блять?",
        "Вперед",
        "maybe the design",
        "Используй синий. А тесты прошли?",
    ],
)
@pytest.mark.parametrize("workers", [False, True])
async def test_entire_message_reaches_contextual_owner_without_status_noise(
    tmp_path: Path, text: str, workers: bool
):
    main = Main()
    replies = []

    async def answer(*args):
        raise AssertionError("A context-free surrogate must not answer")

    async def reply(*args):
        replies.append(args)

    svc = LiveConversationService(LiveInbox(tmp_path / "inbox.json"), answer, reply)
    owner = LiveOwner(1, 10, "/session.jsonl", main, "Build the evaluation harness")
    svc.register(owner)
    async with anyio.create_task_group() as tg:
        if workers:
            svc.attach_workers(tg)
        assert await svc.handle(1, 10, 12, owner.session_key, text)
        with anyio.fail_after(1):
            while not main.messages:
                await anyio.sleep(0.001)
        assert main.messages[0].endswith(text)
        assert not replies  # Durable intake and RPC ACK are internal, not answers.
        receipt = (await svc.inbox.pending(owner.session_key))[0]
        main.entries = [{"role": "user", "content": main.messages[0]}]
        await svc.reconcile(owner)
        assert not replies
        main.entries.append(
            {
                "role": "assistant",
                "content": f"[takopi-considered:{receipt.marker}] 65 harness tests passed; the baseline is not finished yet.",
            }
        )
        await svc.reconcile(owner)
        with anyio.fail_after(1):
            while not replies:
                await anyio.sleep(0.001)
        assert replies == [
            (1, 10, 12, "65 harness tests passed; the baseline is not finished yet.")
        ]
        await svc.reconcile(owner)
        assert len(replies) == 1
        svc.unregister(owner)
        svc.stop_workers()
        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_starting_question_is_durable_and_silent(tmp_path: Path):
    replies = []

    async def reply(*args):
        replies.append(args)

    async def answer(*args):
        raise AssertionError

    inbox = LiveInbox(tmp_path / "inbox.json")
    svc = LiveConversationService(inbox, answer, reply)
    await svc.buffer_starting(1, 10, 12, "/session.jsonl", "Какой статус?")
    pending = await LiveInbox(tmp_path / "inbox.json").pending("/session.jsonl")
    assert [r.text for r in pending] == ["Какой статус?"]
    assert not replies
