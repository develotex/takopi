from __future__ import annotations

from pathlib import Path

import anyio
import pytest

from takopi.telegram.live_conversation import LiveConversationService, LiveOwner
from takopi.telegram.live_inbox import LiveInbox


class Main:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.entries: list[dict] = []
        self.active = True

    async def steer(self, text: str) -> str:
        self.messages.append(text)
        return "queued"

    async def get_messages(self) -> list[dict]:
        return self.entries


async def no_surrogate(*_args):
    raise AssertionError(
        "Questions must use the task owner, not a context-free surrogate"
    )


@pytest.mark.anyio
async def test_question_and_update_same_topic_no_interrupt(tmp_path: Path) -> None:
    main = Main()
    replies = []

    async def reply(*args):
        replies.append(args)

    svc = LiveConversationService(
        LiveInbox(tmp_path / "inbox.json"), no_surrogate, reply
    )
    owner = LiveOwner(1, 10, "/sessions/a.jsonl", main, "write tests")
    assert svc.register(owner)
    assert not svc.owner(1, 11, "/sessions/a.jsonl")
    text = "What's the status? Also use Python 3.12."
    assert await svc.handle(1, 10, 100, owner.session_key, text)
    assert len(main.messages) == 1 and main.messages[0].endswith(text)
    assert not replies
    receipt = (await svc.inbox.pending(owner.session_key))[0]
    assert receipt.state == "submitted"
    main.entries.append(
        {"role": "user", "content": await svc.inbox.delivery_text(receipt.id)}
    )
    await svc.reconcile(owner)
    assert (await svc.inbox.get(receipt.id)).state == "delivered"
    assert not replies
    main.entries.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"[takopi-considered:{receipt.marker}] Tests passed on Python 3.12.",
                }
            ],
        }
    )
    await svc.reconcile(owner)
    assert (await svc.inbox.get(receipt.id)).state == "considered"
    assert replies == [(1, 10, 100, "Tests passed on Python 3.12.")]


@pytest.mark.anyio
async def test_ambiguous_text_retained_but_cross_topic_not_dispatched(
    tmp_path: Path,
) -> None:
    main = Main()
    replies = []

    async def reply(*args):
        replies.append(args)

    svc = LiveConversationService(
        LiveInbox(tmp_path / "inbox.json"), no_surrogate, reply
    )
    owner = LiveOwner(1, 10, "/sessions/a.jsonl", main, "task")
    svc.register(owner)
    assert not await svc.handle(1, 11, 2, owner.session_key, "/update secret")
    assert not await svc.handle(1, 10, 3, "/sessions/b.jsonl", "/update secret")
    assert await svc.handle(1, 10, 4, owner.session_key, "maybe the design")
    assert await svc.handle(1, 10, 5, owner.session_key, "/update use green")
    assert await svc.handle(1, 10, 5, owner.session_key, "/update use green")
    assert [s.splitlines()[-1] for s in main.messages] == [
        "maybe the design",
        "use green",
    ]
    assert not replies


@pytest.mark.anyio
async def test_failed_steer_reports_uncertain_without_retry(tmp_path: Path) -> None:
    main = Main()
    replies = []
    attempts = []

    async def bad_steer(text):
        attempts.append(text)
        raise TimeoutError("response lost")

    main.steer = bad_steer

    async def reply(*args):
        replies.append(args[-1])

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), no_surrogate, reply)
    owner = LiveOwner(1, 10, "a", main, "task")
    svc.register(owner)
    await svc.handle(1, 10, 3, "a", "status?")
    assert (await svc.inbox.pending("a"))[0].state == "uncertain"
    assert "uncertain" in replies[-1].lower()
    await svc.handle(1, 10, 3, "a", "status?")
    assert len(attempts) == 1


@pytest.mark.anyio
async def test_initial_intent_restart_notice_has_id_but_never_replays_prompt(
    tmp_path: Path,
):
    path = tmp_path / "new.jsonl"
    inbox = LiveInbox(tmp_path / "i.json")
    initial = await inbox.receive_initial(
        1, 10, 1, str(path), "PRIVATE_INITIAL_PROMPT", str(tmp_path)
    )
    await inbox.mark_initial_uncertain(initial.id, initial.session_key)
    replies = []

    async def reply(*args):
        replies.append(args)

    restarted = LiveConversationService(
        LiveInbox(tmp_path / "i.json"), no_surrogate, reply
    )
    await restarted.notify_recovery()
    assert replies[-1][:3] == (1, 10, 1)
    assert "retry-initial 1 confirm" in replies[-1][3]
    assert "PRIVATE_INITIAL_PROMPT" not in replies[-1][3]
    assert not path.exists()


@pytest.mark.anyio
async def test_recovery_notice_and_scoped_ambiguous_deferral_refusal_after_restart(
    tmp_path: Path,
):
    path = tmp_path / "inbox.json"
    inbox = LiveInbox(path)
    receipt = await inbox.receive(1, 10, 33, "exact-a", "private update")
    await inbox.mark_uncertain(receipt.id, "response lost")
    replies = []

    async def reply(*args):
        replies.append(args)

    restarted = LiveConversationService(LiveInbox(path), no_surrogate, reply)
    await restarted.notify_recovery()
    assert replies[-1][:3] == (1, 10, 33)
    assert "33" in replies[-1][3] and "duplicate" in replies[-1][3].lower()
    assert "private update" not in replies[-1][3]
    assert not await restarted.recover_receipt(
        1, 11, 99, "exact-a", "defer 33 user chose skip"
    )
    assert (await inbox.get(receipt.id)).state == "uncertain"
    assert await restarted.recover_receipt(
        1, 10, 99, "exact-a", "defer 33 user chose skip"
    )
    assert (await inbox.get(receipt.id)).state == "uncertain"
    assert "refused" in replies[-1][3].lower()


@pytest.mark.anyio
async def test_uncertain_first_blocks_later_question_and_successful_final(
    tmp_path: Path,
):
    main = Main()
    attempted = []

    async def maybe_accepted(text):
        attempted.append(text)
        if len(attempted) == 1:
            raise TimeoutError("response lost")
        return "queued"

    main.steer = maybe_accepted
    replies = []

    async def reply(*args):
        replies.append(args[-1])

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), no_surrogate, reply)
    owner = LiveOwner(1, 10, "a", main, "task")
    svc.register(owner)
    await svc.handle(1, 10, 10, "a", "Use green")
    await svc.handle(1, 10, 11, "a", "Why that choice?")
    assert [r.state for r in await svc.inbox.pending("a")] == ["uncertain", "received"]
    assert len(attempted) == 1
    assert await svc.finalize(owner) is False
    assert any("uncertain" in text.lower() for text in replies)


@pytest.mark.anyio
async def test_questions_not_lost_after_sixteen_and_not_blocked_by_telegram(
    tmp_path: Path,
):
    main = Main()

    async def reply(*_args):
        raise AssertionError("No admission notifications")

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), no_surrogate, reply)
    owner = LiveOwner(1, 10, "a", main, "task")
    svc.register(owner)
    async with anyio.create_task_group() as tg:
        svc.attach_workers(tg)
        for message in range(25):
            await svc.handle(1, 10, message, "a", f"Question {message}?")
        with anyio.fail_after(2):
            while len(main.messages) != 25:
                await anyio.sleep(0.001)
        assert [r.text for r in await svc.inbox.pending("a")] == [
            f"Question {i}?" for i in range(25)
        ]
        svc.unregister(owner)
        svc.stop_workers()
        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_russian_question_mixed_and_correction_preserve_instruction(
    tmp_path: Path,
):
    main = Main()
    replies = []

    async def reply(*args):
        replies.append(args)

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), no_surrogate, reply)
    svc.register(LiveOwner(1, 10, "a", main, "работа"))
    texts = [
        "Как дела?",
        "Как дела? И ещё — не трогай авторизацию",
        "Не трогай авторизацию",
        "Can you use green?",
    ]
    for i, text in enumerate(texts):
        await svc.handle(1, 10, i, "a", text)
    assert [s.splitlines()[-1] for s in main.messages] == texts
    assert not replies


@pytest.mark.anyio
async def test_slow_steer_does_not_block_next_message_or_cancel(tmp_path: Path):
    entered, release = anyio.Event(), anyio.Event()
    main = Main()

    async def steer(text):
        main.messages.append(text)
        entered.set()
        await release.wait()
        return "queued"

    main.steer = steer

    async def reply(*_args):
        raise AssertionError("No admission notices")

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), no_surrogate, reply)
    owner = LiveOwner(1, 10, "a", main, "task")
    svc.register(owner)
    async with anyio.create_task_group() as tg:
        svc.attach_workers(tg)
        await svc.handle(1, 10, 1, "a", "status?")
        await entered.wait()
        with anyio.fail_after(1):
            await svc.handle(1, 10, 2, "a", "Use blue")
        assert [r.text for r in await svc.inbox.pending("a")] == ["status?", "Use blue"]
        release.set()
        svc.unregister(owner)
        svc.stop_workers()
        tg.cancel_scope.cancel()
