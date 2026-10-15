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


@pytest.mark.anyio
async def test_live_send_requires_real_transport_success():
    from takopi.telegram.bridge import send_plain

    class MissingTransport:
        async def send(self, **kwargs):
            return None

    with pytest.raises(ConnectionError):
        from typing import cast
        from takopi.transport import Transport

        await send_plain(
            cast(Transport, MissingTransport()),
            chat_id=1,
            user_msg_id=7,
            text="Real answer",
            require_success=True,
        )


@pytest.mark.anyio
async def test_orphan_reply_retries_without_restart_or_owner(tmp_path: Path):
    import anyio

    inbox = LiveInbox(tmp_path / "inbox.json")
    receipt = await inbox.receive(1, 10, 7, "a", "Tests passed?")
    await inbox.reconcile(
        "a", [{"role": "user", "content": await inbox.delivery_text(receipt.id)}]
    )
    await inbox.mark_considered(receipt.id, "65 tests passed.")
    delivered = anyio.Event()
    attempts = []

    async def reply(*args):
        attempts.append(args)
        if len(attempts) == 1:
            raise ConnectionError("temporary failure")
        delivered.set()

    svc = LiveConversationService(inbox, no_surrogate, reply, reply_retry_interval=0.01)
    async with anyio.create_task_group() as tg:
        svc.attach_workers(tg)
        with anyio.fail_after(1):
            await delivered.wait()
        svc.stop_workers()
    assert not await inbox.pending_replies()
    assert len(attempts) == 2


def test_only_durably_queued_marker_answers_are_removed_from_final():
    from takopi.telegram.live_conversation import LiveRunner
    from tests.test_telegram_live_integration import FakePi, FakeRun

    from typing import cast
    from takopi.runners.pi_rpc import PiRpcRun

    runner = LiveRunner(FakePi(), cast(PiRpcRun, FakeRun()))
    marker = "a" * 32
    text = f"Task result.\n[takopi-considered:{marker}] 65 tests passed.\nMore result."
    assert runner.public_answer(text) == text
    runner.answered_markers.add(marker)
    assert runner.public_answer(text) == "Task result.\n\nMore result."


@pytest.mark.anyio
async def test_waiting_for_outbox_is_not_timed_out_and_requeued(
    tmp_path: Path, monkeypatch
):
    import asyncio
    import anyio
    import takopi.telegram.live_conversation as module

    real_deadline = anyio.fail_after
    monkeypatch.setattr(module.anyio, "fail_after", lambda _: real_deadline(0.01))
    release = asyncio.Event()
    enqueued = []

    async def reply(*args):
        enqueued.append(args)
        await release.wait()

    inbox = LiveInbox(tmp_path / "i.json")
    r = await inbox.receive(1, 10, 1, "a", "question")
    await inbox.reconcile(
        "a", [{"role": "user", "content": await inbox.delivery_text(r.id)}]
    )
    await inbox.mark_considered(r.id, "Actual answer")
    svc = LiveConversationService(inbox, no_surrogate, reply)
    first = asyncio.create_task(svc._send_replies())
    await asyncio.sleep(0.03)
    second = asyncio.create_task(svc._send_replies())
    await asyncio.sleep(0.03)
    assert len(enqueued) == 1
    assert not first.done()
    release.set()
    await asyncio.gather(first, second)
    assert not await inbox.pending_replies()


@pytest.mark.anyio
async def test_long_receipt_reply_does_not_repeat_successful_chunks(tmp_path: Path):
    inbox = LiveInbox(tmp_path / "i.json")
    r = await inbox.receive(1, 10, 1, "a", "question")
    await inbox.reconcile(
        "a", [{"role": "user", "content": await inbox.delivery_text(r.id)}]
    )
    text = "a" * 2000 + "b" * 2000 + "c"
    await inbox.mark_considered(r.id, text)
    sent = []
    fail = True

    async def reply(*args):
        if fail and sent:
            raise ConnectionError()
        sent.append(args[-1])

    svc = LiveConversationService(inbox, no_surrogate, reply)
    await svc._send_replies()
    assert sent == ["a" * 2000]
    fail = False
    await svc._send_replies()
    assert "".join(sent) == text
    assert not await inbox.pending_replies()
