from __future__ import annotations

import anyio
import pytest

from takopi.telegram.loop import ForwardCoalescer, _PendingPrompt, _forward_key
from takopi.telegram.types import TelegramIncomingMessage


@pytest.mark.anyio
async def test_forward_debounce_keeps_prompt_available_until_durable_transition() -> (
    None
):
    original = TelegramIncomingMessage(
        transport="telegram",
        chat_id=-100,
        thread_id=77,
        message_id=1,
        text="Build task",
        reply_to_message_id=None,
        reply_to_text=None,
        sender_id=123,
    )
    pending = _PendingPrompt(
        msg=original,
        text="Build task",
        ambient_context=None,
        chat_project=None,
        topic_key=(-100, 77),
        chat_session_key=None,
        reply_ref=None,
        reply_id=None,
        is_voice_transcribed=False,
        forwards=[],
    )
    queued = {}
    dispatched: list[list[tuple[int, str]]] = []
    done = anyio.Event()

    async def dispatch(item: _PendingPrompt) -> None:
        dispatched.append(list(item.forwards))
        done.set()

    async with anyio.create_task_group() as tg:
        coalescer = ForwardCoalescer(
            task_group=tg, debounce_s=0.01, dispatch=dispatch, pending=queued
        )
        coalescer.schedule(pending)
        barrier = anyio.Event()
        pending.forward_barrier = barrier
        await anyio.sleep(0.04)  # Debounce expired during durable rerouting.
        assert queued.get(_forward_key(original)) is pending
        for message_id, text in ((2, "first"), (3, "second")):
            coalescer.attach_forward(
                TelegramIncomingMessage(
                    transport="telegram",
                    chat_id=-100,
                    thread_id=77,
                    message_id=message_id,
                    text=text,
                    reply_to_message_id=None,
                    reply_to_text=None,
                    sender_id=123,
                    raw={"forward_date": 1},
                )
            )
        assert not dispatched
        barrier.set()
        with anyio.fail_after(1):
            await done.wait()
        assert dispatched == [[(2, "first"), (3, "second")]]
