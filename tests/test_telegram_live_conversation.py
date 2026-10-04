from __future__ import annotations

import asyncio
from pathlib import Path

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


@pytest.mark.anyio
async def test_question_and_update_same_topic_no_interrupt(tmp_path: Path) -> None:
    main = Main()
    replies: list[tuple[int, int, int, str]] = []
    seen: list[str] = []

    async def answer(question: str, snapshot: str) -> str:
        seen.append(snapshot)
        return "Still working; no tool result yet."

    async def reply(chat: int, thread: int, message: int, text: str) -> None:
        replies.append((chat, thread, message, text))

    svc = LiveConversationService(LiveInbox(tmp_path / "inbox.json"), answer, reply)
    owner = LiveOwner(1, 10, "/sessions/a.jsonl", main, "write tests")
    assert svc.register(owner)
    assert not svc.owner(1, 11, "/sessions/a.jsonl")
    assert await svc.handle(
        1, 10, 100, "/sessions/a.jsonl", "What's the status? Also use Python 3.12."
    )
    assert len(main.messages) == 1
    assert main.messages[0].endswith("Also use Python 3.12.")
    assert "What's the status?" not in main.messages[0]
    assert replies[0][:3] == (1, 10, 100)
    assert "received" in replies[0][3]
    assert "not yet confirmed delivered or considered" in replies[0][3]
    assert seen and "/sessions/a.jsonl" not in seen[0]
    receipt = (await svc.inbox.pending(owner.session_key))[0]
    assert receipt.state == "submitted"
    # RPC acceptance alone is not delivery. Observation is required.
    main.entries.append(
        {"role": "user", "content": await svc.inbox.delivery_text(receipt.id)}
    )
    await svc.reconcile(owner)
    assert (await svc.inbox.get(receipt.id)).state == "delivered"
    main.entries.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"[takopi-considered:{receipt.marker}] I'll use Python 3.12.",
                }
            ],
        }
    )
    await svc.reconcile(owner)
    assert (await svc.inbox.get(receipt.id)).state == "considered"
    assert replies[-1][:3] == (1, 10, 100)
    assert "considered" in replies[-1][3]


@pytest.mark.anyio
async def test_ambiguous_and_cross_topic_are_not_dispatched(tmp_path: Path) -> None:
    main = Main()
    replies: list[str] = []

    async def reply(_chat: int, _thread: int, _msg: int, text: str) -> None:
        replies.append(text)

    async def answer(_question: str, _snapshot: str) -> str:
        return "answer"

    svc = LiveConversationService(LiveInbox(tmp_path / "inbox.json"), answer, reply)
    owner = LiveOwner(1, 10, "/sessions/a.jsonl", main, "task")
    svc.register(owner)
    assert not await svc.handle(1, 11, 2, "/sessions/a.jsonl", "/update secret")
    assert not await svc.handle(1, 10, 3, "/sessions/b.jsonl", "/update secret")
    assert await svc.handle(1, 10, 4, owner.session_key, "maybe the design")
    assert not main.messages
    assert "clarify" in replies[-1].lower()
    assert await svc.handle(1, 10, 5, owner.session_key, "/update use green")
    assert main.messages[-1].endswith("use green")
    assert await svc.handle(1, 10, 5, owner.session_key, "/update use green")
    assert len(main.messages) == 1


@pytest.mark.anyio
async def test_failed_steer_reports_uncertain_without_retry(tmp_path: Path) -> None:
    main = Main()
    replies: list[str] = []

    async def bad_steer(_text: str) -> str:
        raise TimeoutError("response lost")

    main.steer = bad_steer  # type: ignore[method-assign]

    async def answer(_question: str, _snapshot: str) -> str:
        return "answer"

    async def reply(_chat: int, _thread: int, _msg: int, text: str) -> None:
        replies.append(text)

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    owner = LiveOwner(1, 10, "a", main, "task")
    svc.register(owner)
    await svc.handle(1, 10, 3, "a", "/update use green")
    assert (await svc.inbox.pending("a"))[0].state == "uncertain"
    assert "uncertain" in replies[-1].lower()
    await svc.handle(1, 10, 3, "a", "/update use green")
    assert len(main.messages) == 0


@pytest.mark.anyio
async def test_clear_imperative_is_stored_without_explicit_command(
    tmp_path: Path,
) -> None:
    main = Main()

    async def reply(*_args: object) -> None:
        return None

    async def answer(*_args: object) -> str:
        return "answer"

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    owner = LiveOwner(1, 10, "a", main, "task")
    svc.register(owner)
    await svc.handle(1, 10, 42, "a", "Use Python 3.12 for the tests.")
    assert main.messages[0].endswith("Use Python 3.12 for the tests.")


@pytest.mark.anyio
async def test_question_timeout_does_not_interrupt_main(tmp_path: Path) -> None:
    main = Main()
    replies: list[str] = []

    async def answer(_question: str, _snapshot: str) -> str:
        await asyncio.sleep(1)
        return "late"

    async def reply(_chat: int, _thread: int, _msg: int, text: str) -> None:
        replies.append(text)

    svc = LiveConversationService(
        LiveInbox(tmp_path / "i.json"), answer, reply, quick_timeout=0.01
    )
    owner = LiveOwner(1, 10, "a", main, "task")
    svc.register(owner)
    assert await svc.handle(1, 10, 7, "a", "status?")
    assert not main.messages
    assert "unavailable" in replies[-1]
