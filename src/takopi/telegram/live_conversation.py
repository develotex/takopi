"""Opt-in, exact-scope Pi live messages. No quick answer enters the writable session."""

from __future__ import annotations

import asyncio
import re
import tempfile

import anyio
from pathlib import Path
from uuid import uuid4
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from ..model import CompletedEvent, ResumeToken, TakopiEvent
from ..runners.pi_rpc import PiRpcClient, PiRpcRun
from .live_inbox import LiveInbox, Receipt


def live_session_path(config_path: Path, chat_id: int, thread_id: int) -> Path:
    return (
        config_path.parent
        / "pi-live-sessions"
        / f"{chat_id}-{thread_id}-{uuid4().hex}.jsonl"
    ).resolve()


def live_route_key(
    chat: int, thread: int | None, resume: ResumeToken | None
) -> tuple[int, int, str] | None:
    if thread is None or resume is None or resume.engine != "pi":
        return None
    path = Path(resume.value)
    if not path.is_absolute() or path.suffix != ".jsonl":
        return None
    return (chat, thread, str(path.resolve()))


async def quick_pi_answer(question: str, snapshot: str) -> str:
    """Ephemeral isolated process: no tools, extensions, skills or project context."""
    with tempfile.TemporaryDirectory(prefix="takopi-quick-") as directory:
        cwd = Path(directory)
        client = PiRpcClient(
            cwd / "quick.jsonl",
            cwd,
            [
                "--no-tools",
                "--no-extensions",
                "--no-skills",
                "--no-context-files",
                "--no-prompt-templates",
                "--no-approve",
            ],
        )
        try:
            answer = ""
            prompt = (
                "Answer briefly using only the public task context below. Do not infer "
                "unobserved work or tool results. Never execute tools.\n"
                f"{snapshot}\nQuestion: {question}"
            )
            async for event in PiRpcRun(client).run(prompt, None):
                if isinstance(event, CompletedEvent):
                    answer = event.answer
            return answer or "No observed result yet."
        finally:
            await client.close()


class MainControl(Protocol):
    async def steer(self, text: str) -> str: ...
    async def get_messages(self) -> list[dict]: ...


class LiveRunner:
    """Bridge-compatible runner wrapping exactly one writable Pi RPC owner."""

    engine = "pi"

    def __init__(
        self, runner: object, rpc: PiRpcRun, ready: anyio.Event | None = None
    ) -> None:
        self.runner = runner
        self.rpc = rpc
        self.client = rpc.client
        self.ready = ready

    def format_resume(self, token: ResumeToken) -> str:
        return self.runner.format_resume(token)  # type: ignore[attr-defined]

    def is_resume_line(self, line: str) -> bool:
        return self.runner.is_resume_line(line)  # type: ignore[attr-defined]

    def extract_resume(self, text: str | None) -> ResumeToken | None:
        return self.runner.extract_resume(text)  # type: ignore[attr-defined]

    async def run(
        self, prompt: str, resume: ResumeToken | None
    ) -> AsyncIterator[TakopiEvent]:
        guidance = (
            "For task updates marked [takopi-live:<marker>], first inspect how they "
            "affect your work. Only after you actually review one, explicitly state "
            "[takopi-considered:<marker>] <specific reason> or "
            "[takopi-deferred:<marker>] <specific reason> in your assistant text. "
            "Do not claim consideration on delivery alone. Continue the main task.\n\n"
        )
        async for event in self.rpc.run(guidance + prompt, resume):
            yield event

    async def steer(self, text: str) -> str:
        if self.ready is not None:
            await asyncio.wait_for(self.ready.wait(), timeout=20)
        if not self.rpc._active:
            # The last agent_settled was already consumed. Continue the same
            # session as a new prompt, never launch a second writable owner.
            async for _ in self.rpc.run(text, None):
                pass
            return "handled"
        return await self.rpc.steer(text)

    async def get_messages(self) -> list[dict]:
        response = await self.client.request("get_messages")
        data = response.get("data")
        messages = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(messages, list):
            raise ValueError("Pi get_messages returned invalid data")
        return messages


QuickAnswer = Callable[[str, str], Awaitable[str]]
Reply = Callable[[int, int, int, str], Awaitable[None]]
_ACK = re.compile(r"\[takopi-(considered|deferred):([a-f0-9]{32})\]\s*([^\n]+)")


@dataclass(slots=True)
class LiveOwner:
    chat_id: int
    thread_id: int
    session_key: str
    main: MainControl
    task: str
    progress: str = "The main task is still running; a pending tool or child has no observed result."
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    quick_slots: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))


def _classify(text: str) -> tuple[str | None, str | None]:
    if text.startswith("/update "):
        return None, text[len("/update ") :].strip() or None
    # Conservatively split an explicit question from a subsequent imperative.
    question, sep, rest = text.partition("?")
    if sep:
        update = rest.strip()
        if update.lower().startswith(
            ("also ", "please ", "use ", "don't ", "do not ", "instead ", "make ")
        ):
            return question.strip() + "?", update
        if not update:
            return text.strip(), None
        return None, None
    if text.lower().startswith(("status", "how is", "what's the status")):
        return text.strip(), None
    if text.lower().startswith(
        (
            "use ",
            "please ",
            "don't ",
            "do not ",
            "instead ",
            "make ",
            "change ",
            "correct ",
            "important: ",
        )
    ):
        return None, text.strip()
    return None, None


def _text(entry: dict) -> str:
    content = entry.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


class LiveConversationService:
    def __init__(
        self,
        inbox: LiveInbox,
        answer: QuickAnswer,
        reply: Reply,
        *,
        quick_timeout: float = 15,
    ) -> None:
        self.inbox = inbox
        self._answer = answer
        self._reply = reply
        self._timeout = quick_timeout
        self._owners: dict[tuple[int, int, str], LiveOwner] = {}

    def register(self, owner: LiveOwner) -> bool:
        key = (owner.chat_id, owner.thread_id, owner.session_key)
        if key in self._owners:
            return False
        self._owners[key] = owner
        return True

    def unregister(self, owner: LiveOwner) -> None:
        key = (owner.chat_id, owner.thread_id, owner.session_key)
        if self._owners.get(key) is owner:
            del self._owners[key]

    def owner(self, chat: int, thread: int, session: str) -> LiveOwner | None:
        return self._owners.get((chat, thread, session))

    async def handle(
        self, chat: int, thread: int, message: int, session: str, text: str
    ) -> bool:
        owner = self.owner(chat, thread, session)
        if owner is None:
            return False
        question, update = _classify(text)
        if question is None and update is None:
            await self._reply(
                chat,
                thread,
                message,
                "Please clarify: ask a question or use /update <task instruction>.",
            )
            return True
        receipt: Receipt | None = None
        if update:
            receipt = await self.inbox.receive(chat, thread, message, session, update)
            # A retry is not a new delivery; uncertain/submitted states are not safe to resend.
            if receipt.state == "received":
                await self._reply(
                    chat,
                    thread,
                    message,
                    "Update received and stored; not yet confirmed delivered or considered.",
                )
            else:
                await self._reply(
                    chat,
                    thread,
                    message,
                    f"Update status: {receipt.state}; not necessarily considered.",
                )
        if question:
            try:
                async with asyncio.timeout(self._timeout):
                    async with owner.quick_slots:
                        # Only explicitly public, scoped context. Never include a session
                        # file, tool args, private transcript, or another topic's data.
                        snapshot = f"Task: {owner.task[:1500]}\nObserved progress: {owner.progress[:1500]}"
                        answer = await self._answer(question, snapshot)
            except Exception:  # noqa: BLE001 - quick reply must not affect main run
                answer = (
                    "Quick answer unavailable; the main task has not been interrupted."
                )
            await self._reply(chat, thread, message, answer)
        if receipt is not None and receipt.state == "received":
            await self.submit(owner, receipt)
        return True

    async def _submit_locked(self, owner: LiveOwner, receipt: Receipt) -> None:
        # The finish flush can beat the incoming handler after durable receive.
        if (await self.inbox.get(receipt.id)).state != "received":
            return
        await self.inbox.mark_uncertain(
            receipt.id, "RPC outcome may be unknown until main-session observation"
        )
        try:
            result = await owner.main.steer(await self.inbox.delivery_text(receipt.id))
            if result not in ("queued", "handled"):
                raise RuntimeError("invalid RPC disposition")
            await self.inbox.mark_submitted(receipt.id)
        except Exception:  # noqa: BLE001 - RPC failure must retain uncertain receipt
            # Preserve uncertain receipt; never silently retry a possibly accepted steer.
            await self._reply(
                receipt.chat_id,
                receipt.thread_id,
                receipt.message_id,
                "Update uncertain: Pi may have accepted it; waiting for main-session observation. Not considered yet.",
            )
            return

    async def submit(self, owner: LiveOwner, receipt: Receipt) -> None:
        async with owner.lock:
            await self._submit_locked(owner, receipt)

    async def flush(self, owner: LiveOwner) -> None:
        async with owner.lock:
            for receipt in await self.inbox.pending(owner.session_key):
                if receipt.state == "received":
                    await self._submit_locked(owner, receipt)

    async def reconcile(self, owner: LiveOwner) -> None:
        async with owner.lock:
            entries = await owner.main.get_messages()
            if not isinstance(entries, list):
                raise ValueError("Pi get_messages returned no message list")
            await self.inbox.reconcile(owner.session_key, entries)
            receipts = {
                r.marker: r for r in await self.inbox.pending(owner.session_key)
            }
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("role") != "assistant":
                    continue
                for match in _ACK.finditer(_text(entry)):
                    action, marker, reason = match.groups()
                    receipt = receipts.get(marker)
                    if receipt is None or receipt.state != "delivered":
                        continue
                    if action == "considered":
                        await self.inbox.mark_considered(receipt.id, reason)
                    else:
                        await self.inbox.mark_deferred(receipt.id, reason)
                    await self._reply(
                        receipt.chat_id,
                        receipt.thread_id,
                        receipt.message_id,
                        f"Update {action}: {reason}",
                    )
                    receipts.pop(marker, None)
