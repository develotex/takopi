"""Opt-in, exact-scope Pi live messages. No quick answer enters the writable session."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile

import anyio
from pathlib import Path
from uuid import uuid4
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ..model import ActionEvent, CompletedEvent, ResumeToken, TakopiEvent
from ..runner_bridge import RunningTask
from ..transport import MessageRef
from ..runners.pi_rpc import PiRpcClient, PiRpcRun
from ..runners.pi import _default_session_dir
from .live_inbox import LiveInbox, Receipt


def live_session_path(config_path: Path, chat_id: int, thread_id: int) -> Path:
    return (
        config_path.parent
        / "pi-live-sessions"
        / f"{chat_id}-{thread_id}-{uuid4().hex}.jsonl"
    ).resolve()


def _session_header(path: Path) -> dict | None:
    try:
        with path.open("rb") as handle:
            line = handle.readline(8193)
        if len(line) > 8192 or not line.endswith(b"\n"):
            return None
        header = json.loads(line)
        return (
            header
            if isinstance(header, dict) and header.get("type") == "session"
            else None
        )
    except (OSError, UnicodeError, ValueError):
        return None


def resolve_legacy_session(partial_id: str, cwd: Path) -> Path:
    """Find one local project session without launching an alias-keyed writer.

    The selected canonical path is only a candidate. The RPC owner's get_state
    must validate the actual identity under its canonical-path lock before prompt.
    """
    if re.fullmatch(r"[a-fA-F0-9-]{8,36}", partial_id) is None:
        raise ValueError("Invalid Pi session ID prefix")
    project = cwd.resolve()
    configured = os.environ.get("PI_CODING_AGENT_SESSION_DIR")
    directory = (
        Path(configured).expanduser() if configured else _default_session_dir(project)
    ).resolve()
    matches: set[Path] = set()
    if directory.is_dir():
        for file in directory.glob("*.jsonl"):
            path = file.resolve()
            if path.parent != directory or not file.is_file():
                continue
            header = _session_header(path)
            if (
                header is not None
                and isinstance(header.get("id"), str)
                and header["id"].lower().startswith(partial_id.lower())
                and isinstance(header.get("cwd"), str)
                and Path(header["cwd"]).resolve() == project
            ):
                matches.add(path)
    if len(matches) != 1:
        raise ValueError("Pi session ID not uniquely found in this project")
    return matches.pop()


def legacy_session_busy(
    running_tasks: Mapping[MessageRef, RunningTask], session_path: Path
) -> bool:
    header = _session_header(session_path)
    if header is None or not isinstance(header.get("id"), str):
        return True
    full_id = header["id"].lower()
    for task in running_tasks.values():
        if task.done.is_set():
            continue
        token = task.resume
        if token is None:
            return True  # Identity is still unresolved; do not risk a second writer.
        if token.engine != "pi":
            continue
        value = token.value
        if (Path(value).is_absolute() and Path(value).resolve() == session_path) or (
            re.fullmatch(r"[a-fA-F0-9-]{8,36}", value) is not None
            and full_id.startswith(value.lower())
        ):
            return True
    return False


async def verify_legacy_session(
    rpc: PiRpcRun, session_path: Path, partial_id: str, cwd: Path
) -> None:
    """Verify Pi's actual identity while its canonical RPC owner is held.

    A get_state response is not permission to prompt until file identity and
    project metadata agree. In particular, never fork a cross-project match.
    """
    before = session_path.stat()
    header = _session_header(session_path)
    response = await rpc.client.request("get_state")
    data = response.get("data")
    after = session_path.stat()
    if (
        not isinstance(data, dict)
        or header is None
        or not isinstance(header.get("id"), str)
        or not isinstance(header.get("cwd"), str)
        or not isinstance(data.get("sessionId"), str)
        or not isinstance(data.get("sessionFile"), str)
        or data.get("isStreaming") is not False
        or data["sessionId"] != header["id"]
        or not data["sessionId"].lower().startswith(partial_id.lower())
        or Path(data["sessionFile"]).resolve() != session_path
        or Path(header["cwd"]).resolve() != cwd.resolve()
        or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
    ):
        raise ValueError(
            "Pi session identity/project mismatch; no prompt was submitted"
        )


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
        self.before_final: Callable[[], Awaitable[None]] | None = None
        self.on_progress: Callable[[str], None] | None = None
        self._followups: list[str] = []

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
        next_prompt = guidance + prompt
        next_resume = resume
        for iteration in range(
            9
        ):  # Bound settlement drain even if updates arrive continuously.
            completed: CompletedEvent | None = None
            async for event in self.rpc.run(next_prompt, next_resume):
                if isinstance(event, CompletedEvent):
                    completed = event
                else:
                    if isinstance(event, ActionEvent) and self.on_progress is not None:
                        # Kind and phase only; titles/details can contain tool arguments.
                        self.on_progress(f"{event.action.kind}: {event.phase}")
                    yield event
            if self.before_final is not None:
                await self.before_final()
            if not self._followups:
                if completed is not None:
                    yield completed
                return
            if iteration == 8:
                raise RuntimeError(
                    "Live Pi settlement drain exceeded eight follow-ups; receipt state retained"
                )
            next_prompt = self._followups.pop(0)
            next_resume = None

    async def steer(self, text: str) -> str:
        if self.ready is not None:
            await asyncio.wait_for(self.ready.wait(), timeout=20)
        if not self.rpc._active:
            # The streaming owner, not this caller, renders the continuation.
            self._followups.append(text)
            return "queued"
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
    closing: bool = False
    public_output: str = ""


def _classify(text: str) -> tuple[str | None, str | None]:
    if text.startswith("/update "):
        return None, text[len("/update ") :].strip() or None
    if text.startswith("/обнови "):
        return None, text[len("/обнови ") :].strip() or None
    # Conservatively split an explicit question from a subsequent imperative.
    question, sep, rest = text.partition("?")
    if sep:
        update = rest.strip()
        if update.lower().startswith(
            (
                "also ",
                "please ",
                "use ",
                "don't ",
                "do not ",
                "instead ",
                "make ",
                "и ещё",
                "и еще",
                "не трогай",
                "используй",
                "пожалуйста",
                "вместо ",
                "исправь ",
            )
        ):
            return question.strip() + "?", update
        if not update:
            return text.strip(), None
        return None, None
    if text.lower().startswith(
        (
            "status",
            "how is",
            "what's the status",
            "как дела",
            "какой статус",
            "что со статусом",
        )
    ):
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
            "не трогай",
            "используй",
            "пожалуйста",
            "вместо ",
            "исправь ",
            "поменяй ",
            "важно: ",
            "не меняй",
            "сделай ",
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
        if key in self._owners or any(
            registered.session_key == owner.session_key
            for registered in self._owners.values()
        ):
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
        if owner is None or owner.closing:
            return False
        question, update = _classify(text)
        if question is None and update is None:
            await self._reply(
                chat,
                thread,
                message,
                "Уточните вопрос или передайте инструкцию через /update <текст>. (Please clarify.)",
            )
            return True
        receipt: Receipt | None = None
        if update:
            async with owner.lock:
                if owner.closing:
                    return False
                receipt = await self.inbox.receive(
                    chat, thread, message, session, update
                )
                # A retry is not a new delivery; reply only after durable receive.
                if receipt.state == "received":
                    await self._reply(
                        chat,
                        thread,
                        message,
                        "Инструкция получена и сохранена (received); not yet confirmed delivered or considered.",
                    )
                    await self._drain_locked(owner)
                else:
                    await self._reply(
                        chat,
                        thread,
                        message,
                        f"Статус инструкции: {receipt.state}; это не означает, что она учтена.",
                    )
        if question:
            try:
                async with asyncio.timeout(self._timeout):
                    async with owner.quick_slots:
                        # Only explicitly public, scoped context. Never include a session
                        # file, tool args, private transcript, or another topic's data.
                        statuses = await self.inbox.for_session(owner.session_key)
                        receipts = "\n".join(
                            f"#{r.message_id}: {r.text[:250]} — {r.state}"
                            + (f" ({r.reason[:200]})" if r.reason else "")
                            for r in statuses[-12:]
                        )
                        snapshot = (
                            f"Task: {owner.task[:1000]}\nObserved progress: {owner.progress[:700]}"
                            f"\nPublic result: {owner.public_output[:700]}\nUser updates: {receipts[:1500]}"
                        )
                        answer = await self._answer(question, snapshot)
            except Exception:  # noqa: BLE001 - quick reply must not affect main run
                answer = (
                    "Quick answer unavailable; the main task has not been interrupted."
                )
            await self._reply(chat, thread, message, answer)
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
        await self.flush(owner)

    async def _drain_locked(self, owner: LiveOwner) -> None:
        for receipt in await self.inbox.pending(owner.session_key):
            if receipt.state == "received":
                await self._submit_locked(owner, receipt)

    async def flush(self, owner: LiveOwner) -> None:
        async with owner.lock:
            await self._drain_locked(owner)

    async def finalize(self, owner: LiveOwner) -> None:
        async with owner.lock:
            await self._drain_locked(owner)
            if isinstance(owner.main, LiveRunner) and owner.main._followups:
                return
            owner.closing = True
            self.unregister(owner)

    async def reconcile(self, owner: LiveOwner) -> None:
        async with owner.lock:
            entries = await owner.main.get_messages()
            if not isinstance(entries, list):
                raise ValueError("Pi get_messages returned no message list")
            delivered = await self.inbox.reconcile(owner.session_key, entries)
            for receipt in delivered:
                await self._reply(
                    receipt.chat_id,
                    receipt.thread_id,
                    receipt.message_id,
                    "Инструкция доставлена Pi (delivered), но ещё не подтверждена как учтённая.",
                )
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
