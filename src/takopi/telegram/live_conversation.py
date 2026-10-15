"""Opt-in, exact-scope conversation with the existing contextual Pi owner."""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile

import anyio
from anyio.abc import TaskGroup
from pathlib import Path
from uuid import uuid4
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol

from ..model import ActionEvent, CompletedEvent, ResumeToken, StartedEvent, TakopiEvent
from ..runner_bridge import RunningTask
from ..transport import MessageRef
from ..runners.pi_rpc import PiRpcClient, PiRpcRun
from ..runners.pi_identity import (
    session_header as _session_header,
    resolve_legacy_session,
)
from .live_inbox import InitialIntent, LiveInbox, Receipt

__all__ = ["resolve_legacy_session"]

logger = logging.getLogger(__name__)


def live_session_path(config_path: Path, chat_id: int, thread_id: int) -> Path:
    return (
        config_path.parent
        / "pi-live-sessions"
        / f"{chat_id}-{thread_id}-{uuid4().hex}.jsonl"
    ).resolve()


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
    rpc.expected_session_id = data["sessionId"]


def session_header_id(session_path: Path) -> str:
    header = _session_header(session_path)
    if header is None or not isinstance(header.get("id"), str):
        raise ValueError("Pi session header is missing")
    return header["id"]


async def verify_bound_session(rpc: PiRpcRun, session_path: Path, cwd: Path) -> None:
    """Verify an already bound canonical path and project before any prompt."""
    header = _session_header(session_path)
    if header is None or not isinstance(header.get("id"), str):
        raise ValueError("Bound Pi session header is missing")
    await verify_legacy_session(rpc, session_path, header["id"], cwd)


async def verify_fresh_session(rpc: PiRpcRun, session_path: Path, cwd: Path) -> str:
    """Provisional identity: Pi writes a NEW JSONL only after first prompt."""
    if (
        session_path.exists()
        or session_path.resolve() != session_path
        or rpc.client.session_path.resolve() != session_path
        or rpc.client.cwd.resolve() != cwd.resolve()
    ):
        raise ValueError("Fresh Pi session path or cwd is no longer unique")
    response = await rpc.client.request("get_state")
    data = response.get("data")
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("sessionId"), str)
        or not data["sessionId"]
        or data.get("sessionFile") != str(session_path)
        or data.get("isStreaming") is not False
    ):
        raise ValueError(
            "Fresh Pi session identity/project mismatch; no prompt was submitted"
        )
    rpc.expected_session_id = data["sessionId"]
    return data["sessionId"]


def verify_provisional_header(session_path: Path, session_id: str, cwd: Path) -> None:
    """Require a real matching header before binding or receipt attestation."""
    header = _session_header(session_path)
    if (
        not session_path.is_file()
        or session_path.resolve() != session_path
        or header is None
        or header.get("id") != session_id
        or not isinstance(header.get("cwd"), str)
        or Path(header["cwd"]).resolve() != cwd.resolve()
    ):
        raise ValueError("New Pi session header has not matched provisional identity")


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
        self,
        runner: object,
        rpc: PiRpcRun,
        ready: anyio.Event | None = None,
        *,
        initial_recovery: bool = False,
    ) -> None:
        self.runner = runner
        self.rpc = rpc
        self.client = rpc.client
        self.ready = ready
        self.initial_recovery = initial_recovery
        self.before_final: Callable[[], Awaitable[bool | None]] | None = None
        self.on_progress: Callable[[str], None] | None = None
        self.on_followup_started: Callable[[str], Awaitable[None]] | None = None
        self._followups: list[str] = []
        self._inflight_followup: str | None = None
        self.cancel_requested: anyio.Event | None = None
        self.settlement_ok = False
        self.delivered_markers: set[str] = set()
        self.suppress_empty_final = True
        self.final_failure = "Live update delivery is uncertain; final result is incomplete. Check the scoped receipt before retrying."

    def public_answer(self, text: str) -> str:
        """Do not echo technical receipts whose actual answer was already sent."""
        return _ACK.sub(
            lambda match: ""
            if match.group(2) in self.delivered_markers
            else match.group(0),
            text,
        ).strip()

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
            "Messages marked [takopi-live:<marker>] are conversation from the user "
            "in this task, including questions and corrections. Interpret the entire "
            "message using the task and conversation history; no /update command is required. "
            "Do not interrupt tools or abandon the main task. When you have an actual "
            "answer or have applied a correction, state "
            "[takopi-considered:<marker>] <your substantive answer to the user> or "
            "[takopi-deferred:<marker>] <specific reason the request cannot be fulfilled>. "
            "Put the entire reply on the same line after the marker; it is sent directly "
            "to the user. Never substitute "
            "receipt counts, tool lifecycle phases, acknowledgements, 'working on it', "
            "or promises to check for an answer. If evidence is missing, do the required "
            "work before answering; do not mark the request considered merely on delivery. "
            "Do not repeat that answer in the final summary. Continue the main task.\n\n"
        )
        next_prompt = prompt if self.initial_recovery else guidance + prompt
        next_resume = resume
        queued_text: str | None = None
        last_answer = ""
        for iteration in range(
            9
        ):  # Bound settlement drain even if updates arrive continuously.
            if self.cancel_requested is not None and self.cancel_requested.is_set():
                yield CompletedEvent(
                    engine="pi",
                    ok=False,
                    answer="",
                    resume=next_resume,
                    error="Pi task cancelled before prompt continuation",
                )
                return
            completed: CompletedEvent | None = None
            async for event in self.rpc.run(next_prompt, next_resume):
                if isinstance(event, CompletedEvent):
                    completed = event
                else:
                    if queued_text is not None and isinstance(event, StartedEvent):
                        if self.on_followup_started is not None:
                            await self.on_followup_started(queued_text)
                        self._inflight_followup = None
                        queued_text = None
                    if isinstance(event, ActionEvent) and self.on_progress is not None:
                        # Kind and phase only; titles/details can contain tool arguments.
                        self.on_progress(f"{event.action.kind}: {event.phase}")
                    yield event
            self.settlement_ok = completed is not None and completed.ok
            if self.before_final is not None and await self.before_final() is False:
                yield CompletedEvent(
                    engine="pi",
                    ok=False,
                    answer=self.final_failure,
                    resume=completed.resume if completed is not None else next_resume,
                    error="Unresolved live update receipt",
                )
                return
            if completed is not None:
                answer = self.public_answer(completed.answer)
                if answer:
                    last_answer = answer
            if not self._followups:
                if completed is not None:
                    yield replace(completed, answer=last_answer)
                return
            if iteration == 8:
                raise RuntimeError(
                    "Live Pi settlement drain exceeded eight follow-ups; receipt state retained"
                )
            next_prompt = self._followups.pop(0)
            queued_text = next_prompt
            self._inflight_followup = queued_text
            next_resume = None

    async def steer(
        self,
        text: str,
        before_rpc: Callable[[], Awaitable[None]] | None = None,
    ) -> str:
        # Never await StartedEvent here: the Telegram poller must remain responsive
        # while the initial RPC prompt is still acquiring its session identity.
        if not self.rpc._active or not getattr(self.rpc, "_prompt_started", True):
            # The streaming owner, not this caller, renders the continuation.
            self._followups.append(text)
            return "local_pending"
        if before_rpc is not None:
            await before_rpc()
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
    accept_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    drain_event: asyncio.Event = field(default_factory=asyncio.Event)
    closing: bool = False
    public_output: str = ""
    running_task: RunningTask | None = None
    identity_verified: bool = True


def _conversation_text(text: str) -> str:
    """Keep user meaning intact; /update remains an optional legacy alias."""
    text = text.strip()
    for command in ("/update", "/обнови"):
        if text == command:
            return ""
        if text.startswith(command + " "):
            return text[len(command) :].strip()
    return text


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
        answer: QuickAnswer | None,
        reply: Reply,
        *,
        quick_timeout: float = 15,
    ) -> None:
        self.inbox = inbox
        # Legacy constructor arguments are accepted for embedding compatibility;
        # the context-free answer callback is intentionally never invoked.
        self._reply = reply
        self._reply_lock = asyncio.Lock()
        self._owners: dict[tuple[int, int, str], LiveOwner] = {}
        self._worker_tg: TaskGroup | None = None
        self._startup_notices: asyncio.Queue[tuple[int, int, int, str]] = asyncio.Queue(
            maxsize=128
        )
        self._startup_notify = asyncio.Event()
        self._worker_stopping = False

    def stop_workers(self) -> None:
        self._worker_stopping = True
        self._startup_notify.set()

    def attach_workers(self, task_group: TaskGroup) -> None:
        self._worker_tg = task_group
        task_group.start_soon(self._startup_ack_worker)
        for owner in self._owners.values():
            self._start_workers(owner)

    async def _startup_reply(
        self, chat: int, thread: int, message: int, text: str
    ) -> None:
        if self._worker_tg is None:
            await self._reply(chat, thread, message, text)
            return
        try:
            self._startup_notices.put_nowait((chat, thread, message, text))
            self._startup_notify.set()
        except asyncio.QueueFull:
            # A quick question has no durable receipt. Do not silently drop its
            # refusal when the bounded notification queue is also saturated.
            try:
                with anyio.fail_after(5):
                    await self._reply(chat, thread, message, text)
            except Exception:  # noqa: BLE001 - Telegram can still be unavailable
                logger.warning(
                    "Startup ACK and direct reply failed; receipt remains durable"
                )

    async def _startup_ack_worker(self) -> None:
        while True:
            await self._startup_notify.wait()
            self._startup_notify.clear()
            while not self._startup_notices.empty():
                chat, thread, message, text = self._startup_notices.get_nowait()
                try:
                    with anyio.fail_after(5):
                        await self._reply(chat, thread, message, text)
                except Exception:  # noqa: BLE001 - intake and notification stay independent
                    logger.warning("Startup ACK failed; receipt remains durable")
            if self._worker_stopping:
                return

    def _start_workers(self, owner: LiveOwner) -> None:
        if self._worker_tg is None:
            return
        self._worker_tg.start_soon(self._drain_worker, owner)

    async def _drain_worker(self, owner: LiveOwner) -> None:
        while self.owner(owner.chat_id, owner.thread_id, owner.session_key) is owner:
            await owner.drain_event.wait()
            owner.drain_event.clear()
            try:
                async with owner.lock:
                    await self._drain_locked(owner)
            except Exception:  # noqa: BLE001 - uncertain receipts are never replayed silently
                logger.warning("Live delivery worker failed; receipts remain durable")

    def register(self, owner: LiveOwner) -> bool:
        key = (owner.chat_id, owner.thread_id, owner.session_key)
        if key in self._owners or any(
            registered.session_key == owner.session_key
            or (registered.chat_id, registered.thread_id)
            == (owner.chat_id, owner.thread_id)
            for registered in self._owners.values()
        ):
            return False
        self._owners[key] = owner
        self._start_workers(owner)
        return True

    def unregister(self, owner: LiveOwner) -> None:
        key = (owner.chat_id, owner.thread_id, owner.session_key)
        if self._owners.get(key) is owner:
            del self._owners[key]
            owner.drain_event.set()

    def owner(self, chat: int, thread: int, session: str) -> LiveOwner | None:
        return self._owners.get((chat, thread, session))

    def topic_owner(self, chat: int, thread: int) -> LiveOwner | None:
        """Include finalizing owners, which still own the session and topic."""
        matches = [
            o
            for o in self._owners.values()
            if o.chat_id == chat and o.thread_id == thread
        ]
        return matches[0] if len(matches) == 1 else None

    def has_topic_owner(self, chat: int, thread: int) -> bool:
        return any(
            o.chat_id == chat and o.thread_id == thread for o in self._owners.values()
        )

    def owner_for_topic(self, chat: int, thread: int) -> LiveOwner | None:
        matches = [
            o
            for o in self._owners.values()
            if o.chat_id == chat and o.thread_id == thread and not o.closing
        ]
        return matches[0] if len(matches) == 1 else None

    async def buffer_starting(
        self, chat: int, thread: int, message: int, session: str, text: str
    ) -> None:
        """Persist updates while the exact queued Pi owner has not registered yet."""
        update = _conversation_text(text)
        if update:
            await self.inbox.receive(chat, thread, message, session, update)

    async def begin_idle_retry(
        self, receipt: Receipt, rpc: PiRpcRun, cwd: Path
    ) -> bool:
        """Inspect exact canonical owner before an explicit retry; never infer absence."""
        path = Path(receipt.session_key)
        header = _session_header(path)
        if (
            not path.is_absolute()
            or path.resolve() != path
            or header is None
            or not isinstance(header.get("id"), str)
            or not isinstance(header.get("cwd"), str)
            or Path(header["cwd"]).resolve() != cwd.resolve()
            or Path(rpc.client.session_path).resolve() != path
        ):
            raise ValueError("Canonical Pi session header or project mismatch")
        pending = await self.inbox.unresolved_for_topic(
            receipt.chat_id, receipt.thread_id, receipt.session_key
        )
        if not pending or pending[0].id != receipt.id:
            raise ValueError("Retry must be the oldest unresolved receipt")
        state = await rpc.client.request("get_state")
        data = state.get("data")
        if (
            not isinstance(data, dict)
            or data.get("sessionId") != header["id"]
            or data.get("sessionFile") != str(path)
            or data.get("isStreaming") is not False
        ):
            raise ValueError("Pi canonical owner identity or idle state mismatch")
        temporary = LiveOwner(
            receipt.chat_id,
            receipt.thread_id,
            str(path),
            LiveRunner(object(), rpc),
            "recovery inspection",
        )
        await self.reconcile(temporary)
        updated = await self.inbox.get(receipt.id)
        if updated.state in ("delivered", "considered", "deferred"):
            await self._reply(
                receipt.chat_id,
                receipt.thread_id,
                receipt.message_id,
                f"Receipt #{receipt.message_id} is {updated.state}; no duplicate retry sent.",
            )
            return False
        await self.inbox.confirm_retry(receipt.id, receipt.session_key)
        await self.inbox.mark_uncertain(
            receipt.id,
            "Explicit user-confirmed retry starting; duplicate effects possible",
        )
        return True

    async def inspect_initial_retry(
        self,
        intent: InitialIntent,
        rpc: PiRpcRun | None,
        cwd: Path,
        live_root: Path,
    ) -> str:
        """Return fresh/retry/continue only after scoped verification where possible."""
        path = Path(intent.session_key)
        if (
            not path.is_absolute()
            or path.resolve() != path
            or Path(intent.cwd).resolve() != cwd.resolve()
            or path.parent != (live_root / "pi-live-sessions").resolve()
            or not path.name.startswith(f"{intent.chat_id}-{intent.thread_id}-")
            or path.suffix != ".jsonl"
        ):
            raise ValueError("Initial Pi intent path or project mismatch")
        if not path.exists():
            if rpc is not None:
                raise ValueError("Unexpected owner for absent initial session")
            return "fresh"
        if rpc is None:
            raise ValueError("Canonical Pi owner required for existing initial session")
        header = _session_header(path)
        if (
            header is None
            or not isinstance(header.get("id"), str)
            or not isinstance(header.get("cwd"), str)
            or Path(header["cwd"]).resolve() != cwd.resolve()
            or Path(rpc.client.session_path).resolve() != path
        ):
            raise ValueError("Initial Pi session identity mismatch")
        response = await rpc.client.request("get_state")
        data = response.get("data")
        if (
            not isinstance(data, dict)
            or data.get("sessionId") != header["id"]
            or data.get("sessionFile") != str(path)
            or data.get("isStreaming") is not False
        ):
            raise ValueError(
                "Initial Pi canonical owner identity or idle state mismatch"
            )
        messages = (await rpc.client.request("get_messages")).get("data")
        if not isinstance(messages, dict) or not isinstance(
            messages.get("messages"), list
        ):
            raise ValueError("Initial Pi messages unavailable")
        observed = any(
            isinstance(entry, dict)
            and entry.get("role") == "user"
            and (content := _text(entry))
            and (content == intent.prompt or content.endswith("\n\n" + intent.prompt))
            for entry in messages["messages"]
        )
        return "continue" if observed else "retry"

    async def notify_recovery(self) -> None:
        await self._send_replies()
        for intent in await self.inbox.initial_unresolved():
            try:
                await self._reply(
                    intent.chat_id,
                    intent.thread_id,
                    intent.message_id,
                    f"Initial Pi task #{intent.message_id}: {intent.state} after restart; no automatic prompt replay. "
                    "Pi may already have acted. In this exact topic use "
                    f"/update retry-initial {intent.message_id} confirm for canonical inspection and possible duplicate-risk continuation, "
                    f"or /update defer-initial {intent.message_id} <reason> to explicitly abandon the initial task.",
                )
            except Exception as exc:  # noqa: BLE001 - other topics remain available
                logger.warning(
                    "Initial Pi recovery notice failed: %s", type(exc).__name__
                )
        for receipt in await self.inbox.unresolved_all():
            try:
                await self._reply(
                    receipt.chat_id,
                    receipt.thread_id,
                    receipt.message_id,
                    f"Receipt #{receipt.message_id}: {receipt.state}; outcome not confirmed after restart. "
                    "Pi may have applied it. No automatic retry. Duplicate effects possible. "
                    "Uncertain Pi-accepted instructions cannot be deferred without proof they stayed local. "
                    f"/update retry {receipt.message_id} confirm starts a canonical-owner inspection before any retry prompt.",
                )
            except Exception as exc:  # noqa: BLE001 - one topic must not stop polling others
                logger.warning("Live recovery notice failed: %s", type(exc).__name__)

    async def recover_receipt(
        self,
        chat: int,
        thread: int,
        message: int,
        session: str,
        command: str,
        *,
        spawn_idle_retry: Callable[[Receipt], None] | None = None,
    ) -> bool:
        match = re.fullmatch(
            r"(defer|retry) (\d+)(?: (.+))?", command.strip(), re.I | re.S
        )
        if match is None:
            return False
        action, number, reason = match.groups()
        receipt_id = (chat, thread, int(number))
        try:
            receipt = await self.inbox.get(receipt_id)
        except KeyError:
            return False
        if receipt.session_key != session:
            return False
        if action.lower() == "defer":
            if not reason or not reason.strip():
                await self._reply(
                    chat, thread, message, "Deferral needs an explicit reason."
                )
                return True
            owner = self.owner(chat, thread, session)

            async def do_defer() -> bool:
                latest = await self.inbox.get(receipt_id)
                if latest.state == "submitted":
                    return False  # Pi accepted this steer; its queue is not locally removable.
                if latest.state == "uncertain" and (
                    latest.ever_attempted is not False
                    or latest.reason != "Only locally queued; not submitted to Pi"
                    or owner is None
                    or not isinstance(owner.main, LiveRunner)
                    or (await self.inbox.delivery_text(receipt_id))
                    not in owner.main._followups
                ):
                    return False  # An uncertain RPC outcome cannot be withdrawn safely.
                if owner is not None and isinstance(owner.main, LiveRunner):
                    delivery = await self.inbox.delivery_text(receipt_id)
                    if owner.main._inflight_followup == delivery:
                        return False  # Prompt may already be in flight.
                    owner.main._followups = [
                        item for item in owner.main._followups if item != delivery
                    ]
                await self.inbox.mark_deferred(
                    receipt_id, f"Explicit user deferral: {reason.strip()}"
                )
                return True

            try:
                if owner is not None:
                    async with owner.lock:
                        deferred = await do_defer()
                else:
                    deferred = await do_defer()
            except ValueError:
                await self._reply(
                    chat, thread, message, "Receipt is already resolved; no change."
                )
                return True
            if not deferred:
                await self._reply(
                    chat,
                    thread,
                    message,
                    "Pi may already have accepted this instruction; deferral refused. Reconcile the exact main-session receipt first.",
                )
                return True
            await self._reply(
                chat,
                thread,
                message,
                f"Receipt #{number} explicitly deferred; it was not considered.",
            )
            owner = self.owner(chat, thread, session)
            if owner is not None and not owner.closing:
                await self.flush(owner)
            return True
        if reason != "confirm":
            await self._reply(
                chat,
                thread,
                message,
                f"Retry can duplicate effects. Explicitly confirm with /update retry {number} confirm.",
            )
            return True
        owner = self.owner(chat, thread, session)
        if owner is None or owner.closing:
            if owner is None and spawn_idle_retry is not None:
                spawn_idle_retry(receipt)
                await self._reply(
                    chat,
                    thread,
                    message,
                    f"Receipt #{number}: inspecting canonical Pi owner before retry. "
                    "If still unobserved, your explicit confirmation authorizes a duplicate-risk prompt.",
                )
            else:
                await self._reply(
                    chat,
                    thread,
                    message,
                    "Pi is still finalizing; retry refused until ownership settles.",
                )
            return True
        try:
            async with owner.lock:
                if owner.closing:
                    result = "Pi is finalizing; retry refused."
                else:
                    await self._reconcile_locked(owner)
                    receipt = await self.inbox.get(receipt_id)
                    pending = await self.inbox.unresolved_for_topic(
                        chat, thread, session
                    )
                    if receipt.state in ("delivered", "considered", "deferred"):
                        result = (
                            f"Receipt #{number}: {receipt.state}; retry refused. "
                            "Delivery is not consideration."
                        )
                    elif not pending or pending[0].id != receipt_id:
                        result = "An earlier receipt is unresolved; retry it or defer it first."
                    elif isinstance(owner.main, LiveRunner) and (
                        (delivery := await self.inbox.delivery_text(receipt_id))
                        in owner.main._followups
                        or owner.main._inflight_followup == delivery
                    ):
                        result = "Follow-up already queued/in flight; retry refused to avoid a duplicate."
                    else:
                        await self.inbox.confirm_retry(receipt_id, session)
                        await self._drain_locked(owner)
                        result = (
                            f"Explicit retry of #{number} accepted; "
                            "duplicate effects remain possible. Check the receipt state."
                        )
        except Exception:  # noqa: BLE001 - observation is required before retry
            result = "Pi observation or receipt transition failed; retry refused."
        await self._reply(chat, thread, message, result)
        return True

    async def handle(
        self,
        chat: int,
        thread: int,
        message: int,
        session: str,
        text: str,
        *,
        spawn_quick: Callable[[Awaitable[None]], None] | None = None,
    ) -> bool:
        owner = self.owner(chat, thread, session)
        if owner is None or owner.closing:
            return False
        # Questions and corrections belong to the same contextual Pi owner.
        # Do not split on punctuation or answer from transport bookkeeping.
        update = _conversation_text(text)
        if not update:
            return True
        if self._worker_tg is not None:
            async with owner.accept_lock:
                if owner.closing:
                    return False
                receipt = await self.inbox.receive(
                    chat, thread, message, session, update
                )
                if receipt.state == "received":
                    owner.drain_event.set()
            return True
        async with owner.lock:
            if owner.closing:
                return False
            receipt = await self.inbox.receive(chat, thread, message, session, update)
            if receipt.state == "received":
                await self._drain_locked(owner)
        return True

    async def _submit_locked(self, owner: LiveOwner, receipt: Receipt) -> None:
        # The finish flush can beat the incoming handler after durable receive.
        if (await self.inbox.get(receipt.id)).state != "received":
            return
        await self.inbox.mark_uncertain(
            receipt.id, "RPC outcome may be unknown until main-session observation"
        )
        try:
            delivery = await self.inbox.delivery_text(receipt.id)
            if isinstance(owner.main, LiveRunner):
                result = await owner.main.steer(
                    delivery, before_rpc=lambda: self.inbox.mark_attempted(receipt.id)
                )
            else:
                await self.inbox.mark_attempted(receipt.id)
                result = await owner.main.steer(delivery)
            if result == "local_pending":
                await self.inbox.mark_local_pending(receipt.id)
            if result not in ("queued", "handled", "local_pending"):
                raise RuntimeError("invalid RPC disposition")
            if result != "local_pending":
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

    async def followup_started(self, owner: LiveOwner, text: str) -> None:
        async with owner.lock:
            for receipt in await self.inbox.pending(owner.session_key):
                if await self.inbox.delivery_text(receipt.id) == text:
                    await self.inbox.mark_submitted(receipt.id)
                    return
            raise ValueError("Unrecognized local follow-up receipt")

    async def submit(self, owner: LiveOwner, receipt: Receipt) -> None:
        await self.flush(owner)

    async def _drain_locked(self, owner: LiveOwner) -> None:
        for receipt in await self.inbox.pending(owner.session_key):
            if receipt.state == "uncertain":
                break  # Ambiguous acceptance gates every later update in this session.
            if receipt.state == "received":
                await self._submit_locked(owner, receipt)
                if (await self.inbox.get(receipt.id)).state == "uncertain":
                    break

    async def flush(self, owner: LiveOwner) -> None:
        async with owner.lock:
            await self._drain_locked(owner)

    async def finalize(self, owner: LiveOwner) -> bool:
        # Establish the intake boundary before a long RPC observation. Never
        # hold the acceptance mutex while waiting for Pi or Telegram network.
        async with owner.accept_lock:
            owner.closing = True
        async with owner.lock:
            try:
                await self._reconcile_locked(owner)
            except Exception as exc:  # noqa: BLE001 - never publish success on unknown state
                logger.warning("Live final observation failed: %s", type(exc).__name__)
                owner.closing = True
                return False
            await self._drain_locked(owner)
            if isinstance(owner.main, LiveRunner) and owner.main._followups:
                async with owner.accept_lock:
                    owner.closing = (
                        False  # The same owner must drain local continuations.
                    )
                return True
            pending = await self.inbox.pending(owner.session_key)
            owner.closing = True
            if pending:
                for receipt in pending:
                    await self._reply(
                        receipt.chat_id,
                        receipt.thread_id,
                        receipt.message_id,
                        f"Receipt #{receipt.message_id}: {receipt.state}. Result incomplete; Pi may have applied this update. Do not retry without explicit confirmation; duplicate effects are possible.",
                    )
                return False
            return True

    async def reconcile(self, owner: LiveOwner) -> None:
        async with owner.lock:
            await self._reconcile_locked(owner)

    async def _reconcile_locked(self, owner: LiveOwner) -> None:
        if not owner.identity_verified:
            return  # provisional new Pi session: no receipt attestation before header
        entries = await owner.main.get_messages()
        if not isinstance(entries, list):
            raise ValueError("Pi get_messages returned no message list")
        await self.inbox.reconcile(owner.session_key, entries)
        receipts = {r.marker: r for r in await self.inbox.pending(owner.session_key)}
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
                    await self.inbox.mark_deferred(
                        receipt.id, reason, reply_pending=True
                    )
                receipts.pop(marker, None)
        await self._send_replies(owner.session_key)
        if isinstance(owner.main, LiveRunner):
            owner.main.delivered_markers.update(
                receipt.marker
                for receipt in await self.inbox.for_session(owner.session_key)
                if receipt.state in ("considered", "deferred")
                and not receipt.reply_pending
            )

    async def _send_replies(self, session_key: str | None = None) -> None:
        # Serialize recovery and live observation: the same answer must not be
        # posted concurrently. Persist the answer before touching Telegram.
        async with self._reply_lock:
            for receipt in await self.inbox.pending_replies(session_key):
                try:
                    with anyio.fail_after(5):
                        await self._reply(
                            receipt.chat_id,
                            receipt.thread_id,
                            receipt.message_id,
                            receipt.reason or "",
                        )
                    await self.inbox.mark_reply_sent(receipt.id)
                except Exception:  # noqa: BLE001 - keep answer durable for retry
                    logger.warning(
                        "Live answer delivery failed; answer retained for retry"
                    )
