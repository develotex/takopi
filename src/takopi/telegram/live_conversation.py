"""Opt-in, exact-scope Pi live messages. No quick answer enters the writable session."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile

import anyio
from pathlib import Path
from uuid import uuid4
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ..model import ActionEvent, CompletedEvent, ResumeToken, StartedEvent, TakopiEvent
from ..runner_bridge import RunningTask
from ..transport import MessageRef
from ..runners.pi_rpc import PiRpcClient, PiRpcRun
from ..runners.pi import _default_session_dir
from .live_inbox import InitialIntent, LiveInbox, Receipt

logger = logging.getLogger(__name__)


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
        self.settlement_ok = False
        self.final_failure = "Live update delivery is uncertain; final result is incomplete. Check the scoped receipt before retrying."

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
        next_prompt = prompt if self.initial_recovery else guidance + prompt
        next_resume = resume
        queued_text: str | None = None
        for iteration in range(
            9
        ):  # Bound settlement drain even if updates arrive continuously.
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
            if not self._followups:
                if completed is not None:
                    yield completed
                return
            if iteration == 8:
                raise RuntimeError(
                    "Live Pi settlement drain exceeded eight follow-ups; receipt state retained"
                )
            next_prompt = self._followups.pop(0)
            queued_text = next_prompt
            self._inflight_followup = queued_text
            next_resume = None

    async def steer(self, text: str) -> str:
        # Never await StartedEvent here: the Telegram poller must remain responsive
        # while the initial RPC prompt is still acquiring its session identity.
        if not self.rpc._active:
            # The streaming owner, not this caller, renders the continuation.
            self._followups.append(text)
            return "local_pending"
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
    running_task: RunningTask | None = None
    identity_verified: bool = True


def _classify(text: str) -> tuple[str | None, str | None]:
    if text.startswith("/update "):
        return None, text[len("/update ") :].strip() or None
    if text.startswith("/обнови "):
        return None, text[len("/обнови ") :].strip() or None
    # Explicit requests phrased as questions still change the main task.
    if text.lower().startswith(
        (
            "can you ",
            "could you ",
            "will you ",
            "would you ",
            "можешь ",
            "сможешь ",
            "не мог бы ",
            "не могла бы ",
        )
    ):
        return None, text.strip()
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
    if text.lower().startswith(("maybe ", "perhaps ", "возможно ", "может быть ")):
        return None, None
    # An ordinary statement in the active task topic is potentially relevant;
    # retaining it is safer than silently dropping an unrecognized imperative.
    return None, text.strip() or None


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
        question, update = _classify(text)
        if update:
            receipt = await self.inbox.receive(chat, thread, message, session, update)
            await self._reply(
                chat,
                thread,
                message,
                f"Instruction stored ({receipt.state}); Pi startup pending, not yet considered.",
            )
        if question:
            await self._reply(
                chat,
                thread,
                message,
                "Pi is starting; no main-task progress has been observed yet.",
            )
        if question is None and update is None:
            await self._reply(
                chat,
                thread,
                message,
                "Please clarify or use /update <text>; nothing was stored.",
            )

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
                    f"In this topic use /update defer {receipt.message_id} <reason> to explicitly skip; "
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
                if owner is not None and isinstance(owner.main, LiveRunner):
                    delivery = await self.inbox.delivery_text(receipt_id)
                    if owner.main._inflight_followup == delivery:
                        return (
                            False  # Prompt may be in-flight; cannot promise deferral.
                        )
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
                    "Pi prompt is in flight; deferral refused. Check observed receipt before acting.",
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
            quick = self._answer_question(owner, chat, thread, message, question)
            if spawn_quick is None:
                await quick
            else:
                spawn_quick(quick)
        return True

    async def _answer_question(
        self, owner: LiveOwner, chat: int, thread: int, message: int, question: str
    ) -> None:
        try:
            async with asyncio.timeout(self._timeout):
                async with owner.quick_slots:
                    statuses = await self.inbox.for_session(owner.session_key)
                    counts = ", ".join(
                        f"{state}: {sum(r.state == state for r in statuses)}"
                        for state in (
                            "received",
                            "uncertain",
                            "submitted",
                            "delivered",
                            "considered",
                            "deferred",
                        )
                    )
                    observed = (
                        owner.progress
                        if re.fullmatch(
                            r"(command|tool|file_change|web_search|subagent|note|turn|warning|telemetry): (started|updated|completed)",
                            owner.progress,
                        )
                        else "No public progress event observed."
                    )
                    snapshot = (
                        f"Observed public phase: {observed}\nReceipt counts: {counts}"
                    )
                    answer = await self._answer(question, snapshot)
        except Exception:  # noqa: BLE001 - quick reply must not affect main run
            answer = "Quick answer unavailable; the main task has not been interrupted."
        await self._reply(chat, thread, message, answer)

    async def _submit_locked(self, owner: LiveOwner, receipt: Receipt) -> None:
        # The finish flush can beat the incoming handler after durable receive.
        if (await self.inbox.get(receipt.id)).state != "received":
            return
        await self.inbox.mark_uncertain(
            receipt.id, "RPC outcome may be unknown until main-session observation"
        )
        try:
            result = await owner.main.steer(await self.inbox.delivery_text(receipt.id))
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
        async with owner.lock:
            try:
                await self._reconcile_locked(owner)
            except Exception as exc:  # noqa: BLE001 - never publish success on unknown state
                logger.warning("Live final observation failed: %s", type(exc).__name__)
                owner.closing = True
                return False
            await self._drain_locked(owner)
            if isinstance(owner.main, LiveRunner) and owner.main._followups:
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
        delivered = await self.inbox.reconcile(owner.session_key, entries)
        for receipt in delivered:
            await self._reply(
                receipt.chat_id,
                receipt.thread_id,
                receipt.message_id,
                "Инструкция доставлена Pi (delivered), но ещё не подтверждена как учтённая.",
            )
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
                    await self.inbox.mark_deferred(receipt.id, reason)
                await self._reply(
                    receipt.chat_id,
                    receipt.thread_id,
                    receipt.message_id,
                    f"Update {action}: {reason}",
                )
                receipts.pop(marker, None)
