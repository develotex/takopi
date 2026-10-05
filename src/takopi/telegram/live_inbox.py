"""Durable per-session Telegram updates; RPC acceptance is not evidence of delivery.

Only the owner of the resolved Pi session may reconcile its user-message stream.
An uncertain command outcome requires inspection of that stream before any retry.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import msgspec

from ..logging import get_logger
from .state_store import JsonStateStore

logger = get_logger(__name__)
STATE_VERSION = 1
STATE_FILENAME = "telegram_live_inbox.json"
type ReceiptId = tuple[int, int, int]
type InitialState = Literal["scheduled", "uncertain", "completed", "deferred"]
type ReceiptState = Literal[
    "received", "uncertain", "submitted", "delivered", "considered", "deferred"
]


def resolve_inbox_path(config_path: Path) -> Path:
    return config_path.with_name(STATE_FILENAME)


@dataclass(frozen=True, slots=True)
class Receipt:
    chat_id: int
    thread_id: int
    message_id: int
    session_key: str
    text: str
    sequence: int
    marker: str
    state: ReceiptState = "received"
    reason: str | None = None

    @property
    def id(self) -> ReceiptId:
        return (self.chat_id, self.thread_id, self.message_id)


@dataclass(frozen=True, slots=True)
class InitialIntent:
    chat_id: int
    thread_id: int
    message_id: int
    session_key: str
    prompt: str
    cwd: str
    state: InitialState = "scheduled"
    reason: str | None = None

    @property
    def id(self) -> ReceiptId:
        return (self.chat_id, self.thread_id, self.message_id)


class _InitialRecord(msgspec.Struct):
    chat_id: int
    thread_id: int
    message_id: int
    session_key: str
    prompt: str
    cwd: str
    state: InitialState
    reason: str | None = None


class _Record(msgspec.Struct):
    chat_id: int
    thread_id: int
    message_id: int
    session_key: str
    text: str
    sequence: int
    marker: str
    state: ReceiptState
    reason: str | None


class _State(msgspec.Struct):
    version: int
    next_sequence: int
    receipts: list[_Record]
    initial_intents: list[_InitialRecord] = msgspec.field(default_factory=list)


def _fresh_state() -> _State:
    return _State(version=STATE_VERSION, next_sequence=1, receipts=[])


def _receipt(record: _Record) -> Receipt:
    return Receipt(**msgspec.to_builtins(record))


def _record(state: _State, receipt_id: ReceiptId) -> _Record | None:
    return next(
        (
            r
            for r in state.receipts
            if (r.chat_id, r.thread_id, r.message_id) == receipt_id
        ),
        None,
    )


def _delivery_text(record: _Record) -> str:
    return f"[takopi-live:{record.marker}]\n{record.text}"


class LiveInbox(JsonStateStore[_State]):
    """Async state store scoped by both Telegram identity and exact Pi session key.

    ``pending`` includes uncertain and submitted receipts for status/finalization,
    not as permission to blindly retry them. Reconcile against user entries from
    the *same* main Pi session before retrying any indeterminate submission.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(
            path,
            version=STATE_VERSION,
            state_type=_State,
            state_factory=_fresh_state,
            log_prefix="telegram.live_inbox",
            logger=logger,
        )

    def _load_locked(self) -> None:
        # The generic JSON store replaces corrupt state with an empty state. For an
        # inbox that would silently erase unacknowledged updates: fail closed.
        if self._path.exists():
            try:
                state = msgspec.json.decode(self._path.read_bytes(), type=_State)
                if state.version != STATE_VERSION:
                    raise ValueError(f"unsupported version {state.version}")
                seen: set[ReceiptId] = set()
                sequences: set[int] = set()
                markers: set[str] = set()
                for item in state.receipts:
                    ident = (item.chat_id, item.thread_id, item.message_id)
                    if (
                        ident in seen
                        or item.sequence in sequences
                        or item.marker in markers
                        or not item.session_key
                        or item.state
                        not in (
                            "received",
                            "uncertain",
                            "submitted",
                            "delivered",
                            "considered",
                            "deferred",
                        )
                    ):
                        raise ValueError("invalid or duplicate receipt")
                    seen.add(ident)
                    sequences.add(item.sequence)
                    markers.add(item.marker)
                if state.next_sequence <= max(sequences, default=0):
                    raise ValueError("invalid sequence cursor")
                seen_intents: set[ReceiptId] = set()
                active_topics: set[tuple[int, int]] = set()
                for intent in state.initial_intents:
                    ident = (intent.chat_id, intent.thread_id, intent.message_id)
                    topic = (intent.chat_id, intent.thread_id)
                    if (
                        ident in seen_intents
                        or not intent.session_key
                        or not intent.prompt
                        or not intent.cwd
                        or intent.state
                        not in ("scheduled", "uncertain", "completed", "deferred")
                        or (
                            intent.state in ("scheduled", "uncertain")
                            and topic in active_topics
                        )
                    ):
                        raise ValueError("invalid or conflicting initial Pi intent")
                    seen_intents.add(ident)
                    if intent.state in ("scheduled", "uncertain"):
                        active_topics.add(topic)
            except (OSError, ValueError, msgspec.DecodeError) as exc:
                raise ValueError(f"Cannot load live inbox {self._path}: {exc}") from exc
            self._state = state
        else:
            self._state = _fresh_state()
        self._mtime_ns = self._stat_mtime_ns()
        self._loaded = True

    def _save_locked(self) -> None:
        """Replace the entire record atomically and sync file and directory."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # A failed replace must not leave mutated in-memory receipts visible.
        self._loaded = False
        name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                delete=False,
            ) as handle:
                name = handle.name
                handle.write(msgspec.json.encode(self._state))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self._path)
            name = None
            if os.name == "posix":
                fd = os.open(self._path.parent, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            self._mtime_ns = self._stat_mtime_ns()
            self._loaded = True
        finally:
            if name is not None:
                Path(name).unlink(missing_ok=True)

    async def receive_initial(
        self,
        chat: int,
        thread: int,
        message: int,
        session_key: str,
        prompt: str,
        cwd: str,
    ) -> InitialIntent:
        if not session_key or not prompt.strip() or not cwd:
            raise ValueError("Initial Pi intent needs path, prompt and project cwd")
        async with self._lock:
            self._reload_locked_if_needed()
            for item in self._state.initial_intents:
                if (item.chat_id, item.thread_id, item.message_id) == (
                    chat,
                    thread,
                    message,
                ):
                    if (item.session_key, item.prompt, item.cwd) != (
                        session_key,
                        prompt,
                        cwd,
                    ):
                        raise ValueError("Initial Pi intent identity mismatch")
                    return InitialIntent(**msgspec.to_builtins(item))
                if (item.chat_id, item.thread_id) == (chat, thread) and item.state in (
                    "scheduled",
                    "uncertain",
                ):
                    raise ValueError("Topic already has unresolved initial Pi intent")
            item = _InitialRecord(
                chat, thread, message, session_key, prompt, cwd, "scheduled"
            )
            self._state.initial_intents.append(item)
            self._save_locked()
            return InitialIntent(**msgspec.to_builtins(item))

    async def initial_for_topic(self, chat: int, thread: int) -> InitialIntent | None:
        async with self._lock:
            self._reload_locked_if_needed()
            for item in reversed(self._state.initial_intents):
                if (item.chat_id, item.thread_id) == (chat, thread) and item.state in (
                    "scheduled",
                    "uncertain",
                ):
                    return InitialIntent(**msgspec.to_builtins(item))
            return None

    async def initial_by_id(self, ident: ReceiptId) -> InitialIntent | None:
        async with self._lock:
            self._reload_locked_if_needed()
            for item in self._state.initial_intents:
                if (item.chat_id, item.thread_id, item.message_id) == ident:
                    return InitialIntent(**msgspec.to_builtins(item))
            return None

    async def initial_unresolved(self) -> list[InitialIntent]:
        async with self._lock:
            self._reload_locked_if_needed()
            return [
                InitialIntent(**msgspec.to_builtins(item))
                for item in self._state.initial_intents
                if item.state in ("scheduled", "uncertain")
            ]

    async def _initial_transition(
        self,
        ident: ReceiptId,
        session_key: str,
        state: InitialState,
        reason: str | None = None,
    ) -> InitialIntent:
        async with self._lock:
            self._reload_locked_if_needed()
            item = next(
                (
                    v
                    for v in self._state.initial_intents
                    if (v.chat_id, v.thread_id, v.message_id) == ident
                ),
                None,
            )
            if (
                item is None
                or item.session_key != session_key
                or item.state not in ("scheduled", "uncertain")
            ):
                raise ValueError("Initial Pi intent scope or state mismatch")
            item.state = state
            item.reason = reason
            self._save_locked()
            return InitialIntent(**msgspec.to_builtins(item))

    async def mark_initial_uncertain(
        self, ident: ReceiptId, session_key: str
    ) -> InitialIntent:
        return await self._initial_transition(
            ident, session_key, "uncertain", "Pi prompt outcome may be unknown"
        )

    async def mark_initial_completed(
        self, ident: ReceiptId, session_key: str
    ) -> InitialIntent:
        return await self._initial_transition(ident, session_key, "completed")

    async def mark_initial_deferred(
        self, ident: ReceiptId, session_key: str, reason: str
    ) -> InitialIntent:
        if not reason.strip():
            raise ValueError("Deferral needs explicit reason")
        return await self._initial_transition(ident, session_key, "deferred", reason)

    async def receive(
        self,
        chat_id: int,
        thread_id: int,
        message_id: int,
        session_key: str,
        text: str,
    ) -> Receipt:
        if not session_key:
            raise ValueError("Live inbox requires an exact session key")
        async with self._lock:
            self._reload_locked_if_needed()
            found = _record(self._state, (chat_id, thread_id, message_id))
            if found is not None:
                if found.session_key != session_key:
                    raise ValueError(
                        "Telegram message already belongs to a different scope"
                    )
                return _receipt(found)
            item = _Record(
                chat_id,
                thread_id,
                message_id,
                session_key,
                text,
                self._state.next_sequence,
                uuid4().hex,
                "received",
                None,
            )
            self._state.receipts.append(item)
            self._state.next_sequence += 1
            self._save_locked()
            return _receipt(item)

    async def get(self, receipt_id: ReceiptId) -> Receipt:
        async with self._lock:
            self._reload_locked_if_needed()
            return _receipt(self._required(receipt_id))

    def _required(self, receipt_id: ReceiptId) -> _Record:
        item = _record(self._state, receipt_id)
        if item is None:
            raise KeyError(receipt_id)
        return item

    async def delivery_text(self, receipt_id: ReceiptId) -> str:
        async with self._lock:
            self._reload_locked_if_needed()
            return _delivery_text(self._required(receipt_id))

    async def for_session(self, session_key: str) -> list[Receipt]:
        async with self._lock:
            self._reload_locked_if_needed()
            return [
                _receipt(r)
                for r in self._state.receipts
                if r.session_key == session_key
            ]

    async def unresolved_for_topic(
        self, chat: int, thread: int, session_key: str
    ) -> list[Receipt]:
        async with self._lock:
            self._reload_locked_if_needed()
            return [
                _receipt(r)
                for r in self._state.receipts
                if r.chat_id == chat
                and r.thread_id == thread
                and r.session_key == session_key
                and r.state not in ("considered", "deferred")
            ]

    async def unresolved_all(self) -> list[Receipt]:
        async with self._lock:
            self._reload_locked_if_needed()
            return [
                _receipt(r)
                for r in self._state.receipts
                if r.state not in ("considered", "deferred")
            ]

    async def confirm_retry(self, receipt_id: ReceiptId, session_key: str) -> Receipt:
        """Explicit user-authorized retry only; absence is never evidence of failure."""
        async with self._lock:
            self._reload_locked_if_needed()
            item = self._required(receipt_id)
            if item.session_key != session_key or item.state not in (
                "uncertain",
                "submitted",
                "received",
            ):
                raise ValueError("Receipt scope or state does not permit retry")
            item.state = "received"
            item.reason = "Explicit user retry; duplicate effects possible"
            self._save_locked()
            return _receipt(item)

    async def pending(self, session_key: str) -> list[Receipt]:
        async with self._lock:
            self._reload_locked_if_needed()
            return [
                _receipt(r)
                for r in self._state.receipts
                if r.session_key == session_key
                and r.state not in ("considered", "deferred")
            ]

    async def _transition(
        self,
        receipt_id: ReceiptId,
        state: ReceiptState,
        *,
        reason: str | None = None,
        from_states: tuple[ReceiptState, ...],
        preserve_states: tuple[ReceiptState, ...] = (),
    ) -> Receipt:
        async with self._lock:
            self._reload_locked_if_needed()
            item = self._required(receipt_id)
            if item.state == state or item.state in preserve_states:
                return _receipt(item)
            if item.state not in from_states:
                raise ValueError(f"Cannot move receipt from {item.state} to {state}")
            item.state = state
            item.reason = reason
            self._save_locked()
            return _receipt(item)

    async def mark_uncertain(self, receipt_id: ReceiptId, reason: str) -> Receipt:
        if not reason.strip():
            raise ValueError("Uncertain receipt needs a reason")
        # Call *before* submitting an RPC command: a lost response might mean
        # Pi accepted it even though the sender never learned its disposition.
        return await self._transition(
            receipt_id,
            "uncertain",
            reason=reason,
            from_states=("received", "submitted"),
        )

    async def mark_submitted(self, receipt_id: ReceiptId) -> Receipt:
        return await self._transition(
            receipt_id,
            "submitted",
            from_states=("received", "uncertain"),
            preserve_states=("delivered", "considered"),
        )

    async def mark_delivered(self, receipt_id: ReceiptId, user_entry: dict) -> Receipt:
        # A successful RPC steer is not evidence: require an observed main-session
        # user entry containing the exact marker and original text.
        async with self._lock:
            self._reload_locked_if_needed()
            item = self._required(receipt_id)
            if not self._matches(item, user_entry):
                raise ValueError("No matching main-session user entry")
            if item.state in ("considered", "deferred", "delivered"):
                return _receipt(item)
            item.state = "delivered"
            item.reason = None
            self._save_locked()
            return _receipt(item)

    @staticmethod
    def _matches(item: _Record, entry: dict) -> bool:
        if entry.get("role") != "user":
            return False
        content = entry.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return content == _delivery_text(item)

    async def reconcile(
        self, session_key: str, user_entries: list[dict]
    ) -> list[Receipt]:
        """Apply only positively observed deliveries; absence never proves non-delivery.

        The caller must load entries from exactly ``session_key``'s main session.
        Returns receipts newly observed in that session.
        """
        async with self._lock:
            self._reload_locked_if_needed()
            changed: list[Receipt] = []
            for item in self._state.receipts:
                if item.session_key != session_key or item.state in (
                    "delivered",
                    "considered",
                    "deferred",
                ):
                    continue
                if any(self._matches(item, entry) for entry in user_entries):
                    item.state = "delivered"
                    item.reason = None
                    changed.append(_receipt(item))
            if changed:
                self._save_locked()
            return changed

    async def mark_considered(self, receipt_id: ReceiptId, reason: str) -> Receipt:
        if not reason.strip():
            raise ValueError("Consideration needs explicit main-agent evidence")
        return await self._transition(
            receipt_id, "considered", reason=reason, from_states=("delivered",)
        )

    async def mark_deferred(self, receipt_id: ReceiptId, reason: str) -> Receipt:
        if not reason.strip():
            raise ValueError("Deferral needs an explicit reason")
        return await self._transition(
            receipt_id,
            "deferred",
            reason=reason,
            from_states=("received", "uncertain", "submitted", "delivered"),
        )
