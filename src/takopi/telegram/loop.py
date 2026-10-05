from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import anyio
from anyio.abc import TaskGroup

from ..config import ConfigError
from ..config_watch import ConfigReload, watch_config as watch_config_changes
from ..commands import list_command_ids
from ..directives import DirectiveError
from ..logging import get_logger
from ..model import EngineId, ResumeToken
from ..runners.run_options import EngineRunOptions, apply_run_options
from ..runners.pi import PiRunner
from ..runners.pi_rpc import PiRpcRun, enable_live_claims
from .live_conversation import (
    LiveConversationService,
    LiveOwner,
    LiveRunner,
    live_route_key,
    legacy_session_busy,
    live_session_path,
    quick_pi_answer,
    resolve_legacy_session,
    session_header_id,
    verify_bound_session,
    verify_fresh_session,
    verify_legacy_session,
    verify_provisional_header,
)
from .live_inbox import InitialIntent, LiveInbox, Receipt, resolve_inbox_path
from ..scheduler import ThreadJob, ThreadScheduler
from ..progress import ProgressTracker
from ..settings import TelegramTransportSettings
from ..transport import MessageRef, SendOptions
from ..transport_runtime import ResolvedMessage
from ..context import RunContext
from ..ids import RESERVED_CHAT_COMMANDS
from .bridge import (
    CANCEL_CALLBACK_DATA,
    STEER_CALLBACK_DATA,
    TelegramBridgeConfig,
    send_plain,
)
from .commands.cancel import (
    handle_callback_cancel,
    handle_callback_steer,
    handle_cancel,
)
from .commands.file_transfer import FILE_PUT_USAGE
from .commands.handlers import (
    dispatch_command,
    handle_agent_command,
    handle_chat_ctx_command,
    handle_chat_new_command,
    handle_ctx_command,
    handle_file_command,
    handle_file_put_default,
    handle_media_group,
    handle_model_command,
    handle_new_command,
    handle_reasoning_command,
    handle_topic_command,
    handle_trigger_command,
    parse_callback_data,
    parse_slash_command,
    get_reserved_commands,
    run_engine,
    save_file_put,
    set_command_menu,
    should_show_resume_line,
)
from .api_models import Sticker
from .commands.parse import is_cancel_command
from .commands.reply import make_reply
from .context import _merge_topic_context, _usage_ctx_set, _usage_topic
from .topics import (
    _maybe_rename_topic,
    _resolve_topics_scope,
    _topic_icon_choice,
    _topic_key,
    _topics_chat_allowed,
    _topics_chat_project,
    _validate_topics_setup,
)
from .client import poll_incoming
from .chat_prefs import ChatPrefsStore, resolve_prefs_path
from .chat_sessions import ChatSessionStore, resolve_sessions_path
from .engine_overrides import merge_overrides
from .engine_defaults import resolve_engine_for_message
from .topic_state import TopicStateStore, resolve_state_path
from .trigger_mode import resolve_trigger_mode, should_trigger_run
from .types import (
    TelegramCallbackQuery,
    TelegramIncomingMessage,
    TelegramIncomingUpdate,
)
from .voice import transcribe_voice

logger = get_logger(__name__)

__all__ = ["poll_updates", "run_main_loop", "send_with_resume"]

ForwardKey = tuple[int, int, int]
MessageKey = tuple[int, int]
_SEEN_MESSAGES_LIMIT = 2048
_SEEN_UPDATES_LIMIT = 4096

_handle_file_put_default = handle_file_put_default


def _chat_session_key(
    msg: TelegramIncomingMessage, *, store: ChatSessionStore | None
) -> tuple[int, int | None] | None:
    if store is None or msg.thread_id is not None:
        return None
    if msg.chat_type == "private":
        return (msg.chat_id, None)
    if msg.sender_id is None:
        return None
    return (msg.chat_id, msg.sender_id)


def _callback_message_thread_id(update: TelegramCallbackQuery) -> int | None:
    if update.raw is None:
        return None
    message = update.raw.get("message")
    if not isinstance(message, dict):
        return None
    thread_id = message.get("message_thread_id")
    return thread_id if isinstance(thread_id, int) else None


def _callback_chat_type(update: TelegramCallbackQuery) -> str | None:
    if update.raw is None:
        return None
    message = update.raw.get("message")
    if not isinstance(message, dict):
        return None
    chat = message.get("chat")
    if not isinstance(chat, dict):
        return None
    chat_type = chat.get("type")
    return chat_type if isinstance(chat_type, str) else None


def _callback_message(update: TelegramCallbackQuery) -> TelegramIncomingMessage:
    return TelegramIncomingMessage(
        transport=update.transport,
        chat_id=update.chat_id,
        message_id=update.message_id,
        text=update.data or "",
        reply_to_message_id=None,
        reply_to_text=None,
        sender_id=update.sender_id,
        thread_id=_callback_message_thread_id(update),
        chat_type=_callback_chat_type(update),
        raw=update.raw,
        update_id=update.update_id,
    )


async def _resolve_engine_run_options(
    chat_id: int,
    thread_id: int | None,
    engine: EngineId,
    chat_prefs: ChatPrefsStore | None,
    topic_store: TopicStateStore | None,
) -> EngineRunOptions | None:
    topic_override = None
    if topic_store is not None and thread_id is not None:
        topic_override = await topic_store.get_engine_override(
            chat_id, thread_id, engine
        )
    chat_override = None
    if chat_prefs is not None:
        chat_override = await chat_prefs.get_engine_override(chat_id, engine)
    merged = merge_overrides(topic_override, chat_override)
    if merged is None:
        return None
    return EngineRunOptions(model=merged.model, reasoning=merged.reasoning)


def _allowed_chat_ids(cfg: TelegramBridgeConfig) -> set[int]:
    allowed = set(cfg.chat_ids or ())
    allowed.add(cfg.chat_id)
    allowed.update(cfg.runtime.project_chat_ids())
    allowed.update(cfg.allowed_user_ids)
    return allowed


async def _send_startup(cfg: TelegramBridgeConfig) -> None:
    from ..markdown import MarkdownParts
    from ..transport import RenderedMessage
    from .render import prepare_telegram

    logger.debug("startup.message", text=cfg.startup_msg)
    parts = MarkdownParts(header=cfg.startup_msg)
    text, entities = prepare_telegram(parts)
    message = RenderedMessage(text=text, extra={"entities": entities})
    sent = await cfg.exec_cfg.transport.send(
        channel_id=cfg.chat_id,
        message=message,
    )
    if sent is not None:
        logger.info("startup.sent", chat_id=cfg.chat_id)


def _dispatch_builtin_command(
    *,
    ctx: TelegramCommandContext,
    command_id: str,
) -> bool:
    cfg = ctx.cfg
    msg = ctx.msg
    args_text = ctx.args_text
    ambient_context = ctx.ambient_context
    topic_store = ctx.topic_store
    chat_prefs = ctx.chat_prefs
    resolved_scope = ctx.resolved_scope
    scope_chat_ids = ctx.scope_chat_ids
    reply = ctx.reply
    task_group = ctx.task_group
    if command_id == "file":
        if not cfg.files.enabled:
            handler = partial(
                reply,
                text="file transfer disabled; enable `[transports.telegram.files]`.",
            )
        else:
            handler = partial(
                handle_file_command,
                cfg,
                msg,
                args_text,
                ambient_context,
                topic_store,
            )
        task_group.start_soon(handler)
        return True

    if command_id == "ctx":
        topic_key = (
            _topic_key(msg, cfg, scope_chat_ids=scope_chat_ids)
            if cfg.topics.enabled and topic_store is not None
            else None
        )
        if topic_key is not None:
            handler = partial(
                handle_ctx_command,
                cfg,
                msg,
                args_text,
                topic_store,
                resolved_scope=resolved_scope,
                scope_chat_ids=scope_chat_ids,
            )
        else:
            handler = partial(
                handle_chat_ctx_command,
                cfg,
                msg,
                args_text,
                chat_prefs,
            )
        task_group.start_soon(handler)
        return True

    if cfg.topics.enabled and topic_store is not None:
        if command_id == "new":
            handler = partial(
                handle_new_command,
                cfg,
                msg,
                topic_store,
                resolved_scope=resolved_scope,
                scope_chat_ids=scope_chat_ids,
            )
        elif command_id == "topic":
            handler = partial(
                handle_topic_command,
                cfg,
                msg,
                args_text,
                topic_store,
                resolved_scope=resolved_scope,
                scope_chat_ids=scope_chat_ids,
            )
        else:
            handler = None
        if handler is not None:
            task_group.start_soon(handler)
            return True

    if command_id == "model":
        handler = partial(
            handle_model_command,
            cfg,
            msg,
            args_text,
            ambient_context,
            topic_store,
            chat_prefs,
            resolved_scope=resolved_scope,
            scope_chat_ids=scope_chat_ids,
        )
        task_group.start_soon(handler)
        return True

    if command_id == "agent":
        handler = partial(
            handle_agent_command,
            cfg,
            msg,
            args_text,
            ambient_context,
            topic_store,
            chat_prefs,
            resolved_scope=resolved_scope,
            scope_chat_ids=scope_chat_ids,
        )
        task_group.start_soon(handler)
        return True

    if command_id == "reasoning":
        handler = partial(
            handle_reasoning_command,
            cfg,
            msg,
            args_text,
            ambient_context,
            topic_store,
            chat_prefs,
            resolved_scope=resolved_scope,
            scope_chat_ids=scope_chat_ids,
        )
        task_group.start_soon(handler)
        return True

    if command_id == "trigger":
        handler = partial(
            handle_trigger_command,
            cfg,
            msg,
            args_text,
            ambient_context,
            topic_store,
            chat_prefs,
            resolved_scope=resolved_scope,
            scope_chat_ids=scope_chat_ids,
        )
        task_group.start_soon(handler)
        return True

    return False


async def _drain_backlog(cfg: TelegramBridgeConfig, offset: int | None) -> int | None:
    drained = 0
    while True:
        updates = await cfg.bot.get_updates(
            offset=offset,
            timeout_s=0,
            allowed_updates=["message", "callback_query"],
        )
        if updates is None:
            logger.info("startup.backlog.failed")
            return offset
        logger.debug("startup.backlog.updates", updates=updates)
        if not updates:
            if drained:
                logger.info("startup.backlog.drained", count=drained)
            return offset
        offset = updates[-1].update_id + 1
        drained += len(updates)


async def poll_updates(
    cfg: TelegramBridgeConfig,
    *,
    sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
) -> AsyncIterator[TelegramIncomingUpdate]:
    offset: int | None = None
    offset = await _drain_backlog(cfg, offset)
    await _send_startup(cfg)

    async for msg in poll_incoming(
        cfg.bot,
        chat_ids=lambda: _allowed_chat_ids(cfg),
        offset=offset,
        sleep=sleep,
    ):
        yield msg


@dataclass(slots=True)
class _MediaGroupState:
    messages: list[TelegramIncomingMessage]
    token: int = 0


@dataclass(slots=True)
class _PendingPrompt:
    msg: TelegramIncomingMessage
    text: str
    ambient_context: RunContext | None
    chat_project: str | None
    topic_key: tuple[int, int] | None
    chat_session_key: tuple[int, int | None] | None
    reply_ref: MessageRef | None
    reply_id: int | None
    is_voice_transcribed: bool
    forwards: list[tuple[int, str]]
    cancel_scope: anyio.CancelScope | None = None


@dataclass(frozen=True, slots=True)
class TelegramMsgContext:
    chat_id: int
    thread_id: int | None
    reply_id: int | None
    reply_ref: MessageRef | None
    topic_key: tuple[int, int] | None
    chat_session_key: tuple[int, int | None] | None
    stateful_mode: bool
    chat_project: str | None
    ambient_context: RunContext | None


@dataclass(frozen=True, slots=True)
class MessageClassification:
    text: str
    command_id: str | None
    args_text: str
    is_cancel: bool
    is_forward_candidate: bool
    is_media_group_document: bool


@dataclass(frozen=True, slots=True)
class TelegramCommandContext:
    cfg: TelegramBridgeConfig
    msg: TelegramIncomingMessage
    args_text: str
    ambient_context: RunContext | None
    topic_store: TopicStateStore | None
    chat_prefs: ChatPrefsStore | None
    resolved_scope: str | None
    scope_chat_ids: frozenset[int]
    reply: Callable[..., Awaitable[None]]
    task_group: TaskGroup


def _classify_message(
    msg: TelegramIncomingMessage, *, files_enabled: bool
) -> MessageClassification:
    text = msg.text
    command_id, args_text = parse_slash_command(text)
    is_forward_candidate = (
        _is_forwarded(msg.raw)
        and msg.document is None
        and msg.voice is None
        and msg.media_group_id is None
    )
    is_media_group_document = (
        files_enabled and msg.document is not None and msg.media_group_id is not None
    )
    return MessageClassification(
        text=text,
        command_id=command_id,
        args_text=args_text,
        is_cancel=is_cancel_command(text),
        is_forward_candidate=is_forward_candidate,
        is_media_group_document=is_media_group_document,
    )


@dataclass(slots=True)
class TelegramLoopState:
    running_tasks: RunningTasks
    topic_icon_stickers: list[Sticker] | None
    pending_prompts: dict[ForwardKey, _PendingPrompt]
    media_groups: dict[tuple[int, str], _MediaGroupState]
    command_ids: set[str]
    reserved_commands: set[str]
    reserved_chat_commands: set[str]
    transport_snapshot: dict[str, object] | None
    topic_store: TopicStateStore | None
    chat_session_store: ChatSessionStore | None
    chat_prefs: ChatPrefsStore | None
    resolved_topics_scope: str | None
    topics_chat_ids: frozenset[int]
    bot_username: str | None
    forward_coalesce_s: float
    media_group_debounce_s: float
    transport_id: str | None
    seen_update_ids: set[int]
    seen_update_order: deque[int]
    seen_message_keys: set[MessageKey]
    seen_messages_order: deque[MessageKey]


if TYPE_CHECKING:
    from ..runner_bridge import RunningTask, RunningTasks


_FORWARD_FIELDS = (
    "forward_origin",
    "forward_from",
    "forward_from_chat",
    "forward_from_message_id",
    "forward_sender_name",
    "forward_signature",
    "forward_date",
    "is_automatic_forward",
)


def _forward_key(msg: TelegramIncomingMessage) -> ForwardKey:
    return (msg.chat_id, msg.thread_id or 0, msg.sender_id or 0)


def _is_forwarded(raw: dict[str, object] | None) -> bool:
    if not isinstance(raw, dict):
        return False
    return any(raw.get(field) is not None for field in _FORWARD_FIELDS)


def _forward_fields_present(raw: dict[str, object] | None) -> list[str]:
    if not isinstance(raw, dict):
        return []
    return [field for field in _FORWARD_FIELDS if raw.get(field) is not None]


def _format_forwarded_prompt(forwarded: list[str], prompt: str) -> str:
    if not forwarded:
        return prompt
    separator = "\n\n"
    forward_block = separator.join(forwarded)
    if prompt.strip():
        return f"{prompt}{separator}{forward_block}"
    return forward_block


class ForwardCoalescer:
    def __init__(
        self,
        *,
        task_group: TaskGroup,
        debounce_s: float,
        sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
        dispatch: Callable[[_PendingPrompt], Awaitable[None]],
        pending: dict[ForwardKey, _PendingPrompt],
    ) -> None:
        self._task_group = task_group
        self._debounce_s = debounce_s
        self._sleep = sleep
        self._dispatch = dispatch
        self._pending = pending

    def cancel(self, key: ForwardKey) -> None:
        pending = self._pending.pop(key, None)
        if pending is None:
            return
        if pending.cancel_scope is not None:
            pending.cancel_scope.cancel()
        logger.debug(
            "forward.prompt.cancelled",
            chat_id=pending.msg.chat_id,
            thread_id=pending.msg.thread_id,
            sender_id=pending.msg.sender_id,
            message_id=pending.msg.message_id,
            forward_count=len(pending.forwards),
        )

    def schedule(self, pending: _PendingPrompt) -> None:
        if pending.msg.sender_id is None:
            logger.debug(
                "forward.prompt.bypass",
                chat_id=pending.msg.chat_id,
                thread_id=pending.msg.thread_id,
                sender_id=pending.msg.sender_id,
                message_id=pending.msg.message_id,
                reason="missing_sender",
            )
            self._task_group.start_soon(self._dispatch, pending)
            return
        if self._debounce_s <= 0:
            logger.debug(
                "forward.prompt.bypass",
                chat_id=pending.msg.chat_id,
                thread_id=pending.msg.thread_id,
                sender_id=pending.msg.sender_id,
                message_id=pending.msg.message_id,
                reason="disabled",
            )
            self._task_group.start_soon(self._dispatch, pending)
            return
        key = _forward_key(pending.msg)
        existing = self._pending.get(key)
        if existing is not None:
            if existing.cancel_scope is not None:
                existing.cancel_scope.cancel()
            if existing.forwards:
                pending.forwards = list(existing.forwards)
            logger.debug(
                "forward.prompt.replace",
                chat_id=pending.msg.chat_id,
                thread_id=pending.msg.thread_id,
                sender_id=pending.msg.sender_id,
                old_message_id=existing.msg.message_id,
                new_message_id=pending.msg.message_id,
                forward_count=len(pending.forwards),
            )
        self._pending[key] = pending
        logger.debug(
            "forward.prompt.schedule",
            chat_id=pending.msg.chat_id,
            thread_id=pending.msg.thread_id,
            sender_id=pending.msg.sender_id,
            message_id=pending.msg.message_id,
            debounce_s=self._debounce_s,
        )
        self._reschedule(key, pending)

    def attach_forward(self, msg: TelegramIncomingMessage) -> None:
        if msg.sender_id is None:
            logger.debug(
                "forward.message.ignored",
                chat_id=msg.chat_id,
                thread_id=msg.thread_id,
                sender_id=msg.sender_id,
                message_id=msg.message_id,
                reason="missing_sender",
            )
            return
        key = _forward_key(msg)
        pending = self._pending.get(key)
        if pending is None:
            logger.debug(
                "forward.message.ignored",
                chat_id=msg.chat_id,
                thread_id=msg.thread_id,
                sender_id=msg.sender_id,
                message_id=msg.message_id,
                reason="no_pending_prompt",
            )
            return
        text = msg.text
        if not text.strip():
            logger.debug(
                "forward.message.ignored",
                chat_id=msg.chat_id,
                thread_id=msg.thread_id,
                sender_id=msg.sender_id,
                message_id=msg.message_id,
                reason="empty_text",
            )
            return
        pending.forwards.append((msg.message_id, text))
        logger.debug(
            "forward.message.attached",
            chat_id=msg.chat_id,
            thread_id=msg.thread_id,
            sender_id=msg.sender_id,
            message_id=msg.message_id,
            prompt_message_id=pending.msg.message_id,
            forward_count=len(pending.forwards),
            forward_fields=_forward_fields_present(msg.raw),
            forward_date=msg.raw.get("forward_date") if msg.raw else None,
            message_date=msg.raw.get("date") if msg.raw else None,
            text_len=len(text),
        )
        self._reschedule(key, pending)

    def _reschedule(self, key: ForwardKey, pending: _PendingPrompt) -> None:
        if pending.cancel_scope is not None:
            pending.cancel_scope.cancel()
        pending.cancel_scope = None
        self._task_group.start_soon(self._debounce_prompt_run, key, pending)

    async def _debounce_prompt_run(
        self,
        key: ForwardKey,
        pending: _PendingPrompt,
    ) -> None:
        try:
            with anyio.CancelScope() as scope:
                pending.cancel_scope = scope
                await self._sleep(self._debounce_s)
        except anyio.get_cancelled_exc_class():
            return
        if self._pending.get(key) is not pending:
            return
        self._pending.pop(key, None)
        logger.debug(
            "forward.prompt.run",
            chat_id=pending.msg.chat_id,
            thread_id=pending.msg.thread_id,
            sender_id=pending.msg.sender_id,
            message_id=pending.msg.message_id,
            forward_count=len(pending.forwards),
            debounce_s=self._debounce_s,
        )
        await self._dispatch(pending)


@dataclass(frozen=True, slots=True)
class ResumeDecision:
    resume_token: ResumeToken | None
    handled_by_running_task: bool


class ResumeResolver:
    def __init__(
        self,
        *,
        cfg: TelegramBridgeConfig,
        task_group: TaskGroup,
        running_tasks: Mapping[MessageRef, object],
        enqueue_resume: Callable[
            [
                int,
                int,
                str,
                ResumeToken,
                RunContext | None,
                int | None,
                tuple[int, int | None] | None,
                MessageRef | None,
            ],
            Awaitable[None],
        ],
        topic_store: TopicStateStore | None,
        chat_session_store: ChatSessionStore | None,
    ) -> None:
        self._cfg = cfg
        self._task_group = task_group
        self._running_tasks = running_tasks
        self._enqueue_resume = enqueue_resume
        self._topic_store = topic_store
        self._chat_session_store = chat_session_store

    async def resolve(
        self,
        *,
        resume_token: ResumeToken | None,
        reply_id: int | None,
        chat_id: int,
        user_msg_id: int,
        thread_id: int | None,
        chat_session_key: tuple[int, int | None] | None,
        topic_key: tuple[int, int] | None,
        engine_for_session: EngineId,
        prompt_text: str,
    ) -> ResumeDecision:
        if resume_token is not None:
            return ResumeDecision(
                resume_token=resume_token, handled_by_running_task=False
            )
        if reply_id is not None:
            running_task = self._running_tasks.get(
                MessageRef(channel_id=chat_id, message_id=reply_id)
            )
            if running_task is not None:
                self._task_group.start_soon(
                    send_with_resume,
                    self._cfg,
                    self._enqueue_resume,
                    running_task,
                    chat_id,
                    user_msg_id,
                    thread_id,
                    chat_session_key,
                    prompt_text,
                )
                return ResumeDecision(resume_token=None, handled_by_running_task=True)
        if self._topic_store is not None and topic_key is not None:
            stored = await self._topic_store.get_session_resume(
                topic_key[0],
                topic_key[1],
                engine_for_session,
            )
            if stored is not None:
                resume_token = stored
        if (
            resume_token is None
            and self._chat_session_store is not None
            and chat_session_key is not None
        ):
            stored = await self._chat_session_store.get_session_resume(
                chat_session_key[0],
                chat_session_key[1],
                engine_for_session,
            )
            if stored is not None:
                resume_token = stored
        return ResumeDecision(resume_token=resume_token, handled_by_running_task=False)


class MediaGroupBuffer:
    def __init__(
        self,
        *,
        task_group: TaskGroup,
        debounce_s: float,
        sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
        cfg: TelegramBridgeConfig,
        chat_prefs: ChatPrefsStore | None,
        topic_store: TopicStateStore | None,
        bot_username: str | None,
        command_ids: Callable[[], set[str]],
        reserved_chat_commands: set[str],
        groups: dict[tuple[int, str], _MediaGroupState],
        run_prompt_from_upload: Callable[
            [TelegramIncomingMessage, str, ResolvedMessage], Awaitable[None]
        ],
        resolve_prompt_message: Callable[
            [TelegramIncomingMessage, str, RunContext | None],
            Awaitable[ResolvedMessage | None],
        ],
    ) -> None:
        self._task_group = task_group
        self._debounce_s = debounce_s
        self._sleep = sleep
        self._cfg = cfg
        self._chat_prefs = chat_prefs
        self._topic_store = topic_store
        self._bot_username = bot_username
        self._command_ids = command_ids
        self._reserved_chat_commands = reserved_chat_commands
        self._groups = groups
        self._run_prompt_from_upload = run_prompt_from_upload
        self._resolve_prompt_message = resolve_prompt_message

    def add(self, msg: TelegramIncomingMessage) -> None:
        if msg.media_group_id is None:
            return
        key = (msg.chat_id, msg.media_group_id)
        state = self._groups.get(key)
        if state is None:
            state = _MediaGroupState(messages=[])
            self._groups[key] = state
            self._task_group.start_soon(self._flush_media_group, key)
        state.messages.append(msg)
        state.token += 1

    async def _flush_media_group(self, key: tuple[int, str]) -> None:
        while True:
            state = self._groups.get(key)
            if state is None:
                return
            token = state.token
            await self._sleep(self._debounce_s)
            state = self._groups.get(key)
            if state is None:
                return
            if state.token != token:
                continue
            messages = list(state.messages)
            del self._groups[key]
            if not messages:
                return
            trigger_mode = await resolve_trigger_mode(
                chat_id=messages[0].chat_id,
                thread_id=messages[0].thread_id,
                chat_prefs=self._chat_prefs,
                topic_store=self._topic_store,
            )
            command_ids = self._command_ids()
            if trigger_mode == "mentions" and not any(
                should_trigger_run(
                    msg,
                    bot_username=self._bot_username,
                    runtime=self._cfg.runtime,
                    command_ids=command_ids,
                    reserved_chat_commands=self._reserved_chat_commands,
                )
                for msg in messages
            ):
                return
            await handle_media_group(
                self._cfg,
                messages,
                self._topic_store,
                self._run_prompt_from_upload,
                self._resolve_prompt_message,
            )
            return


def _diff_keys(old: dict[str, object], new: dict[str, object]) -> list[str]:
    keys = set(old) | set(new)
    return sorted(key for key in keys if old.get(key) != new.get(key))


async def _wait_for_resume(running_task) -> ResumeToken | None:
    if running_task.resume is not None:
        return running_task.resume
    resume: ResumeToken | None = None

    async with anyio.create_task_group() as tg:

        async def wait_resume() -> None:
            nonlocal resume
            await running_task.resume_ready.wait()
            resume = running_task.resume
            tg.cancel_scope.cancel()

        async def wait_done() -> None:
            await running_task.done.wait()
            tg.cancel_scope.cancel()

        tg.start_soon(wait_resume)
        tg.start_soon(wait_done)

    return resume


async def _send_queued_progress(
    cfg: TelegramBridgeConfig,
    *,
    chat_id: int,
    user_msg_id: int,
    thread_id: int | None,
    resume_token: ResumeToken,
    context: RunContext | None,
    steerable: bool,
) -> MessageRef | None:
    if cfg.exec_cfg.progress_updates == "none":
        return None
    tracker = ProgressTracker(engine=resume_token.engine)
    tracker.set_resume(resume_token)
    context_line = cfg.runtime.format_context_line(context)
    resume_formatter = None
    if should_show_resume_line(
        show_resume_line=cfg.show_resume_line,
        stateful_mode=cfg.session_mode == "chat",
        context=context,
    ):
        resume_formatter = cfg.runtime.resolve_runner(
            resume_token=resume_token,
            engine_override=None,
        ).runner.format_resume
    state = tracker.snapshot(
        resume_formatter=resume_formatter,
        context_line=context_line,
    )
    message = cfg.exec_cfg.presenter.render_progress(
        state,
        elapsed_s=0.0,
        label="queued" if steerable else "starting",
    )
    reply_ref = MessageRef(
        channel_id=chat_id,
        message_id=user_msg_id,
        thread_id=thread_id,
    )
    return await cfg.exec_cfg.transport.send(
        channel_id=chat_id,
        message=message,
        options=SendOptions(reply_to=reply_ref, notify=False, thread_id=thread_id),
    )


async def send_with_resume(
    cfg: TelegramBridgeConfig,
    enqueue: Callable[
        [
            int,
            int,
            str,
            ResumeToken,
            RunContext | None,
            int | None,
            tuple[int, int | None] | None,
            MessageRef | None,
        ],
        Awaitable[None],
    ],
    running_task,
    chat_id: int,
    user_msg_id: int,
    thread_id: int | None,
    session_key: tuple[int, int | None] | None,
    text: str,
) -> None:
    reply = partial(
        send_plain,
        cfg.exec_cfg.transport,
        chat_id=chat_id,
        user_msg_id=user_msg_id,
        thread_id=thread_id,
    )
    resume = await _wait_for_resume(running_task)
    if resume is None:
        await reply(
            text="resume token not ready yet; try replying to the final message.",
            notify=False,
        )
        return
    progress_ref = await _send_queued_progress(
        cfg,
        chat_id=chat_id,
        user_msg_id=user_msg_id,
        thread_id=thread_id,
        resume_token=resume,
        context=running_task.context,
        steerable=not running_task.done.is_set(),
    )
    await enqueue(
        chat_id,
        user_msg_id,
        text,
        resume,
        running_task.context,
        thread_id,
        session_key,
        progress_ref,
    )


async def run_main_loop(
    cfg: TelegramBridgeConfig,
    poller: Callable[
        [TelegramBridgeConfig], AsyncIterator[TelegramIncomingUpdate]
    ] = poll_updates,
    *,
    watch_config: bool | None = None,
    default_engine_override: str | None = None,
    transport_id: str | None = None,
    transport_config: TelegramTransportSettings | None = None,
    sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
) -> None:
    if not cfg.pi_live_conversation:
        await _run_main_loop_impl(
            cfg,
            poller,
            watch_config=watch_config,
            default_engine_override=default_engine_override,
            transport_id=transport_id,
            transport_config=transport_config,
            sleep=sleep,
        )
        return
    with enable_live_claims():
        await _run_main_loop_impl(
            cfg,
            poller,
            watch_config=watch_config,
            default_engine_override=default_engine_override,
            transport_id=transport_id,
            transport_config=transport_config,
            sleep=sleep,
        )


async def _run_main_loop_impl(
    cfg: TelegramBridgeConfig,
    poller: Callable[
        [TelegramBridgeConfig], AsyncIterator[TelegramIncomingUpdate]
    ] = poll_updates,
    *,
    watch_config: bool | None = None,
    default_engine_override: str | None = None,
    transport_id: str | None = None,
    transport_config: TelegramTransportSettings | None = None,
    sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
) -> None:
    state = TelegramLoopState(
        running_tasks={},
        topic_icon_stickers=None,
        pending_prompts={},
        media_groups={},
        command_ids={
            command_id.lower()
            for command_id in list_command_ids(allowlist=cfg.runtime.allowlist)
        },
        reserved_commands=get_reserved_commands(cfg.runtime),
        reserved_chat_commands=set(RESERVED_CHAT_COMMANDS),
        transport_snapshot=(
            transport_config.model_dump() if transport_config is not None else None
        ),
        topic_store=None,
        chat_session_store=None,
        chat_prefs=None,
        resolved_topics_scope=None,
        topics_chat_ids=frozenset(),
        bot_username=None,
        forward_coalesce_s=max(0.0, float(cfg.forward_coalesce_s)),
        media_group_debounce_s=max(0.0, float(cfg.media_group_debounce_s)),
        transport_id=transport_id,
        seen_update_ids=set(),
        seen_update_order=deque(),
        seen_message_keys=set(),
        seen_messages_order=deque(),
    )

    def refresh_topics_scope() -> None:
        if cfg.topics.enabled:
            (
                state.resolved_topics_scope,
                state.topics_chat_ids,
            ) = _resolve_topics_scope(cfg)
        else:
            state.resolved_topics_scope = None
            state.topics_chat_ids = frozenset()

    def refresh_commands() -> None:
        allowlist = cfg.runtime.allowlist
        state.command_ids = {
            command_id.lower() for command_id in list_command_ids(allowlist=allowlist)
        }
        state.reserved_commands = get_reserved_commands(cfg.runtime)

    try:
        config_path = cfg.runtime.config_path
        if config_path is not None:
            state.chat_prefs = ChatPrefsStore(resolve_prefs_path(config_path))
            logger.info(
                "chat_prefs.enabled",
                state_path=str(resolve_prefs_path(config_path)),
            )
        if cfg.session_mode == "chat":
            if config_path is None:
                raise ConfigError(
                    "session_mode=chat but config path is not set; cannot locate state file."
                )
            state.chat_session_store = ChatSessionStore(
                resolve_sessions_path(config_path)
            )
            cleared = await state.chat_session_store.sync_startup_cwd(Path.cwd())
            if cleared:
                logger.info(
                    "chat_sessions.cleared",
                    reason="startup_cwd_changed",
                    cwd=str(Path.cwd()),
                    state_path=str(resolve_sessions_path(config_path)),
                )
            logger.info(
                "chat_sessions.enabled",
                state_path=str(resolve_sessions_path(config_path)),
            )
        if cfg.topics.enabled:
            if config_path is None:
                raise ConfigError(
                    "topics enabled but config path is not set; cannot locate state file."
                )
            state.topic_store = TopicStateStore(resolve_state_path(config_path))
            await _validate_topics_setup(cfg)
            refresh_topics_scope()
            logger.info(
                "topics.enabled",
                scope=cfg.topics.scope,
                resolved_scope=state.resolved_topics_scope,
                state_path=str(resolve_state_path(config_path)),
            )
        await set_command_menu(cfg)
        try:
            me = await cfg.bot.get_me()
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "trigger_mode.bot_username.failed",
                error=str(exc),
                error_type=exc.__class__.__name__,
            )
            me = None
        if me is not None and me.username:
            state.bot_username = me.username.lower()
        else:
            logger.info("trigger_mode.bot_username.unavailable")
        async with anyio.create_task_group() as tg:
            poller_fn: Callable[
                [TelegramBridgeConfig], AsyncIterator[TelegramIncomingUpdate]
            ]
            if poller is poll_updates:
                poller_fn = cast(
                    Callable[
                        [TelegramBridgeConfig], AsyncIterator[TelegramIncomingUpdate]
                    ],
                    partial(poll_updates, sleep=sleep),
                )
            else:
                poller_fn = poller
            config_path = cfg.runtime.config_path
            watch_enabled = bool(watch_config) and config_path is not None

            async def handle_reload(reload: ConfigReload) -> None:
                refresh_commands()
                refresh_topics_scope()
                await set_command_menu(cfg)
                if state.transport_snapshot is not None:
                    new_snapshot = reload.settings.transports.telegram.model_dump()
                    changed = _diff_keys(state.transport_snapshot, new_snapshot)
                    if changed:
                        logger.warning(
                            "config.reload.transport_config_changed",
                            transport="telegram",
                            keys=changed,
                            restart_required=True,
                        )
                        state.transport_snapshot = new_snapshot
                if (
                    state.transport_id is not None
                    and reload.settings.transport != state.transport_id
                ):
                    logger.warning(
                        "config.reload.transport_changed",
                        old=state.transport_id,
                        new=reload.settings.transport,
                        restart_required=True,
                    )
                    state.transport_id = reload.settings.transport

            if watch_enabled and config_path is not None:

                async def run_config_watch() -> None:
                    await watch_config_changes(
                        config_path=config_path,
                        runtime=cfg.runtime,
                        default_engine_override=default_engine_override,
                        on_reload=handle_reload,
                    )

                tg.start_soon(run_config_watch)

            async def resolve_topic_icon(title: str) -> tuple[str, str | None]:
                first, separator, remainder = title.partition(" ")
                if not separator or not remainder.strip():
                    return title, None

                if state.topic_icon_stickers is None:
                    stickers = await cfg.bot.get_forum_topic_icon_stickers()
                    state.topic_icon_stickers = stickers or []

                return _topic_icon_choice(title, state.topic_icon_stickers)

            def wrap_on_thread_known(
                base_cb: Callable[[ResumeToken, anyio.Event], Awaitable[None]] | None,
                topic_key: tuple[int, int] | None,
                chat_session_key: tuple[int, int | None] | None,
                running_task: RunningTask | None = None,
                done_override: anyio.Event | None = None,
            ) -> Callable[[ResumeToken, anyio.Event], Awaitable[None]] | None:
                if base_cb is None and topic_key is None and chat_session_key is None:
                    return None

                async def _wrapped(token: ResumeToken, done: anyio.Event) -> None:
                    if base_cb is not None:
                        await base_cb(token, done_override or done)
                    if state.topic_store is not None and topic_key is not None:
                        await state.topic_store.set_session_resume(
                            topic_key[0], topic_key[1], token
                        )
                        if running_task is not None and running_task.title:
                            # Check if the title actually changed to avoid spamming the API.
                            snapshot = await state.topic_store.get_thread(*topic_key)
                            context = snapshot.context if snapshot else None
                            new_title, icon_id = await resolve_topic_icon(
                                running_task.title
                            )
                            current_title = snapshot.topic_title if snapshot else None
                            is_bound = context is not None and (
                                context.project or context.branch
                            )
                            if not is_bound and current_title != new_title:
                                updated = await cfg.bot.edit_forum_topic(
                                    chat_id=topic_key[0],
                                    message_thread_id=topic_key[1],
                                    name=new_title,
                                    icon_custom_emoji_id=icon_id,
                                )
                                if updated:
                                    await state.topic_store.set_context(
                                        *topic_key,
                                        context
                                        or RunContext(project=None, branch=None),
                                        topic_title=new_title,
                                    )
                    if (
                        state.chat_session_store is not None
                        and chat_session_key is not None
                    ):
                        await state.chat_session_store.set_session_resume(
                            chat_session_key[0], chat_session_key[1], token
                        )

                return _wrapped

            async def run_job(
                chat_id: int,
                user_msg_id: int,
                text: str,
                resume_token: ResumeToken | None,
                context: RunContext | None,
                thread_id: int | None = None,
                chat_session_key: tuple[int, int | None] | None = None,
                reply_ref: MessageRef | None = None,
                on_thread_known: Callable[[ResumeToken, anyio.Event], Awaitable[None]]
                | None = None,
                engine_override: EngineId | None = None,
                progress_ref: MessageRef | None = None,
                recovery_rpc: PiRpcRun | None = None,
                recovery_initial_id: tuple[int, int, int] | None = None,
                allow_live: bool = True,
            ) -> None:
                topic_key = (
                    (chat_id, thread_id)
                    if state.topic_store is not None
                    and thread_id is not None
                    and _topics_chat_allowed(
                        cfg, chat_id, scope_chat_ids=state.topics_chat_ids
                    )
                    else None
                )
                stateful_mode = topic_key is not None or chat_session_key is not None
                show_resume_line = should_show_resume_line(
                    show_resume_line=cfg.show_resume_line,
                    stateful_mode=stateful_mode,
                    context=context,
                )
                engine_for_overrides = (
                    resume_token.engine
                    if resume_token is not None
                    else engine_override
                    if engine_override is not None
                    else cfg.runtime.resolve_engine(
                        engine_override=None,
                        context=context,
                    )
                )
                overrides_thread_id = topic_key[1] if topic_key is not None else None
                run_options = await _resolve_engine_run_options(
                    chat_id,
                    overrides_thread_id,
                    engine_for_overrides,
                    chat_prefs=state.chat_prefs,
                    topic_store=state.topic_store,
                )
                # All Pi launch paths (including media, directives and queued
                # one-shot jobs) pass this gate, not just the opt-in RPC branch.
                # Otherwise a legacy prompt can overtake an unresolved live
                # instruction or write a Pi file bound to another topic.
                if (
                    live is not None
                    and engine_for_overrides == "pi"
                    and recovery_initial_id is None
                    and recovery_rpc is None
                ):
                    if topic_key is not None:
                        intent = await live.inbox.initial_for_topic(*topic_key)
                        original_scheduled = (
                            intent is not None
                            and allow_live
                            and resume_token is None
                            and intent.state == "scheduled"
                            and intent.message_id == user_msg_id
                            and fresh_pending.get(topic_key, (None, 0))[0]
                            == Path(intent.session_key)
                            and intent.prompt == text
                        )
                        if intent is not None and not original_scheduled:
                            await send_plain(
                                cfg.exec_cfg.transport,
                                chat_id=chat_id,
                                user_msg_id=user_msg_id,
                                thread_id=thread_id,
                                text=f"Initial Pi task #{intent.message_id} remains unresolved; no prompt submitted. Use scoped recovery.",
                            )
                            return
                        outstanding = [
                            receipt
                            for receipt in await live.inbox.unresolved_all()
                            if receipt.chat_id == chat_id
                            and receipt.thread_id == topic_key[1]
                            and receipt.message_id <= user_msg_id
                        ]
                        if outstanding and not original_scheduled:
                            await send_plain(
                                cfg.exec_cfg.transport,
                                chat_id=chat_id,
                                user_msg_id=user_msg_id,
                                thread_id=thread_id,
                                text="Prior Pi update is unresolved; no prompt submitted. Reconcile, retry or defer its receipt first.",
                            )
                            return
                    if resume_token is not None and state.topic_store is not None:
                        try:
                            cwd_for_claim = (
                                cfg.runtime.resolve_run_cwd(context) or Path.cwd()
                            )
                            candidate_path = Path(resume_token.value)
                            if not candidate_path.is_absolute():
                                candidate_path = (
                                    (cwd_for_claim / candidate_path).resolve()
                                    if candidate_path.suffix == ".jsonl"
                                    else resolve_legacy_session(
                                        resume_token.value, cwd_for_claim
                                    )
                                )
                            candidate_path = candidate_path.resolve()
                            other_owners = await state.topic_store.session_owners(
                                "pi",
                                str(candidate_path),
                                session_header_id(candidate_path)
                                if candidate_path.is_file()
                                else None,
                            )
                        except (ValueError, OSError, ConfigError):
                            other_owners = None
                        if other_owners is None or (
                            other_owners
                            and (topic_key is None or other_owners - {topic_key})
                        ):
                            await send_plain(
                                cfg.exec_cfg.transport,
                                chat_id=chat_id,
                                user_msg_id=user_msg_id,
                                thread_id=thread_id,
                                text=(
                                    "Pi session ID не найден однозначно в этом проекте; no prompt submitted."
                                    if not Path(resume_token.value).is_absolute()
                                    and Path(resume_token.value).suffix != ".jsonl"
                                    else "Pi session identity or topic owner cannot be verified; no prompt submitted."
                                ),
                            )
                            return
                from ..runner_bridge import RunningTask

                running_task = RunningTask(context=context)
                live_runner: LiveRunner | None = None
                owner: LiveOwner | None = None
                owner_done = anyio.Event()
                if (
                    live is not None
                    and allow_live
                    and topic_key is not None
                    and engine_for_overrides == "pi"
                    and config_path is not None
                ):
                    entry = cfg.runtime.resolve_runner(
                        resume_token=resume_token, engine_override=engine_override
                    )
                    root = (config_path.parent / "pi-live-sessions").resolve()
                    path = (
                        Path(resume_token.value).resolve()
                        if resume_token is not None
                        else fresh_pending.get(topic_key, (None, 0))[0]
                        or live_session_path(config_path, chat_id, topic_key[1])
                    )
                    # A short ID is only a lookup hint. Resolve to one local
                    # project file before claiming the canonical RPC owner.
                    bound = (
                        await state.topic_store.get_session_resume(
                            chat_id, topic_key[1], "pi"
                        )
                        if state.topic_store is not None
                        else None
                    )
                    fresh = (
                        path.parent == root
                        and path.name.startswith(f"{chat_id}-{topic_key[1]}-")
                        and resume_token is None
                    )
                    existing = (
                        resume_token is not None
                        and Path(resume_token.value).is_absolute()
                        and path.suffix == ".jsonl"
                        and path.is_file()
                        and bound == resume_token
                    )
                    cwd = cfg.runtime.resolve_run_cwd(context) or Path.cwd()
                    unresolved_initial = await live.inbox.initial_for_topic(
                        chat_id, topic_key[1]
                    )
                    if unresolved_initial is not None and not (
                        unresolved_initial.session_key == str(path)
                        and (
                            recovery_initial_id == unresolved_initial.id
                            or (
                                recovery_initial_id is None
                                and resume_token is None
                                and user_msg_id == unresolved_initial.message_id
                            )
                        )
                    ):
                        await send_plain(
                            cfg.exec_cfg.transport,
                            chat_id=chat_id,
                            user_msg_id=user_msg_id,
                            thread_id=thread_id,
                            text=f"Initial Pi task #{unresolved_initial.message_id} remains unresolved; no new prompt submitted. Explicitly retry-initial or defer-initial in this topic.",
                        )
                        return
                    legacy_id: str | None = None
                    if (
                        resume_token is not None
                        and not Path(resume_token.value).is_absolute()
                        and bound == resume_token
                        and isinstance(entry.runner, PiRunner)
                        and entry.available
                    ):
                        legacy_id = resume_token.value
                        try:
                            path = resolve_legacy_session(legacy_id, cwd)
                        except ValueError:
                            await send_plain(
                                cfg.exec_cfg.transport,
                                chat_id=chat_id,
                                user_msg_id=user_msg_id,
                                thread_id=thread_id,
                                text="Pi session ID не найден однозначно в этом проекте. Работа не запущена; уточните путь к сессии.",
                            )
                            return
                        if legacy_session_busy(state.running_tasks, path):
                            await send_plain(
                                cfg.exec_cfg.transport,
                                chat_id=chat_id,
                                user_msg_id=user_msg_id,
                                thread_id=thread_id,
                                text="Эта Pi-сессия уже выполняется; дождитесь завершения текущего запуска.",
                            )
                            return
                    if (
                        resume_token is not None
                        and recovery_rpc is None
                        and any(
                            receipt.message_id <= user_msg_id
                            for receipt in await live.inbox.unresolved_for_topic(
                                chat_id, topic_key[1], str(path)
                            )
                        )
                    ):
                        await send_plain(
                            cfg.exec_cfg.transport,
                            chat_id=chat_id,
                            user_msg_id=user_msg_id,
                            thread_id=thread_id,
                            text="Prior Pi update remains unresolved; no new writable prompt submitted. Reconcile, retry or defer the exact receipt first.",
                        )
                        return
                    if existing and state.topic_store is not None:
                        try:
                            owners = await state.topic_store.session_owners(
                                "pi", str(path), session_header_id(path)
                            )
                            if owners - {(chat_id, topic_key[1])}:
                                raise ValueError("Another topic owns this Pi session")
                        except ValueError:
                            await send_plain(
                                cfg.exec_cfg.transport,
                                chat_id=chat_id,
                                user_msg_id=user_msg_id,
                                thread_id=thread_id,
                                text="Pi session identity or topic ownership is conflicting; no prompt submitted.",
                            )
                            return
                    if resume_token is not None and not existing and legacy_id is None:
                        await send_plain(
                            cfg.exec_cfg.transport,
                            chat_id=chat_id,
                            user_msg_id=user_msg_id,
                            thread_id=thread_id,
                            text="Bound Pi session is missing or mismatched; no prompt submitted. Verify this exact topic before retrying.",
                        )
                        return
                    if (
                        isinstance(entry.runner, PiRunner)
                        and entry.available
                        and (fresh or existing or legacy_id is not None)
                    ):
                        if recovery_rpc is not None:
                            if Path(recovery_rpc.client.session_path).resolve() != path:
                                raise ValueError("Retry RPC owner path mismatch")
                            rpc = recovery_rpc
                        else:
                            with apply_run_options(run_options):
                                rpc = entry.runner.rpc_run(path, cwd=cwd)
                        provisional_id: str | None = None
                        if resume_token is None or existing:
                            try:
                                if existing:
                                    await verify_bound_session(rpc, path, cwd)
                                else:
                                    provisional_id = await verify_fresh_session(
                                        rpc, path, cwd
                                    )
                            except Exception as exc:  # noqa: BLE001 - canonical identity required before prompt
                                await rpc.client.close()
                                await send_plain(
                                    cfg.exec_cfg.transport,
                                    chat_id=chat_id,
                                    user_msg_id=user_msg_id,
                                    thread_id=thread_id,
                                    text=f"Pi session identity could not be verified ({type(exc).__name__}); no prompt submitted. Confirm recovery in this topic.",
                                )
                                return
                        live_runner = LiveRunner(
                            entry.runner,
                            rpc,
                            running_task.resume_ready,
                            initial_recovery=recovery_rpc is not None,
                        )
                        owner = LiveOwner(
                            chat_id,
                            topic_key[1],
                            str(path),
                            live_runner,
                            text[:1500],
                            running_task=running_task,
                            identity_verified=provisional_id is None,
                        )
                        if not live.register(owner):
                            await send_plain(
                                cfg.exec_cfg.transport,
                                chat_id=chat_id,
                                user_msg_id=user_msg_id,
                                thread_id=thread_id,
                                text="This Pi session is already active; try your message again.",
                            )
                            return
                        if legacy_id is not None:
                            try:
                                await verify_legacy_session(rpc, path, legacy_id, cwd)
                                assert state.topic_store is not None
                                owners = await state.topic_store.session_owners(
                                    "pi", str(path), session_header_id(path)
                                )
                                if owners - {(chat_id, topic_key[1])}:
                                    raise ValueError(
                                        "Another topic owns this Pi session"
                                    )
                                canonical = ResumeToken(engine="pi", value=str(path))
                                await state.topic_store.set_session_resume(
                                    chat_id, topic_key[1], canonical
                                )
                                resume_token = canonical
                            except Exception as exc:  # noqa: BLE001 - no prompt on failed identity
                                live.unregister(owner)
                                await rpc.client.close()
                                logger.warning(
                                    "live.legacy_identity.failed", error=str(exc)
                                )
                                await send_plain(
                                    cfg.exec_cfg.transport,
                                    chat_id=chat_id,
                                    user_msg_id=user_msg_id,
                                    thread_id=thread_id,
                                    text="Идентичность Pi-сессии не подтверждена. Работа не запущена; привязка темы сохранена.",
                                )
                                return
                        starting_live.discard(
                            (chat_id, topic_key[1], owner.session_key)
                        )
                        if fresh_pending.get(topic_key, (None, 0))[0] == path:
                            fresh_pending.pop(topic_key, None)
                        await live.flush(owner)
                        initial_intent = await live.inbox.initial_for_topic(
                            chat_id, topic_key[1]
                        )
                        if (
                            initial_intent is not None
                            and initial_intent.session_key == owner.session_key
                        ):
                            await live.inbox.mark_initial_uncertain(
                                initial_intent.id, owner.session_key
                            )

                        owned_live = live
                        assert owned_live is not None

                        async def finalize_owned() -> bool:
                            if not await verify_owned_header():
                                owner.closing = True
                                live_runner.final_failure = "New Pi session header missing/mismatched; task result incomplete. Initial task and updates require scoped recovery."
                                return False
                            result = await owned_live.finalize(owner)
                            if (
                                result
                                and not live_runner._followups
                                and live_runner.settlement_ok
                                and initial_intent is not None
                                and initial_intent.session_key == owner.session_key
                            ):
                                await owned_live.inbox.mark_initial_completed(
                                    initial_intent.id, owner.session_key
                                )
                            return result

                        live_runner.before_final = finalize_owned
                        live_runner.on_progress = lambda status: setattr(
                            owner, "progress", status
                        )
                        live_runner.on_followup_started = partial(
                            live.followup_started, owner
                        )
                    elif resume_token is not None:
                        await send_plain(
                            cfg.exec_cfg.transport,
                            chat_id=chat_id,
                            user_msg_id=user_msg_id,
                            thread_id=thread_id,
                            text="Не удалось безопасно открыть Pi live для этой сессии; проверьте привязку темы и путь к файлу JSONL. Работа не запущена.",
                        )
                        return

                known_callback = wrap_on_thread_known(
                    on_thread_known,
                    topic_key,
                    chat_session_key,
                    running_task,
                    owner_done if owner is not None else None,
                )

                async def guarded_thread_known(
                    token: ResumeToken, done: anyio.Event
                ) -> None:
                    if owner is not None and not owner.identity_verified:
                        if token.engine != "pi" or token.value != owner.session_key:
                            raise ValueError(
                                "Provisional Pi owner produced a different session"
                            )
                        return
                    if known_callback is not None:
                        await known_callback(token, done)

                async def verify_owned_header() -> bool:
                    if owner is None:
                        return True
                    async with owner.lock:
                        if owner.identity_verified:
                            return True
                        if provisional_id is None or not path.is_file():
                            return False
                        try:
                            verify_provisional_header(path, provisional_id, cwd)
                        except ValueError:
                            return False
                        if known_callback is not None:
                            await known_callback(
                                ResumeToken(engine="pi", value=owner.session_key),
                                owner_done,
                            )
                        owner.identity_verified = True
                        return True

                async def monitor() -> None:
                    assert live is not None and owner is not None
                    while not running_task.done.is_set():
                        await anyio.sleep(1)
                        try:
                            if not await verify_owned_header():
                                continue
                            await live.reconcile(owner)
                        except Exception as exc:  # noqa: BLE001 - RPC can exit while observed
                            logger.debug("live.reconcile.failed", error=str(exc))

                try:
                    async with anyio.create_task_group() as live_group:
                        if owner is not None:
                            live_group.start_soon(monitor)
                        await run_engine(
                            exec_cfg=cfg.exec_cfg,
                            runtime=cfg.runtime,
                            running_tasks=state.running_tasks,
                            chat_id=chat_id,
                            user_msg_id=user_msg_id,
                            text=text,
                            resume_token=resume_token,
                            context=context,
                            reply_ref=reply_ref,
                            running_task=running_task,
                            on_thread_known=guarded_thread_known,
                            engine_override=engine_override,
                            thread_id=thread_id,
                            show_resume_line=show_resume_line,
                            progress_ref=progress_ref,
                            run_options=run_options,
                            runner_override=live_runner,
                        )
                        live_group.cancel_scope.cancel()
                    if live is not None and owner is not None:
                        try:
                            await live.reconcile(owner)
                        except Exception as exc:  # noqa: BLE001 - retain receipts on failure
                            logger.warning(
                                "live.final_reconcile.failed", error=str(exc)
                            )
                finally:
                    try:
                        if live_runner is not None:
                            await live_runner.client.close()
                    finally:
                        with anyio.CancelScope(shield=True):
                            try:
                                if (
                                    live is not None
                                    and owner is not None
                                    and topic_key is not None
                                ):
                                    token = ResumeToken(
                                        engine="pi", value=owner.session_key
                                    )
                                    if await scheduler.has_pending_for_topic(
                                        token, chat_id, topic_key[1]
                                    ):
                                        starting_live.add(
                                            (chat_id, topic_key[1], owner.session_key)
                                        )
                            finally:
                                if live is not None and owner is not None:
                                    live.unregister(owner)
                                owner_done.set()

            async def run_thread_job(job: ThreadJob) -> None:
                try:
                    await run_job(
                        cast(int, job.chat_id),
                        cast(int, job.user_msg_id),
                        job.text,
                        job.resume_token,
                        job.context,
                        cast(int | None, job.thread_id),
                        job.session_key,
                        None,
                        scheduler.note_thread_known,
                        None,
                        job.progress_ref,
                        allow_live=job.allow_live,
                    )
                finally:
                    if isinstance(job.chat_id, int) and isinstance(job.thread_id, int):
                        start_key = live_route_key(
                            job.chat_id, job.thread_id, job.resume_token
                        )
                        if (
                            start_key is not None
                            and start_key not in retry_starting
                            and not await scheduler.has_pending_for_topic(
                                job.resume_token, job.chat_id, job.thread_id
                            )
                        ):
                            starting_live.discard(start_key)

            scheduler = ThreadScheduler(task_group=tg, run_job=run_thread_job)
            starting_live: set[tuple[int, int, str]] = set()
            retry_starting: set[tuple[int, int, str]] = set()
            fresh_pending: dict[tuple[int, int], tuple[Path, int]] = {}
            legacy_pending: set[tuple[int, int]] = set()
            recovery_queue: asyncio.Queue[Callable[[], Awaitable[None]]] = (
                asyncio.Queue(maxsize=32)
            )
            recovery_notice = asyncio.Event()
            recovery_stopping = False

            def enqueue_recovery(work: Callable[[], Awaitable[None]]) -> bool:
                try:
                    recovery_queue.put_nowait(work)
                    recovery_notice.set()
                    return True
                except asyncio.QueueFull:
                    return False

            async def recovery_worker() -> None:
                while True:
                    await recovery_notice.wait()
                    recovery_notice.clear()
                    while not recovery_queue.empty():
                        work = recovery_queue.get_nowait()
                        try:
                            await work()
                        except Exception as exc:  # noqa: BLE001 - other topics stay available
                            logger.warning(
                                "live.recovery.worker.failed",
                                error_type=type(exc).__name__,
                            )
                    if recovery_stopping:
                        return

            if cfg.pi_live_conversation:
                tg.start_soon(recovery_worker)
                tg.start_soon(recovery_worker)

            async def _run_quick(coro: Awaitable[None]) -> None:
                await coro

            def spawn_quick(coro: Awaitable[None]) -> None:
                tg.start_soon(_run_quick, coro)

            live: LiveConversationService | None = None
            if cfg.pi_live_conversation:
                if config_path is None:
                    raise ConfigError(
                        "Pi live conversation needs a config path for durable receipts"
                    )

                async def live_reply(
                    chat: int, thread: int, message: int, text: str
                ) -> None:
                    await send_plain(
                        cfg.exec_cfg.transport,
                        chat_id=chat,
                        user_msg_id=message,
                        thread_id=thread,
                        text=text,
                        notify=False,
                    )

                live = LiveConversationService(
                    LiveInbox(resolve_inbox_path(config_path)),
                    quick_pi_answer,
                    live_reply,
                )
                live.attach_workers(tg)
                tg.start_soon(live.notify_recovery)

            def resolve_topic_key(
                msg: TelegramIncomingMessage,
            ) -> tuple[int, int] | None:
                if state.topic_store is None:
                    return None
                return _topic_key(msg, cfg, scope_chat_ids=state.topics_chat_ids)

            def _build_upload_prompt(base: str, annotation: str) -> str:
                if base and base.strip():
                    return f"{base}\n\n{annotation}"
                return annotation

            async def resolve_prompt_message(
                msg: TelegramIncomingMessage,
                text: str,
                ambient_context: RunContext | None,
            ) -> ResolvedMessage | None:
                reply = make_reply(cfg, msg)
                try:
                    resolved = cfg.runtime.resolve_message(
                        text=text,
                        reply_text=msg.reply_to_text,
                        ambient_context=ambient_context,
                        chat_id=msg.chat_id,
                    )
                except DirectiveError as exc:
                    await reply(text=f"error:\n{exc}")
                    return None
                topic_key = resolve_topic_key(msg)
                chat_project = (
                    _topics_chat_project(cfg, msg.chat_id)
                    if cfg.topics.enabled
                    else None
                )
                _, ok = await ensure_topic_context(
                    resolved=resolved,
                    ambient_context=ambient_context,
                    topic_key=topic_key,
                    chat_project=chat_project,
                    reply=reply,
                )
                if not ok:
                    return None
                return resolved

            async def resolve_engine_defaults(
                *,
                explicit_engine: EngineId | None,
                context: RunContext | None,
                chat_id: int,
                topic_key: tuple[int, int] | None,
            ):
                return await resolve_engine_for_message(
                    runtime=cfg.runtime,
                    context=context,
                    explicit_engine=explicit_engine,
                    chat_id=chat_id,
                    topic_key=topic_key,
                    topic_store=state.topic_store,
                    chat_prefs=state.chat_prefs,
                )

            async def ensure_topic_context(
                *,
                resolved: ResolvedMessage,
                ambient_context: RunContext | None,
                topic_key: tuple[int, int] | None,
                chat_project: str | None,
                reply: Callable[..., Awaitable[None]],
            ) -> tuple[RunContext | None, bool]:
                effective_context = ambient_context
                if (
                    state.topic_store is not None
                    and topic_key is not None
                    and resolved.context is not None
                    and resolved.context_source == "directives"
                ):
                    await state.topic_store.set_context(*topic_key, resolved.context)
                    await _maybe_rename_topic(
                        cfg,
                        state.topic_store,
                        chat_id=topic_key[0],
                        thread_id=topic_key[1],
                        context=resolved.context,
                    )
                    effective_context = resolved.context
                if (
                    state.topic_store is not None
                    and topic_key is not None
                    and effective_context is None
                    and resolved.context_source not in {"directives", "reply_ctx"}
                ):
                    await reply(
                        text="this topic isn't bound to a project yet.\n"
                        f"{_usage_ctx_set(chat_project=chat_project)} or "
                        f"{_usage_topic(chat_project=chat_project)}",
                    )
                    return effective_context, False
                return effective_context, True

            resume_resolver = ResumeResolver(
                cfg=cfg,
                task_group=tg,
                running_tasks=state.running_tasks,
                enqueue_resume=scheduler.enqueue_resume,
                topic_store=state.topic_store,
                chat_session_store=state.chat_session_store,
            )

            async def dispatch_prompt_run(
                *,
                msg: TelegramIncomingMessage,
                prompt_text: str,
                resolved: ResolvedMessage,
                topic_key: tuple[int, int] | None,
                chat_session_key: tuple[int, int | None] | None,
                reply_ref: MessageRef | None,
                reply_id: int | None,
                live_eligible: bool = True,
            ) -> None:
                chat_id = msg.chat_id
                user_msg_id = msg.message_id
                context = resolved.context
                engine_resolution = await resolve_engine_defaults(
                    explicit_engine=resolved.engine_override,
                    context=context,
                    chat_id=chat_id,
                    topic_key=topic_key,
                )
                engine_override = engine_resolution.engine
                if (
                    live is not None
                    and resolved.resume_token is not None
                    and resolved.resume_token.engine == "pi"
                    and config_path is not None
                ):
                    explicit_path = Path(resolved.resume_token.value)
                    root = (config_path.parent / "pi-live-sessions").resolve()
                    if (
                        explicit_path.is_absolute()
                        and explicit_path.resolve().parent == root
                        and (
                            topic_key is None
                            or not explicit_path.name.startswith(
                                f"{chat_id}-{topic_key[1]}-"
                            )
                        )
                    ):
                        await send_plain(
                            cfg.exec_cfg.transport,
                            chat_id=chat_id,
                            user_msg_id=user_msg_id,
                            thread_id=msg.thread_id,
                            text="Pi live session belongs to a different topic.",
                        )
                        return
                if (
                    live_eligible
                    and live is not None
                    and topic_key is not None
                    and engine_override == "pi"
                ):
                    candidate = resolved.resume_token
                    if candidate is None and state.topic_store is not None:
                        candidate = await state.topic_store.get_session_resume(
                            chat_id, topic_key[1], "pi"
                        )
                    key = live_route_key(chat_id, msg.thread_id, candidate)
                    initial_fresh = False
                    if candidate is None and topic_key in fresh_pending:
                        provisional, first_id = fresh_pending[topic_key]
                        key = (chat_id, topic_key[1], str(provisional))
                        initial_fresh = user_msg_id == first_id
                    replied_task = (
                        state.running_tasks.get(reply_ref) if reply_ref else None
                    )
                    active = (
                        live.owner(*key)
                        if key is not None
                        else (
                            live.owner_for_topic(chat_id, topic_key[1])
                            if candidate is None
                            else None
                        )
                    )
                    if active is not None and key is None:
                        key = (chat_id, topic_key[1], active.session_key)
                    if active is not None and (
                        active.running_task is None
                        or resolved.context != active.running_task.context
                    ):
                        await live._startup_reply(
                            chat_id,
                            topic_key[1],
                            user_msg_id,
                            "Project/branch directive was not sent to the active Pi task. Wait for completion or use a separate topic.",
                        )
                        return
                    if (
                        key is not None
                        and active is None
                        and key in starting_live
                        and not initial_fresh
                    ):
                        await live.buffer_starting(
                            chat_id, key[1], user_msg_id, key[2], prompt_text
                        )
                        return
                    if key is not None and active is None and not initial_fresh:
                        unresolved = await live.inbox.unresolved_for_topic(*key)
                        if unresolved:
                            ids = ", ".join(str(r.message_id) for r in unresolved[:10])
                            await send_plain(
                                cfg.exec_cfg.transport,
                                chat_id=chat_id,
                                user_msg_id=user_msg_id,
                                thread_id=topic_key[1],
                                text=f"Pi has unresolved receipts #{ids}; no new prompt sent. "
                                "Check the scoped recovery notice and explicitly defer or reconcile first.",
                            )
                            return
                    if active is not None and active.closing and key is not None:
                        if await live.inbox.unresolved_for_topic(
                            *key
                        ) or await live.inbox.initial_for_topic(chat_id, key[1]):
                            await live._startup_reply(
                                chat_id,
                                key[1],
                                user_msg_id,
                                "Prior Pi task or update remains unresolved; this new prompt was not queued or submitted. Reconcile it before retrying.",
                            )
                            return
                        token = ResumeToken(engine="pi", value=active.session_key)
                        progress_ref = await _send_queued_progress(
                            cfg,
                            chat_id=chat_id,
                            user_msg_id=user_msg_id,
                            thread_id=msg.thread_id,
                            resume_token=token,
                            context=context,
                            steerable=False,
                        )
                        await scheduler.enqueue_resume(
                            chat_id,
                            user_msg_id,
                            prompt_text,
                            token,
                            context,
                            msg.thread_id,
                            chat_session_key,
                            progress_ref,
                        )
                        return
                    if (
                        active is not None
                        and (
                            replied_task is None
                            or replied_task is active.running_task
                            or (
                                replied_task.resume is not None
                                and replied_task.resume.value == active.session_key
                            )
                        )
                        and key is not None
                        and await live.handle(
                            chat_id,
                            key[1],
                            user_msg_id,
                            key[2],
                            prompt_text,
                            spawn_quick=spawn_quick,
                        )
                    ):
                        return
                resume_decision = await resume_resolver.resolve(
                    resume_token=resolved.resume_token,
                    reply_id=reply_id,
                    chat_id=chat_id,
                    user_msg_id=user_msg_id,
                    thread_id=msg.thread_id,
                    chat_session_key=chat_session_key,
                    topic_key=topic_key,
                    engine_for_session=engine_resolution.engine,
                    prompt_text=prompt_text,
                )
                if resume_decision.handled_by_running_task:
                    return
                resume_token = resume_decision.resume_token
                if live_eligible and live is not None and resume_token is not None:
                    key = live_route_key(chat_id, msg.thread_id, resume_token)
                    if key is not None and await live.handle(
                        chat_id,
                        key[1],
                        user_msg_id,
                        key[2],
                        prompt_text,
                        spawn_quick=spawn_quick,
                    ):
                        return
                if resume_token is None:
                    try:
                        await run_job(
                            chat_id,
                            user_msg_id,
                            prompt_text,
                            None,
                            context,
                            msg.thread_id,
                            chat_session_key,
                            reply_ref,
                            scheduler.note_thread_known,
                            engine_override,
                            allow_live=live_eligible,
                        )
                    finally:
                        if topic_key is not None and topic_key in fresh_pending:
                            provisional, first_id = fresh_pending[topic_key]
                            if first_id == user_msg_id:
                                fresh_pending.pop(topic_key, None)
                                starting_live.discard(
                                    (chat_id, topic_key[1], str(provisional))
                                )
                    return
                start_key = live_route_key(chat_id, msg.thread_id, resume_token)
                if live_eligible and live is not None and start_key is not None:
                    starting_live.add(start_key)
                progress_ref = await _send_queued_progress(
                    cfg,
                    chat_id=chat_id,
                    user_msg_id=user_msg_id,
                    thread_id=msg.thread_id,
                    resume_token=resume_token,
                    context=context,
                    steerable=await scheduler.is_busy(resume_token),
                )
                await scheduler.enqueue_resume(
                    chat_id,
                    user_msg_id,
                    prompt_text,
                    resume_token,
                    context,
                    msg.thread_id,
                    chat_session_key,
                    progress_ref,
                    allow_live=live_eligible,
                )

            async def run_prompt_from_upload(
                msg: TelegramIncomingMessage,
                prompt_text: str,
                resolved: ResolvedMessage,
            ) -> None:
                reply_id = msg.reply_to_message_id
                reply_ref = (
                    MessageRef(
                        channel_id=msg.chat_id,
                        message_id=msg.reply_to_message_id,
                        thread_id=msg.thread_id,
                    )
                    if msg.reply_to_message_id is not None
                    else None
                )
                chat_session_key = _chat_session_key(
                    msg, store=state.chat_session_store
                )
                topic_key = resolve_topic_key(msg)
                await dispatch_prompt_run(
                    msg=msg,
                    prompt_text=prompt_text,
                    resolved=resolved,
                    topic_key=topic_key,
                    chat_session_key=chat_session_key,
                    reply_ref=reply_ref,
                    reply_id=reply_id,
                    live_eligible=False,
                )

            async def _dispatch_pending_prompt(pending: _PendingPrompt) -> None:
                msg = pending.msg
                reply = make_reply(cfg, msg)
                try:
                    resolved = cfg.runtime.resolve_message(
                        text=pending.text,
                        reply_text=msg.reply_to_text,
                        ambient_context=pending.ambient_context,
                        chat_id=msg.chat_id,
                    )
                except DirectiveError as exc:
                    await reply(text=f"error:\n{exc}")
                    return
                if pending.is_voice_transcribed:
                    resolved = ResolvedMessage(
                        prompt=f"(voice transcribed) {resolved.prompt}",
                        resume_token=resolved.resume_token,
                        engine_override=resolved.engine_override,
                        context=resolved.context,
                        context_source=resolved.context_source,
                    )

                prompt_text = resolved.prompt
                if pending.forwards:
                    forwarded = [
                        text
                        for _, text in sorted(
                            pending.forwards,
                            key=lambda item: item[0],
                        )
                    ]
                    prompt_text = _format_forwarded_prompt(
                        forwarded,
                        prompt_text,
                    )

                _effective_context, ok = await ensure_topic_context(
                    resolved=resolved,
                    ambient_context=pending.ambient_context,
                    topic_key=pending.topic_key,
                    chat_project=pending.chat_project,
                    reply=reply,
                )
                if not ok:
                    if pending.topic_key is not None and pending.forwards:
                        legacy_pending.discard(pending.topic_key)
                    return
                initial_directive = False
                if (
                    live is not None
                    and pending.topic_key is not None
                    and resolved.engine_override in (None, "pi")
                    and not pending.forwards
                    and not pending.is_voice_transcribed
                    and fresh_pending.get(pending.topic_key, (None, 0))[1]
                    == msg.message_id
                ):
                    intent = await live.inbox.initial_for_topic(*pending.topic_key)
                    initial_directive = (
                        intent is not None
                        and intent.message_id == msg.message_id
                        and intent.prompt == prompt_text
                    )
                try:
                    await dispatch_prompt_run(
                        msg=msg,
                        prompt_text=prompt_text,
                        resolved=resolved,
                        topic_key=pending.topic_key,
                        chat_session_key=pending.chat_session_key,
                        reply_ref=pending.reply_ref,
                        reply_id=pending.reply_id,
                        live_eligible=not pending.is_voice_transcribed
                        and not pending.forwards
                        and (
                            initial_directive
                            or (
                                resolved.prompt == pending.text.strip()
                                and resolved.context_source != "directives"
                            )
                        ),
                    )
                finally:
                    if pending.topic_key is not None and pending.forwards:
                        legacy_pending.discard(pending.topic_key)

            forward_coalescer = ForwardCoalescer(
                task_group=tg,
                debounce_s=state.forward_coalesce_s,
                sleep=sleep,
                dispatch=_dispatch_pending_prompt,
                pending=state.pending_prompts,
            )

            async def handle_prompt_upload(
                msg: TelegramIncomingMessage,
                caption_text: str,
                ambient_context: RunContext | None,
                topic_store: TopicStateStore | None,
            ) -> None:
                resolved = await resolve_prompt_message(
                    msg,
                    caption_text,
                    ambient_context,
                )
                if resolved is None:
                    return
                saved = await save_file_put(
                    cfg,
                    msg,
                    "",
                    resolved.context,
                    topic_store,
                )
                if saved is None:
                    return
                annotation = f"[uploaded file: {saved.rel_path.as_posix()}]"
                prompt = _build_upload_prompt(resolved.prompt, annotation)
                await run_prompt_from_upload(msg, prompt, resolved)

            media_group_buffer = MediaGroupBuffer(
                task_group=tg,
                debounce_s=state.media_group_debounce_s,
                sleep=sleep,
                cfg=cfg,
                chat_prefs=state.chat_prefs,
                topic_store=state.topic_store,
                bot_username=state.bot_username,
                command_ids=lambda: state.command_ids,
                reserved_chat_commands=state.reserved_chat_commands,
                groups=state.media_groups,
                run_prompt_from_upload=run_prompt_from_upload,
                resolve_prompt_message=resolve_prompt_message,
            )

            async def build_message_context(
                msg: TelegramIncomingMessage,
            ) -> TelegramMsgContext:
                chat_id = msg.chat_id
                reply_id = msg.reply_to_message_id
                reply_ref = (
                    MessageRef(channel_id=chat_id, message_id=reply_id)
                    if reply_id is not None
                    else None
                )
                topic_key = resolve_topic_key(msg)
                chat_session_key = _chat_session_key(
                    msg, store=state.chat_session_store
                )
                stateful_mode = topic_key is not None or chat_session_key is not None
                chat_project = (
                    _topics_chat_project(cfg, chat_id) if cfg.topics.enabled else None
                )
                bound_context = (
                    await state.topic_store.get_context(*topic_key)
                    if state.topic_store is not None and topic_key is not None
                    else None
                )
                chat_bound_context = None
                if state.chat_prefs is not None:
                    chat_bound_context = await state.chat_prefs.get_context(chat_id)
                if bound_context is not None:
                    ambient_context = _merge_topic_context(
                        chat_project=chat_project, bound=bound_context
                    )
                elif chat_bound_context is not None:
                    ambient_context = chat_bound_context
                else:
                    ambient_context = _merge_topic_context(
                        chat_project=chat_project, bound=None
                    )
                return TelegramMsgContext(
                    chat_id=chat_id,
                    thread_id=msg.thread_id,
                    reply_id=reply_id,
                    reply_ref=reply_ref,
                    topic_key=topic_key,
                    chat_session_key=chat_session_key,
                    stateful_mode=stateful_mode,
                    chat_project=chat_project,
                    ambient_context=ambient_context,
                )

            async def route_message(msg: TelegramIncomingMessage) -> None:
                reply = make_reply(cfg, msg)
                classification = _classify_message(msg, files_enabled=cfg.files.enabled)
                text = classification.text
                is_voice_transcribed = False
                if classification.is_forward_candidate:
                    pending_forward = forward_coalescer._pending.get(_forward_key(msg))
                    if (
                        live is not None
                        and pending_forward is not None
                        and pending_forward.topic_key is not None
                    ):
                        forward_topic = pending_forward.topic_key
                        intent = await live.inbox.initial_for_topic(*forward_topic)
                        if (
                            intent is not None
                            and intent.message_id == pending_forward.msg.message_id
                        ):
                            dependent = await live.inbox.unresolved_for_topic(
                                forward_topic[0], forward_topic[1], intent.session_key
                            )
                            if dependent or intent.state != "scheduled":
                                await live._startup_reply(
                                    msg.chat_id,
                                    forward_topic[1],
                                    msg.message_id,
                                    "Forwarded message not accepted: initial Pi task has dependent updates or may already have started. Resend after resolving the live task.",
                                )
                                return
                            await live.inbox.mark_initial_deferred(
                                intent.id,
                                intent.session_key,
                                "Transport rerouted forwarded composition to legacy one-shot before Pi submission",
                            )
                            fresh_pending.pop(forward_topic, None)
                            starting_live.discard(
                                (forward_topic[0], forward_topic[1], intent.session_key)
                            )
                            legacy_pending.add(forward_topic)
                    forward_coalescer.attach_forward(msg)
                    return
                forward_key = _forward_key(msg)
                if classification.is_media_group_document:
                    media_group_buffer.add(msg)
                    return
                ctx = await build_message_context(msg)
                chat_id = ctx.chat_id
                reply_id = ctx.reply_id
                reply_ref = ctx.reply_ref
                topic_key = ctx.topic_key
                if (
                    live is not None
                    and topic_key in legacy_pending
                    and not classification.is_cancel
                ):
                    await live._startup_reply(
                        chat_id,
                        topic_key[1],
                        msg.message_id,
                        "Forwarded task is running through legacy one-shot; this message was not accepted. Resend after the final reply.",
                    )
                    return
                chat_session_key = ctx.chat_session_key
                stateful_mode = ctx.stateful_mode
                chat_project = ctx.chat_project
                ambient_context = ctx.ambient_context

                if classification.is_cancel:
                    tg.start_soon(
                        handle_cancel, cfg, msg, state.running_tasks, scheduler
                    )
                    return

                command_id = classification.command_id
                args_text = classification.args_text
                # Project commands can bypass ordinary-text live interception via
                # dispatch_command. Never let a command for another project/branch
                # run or steer under this topic's active Pi session identity.
                if (
                    live is not None
                    and topic_key is not None
                    and command_id is not None
                ):
                    active_pi = live.owner_for_topic(chat_id, topic_key[1])
                    if active_pi is not None:
                        try:
                            command_target = cfg.runtime.resolve_message(
                                text=text,
                                reply_text=msg.reply_to_text,
                                ambient_context=ambient_context,
                                chat_id=chat_id,
                            )
                        except DirectiveError:
                            command_target = None
                        if (
                            command_target is not None
                            and command_target.context_source == "directives"
                            and (
                                active_pi.running_task is None
                                or command_target.context
                                != active_pi.running_task.context
                            )
                        ):
                            await live._startup_reply(
                                chat_id,
                                topic_key[1],
                                msg.message_id,
                                "Project/branch command cannot target the active Pi task; use another topic or wait for completion.",
                            )
                            return
                if command_id == "new":
                    if topic_key is not None and live is not None:
                        initial = await live.inbox.initial_for_topic(
                            chat_id, topic_key[1]
                        )
                        if initial is not None:
                            await reply(
                                text=f"Cannot /new while initial Pi task #{initial.message_id} is unresolved. Explicitly retry or defer it first."
                            )
                            return
                    if topic_key is not None and live is not None:
                        provisional = fresh_pending.get(topic_key, (None, 0))[0]
                        if (
                            provisional is not None
                            and await live.inbox.unresolved_for_topic(
                                chat_id, topic_key[1], str(provisional)
                            )
                        ):
                            await reply(
                                text="Cannot /new while startup has unresolved live updates. Wait for Pi or explicitly defer the receipt first."
                            )
                            return
                        if any(
                            r.chat_id == chat_id and r.thread_id == topic_key[1]
                            for r in await live.inbox.unresolved_all()
                        ):
                            await reply(
                                text="Cannot /new with unresolved live receipts in this topic. Reconcile/retry or explicitly defer them first."
                            )
                            return
                    forward_coalescer.cancel(forward_key)
                    if topic_key is not None and topic_key in fresh_pending:
                        provisional, _ = fresh_pending.pop(topic_key)
                        starting_live.discard((chat_id, topic_key[1], str(provisional)))
                    if state.topic_store is not None and topic_key is not None:
                        tg.start_soon(
                            partial(
                                handle_new_command,
                                cfg,
                                msg,
                                state.topic_store,
                                resolved_scope=state.resolved_topics_scope,
                                scope_chat_ids=state.topics_chat_ids,
                            )
                        )
                        return
                    if state.chat_session_store is not None:
                        tg.start_soon(
                            handle_chat_new_command,
                            cfg,
                            msg,
                            state.chat_session_store,
                            chat_session_key,
                        )
                        return
                    if state.topic_store is not None:
                        tg.start_soon(
                            partial(
                                handle_new_command,
                                cfg,
                                msg,
                                state.topic_store,
                                resolved_scope=state.resolved_topics_scope,
                                scope_chat_ids=state.topics_chat_ids,
                            )
                        )
                        return
                if command_id is not None and _dispatch_builtin_command(
                    ctx=TelegramCommandContext(
                        cfg=cfg,
                        msg=msg,
                        args_text=args_text,
                        ambient_context=ambient_context,
                        topic_store=state.topic_store,
                        chat_prefs=state.chat_prefs,
                        resolved_scope=state.resolved_topics_scope,
                        scope_chat_ids=state.topics_chat_ids,
                        reply=reply,
                        task_group=tg,
                    ),
                    command_id=command_id,
                ):
                    return

                trigger_mode = await resolve_trigger_mode(
                    chat_id=chat_id,
                    thread_id=msg.thread_id,
                    chat_prefs=state.chat_prefs,
                    topic_store=state.topic_store,
                )
                if trigger_mode == "mentions" and not should_trigger_run(
                    msg,
                    bot_username=state.bot_username,
                    runtime=cfg.runtime,
                    command_ids=state.command_ids,
                    reserved_chat_commands=state.reserved_chat_commands,
                ):
                    return

                if msg.voice is not None:
                    text = await transcribe_voice(
                        bot=cfg.bot,
                        msg=msg,
                        enabled=cfg.voice_transcription,
                        model=cfg.voice_transcription_model,
                        max_bytes=cfg.voice_max_bytes,
                        reply=reply,
                        base_url=cfg.voice_transcription_base_url,
                        api_key=cfg.voice_transcription_api_key,
                    )
                    if text is None:
                        return
                    is_voice_transcribed = True
                if msg.document is not None:
                    if cfg.files.enabled and cfg.files.auto_put:
                        caption_text = text.strip()
                        if cfg.files.auto_put_mode == "prompt" and caption_text:
                            tg.start_soon(
                                handle_prompt_upload,
                                msg,
                                caption_text,
                                ambient_context,
                                state.topic_store,
                            )
                        elif not caption_text:
                            tg.start_soon(
                                handle_file_put_default,
                                cfg,
                                msg,
                                ambient_context,
                                state.topic_store,
                            )
                        else:
                            tg.start_soon(
                                partial(reply, text=FILE_PUT_USAGE),
                            )
                    elif cfg.files.enabled:
                        tg.start_soon(
                            partial(reply, text=FILE_PUT_USAGE),
                        )
                    return
                if command_id == "update" and live is not None:
                    candidate = (
                        await state.topic_store.get_session_resume(
                            chat_id, topic_key[1], "pi"
                        )
                        if state.topic_store is not None and topic_key is not None
                        else None
                    )
                    key = live_route_key(chat_id, msg.thread_id, candidate)
                    if key is None and candidate is None and topic_key in fresh_pending:
                        provisional, _ = fresh_pending[topic_key]
                        key = (chat_id, topic_key[1], str(provisional))
                    active = (
                        live.owner(*key)
                        if key is not None
                        else (
                            live.owner_for_topic(chat_id, topic_key[1])
                            if topic_key is not None and candidate is None
                            else None
                        )
                    )
                    parts = args_text.strip().split(" ", 2)
                    if (
                        key is None
                        and candidate is None
                        and topic_key is not None
                        and parts[0] in ("defer", "retry")
                        and len(parts) >= 2
                        and parts[1].isdigit()
                    ):
                        try:
                            orphan = await live.inbox.get(
                                (chat_id, topic_key[1], int(parts[1]))
                            )
                        except KeyError:
                            orphan = None
                        if orphan is not None:
                            key = (chat_id, topic_key[1], orphan.session_key)
                            if (
                                parts[0] == "retry"
                                and await live.inbox.initial_for_topic(
                                    chat_id, topic_key[1]
                                )
                                is not None
                            ):
                                await reply(
                                    text="Initial Pi task must be reconciled before retrying its dependent update."
                                )
                                return
                    if (
                        topic_key is not None
                        and parts[0] in ("retry-initial", "defer-initial")
                        and len(parts) >= 2
                        and parts[1].isdigit()
                    ):
                        initial = await live.inbox.initial_by_id(
                            (chat_id, topic_key[1], int(parts[1]))
                        )
                        if initial is None or initial.state not in (
                            "scheduled",
                            "uncertain",
                        ):
                            await reply(
                                text="Initial Pi task ID is not unresolved in this exact topic; nothing changed."
                            )
                            return
                        intent_key = (chat_id, topic_key[1], initial.session_key)
                        if parts[0] == "defer-initial":
                            if len(parts) < 3 or not parts[2].strip():
                                await reply(
                                    text="Explicit initial-task deferral requires a reason."
                                )
                            elif (
                                intent_key in starting_live
                                or live.owner(*intent_key) is not None
                            ):
                                await reply(
                                    text="Initial Pi task is in flight; deferral refused until ownership settles."
                                )
                            else:
                                await live.inbox.mark_initial_deferred(
                                    initial.id, initial.session_key, parts[2].strip()
                                )
                                await reply(
                                    text=f"Initial Pi task #{initial.message_id} explicitly deferred. Pi may already have acted; dependent updates must be resolved separately."
                                )
                            return
                        if len(parts) != 3 or parts[2] != "confirm":
                            await reply(
                                text=f"Retry may duplicate tool effects. Confirm with /update retry-initial {initial.message_id} confirm."
                            )
                            return
                        if (
                            intent_key in starting_live
                            or live.owner(*intent_key) is not None
                        ):
                            await reply(
                                text="Initial Pi owner already starting; no duplicate retry scheduled."
                            )
                            return
                        if candidate is not None and (
                            live_route_key(chat_id, msg.thread_id, candidate)
                            != intent_key
                        ):
                            await reply(
                                text="Topic is bound to another Pi session; initial retry refused."
                            )
                            return

                        async def run_initial_retry(intent: InitialIntent) -> None:
                            rpc: PiRpcRun | None = None
                            try:
                                if config_path is None:
                                    raise ValueError("Live config path unavailable")
                                cwd = (
                                    cfg.runtime.resolve_run_cwd(ambient_context)
                                    or Path.cwd()
                                )
                                if str(cwd.resolve()) != intent.cwd:
                                    raise ValueError("Initial task project changed")
                                path = Path(intent.session_key)
                                assert state.topic_store is not None
                                owners = await state.topic_store.session_owners(
                                    "pi",
                                    str(path),
                                    session_header_id(path) if path.exists() else None,
                                )
                                if owners - {(chat_id, topic_key[1])}:
                                    raise ValueError(
                                        "Initial Pi session is bound to another topic"
                                    )
                                token: ResumeToken | None = None
                                if path.exists():
                                    entry = cfg.runtime.resolve_runner(
                                        resume_token=ResumeToken(
                                            engine="pi", value=str(path)
                                        ),
                                        engine_override="pi",
                                    )
                                    if (
                                        not isinstance(entry.runner, PiRunner)
                                        or not entry.available
                                    ):
                                        raise ValueError("Pi runner unavailable")
                                    opts = await _resolve_engine_run_options(
                                        chat_id,
                                        topic_key[1],
                                        "pi",
                                        chat_prefs=state.chat_prefs,
                                        topic_store=state.topic_store,
                                    )
                                    with apply_run_options(opts):
                                        rpc = entry.runner.rpc_run(path, cwd=cwd)
                                decision = await live.inspect_initial_retry(
                                    intent, rpc, cwd, config_path.parent
                                )
                                if decision == "fresh":
                                    prompt = intent.prompt
                                else:
                                    token = ResumeToken(
                                        engine="pi", value=intent.session_key
                                    )
                                    assert state.topic_store is not None
                                    await state.topic_store.set_session_resume(
                                        chat_id, topic_key[1], token
                                    )
                                    prompt = (
                                        intent.prompt
                                        if decision == "retry"
                                        else "Continue the original task from its observed main-session state; do not redo finished tool effects. Consider queued updates before finalizing."
                                    )
                                await run_job(
                                    chat_id,
                                    msg.message_id,
                                    prompt,
                                    token,
                                    ambient_context,
                                    topic_key[1],
                                    chat_session_key,
                                    None,
                                    scheduler.note_thread_known,
                                    "pi",
                                    None,
                                    recovery_rpc=rpc,
                                    recovery_initial_id=intent.id,
                                )
                            except Exception as exc:  # noqa: BLE001 - preserve ambiguous initial task
                                logger.warning(
                                    "live.initial_retry.failed",
                                    error_type=type(exc).__name__,
                                )
                                await reply(
                                    text=f"Initial Pi task #{intent.message_id}: retry failed/owner conflicted; no success claimed. Inspect the scoped state before another explicit retry."
                                )
                            finally:
                                if rpc is not None:
                                    await rpc.client.close()
                                retry_starting.discard(intent_key)
                                if fresh_pending.get(topic_key, (None, 0))[0] == Path(
                                    intent.session_key
                                ):
                                    fresh_pending.pop(topic_key, None)
                                if not await scheduler.has_pending_for_topic(
                                    ResumeToken(engine="pi", value=intent.session_key),
                                    chat_id,
                                    topic_key[1],
                                ):
                                    starting_live.discard(intent_key)

                        confirmed_initial = cast(InitialIntent, initial)

                        async def initial_work() -> None:
                            await run_initial_retry(confirmed_initial)

                        if not enqueue_recovery(initial_work):
                            await live._startup_reply(
                                chat_id,
                                topic_key[1],
                                msg.message_id,
                                "Pi recovery queue full; initial retry not started. Please retry later.",
                            )
                            return
                        fresh_pending[topic_key] = (
                            Path(initial.session_key),
                            initial.message_id,
                        )
                        starting_live.add(intent_key)
                        retry_starting.add(intent_key)
                        await live._startup_reply(
                            chat_id,
                            topic_key[1],
                            msg.message_id,
                            f"Initial Pi task #{initial.message_id}: canonical inspection scheduled. Your explicit confirmation permits a duplicate-risk prompt only if unobserved; no automatic replay.",
                        )
                        return
                    if key is not None and args_text.startswith(("defer ", "retry ")):
                        recovery_key = cast(tuple[int, int, str], key)
                        if (
                            args_text.startswith("retry ")
                            and recovery_key in starting_live
                        ):
                            await reply(
                                text="Pi owner is starting; retry already pending. No second replay started."
                            )
                            return

                        async def run_idle_retry(receipt: Receipt) -> None:
                            rpc: PiRpcRun | None = None
                            try:
                                token = ResumeToken(
                                    engine="pi", value=receipt.session_key
                                )
                                entry = cfg.runtime.resolve_runner(
                                    resume_token=token, engine_override="pi"
                                )
                                if (
                                    not isinstance(entry.runner, PiRunner)
                                    or not entry.available
                                ):
                                    raise ValueError(
                                        "Pi runner unavailable for recovery"
                                    )
                                cwd = (
                                    cfg.runtime.resolve_run_cwd(ambient_context)
                                    or Path.cwd()
                                )
                                opts = await _resolve_engine_run_options(
                                    chat_id,
                                    recovery_key[1],
                                    "pi",
                                    chat_prefs=state.chat_prefs,
                                    topic_store=state.topic_store,
                                )
                                with apply_run_options(opts):
                                    rpc = entry.runner.rpc_run(
                                        Path(receipt.session_key), cwd=cwd
                                    )
                                if not await live.begin_idle_retry(receipt, rpc, cwd):
                                    return
                                await run_job(
                                    chat_id,
                                    msg.message_id,
                                    await live.inbox.delivery_text(receipt.id),
                                    token,
                                    ambient_context,
                                    recovery_key[1],
                                    chat_session_key,
                                    None,
                                    scheduler.note_thread_known,
                                    "pi",
                                    None,
                                    recovery_rpc=rpc,
                                )
                            except Exception as exc:  # noqa: BLE001 - unknown RPC outcome stays durable
                                logger.warning(
                                    "live.retry.failed", error_type=type(exc).__name__
                                )
                                await reply(
                                    text=f"Receipt #{receipt.message_id}: retry failed or owner conflicted; no success claimed. Inspect state before confirming again."
                                )
                            finally:
                                if rpc is not None:
                                    await rpc.client.close()
                                retry_starting.discard(recovery_key)
                                if not await scheduler.has_pending_for_topic(
                                    token, chat_id, recovery_key[1]
                                ):
                                    starting_live.discard(recovery_key)

                        def spawn_idle_retry(receipt: Receipt) -> None:
                            async def work() -> None:
                                await run_idle_retry(receipt)

                            if not enqueue_recovery(work):
                                raise RuntimeError("Pi recovery queue is full")
                            starting_live.add(recovery_key)
                            retry_starting.add(recovery_key)

                        async def inspect_receipt() -> None:
                            try:
                                handled = await live.recover_receipt(
                                    chat_id,
                                    recovery_key[1],
                                    msg.message_id,
                                    recovery_key[2],
                                    args_text,
                                    spawn_idle_retry=spawn_idle_retry,
                                )
                                if not handled:
                                    await reply(
                                        text="Receipt not found in this exact topic/session; nothing changed."
                                    )
                            except RuntimeError:
                                await live._startup_reply(
                                    chat_id,
                                    recovery_key[1],
                                    msg.message_id,
                                    "Pi recovery queue full; no retry started. Please try later.",
                                )

                        if not enqueue_recovery(inspect_receipt):
                            await live._startup_reply(
                                chat_id,
                                key[1],
                                msg.message_id,
                                "Pi recovery queue full; nothing submitted. Please retry later.",
                            )
                        return
                    if active is None and key is not None and key in starting_live:
                        await live.buffer_starting(
                            chat_id,
                            key[1],
                            msg.message_id,
                            key[2],
                            f"/update {args_text}",
                        )
                    elif active is not None and not active.closing:
                        await live.handle(
                            chat_id,
                            active.thread_id,
                            msg.message_id,
                            active.session_key,
                            f"/update {args_text}",
                            spawn_quick=spawn_quick,
                        )
                    elif active is not None:
                        await reply(
                            text="Pi is finalizing; update was not accepted. Please resend after the final reply."
                        )
                    else:
                        tg.start_soon(
                            partial(
                                reply, text="/update requires an active Pi live topic."
                            )
                        )
                    return
                if command_id is not None and command_id not in state.reserved_commands:
                    if command_id not in state.command_ids:
                        refresh_commands()
                    if command_id in state.command_ids:
                        engine_resolution = await resolve_engine_defaults(
                            explicit_engine=None,
                            context=ambient_context,
                            chat_id=chat_id,
                            topic_key=topic_key,
                        )
                        default_engine_override = (
                            engine_resolution.engine
                            if engine_resolution.source
                            in {"directive", "topic_default", "chat_default"}
                            else None
                        )
                        overrides_thread_id = (
                            topic_key[1] if topic_key is not None else None
                        )
                        engine_overrides_resolver = partial(
                            _resolve_engine_run_options,
                            chat_id,
                            overrides_thread_id,
                            chat_prefs=state.chat_prefs,
                            topic_store=state.topic_store,
                        )
                        tg.start_soon(
                            dispatch_command,
                            cfg,
                            msg,
                            text,
                            command_id,
                            args_text,
                            state.running_tasks,
                            scheduler,
                            wrap_on_thread_known(
                                scheduler.note_thread_known,
                                topic_key,
                                chat_session_key,
                                None,
                            ),
                            stateful_mode,
                            default_engine_override,
                            engine_overrides_resolver,
                        )
                        return

                pending = _PendingPrompt(
                    msg=msg,
                    text=text,
                    ambient_context=ambient_context,
                    chat_project=chat_project,
                    topic_key=topic_key,
                    chat_session_key=chat_session_key,
                    reply_ref=reply_ref,
                    reply_id=reply_id,
                    is_voice_transcribed=is_voice_transcribed,
                    forwards=[],
                )
                if reply_id is not None and state.running_tasks.get(
                    MessageRef(channel_id=chat_id, message_id=reply_id)
                ):
                    logger.debug(
                        "forward.prompt.bypass",
                        chat_id=chat_id,
                        thread_id=msg.thread_id,
                        sender_id=msg.sender_id,
                        message_id=msg.message_id,
                        reason="reply_resume",
                    )
                    tg.start_soon(_dispatch_pending_prompt, pending)
                    return
                # Active ordinary text must be accepted before the destructive
                # forward debounce can replace it with a later status question.
                if (
                    live is not None
                    and topic_key is not None
                    and not is_voice_transcribed
                    and msg.voice is None
                    and msg.document is None
                    and msg.media_group_id is None
                    and text.strip()
                ):
                    try:
                        resolved_live = cfg.runtime.resolve_message(
                            text=text,
                            reply_text=msg.reply_to_text,
                            ambient_context=ambient_context,
                            chat_id=chat_id,
                        )
                    except DirectiveError:
                        resolved_live = None
                    if (
                        resolved_live is not None
                        and resolved_live.context_source == "directives"
                        and (initial := await live.inbox.initial_for_topic(*topic_key))
                        is not None
                        and initial.message_id != msg.message_id
                    ):
                        await live._startup_reply(
                            chat_id,
                            topic_key[1],
                            msg.message_id,
                            f"Directive not accepted: initial Pi task #{initial.message_id} remains unresolved. Retry or defer it first.",
                        )
                        return
                    if (
                        resolved_live is not None
                        and resolved_live.prompt.strip()
                        and resolved_live.engine_override in (None, "pi")
                    ):
                        engine_live = await resolve_engine_defaults(
                            explicit_engine=resolved_live.engine_override,
                            context=resolved_live.context,
                            chat_id=chat_id,
                            topic_key=topic_key,
                        )
                        candidate = (
                            resolved_live.resume_token
                            or await state.topic_store.get_session_resume(  # type: ignore[union-attr]
                                chat_id, topic_key[1], "pi"
                            )
                        )
                        key = live_route_key(chat_id, msg.thread_id, candidate)
                        if (
                            key is None
                            and candidate is None
                            and topic_key in fresh_pending
                        ):
                            provisional, _ = fresh_pending[topic_key]
                            key = (chat_id, topic_key[1], str(provisional))
                        active = (
                            live.owner(*key)
                            if key is not None
                            else (
                                live.owner_for_topic(chat_id, topic_key[1])
                                if candidate is None
                                else None
                            )
                        )
                        if active is not None and key is None:
                            key = (chat_id, topic_key[1], active.session_key)
                        if active is not None and (
                            active.running_task is None
                            or resolved_live.context != active.running_task.context
                        ):
                            await live._startup_reply(
                                chat_id,
                                topic_key[1],
                                msg.message_id,
                                "Project/branch directive was not sent to the active Pi task. Wait for completion or use a separate topic.",
                            )
                            return
                        replied_task = (
                            state.running_tasks.get(reply_ref) if reply_ref else None
                        )
                        if (
                            engine_live.engine == "pi"
                            and key is not None
                            and active is None
                            and key in starting_live
                            and state.topic_store is not None
                        ):
                            bound_context = await state.topic_store.get_context(
                                *topic_key
                            )
                            initial = await live.inbox.initial_for_topic(*topic_key)
                            context_mismatch = (
                                bound_context is not None
                                and resolved_live.context != bound_context
                            )
                            if bound_context is None and initial is not None:
                                try:
                                    target_cwd = cfg.runtime.resolve_run_cwd(
                                        resolved_live.context
                                    )
                                    context_mismatch = (
                                        target_cwd is None
                                        or str(target_cwd.resolve()) != initial.cwd
                                    )
                                except ConfigError:
                                    context_mismatch = True
                            if context_mismatch:
                                await live._startup_reply(
                                    chat_id,
                                    topic_key[1],
                                    msg.message_id,
                                    "Project/branch directive was not sent to the starting Pi task. Use another topic or wait for completion.",
                                )
                                return
                        if (
                            engine_live.engine == "pi"
                            and key is not None
                            and active is None
                            and key in starting_live
                        ):
                            await live.buffer_starting(
                                chat_id,
                                key[1],
                                msg.message_id,
                                key[2],
                                resolved_live.prompt,
                            )
                            return
                        if (
                            engine_live.engine == "pi"
                            and key is not None
                            and active is not None
                            and active.closing
                        ):
                            await _dispatch_pending_prompt(pending)
                            return
                        if (
                            engine_live.engine == "pi"
                            and key is not None
                            and active is not None
                            and (
                                replied_task is None
                                or replied_task is active.running_task
                                or (
                                    replied_task.resume is not None
                                    and replied_task.resume.value == active.session_key
                                )
                            )
                            and await live.handle(
                                chat_id,
                                key[1],
                                msg.message_id,
                                key[2],
                                resolved_live.prompt,
                                spawn_quick=spawn_quick,
                            )
                        ):
                            return
                        if (
                            engine_live.engine == "pi"
                            and candidate is None
                            and active is None
                            and topic_key not in fresh_pending
                            and config_path is not None
                        ):
                            existing_initial = await live.inbox.initial_for_topic(
                                chat_id, topic_key[1]
                            )
                            if existing_initial is not None:
                                await reply(
                                    text=f"Initial Pi task #{existing_initial.message_id} is unresolved after restart; no new prompt started. Use scoped retry-initial or defer-initial."
                                )
                                return
                            provisional = live_session_path(
                                config_path, chat_id, topic_key[1]
                            )
                            cwd = (
                                cfg.runtime.resolve_run_cwd(resolved_live.context)
                                or Path.cwd()
                            )
                            await live.inbox.receive_initial(
                                chat_id,
                                topic_key[1],
                                msg.message_id,
                                str(provisional),
                                resolved_live.prompt,
                                str(cwd.resolve()),
                            )
                            fresh_pending[topic_key] = (provisional, msg.message_id)
                            starting_live.add((chat_id, topic_key[1], str(provisional)))
                forward_coalescer.schedule(pending)

            allowed_user_ids = set(cfg.allowed_user_ids)

            async def route_update(update: TelegramIncomingUpdate) -> None:
                if allowed_user_ids:
                    sender_id = update.sender_id
                    if sender_id is None or sender_id not in allowed_user_ids:
                        logger.debug(
                            "update.ignored",
                            reason="sender_not_allowed",
                            chat_id=update.chat_id,
                            sender_id=sender_id,
                        )
                        return
                if update.update_id is not None:
                    update_id = update.update_id
                    if update_id in state.seen_update_ids:
                        logger.debug(
                            "update.ignored",
                            reason="duplicate_update",
                            update_id=update_id,
                            chat_id=update.chat_id,
                            sender_id=update.sender_id,
                        )
                        return
                    state.seen_update_ids.add(update_id)
                    state.seen_update_order.append(update_id)
                    if len(state.seen_update_order) > _SEEN_UPDATES_LIMIT:
                        oldest_update_id = state.seen_update_order.popleft()
                        state.seen_update_ids.discard(oldest_update_id)
                elif isinstance(update, TelegramIncomingMessage):
                    key = (update.chat_id, update.message_id)
                    if key in state.seen_message_keys:
                        logger.debug(
                            "update.ignored",
                            reason="duplicate_message",
                            chat_id=update.chat_id,
                            message_id=update.message_id,
                            sender_id=update.sender_id,
                        )
                        return
                    state.seen_message_keys.add(key)
                    state.seen_messages_order.append(key)
                    if len(state.seen_messages_order) > _SEEN_MESSAGES_LIMIT:
                        oldest = state.seen_messages_order.popleft()
                        state.seen_message_keys.discard(oldest)
                if isinstance(update, TelegramCallbackQuery):
                    if update.data == CANCEL_CALLBACK_DATA:
                        tg.start_soon(
                            handle_callback_cancel,
                            cfg,
                            update,
                            state.running_tasks,
                            scheduler,
                        )
                    elif update.data == STEER_CALLBACK_DATA:
                        tg.start_soon(
                            handle_callback_steer,
                            cfg,
                            update,
                            state.running_tasks,
                            scheduler,
                        )
                    elif update.data:
                        command_id, args_text = parse_callback_data(update.data)
                        if command_id not in state.command_ids:
                            refresh_commands()
                        if command_id in state.command_ids:
                            callback_msg = _callback_message(update)
                            ctx = await build_message_context(callback_msg)
                            engine_resolution = await resolve_engine_defaults(
                                explicit_engine=None,
                                context=ctx.ambient_context,
                                chat_id=ctx.chat_id,
                                topic_key=ctx.topic_key,
                            )
                            default_engine_override = (
                                engine_resolution.engine
                                if engine_resolution.source
                                in {"directive", "topic_default", "chat_default"}
                                else None
                            )
                            overrides_thread_id = (
                                ctx.topic_key[1]
                                if ctx.topic_key is not None
                                else callback_msg.thread_id
                            )
                            engine_overrides_resolver = partial(
                                _resolve_engine_run_options,
                                ctx.chat_id,
                                overrides_thread_id,
                                chat_prefs=state.chat_prefs,
                                topic_store=state.topic_store,
                            )
                            tg.start_soon(
                                cfg.bot.answer_callback_query,
                                update.callback_query_id,
                            )
                            tg.start_soon(
                                dispatch_command,
                                cfg,
                                callback_msg,
                                update.data,
                                command_id,
                                args_text,
                                state.running_tasks,
                                scheduler,
                                wrap_on_thread_known(
                                    scheduler.note_thread_known,
                                    ctx.topic_key,
                                    ctx.chat_session_key,
                                    None,
                                ),
                                ctx.stateful_mode,
                                default_engine_override,
                                engine_overrides_resolver,
                            )
                        else:
                            tg.start_soon(
                                cfg.bot.answer_callback_query,
                                update.callback_query_id,
                            )
                    else:
                        tg.start_soon(
                            cfg.bot.answer_callback_query,
                            update.callback_query_id,
                        )
                    return
                await route_message(update)

            try:
                async for update in poller_fn(cfg):
                    await route_update(update)
            finally:
                if live is not None:
                    live.stop_workers()
                    recovery_stopping = True
                    recovery_notice.set()
    finally:
        await cfg.exec_cfg.transport.close()
