from __future__ import annotations

import json
import anyio
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest

from takopi.config import ProjectConfig, ProjectsConfig
from takopi.context import RunContext
from takopi.markdown import MarkdownPresenter
from takopi.model import CompletedEvent, ResumeToken, StartedEvent
from takopi.router import AutoRouter, RunnerEntry
from takopi.runner_bridge import ExecBridgeConfig
from takopi.runners.pi import PiRunner
from takopi.runners.pi_rpc import PiRpcRun
from takopi.settings import TelegramTopicsSettings
from takopi.telegram.bridge import TelegramBridgeConfig, run_main_loop
from takopi.telegram.live_inbox import LiveInbox, resolve_inbox_path
from takopi.telegram.topic_state import TopicStateStore, resolve_state_path
from takopi.telegram.types import TelegramIncomingMessage
from takopi.transport import Transport
from takopi.transport_runtime import TransportRuntime
from tests.telegram_fakes import FakeBot, FakeTransport


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case",
    [
        "retry",
        "retry_fifo",
        "retry_cancel",
        "finalize_order",
        "observed",
        "conflict",
        "wrong_topic",
        "wrong_session",
        "wrong_id",
        "mismatch",
        "streaming",
        "wrong_project",
    ],
)
async def test_idle_restart_retry_uses_exact_canonical_owner_and_first_marker(
    tmp_path: Path,
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tmp_path / "existing.jsonl"
    ident = uuid4().hex
    session.write_text(
        json.dumps(
            {
                "type": "session",
                "id": ident,
                "cwd": str(tmp_path / "other")
                if case == "wrong_project"
                else str(tmp_path),
            }
        )
        + "\n"
    )
    config = tmp_path / "takopi.toml"
    inbox = LiveInbox(resolve_inbox_path(config))
    receipt = await inbox.receive(-100, 77, 33, str(session), "Use green")
    await inbox.mark_uncertain(receipt.id, "lost RPC response")

    class Rpc:
        def __init__(self):
            self.client = self
            self.session_path = session
            self._active = False
            self.prompts: list[str] = []
            self.entries: list[dict] = []
            self.requests: list[str] = []
            self.closed = False
            self.started = anyio.Event()
            self.release = anyio.Event()
            self.steered: list[str] = []

        async def request(self, typ: str):
            self.requests.append(typ)
            if typ == "get_state":
                if case == "conflict":
                    raise RuntimeError("canonical owner conflict")
                return {
                    "data": {
                        "sessionId": "wrong" if case == "mismatch" else ident,
                        "sessionFile": str(session),
                        "isStreaming": case == "streaming",
                    }
                }
            if typ == "get_messages":
                return {"data": {"messages": self.entries}}
            raise AssertionError(f"Unexpected request {typ}")

        async def run(self, prompt, resume):
            self.prompts.append(prompt)
            assert resume == (
                ResumeToken(engine="pi", value=str(session))
                if len(self.prompts) == 1 or case == "finalize_order"
                else None
            )
            self._active = True
            yield StartedEvent(engine="pi", resume=resume)
            self.started.set()
            if case in ("retry_fifo", "retry_cancel"):
                await self.release.wait()
            self.entries.append({"role": "user", "content": prompt})
            self.entries.append(
                {
                    "role": "assistant",
                    "content": f"[takopi-considered:{receipt.marker}] Applied green to the task.",
                }
            )
            self._active = False
            yield CompletedEvent(
                engine="pi",
                ok=True,
                answer="Final with green"
                if len(self.prompts) == 1
                else "Next run final",
                resume=resume,
            )

        async def steer(self, text):
            assert self.prompts, "steer may not overtake retry prompt"
            self.steered.append(text)
            return "queued"

        async def close(self):
            self.closed = True

    rpc = Rpc()
    if case == "observed":
        rpc.entries = [
            {"role": "user", "content": await inbox.delivery_text(receipt.id)}
        ]

    class Runner(PiRunner):
        async def run(self, prompt, resume):
            raise AssertionError("One-shot Pi fallback forbidden")
            yield  # pragma: no cover

        def rpc_run(self, session_path: Path, *, cwd: Path) -> PiRpcRun:
            assert session_path == session and cwd == tmp_path
            return cast(PiRpcRun, rpc)

    runner = Runner(extra_args=[], model=None, provider=None)
    projects = ProjectsConfig(
        projects={
            "test": ProjectConfig(
                alias="test",
                path=tmp_path,
                worktrees_dir=tmp_path / ".worktrees",
                chat_id=-100,
            )
        },
        default_project=None,
        chat_map={-100: "test"},
    )
    runtime = TransportRuntime(
        router=AutoRouter([RunnerEntry(engine="pi", runner=runner)], "pi"),
        projects=projects,
        config_path=config,
    )
    store = TopicStateStore(resolve_state_path(config))
    await store.set_context(-100, 77, RunContext(project="test"))
    await store.set_session_resume(
        -100, 77, ResumeToken(engine="pi", value=str(session))
    )
    thread = 78 if case == "wrong_topic" else 77
    if case == "wrong_topic":
        await store.set_context(-100, 78, RunContext(project="test"))
        await store.set_session_resume(
            -100, 78, ResumeToken(engine="pi", value=str(session))
        )
    if case == "wrong_session":
        other = tmp_path / "other.jsonl"
        other.write_text(
            json.dumps({"type": "session", "id": uuid4().hex, "cwd": str(tmp_path)})
            + "\n"
        )
        await store.set_session_resume(
            -100, 77, ResumeToken(engine="pi", value=str(other))
        )

    class HeldFinalTransport(FakeTransport):
        def __init__(self):
            super().__init__()
            self.final_entered = anyio.Event()
            self.final_release = anyio.Event()
            self.held = False

        async def send(self, *, channel_id, message, options=None):
            if "Final with green" in message.text and not self.held:
                self.held = True
                self.final_entered.set()
                await self.final_release.wait()
            return await super().send(
                channel_id=channel_id, message=message, options=options
            )

    held_transport = HeldFinalTransport()
    transport = held_transport if case == "finalize_order" else FakeTransport()
    cfg = TelegramBridgeConfig(
        bot=FakeBot(),
        runtime=runtime,
        chat_id=-100,
        startup_msg="",
        exec_cfg=ExecBridgeConfig(
            transport=cast(Transport, transport),
            presenter=MarkdownPresenter(),
            final_notify=True,
        ),
        topics=TelegramTopicsSettings(enabled=True, scope="projects"),
        pi_live_conversation=True,
        forward_coalesce_s=0,
    )

    cancel_seen = anyio.Event()
    if case == "retry_cancel":
        import takopi.telegram.loop as loop

        original_cancel = loop.handle_cancel

        async def trace_cancel(*args):
            cancel_seen.set()
            await original_cancel(*args)

        monkeypatch.setattr(loop, "handle_cancel", trace_cancel)
    emitted = anyio.Event()

    async def poller(_cfg):
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=thread,
            message_id=50,
            text="/update retry 999 confirm"
            if case == "wrong_id"
            else "/update retry 33 confirm",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )
        if case in ("retry_fifo", "retry_cancel", "finalize_order"):
            if case == "finalize_order":
                await held_transport.final_entered.wait()
            else:
                await rpc.started.wait()
            yield TelegramIncomingMessage(
                transport="telegram",
                chat_id=-100,
                thread_id=77,
                message_id=51,
                text=(
                    "Use blue"
                    if case == "retry_fifo"
                    else "Next task"
                    if case == "finalize_order"
                    else "/cancel"
                ),
                reply_to_message_id=50 if case == "retry_cancel" else None,
                reply_to_text="/update retry 33 confirm"
                if case == "retry_cancel"
                else None,
                sender_id=123,
            )
            emitted.set()
            if case == "finalize_order":
                await held_transport.final_release.wait()
            else:
                await rpc.release.wait()

    if case in ("retry_fifo", "retry_cancel", "finalize_order"):
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_main_loop, cfg, poller)
            try:
                with anyio.fail_after(2):
                    await emitted.wait()
                    if case == "retry_fifo":
                        pending = await inbox.pending(str(session))
                        assert [r.text for r in pending] == ["Use green", "Use blue"]
                        assert [r.state for r in pending] == ["uncertain", "received"]
                        assert not rpc.steered, (
                            "later receipt must not overtake uncertain retry"
                        )
                        assert rpc.prompts[0] == await inbox.delivery_text(receipt.id)
                        rpc.release.set()
                        while len(rpc.prompts) < 2:
                            await anyio.sleep(0.01)
                        assert rpc.prompts[1].endswith("Use blue")
                    elif case == "finalize_order":
                        assert len(rpc.prompts) == 1, (
                            "new job must wait for final rendering"
                        )
                        held_transport.final_release.set()
                        while len(rpc.prompts) < 2:
                            await anyio.sleep(0.01)
                        assert rpc.prompts[1].endswith("Next task")
                    else:
                        await cancel_seen.wait()
                        assert rpc.prompts[0] == await inbox.delivery_text(receipt.id)
            finally:
                rpc.release.set()
                if case == "finalize_order":
                    held_transport.final_release.set()
                tg.cancel_scope.cancel()
        return

    await run_main_loop(cfg, poller)
    final = await inbox.get(receipt.id)
    if case == "retry":
        assert rpc.requests[:2] == ["get_state", "get_messages"]
        assert rpc.prompts == [await inbox.delivery_text(receipt.id)], (
            "receipt marker must be first action"
        )
        assert final.state == "considered"
        assert any(
            "Final with green" in call["message"].text for call in transport.send_calls
        )
    elif case == "observed":
        assert not rpc.prompts and final.state == "delivered"
    else:
        assert not rpc.prompts and final.state == "uncertain"
        if case in ("conflict", "mismatch", "streaming"):
            assert "get_state" in rpc.requests
        else:
            assert not rpc.requests
