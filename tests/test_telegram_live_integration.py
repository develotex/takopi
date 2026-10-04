from __future__ import annotations

import pytest

from takopi.telegram.live_conversation import (
    LiveConversationService,
    LiveOwner,
    LiveRunner,
)
from takopi.telegram.live_inbox import LiveInbox
from pathlib import Path
from typing import cast

import anyio
from takopi.config import ProjectConfig, ProjectsConfig
from takopi.context import RunContext
from takopi.markdown import MarkdownPresenter
from takopi.model import ResumeToken, StartedEvent
from takopi.router import AutoRouter, RunnerEntry
from takopi.runner_bridge import ExecBridgeConfig
from takopi.runners.pi import PiRunner
from takopi.settings import TelegramTopicsSettings
from takopi.telegram.bridge import TelegramBridgeConfig, run_main_loop
from takopi.telegram.live_inbox import resolve_inbox_path
from takopi.telegram.topic_state import TopicStateStore, resolve_state_path
from takopi.telegram.types import TelegramIncomingMessage
from takopi.transport_runtime import TransportRuntime
from takopi.transport import Transport
from tests.telegram_fakes import FakeBot, FakeTransport
from takopi.model import CompletedEvent
from takopi.runners.pi_rpc import PiRpcRun


class FakePi:
    engine = "pi"

    def format_resume(self, token):
        return token.value

    def is_resume_line(self, line):
        return False

    def extract_resume(self, text):
        return None


class FakeRun:
    def __init__(self):
        self.client = self
        self._active = True
        self.sent: list[str] = []
        self.entries: list[dict] = []

    async def steer(self, text):
        self.sent.append(text)
        return "queued"

    async def request(self, typ):
        assert typ == "get_messages"
        return {"data": {"messages": self.entries}}

    async def run(self, prompt, resume):
        from takopi.model import CompletedEvent

        self.sent.append(prompt)
        yield CompletedEvent(engine="pi", ok=True, answer="done")


@pytest.mark.anyio
async def test_adapter_uses_same_rpc_owner_and_observed_messages(tmp_path: Path):
    rpc = FakeRun()
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))

    async def answer(*_):
        return "answer"

    async def reply(*_):
        return None

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    owner = LiveOwner(1, 10, str(tmp_path / "s.jsonl"), runner, "task")
    svc.register(owner)
    assert await svc.handle(1, 10, 33, owner.session_key, "/update use green")
    assert rpc.sent[0].endswith("use green")
    receipt = (await svc.inbox.pending(owner.session_key))[0]
    assert receipt.state == "submitted"
    rpc.entries = [
        {"role": "user", "content": await svc.inbox.delivery_text(receipt.id)}
    ]
    await svc.reconcile(owner)
    assert (await svc.inbox.get(receipt.id)).state == "delivered"


@pytest.mark.anyio
async def test_settlement_race_uses_same_owner_followup_once(tmp_path: Path):
    rpc = FakeRun()
    rpc._active = False
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))
    replies = []

    async def answer(*_):
        return "answer"

    async def reply(*args):
        replies.append(args)

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    owner = LiveOwner(1, 10, str(tmp_path / "s.jsonl"), runner, "task")
    svc.register(owner)
    receipt = await svc.inbox.receive(1, 10, 33, owner.session_key, "use green")
    await svc.flush(owner)
    assert runner._followups == [await svc.inbox.delivery_text(receipt.id)]
    await svc.submit(owner, receipt)
    assert len(runner._followups) == 1
    assert not rpc.sent, "only the streaming run may start the continuation"
    assert (await svc.inbox.get(receipt.id)).state == "submitted"


@pytest.mark.anyio
@pytest.mark.parametrize("legacy", [False, True])
async def test_real_loop_accepts_rapid_live_text_before_forward_coalescing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy: bool,
):
    class HeldRpc(FakeRun):
        def __init__(self, path: Path):
            super().__init__()
            self.path = path
            self.release = anyio.Event()
            self._active = False

        async def run(self, prompt, resume):
            self._active = True
            yield StartedEvent(
                engine="pi", resume=ResumeToken(engine="pi", value=str(self.path))
            )
            try:
                await self.release.wait()
                yield CompletedEvent(engine="pi", ok=True, answer="done")
            finally:
                self._active = False

        async def close(self):
            return None

    class TestPi(PiRunner):
        async def run(self, prompt, resume):
            raise AssertionError("live mode must not silently use one-shot Pi")
            yield  # pragma: no cover

        def rpc_run(self, session_path: Path, *, cwd: Path) -> PiRpcRun:
            assert session_path == path
            return cast(PiRpcRun, rpc)

    import takopi.telegram.loop as loop

    async def quick(_question: str, _snapshot: str) -> str:
        return "No observed result yet."

    monkeypatch.setattr(loop, "quick_pi_answer", quick)
    path = (
        tmp_path
        / ("existing" if legacy else "pi-live-sessions")
        / ("old-topic.jsonl" if legacy else "-100-77-test.jsonl")
    ).resolve()
    if legacy:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("existing session fixture")
    rpc = HeldRpc(path)
    runner = TestPi(extra_args=[], model=None, provider=None)
    config = tmp_path / "takopi.toml"
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
    await store.set_session_resume(-100, 77, ResumeToken(engine="pi", value=str(path)))
    progress_ready = anyio.Event()
    transport = FakeTransport(progress_ready=progress_ready)
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
        forward_coalesce_s=0.4,
    )
    emitted = anyio.Event()

    async def poller(_cfg):
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=77,
            message_id=1,
            text="start",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )
        await progress_ready.wait()
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=77,
            message_id=2,
            text="Не трогай авторизацию",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=77,
            message_id=3,
            text="Как дела?",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )
        emitted.set()
        await rpc.release.wait()

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_main_loop, cfg, poller)
        try:
            with anyio.fail_after(3):
                await emitted.wait()
            pending = await LiveInbox(resolve_inbox_path(config)).pending(str(path))
            assert [item.text for item in pending] == ["Не трогай авторизацию"]
            assert [item.state for item in pending] == ["submitted"]
            assert len(rpc.sent) == 1
        finally:
            rpc.release.set()
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_short_legacy_pi_topic_does_not_silently_fallback_to_one_shot(
    tmp_path: Path,
):
    class ShortIdPi(PiRunner):
        async def run(self, prompt, resume):
            raise AssertionError("unsafe one-shot fallback")
            yield  # pragma: no cover

    runner = ShortIdPi(extra_args=[], model=None, provider=None)
    config = tmp_path / "takopi.toml"
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
        -100, 77, ResumeToken(engine="pi", value="legacy-short-id")
    )
    transport = FakeTransport()
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

    async def poller(_cfg):
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=77,
            message_id=1,
            text="start",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )

    await run_main_loop(cfg, poller)
    assert any(
        "не найден однозначно" in item["message"].text for item in transport.send_calls
    )
    assert await store.get_session_resume(-100, 77, "pi") == ResumeToken(
        engine="pi", value="legacy-short-id"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("wrong_state", [False, True])
async def test_bound_short_id_migrates_only_after_canonical_get_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrong_state: bool
):
    import json
    from uuid import uuid4
    from takopi.runners.pi import _default_session_dir

    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi"))
    ident = uuid4().hex
    prefix = ident[:10]
    session_dir = _default_session_dir(tmp_path)
    session_dir.mkdir(parents=True)
    session = session_dir / f"2026-10-04_{ident}.jsonl"
    session.write_text(
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": ident,
                "cwd": str(tmp_path),
                "timestamp": "2026-10-04T00:00:00.000Z",
            }
        )
        + "\n"
    )

    class MigratingRpc(FakeRun):
        def __init__(self):
            super().__init__()
            self._active = False
            self.calls: list[str] = []

        async def request(self, typ):
            self.calls.append(typ)
            if typ == "get_state":
                return {
                    "data": {
                        "sessionId": "wrong-id" if wrong_state else ident,
                        "sessionFile": str(session),
                        "isStreaming": False,
                    }
                }
            return {"data": {"messages": []}}

        async def run(self, prompt, resume):
            self.calls.append("prompt")
            yield StartedEvent(
                engine="pi", resume=ResumeToken(engine="pi", value=str(session))
            )
            yield CompletedEvent(engine="pi", ok=True, answer="done")

        async def close(self):
            pass

    rpc = MigratingRpc()

    class MigratingPi(PiRunner):
        async def run(self, prompt, resume):
            raise AssertionError("unsafe one-shot fallback")
            yield  # pragma: no cover

        def rpc_run(self, session_path: Path, *, cwd: Path) -> PiRpcRun:
            assert session_path == session.resolve()
            return cast(PiRpcRun, rpc)

    runner = MigratingPi(extra_args=[], model=None, provider=None)
    config = tmp_path / "takopi.toml"
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
    await store.set_session_resume(-100, 77, ResumeToken(engine="pi", value=prefix))
    transport = FakeTransport()
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

    async def poller(_cfg):
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=77,
            message_id=1,
            text="start",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )

    await run_main_loop(cfg, poller)
    if wrong_state:
        assert rpc.calls == ["get_state"]
        assert await store.get_session_resume(-100, 77, "pi") == ResumeToken(
            "pi", prefix
        )
    else:
        assert rpc.calls[:2] == ["get_state", "prompt"]
        assert await store.get_session_resume(-100, 77, "pi") == ResumeToken(
            "pi", str(session.resolve())
        )


@pytest.mark.anyio
@pytest.mark.parametrize("cancel", [False, True])
async def test_settlement_continuation_is_rendered_before_final_and_cancellable(
    tmp_path: Path, cancel: bool
):
    import asyncio

    class SettlingRun(FakeRun):
        def __init__(self):
            super().__init__()
            self.started_followup = asyncio.Event()
            self.release_followup = asyncio.Event()

        async def run(self, prompt, resume):
            self._active = True
            self.sent.append(prompt)
            if len(self.sent) == 2:
                self.started_followup.set()
                await self.release_followup.wait()
                answer = "UPDATED FINAL RESULT"
            else:
                answer = "outdated result"
            self._active = False
            yield CompletedEvent(engine="pi", ok=True, answer=answer)

    rpc = SettlingRun()
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))

    async def answer(*_):
        return "answer"

    async def reply(*_):
        return None

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    owner = LiveOwner(1, 10, str(tmp_path / "s.jsonl"), runner, "task")
    svc.register(owner)
    receipt = await svc.inbox.receive(1, 10, 33, owner.session_key, "use green")
    runner.before_final = lambda: svc.flush(owner)
    emitted: list[CompletedEvent] = []

    async def consume():
        emitted.extend(
            [
                event
                async for event in runner.run("build", None)
                if isinstance(event, CompletedEvent)
            ]
        )

    task = asyncio.create_task(consume())
    await asyncio.wait_for(rpc.started_followup.wait(), 1)
    assert not emitted, "old answer must not be final before queued update completes"
    assert (await svc.inbox.get(receipt.id)).state == "submitted"
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        rpc.release_followup.set()
        assert not emitted, "cancelled follow-up must not report success"
    else:
        rpc.release_followup.set()
        await asyncio.wait_for(task, 1)
        assert [event.answer for event in emitted] == ["UPDATED FINAL RESULT"]


@pytest.mark.anyio
async def test_main_prompt_teaches_explicit_ack_without_quick_transcript():
    rpc = FakeRun()
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))
    _ = [event async for event in runner.run("build feature", None)]
    assert "[takopi-considered:" in rpc.sent[0]
    assert "[takopi-deferred:" in rpc.sent[0]
    assert "build feature" in rpc.sent[0]
