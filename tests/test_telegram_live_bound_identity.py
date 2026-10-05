from __future__ import annotations

import asyncio
import json
from pathlib import Path

import anyio
from typing import cast

import pytest

from takopi.config import ProjectConfig, ProjectsConfig
from takopi.context import RunContext
from takopi.markdown import MarkdownPresenter
from takopi.model import CompletedEvent, ResumeToken, StartedEvent
from takopi.router import AutoRouter, RunnerEntry
from takopi.runner_bridge import ExecBridgeConfig
from takopi.runners.pi import PiRunner
from takopi.runners.pi_rpc import PiRpcClient, PiRpcRun
from takopi.settings import TelegramTopicsSettings
from takopi.telegram.bridge import TelegramBridgeConfig, run_main_loop
from takopi.telegram.live_inbox import LiveInbox, resolve_inbox_path
from takopi.telegram.topic_state import TopicStateStore, resolve_state_path
from takopi.telegram.types import TelegramCallbackQuery, TelegramIncomingMessage
from takopi.transport import Transport
from takopi.transport_runtime import TransportRuntime
from tests.telegram_fakes import FakeBot, FakeTransport


@pytest.mark.anyio
@pytest.mark.parametrize("case", ["missing", "foreign_project", "relative"])
async def test_bound_pi_session_fails_closed_before_any_prompt(
    tmp_path: Path, case: str
):
    config = tmp_path / "takopi.toml"
    path = tmp_path / "pi-live-sessions" / "-100-77-existing.jsonl"
    if case in ("foreign_project", "relative"):
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "type": "session",
                    "id": "abc",
                    "cwd": str(tmp_path if case == "relative" else tmp_path / "other"),
                }
            )
            + "\n"
        )

    class Rpc:
        def __init__(self):
            self.client = self
            self.session_path = path
            self.requests = []
            self.prompts = []
            self.cwd = tmp_path
            self._active = False
            self._prompt_started = False

        async def request(self, typ):
            self.requests.append(typ)
            assert typ == "get_state"
            return {
                "data": {
                    "sessionId": "abc",
                    "sessionFile": str(path),
                    "isStreaming": False,
                }
            }

        async def run(self, prompt, _resume):
            if case != "relative":
                raise AssertionError("Unverified bound Pi prompt forbidden")
            self.prompts.append(prompt)
            yield StartedEvent(engine="pi", resume=ResumeToken("pi", str(path)))
            yield CompletedEvent(engine="pi", ok=True, answer="done")

        async def close(self):
            pass

    rpc = Rpc()

    class Runner(PiRunner):
        async def run(self, *_):
            raise AssertionError("One-shot fallback forbidden")
            yield  # pragma: no cover

        def rpc_run(self, session_path: Path, *, cwd: Path) -> PiRpcRun:
            assert session_path == path and cwd == tmp_path
            return cast(PiRpcRun, rpc)

    runner = Runner(extra_args=[], model=None, provider=None)
    runtime = TransportRuntime(
        router=AutoRouter([RunnerEntry(engine="pi", runner=runner)], "pi"),
        projects=ProjectsConfig(
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
        ),
        config_path=config,
    )
    store = TopicStateStore(resolve_state_path(config))
    await store.set_context(-100, 77, RunContext(project="test"))
    await store.set_session_resume(
        -100,
        77,
        ResumeToken(
            engine="pi",
            value="pi-live-sessions/-100-77-existing.jsonl"
            if case == "relative"
            else str(path),
        ),
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
            message_id=50,
            text="Perform task",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )

    await run_main_loop(cfg, poller)
    if case == "relative":
        assert rpc.prompts and rpc.prompts[0].endswith("Perform task")
        assert await store.get_session_resume(-100, 77, "pi") == ResumeToken(
            "pi", str(path)
        )
        return
    assert rpc.requests == ([] if case == "missing" else ["get_state"])
    assert any(
        "no prompt submitted" in item["message"].text.lower()
        for item in transport.send_calls
    )


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["delayed_plugin", "unknown_callback"])
async def test_plugin_cannot_race_live_start_or_run_with_unknown_topic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    import takopi.telegram.loop as loop

    config = tmp_path / "takopi.toml"
    entered, release, emitted = anyio.Event(), anyio.Event(), anyio.Event()
    dispatches: list[str] = []
    monkeypatch.setattr(loop, "list_command_ids", lambda **_: ["spy"])

    async def fake_dispatch(*_args, **_kwargs):
        dispatches.append("started")
        entered.set()
        await release.wait()

    monkeypatch.setattr(loop, "dispatch_command", fake_dispatch)

    class Runner(PiRunner):
        def rpc_run(self, *_args, **_kwargs):
            raise AssertionError("live start must wait for delayed plugin")

        async def run(self, *_args):
            raise AssertionError("plugin/initial must not start Pi concurrently")
            yield  # pragma: no cover

    runtime = TransportRuntime(
        router=AutoRouter(
            [
                RunnerEntry(
                    engine="pi", runner=Runner(extra_args=[], model=None, provider=None)
                )
            ],
            "pi",
        ),
        projects=ProjectsConfig(
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
        ),
        config_path=config,
    )
    cfg = TelegramBridgeConfig(
        bot=FakeBot(),
        runtime=runtime,
        chat_id=-100,
        startup_msg="",
        exec_cfg=ExecBridgeConfig(
            transport=cast(Transport, FakeTransport()),
            presenter=MarkdownPresenter(),
            final_notify=True,
        ),
        topics=TelegramTopicsSettings(enabled=True, scope="projects"),
        pi_live_conversation=True,
        forward_coalesce_s=0,
    )

    async def poller(_cfg):
        if mode == "unknown_callback":
            yield TelegramCallbackQuery(
                transport="telegram",
                chat_id=-100,
                message_id=1,
                callback_query_id="missing-thread",
                data="spy:run",
                sender_id=123,
                raw={"message": {}},
            )
        else:
            yield TelegramIncomingMessage(
                transport="telegram",
                chat_id=-100,
                thread_id=77,
                message_id=1,
                text="/spy run",
                reply_to_message_id=None,
                reply_to_text=None,
                sender_id=123,
            )
            await entered.wait()
            yield TelegramIncomingMessage(
                transport="telegram",
                chat_id=-100,
                thread_id=77,
                message_id=2,
                text="Build task",
                reply_to_message_id=None,
                reply_to_text=None,
                sender_id=123,
            )
        emitted.set()
        await release.wait()

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_main_loop, cfg, poller)
        try:
            with anyio.fail_after(2):
                await emitted.wait()
            await anyio.sleep(0.05)
            if mode == "unknown_callback":
                assert dispatches == [], (
                    "plugin callback with unproven topic must fail closed"
                )
            else:
                assert dispatches == ["started"]
                assert (
                    await LiveInbox(resolve_inbox_path(config)).initial_for_topic(
                        -100, 77
                    )
                    is None
                )
        finally:
            release.set()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("phase", "reply_kind"),
    [
        ("identity", "none"),
        ("identity", "original"),
        ("identity", "root"),
        ("debounce", "none"),
        ("debounce", "root"),
        ("empty_forward", "none"),
    ],
)
async def test_telegram_cancel_while_fresh_identity_is_pending_prevents_prompt(
    tmp_path: Path,
    phase: str,
    reply_kind: str,
) -> None:
    config = tmp_path / "takopi.toml"
    entered, release, emitted = anyio.Event(), anyio.Event(), anyio.Event()
    rpc_runs: list[Rpc] = []

    class Rpc:
        def __init__(self, session_path: Path):
            self.client = self
            self.session_path = session_path
            self.cwd = tmp_path
            self.prompt_sent = False

        async def request(self, typ: str):
            assert typ == "get_state"
            entered.set()
            await release.wait()
            return {
                "data": {
                    "sessionId": "fixture",
                    "sessionFile": str(self.session_path),
                    "isStreaming": False,
                }
            }

        async def run(self, *_args):
            self.prompt_sent = True
            if phase == "empty_forward":
                yield StartedEvent(
                    engine="pi", resume=ResumeToken("pi", str(self.session_path))
                )
                yield CompletedEvent(engine="pi", ok=True, answer="done")
                return
            raise AssertionError("cancelled startup must not submit a Pi prompt")

        async def close(self):
            pass

    class Runner(PiRunner):
        def rpc_run(self, session_path: Path, *, cwd: Path) -> PiRpcRun:
            rpc = Rpc(session_path)
            rpc_runs.append(rpc)
            return cast(PiRpcRun, rpc)

    runtime = TransportRuntime(
        router=AutoRouter(
            [
                RunnerEntry(
                    engine="pi", runner=Runner(extra_args=[], model=None, provider=None)
                )
            ],
            "pi",
        ),
        projects=ProjectsConfig(
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
        ),
        config_path=config,
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
        forward_coalesce_s=0.3 if phase in ("debounce", "empty_forward") else 0,
    )

    async def poller(_cfg):
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=77,
            message_id=1,
            text="Build task",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )
        if phase == "empty_forward":
            await anyio.sleep(0.05)
            yield TelegramIncomingMessage(
                transport="telegram",
                chat_id=-100,
                thread_id=77,
                message_id=2,
                text="",
                reply_to_message_id=None,
                reply_to_text=None,
                sender_id=123,
                raw={"forward_date": 1},
            )
            with anyio.fail_after(2):
                await entered.wait()
            yield TelegramIncomingMessage(
                transport="telegram",
                chat_id=-100,
                thread_id=77,
                message_id=3,
                text="/update Use green",
                reply_to_message_id=None,
                reply_to_text=None,
                sender_id=123,
            )
            emitted.set()
            await asyncio.Event().wait()
            return
        if phase == "identity":
            await entered.wait()
        else:
            await anyio.sleep(0.05)
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=77,
            message_id=2,
            text="/cancel",
            reply_to_message_id={"original": 1, "root": 77}.get(reply_kind),
            reply_to_text="Build task" if reply_kind == "original" else None,
            sender_id=123,
        )
        emitted.set()
        await asyncio.Event().wait()

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_main_loop, cfg, poller)
        with anyio.fail_after(2):
            await emitted.wait()
        await anyio.sleep(0.05)
        if phase == "empty_forward":
            assert rpc_runs
            assert (
                await LiveInbox(resolve_inbox_path(config)).get((-100, 77, 3))
            ).text == "Use green"
            release.set()
            tg.cancel_scope.cancel()
            return
        release.set()
        await anyio.sleep(0.4 if phase == "debounce" else 0.1)
        assert all(not rpc.prompt_sent for rpc in rpc_runs)
        if phase == "identity":
            assert rpc_runs
        assert (
            await LiveInbox(resolve_inbox_path(config)).initial_for_topic(-100, 77)
            is None
        )
        assert not any(
            "nothing is currently running" in call["message"].text
            for call in transport.send_calls
        )
        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_cancel_during_fresh_identity_request_releases_rpc_claim_and_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from takopi.runners import pi_rpc

    entered = anyio.Event()
    config = tmp_path / "takopi.toml"
    client: PiRpcClient | None = None

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.Event().wait()
            raise StopAsyncIteration

    class Stdin:
        async def send(self, _data: bytes):
            entered.set()

        async def aclose(self):
            pass

    class Proc:
        pid = 999999
        stdin = Stdin()
        stdout = Stream()
        stderr = Stream()
        returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

    async def fake_open(*_args, **_kwargs):
        return Proc()

    monkeypatch.setattr(pi_rpc.anyio, "open_process", fake_open)
    monkeypatch.setattr(pi_rpc.os, "killpg", lambda *_args: None)

    class Runner(PiRunner):
        def rpc_run(self, session_path: Path, *, cwd: Path) -> PiRpcRun:
            nonlocal client
            assert cwd == tmp_path
            client = PiRpcClient(session_path, tmp_path, [])
            return PiRpcRun(client)

    runtime = TransportRuntime(
        router=AutoRouter(
            [
                RunnerEntry(
                    engine="pi", runner=Runner(extra_args=[], model=None, provider=None)
                )
            ],
            "pi",
        ),
        projects=ProjectsConfig(
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
        ),
        config_path=config,
    )
    cfg = TelegramBridgeConfig(
        bot=FakeBot(),
        runtime=runtime,
        chat_id=-100,
        startup_msg="",
        exec_cfg=ExecBridgeConfig(
            transport=cast(Transport, FakeTransport()),
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
            text="Build task",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )
        await asyncio.Event().wait()

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_main_loop, cfg, poller)
            with anyio.fail_after(2):
                await entered.wait()
            tg.cancel_scope.cancel()
        assert client is not None
        assert client._closed, "cancelled identity verification must close RPC"
        assert pi_rpc._OWNERS.get(client.session_path) is None
        assert all(task.done() for task in client._tasks)
    finally:
        if client is not None:
            await client.close()
