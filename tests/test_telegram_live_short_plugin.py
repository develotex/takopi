from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import anyio
import pytest

from takopi.config import ProjectConfig, ProjectsConfig
from takopi.context import RunContext
from takopi.markdown import MarkdownPresenter
from takopi.model import CompletedEvent, ResumeToken, StartedEvent
from takopi.router import AutoRouter, RunnerEntry
from takopi.runner_bridge import ExecBridgeConfig
from takopi.runners.pi import PiRunner, _default_session_dir
from takopi.runners.pi_rpc import PiRpcRun
from takopi.settings import TelegramTopicsSettings
from takopi.telegram.bridge import TelegramBridgeConfig, run_main_loop
from takopi.telegram.topic_state import TopicStateStore, resolve_state_path
from takopi.telegram.types import TelegramCallbackQuery, TelegramIncomingMessage
from takopi.transport import Transport
from takopi.transport_runtime import TransportRuntime
from tests.telegram_fakes import FakeBot, FakeTransport


@pytest.mark.anyio
async def test_short_id_migration_reserves_topic_before_awaiting_owner_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import takopi.telegram.loop as loop

    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi"))
    config = tmp_path / "takopi.toml"
    ident = "abcdef0123456789abcdef0123456789"
    path = _default_session_dir(tmp_path) / f"2026-10-05_{ident}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"type": "session", "version": 3, "id": ident, "cwd": str(tmp_path)})
        + "\n"
    )
    entered, release = anyio.Event(), anyio.Event()
    dispatched: list[str] = []
    prompts: list[str] = []
    original_owners = TopicStateStore.session_owners

    async def delayed_owners(self, *args, **kwargs):
        if not entered.is_set():
            entered.set()
            await release.wait()
        return await original_owners(self, *args, **kwargs)

    monkeypatch.setattr(TopicStateStore, "session_owners", delayed_owners)
    monkeypatch.setattr(loop, "list_command_ids", lambda **_: ["spy"])

    async def fake_dispatch(*_args, **_kwargs):
        dispatched.append("plugin")

    monkeypatch.setattr(loop, "dispatch_command", fake_dispatch)

    class Rpc:
        session_path = path
        cwd = tmp_path
        client = None
        _active = False
        _prompt_started = False

        async def request(self, typ):
            assert typ == "get_state"
            return {
                "data": {
                    "sessionId": ident,
                    "sessionFile": str(path),
                    "isStreaming": False,
                }
            }

        async def run(self, prompt, _resume):
            prompts.append(prompt)
            yield StartedEvent(engine="pi", resume=ResumeToken("pi", str(path)))
            yield CompletedEvent(engine="pi", ok=True, answer="done")

        async def close(self):
            pass

    rpc = Rpc()
    rpc.client = rpc

    class Runner(PiRunner):
        async def run(self, *_args):
            raise AssertionError("legacy short ID must migrate to Pi RPC")
            yield  # pragma: no cover

        def rpc_run(self, session_path: Path, *, cwd: Path) -> PiRpcRun:
            assert session_path == path and cwd == tmp_path
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
    store = TopicStateStore(resolve_state_path(config))
    await store.set_context(-100, 77, RunContext(project="test"))
    await store.set_session_resume(-100, 77, ResumeToken("pi", ident[:12]))
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
        with anyio.fail_after(2):
            await entered.wait()
        yield TelegramCallbackQuery(
            transport="telegram",
            chat_id=-100,
            message_id=2,
            callback_query_id="during-migration",
            data="spy:run",
            sender_id=123,
            raw={"message": {"message_thread_id": 77}},
        )
        await anyio.sleep(0.05)
        release.set()

    try:
        await run_main_loop(cfg, poller)
    finally:
        release.set()
    assert dispatched == [], "Plugin callback must not overlap short-ID migration"
    assert prompts, "Bound short ID should still resume after migration"
