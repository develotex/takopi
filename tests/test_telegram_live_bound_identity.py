from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from takopi.config import ProjectConfig, ProjectsConfig
from takopi.context import RunContext
from takopi.markdown import MarkdownPresenter
from takopi.model import ResumeToken
from takopi.router import AutoRouter, RunnerEntry
from takopi.runner_bridge import ExecBridgeConfig
from takopi.runners.pi import PiRunner
from takopi.runners.pi_rpc import PiRpcRun
from takopi.settings import TelegramTopicsSettings
from takopi.telegram.bridge import TelegramBridgeConfig, run_main_loop
from takopi.telegram.topic_state import TopicStateStore, resolve_state_path
from takopi.telegram.types import TelegramIncomingMessage
from takopi.transport import Transport
from takopi.transport_runtime import TransportRuntime
from tests.telegram_fakes import FakeBot, FakeTransport


@pytest.mark.anyio
@pytest.mark.parametrize("case", ["missing", "foreign_project"])
async def test_bound_pi_session_fails_closed_before_any_prompt(
    tmp_path: Path, case: str
):
    config = tmp_path / "takopi.toml"
    path = tmp_path / "pi-live-sessions" / "-100-77-existing.jsonl"
    if case == "foreign_project":
        path.parent.mkdir()
        path.write_text(
            json.dumps({"type": "session", "id": "abc", "cwd": str(tmp_path / "other")})
            + "\n"
        )

    class Rpc:
        def __init__(self):
            self.client = self
            self.session_path = path
            self.requests = []

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

        async def run(self, *_):
            raise AssertionError("Unverified bound Pi prompt forbidden")
            yield  # pragma: no cover

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
    await store.set_session_resume(-100, 77, ResumeToken(engine="pi", value=str(path)))
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
    assert rpc.requests == ([] if case == "missing" else ["get_state"])
    assert any(
        "no prompt submitted" in item["message"].text.lower()
        for item in transport.send_calls
    )
