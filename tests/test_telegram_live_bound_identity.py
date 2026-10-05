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
from takopi.model import ResumeToken
from takopi.router import AutoRouter, RunnerEntry
from takopi.runner_bridge import ExecBridgeConfig
from takopi.runners.pi import PiRunner
from takopi.runners.pi_rpc import PiRpcClient, PiRpcRun
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
