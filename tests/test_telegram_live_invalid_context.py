from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from takopi.config import ProjectConfig, ProjectsConfig
from takopi.markdown import MarkdownPresenter
from takopi.router import AutoRouter, RunnerEntry
from takopi.runner_bridge import ExecBridgeConfig
from takopi.runners.pi import PiRunner
from takopi.settings import TelegramTopicsSettings
from takopi.telegram.bridge import TelegramBridgeConfig, run_main_loop
from takopi.telegram.live_inbox import LiveInbox, resolve_inbox_path
from takopi.telegram.types import TelegramIncomingMessage
from takopi.transport import Transport
from takopi.transport_runtime import TransportRuntime
from tests.telegram_fakes import FakeBot, FakeTransport


@pytest.mark.anyio
async def test_invalid_project_directive_does_not_terminate_live_poller(
    tmp_path: Path,
) -> None:
    config = tmp_path / "takopi.toml"
    runtime = TransportRuntime(
        router=AutoRouter(
            [
                RunnerEntry(
                    engine="pi",
                    runner=PiRunner(extra_args=[], model=None, provider=None),
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
        forward_coalesce_s=0,
    )

    async def poller(_cfg):
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=77,
            message_id=1,
            text="@../escape Build task",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=77,
            message_id=2,
            text="/cancel",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )

    await run_main_loop(cfg, poller)
    assert any("error:" in item["message"].text for item in transport.send_calls)
    assert (
        await LiveInbox(resolve_inbox_path(config)).initial_for_topic(-100, 77) is None
    )
