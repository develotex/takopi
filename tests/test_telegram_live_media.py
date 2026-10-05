"""Media-caption tasks must not claim a live initial intent without its full prompt."""

from pathlib import Path
from typing import cast

import pytest

from takopi.config import ProjectConfig, ProjectsConfig
from takopi.markdown import MarkdownPresenter
from takopi.model import CompletedEvent, StartedEvent, ResumeToken
from takopi.router import AutoRouter, RunnerEntry
from takopi.runner_bridge import ExecBridgeConfig
from takopi.runners.pi import PiRunner
from takopi.runners.pi_rpc import PiRpcRun
from takopi.settings import TelegramFilesSettings, TelegramTopicsSettings
from takopi.telegram.bridge import TelegramBridgeConfig, run_main_loop
from takopi.telegram.live_inbox import LiveInbox, resolve_inbox_path
from takopi.telegram.types import TelegramIncomingMessage, TelegramDocument
from takopi.telegram.topic_state import TopicStateStore, resolve_state_path
from takopi.transport import Transport
from takopi.transport_runtime import TransportRuntime
from tests.telegram_fakes import FakeBot, FakeTransport


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["fresh_caption", "bound_caption", "forward"])
async def test_composed_initial_prompt_routes_through_one_shot_not_live_rpc(
    tmp_path: Path, mode: str
):
    from takopi.telegram.api_models import File

    class UploadBot(FakeBot):
        async def get_file(self, file_id: str) -> File | None:
            return File(file_path="files/hello.txt")

        async def download_file(self, file_path: str) -> bytes | None:
            return b"hello"

    class Runner(PiRunner):
        def __init__(self):
            super().__init__(extra_args=[], model=None, provider=None)
            self.prompts: list[str] = []

        async def run(self, prompt: str, resume: ResumeToken | None):
            self.prompts.append(prompt)
            yield StartedEvent(
                engine="pi", resume=ResumeToken("pi", str(tmp_path / "legacy.jsonl"))
            )
            yield CompletedEvent(
                engine="pi",
                ok=True,
                answer="done",
                resume=ResumeToken("pi", str(tmp_path / "legacy.jsonl")),
            )

        def rpc_run(self, session_path: Path, *, cwd: Path) -> PiRpcRun:
            raise AssertionError(
                "media caption cannot create an incomplete live initial task"
            )

    runner = Runner()
    config = tmp_path / "takopi.toml"
    if mode == "bound_caption":
        await TopicStateStore(resolve_state_path(config)).set_session_resume(
            -100, 77, ResumeToken("pi", str(tmp_path / "existing.jsonl"))
        )
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
    transport = FakeTransport()
    cfg = TelegramBridgeConfig(
        bot=UploadBot(),
        runtime=runtime,
        chat_id=-100,
        startup_msg="",
        exec_cfg=ExecBridgeConfig(
            transport=cast(Transport, transport),
            presenter=MarkdownPresenter(),
            final_notify=True,
        ),
        topics=TelegramTopicsSettings(enabled=True, scope="projects"),
        files=TelegramFilesSettings(
            enabled=True, auto_put=True, auto_put_mode="prompt"
        ),
        pi_live_conversation=True,
        forward_coalesce_s=0.04 if mode == "forward" else 0,
    )

    async def poller(_cfg):
        if mode == "forward":
            yield TelegramIncomingMessage(
                transport="telegram",
                chat_id=-100,
                thread_id=77,
                message_id=1,
                text="Summarize this",
                reply_to_message_id=None,
                reply_to_text=None,
                sender_id=123,
            )
            yield TelegramIncomingMessage(
                transport="telegram",
                chat_id=-100,
                thread_id=77,
                message_id=2,
                text="Forwarded source content",
                reply_to_message_id=None,
                reply_to_text=None,
                sender_id=123,
                raw={"forward_origin": {"type": "user"}},
            )
        else:
            yield TelegramIncomingMessage(
                transport="telegram",
                chat_id=-100,
                thread_id=77,
                message_id=1,
                text="Review this file",
                reply_to_message_id=None,
                reply_to_text=None,
                sender_id=123,
                document=TelegramDocument(
                    file_id="doc-1",
                    file_name="hello.txt",
                    mime_type="text/plain",
                    file_size=5,
                    raw={"file_id": "doc-1"},
                ),
            )

    await run_main_loop(cfg, poller)
    assert len(runner.prompts) == 1
    assert (
        "Forwarded source content" if mode == "forward" else "[uploaded file:"
    ) in runner.prompts[0]
    assert (
        await LiveInbox(resolve_inbox_path(config)).initial_for_topic(-100, 77) is None
    )
