"""An incoming media item must never replace a persisted initial Pi prompt."""

import json
from pathlib import Path
from typing import cast

import pytest

from takopi.config import ProjectConfig, ProjectsConfig
from takopi.context import RunContext
from takopi.markdown import MarkdownPresenter
from takopi.model import CompletedEvent, ResumeToken, StartedEvent
from takopi.router import AutoRouter, RunnerEntry
from takopi.runner_bridge import ExecBridgeConfig
from takopi.runners.pi import PiRunner
from takopi.runners.pi_rpc import PiRpcRun
from takopi.settings import TelegramFilesSettings, TelegramTopicsSettings
from takopi.telegram.bridge import TelegramBridgeConfig, run_main_loop
from takopi.telegram.live_inbox import LiveInbox, resolve_inbox_path
from takopi.telegram.topic_state import TopicStateStore, resolve_state_path
from takopi.telegram.types import (
    TelegramDocument,
    TelegramIncomingMessage,
    TelegramVoice,
)
from takopi.transport import Transport
from takopi.transport_runtime import TransportRuntime
from tests.telegram_fakes import FakeBot, FakeTransport


@pytest.mark.anyio
@pytest.mark.parametrize("media", ["voice", "album", "document_album"])
async def test_media_in_forward_window_cannot_replace_initial_pi_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, media: str
) -> None:
    import takopi.telegram.loop as loop

    class Rpc:
        def __init__(self, path: Path) -> None:
            self.client = self
            self.session_path = path
            self.cwd = tmp_path
            self._active = False
            self.prompts: list[str] = []
            self.entries: list[dict] = []

        async def request(self, typ: str, **_kwargs):
            if typ == "get_state":
                return {
                    "data": {
                        "sessionId": "fixture",
                        "sessionFile": str(self.session_path),
                        "isStreaming": False,
                    }
                }
            if typ == "get_messages":
                return {"data": {"messages": self.entries}}
            raise AssertionError(typ)

        async def run(self, prompt: str, _resume):
            self._active = True
            self.prompts.append(prompt)
            self.session_path.parent.mkdir(exist_ok=True)
            if not self.session_path.exists():
                self.session_path.write_text(
                    json.dumps(
                        {"type": "session", "id": "fixture", "cwd": str(tmp_path)}
                    )
                    + "\n"
                )
            self.entries.append({"role": "user", "content": prompt})
            yield StartedEvent(
                engine="pi", resume=ResumeToken("pi", str(self.session_path))
            )
            self._active = False
            yield CompletedEvent(
                engine="pi",
                ok=True,
                answer="done",
                resume=ResumeToken("pi", str(self.session_path)),
            )

        async def close(self) -> None:
            pass

    class Runner(PiRunner):
        def __init__(self) -> None:
            super().__init__(extra_args=[], model=None, provider=None)
            self.rpc: Rpc | None = None

        async def run(self, _prompt: str, _resume):
            raise AssertionError("second media prompt must not steal the initial task")
            yield  # pragma: no cover

        def rpc_run(self, session_path: Path, *, cwd: Path) -> PiRpcRun:
            assert cwd == tmp_path
            self.rpc = Rpc(session_path)
            return cast(PiRpcRun, self.rpc)

    if media == "voice":

        async def transcribe_voice(**_kwargs):
            return "Use green"

        monkeypatch.setattr(loop, "transcribe_voice", transcribe_voice)
    config = tmp_path / "takopi.toml"
    runner = Runner()
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
    await TopicStateStore(resolve_state_path(config)).set_context(
        -100, 77, RunContext(project="test")
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
        files=TelegramFilesSettings(
            enabled=True, auto_put=True, auto_put_mode="prompt"
        ),
        pi_live_conversation=True,
        forward_coalesce_s=0.05,
    )

    async def poller(_cfg):
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=77,
            message_id=1,
            text="Build original task",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=77,
            message_id=2,
            text="Album caption" if media == "album" else "",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
            voice=TelegramVoice("v", "audio/ogg", 100, 1, {})
            if media == "voice"
            else None,
            document=TelegramDocument("d", "hello.txt", "text/plain", 100, {})
            if media == "document_album"
            else None,
            media_group_id="album-1" if media != "voice" else None,
        )

    await run_main_loop(cfg, poller)
    assert runner.rpc is not None
    assert runner.rpc.prompts[0].endswith("Build original task")
    inbox = LiveInbox(resolve_inbox_path(config))
    assert await inbox.for_session(str(runner.rpc.session_path)) == []
    assert any(
        "not accepted" in call["message"].text.lower() for call in transport.send_calls
    )
