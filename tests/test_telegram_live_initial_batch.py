"""An incoming media item must never replace a persisted initial Pi prompt."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import anyio
import pytest

from takopi.config import ProjectConfig, ProjectsConfig
from takopi.context import RunContext
from takopi.markdown import MarkdownPresenter
from takopi.model import CompletedEvent, ResumeToken, StartedEvent, TakopiEvent
from takopi.router import AutoRouter, RunnerEntry
from takopi.runner_bridge import ExecBridgeConfig
from takopi.runners.pi import PiRunner
from takopi.runners.pi_rpc import PiRpcRun
from takopi.settings import TelegramFilesSettings, TelegramTopicsSettings
from takopi.telegram.api_models import File
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
@pytest.mark.parametrize(
    "media",
    [
        "voice",
        "album",
        "document_album",
        "single_document",
        "document_before_initial",
        "invalid_directive",
        "document_before_running",
        "album_flush_before_initial",
        "initial_reservation_race",
        "plugin_during_media",
    ],
)
async def test_media_in_forward_window_cannot_replace_initial_pi_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, media: str
) -> None:
    import takopi.telegram.loop as loop

    download_started, download_release = anyio.Event(), anyio.Event()
    plugin_calls: list[str] = []
    if media == "plugin_during_media":
        monkeypatch.setattr(loop, "list_command_ids", lambda **_: ["spy"])

        async def fake_dispatch(*_args, **_kwargs):
            plugin_calls.append("dispatched")

        monkeypatch.setattr(loop, "dispatch_command", fake_dispatch)
    if media == "initial_reservation_race":
        original_receive_initial = LiveInbox.receive_initial

        async def slow_receive_initial(self, *args):
            await anyio.sleep(0.06)
            return await original_receive_initial(self, *args)

        monkeypatch.setattr(LiveInbox, "receive_initial", slow_receive_initial)

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
            if media == "document_before_running":
                await anyio.sleep(0.15)
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

        async def run(
            self, prompt: str, resume: ResumeToken | None
        ) -> AsyncIterator[TakopiEvent]:
            if media in ("album_flush_before_initial", "plugin_during_media"):
                yield StartedEvent(
                    engine="pi",
                    resume=ResumeToken("pi", str(tmp_path / "legacy.jsonl")),
                )
                yield CompletedEvent(engine="pi", ok=True, answer="done")
                return
            raise AssertionError("second media prompt must not steal the initial task")
            yield  # pragma: no cover

        def rpc_run(self, session_path: Path, *, cwd: Path) -> PiRpcRun:
            assert cwd == tmp_path
            self.rpc = Rpc(session_path)
            return cast(PiRpcRun, self.rpc)

    if media == "voice":

        async def transcribe_voice(**_kwargs):
            raise AssertionError("rejected voice must not be transcribed")

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

    class UploadBot(FakeBot):
        async def get_file(self, file_id: str) -> File | None:
            return File(file_path="files/hello.txt")

        async def download_file(self, file_path: str) -> bytes | None:
            if media in ("album_flush_before_initial", "plugin_during_media"):
                download_started.set()
                await download_release.wait()
                return b"hello"
            raise AssertionError(
                "media must not upload while initial Pi prompt is reserved"
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
        forward_coalesce_s=0.05,
        media_group_debounce_s=0.08 if media == "document_before_running" else 0.02,
    )

    async def poller(_cfg):
        initial = TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=77,
            message_id=2
            if media
            in (
                "document_before_initial",
                "document_before_running",
                "album_flush_before_initial",
                "initial_reservation_race",
                "plugin_during_media",
            )
            else 1,
            text="Build original task",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )
        incoming_media = TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=77,
            message_id=1
            if media
            in (
                "document_before_initial",
                "document_before_running",
                "album_flush_before_initial",
                "initial_reservation_race",
                "plugin_during_media",
            )
            else 2,
            text="Album caption"
            if media
            in (
                "album",
                "document_album",
                "single_document",
                "album_flush_before_initial",
                "initial_reservation_race",
                "plugin_during_media",
            )
            else "@one @two Replace original"
            if media == "invalid_directive"
            else "",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
            voice=TelegramVoice("v", "audio/ogg", 100, 1, {})
            if media == "voice"
            else None,
            document=TelegramDocument("d", "hello.txt", "text/plain", 100, {})
            if media
            in (
                "document_album",
                "single_document",
                "document_before_initial",
                "document_before_running",
                "album_flush_before_initial",
                "initial_reservation_race",
                "plugin_during_media",
            )
            else None,
            media_group_id="album-1"
            if media
            in (
                "album",
                "document_album",
                "document_before_initial",
                "document_before_running",
                "album_flush_before_initial",
                "initial_reservation_race",
                "plugin_during_media",
            )
            else None,
        )
        if media in (
            "document_before_initial",
            "document_before_running",
            "album_flush_before_initial",
            "initial_reservation_race",
            "plugin_during_media",
        ):
            yield incoming_media
            if media in ("album_flush_before_initial", "plugin_during_media"):
                with anyio.fail_after(2):
                    await download_started.wait()
            if media == "plugin_during_media":
                yield TelegramIncomingMessage(
                    transport="telegram",
                    chat_id=-100,
                    thread_id=77,
                    message_id=2,
                    text="/spy change",
                    reply_to_message_id=None,
                    reply_to_text=None,
                    sender_id=123,
                )
            else:
                yield initial
            if media in ("album_flush_before_initial", "plugin_during_media"):
                await anyio.sleep(0.08)
                assert runner.rpc is None, (
                    "initial Pi run overtook an active album upload"
                )
                download_release.set()
        else:
            yield initial
            yield incoming_media

    await run_main_loop(cfg, poller)
    if media in ("album_flush_before_initial", "plugin_during_media"):
        assert not plugin_calls, "plugin must not run during album processing"
        assert runner.rpc is None, (
            "Pi must not start concurrently with accepted album upload"
        )
        assert any(
            "not accepted" in call["message"].text.lower()
            for call in transport.send_calls
        )
        return
    assert runner.rpc is not None
    assert runner.rpc.prompts[0].endswith("Build original task")
    inbox = LiveInbox(resolve_inbox_path(config))
    assert await inbox.for_session(str(runner.rpc.session_path)) == []
    assert any(
        "not accepted" in call["message"].text.lower() for call in transport.send_calls
    )
