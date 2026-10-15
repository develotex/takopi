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
from takopi.runner_bridge import ExecBridgeConfig, RunningTask
from takopi.runners.pi import PiRunner
from takopi.runners.pi_rpc import PiRpcRun
from takopi.settings import TelegramTopicsSettings
from takopi.telegram.bridge import TelegramBridgeConfig, run_main_loop
from takopi.telegram.live_conversation import LiveConversationService
from takopi.telegram.live_inbox import LiveInbox, resolve_inbox_path
from takopi.telegram.topic_state import TopicStateStore, resolve_state_path
from takopi.telegram.types import TelegramIncomingMessage
from takopi.transport import MessageRef, Transport
from takopi.transport_runtime import TransportRuntime
from tests.telegram_fakes import FakeBot, FakeTransport


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case",
    [
        "before_file",
        "after_file_unobserved",
        "after_file_observed",
        "wrong_topic",
        "owner_conflict",
        "other_bound",
        "other_bound_short",
        "header_mismatch",
        "header_absent",
        "bound_bypass",
        "cancel_before_file",
        "cancel_after_file",
        "cancel_root_reply",
        "cancel_original_reply",
        "cancel_verify_fresh_original",
        "cancel_ambiguous",
    ],
)
async def test_explicit_initial_recovery_preserves_original_prompt_and_dependent_fifo(
    tmp_path: Path,
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "takopi.toml"
    entered, release = anyio.Event(), anyio.Event()
    other_task = RunningTask(
        context=RunContext(project="test"), thread_id=77, user_message_id=999
    )
    if case == "cancel_ambiguous":
        import takopi.telegram.loop as telegram_loop

        original_state = telegram_loop.TelegramLoopState

        def capture_state(*args, **kwargs):
            state = original_state(*args, **kwargs)
            state.running_tasks[
                MessageRef(channel_id=-100, message_id=999, thread_id=77)
            ] = other_task
            return state

        monkeypatch.setattr(telegram_loop, "TelegramLoopState", capture_state)
    path = tmp_path / "pi-live-sessions" / f"-100-77-{uuid4().hex}.jsonl"
    ident = uuid4().hex
    inbox = LiveInbox(resolve_inbox_path(config))
    initial = await inbox.receive_initial(
        -100, 77, 1, str(path), "Build original task", str(tmp_path)
    )
    await inbox.mark_initial_uncertain(initial.id, initial.session_key)
    if case == "cancel_before_file":
        original_inspect = LiveConversationService.inspect_initial_retry

        async def delayed_inspect(self, *args):
            entered.set()
            await release.wait()
            return await original_inspect(self, *args)

        monkeypatch.setattr(
            LiveConversationService, "inspect_initial_retry", delayed_inspect
        )
    update = (
        None
        if case == "bound_bypass"
        else await inbox.receive(-100, 77, 2, str(path), "Use green")
    )
    if case in (
        "after_file_unobserved",
        "after_file_observed",
        "owner_conflict",
        "other_bound",
        "other_bound_short",
        "bound_bypass",
        "cancel_after_file",
        "cancel_root_reply",
        "cancel_original_reply",
        "cancel_ambiguous",
    ):
        path.parent.mkdir()
        path.write_text(
            json.dumps({"type": "session", "id": ident, "cwd": str(tmp_path)}) + "\n"
        )

    class Rpc:
        def __init__(self):
            self.client = self
            self.session_path = path
            self.cwd = tmp_path
            self._active = False
            self.prompts: list[str] = []
            self.calls: list[str] = []
            self.entries: list[dict] = (
                [{"role": "user", "content": "Build original task"}]
                if case == "after_file_observed"
                else []
            )

        async def request(self, typ):
            self.calls.append(typ)
            if typ == "get_state":
                if case.startswith("cancel_") and case != "cancel_before_file":
                    entered.set()
                    await release.wait()
                if case in ("before_file", "header_mismatch", "header_absent"):
                    assert not path.exists(), "new Pi JSONL is absent during get_state"
                if case == "owner_conflict":
                    raise RuntimeError("canonical owner conflict")
                return {
                    "data": {
                        "sessionId": ident,
                        "sessionFile": str(path),
                        "isStreaming": False,
                    }
                }
            if typ == "get_messages":
                return {"data": {"messages": self.entries}}
            raise AssertionError(typ)

        async def run(self, prompt, resume):
            self.prompts.append(prompt)
            self._active = True
            token = ResumeToken(engine="pi", value=str(path))
            yield StartedEvent(engine="pi", resume=token)
            if not path.exists() and case != "header_absent":
                path.parent.mkdir(exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "type": "session",
                            "id": "wrong" if case == "header_mismatch" else ident,
                            "cwd": str(tmp_path),
                        }
                    )
                    + "\n"
                )
            self.entries.append({"role": "user", "content": prompt})
            if update is not None and prompt == await inbox.delivery_text(update.id):
                self.entries.append(
                    {
                        "role": "assistant",
                        "content": f"[takopi-considered:{update.marker}] Applied green.",
                    }
                )
            self._active = False
            yield CompletedEvent(engine="pi", ok=True, answer="Done", resume=token)

        async def steer(self, _text):
            raise AssertionError(
                "Dependent receipt must wait until initial prompt settles"
            )

        async def close(self):
            pass

    rpc = Rpc()

    class Runner(PiRunner):
        async def run(self, prompt, resume):
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
    if case == "wrong_topic":
        await store.set_context(-100, 78, RunContext(project="test"))
    if case in ("other_bound", "other_bound_short"):
        await store.set_session_resume(
            -100,
            78,
            ResumeToken(
                engine="pi", value=str(path) if case == "other_bound" else ident[:12]
            ),
        )
    if case == "bound_bypass":
        await store.set_session_resume(
            -100, 77, ResumeToken(engine="pi", value=str(path))
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
            thread_id=78 if case == "wrong_topic" else 77,
            message_id=50,
            text="Another task"
            if case == "bound_bypass"
            else "/update retry-initial 1 confirm",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )
        if case.startswith("cancel_"):
            with anyio.fail_after(2):
                await entered.wait()
            yield TelegramIncomingMessage(
                transport="telegram",
                chat_id=-100,
                thread_id=77,
                message_id=51,
                text="/cancel",
                reply_to_message_id=(
                    77
                    if case == "cancel_root_reply"
                    else 1
                    if case in ("cancel_original_reply", "cancel_verify_fresh_original")
                    else None
                ),
                reply_to_text=None,
                sender_id=123,
            )
            await anyio.sleep(0.05)
            release.set()

    await run_main_loop(cfg, poller)
    current = await inbox.initial_by_id(initial.id)
    assert current is not None
    if case == "cancel_ambiguous":
        assert rpc.prompts, "Ambiguous bare /cancel must not select the Pi retry"
        assert not other_task.cancel_requested.is_set()
        assert any(
            "multiple runs are active" in item["message"].text
            for item in transport.send_calls
        )
        return
    if case.startswith("cancel_"):
        assert not rpc.prompts, "Cancelling inspection must not submit a retry prompt"
        assert current.state == "uncertain"
        return
    if case in ("header_mismatch", "header_absent"):
        assert current.state == "uncertain"
        assert update is not None
        assert (await inbox.get(update.id)).state == "uncertain"
        assert len(rpc.prompts) == 1
        assert await store.get_session_resume(-100, 77, "pi") is None
        assert any(
            "header" in item["message"].text.lower() for item in transport.send_calls
        )
        return
    if case in (
        "wrong_topic",
        "owner_conflict",
        "other_bound",
        "other_bound_short",
        "bound_bypass",
    ):
        assert current.state == "uncertain"
        assert not rpc.prompts
        if case in ("wrong_topic", "other_bound", "other_bound_short", "bound_bypass"):
            assert not rpc.calls
        else:
            assert "get_state" in rpc.calls
        return
    assert current.state == "completed"
    assert update is not None
    assert (await inbox.get(update.id)).state == "considered"
    assert len(rpc.prompts) == 2
    if case == "after_file_observed":
        assert rpc.prompts[0].startswith("Continue the original task")
        assert rpc.prompts[0] != initial.prompt
    else:
        assert rpc.prompts[0].endswith(initial.prompt)
    assert rpc.prompts[1] == await inbox.delivery_text(update.id)
    if case == "before_file":
        assert rpc.calls[0] == "get_state", (
            "provisional canonical path must be verified before prompt"
        )
        assert await store.get_session_resume(-100, 77, "pi") == ResumeToken(
            engine="pi", value=str(path)
        )
    else:
        assert rpc.calls[:2] == ["get_state", "get_messages"]
