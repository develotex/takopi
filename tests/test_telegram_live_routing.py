from __future__ import annotations

from pathlib import Path

import pytest

from takopi.model import ResumeToken
from takopi.settings import TelegramTransportSettings
from takopi.telegram.live_conversation import live_session_path, live_route_key


@pytest.mark.anyio
async def test_scheduler_identifies_queued_job_in_exact_topic() -> None:
    from takopi.scheduler import ThreadScheduler, ThreadJob

    class TaskGroup:
        def start_soon(self, *_args):
            pass

    async def run(_job):
        pass

    scheduler = ThreadScheduler(task_group=TaskGroup(), run_job=run)
    token = ResumeToken(engine="pi", value="/session.jsonl")
    await scheduler.enqueue(
        ThreadJob(
            chat_id=1, user_msg_id=4, text="next", resume_token=token, thread_id=10
        )
    )
    assert await scheduler.has_pending_for_topic(token, 1, 10)
    assert not await scheduler.has_pending_for_topic(token, 1, 11)


def test_live_opt_in_defaults_off_and_exact_topic_paths(tmp_path: Path) -> None:
    cfg = TelegramTransportSettings(bot_token="test", chat_id=1)
    assert not cfg.pi_live_conversation
    cfg = TelegramTransportSettings(
        bot_token="test", chat_id=1, pi_live_conversation=True
    )
    assert cfg.pi_live_conversation
    a = live_session_path(tmp_path, 1, 10)
    b = live_session_path(tmp_path, 1, 11)
    assert a != b and a.is_absolute() and a.suffix == ".jsonl"
    assert live_route_key(1, 10, ResumeToken(engine="pi", value=str(a))) == (
        1,
        10,
        str(a),
    )
    assert live_route_key(1, 10, ResumeToken(engine="pi", value="short-id")) is None
    assert live_route_key(1, 10, ResumeToken(engine="codex", value=str(a))) is None
    assert live_route_key(1, None, ResumeToken(engine="pi", value=str(a))) is None


@pytest.mark.anyio
async def test_quick_responder_uses_ephemeral_no_tool_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from takopi.telegram import live_conversation as module

    calls: list[tuple[Path, list[str], str]] = []

    class FakeClient:
        def __init__(self, session_path: Path, cwd: Path, args: list[str]) -> None:
            calls.append((session_path, args, str(cwd)))

        async def close(self) -> None:
            pass

    class FakeRun:
        def __init__(self, client: FakeClient) -> None:
            pass

        async def run(self, prompt: str, resume: None):
            from takopi.model import CompletedEvent

            yield CompletedEvent(engine="pi", answer="safe", ok=True)

    monkeypatch.setattr(module, "PiRpcClient", FakeClient)
    monkeypatch.setattr(module, "PiRpcRun", FakeRun)
    assert await module.quick_pi_answer("status?", "Task: demo") == "safe"
    assert "--no-tools" in calls[0][1]
    assert "--no-extensions" in calls[0][1]
    assert "--no-context-files" in calls[0][1]
    assert calls[0][0].parent == Path(calls[0][2])
