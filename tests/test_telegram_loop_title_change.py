import asyncio
from typing import cast

import anyio
import pytest

from takopi.model import CompletedEvent, ResumeToken, StartedEvent, TitleChangedEvent
from takopi.runner import Runner
from takopi.runner_bridge import ProgressEdits, RunningTask, run_runner_with_cancel
from takopi.runners.mock import Emit, Return, ScriptRunner


class _Edits:
    async def on_event(self, _event) -> None:
        pass


@pytest.mark.anyio
async def test_cancelled_completion_waits_for_interrupt_and_never_reports_success() -> (
    None
):
    settled, interrupt_entered, allow_abort = (
        anyio.Event(),
        anyio.Event(),
        anyio.Event(),
    )
    token = ResumeToken(engine="pi", value="test-session")

    class Control:
        aborted = False

        async def interrupt(self):
            interrupt_entered.set()
            await allow_abort.wait()
            self.aborted = True
            return True

    control = Control()

    class FakeRunner:
        async def run(self, _prompt, _resume):
            yield StartedEvent(engine="pi", resume=token, meta={"control": control})
            await settled.wait()
            yield CompletedEvent(engine="pi", ok=True, answer="settled", resume=token)

    running = RunningTask()
    run = asyncio.create_task(
        run_runner_with_cancel(
            cast(Runner, FakeRunner()),
            prompt="task",
            resume_token=None,
            edits=cast(ProgressEdits, _Edits()),
            running_task=running,
            on_thread_known=None,
        )
    )
    try:
        with anyio.fail_after(1):
            await running.resume_ready.wait()
        running.cancel_requested.set()
        with anyio.fail_after(1):
            await interrupt_entered.wait()
        settled.set()
        await anyio.sleep(0.03)
        assert not run.done(), "Observed cancel cannot discard an in-flight abort"
    finally:
        allow_abort.set()
    outcome = await asyncio.wait_for(run, 1)
    assert control.aborted
    assert outcome.cancelled


def test_title_changed_event() -> None:
    event = TitleChangedEvent(engine="pi", title="New Title")
    assert event.type == "title_changed"
    assert event.title == "New Title"


@pytest.mark.anyio
async def test_prompt_fallback_title_is_only_generated_for_new_session() -> None:
    token = ResumeToken(engine="pi", value="existing-session")
    runner = ScriptRunner(
        [Return("ok")],
        engine="pi",
        resume_value=token.value,
    )
    task = RunningTask()
    callback_titles: list[str | None] = []

    async def on_thread_known(_token: ResumeToken, _done: anyio.Event) -> None:
        callback_titles.append(task.title)

    await run_runner_with_cancel(
        runner,
        prompt="This must not become a new topic title",
        resume_token=token,
        edits=_Edits(),
        running_task=task,
        on_thread_known=on_thread_known,
    )

    assert task.title is None
    assert callback_titles == [None]


@pytest.mark.anyio
async def test_pi_title_can_still_rename_resumed_session() -> None:
    token = ResumeToken(engine="pi", value="existing-session")
    runner = ScriptRunner(
        [
            Emit(TitleChangedEvent(engine="pi", title="Stable session title")),
            Return("ok"),
        ],
        engine="pi",
        resume_value=token.value,
    )
    task = RunningTask()
    callback_titles: list[str | None] = []

    async def on_thread_known(_token: ResumeToken, _done: anyio.Event) -> None:
        callback_titles.append(task.title)

    await run_runner_with_cancel(
        runner,
        prompt="A later message",
        resume_token=token,
        edits=_Edits(),
        running_task=task,
        on_thread_known=on_thread_known,
    )

    assert task.title == "Stable session title"
    assert callback_titles == [None, "Stable session title"]
