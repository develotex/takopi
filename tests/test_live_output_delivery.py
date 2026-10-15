from typing import cast
from takopi.runners.pi_rpc import PiRpcRun
import pytest

from takopi.telegram.live_conversation import LiveConversationService, LiveRunner
from takopi.telegram.live_inbox import LiveInbox
from takopi.model import CompletedEvent, StartedEvent, ResumeToken
from tests.test_telegram_live_conversation import no_surrogate
from tests.test_telegram_live_integration import FakeRun, FakePi


@pytest.mark.anyio
async def test_settlement_publishes_original_result_before_question_not_as_stale_final(
    tmp_path,
):
    marker = "b" * 32

    class RPC(FakeRun):
        async def run(self, prompt, resume):
            self.sent.append(prompt)
            yield StartedEvent(engine="pi", resume=ResumeToken("pi", "a"))
            yield CompletedEvent(
                engine="pi",
                ok=True,
                answer=(
                    "Original task deliverable"
                    if len(self.sent) == 1
                    else f"[takopi-considered:{marker}] Actual question answer"
                ),
            )

    rpc = RPC()
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))
    replies = []

    async def reply(*args):
        replies.append(args)

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), no_surrogate, reply)

    async def intermediate(text):
        await svc.queue_output(1, 10, 1, "a", text)

    runner.on_intermediate_answer = intermediate
    runner.answered_markers.add(marker)
    runner._followups.append("question")
    emitted = [
        e async for e in runner.run("task", None) if isinstance(e, CompletedEvent)
    ]
    assert replies == [(1, 10, 1, "Original task deliverable")]
    assert [e.answer for e in emitted] == [""]
    assert runner.suppress_empty_final


@pytest.mark.anyio
async def test_two_completed_turns_in_one_rpc_run_preserve_main_deliverable():
    marker = "c" * 32

    class RPC(FakeRun):
        async def run(self, prompt, resume):
            self.completed_answers = [
                "Original deliverable",
                f"[takopi-considered:{marker}] The real answer",
            ]
            yield CompletedEvent(
                engine="pi", ok=True, answer=self.completed_answers[-1]
            )

    runner = LiveRunner(FakePi(), cast(PiRpcRun, RPC()))
    captured = []

    async def output(text):
        captured.append(text)

    runner.on_intermediate_answer = output
    runner.answered_markers.add(marker)
    final = [e async for e in runner.run("task", None) if isinstance(e, CompletedEvent)]
    assert captured == ["Original deliverable"]
    assert final[-1].answer == ""


@pytest.mark.anyio
async def test_genuinely_missing_final_is_not_suppressed():
    class RPC(FakeRun):
        async def run(self, prompt, resume):
            yield CompletedEvent(engine="pi", ok=True, answer="")

    runner = LiveRunner(FakePi(), cast(PiRpcRun, RPC()))
    _ = [e async for e in runner.run("task", None)]
    assert not runner.suppress_empty_final


@pytest.mark.anyio
async def test_long_durable_output_resumes_at_failed_chunk_after_restart(tmp_path):
    inbox = LiveInbox(tmp_path / "i.json")
    sent = []

    async def reply(*args):
        if sent:
            raise ConnectionError("second chunk failed")
        sent.append(args[-1])

    svc = LiveConversationService(inbox, no_surrogate, reply)
    text = "x" * 2000 + "😀" * 2001
    await svc.queue_output(1, 10, 1, "a", text)
    assert sent == ["x" * 2000]

    async def recovered(*args):
        sent.append(args[-1])

    restarted = LiveConversationService(
        LiveInbox(tmp_path / "i.json"), no_surrogate, recovered
    )
    await restarted.notify_recovery()
    assert "".join(sent) == text
    assert all(len(chunk.encode("utf-16-le")) // 2 <= 4096 for chunk in sent)
    assert not await inbox.pending_outputs()
