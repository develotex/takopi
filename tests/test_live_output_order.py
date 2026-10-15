from typing import cast

import anyio
import pytest

from takopi.model import CompletedEvent
from takopi.runners.pi_rpc import PiRpcRun
from takopi.telegram.live_conversation import (
    LiveConversationService,
    LiveOwner,
    LiveRunner,
)
from takopi.telegram.live_inbox import LiveInbox
from tests.test_telegram_live_integration import FakePi, FakeRun


@pytest.mark.anyio
@pytest.mark.parametrize("workers", [False, True])
@pytest.mark.parametrize("fail_send", [False, True])
@pytest.mark.parametrize("ending", ["correction", "normal_final", "unresolved"])
async def test_rpc_output_chronology_including_retry_and_failed_settlement(
    tmp_path, workers, fail_send, ending
):
    inbox = LiveInbox(tmp_path / "i.json")
    receipt = await inbox.receive(1, 10, 2, "a", "Actually use 20, not 10")
    delivery = await inbox.delivery_text(receipt.id)
    corrected = (
        f"[takopi-considered:{receipt.marker}] Corrected configuration: limit=20"
    )
    older = "Earlier result: limit=10"
    final = "Final configuration: limit=20"

    class RPC(FakeRun):
        def __init__(self):
            super().__init__()
            self.assistant_outputs = []
            self.output_generation = 0
            self.completed_answers = []

        async def run(self, prompt, resume):
            self.output_generation += 1
            texts = [older, corrected]
            if ending == "normal_final":
                texts.append(final)
            self.assistant_outputs = [(text, True) for text in texts]
            self.completed_answers = texts
            self.entries = (
                [{"role": "user", "content": delivery}]
                if ending != "unresolved"
                else []
            ) + [{"role": "assistant", "content": text} for text in texts]
            yield CompletedEvent(engine="pi", ok=True, answer=texts[-1])

    rpc = RPC()
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))
    sent = []
    broken = fail_send

    async def reply(chat, thread, message, text):
        if broken:
            raise ConnectionError("Telegram down")
        sent.append(text)

    svc = LiveConversationService(inbox, None, reply, reply_retry_interval=0.01)
    owner = LiveOwner(1, 10, "a", runner, "task")
    svc.register(owner)
    runner.before_final = lambda: svc.finalize(owner)
    runner.on_intermediate_answer = lambda text: svc.queue_output(1, 10, 1, "a", text)
    async with anyio.create_task_group() as tg:
        if workers:
            svc.attach_workers(tg)
        try:
            completed = [
                e
                async for e in runner.run("task", None)
                if isinstance(e, CompletedEvent)
            ]
        except ConnectionError:
            assert ending == "unresolved" and fail_send
            completed = []  # Even failure of the actionable notice must retain the result.
        broken = False
        await svc._send_replies()
        # Ignore actionable unsettled-receipt notices, but preserve all actual outputs.
        actual = [
            text
            for text in sent
            if text.startswith(
                ("Earlier result:", "Corrected configuration:", "Final configuration:")
            )
        ]
        expected = [older]
        if ending != "unresolved":
            expected.append("Corrected configuration: limit=20")
        if ending == "normal_final":
            expected.append(final)
        assert actual == expected
        if ending != "unresolved":
            assert completed[-1].answer == "", (
                "normal finals must not bypass durable output ordering"
            )
            assert runner.suppress_empty_final
        else:
            assert not completed or not completed[-1].ok
        svc.unregister(owner)
        svc.stop_workers()
        tg.cancel_scope.cancel()
