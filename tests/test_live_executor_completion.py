"""Live output must survive the real executor's presentation wrappers."""

from typing import cast

import pytest

from takopi.markdown import MarkdownPresenter
from takopi.model import CompletedEvent, ResumeToken, StartedEvent
from takopi.runner_bridge import ExecBridgeConfig
from takopi.runners.pi_rpc import PiRpcRun
from takopi.runners.run_options import EngineRunOptions
from takopi.telegram.bridge import send_plain
from takopi.telegram.commands.executor import _run_engine
from takopi.telegram.live_conversation import (
    LiveConversationService,
    LiveOwner,
    LiveRunner,
)
from takopi.telegram.live_inbox import LiveInbox
from takopi.transport import Transport
from takopi.transport_runtime import TransportRuntime
from tests.telegram_fakes import FakeTransport, _empty_projects, _make_router
from tests.test_live_history_and_segments import BufferedRun
from tests.test_telegram_live_integration import FakePi


@pytest.mark.anyio
@pytest.mark.parametrize("show_resume", [False, True])
@pytest.mark.parametrize("warning", [False, True])
@pytest.mark.parametrize("progress", ["none", "once"])
@pytest.mark.parametrize("ending", ["published", "empty", "error"])
async def test_live_final_through_production_executor(
    tmp_path, monkeypatch, show_resume, warning, progress, ending
):
    class RPC(BufferedRun):
        async def run(self, prompt, resume):
            token = ResumeToken("pi", "a")
            yield StartedEvent(engine="pi", resume=token)
            if ending == "published":
                answer, ok, error = "Да, тут.", True, None
                self.assistant_outputs = [(answer, True)]
                self.entries = [{"role": "assistant", "content": answer}]
            elif ending == "error":
                answer, ok, error = "", False, "Real provider failure"
            else:
                answer, ok, error = "", True, None
            yield CompletedEvent(
                engine="pi", answer=answer, ok=ok, error=error, resume=token
            )

    rpc = RPC()
    main = LiveRunner(FakePi(), cast(PiRpcRun, rpc))
    transport = FakeTransport()

    async def reply(chat, thread, message, text):
        await send_plain(
            cast(Transport, transport),
            chat_id=chat,
            thread_id=thread,
            user_msg_id=message,
            text=text,
            require_success=True,
        )

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), None, reply)
    owner = LiveOwner(1, 10, "a", main, "task")
    main.before_final = lambda: svc.finalize(owner)
    main.on_intermediate_answer = lambda text: svc.queue_output(1, 10, 1, "a", text)
    monkeypatch.setattr(
        "takopi.telegram.commands.executor.supports_reasoning", lambda _: not warning
    )
    await _run_engine(
        exec_cfg=ExecBridgeConfig(
            transport=cast(Transport, transport),
            presenter=MarkdownPresenter(),
            progress_updates=progress,
            final_notify=True,
        ),
        runtime=TransportRuntime(router=_make_router(main), projects=_empty_projects()),
        running_tasks={},
        chat_id=1,
        user_msg_id=1,
        text="ты тут?",
        resume_token=None,
        context=None,
        reply_ref=None,
        on_thread_known=None,
        engine_override=None,
        thread_id=10,
        show_resume_line=show_resume,
        run_options=EngineRunOptions(reasoning="high") if warning else None,
        runner_override=main,
    )
    rendered = [c["message"].text for c in transport.send_calls + transport.edit_calls]
    if ending == "published":
        assert main.suppress_empty_final
        assert sum("Да, тут." in text for text in rendered) == 1
        assert not any("error" in text.lower() for text in rendered)
        if progress == "once":
            assert transport.delete_calls, "initial progress should be cleaned up"
    else:
        assert not main.suppress_empty_final
        assert any("error" in text.lower() for text in rendered)
        if ending == "error":
            assert any("Real provider failure" in text for text in rendered)
