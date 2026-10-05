from __future__ import annotations

import pytest

from takopi.telegram.live_conversation import (
    LiveConversationService,
    LiveOwner,
    LiveRunner,
)
from takopi.telegram.live_inbox import LiveInbox, Receipt
from pathlib import Path
from typing import cast

import anyio
from takopi.config import ProjectConfig, ProjectsConfig
from takopi.context import RunContext
from takopi.markdown import MarkdownPresenter
from takopi.model import ResumeToken, StartedEvent
from takopi.router import AutoRouter, RunnerEntry
from takopi.runner_bridge import ExecBridgeConfig
from takopi.runners.pi import PiRunner
from takopi.settings import TelegramTopicsSettings
from takopi.telegram.bridge import TelegramBridgeConfig, run_main_loop
from takopi.telegram.live_inbox import resolve_inbox_path
from takopi.telegram.topic_state import TopicStateStore, resolve_state_path
from takopi.telegram.types import TelegramIncomingMessage
from takopi.transport_runtime import TransportRuntime
from takopi.transport import Transport
from tests.telegram_fakes import FakeBot, FakeTransport
from takopi.model import CompletedEvent
from takopi.runners.pi_rpc import PiRpcRun


class FakePi:
    engine = "pi"

    def format_resume(self, token):
        return token.value

    def is_resume_line(self, line):
        return False

    def extract_resume(self, text):
        return None


class FakeRun:
    def __init__(self):
        self.client = self
        self._active = True
        self.sent: list[str] = []
        self.entries: list[dict] = []

    async def steer(self, text):
        self.sent.append(text)
        return "queued"

    async def request(self, typ):
        assert typ == "get_messages"
        return {"data": {"messages": self.entries}}

    async def run(self, prompt, resume):
        from takopi.model import CompletedEvent

        self.sent.append(prompt)
        yield CompletedEvent(engine="pi", ok=True, answer="done")


@pytest.mark.anyio
async def test_finalizing_owner_still_excludes_another_session_in_same_topic(
    tmp_path: Path,
):
    async def answer(*_):
        return "answer"

    async def reply(*_):
        return None

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    first = LiveOwner(
        1,
        10,
        str(tmp_path / "first.jsonl"),
        LiveRunner(FakePi(), cast(PiRpcRun, FakeRun())),
        "task",
    )
    second = LiveOwner(
        1,
        10,
        str(tmp_path / "second.jsonl"),
        LiveRunner(FakePi(), cast(PiRpcRun, FakeRun())),
        "task",
    )
    assert svc.register(first)
    first.closing = True
    assert svc.has_topic_owner(1, 10)
    assert not svc.register(second)
    svc.unregister(first)
    assert svc.register(second)
    svc.unregister(second)


@pytest.mark.anyio
async def test_adapter_uses_same_rpc_owner_and_observed_messages(tmp_path: Path):
    rpc = FakeRun()
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))

    async def answer(*_):
        return "answer"

    async def reply(*_):
        return None

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    owner = LiveOwner(1, 10, str(tmp_path / "s.jsonl"), runner, "task")
    svc.register(owner)
    assert await svc.handle(1, 10, 33, owner.session_key, "/update use green")
    assert rpc.sent[0].endswith("use green")
    receipt = (await svc.inbox.pending(owner.session_key))[0]
    assert receipt.state == "submitted"
    rpc.entries = [
        {"role": "user", "content": await svc.inbox.delivery_text(receipt.id)}
    ]
    await svc.reconcile(owner)
    assert (await svc.inbox.get(receipt.id)).state == "delivered"


@pytest.mark.anyio
async def test_fresh_pi_session_identity_is_verified_before_initial_prompt(
    tmp_path: Path,
):
    import json
    from takopi.telegram.live_conversation import (
        verify_fresh_session,
        verify_provisional_header,
    )

    path = tmp_path / "pi-live-sessions" / "1-10-test.jsonl"
    path.parent.mkdir()

    class Inspect:
        def __init__(self, actual: Path):
            self.client = self
            self.session_path = path
            self.cwd = tmp_path
            self.actual = actual
            self.calls = []

        async def request(self, typ):
            self.calls.append(typ)
            assert typ == "get_state"
            assert not path.exists(), "Pi has not written a new header before prompt"
            return {
                "data": {
                    "sessionId": "abc",
                    "sessionFile": str(self.actual),
                    "isStreaming": False,
                }
            }

    ok = Inspect(path)
    assert await verify_fresh_session(cast(PiRpcRun, ok), path, tmp_path) == "abc"
    assert ok.calls == ["get_state"]
    assert not path.exists()
    with pytest.raises(ValueError):
        verify_provisional_header(path, "abc", tmp_path)
    path.write_text(
        json.dumps({"type": "session", "id": "abc", "cwd": str(tmp_path)}) + "\n"
    )
    verify_provisional_header(path, "abc", tmp_path)
    with pytest.raises(ValueError):
        verify_provisional_header(path, "wrong", tmp_path)
    path.unlink()
    wrong = Inspect(tmp_path / "different.jsonl")
    with pytest.raises(ValueError):
        await verify_fresh_session(cast(PiRpcRun, wrong), path, tmp_path)
    assert wrong.calls == ["get_state"]


@pytest.mark.anyio
async def test_initial_intent_retry_inspects_missing_and_existing_session(
    tmp_path: Path,
):
    import json

    root = tmp_path / "config"
    root.mkdir()
    directory = root / "pi-live-sessions"
    directory.mkdir()
    path = directory / "1-10-test.jsonl"
    inbox = LiveInbox(tmp_path / "i.json")
    intent = await inbox.receive_initial(
        1, 10, 1, str(path), "Build task", str(tmp_path)
    )

    async def answer(*_):
        return "answer"

    async def reply(*_):
        return None

    svc = LiveConversationService(inbox, answer, reply)
    assert await svc.inspect_initial_retry(intent, None, tmp_path, root) == "fresh"
    path.write_text(
        json.dumps({"type": "session", "id": "abc", "cwd": str(tmp_path)}) + "\n"
    )

    class Inspect:
        def __init__(self):
            self.client = self
            self.session_path = path
            self.entries = []

        async def request(self, typ):
            if typ == "get_state":
                return {
                    "data": {
                        "sessionId": "abc",
                        "sessionFile": str(path),
                        "isStreaming": False,
                    }
                }
            return {"data": {"messages": self.entries}}

    rpc = Inspect()
    assert (
        await svc.inspect_initial_retry(intent, cast(PiRpcRun, rpc), tmp_path, root)
        == "retry"
    )
    rpc.entries = [{"role": "user", "content": "Build task"}]
    assert (
        await svc.inspect_initial_retry(intent, cast(PiRpcRun, rpc), tmp_path, root)
        == "continue"
    )
    with pytest.raises(ValueError):
        await svc.inspect_initial_retry(
            intent, cast(PiRpcRun, rpc), tmp_path / "other", root
        )


@pytest.mark.anyio
async def test_idle_retry_reconciles_canonical_owner_before_explicit_replay(
    tmp_path: Path,
):
    import json

    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps({"type": "session", "id": "abc-def", "cwd": str(tmp_path)}) + "\n"
    )
    inbox = LiveInbox(tmp_path / "i.json")
    receipt = await inbox.receive(1, 10, 33, str(path), "Use green")
    await inbox.mark_uncertain(receipt.id, "RPC timeout")

    class Inspect:
        def __init__(self, *, observed: bool = False):
            self.observed = observed
            self.calls: list[str] = []
            self.client = self
            self.session_path = path

        async def request(self, typ):
            self.calls.append(typ)
            if typ == "get_state":
                return {
                    "data": {
                        "sessionId": "abc-def",
                        "sessionFile": str(path),
                        "isStreaming": False,
                    }
                }
            if typ == "get_messages":
                return {
                    "data": {
                        "messages": (
                            [
                                {
                                    "role": "user",
                                    "content": await inbox.delivery_text(receipt.id),
                                }
                            ]
                            if self.observed
                            else []
                        )
                    }
                }
            raise AssertionError("no prompt without a confirmed retry")

    async def answer(*_):
        return "answer"

    async def reply(*_):
        return None

    svc = LiveConversationService(inbox, answer, reply)
    observed = Inspect(observed=True)
    assert not await svc.begin_idle_retry(receipt, cast(PiRpcRun, observed), tmp_path)
    assert observed.calls == ["get_state", "get_messages"]
    assert (await inbox.get(receipt.id)).state == "delivered"
    await inbox.mark_deferred(receipt.id, "explicit user decision after delivery")
    other = await inbox.receive(1, 10, 34, str(path), "Use blue")
    await inbox.mark_uncertain(other.id, "RPC timeout")
    fresh = Inspect()
    assert await svc.begin_idle_retry(other, cast(PiRpcRun, fresh), tmp_path)
    assert fresh.calls == ["get_state", "get_messages"]
    assert (await inbox.get(other.id)).state == "uncertain"


@pytest.mark.anyio
async def test_final_boundary_cannot_miss_receipt_accepted_during_pending_snapshot(
    tmp_path: Path,
):
    class SnapshotInbox(LiveInbox):
        def __init__(self, path):
            super().__init__(path)
            self.calls = 0
            self.entered = anyio.Event()
            self.release = anyio.Event()

        async def pending(self, session_key: str) -> list[Receipt]:
            self.calls += 1
            result = await super().pending(session_key)
            if self.calls == 3:
                self.entered.set()
                await self.release.wait()
            return result

    inbox = SnapshotInbox(tmp_path / "i.json")
    rpc = FakeRun()
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))

    async def reply(*_):
        pass

    async def answer(*_):
        return "answer"

    svc = LiveConversationService(inbox, answer, reply)
    owner = LiveOwner(1, 10, str(tmp_path / "s.jsonl"), runner, "task")
    svc.register(owner)
    finalized: list[bool] = []

    async def finish():
        finalized.append(await svc.finalize(owner))

    async with anyio.create_task_group() as tg:
        svc.attach_workers(tg)
        tg.start_soon(finish)
        with anyio.fail_after(1):
            await inbox.entered.wait()
        accepted = await svc.handle(1, 10, 33, owner.session_key, "/update Use green")
        inbox.release.set()
        with anyio.fail_after(1):
            while not finalized:
                await anyio.sleep(0.01)
        assert not accepted or not finalized[0], (
            "accepted instruction must not be omitted from a successful final"
        )
        svc.unregister(owner)
        svc.stop_workers()
        tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_startup_buffer_persists_later_receipts_while_ack_network_is_held(
    tmp_path: Path,
):
    entered = anyio.Event()
    release = anyio.Event()

    async def reply(*_):
        entered.set()
        await release.wait()

    async def answer(*_):
        return "answer"

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    try:
        async with anyio.create_task_group() as tg:
            if hasattr(svc, "attach_workers"):
                svc.attach_workers(tg)
            tg.start_soon(svc.buffer_starting, 1, 10, 33, "session", "/update First")
            with anyio.fail_after(1):
                await entered.wait()
            with anyio.fail_after(0.2):
                await svc.buffer_starting(1, 10, 34, "session", "/update Second")
            assert [r.text for r in await svc.inbox.pending("session")] == [
                "First",
                "Second",
            ]
            release.set()
            tg.cancel_scope.cancel()
    finally:
        release.set()


@pytest.mark.anyio
async def test_live_poll_accepts_second_receipt_while_first_steer_is_held(
    tmp_path: Path,
):
    class HeldRun(FakeRun):
        def __init__(self):
            super().__init__()
            self.first_entered = anyio.Event()
            self.release = anyio.Event()

        async def steer(self, text: str) -> str:
            self.first_entered.set()
            await self.release.wait()
            return "queued"

    rpc = HeldRun()
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))

    async def reply(_chat, _thread, _msg, _text):
        pass

    async def answer(*_):
        return "answer"

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    owner = LiveOwner(1, 10, str(tmp_path / "s.jsonl"), runner, "task")
    svc.register(owner)
    try:
        async with anyio.create_task_group() as tg:
            if hasattr(svc, "attach_workers"):
                svc.attach_workers(tg)
            tg.start_soon(svc.handle, 1, 10, 33, owner.session_key, "/update First")
            with anyio.fail_after(1):
                await rpc.first_entered.wait()
            with anyio.fail_after(0.2):
                assert await svc.handle(1, 10, 34, owner.session_key, "/update Second")
            assert (await svc.inbox.get((1, 10, 34))).state == "received"
            rpc.release.set()
            tg.cancel_scope.cancel()
    finally:
        rpc.release.set()
        svc.unregister(owner)


@pytest.mark.anyio
async def test_early_rpc_active_flag_does_not_steer_before_initial_prompt_started():
    class EarlyRpc(FakeRun):
        def __init__(self):
            super().__init__()
            self._active = True
            self._prompt_started = False

    rpc = EarlyRpc()
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))
    assert await runner.steer("first update") == "local_pending"
    assert runner._followups == ["first update"]
    assert not rpc.sent


@pytest.mark.anyio
async def test_local_retry_queue_cannot_erase_an_earlier_pi_accepted_submission(
    tmp_path: Path,
):
    rpc = FakeRun()
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))
    replies: list[str] = []

    async def reply(_chat, _thread, _msg, text):
        replies.append(text)

    async def answer(*_):
        return "answer"

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    owner = LiveOwner(1, 10, str(tmp_path / "s.jsonl"), runner, "task")
    svc.register(owner)
    assert await svc.handle(1, 10, 33, owner.session_key, "/update Do X")
    receipt = await svc.inbox.get((1, 10, 33))
    assert receipt.state == "submitted" and rpc.sent
    await svc.inbox.confirm_retry(receipt.id, owner.session_key)
    rpc._active = False
    await svc.flush(owner)
    assert (await svc.inbox.get(receipt.id)).state == "uncertain"
    assert runner._followups == [await svc.inbox.delivery_text(receipt.id)]
    assert await svc.recover_receipt(
        1, 10, 50, owner.session_key, "defer 33 changed mind"
    )
    assert (await svc.inbox.get(receipt.id)).state == "uncertain"
    assert any("refused" in text.lower() for text in replies)


@pytest.mark.anyio
async def test_defer_refuses_pi_accepted_queued_steer_even_before_tool_releases(
    tmp_path: Path,
):
    class QueuedRpc(FakeRun):
        def __init__(self):
            super().__init__()
            self.queued: list[str] = []
            self.executed: list[str] = []

        async def steer(self, text: str) -> str:
            self.queued.append(text)
            return "queued"

        def release_tool(self) -> None:
            self.executed.extend(self.queued)
            self.queued.clear()

    rpc = QueuedRpc()
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))
    replies: list[str] = []

    async def reply(_chat, _thread, _msg, text):
        replies.append(text)

    async def answer(*_):
        return "answer"

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    owner = LiveOwner(1, 10, str(tmp_path / "s.jsonl"), runner, "task")
    svc.register(owner)
    assert await svc.handle(1, 10, 33, owner.session_key, "/update Do X")
    receipt = await svc.inbox.get((1, 10, 33))
    assert receipt.state == "submitted" and len(rpc.queued) == 1
    assert await svc.recover_receipt(
        1, 10, 50, owner.session_key, "defer 33 changed mind"
    )
    assert (await svc.inbox.get(receipt.id)).state == "submitted"
    assert any("refused" in text.lower() for text in replies)
    rpc.release_tool()
    assert rpc.executed == [await svc.inbox.delivery_text(receipt.id)]


@pytest.mark.anyio
async def test_explicit_deferral_removes_locally_queued_followup(tmp_path: Path):
    rpc = FakeRun()
    rpc._active = False
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))
    replies: list[str] = []

    async def reply(_chat, _thread, _msg, text):
        replies.append(text)

    async def answer(*_):
        return "answer"

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    owner = LiveOwner(1, 10, str(tmp_path / "s.jsonl"), runner, "task")
    svc.register(owner)
    await svc.handle(1, 10, 33, owner.session_key, "/update Use green")
    assert len(runner._followups) == 1
    assert await svc.recover_receipt(
        1, 10, 50, owner.session_key, "defer 33 user chose skip"
    )
    assert not runner._followups
    assert (await svc.inbox.get((1, 10, 33))).state == "deferred"


@pytest.mark.anyio
async def test_local_followup_stays_uncertain_until_real_prompt_start_and_restart_does_not_replay(
    tmp_path: Path,
):
    rpc = FakeRun()
    rpc._active = False
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))

    async def answer(*_):
        return "answer"

    async def reply(*_):
        return None

    path = tmp_path / "inbox.json"
    svc = LiveConversationService(LiveInbox(path), answer, reply)
    owner = LiveOwner(1, 10, str(tmp_path / "s.jsonl"), runner, "task")
    svc.register(owner)
    await svc.handle(1, 10, 33, owner.session_key, "/update Use green")
    assert (await svc.inbox.pending(owner.session_key))[0].state == "uncertain"
    recovered = LiveConversationService(LiveInbox(path), answer, reply)
    assert (await recovered.inbox.pending(owner.session_key))[0].state == "uncertain"
    assert not recovered.owner(1, 10, owner.session_key)
    await recovered.flush(owner)
    assert len(runner._followups) == 1, (
        "restart must not enqueue uncertain receipt again"
    )


@pytest.mark.anyio
async def test_settlement_race_uses_same_owner_followup_once(tmp_path: Path):
    rpc = FakeRun()
    rpc._active = False
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))
    replies = []

    async def answer(*_):
        return "answer"

    async def reply(*args):
        replies.append(args)

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    owner = LiveOwner(1, 10, str(tmp_path / "s.jsonl"), runner, "task")
    svc.register(owner)
    receipt = await svc.inbox.receive(1, 10, 33, owner.session_key, "use green")
    await svc.flush(owner)
    assert runner._followups == [await svc.inbox.delivery_text(receipt.id)]
    await svc.submit(owner, receipt)
    assert len(runner._followups) == 1
    assert not rpc.sent, "only the streaming run may start the continuation"
    assert (await svc.inbox.get(receipt.id)).state == "uncertain"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "mode",
    [
        "bound",
        "legacy",
        "unbound",
        "unbound_predebounce",
        "unbound_directive_first",
        "unbound_pi_directive_first",
        "unbound_directive_after",
        "unbound_forward_after_update",
        "bound_held_steer_cancel",
        "bound_other_project",
        "unbound_other_project",
        "bound_plugin_command",
        "bound_plugin_callback",
        "bound_new_during_active",
        "bound_update_after_branch_rebind",
    ],
)
@pytest.mark.parametrize("command_first", [False, True])
async def test_real_loop_accepts_rapid_live_text_before_forward_coalescing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    command_first: bool,
):
    legacy = mode == "legacy"
    unbound = mode in (
        "unbound",
        "unbound_predebounce",
        "unbound_directive_first",
        "unbound_pi_directive_first",
        "unbound_directive_after",
        "unbound_forward_after_update",
        "unbound_other_project",
    )
    predebounce = mode in (
        "unbound_predebounce",
        "unbound_directive_first",
        "unbound_pi_directive_first",
        "unbound_directive_after",
        "unbound_forward_after_update",
        "unbound_other_project",
    )
    held_steer = mode == "bound_held_steer_cancel"
    plugin_mode = mode in ("bound_plugin_command", "bound_plugin_callback")
    plugin_calls: list[str] = []
    steer_entered = anyio.Event()
    steer_release = anyio.Event()

    class HeldRpc(FakeRun):
        def __init__(self, path: Path):
            super().__init__()
            self.path = path
            self.client = self
            self.session_path = path
            self.cwd = tmp_path
            self.release = anyio.Event()
            self.start_gate = anyio.Event()
            self.run_started = anyio.Event()
            self.prompts: list[str] = []
            self._active = False

        async def request(self, typ):
            if typ != "get_state":
                return await super().request(typ)
            if unbound:
                assert not self.path.exists(), "fresh get_state cannot create JSONL"
            else:
                assert self.path.exists()
            return {
                "data": {
                    "sessionId": "fixture",
                    "sessionFile": str(self.path),
                    "isStreaming": False,
                }
            }

        async def run(self, prompt, resume):
            self._active = True
            self.prompts.append(prompt)
            self.run_started.set()
            if unbound:
                await self.start_gate.wait()
            yield StartedEvent(
                engine="pi", resume=ResumeToken(engine="pi", value=str(self.path))
            )
            if unbound:
                import json

                self.path.parent.mkdir(exist_ok=True)
                self.path.write_text(
                    json.dumps(
                        {"type": "session", "id": "fixture", "cwd": str(tmp_path)}
                    )
                    + "\n"
                )
            try:
                await self.release.wait()
                yield CompletedEvent(engine="pi", ok=True, answer="done")
            finally:
                self._active = False

        async def steer(self, text):
            if held_steer:
                steer_entered.set()
                await steer_release.wait()
            return await super().steer(text)

        async def close(self):
            return None

    class TestPi(PiRunner):
        async def run(self, prompt, resume):
            raise AssertionError("live mode must not silently use one-shot Pi")
            yield  # pragma: no cover

        def rpc_run(self, session_path: Path, *, cwd: Path) -> PiRpcRun:
            if not unbound:
                assert session_path == path
            rpc.path = session_path
            rpc.session_path = session_path
            return cast(PiRpcRun, rpc)

    import takopi.telegram.loop as loop

    if plugin_mode:
        monkeypatch.setattr(loop, "list_command_ids", lambda **_: ["spy"])

        async def fake_dispatch(*_args, **_kwargs):
            plugin_calls.append("dispatched")

        monkeypatch.setattr(loop, "dispatch_command", fake_dispatch)

    slow_quick = mode == "bound" and not command_first
    quick_entered = anyio.Event()
    quick_release = anyio.Event()
    cancel_entered = anyio.Event()
    cancel_finished = anyio.Event()
    original_cancel = loop.handle_cancel

    async def trace_cancel(*args):
        cancel_entered.set()
        await original_cancel(*args)
        cancel_finished.set()

    if slow_quick or held_steer:
        monkeypatch.setattr(loop, "handle_cancel", trace_cancel)

    async def quick(_question: str, _snapshot: str) -> str:
        if slow_quick:
            quick_entered.set()
            await quick_release.wait()
        return "No observed result yet."

    monkeypatch.setattr(loop, "quick_pi_answer", quick)
    from takopi.telegram.live_conversation import live_session_path

    path = (
        live_session_path(tmp_path, -100, 77)
        if unbound
        else (
            tmp_path
            / ("existing" if legacy else "pi-live-sessions")
            / ("old-topic.jsonl" if legacy else "-100-77-test.jsonl")
        )
    ).resolve()
    if not unbound:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"type": "session", "id": "fixture", "cwd": str(tmp_path)})
            + "\n"
        )
    rpc = HeldRpc(path)
    runner = TestPi(extra_args=[], model=None, provider=None)
    config = tmp_path / "takopi.toml"
    projects = ProjectsConfig(
        projects={
            "test": ProjectConfig(
                alias="test",
                path=tmp_path,
                worktrees_dir=tmp_path / ".worktrees",
                chat_id=-100,
            ),
            "other": ProjectConfig(
                alias="other",
                path=tmp_path / "other",
                worktrees_dir=tmp_path / "other" / ".worktrees",
                chat_id=-100,
            ),
        },
        default_project=None,
        chat_map={-100: "test"},
    )
    runtime = TransportRuntime(
        router=AutoRouter([RunnerEntry(engine="pi", runner=runner)], "pi"),
        projects=projects,
        config_path=config,
    )
    store = TopicStateStore(resolve_state_path(config))
    await store.set_context(-100, 77, RunContext(project="test"))
    if not unbound:
        await store.set_session_resume(
            -100, 77, ResumeToken(engine="pi", value=str(path))
        )
    progress_ready = anyio.Event()
    transport = FakeTransport(progress_ready=progress_ready)
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
        forward_coalesce_s=0.4,
    )
    emitted = anyio.Event()

    async def poller(_cfg):
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=77,
            message_id=1,
            text=(
                "/pi Build original task"
                if mode == "unbound_pi_directive_first"
                else "/test start"
                if mode == "unbound_directive_first"
                else "start"
            ),
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )
        if not predebounce:
            if (
                unbound
                or held_steer
                or mode in ("bound_other_project", "bound_update_after_branch_rebind")
            ):
                await rpc.run_started.wait()
            else:
                await progress_ready.wait()
        if mode == "bound_update_after_branch_rebind":
            await store.set_context(-100, 77, RunContext(project="test", branch="dev"))
        if mode == "bound_plugin_callback":
            from takopi.telegram.types import TelegramCallbackQuery

            yield TelegramCallbackQuery(
                transport="telegram",
                chat_id=-100,
                message_id=2,
                callback_query_id="spy-2",
                data="spy:change",
                sender_id=123,
                raw={"message": {"message_thread_id": 77}},
            )
        else:
            yield TelegramIncomingMessage(
                transport="telegram",
                chat_id=-100,
                thread_id=77,
                message_id=2,
                text="/spy change"
                if mode == "bound_plugin_command"
                else "/new"
                if mode == "bound_new_during_active"
                else "/update Please deploy branch changes"
                if mode == "bound_update_after_branch_rebind"
                else "/test Change goal"
                if mode == "unbound_directive_after"
                else "/other Please deploy change"
                if mode in ("bound_other_project", "unbound_other_project")
                else "/update Используй синий"
                if command_first
                else "Не трогай авторизацию",
                reply_to_message_id=None,
                reply_to_text=None,
                sender_id=123,
            )
        if held_steer:
            with anyio.fail_after(2):
                await steer_entered.wait()
        yield TelegramIncomingMessage(
            transport="telegram",
            chat_id=-100,
            thread_id=77,
            message_id=3,
            text="Forwarded source content"
            if mode == "unbound_forward_after_update"
            else "Не трогай авторизацию"
            if command_first
            else "Как дела?",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
            raw={"forward_origin": {"type": "user"}}
            if mode == "unbound_forward_after_update"
            else {},
        )
        if command_first:
            yield TelegramIncomingMessage(
                transport="telegram",
                chat_id=-100,
                thread_id=77,
                message_id=4,
                text="Как дела?",
                reply_to_message_id=None,
                reply_to_text=None,
                sender_id=123,
            )
        if slow_quick or held_steer:
            yield TelegramIncomingMessage(
                transport="telegram",
                chat_id=-100,
                thread_id=77,
                message_id=5,
                text="/cancel",
                reply_to_message_id=1,
                reply_to_text="start",
                sender_id=123,
            )
        emitted.set()
        await rpc.release.wait()

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_main_loop, cfg, poller)
        try:
            with anyio.fail_after(3):
                await emitted.wait()
                if predebounce:
                    await rpc.run_started.wait()
            if slow_quick or held_steer:
                with anyio.fail_after(1):
                    if slow_quick:
                        await quick_entered.wait()
                    await cancel_entered.wait()
                    if held_steer:
                        await cancel_finished.wait()
                if slow_quick:
                    assert not quick_release.is_set(), (
                        "cancel must pass a blocked quick responder"
                    )
                if held_steer:
                    assert not steer_release.is_set(), (
                        "cancel must pass a held Pi steer response"
                    )
            pending = await LiveInbox(resolve_inbox_path(config)).pending(str(rpc.path))
            if plugin_mode or mode in (
                "bound_new_during_active",
                "bound_update_after_branch_rebind",
            ):
                assert not plugin_calls, (
                    "plugin cannot bypass live receipt and session binding"
                )
                if mode == "bound_update_after_branch_rebind":
                    assert all(
                        "Please deploy branch changes" not in text for text in rpc.sent
                    )
                    assert all(r.message_id != 2 for r in pending)
                elif mode == "bound_new_during_active":
                    assert (
                        await store.get_session_resume(-100, 77, "pi")
                    ) == ResumeToken("pi", str(path))
                    assert all(r.message_id != 2 for r in pending)
                    assert [(r.message_id, r.text) for r in pending] == (
                        [(3, "Не трогай авторизацию")] if command_first else []
                    ), "message after rejected /new must retain the old live owner"
                else:
                    assert all(r.message_id != 2 for r in pending)
                    assert (
                        await store.get_session_resume(-100, 77, "pi")
                    ) == ResumeToken("pi", str(path))
                return
            if mode in ("bound_other_project", "unbound_other_project"):
                assert [(r.message_id, r.text) for r in pending] == (
                    [(3, "Не трогай авторизацию")] if command_first else []
                ), "other-project directive must not reach this Pi"
                assert all("Please deploy change" not in text for text in rpc.sent)
                assert (await store.get_context(-100, 77)) == RunContext(project="test")
                return
            assert [item.text for item in pending] == (
                (["Используй синий"] if command_first else ["Не трогай авторизацию"])
                if mode == "unbound_forward_after_update"
                else (["Не трогай авторизацию"] if command_first else [])
                if mode == "unbound_directive_after"
                else ["Используй синий", "Не трогай авторизацию"]
                if command_first
                else ["Не трогай авторизацию"]
            )
            if mode == "unbound_forward_after_update":
                assert any(
                    "forwarded message not accepted" in call["message"].text.lower()
                    for call in transport.send_calls
                )
                assert rpc.prompts[0].endswith("start")
            elif held_steer:
                assert pending[0].state in ("uncertain", "received")
                assert len(rpc.prompts) == 1
            elif mode == "unbound_directive_after":
                assert any(
                    "not accepted" in call["message"].text.lower()
                    for call in transport.send_calls
                )
                assert rpc.prompts[0].endswith("start")
            elif predebounce:
                assert [item.state for item in pending] == (
                    ["uncertain", "received"] if command_first else ["uncertain"]
                )
                assert rpc.prompts[0].endswith(
                    "Build original task"
                    if mode == "unbound_pi_directive_first"
                    else "start"
                ), "initial task must not be replaced by rapid text"
                assert not rpc.sent
            elif command_first and not unbound:
                assert [item.state for item in pending] == ["uncertain", "received"]
                assert not rpc.sent, "local startup queue is not an accepted Pi prompt"
                rpc.release.set()
                with anyio.fail_after(2):
                    while len(rpc.prompts) != 3:
                        await anyio.sleep(0.01)
                assert rpc.prompts[1].endswith("Используй синий")
                assert rpc.prompts[2].endswith("Не трогай авторизацию")
            else:
                assert all(item.state in ("submitted", "uncertain") for item in pending)
                if any(item.state == "uncertain" for item in pending):
                    assert not rpc.sent, "startup/cancel ambiguity is not acceptance"
                    if mode == "legacy" and not slow_quick:
                        rpc.release.set()
                        with anyio.fail_after(2):
                            while len(rpc.prompts) < 2:
                                await anyio.sleep(0.01)
                        assert rpc.prompts[1] == await LiveInbox(
                            resolve_inbox_path(config)
                        ).delivery_text(pending[0].id)
                else:
                    assert len(rpc.sent) == len(pending)
            if command_first and unbound and not predebounce:
                assert rpc.sent[0].endswith("Используй синий")
                assert rpc.sent[1].endswith("Не трогай авторизацию")
        finally:
            quick_release.set()
            steer_release.set()
            rpc.start_gate.set()
            rpc.release.set()
            tg.cancel_scope.cancel()


@pytest.mark.anyio
async def test_short_legacy_pi_topic_does_not_silently_fallback_to_one_shot(
    tmp_path: Path,
):
    class ShortIdPi(PiRunner):
        async def run(self, prompt, resume):
            raise AssertionError("unsafe one-shot fallback")
            yield  # pragma: no cover

    runner = ShortIdPi(extra_args=[], model=None, provider=None)
    config = tmp_path / "takopi.toml"
    projects = ProjectsConfig(
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
    )
    runtime = TransportRuntime(
        router=AutoRouter([RunnerEntry(engine="pi", runner=runner)], "pi"),
        projects=projects,
        config_path=config,
    )
    store = TopicStateStore(resolve_state_path(config))
    await store.set_context(-100, 77, RunContext(project="test"))
    await store.set_session_resume(
        -100, 77, ResumeToken(engine="pi", value="legacy-short-id")
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
            text="start",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )

    await run_main_loop(cfg, poller)
    assert any(
        "не найден однозначно" in item["message"].text for item in transport.send_calls
    )
    assert await store.get_session_resume(-100, 77, "pi") == ResumeToken(
        engine="pi", value="legacy-short-id"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("wrong_state", [False, True])
async def test_bound_short_id_migrates_only_after_canonical_get_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrong_state: bool
):
    import json
    from uuid import uuid4
    from takopi.runners.pi import _default_session_dir

    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi"))
    ident = uuid4().hex
    prefix = ident[:10]
    session_dir = _default_session_dir(tmp_path)
    session_dir.mkdir(parents=True)
    session = session_dir / f"2026-10-04_{ident}.jsonl"
    session.write_text(
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": ident,
                "cwd": str(tmp_path),
                "timestamp": "2026-10-04T00:00:00.000Z",
            }
        )
        + "\n"
    )

    class MigratingRpc(FakeRun):
        def __init__(self):
            super().__init__()
            self._active = False
            self.calls: list[str] = []

        async def request(self, typ):
            self.calls.append(typ)
            if typ == "get_state":
                return {
                    "data": {
                        "sessionId": "wrong-id" if wrong_state else ident,
                        "sessionFile": str(session),
                        "isStreaming": False,
                    }
                }
            return {"data": {"messages": []}}

        async def run(self, prompt, resume):
            self.calls.append("prompt")
            yield StartedEvent(
                engine="pi", resume=ResumeToken(engine="pi", value=str(session))
            )
            yield CompletedEvent(engine="pi", ok=True, answer="done")

        async def close(self):
            pass

    rpc = MigratingRpc()

    class MigratingPi(PiRunner):
        async def run(self, prompt, resume):
            raise AssertionError("unsafe one-shot fallback")
            yield  # pragma: no cover

        def rpc_run(self, session_path: Path, *, cwd: Path) -> PiRpcRun:
            assert session_path == session.resolve()
            return cast(PiRpcRun, rpc)

    runner = MigratingPi(extra_args=[], model=None, provider=None)
    config = tmp_path / "takopi.toml"
    projects = ProjectsConfig(
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
    )
    runtime = TransportRuntime(
        router=AutoRouter([RunnerEntry(engine="pi", runner=runner)], "pi"),
        projects=projects,
        config_path=config,
    )
    store = TopicStateStore(resolve_state_path(config))
    await store.set_context(-100, 77, RunContext(project="test"))
    await store.set_session_resume(-100, 77, ResumeToken(engine="pi", value=prefix))
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
            text="start",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )

    await run_main_loop(cfg, poller)
    if wrong_state:
        assert rpc.calls == ["get_state"]
        assert await store.get_session_resume(-100, 77, "pi") == ResumeToken(
            "pi", prefix
        )
    else:
        assert rpc.calls[:2] == ["get_state", "prompt"]
        assert await store.get_session_resume(-100, 77, "pi") == ResumeToken(
            "pi", str(session.resolve())
        )


@pytest.mark.anyio
@pytest.mark.parametrize("cancel", [False, True])
async def test_settlement_continuation_is_rendered_before_final_and_cancellable(
    tmp_path: Path, cancel: bool
):
    import asyncio

    class SettlingRun(FakeRun):
        def __init__(self):
            super().__init__()
            self.started_followup = asyncio.Event()
            self.release_followup = asyncio.Event()

        async def run(self, prompt, resume):
            self._active = True
            self.sent.append(prompt)
            yield StartedEvent(engine="pi", resume=ResumeToken(engine="pi", value="a"))
            if len(self.sent) == 2:
                self.started_followup.set()
                await self.release_followup.wait()
                answer = "UPDATED FINAL RESULT"
            else:
                answer = "outdated result"
            self._active = False
            yield CompletedEvent(engine="pi", ok=True, answer=answer)

    rpc = SettlingRun()
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))

    async def answer(*_):
        return "answer"

    async def reply(*_):
        return None

    svc = LiveConversationService(LiveInbox(tmp_path / "i.json"), answer, reply)
    owner = LiveOwner(1, 10, str(tmp_path / "s.jsonl"), runner, "task")
    svc.register(owner)
    receipt = await svc.inbox.receive(1, 10, 33, owner.session_key, "use green")
    runner.before_final = lambda: svc.flush(owner)
    runner.on_followup_started = lambda text: svc.followup_started(owner, text)
    emitted: list[CompletedEvent] = []

    async def consume():
        emitted.extend(
            [
                event
                async for event in runner.run("build", None)
                if isinstance(event, CompletedEvent)
            ]
        )

    task = asyncio.create_task(consume())
    await asyncio.wait_for(rpc.started_followup.wait(), 1)
    assert not emitted, "old answer must not be final before queued update completes"
    assert (await svc.inbox.get(receipt.id)).state == "submitted"
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        rpc.release_followup.set()
        assert not emitted, "cancelled follow-up must not report success"
    else:
        rpc.release_followup.set()
        await asyncio.wait_for(task, 1)
        assert [event.answer for event in emitted] == ["UPDATED FINAL RESULT"]


@pytest.mark.anyio
async def test_main_prompt_teaches_explicit_ack_without_quick_transcript():
    rpc = FakeRun()
    runner = LiveRunner(FakePi(), cast(PiRpcRun, rpc))
    _ = [event async for event in runner.run("build feature", None)]
    assert "[takopi-considered:" in rpc.sent[0]
    assert "[takopi-deferred:" in rpc.sent[0]
    assert "build feature" in rpc.sent[0]
