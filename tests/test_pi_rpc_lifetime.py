"""Compatibility drain failures must not become successful RPC completions."""

from pathlib import Path
from typing import Any

import pytest

from takopi.model import CompletedEvent
from takopi.runners.pi_rpc import PiRpcRun


class Records:
    cwd = Path("/mnt")
    session_path = Path("/mnt/lifetime-test.jsonl")

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    async def start(self) -> None:
        pass

    async def request(self, kind: str, **_: Any) -> dict[str, Any]:
        if kind == "get_state":
            return {
                "data": {"sessionId": "owner", "sessionFile": str(self.session_path)}
            }
        return {"data": {"disposition": "started"}}

    async def events(self):
        for record in self.records:
            callback = getattr(self, "before_record", None)
            if callback is not None:
                callback(record)
            yield record


def events(error: str, event: str = "agent_end") -> list[dict[str, Any]]:
    return [
        {"type": "agent_start"},
        {
            "type": "extension_error",
            "extensionPath": "/copy/pi-subagents/index.js",
            "event": event,
            "error": error,
        },
        {"type": "agent_end", "messages": [], "willRetry": False},
        {"type": "agent_settled"},
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "reason",
    ["timed out", "child failed", "needs attention", "session changed", "cancelled"],
)
async def test_lifetime_error_cannot_settle_successfully(reason: str) -> None:
    run = PiRpcRun(Records(events(f"TAKOPI_RPC_LIFETIME: {reason}")))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match=reason):
        _ = [record async for record in run.run("hello", None)]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error,event",
    [
        ("unrelated extension failure", "agent_end"),
        ("Naming example mentions TAKOPI_RPC_LIFETIME: but is unrelated", "agent_end"),
        ("TAKOPI_RPC_LIFETIME: example", "tool_call"),
    ],
)
async def test_unrelated_extension_errors_preserve_existing_behavior(
    error: str, event: str
) -> None:
    run = PiRpcRun(Records(events(error, event)))  # type: ignore[arg-type]
    output = [record async for record in run.run("hello", None)]
    assert len([record for record in output if isinstance(record, CompletedEvent)]) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("known_cancel", ["interrupting", "requested"])
async def test_expected_machine_cancellation_consumes_settlement(
    known_cancel: str,
) -> None:
    import anyio

    client = Records(events("TAKOPI_RPC_LIFETIME:CANCELLED: operation stopped"))
    run = PiRpcRun(client)  # type: ignore[arg-type]
    run.cancel_requested = anyio.Event()

    def before_record(record: dict[str, Any]) -> None:
        if record.get("type") == "extension_error":
            if known_cancel == "interrupting":
                run._interrupting = True
            else:
                assert run.cancel_requested is not None
                run.cancel_requested.set()

    client.before_record = before_record  # type: ignore[attr-defined]
    _ = [record async for record in run.run("hello", None)]
    assert not run._unsettled


@pytest.mark.anyio
@pytest.mark.parametrize(
    "marker",
    [
        "TAKOPI_RPC_LIFETIME:CANCELLED: unexpected abort",
        "TAKOPI_RPC_LIFETIME: This operation was aborted",
        "TAKOPI_RPC_LIFETIME: child failed",
    ],
)
async def test_unknown_cancel_or_real_failure_remains_terminal(marker: str) -> None:
    run = PiRpcRun(Records(events(marker)))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="lifetime failed"):
        _ = [record async for record in run.run("hello", None)]


@pytest.mark.anyio
async def test_reconcile_expected_cancel_before_next_prompt() -> None:
    run = PiRpcRun(Records(events("TAKOPI_RPC_LIFETIME:CANCELLED: operation stopped")))  # type: ignore[arg-type]
    run._unsettled = True
    run._interrupting = True
    await run._reconcile()
    assert not run._unsettled


@pytest.mark.anyio
async def test_expected_drain_abort_renders_cancelled_not_error() -> None:
    import asyncio
    import anyio

    from takopi.markdown import MarkdownPresenter
    from takopi.runners.pi import PiRunner
    from takopi.runner_bridge import ExecBridgeConfig, IncomingMessage, handle_message
    from takopi.transport import MessageRef
    from tests.test_runner_bridge_progress import FakeTransport

    class CancelRecords(Records):
        def __init__(self) -> None:
            super().__init__([])
            self.abort_sent = anyio.Event()

        async def send_request(self, kind: str, **_: Any):
            future = asyncio.get_running_loop().create_future()
            future.set_result({"success": True})
            return kind, future

        async def wait_response(self, _ident: str, future: Any):
            return await future

        async def request(self, kind: str, **fields: Any):
            if kind == "abort":
                self.abort_sent.set()
                await anyio.sleep(0)
                return {"success": True}
            return await super().request(kind, **fields)

        async def events(self):
            yield {"type": "agent_start"}
            await self.abort_sent.wait()
            for record in events("TAKOPI_RPC_LIFETIME:CANCELLED: operation stopped")[
                1:
            ]:
                yield record

    rpc = PiRpcRun(CancelRecords())  # type: ignore[arg-type]

    class Runner(PiRunner):
        def run(self, prompt: str, resume: Any):
            return rpc.run(prompt, resume)

    transport = FakeTransport()
    running = {}
    cfg = ExecBridgeConfig(
        transport=transport,
        presenter=MarkdownPresenter(),
        final_notify=True,
        progress_updates="none",
        show_typing=False,
    )

    async def execute() -> None:
        await handle_message(
            cfg,
            runner=Runner(extra_args=[], model=None, provider=None),
            incoming=IncomingMessage(channel_id=123, message_id=10, text="hello"),
            resume_token=None,
            running_tasks=running,
        )  # type: ignore[arg-type]

    with anyio.fail_after(3):
        async with anyio.create_task_group() as group:
            group.start_soon(execute)
            while (
                not running
                or running[MessageRef(channel_id=123, message_id=10)].control is None
            ):
                await anyio.sleep(0)
            running[MessageRef(channel_id=123, message_id=10)].cancel_requested.set()
    rendered = [
        call["message"].text for call in transport.send_calls + transport.edit_calls
    ]
    assert any("cancelled" in text.lower() for text in rendered), rendered
    assert not any("error" in text.lower() for text in rendered), rendered


@pytest.mark.anyio
async def test_reconcile_does_not_hide_lifetime_error() -> None:
    run = PiRpcRun(Records(events("TAKOPI_RPC_LIFETIME: child failed")))  # type: ignore[arg-type]
    run._unsettled = True
    with pytest.raises(RuntimeError, match="child failed"):
        _ = [record async for record in run.run("do not retry", None)]
