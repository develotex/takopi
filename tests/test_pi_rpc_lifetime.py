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
async def test_reconcile_does_not_hide_lifetime_error() -> None:
    run = PiRpcRun(Records(events("TAKOPI_RPC_LIFETIME: child failed")))  # type: ignore[arg-type]
    run._unsettled = True
    with pytest.raises(RuntimeError, match="child failed"):
        _ = [record async for record in run.run("do not retry", None)]
