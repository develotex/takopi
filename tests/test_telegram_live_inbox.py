from __future__ import annotations

import json
from pathlib import Path

import pytest

from takopi.telegram.live_inbox import LiveInbox, resolve_inbox_path


@pytest.mark.anyio
async def test_duplicate_telegram_message_is_one_receipt_and_preserves_original(
    tmp_path: Path,
) -> None:
    inbox = LiveInbox(tmp_path / "inbox.json")
    first = await inbox.receive(1, 10, 100, "session-a", "original")
    duplicate = await inbox.receive(1, 10, 100, "session-a", "edited")
    assert duplicate == first
    assert first.text == "original"
    assert first.state == "received"
    assert len(await inbox.pending("session-a")) == 1
    with pytest.raises(ValueError, match="different scope"):
        await inbox.receive(1, 10, 100, "session-b", "original")


@pytest.mark.anyio
async def test_fifo_and_cross_topic_session_isolation(tmp_path: Path) -> None:
    inbox = LiveInbox(tmp_path / "inbox.json")
    second = await inbox.receive(1, 10, 102, "session-a", "second")
    await inbox.receive(1, 11, 101, "session-b", "private")
    first = await inbox.receive(1, 10, 101, "session-a", "first")
    assert [r.id for r in await inbox.pending("session-a")] == [second.id, first.id]
    assert [r.text for r in await inbox.pending("session-b")] == ["private"]
    assert first.sequence > second.sequence
    assert first.marker != second.marker


@pytest.mark.anyio
async def test_each_receipt_state_survives_restart_without_claiming_delivery(
    tmp_path: Path,
) -> None:
    path = tmp_path / "inbox.json"
    inbox = LiveInbox(path)
    received = await inbox.receive(1, 10, 1, "session", "received")
    uncertain = await inbox.receive(1, 10, 2, "session", "uncertain")
    submitted = await inbox.receive(1, 10, 3, "session", "submitted")
    delivered = await inbox.receive(1, 10, 4, "session", "delivered")
    considered = await inbox.receive(1, 10, 5, "session", "considered")
    deferred = await inbox.receive(1, 10, 6, "session", "deferred")
    await inbox.mark_uncertain(uncertain.id, "command may have reached Pi")
    await inbox.mark_submitted(submitted.id)
    await inbox.mark_submitted(delivered.id)
    await inbox.reconcile(
        "session",
        [{"role": "user", "content": await inbox.delivery_text(delivered.id)}],
    )
    await inbox.mark_submitted(considered.id)
    await inbox.reconcile(
        "session",
        [{"role": "user", "content": await inbox.delivery_text(considered.id)}],
    )
    await inbox.mark_considered(considered.id, "applied to plan")
    await inbox.mark_deferred(deferred.id, "needs clarification")

    restarted = LiveInbox(path)
    assert [r.state for r in await restarted.pending("session")] == [
        "received",
        "uncertain",
        "submitted",
        "delivered",
    ]
    assert (await restarted.get(considered.id)).reason == "applied to plan"
    assert (await restarted.get(deferred.id)).reason == "needs clarification"
    assert await restarted.receive(
        1, 10, 3, "session", "submitted"
    ) == await restarted.get(submitted.id)
    assert received.marker == (await restarted.get(received.id)).marker
    next_receipt = await restarted.receive(1, 10, 7, "session", "next")
    assert next_receipt.sequence > deferred.sequence


@pytest.mark.anyio
async def test_reconcile_only_observed_user_message_in_exact_session(
    tmp_path: Path,
) -> None:
    inbox = LiveInbox(tmp_path / "inbox.json")
    target = await inbox.receive(1, 10, 1, "session-a", "constraint")
    other = await inbox.receive(1, 11, 2, "session-b", "private")
    await inbox.mark_submitted(target.id)  # Accepted RPC command is NOT delivery.
    assert (await inbox.get(target.id)).state == "submitted"
    text = await inbox.delivery_text(target.id)
    await inbox.reconcile("session-b", [{"role": "user", "content": text}])
    await inbox.reconcile("session-a", [{"role": "assistant", "content": text}])
    await inbox.reconcile("session-a", [{"role": "user", "content": "prefix " + text}])
    assert (await inbox.get(target.id)).state == "submitted"
    await inbox.reconcile("session-a", [{"role": "user", "content": text}])
    assert (await inbox.get(target.id)).state == "delivered"
    assert (await inbox.get(other.id)).state == "received"
    assert (await inbox.get(target.id)).reason is None


@pytest.mark.anyio
async def test_uncertain_receipt_is_not_resent_without_reconciliation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "inbox.json"
    inbox = LiveInbox(path)
    receipt = await inbox.receive(1, 10, 1, "session", "constraint")
    await inbox.mark_uncertain(receipt.id, "RPC timed out before response")
    restarted = LiveInbox(path)
    assert (await restarted.get(receipt.id)).state == "uncertain"
    assert [r.id for r in await restarted.pending("session")] == [receipt.id]
    await restarted.reconcile("session", [])
    assert (await restarted.get(receipt.id)).state == "uncertain"
    await restarted.reconcile(
        "session",
        [{"role": "user", "content": await restarted.delivery_text(receipt.id)}],
    )
    assert (await restarted.get(receipt.id)).state == "delivered"


@pytest.mark.anyio
async def test_settlement_race_remains_pending_until_explicit_consideration(
    tmp_path: Path,
) -> None:
    inbox = LiveInbox(tmp_path / "inbox.json")
    receipt = await inbox.receive(1, 10, 1, "session", "late update")
    assert [r.id for r in await inbox.pending("session")] == [receipt.id]
    await inbox.mark_submitted(receipt.id)
    await inbox.reconcile(
        "session", [{"role": "user", "content": await inbox.delivery_text(receipt.id)}]
    )
    assert [r.state for r in await inbox.pending("session")] == ["delivered"]
    with pytest.raises(ValueError, match="explicit"):
        await inbox.mark_considered(receipt.id, "")
    await inbox.mark_considered(receipt.id, "main agent confirmed application")
    assert await inbox.pending("session") == []


@pytest.mark.anyio
async def test_corrupt_or_truncated_existing_store_fails_closed_without_erasure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "inbox.json"
    path.write_bytes(b'{"version":1,"receipts":[')
    inbox = LiveInbox(path)
    with pytest.raises(ValueError, match="inbox"):
        await inbox.receive(1, 10, 1, "session", "cannot overwrite")
    assert path.read_bytes() == b'{"version":1,"receipts":['
    path.write_text(json.dumps({"version": 999, "receipts": []}))
    with pytest.raises(ValueError, match="inbox"):
        await LiveInbox(path).pending("session")
    path.write_text(json.dumps({"version": 1}))
    with pytest.raises(ValueError, match="inbox"):
        await LiveInbox(path).pending("session")
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "next_sequence": 2,
                "receipts": [
                    {
                        "chat_id": 1,
                        "thread_id": 10,
                        "message_id": 1,
                        "session_key": "session",
                        "text": "update",
                        "sequence": 1,
                        "marker": "abc",
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="inbox"):
        await LiveInbox(path).pending("session")


@pytest.mark.anyio
async def test_failed_atomic_save_does_not_leave_ghost_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import takopi.telegram.live_inbox as module

    path = tmp_path / "inbox.json"
    inbox = LiveInbox(path)
    await inbox.receive(1, 10, 1, "session", "committed")
    real_replace = module.os.replace

    def fail_replace(src: str, dst: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        await inbox.receive(1, 10, 2, "session", "not committed")
    monkeypatch.setattr(module.os, "replace", real_replace)
    assert [r.text for r in await inbox.pending("session")] == ["committed"]
    assert [r.text for r in await LiveInbox(path).pending("session")] == ["committed"]


@pytest.mark.anyio
async def test_delivery_requires_exact_evidence_and_consideration_requires_delivery(
    tmp_path: Path,
) -> None:
    inbox = LiveInbox(tmp_path / "inbox.json")
    receipt = await inbox.receive(1, 10, 1, "session", "change")
    with pytest.raises(ValueError, match="Cannot move"):
        await inbox.mark_considered(receipt.id, "applied")
    with pytest.raises(ValueError, match="matching"):
        await inbox.mark_delivered(
            receipt.id,
            {"role": "assistant", "content": await inbox.delivery_text(receipt.id)},
        )
    delivered = await inbox.mark_delivered(
        receipt.id,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": await inbox.delivery_text(receipt.id)}
            ],
        },
    )
    assert delivered.state == "delivered"


def test_inbox_resolves_next_to_config(tmp_path: Path) -> None:
    assert (
        resolve_inbox_path(tmp_path / "config.toml")
        == tmp_path / "telegram_live_inbox.json"
    )
