"""Legacy short session IDs must resolve to one exact project session before prompt."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from takopi.runners.pi import _default_session_dir
from takopi.telegram import live_conversation
from takopi.telegram.live_conversation import LiveConversationService, LiveOwner
from takopi.telegram.live_inbox import LiveInbox
from takopi.runner_bridge import RunningTask
from takopi.model import ResumeToken
from takopi.transport import MessageRef


def _session(cwd: Path, ident: str) -> Path:
    directory = _default_session_dir(cwd)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"2026-10-04_{ident}.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": ident,
                "timestamp": "2026-10-04T00:00:00.000Z",
                "cwd": str(cwd),
            }
        )
        + "\n"
    )
    return path.resolve()


def test_short_id_resolves_only_matching_project_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi"))
    cwd = tmp_path / "project"
    cwd.mkdir()
    ident = uuid4().hex
    target = _session(cwd, ident)
    assert live_conversation.resolve_legacy_session(ident[:12], cwd) == target
    assert target.read_text().count("\n") == 1


def test_aliases_and_topics_cannot_both_own_same_canonical_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi"))
    cwd = tmp_path / "project"
    cwd.mkdir()
    ident = uuid4().hex
    target = _session(cwd, ident)
    assert live_conversation.resolve_legacy_session(ident[:8], cwd) == target
    assert live_conversation.resolve_legacy_session(ident[:12], cwd) == target

    async def answer(*_):
        return "answer"

    async def reply(*_):
        return None

    svc = LiveConversationService(LiveInbox(tmp_path / "inbox.json"), answer, reply)
    first = LiveOwner(1, 10, str(target), object(), "first")  # type: ignore[arg-type]
    second = LiveOwner(1, 11, str(target), object(), "second")  # type: ignore[arg-type]
    assert svc.register(first)
    assert not svc.register(second)


def test_inflight_one_shot_alias_blocks_migration_until_settled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi"))
    cwd = tmp_path / "project"
    cwd.mkdir()
    ident = uuid4().hex
    path = _session(cwd, ident)
    running = RunningTask(resume=ResumeToken("pi", ident[:8]))
    tasks = {MessageRef(channel_id=1, message_id=5): running}
    assert live_conversation.legacy_session_busy(tasks, path)
    running.done.set()
    assert not live_conversation.legacy_session_busy(tasks, path)


def test_unknown_ambiguous_and_cross_project_short_ids_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi"))
    cwd = tmp_path / "project"
    other = tmp_path / "other"
    cwd.mkdir()
    other.mkdir()
    _session(cwd, "abcde123" + uuid4().hex)
    _session(cwd, "abcde123" + uuid4().hex)
    _session(other, "other123" + uuid4().hex)
    for prefix in ("abcde123", "missing-id", "other123"):
        with pytest.raises(ValueError):
            live_conversation.resolve_legacy_session(prefix, cwd)
