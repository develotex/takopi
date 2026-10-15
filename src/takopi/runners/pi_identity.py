"""Read-only canonical Pi session identity discovery (never starts a writer)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .pi import _default_session_dir


def session_header(path: Path) -> dict | None:
    try:
        with path.open("rb") as handle:
            line = handle.readline(8193)
        if len(line) > 8192 or not line.endswith(b"\n"):
            return None
        header = json.loads(line)
        return (
            header
            if isinstance(header, dict) and header.get("type") == "session"
            else None
        )
    except (OSError, UnicodeError, ValueError):
        return None


def resolve_legacy_session(partial_id: str, cwd: Path) -> Path:
    """Identify exactly one same-project file by its header before claiming a writer.

    A match remains a candidate: the RPC owner validates get_state under the
    canonical claim before prompting. One-shot Pi starts only after the claim.
    """
    if re.fullmatch(r"[a-fA-F0-9-]{8,36}", partial_id) is None:
        raise ValueError("Invalid Pi session ID prefix")
    project = cwd.resolve()
    configured = os.environ.get("PI_CODING_AGENT_SESSION_DIR")
    directory = (
        Path(configured).expanduser() if configured else _default_session_dir(project)
    ).resolve()
    matches: set[Path] = set()
    if directory.is_dir():
        for file in directory.glob("*.jsonl"):
            path = file.resolve()
            if path.parent != directory or not file.is_file():
                continue
            header = session_header(path)
            if (
                header is not None
                and isinstance(header.get("id"), str)
                and header["id"].lower().startswith(partial_id.lower())
                and isinstance(header.get("cwd"), str)
                and Path(header["cwd"]).resolve() == project
            ):
                matches.add(path)
    if len(matches) != 1:
        raise ValueError("Pi session ID not uniquely found in this project")
    return matches.pop()
