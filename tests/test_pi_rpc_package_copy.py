"""Atomic compatibility preparation never edits the installed source package."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "rpc_copy", Path(__file__).parents[1] / "scripts/prepare_pi_rpc_lifetime.py"
)


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert SPEC and SPEC.loader
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    patch = tmp_path / "patch"
    patch.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (source / "package.json").write_text('{"name":"pi-subagents","version":"0.74.0"}')
    (source / "original.js").write_text("old\n")
    pi = tmp_path / "pi"
    pi.mkdir()
    (pi / "package.json").write_text('{"version":"0.99.2"}')

    def digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    (patch / "manifest.json").write_text(
        json.dumps(
            {
                "package": "pi-subagents",
                "version": "0.74.0",
                "piVersion": "0.99.2",
                "files": {
                    "original.js": {
                        "original": digest("old\n"),
                        "patched": digest("new\n"),
                    },
                    "src/extension/takopi-rpc-lifetime.js": {
                        "original": None,
                        "patched": digest("helper\n"),
                    },
                },
            }
        )
    )
    (patch / "rpc-lifetime.patch").write_text(
        "--- a/original.js\n+++ b/original.js\n@@ -1 +1 @@\n-old\n+new\n"
    )
    (patch / "takopi-rpc-lifetime.js").write_text("helper\n")
    monkeypatch.setattr(module, "PATCH_ROOT", patch)
    return module, source, tmp_path / "copy", pi


def test_copy_is_atomic_preserves_original_and_verifies_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, source, destination, pi = fixture(tmp_path, monkeypatch)
    module.prepare(source, destination, pi)
    assert (source / "original.js").read_text() == "old\n"
    assert (destination / "original.js").read_text() == "new\n"
    module.prepare(source, destination, pi)
    module.verify(destination, pi)


@pytest.mark.parametrize("bad", ["source", "version", "pi", "patch"])
def test_mismatch_never_leaves_partial_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    module, source, destination, pi = fixture(tmp_path, monkeypatch)
    if bad == "source":
        (source / "original.js").write_text("unexpected\n")
    elif bad == "version":
        (source / "package.json").write_text('{"version":"0.75.0"}')
    elif bad == "pi":
        (pi / "package.json").write_text('{"version":"0.99.3"}')
    else:
        (module.PATCH_ROOT / "rpc-lifetime.patch").write_text("invalid patch\n")
    with pytest.raises((ValueError, RuntimeError)):
        module.prepare(source, destination, pi)
    assert not destination.exists()
    assert not list(tmp_path.glob(".copy-*"))


def test_modified_destination_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, source, destination, pi = fixture(tmp_path, monkeypatch)
    module.prepare(source, destination, pi)
    (destination / "original.js").write_text("manual change\n")
    with pytest.raises(ValueError, match="digest"):
        module.prepare(source, destination, pi)
    assert (destination / "original.js").read_text() == "manual change\n"
