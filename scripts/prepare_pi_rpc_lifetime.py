#!/usr/bin/env python3
"""Prepare/verify one pinned compatibility package COPY; never modify its source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

PATCH_ROOT = Path(__file__).resolve().parents[1] / "patches/pi-subagents-0.74.0"
HELPER = "src/extension/takopi-rpc-lifetime.js"


def _manifest() -> dict:
    return json.loads((PATCH_ROOT / "manifest.json").read_text())


def _versions(package: Path, pi: Path, manifest: dict) -> None:
    metadata = json.loads((package / "package.json").read_text())
    if (
        metadata.get("name") != manifest["package"]
        or metadata.get("version") != manifest["version"]
    ):
        raise ValueError("Expected exact pi-subagents 0.74.0 package")
    if (
        json.loads((pi / "package.json").read_text()).get("version")
        != manifest["piVersion"]
    ):
        raise ValueError(
            "Expected exact Pi 0.99.2; revalidate compatibility before upgrading"
        )


def _digests(package: Path, manifest: dict, kind: str) -> None:
    for name, digests in manifest["files"].items():
        file = package / name
        expected = digests[kind]
        if file.is_symlink() or any(
            parent.is_symlink()
            for parent in file.parents
            if parent != package and package in parent.parents
        ):
            raise ValueError(f"Refusing symlinked patch target: {name}")
        actual = (
            hashlib.sha256(file.read_bytes()).hexdigest() if file.exists() else None
        )
        if actual != expected:
            raise ValueError(f"{kind} source digest mismatch: {name}")


def verify(package: Path, pi: Path) -> None:
    manifest = _manifest()
    _versions(package, pi, manifest)
    _digests(package, manifest, "patched")


def prepare(source: Path, destination: Path, pi: Path) -> None:
    source, destination, pi = source.resolve(), destination.resolve(), pi.resolve()
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError(
            "Source and destination must be separate non-nested directories"
        )
    manifest = _manifest()
    # Validate EVERY source before writing anything. An existing destination is
    # only verified, never repaired or partially overwritten.
    _versions(source, pi, manifest)
    _digests(source, manifest, "original")
    if destination.exists():
        verify(destination, pi)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        shutil.copytree(source, stage, dirs_exist_ok=True, symlinks=True)
        result = subprocess.run(
            [
                "patch",
                "--batch",
                "--forward",
                "-p1",
                "-i",
                str((PATCH_ROOT / "rpc-lifetime.patch").resolve()),
            ],
            cwd=stage,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"Compatibility patch failed: {result.stdout}{result.stderr}"
            )
        helper = stage / HELPER
        helper.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PATCH_ROOT / "takopi-rpc-lifetime.js", helper)
        # Dependencies remain read-only links to the original dependency tree;
        # no npm install, package discovery/config mutation, or live patching.
        if not (stage / "node_modules").exists():
            (stage / "node_modules").symlink_to(source.parent, target_is_directory=True)
        verify(stage, pi)
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--pi-package", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.destination, args.pi_package)
    elif args.source:
        prepare(args.source, args.destination, args.pi_package)
    else:
        parser.error("--source is required unless --verify is used")
    print(f"Verified pinned RPC lifetime package: {args.destination}")


if __name__ == "__main__":
    main()
