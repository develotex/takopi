import os
import signal
import sys
from pathlib import Path

import anyio

import pytest

from takopi.utils import subprocess as subprocess_utils


@pytest.mark.anyio
async def test_manage_subprocess_kills_when_terminate_times_out(
    monkeypatch,
) -> None:
    async def fake_wait_for_process(_proc, timeout: float) -> bool:
        _ = timeout
        return True

    monkeypatch.setattr(subprocess_utils, "wait_for_process", fake_wait_for_process)

    async with subprocess_utils.manage_subprocess(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(10)",
        ]
    ) as proc:
        assert proc.returncode is None

    assert proc.returncode is not None
    assert proc.returncode != 0


@pytest.mark.anyio
@pytest.mark.parametrize("kill_descendants", [False, True])
@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc").exists(), reason="Linux groups"
)
async def test_parent_exit_preserves_legacy_tools_but_fences_claimed_pi(
    tmp_path: Path, kill_descendants: bool
) -> None:
    pid_file = tmp_path / "tool.pid"
    script = (
        "import subprocess, sys, pathlib\n"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'], "
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid))\n"
    )
    child_pid = None
    try:
        async with subprocess_utils.manage_subprocess(
            [sys.executable, "-c", script], kill_descendants=kill_descendants
        ) as proc:
            with anyio.fail_after(2):
                while not pid_file.exists():
                    await anyio.sleep(0.01)
            child_pid = int(pid_file.read_text())
            assert await proc.wait() == 0
        stat = Path(f"/proc/{child_pid}/stat")
        if kill_descendants:
            with anyio.fail_after(1):
                while stat.exists() and stat.read_text().split()[2] != "Z":
                    await anyio.sleep(0.02)
        else:
            assert stat.exists() and stat.read_text().split()[2] != "Z", (
                "Non-Pi/opt-in-off tools retain legacy background behavior"
            )
    finally:
        if child_pid is not None:
            stat = Path(f"/proc/{child_pid}/stat")
            if stat.exists() and stat.read_text().split()[2] != "Z":
                os.kill(child_pid, signal.SIGKILL)  # Synthetic test child only.
