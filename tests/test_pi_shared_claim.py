"""Pre-spawn in-process canonical ownership for Pi RPC and one-shot runners."""

import json
import sys
from uuid import uuid4
from pathlib import Path

import pytest

from takopi.model import ResumeToken
from takopi.runners.pi import PiRunner
from takopi.runners.pi_rpc import PiRpcClient


async def _running_rpc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PiRpcClient:
    from takopi.runners import pi_rpc

    script = tmp_path / "fake_pi.py"
    script.write_text("import sys\nfor line in sys.stdin: pass\n")
    monkeypatch.setattr(pi_rpc, "_PI_COMMAND", sys.executable)
    rpc = PiRpcClient(tmp_path / "owned.jsonl", tmp_path, ["-u", str(script)])
    await rpc.start()
    return rpc


@pytest.mark.anyio
async def test_one_shot_cannot_spawn_over_rpc_canonical_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from takopi.runners import pi_rpc

    rpc = await _running_rpc(tmp_path, monkeypatch)
    spawned = False

    async def forbidden_spawn(*args, **kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("one-shot must not spawn against a claimed Pi session")

    monkeypatch.setattr("takopi.utils.subprocess.anyio.open_process", forbidden_spawn)
    runner = PiRunner(extra_args=[], model=None, provider=None)
    try:
        with pi_rpc.enable_live_claims(), pytest.raises(RuntimeError, match="owner"):
            await anext(
                runner.run("Another task", ResumeToken("pi", str(rpc.session_path)))
            )
        assert not spawned
    finally:
        await rpc.close()


@pytest.mark.anyio
async def test_short_id_alias_cannot_spawn_over_rpc_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from takopi.runners import pi_rpc

    rpc = await _running_rpc(tmp_path, monkeypatch)
    ident = uuid4().hex
    rpc.session_path.write_text(
        json.dumps({"type": "session", "id": ident, "cwd": str(Path.cwd())}) + "\n"
    )
    monkeypatch.setenv("PI_CODING_AGENT_SESSION_DIR", str(tmp_path))
    spawned = False

    async def forbidden_spawn(*args, **kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("short ID cannot spawn a second owner")

    monkeypatch.setattr("takopi.utils.subprocess.anyio.open_process", forbidden_spawn)
    runner = PiRunner(extra_args=[], model=None, provider=None)
    try:
        with pi_rpc.enable_live_claims(), pytest.raises(RuntimeError, match="owner"):
            await anext(runner.run("Another task", ResumeToken("pi", ident[:12])))
        assert not spawned
    finally:
        await rpc.close()


@pytest.mark.anyio
async def test_reverse_claim_blocks_rpc_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from takopi.runners import pi_rpc

    path = tmp_path / "owned.jsonl"
    async with pi_rpc.claim_one_shot(path):
        rpc = PiRpcClient(path, tmp_path, [])
        with pytest.raises(RuntimeError, match="owner"):
            await rpc.start()
    # The rejected RPC client may not remove another claimant's lease.
    async with pi_rpc.claim_one_shot(path):
        pass


@pytest.mark.anyio
async def test_missing_short_id_fails_closed_only_when_live_opted_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from takopi.runners import pi_rpc

    runner = PiRunner(extra_args=[], model=None, provider=None)
    spawned = False

    async def forbidden_spawn(*args, **kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("legacy one-shot proceeds to child without a live claim")

    monkeypatch.setattr("takopi.utils.subprocess.anyio.open_process", forbidden_spawn)
    with (
        pi_rpc.enable_live_claims(),
        pytest.raises(ValueError, match="not uniquely found"),
    ):
        await anext(runner.run("Legacy", ResumeToken("pi", "deadbeef0000")))
    assert not spawned
    with pytest.raises(AssertionError, match="legacy one-shot proceeds"):
        await anext(runner.run("Legacy", ResumeToken("pi", "deadbeef0000")))
    assert spawned


@pytest.mark.anyio
async def test_rejected_one_shot_never_releases_the_real_rpc_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from takopi.runners import pi_rpc

    rpc = await _running_rpc(tmp_path, monkeypatch)

    async def forbidden_spawn(*args, **kwargs):
        raise AssertionError("must refuse before spawn")

    monkeypatch.setattr("takopi.utils.subprocess.anyio.open_process", forbidden_spawn)
    runner = PiRunner(extra_args=[], model=None, provider=None)
    try:
        with pi_rpc.enable_live_claims():
            for _ in range(2):
                with pytest.raises(RuntimeError, match="owner"):
                    await anext(
                        runner.run(
                            "Another task", ResumeToken("pi", str(rpc.session_path))
                        )
                    )
    finally:
        await rpc.close()
