"""Per-agent sandbox access: the commands built, when setup runs, and (as root) the OS refusing."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import actant.sandbox.local as local_backend
from actant.sandbox import ExecResult, LocalSandbox, SandboxAccess
from actant.sandbox.access import enforceable, run_argv, setup_argv
from actant.sandbox.modal import ModalSandbox


def test_access_rejects_bad_users_and_paths() -> None:
    for user in ("", "Agent", "1agent", "a" * 33, "a b"):
        with pytest.raises(ValueError, match="Unix user"):
            SandboxAccess(user)
    for path in ("", "/abs", "a/../b", "a//b", "./a"):
        with pytest.raises(ValueError, match="relative"):
            SandboxAccess("agent", writable=[path])


def test_user_for_is_stable_and_valid() -> None:
    user = SandboxAccess.user_for("thread-Kitchen/1")
    assert user == SandboxAccess.user_for("thread-Kitchen/1") != SandboxAccess.user_for("x")
    assert SandboxAccess(user).user == user and len(user) <= 32


def test_run_argv_runs_as_the_user_with_its_scratch() -> None:
    argv = ["bash", "-lc", "echo hi"]
    scratch = "/tmp/actant-agent"
    assert run_argv(SandboxAccess("agent"), argv) == [
        *("runuser", "-u", "agent", "--", "env", f"HOME={scratch}", f"TMPDIR={scratch}"),
        *argv,
    ]
    unscratched = run_argv(SandboxAccess("agent", scratch=False), argv)
    assert unscratched == ["runuser", "-u", "agent", "--", "env", *argv]


def test_setup_argv_passes_paths_as_arguments() -> None:
    access = SandboxAccess("agent", writable=["objects/chair", "room.py"])
    argv = setup_argv(access, "/ws", lock=True)
    assert argv[:2] == ["bash", "-c"] and "useradd -M -s /bin/bash" in argv[2]
    assert argv[4:] == [
        *("agent", "/tmp/actant-agent", "/ws", "lock"),
        *("/ws/objects/chair", "/ws/room.py"),
    ]
    unlocked = setup_argv(SandboxAccess("agent", scratch=False), "/ws", lock=False)
    assert unlocked[4:] == ["agent", "", "/ws", ""]


class _Modal:
    """A Modal sandbox handle that records each ``exec`` and exits with ``code``."""

    def __init__(self, code: int = 0) -> None:
        self.execs: list[tuple[str, ...]] = []
        self.object_id = "sb-1"

        async def run(*argv: str, **_: Any) -> Any:
            self.execs.append(argv)
            stderr = "useradd: denied" if code else ""
            return SimpleNamespace(
                stdout=SimpleNamespace(read=SimpleNamespace(aio=_value(""))),
                stderr=SimpleNamespace(read=SimpleNamespace(aio=_value(stderr))),
                wait=SimpleNamespace(aio=_value(code)),
            )

        self.exec = SimpleNamespace(aio=run)


def _value(value: object) -> Any:
    async def fn() -> object:
        return value

    return fn


async def test_modal_exec_with_access_readies_then_runs_as_the_user() -> None:
    fake = _Modal()
    sandbox = ModalSandbox(fake, {}, root="/ws", scrub_env=("KEY",))
    access = SandboxAccess("agent", writable=["mine"])
    await sandbox.exec(["bash", "-lc", "true"], timeout=5, access=access)
    await sandbox.exec(["bash", "-lc", "true"], timeout=5, access=access)
    setup, command, again, _ = fake.execs
    # The workspace is locked once; the agent's paths are chowned on every call.
    assert setup[:2] == ("timeout", "120") and setup[-2:] == ("lock", "/ws/mine")
    assert again[-2:] == ("", "/ws/mine")
    assert command[2:] == ("env", "-uKEY", *run_argv(access, ["bash", "-lc", "true"]))


async def test_modal_exec_without_access_is_unchanged() -> None:
    fake = _Modal()
    await ModalSandbox(fake, {}, root="/ws").exec(["true"], timeout=5)
    assert fake.execs == [("timeout", "5", "true")]


async def test_modal_setup_failure_raises() -> None:
    sandbox = ModalSandbox(_Modal(code=1), {}, root="/ws")
    with pytest.raises(RuntimeError, match="useradd: denied"):
        await sandbox.exec(["true"], timeout=5, access=SandboxAccess("agent"))


async def test_local_exec_as_root_readies_once_then_runs_as_the_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[list[str]] = []

    async def run(
        self: LocalSandbox,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout: float,
        env: Mapping[str, str],
    ) -> ExecResult:
        del self, cwd, timeout, env
        ran.append(list(argv))
        return ExecResult(0, "", "")

    monkeypatch.setattr(local_backend, "enforceable", lambda: True)
    monkeypatch.setattr(LocalSandbox, "_run", run)
    sandbox = LocalSandbox(tmp_path)
    access = SandboxAccess("agent", writable=["mine"])
    await sandbox.exec(["true"], timeout=5, access=access)
    await sandbox.exec(["true"], timeout=5, access=access)
    root = str(sandbox.root)
    assert ran[0][-2:] == ["lock", f"{root}/mine"] and ran[2][-2:] == ["", f"{root}/mine"]
    assert ran[1] == ran[3] == run_argv(access, ["true"])


async def test_local_exec_off_root_runs_unenforced_and_warns_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(local_backend, "enforceable", lambda: False)
    sandbox = LocalSandbox(tmp_path)
    with caplog.at_level(logging.WARNING, logger=local_backend.__name__):
        for _ in range(2):
            result = await sandbox.exec(
                ["sh", "-c", "echo ok"], timeout=5, access=SandboxAccess("agent")
            )
            assert result.stdout == "ok\n"
    assert [r.getMessage() for r in caplog.records if "not enforced" in r.getMessage()] == [
        f"{sandbox.id}: not root, so sandbox access for 'agent' is not enforced"
    ]


@pytest.mark.skipif(not enforceable(), reason="needs root, useradd and runuser (a sandbox)")
async def test_as_root_the_os_refuses_another_agents_files(tmp_path: Path) -> None:
    os.chmod(tmp_path, 0o755)
    sandbox = LocalSandbox(tmp_path)
    (tmp_path / "mine").mkdir()
    (tmp_path / "theirs").mkdir()
    (tmp_path / "shared.txt").write_text("root's\n")
    user = SandboxAccess.user_for(str(tmp_path), prefix="actest-")
    access = SandboxAccess(user, writable=["mine"])
    try:
        script = (
            "echo a > mine/a.txt; echo b > theirs/b.txt; echo c >> shared.txt; "
            'cat shared.txt; echo d > "$TMPDIR/d.txt" && echo scratch ok'
        )
        result = await sandbox.exec(["bash", "-c", script], timeout=30, access=access)
        assert result.stderr.count("Permission denied") == 2
        assert "root's" in result.stdout and "scratch ok" in result.stdout
        assert (tmp_path / "mine" / "a.txt").is_file()
        assert not (tmp_path / "theirs" / "b.txt").exists()
        assert (tmp_path / "shared.txt").read_text() == "root's\n"
        plain = await sandbox.exec(["bash", "-c", "echo e > theirs/e.txt"], timeout=30)
        assert plain.returncode == 0  # without access, as before
    finally:
        shutil.rmtree(f"/tmp/actant-{user}", ignore_errors=True)
        subprocess.run(["userdel", user], capture_output=True, check=False)
