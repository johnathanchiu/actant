"""Per-agent sandbox access: the commands built, when setup runs, and (as root) the OS refusing."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import actant.sandbox.local as local_backend
from actant.sandbox import ExecResult, LocalSandbox, SandboxAccess
from actant.sandbox.access import enforceable, run_argv, setup_argv
from actant.sandbox.modal import ModalSandbox
from actant.agents import AgentDefinition
from actant.llm.providers.fake import FakeLLM
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.stores.postgres.conversion import set_thread_access, thread_access
from actant.runtime.stores.postgres.models import ActantThreadModel
from actant.runtime.temporal.activities.context import ActivityContext
from actant.runtime.temporal.activities.runs import RunActivities
from actant.runtime.temporal.activities.tools import ToolActivities
from actant.runtime.temporal.types import StartRunInput
from actant.tools.calls import ToolCallRecord
from actant.tools.registry import ToolRegistry
from runtime_fixtures import static_agents


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
    argv = setup_argv(access, "/ws", restrict_workspace=True, new_user=True)
    assert argv[:2] == ["bash", "-c"] and "useradd -M -s /bin/bash" in argv[2]
    user, scratch, root, marker, new, *paths = argv[4:]
    assert (user, scratch, root, new) == ("agent", "/tmp/actant-agent", "/ws", "new")
    assert marker.startswith("/tmp/actant-restricted-")
    assert paths == ["/ws/objects/chair", "/ws/room.py"]
    again = setup_argv(
        SandboxAccess("agent", scratch=False), "/ws", restrict_workspace=False, new_user=False
    )
    assert again[4:] == ["agent", "", "/ws", "", ""]


def _setup(argv: Sequence[str]) -> dict[str, Any] | None:
    """A setup command's fields, or ``None`` for an agent's command."""
    if "actant-access" not in argv:
        return None
    user, _, _, marker, new, *paths = argv[argv.index("actant-access") + 1 :]
    return {"user": user, "restrict": bool(marker), "new": bool(new), "paths": paths}


class _Modal:
    """A Modal sandbox handle that records each ``exec``. A setup exits with the next of
    ``codes`` (0 once they run out) after ``setup_s``, printing the paths it owns; an
    agent's command waits on ``command`` when given."""

    def __init__(
        self,
        codes: Sequence[int] = (),
        *,
        setup_s: float = 0.0,
        command: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.execs: list[tuple[str, ...]] = []
        self.object_id = "sb-1"
        self.active_setups = 0
        self.most_setups = 0
        codes = list(codes)

        async def run(*argv: str, **_: Any) -> Any:
            self.execs.append(argv)
            setup = _setup(argv)
            code, stdout = 0, ""
            if setup is not None:
                self.active_setups += 1
                self.most_setups = max(self.most_setups, self.active_setups)
                try:
                    await asyncio.sleep(setup_s)
                finally:
                    self.active_setups -= 1
                code = codes.pop(0) if codes else 0
                stdout = "".join(f"{path}\n" for path in setup["paths"])
            elif command is not None:
                await command()
            return SimpleNamespace(
                stdout=SimpleNamespace(read=SimpleNamespace(aio=_value(stdout))),
                stderr=SimpleNamespace(read=SimpleNamespace(aio=_value("denied" if code else ""))),
                wait=SimpleNamespace(aio=_value(code)),
            )

        self.exec = SimpleNamespace(aio=run)

    def setups(self) -> list[dict[str, Any]]:
        return [s for s in map(_setup, self.execs) if s is not None]


def _value(value: object) -> Any:
    async def fn() -> object:
        return value

    return fn


async def test_modal_exec_readies_once_then_runs_as_the_user() -> None:
    fake = _Modal()
    sandbox = ModalSandbox(fake, {}, root="/ws", scrub_env=("KEY",))
    access = SandboxAccess("agent", writable=["mine"])
    await sandbox.exec(["bash", "-lc", "true"], timeout=5, access=access)
    await sandbox.exec(["bash", "-lc", "true"], timeout=5, access=access)
    # One setup: once the agent owns its paths, later calls skip it.
    assert fake.setups() == [
        {"user": "agent", "restrict": True, "new": True, "paths": ["/ws/mine"]}
    ]
    setup, command, again = fake.execs
    assert setup[:2] == ("timeout", "120") and command == again
    assert command[2:] == ("env", "-uKEY", *run_argv(access, ["bash", "-lc", "true"]))


async def test_a_path_that_appears_later_is_chowned_then() -> None:
    fake = _Modal()
    sandbox = ModalSandbox(fake, {}, root="/ws")
    await sandbox.exec(["true"], timeout=5, access=SandboxAccess("agent", writable=["a"]))
    await sandbox.exec(["true"], timeout=5, access=SandboxAccess("agent", writable=["a", "b"]))
    assert fake.setups()[1] == {
        "user": "agent",
        "restrict": False,
        "new": False,
        "paths": ["/ws/a", "/ws/b"],
    }


async def test_modal_exec_without_access_is_unchanged() -> None:
    fake = _Modal()
    await ModalSandbox(fake, {}, root="/ws").exec(["true"], timeout=5)
    assert fake.execs == [("timeout", "5", "true")]


async def test_concurrent_first_execs_restrict_the_workspace_once() -> None:
    fake = _Modal(setup_s=0.05)
    sandbox = ModalSandbox(fake, {}, root="/ws")
    chair = SandboxAccess("chair", writable=["chair"])
    table = SandboxAccess("table", writable=["table"])
    await asyncio.gather(
        sandbox.exec(["true"], timeout=5, access=chair),
        sandbox.exec(["true"], timeout=5, access=table),
    )
    setups = fake.setups()
    assert fake.most_setups == 1  # serialized: no chown races the restriction
    assert [s["restrict"] for s in setups] == [True, False]
    assert sorted((s["user"], tuple(s["paths"])) for s in setups) == [
        ("chair", ("/ws/chair",)),
        ("table", ("/ws/table",)),
    ]


async def test_agents_commands_run_in_parallel(caplog: pytest.LogCaptureFixture) -> None:
    started = 0
    both = asyncio.Event()

    async def command() -> None:
        nonlocal started
        started += 1
        if started == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 2)  # times out if commands were serialized

    fake = _Modal(setup_s=0.15, command=command)
    sandbox = ModalSandbox(fake, {}, root="/ws")
    with caplog.at_level(logging.INFO, logger="actant.sandbox.access"):
        await asyncio.gather(
            sandbox.exec(["sleep"], timeout=5, access=SandboxAccess("chair")),
            sandbox.exec(["sleep"], timeout=5, access=SandboxAccess("table")),
        )
    assert any("waited" in r.getMessage() for r in caplog.records)


async def test_a_failed_setup_releases_the_mutex_and_is_retried() -> None:
    fake = _Modal(codes=[1])
    sandbox = ModalSandbox(fake, {}, root="/ws")
    access = SandboxAccess("agent", writable=["mine"])
    with pytest.raises(RuntimeError, match="denied"):
        await sandbox.exec(["true"], timeout=5, access=access)
    await asyncio.wait_for(sandbox.exec(["true"], timeout=5, access=access), 1)
    assert [s["restrict"] for s in fake.setups()] == [True, True]


async def test_a_cancelled_setup_releases_the_mutex() -> None:
    fake = _Modal(setup_s=10)
    sandbox = ModalSandbox(fake, {}, root="/ws")
    task = asyncio.create_task(sandbox.exec(["true"], timeout=5, access=SandboxAccess("chair")))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    fake_fast = _Modal()
    sandbox._sandbox = fake_fast  # the next setup returns at once
    await asyncio.wait_for(sandbox.exec(["true"], timeout=5, access=SandboxAccess("table")), 1)
    assert fake_fast.setups()[0]["restrict"]  # the cancelled one never recorded it


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
        setup = _setup(argv)
        return ExecResult(0, "\n".join(setup["paths"]) if setup else "", "")

    monkeypatch.setattr(local_backend, "enforceable", lambda: True)
    monkeypatch.setattr(LocalSandbox, "_run", run)
    sandbox = LocalSandbox(tmp_path)
    access = SandboxAccess("agent", writable=["mine"])
    await sandbox.exec(["true"], timeout=5, access=access)
    await sandbox.exec(["true"], timeout=5, access=access)
    setup, first, second = ran
    assert _setup(setup) == {
        "user": "agent",
        "restrict": True,
        "new": True,
        "paths": [f"{sandbox.root}/mine"],
    }
    assert first == second == run_argv(access, ["true"])


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


def test_two_threads_of_one_definition_own_only_their_folders() -> None:
    chair = SandboxAccess(SandboxAccess.user_for("thread-chair"), writable=["objects/chair"])
    table = SandboxAccess(SandboxAccess.user_for("thread-table"), writable=["objects/table"])
    assert chair.user != table.user
    for mine, theirs in ((chair, table), (table, chair)):
        setup = _setup(setup_argv(mine, "/ws", restrict_workspace=False, new_user=True))
        assert setup == {
            "user": mine.user,
            "restrict": False,
            "new": True,
            "paths": [f"/ws/{mine.writable[0]}"],
        }
        assert run_argv(mine, ["true"])[:3] == ["runuser", "-u", mine.user]


async def test_access_round_trips_through_a_thread_row() -> None:
    access = SandboxAccess("agent", writable=["a", "b/c"], scratch=False)
    row = ActantThreadModel(agent_id="a", thread_id="t", status="idle")
    set_thread_access(row, access)
    assert (row.sandbox_user, row.sandbox_writable, row.sandbox_scratch) == (
        "agent",
        ["a", "b/c"],
        False,
    )
    assert thread_access(row) == access
    set_thread_access(row, None)
    assert thread_access(row) is None and row.sandbox_writable is None


_DEFAULT = SandboxAccess("agent-default", writable=["shared"])


async def _context_access(
    stores: InMemoryRuntimeStores, thread_id: str, monkeypatch: pytest.MonkeyPatch
) -> SandboxAccess | None:
    agent = AgentDefinition(
        id="author",
        name="author",
        persona="",
        llm=FakeLLM([]),
        tools=ToolRegistry([]),
        sandbox_access=_DEFAULT,
    )
    context = ActivityContext(stores=stores, resolve_agent=static_agents({}))

    async def sandbox_for(*_: object) -> LocalSandbox:
        return LocalSandbox(Path("/tmp"))

    monkeypatch.setattr(context, "sandbox_for", sandbox_for)
    record = SimpleNamespace(
        agent_id="author", thread_id=thread_id, run_id="r", id="tc", turn_id="t"
    )
    tool = SimpleNamespace(needs_sandbox=True)
    ctx = await ToolActivities(context)._call_context(agent, tool, cast(ToolCallRecord, record))
    return ctx.sandbox_access


async def test_a_thread_started_with_access_keeps_it_over_the_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = InMemoryRuntimeStores()
    runs = RunActivities(ActivityContext(stores=stores, resolve_agent=static_agents({})))
    chair = SandboxAccess("author-chair", writable=["objects/chair"])
    await runs.start_run(StartRunInput("author", "chair", "r1", None, sandbox_access=chair))
    # A later start (a restart, a replay) without it keeps what was recorded.
    await runs.start_run(StartRunInput("author", "chair", "r2", None))
    await stores.threads.get_or_create("author", "plain")
    assert (await stores.threads.get("author", "chair")).sandbox_access == chair
    assert await _context_access(stores, "chair", monkeypatch) == chair
    assert await _context_access(stores, "plain", monkeypatch) == _DEFAULT


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
        # A second thread of the same definition owns its own folder, not the first's; both
        # start at once, so their setups race the workspace's restriction.
        other = SandboxAccess(f"{user}-2", writable=["theirs"])
        result, theirs = await asyncio.gather(
            sandbox.exec(["bash", "-c", script], timeout=30, access=access),
            sandbox.exec(
                ["bash", "-c", "echo f > theirs/f.txt; echo g > mine/g.txt"],
                timeout=30,
                access=other,
            ),
        )
        assert result.stderr.count("Permission denied") == 2
        assert "root's" in result.stdout and "scratch ok" in result.stdout
        assert (tmp_path / "mine" / "a.txt").is_file()
        assert not (tmp_path / "theirs" / "b.txt").exists()
        assert (tmp_path / "shared.txt").read_text() == "root's\n"
        assert theirs.stderr.count("Permission denied") == 1
        assert (tmp_path / "theirs" / "f.txt").is_file()
        assert not (tmp_path / "mine" / "g.txt").exists()
        plain = await sandbox.exec(["bash", "-c", "echo e > theirs/e.txt"], timeout=30)
        assert plain.returncode == 0  # without access, as before
    finally:
        marker = setup_argv(access, str(sandbox.root), restrict_workspace=True, new_user=False)[7]
        shutil.rmtree(marker, ignore_errors=True)
        for name in (user, f"{user}-2"):
            shutil.rmtree(f"/tmp/actant-{name}", ignore_errors=True)
            subprocess.run(["userdel", name], capture_output=True, check=False)
