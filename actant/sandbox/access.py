"""Per-agent write access inside a shared sandbox, enforced by the OS.

Several agents can share one sandbox (one key, many threads). An agent declares the
paths it owns with a :class:`SandboxAccess` on its definition; a command run with it
runs as that agent's own Unix user, which owns those paths and nothing else in the
workspace. A write anywhere else fails with ``Permission denied`` in the command's
stderr, the same as any other failed write, so the model sees it and adapts.

Enforcing needs root and ``runuser`` where the command runs (a container backend runs
as root). The workspace must be a real disk that keeps owners and modes: ``disk_sync``
storage or ``local``, not a ``mount`` bucket.

This module only builds argv; the backends run it (:meth:`Sandbox.exec`). The ``local``
backend, which isolates nothing, runs the command unenforced (with a warning) where it is
not root, as it ignores a spec's ``image`` and ``gpu``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

_log = logging.getLogger(__name__)

#: A setup that waits longer than this for another agent's is logged.
SLOW_WAIT_S = 0.1

#: How long readying an agent's user and paths may take (a ``chown -R`` of its folders).
SETUP_TIMEOUT_S = 120

#: A portable Unix user name: what ``useradd`` accepts everywhere.
_USER = re.compile(r"[a-z_][a-z0-9_-]{0,31}")

#: Run as root before an agent's command. Positional: user, scratch dir ("" for none),
#: workspace root, the marker that records the workspace was restricted ("" to skip that
#: step), "new" for a user this sandbox handle has not readied ("" otherwise), then the
#: absolute writable paths. Restricting makes the workspace root's and read-only to others,
#: once per workspace: ``mkdir`` of the marker is atomic, so of two handles one restricts.
#: Then the agent's paths that exist are chowned (printed, one per line, so the caller
#: knows which it owns now); one that does not exist yet is skipped.
_SETUP = """set -e
user=$1 scratch=$2 root=$3 marker=$4 new=$5; shift 5
if [ -n "$marker" ] && mkdir "$marker" 2>/dev/null; then
  { chown -R 0:0 "$root" && chmod -R go-w,a+rX "$root"; } || { rmdir "$marker"; exit 1; }
fi
if [ -n "$new" ]; then
  id -u "$user" >/dev/null 2>&1 || useradd -M -s /bin/bash "$user"
  if [ -n "$scratch" ]; then mkdir -p "$scratch"; chown "$user" "$scratch"; chmod 700 "$scratch"; fi
fi
for path in "$@"; do if [ -e "$path" ]; then chown -R "$user" "$path"; echo "$path"; fi; done
"""


@dataclass(frozen=True)
class SandboxAccess:
    """What one agent may write in its sandbox. Declared on :class:`~actant.agents.AgentDefinition`
    and passed to :meth:`Sandbox.exec`.

    ``user`` is the agent's Unix user, made on first use (:meth:`user_for` derives a valid one
    from an id). ``writable`` are paths relative to the workspace root that it owns; the rest
    of the workspace it can read but not change. ``scratch`` gives it a private directory
    outside the workspace as ``HOME`` and ``TMPDIR``.
    """

    user: str
    writable: Sequence[str] = ()
    scratch: bool = True

    def __post_init__(self) -> None:
        if not _USER.fullmatch(self.user):
            raise ValueError(f"not a Unix user name: {self.user!r}; see SandboxAccess.user_for")
        object.__setattr__(self, "writable", tuple(self.writable))
        for path in self.writable:
            if (
                not path
                or path.startswith("/")
                or any(part in ("", ".", "..") for part in path.split("/"))
            ):
                raise ValueError(f"writable paths are relative and normalized: {path!r}")

    @staticmethod
    def user_for(key: str, prefix: str = "agent-") -> str:
        """A stable, valid Unix user name for an agent or thread id."""
        return prefix + hashlib.sha256(key.encode()).hexdigest()[:12]

    def scratch_dir(self, tmp: str = "/tmp") -> str | None:
        return f"{tmp}/actant-{self.user}" if self.scratch else None


def enforceable() -> bool:
    """Whether this process can run commands as other users: root, with ``runuser``."""
    return (
        os.geteuid() == 0
        and shutil.which("runuser") is not None
        and shutil.which("useradd") is not None
    )


def setup_argv(
    access: SandboxAccess,
    root: str,
    *,
    restrict_workspace: bool,
    new_user: bool,
    tmp: str = "/tmp",
) -> list[str]:
    """The root command that readies ``access`` in the workspace at ``root`` (absolute)."""
    marker = f"{tmp}/actant-restricted-{hashlib.sha256(root.encode()).hexdigest()[:12]}"
    return [
        "bash",
        "-c",
        _SETUP,
        "actant-access",
        access.user,
        access.scratch_dir(tmp) or "",
        root,
        marker if restrict_workspace else "",
        "new" if new_user else "",
        *(f"{root}/{path}" for path in access.writable),
    ]


class _Ran(Protocol):
    """What ``AccessSetup`` reads of a finished command (an ``ExecResult``)."""

    @property
    def returncode(self) -> int: ...
    @property
    def stdout(self) -> str: ...
    @property
    def stderr(self) -> str: ...


class AccessSetup:
    """One sandbox's readying of its agents.

    The first setup restricts the workspace; an agent's first makes its user and scratch dir
    and chowns its paths. Setups take a mutex, so no agent's chown races the workspace's
    restriction; commands never hold it, so agents' commands run in parallel. Once an agent
    owns all its ``writable`` paths, later calls skip setup and the mutex altogether; a path
    that did not exist yet is chowned on a later call once it does. ``run`` runs as root.
    """

    def __init__(self, root: str, run: Callable[[list[str]], Awaitable[_Ran]]) -> None:
        self._root = root
        self._run = run
        self._mutex = asyncio.Lock()
        self._restricted = False
        #: Per readied user, the absolute paths it owns.
        self._owned: dict[str, set[str]] = {}

    def _missing(self, access: SandboxAccess) -> set[str] | None:
        """The writable paths ``access.user`` does not own yet; ``None`` for a new user."""
        owned = self._owned.get(access.user)
        if owned is None:
            return None
        return {f"{self._root}/{path}" for path in access.writable} - owned

    async def ready(self, access: SandboxAccess) -> None:
        if self._missing(access) == set():
            return
        waited = time.monotonic()
        async with self._mutex:
            waited = time.monotonic() - waited
            if waited > SLOW_WAIT_S:
                _log.info("sandbox access for %r waited %.2fs for setup", access.user, waited)
            missing = self._missing(access)
            if missing == set():
                return
            argv = setup_argv(
                access,
                self._root,
                restrict_workspace=not self._restricted,
                new_user=missing is None,
            )
            result = await self._run(argv)
            if result.returncode:
                raise RuntimeError(
                    f"sandbox access for {access.user!r} failed: {result.stderr.strip()}"
                )
            self._restricted = True
            owned = self._owned.setdefault(access.user, set())
            owned.update(line for line in result.stdout.splitlines() if line)


def run_argv(access: SandboxAccess, argv: Sequence[str], *, tmp: str = "/tmp") -> list[str]:
    """``argv`` run as ``access.user``, its scratch dir as ``HOME`` and ``TMPDIR``."""
    scratch = access.scratch_dir(tmp)
    home = [f"HOME={scratch}", f"TMPDIR={scratch}"] if scratch else []
    return ["runuser", "-u", access.user, "--", "env", *home, *argv]
