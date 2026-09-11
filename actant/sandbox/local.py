"""The local backend: a directory on this machine, commands as subprocesses.

No isolation. It is the development backend and the one tests use; in
production the directory is whatever the operator mounted there.
"""

from __future__ import annotations

import asyncio
import os
import signal
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from actant.sandbox.base import Entry, ExecResult, Sandbox, SandboxSpec


class LocalSandbox:
    def __init__(self, root: Path, env: Mapping[str, str] | None = None) -> None:
        self.root = root.resolve()
        self.id = str(self.root)
        self._env = dict(env or {})

    def _path(self, path: str) -> Path:
        target = (self.root / path).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(f"path escapes the sandbox: {path!r}")
        return target

    async def read(self, path: str) -> bytes:
        return await asyncio.to_thread(self._path(path).read_bytes)

    async def write(self, path: str, data: bytes) -> None:
        target = self._path(path)

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await asyncio.to_thread(_write)

    async def ls(self, pattern: str) -> list[Entry]:
        def _ls() -> list[Entry]:
            entries = []
            for match in sorted(self.root.glob(pattern)):
                if match.is_file():
                    stat = match.stat()
                    entries.append(
                        Entry(str(match.relative_to(self.root)), stat.st_size, stat.st_mtime)
                    )
            return entries

        return await asyncio.to_thread(_ls)

    async def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        timeout: float,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self._path(cwd) if cwd else self.root,
            env={**os.environ, **self._env, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except (TimeoutError, asyncio.CancelledError):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = await process.communicate()
            return ExecResult(
                124,
                stdout.decode(errors="replace"),
                stderr.decode(errors="replace") + "\nTimed out.\n",
                timed_out=True,
            )
        return ExecResult(
            process.returncode or 0,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )

    async def close(self) -> None:
        """The directory is the durable root; nothing to release."""


class LocalSandboxProvider:
    """Sandboxes under ``spec.mount`` (or ``root``, or a temp dir), one directory per thread."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root

    async def open(self, spec: SandboxSpec, *, agent_id: str, thread_id: str) -> Sandbox:
        del agent_id
        base = Path(spec.mount) if spec.mount else (self.root or Path(tempfile.mkdtemp("-actant")))
        root = base / thread_id
        root.mkdir(parents=True, exist_ok=True)
        return LocalSandbox(root, spec.env)

    async def attach(self, spec: SandboxSpec, sandbox_id: str) -> Sandbox:
        root = Path(sandbox_id)
        if not root.is_dir():
            raise KeyError(sandbox_id)
        return LocalSandbox(root, spec.env)
