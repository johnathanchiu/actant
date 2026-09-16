"""The local backend: a directory on this machine, commands as subprocesses.

No isolation. It is the development backend and the one tests use; in
production the directory is whatever the operator mounted there. A spec's
services are served by a host subprocess on 127.0.0.1 with a random token.
Given an :class:`~actant.sandbox.base.ImageBucket`, that host uploads the images its
services return and returns durable asset references (bucket keys from the environment).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import signal
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import actant.sandbox.host as host
from actant.sandbox.base import Endpoint, Entry, ExecResult, ImageBucket, Sandbox, SandboxSpec
from actant.sandbox.protocol import EntryConfig, Header, HostConfig

#: How long a host may take to import its services and bind.
HOST_START_TIMEOUT_S = 60
#: :mod:`actant.sandbox.entry`, named not imported: running a module this package imports
#: with ``-m`` would load it twice.
ENTRY_MODULE = "actant.sandbox.entry"


def _environment(env: Mapping[str, str]) -> dict[str, str]:
    # ``python`` in argv is this interpreter: the one the services' own package is
    # installed in, which is what a container backend bakes into its image.
    return {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        **env,
    }


class LocalSandbox:
    def __init__(
        self,
        root: Path,
        env: Mapping[str, str] | None = None,
        scrub_env: Sequence[str] = (),
        *,
        endpoint: Endpoint | None = None,
        host_process: asyncio.subprocess.Process | None = None,
    ) -> None:
        self.root = root.resolve()
        self.id = str(self.root)
        self._endpoint = endpoint
        self.host_process = host_process
        self._env = dict(env or {})
        self._scrub = tuple(scrub_env)

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
        if pattern.startswith("/") or ".." in pattern.split("/"):
            raise ValueError(f"path escapes the sandbox: {pattern!r}")

        def _ls() -> list[Entry]:
            entries = []
            for match in sorted(self.root.glob(pattern)):
                if match.is_file() and self.root in match.resolve().parents:
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
        inherited = {k: v for k, v in _environment(self._env).items() if k not in self._scrub}
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self._path(cwd) if cwd else self.root,
            env={**inherited, **(env or {})},
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

    async def sync(self) -> ExecResult:
        """The directory is already durable; nothing to push."""
        return ExecResult(0, "", "")

    async def endpoint(self, *, refresh: bool = False) -> Endpoint | None:
        """The host's fixed loopback URL and token; nothing to refresh."""
        del refresh
        return self._endpoint

    async def close(self) -> None:
        """Stop the service host, if any. The directory is the durable root."""
        process = self.host_process
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 10)
        except TimeoutError:
            process.kill()
            await process.wait()


async def start_host(
    spec: SandboxSpec, root: Path, images: ImageBucket | None = None
) -> tuple[Endpoint, asyncio.subprocess.Process]:
    """Launch the host for ``spec.services`` in ``root`` on a free port; return once it listens.
    With ``images``, the host uploads returned images under the thread ``root`` names."""
    assert spec.services
    token = secrets.token_urlsafe(32)
    config = EntryConfig(
        host=HostConfig(
            services=dict(spec.services),
            port=0,
            bind="127.0.0.1",
            scrub=list(spec.scrub_env),
            images=None if images is None else host.image_upload_config(images, spec, root.name),
        )
    )
    argv = [sys.executable, "-m", ENTRY_MODULE, config.model_dump_json()]
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=root,
        env={**_environment(spec.env), host.TOKEN_ENV: token},
        stdout=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        line = await asyncio.wait_for(process.stdout.readline(), HOST_START_TIMEOUT_S)
    except TimeoutError:
        line = b""
    prefix, _, port = line.decode().strip().partition(" ")
    if prefix != host.READY_PREFIX:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()
        raise RuntimeError(
            f"service host for {dict(spec.services)} did not start (exit {process.returncode})"
        )
    # Keep reading: a service method that prints would otherwise fill the pipe and stall the host.
    _drains.add(task := asyncio.create_task(_forward(process.stdout)))
    task.add_done_callback(_drains.discard)
    endpoint = Endpoint(f"http://127.0.0.1:{port}", {Header.AUTHORIZATION: f"Bearer {token}"})
    return endpoint, process


_drains: set[asyncio.Task[None]] = set()


async def _forward(stream: asyncio.StreamReader) -> None:
    while line := await stream.readline():
        sys.stdout.write(line.decode(errors="replace"))


class LocalSandboxProvider:
    """Sandboxes under ``spec.mount`` (or ``root``, or a temp dir), one directory per thread.

    Service hosts live as long as this provider's process; ``attach`` reuses a
    running one and restarts a dead one. ``images`` makes hosts return durable asset references.
    """

    def __init__(self, root: Path | None = None, *, images: ImageBucket | None = None) -> None:
        self.root = root
        self.images = images
        self._live: dict[Path, LocalSandbox] = {}

    async def open(self, spec: SandboxSpec, *, agent_id: str, thread_id: str) -> Sandbox:
        del agent_id
        base = Path(spec.mount) if spec.mount else (self.root or Path(tempfile.mkdtemp("-actant")))
        root = base / thread_id
        root.mkdir(parents=True, exist_ok=True)
        return await self._sandbox(spec, root)

    async def attach(self, spec: SandboxSpec, sandbox_id: str) -> Sandbox:
        root = Path(sandbox_id)
        if not root.is_dir():
            raise KeyError(sandbox_id)
        return await self._sandbox(spec, root)

    async def _sandbox(self, spec: SandboxSpec, root: Path) -> LocalSandbox:
        root = root.resolve()
        live = self._live.get(root)
        if live and live.host_process and live.host_process.returncode is None:
            return live
        if not spec.services:
            return LocalSandbox(root, spec.env, spec.scrub_env)
        endpoint, process = await start_host(spec, root, self.images)
        sandbox = LocalSandbox(
            root, spec.env, spec.scrub_env, endpoint=endpoint, host_process=process
        )
        self._live[root] = sandbox
        return sandbox
