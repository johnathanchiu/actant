"""The Modal backend: a network-blocked ``modal.Sandbox`` over a mounted bucket prefix.

Files live in the product's object storage (S3, R2, or MinIO in development),
mounted into the container with ``CloudBucketMount`` under a per-thread
prefix. Nothing is stored on Modal: a dead sandbox reopens on the same
prefix, and the worker reads results back through its own blob client.

Requires the ``modal`` extra. Imported lazily so the package stays importable
without it.
"""

from __future__ import annotations

import importlib
import json
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from actant.sandbox.base import Entry, ExecResult, Sandbox, SandboxSpec

MOUNT_PATH = "/mnt/sandbox"

_LS = (
    "import glob, json, os, sys\n"
    "root, pattern = sys.argv[1], sys.argv[2]\n"
    "out = []\n"
    "for p in sorted(glob.glob(os.path.join(root, pattern), recursive=True)):\n"
    "    if os.path.isfile(p):\n"
    "        s = os.stat(p)\n"
    "        out.append([os.path.relpath(p, root), s.st_size, s.st_mtime])\n"
    "print(json.dumps(out))\n"
)


@dataclass
class ModalSandboxProvider:
    """``app_name`` groups the sandboxes on Modal; the bucket settings describe the mount.

    ``secret_name`` is a Modal secret holding the bucket's access keys, the
    only credential the container ever sees, and it is consumed by the mount
    rather than exposed to the process.
    """

    app_name: str
    bucket: str
    key_prefix: str = "sandboxes/"
    endpoint_url: str | None = None
    secret_name: str | None = None
    client: Any = None

    async def open(self, spec: SandboxSpec, *, agent_id: str, thread_id: str) -> Sandbox:
        del agent_id
        modal = importlib.import_module("modal")
        app = await modal.App.lookup.aio(self.app_name, create_if_missing=True, client=self.client)
        mount = modal.CloudBucketMount(
            self.bucket,
            key_prefix=f"{self.key_prefix}{thread_id}/",
            bucket_endpoint_url=self.endpoint_url,
            secret=(modal.Secret.from_name(self.secret_name) if self.secret_name else None),
        )
        sandbox = await modal.Sandbox.create.aio(
            app=app,
            image=spec.image,
            cpu=spec.cpu,
            memory=spec.memory_mb,
            timeout=spec.timeout_s,
            idle_timeout=spec.idle_timeout_s,
            block_network=True,
            volumes={MOUNT_PATH: mount},
            workdir=MOUNT_PATH,
            client=self.client,
        )
        return ModalSandbox(sandbox, spec.env)

    async def attach(self, spec: SandboxSpec, sandbox_id: str) -> Sandbox:
        modal = importlib.import_module("modal")
        try:
            sandbox = await modal.Sandbox.from_id.aio(sandbox_id, client=self.client)
        except Exception as exc:  # noqa: BLE001 -- any lookup failure means "gone"
            raise KeyError(sandbox_id) from exc
        if await sandbox.poll.aio() is not None:
            raise KeyError(sandbox_id)
        return ModalSandbox(sandbox, spec.env)


class ModalSandbox:
    def __init__(self, sandbox: Any, env: Mapping[str, str]) -> None:
        self._sandbox = sandbox
        self._env = dict(env)
        self.id = str(sandbox.object_id)

    @staticmethod
    def _check(path: str) -> str:
        if path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"path escapes the sandbox: {path!r}")
        return path

    async def read(self, path: str) -> bytes:
        return await self._sandbox.filesystem.read_bytes.aio(f"{MOUNT_PATH}/{self._check(path)}")

    async def write(self, path: str, data: bytes) -> None:
        target = f"{MOUNT_PATH}/{self._check(path)}"
        await self._sandbox.filesystem.make_directory.aio(target.rpartition("/")[0])
        await self._sandbox.filesystem.write_bytes.aio(data, target)

    async def ls(self, pattern: str) -> list[Entry]:
        result = await self.exec(["python", "-c", _LS, MOUNT_PATH, pattern], timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"ls failed: {result.stderr}")
        return [Entry(p, s, m) for p, s, m in json.loads(result.stdout or "[]")]

    async def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        timeout: float,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        workdir = f"{MOUNT_PATH}/{self._check(cwd)}" if cwd else MOUNT_PATH
        # ``timeout`` runs inside the container, so the process group dies
        # there; 124 is its exit code, the same one the local backend uses.
        process = await self._sandbox.exec.aio(
            "timeout",
            str(int(timeout)),
            *argv,
            workdir=workdir,
            env={**self._env, **(env or {})},
        )
        stdout, stderr = await process.stdout.read.aio(), await process.stderr.read.aio()
        code = await process.wait.aio()
        return ExecResult(code, stdout, stderr, timed_out=code == 124)

    async def close(self) -> None:
        # ``terminate`` only requests the stop; wait so ``attach`` sees it finished.
        await self._sandbox.terminate.aio()
        await self._sandbox.wait.aio(raise_on_termination=False)

    def __repr__(self) -> str:
        return f"ModalSandbox({shlex.quote(self.id)})"
