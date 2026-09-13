"""The Modal backend: a ``modal.Sandbox`` whose files are backed by a bucket prefix.

Files live in the product's object storage (S3, R2, or MinIO in development)
under a per-thread prefix, in one of two ways (``SandboxSpec.storage``):

``"mount"``
    The prefix is mounted with ``CloudBucketMount``. Whole-file writes only (no
    append, rename or seek). The bucket keys go to the mount, never the process.
``"disk_sync"``
    Files live on the container's own disk under :data:`DISK_PATH`, with normal
    POSIX semantics. ``open`` restores the prefix onto the disk with s5cmd and
    :meth:`ModalSandbox.sync` (or :func:`sync_command` run after each tool call)
    pushes it back, deleting remote files removed locally. The tradeoff: s5cmd
    runs in the container, so the bucket keys are in the sandbox's environment;
    list them in ``scrub_env`` so agent-run code does not see them. The image
    needs s5cmd (:func:`with_s5cmd`).

Either way a dead sandbox reopens on the same prefix. ``secret_name`` names a
Modal secret holding ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY`` (plus
``AWS_REGION``: ``auto`` for R2, ``us-east-1`` for MinIO), or pass them inline as
``bucket_env``. The endpoint comes from ``endpoint_url`` (any https URL, a
cloudflared tunnel included); s5cmd uses path-style addressing for any custom
endpoint, which MinIO needs.

Requires the ``modal`` extra. Imported lazily so the package stays importable
without it.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from actant.sandbox.base import Entry, ExecResult, Sandbox, SandboxSpec

MOUNT_PATH = "/mnt/sandbox"
DISK_PATH = "/root/sandbox"
THREAD_TAG = "actant_thread"
SYNC_TIMEOUT_S = 1800
#: How long ``close`` lets the final push run before terminating anyway.
CLOSE_SYNC_TIMEOUT_S = 300
#: Tool-host request files (:data:`actant.sandbox.remote.REQUESTS_DIR`): never synced.
SYNC_EXCLUDE = ".actant/*"
S5CMD_VERSION = "2.3.0"
S5CMD_URL = (
    f"https://github.com/peak/s5cmd/releases/download/v{S5CMD_VERSION}/"
    f"s5cmd_{S5CMD_VERSION}_Linux-64bit.tar.gz"
)

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

    ``secret_name`` is a Modal secret holding the bucket's access keys. With
    ``storage="mount"`` the mount consumes it and the process never sees it;
    with ``"disk_sync"`` it is in the sandbox environment for s5cmd.
    """

    app_name: str
    bucket: str
    key_prefix: str = "sandboxes/"
    endpoint_url: str | None = None
    secret_name: str | None = None
    #: Inline bucket credentials (same keys as ``secret_name``), sent with
    #: ``modal.Secret.from_dict`` so a test needs no persisted Modal secret.
    #: ``AWS_REGION`` defaults to ``us-east-1``. Takes precedence over ``secret_name``.
    bucket_env: Mapping[str, str] | None = None
    client: Any = None

    async def open(self, spec: SandboxSpec, *, agent_id: str, thread_id: str) -> Sandbox:
        del agent_id
        modal = importlib.import_module("modal")
        app = await modal.App.lookup.aio(self.app_name, create_if_missing=True, client=self.client)
        if self.bucket_env is not None:
            bucket_secret = modal.Secret.from_dict({"AWS_REGION": "us-east-1", **self.bucket_env})
        elif self.secret_name:
            bucket_secret = modal.Secret.from_name(self.secret_name)
        else:
            bucket_secret = None
        secrets = [modal.Secret.from_name(name) for name in spec.secrets]
        if spec.storage == "disk_sync":
            root, volumes = DISK_PATH, {}
            if bucket_secret is not None:
                secrets.append(bucket_secret)
        elif spec.storage == "mount":
            root = MOUNT_PATH
            volumes = {
                MOUNT_PATH: modal.CloudBucketMount(
                    self.bucket,
                    key_prefix=f"{self.key_prefix}{thread_id}/",
                    bucket_endpoint_url=self.endpoint_url,
                    secret=bucket_secret,
                )
            }
        else:
            raise ValueError(f"unknown sandbox storage: {spec.storage!r}")
        sandbox = await modal.Sandbox.create.aio(
            app=app,
            image=spec.image,
            cpu=spec.cpu,
            memory=spec.memory_mb,
            gpu=spec.gpu,
            timeout=spec.timeout_s,
            idle_timeout=spec.idle_timeout_s,
            block_network=not spec.network,
            secrets=secrets,
            # Not scrubbed here: the tool server needs these. In-sandbox code reads
            # ACTANT_SCRUB_ENV and removes them from what the agent's own code runs.
            env={"ACTANT_SCRUB_ENV": ",".join(spec.scrub_env)} if spec.scrub_env else None,
            # ``attach`` gets no thread id; the tag carries it for ``sync``.
            tags={THREAD_TAG: thread_id},
            volumes=volumes,
            workdir=root,
            client=self.client,
        )
        if spec.storage == "mount":
            return ModalSandbox(sandbox, spec.env, scrub_env=spec.scrub_env)
        handle = ModalSandbox(
            sandbox,
            spec.env,
            root=DISK_PATH,
            sync_argv=shlex.split(sync_command(self, thread_id)),
            scrub_env=spec.scrub_env,
        )
        restore = await handle.exec(
            shlex.split(restore_command(self, thread_id)), timeout=SYNC_TIMEOUT_S, keep_env=True
        )
        # A new thread has an empty prefix, which s5cmd reports as an error.
        if restore.returncode != 0 and "no object found" not in restore.stderr:
            with contextlib.suppress(Exception):
                await handle.close()
            raise RuntimeError(f"restoring {self._remote(thread_id)} failed: {restore.stderr}")
        return handle

    async def attach(self, spec: SandboxSpec, sandbox_id: str) -> Sandbox:
        modal = importlib.import_module("modal")
        try:
            sandbox = await modal.Sandbox.from_id.aio(sandbox_id, client=self.client)
        except Exception as exc:  # noqa: BLE001 -- any lookup failure means "gone"
            raise KeyError(sandbox_id) from exc
        if await sandbox.poll.aio() is not None:
            raise KeyError(sandbox_id)
        if spec.storage != "disk_sync":
            return ModalSandbox(sandbox, spec.env, scrub_env=spec.scrub_env)
        # A live sandbox still has its disk: nothing to restore.
        thread_id = (await sandbox.get_tags.aio())[THREAD_TAG]
        return ModalSandbox(
            sandbox,
            spec.env,
            root=DISK_PATH,
            sync_argv=shlex.split(sync_command(self, thread_id)),
            scrub_env=spec.scrub_env,
        )

    def _remote(self, thread_id: str) -> str:
        return f"s3://{self.bucket}/{self.key_prefix}{thread_id}/"

    def _s5cmd(self, *args: str) -> list[str]:
        endpoint = ["--endpoint-url", self.endpoint_url] if self.endpoint_url else []
        return ["s5cmd", *endpoint, *args]


def sync_command(provider: ModalSandboxProvider, thread_id: str) -> str:
    """The shell command that pushes a disk_sync sandbox's disk to its bucket prefix.

    Mirrors exactly: remote files removed locally are deleted. A product hands
    this to the host server's ``--after`` hook to push after every tool call.
    """
    return shlex.join(
        provider._s5cmd(
            "sync",
            "--delete",
            "--exclude",
            SYNC_EXCLUDE,
            f"{DISK_PATH}/",
            provider._remote(thread_id),
        )
    )


def restore_command(provider: ModalSandboxProvider, thread_id: str) -> str:
    """The shell command that pulls a thread's bucket prefix onto the disk (never deletes)."""
    return shlex.join(
        provider._s5cmd(
            "sync", "--exclude", SYNC_EXCLUDE, f"{provider._remote(thread_id)}*", f"{DISK_PATH}/"
        )
    )


def with_s5cmd(image: Any) -> Any:
    """``image`` with the pinned s5cmd binary in ``/usr/local/bin`` (``disk_sync`` needs it)."""
    return image.run_commands(
        f"curl -fsSL {S5CMD_URL} | tar -xz -C /usr/local/bin s5cmd", "s5cmd version"
    )


class ModalSandbox:
    def __init__(
        self,
        sandbox: Any,
        env: Mapping[str, str],
        *,
        root: str = MOUNT_PATH,
        sync_argv: Sequence[str] | None = None,
        scrub_env: Sequence[str] = (),
    ) -> None:
        self._sandbox = sandbox
        self._env = dict(env)
        self._scrub = tuple(scrub_env)
        self._root = root
        self._sync_argv = list(sync_argv) if sync_argv else None
        self.id = str(sandbox.object_id)

    async def sync(self) -> ExecResult:
        """Push the disk to the bucket prefix (``disk_sync``). A no-op for a mount."""
        if self._sync_argv is None:
            return ExecResult(0, "", "")
        # s5cmd needs the bucket keys that scrub_env usually lists.
        return await self.exec(self._sync_argv, timeout=SYNC_TIMEOUT_S, keep_env=True)

    @staticmethod
    def _check(path: str) -> str:
        if path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"path escapes the sandbox: {path!r}")
        return path

    async def read(self, path: str) -> bytes:
        return await self._sandbox.filesystem.read_bytes.aio(f"{self._root}/{self._check(path)}")

    async def write(self, path: str, data: bytes) -> None:
        target = f"{self._root}/{self._check(path)}"
        await self._sandbox.filesystem.make_directory.aio(target.rpartition("/")[0])
        await self._sandbox.filesystem.write_bytes.aio(data, target)

    async def ls(self, pattern: str) -> list[Entry]:
        pattern = self._check(pattern)
        result = await self.exec(["python", "-c", _LS, self._root, pattern], timeout=60)
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
        keep_env: bool = False,
    ) -> ExecResult:
        workdir = f"{self._root}/{self._check(cwd)}" if cwd else self._root
        # Secrets are in the container's environment, and Modal's ``env`` can only
        # add variables, so ``env -u`` removes the scrubbed ones in the container.
        unset = (
            [] if keep_env else [f"-u{name}" for name in self._scrub if name not in (env or {})]
        )
        # ``timeout`` runs inside the container, so the process group dies
        # there; 124 is its exit code, the same one the local backend uses.
        process = await self._sandbox.exec.aio(
            *(["env", *unset] if unset else []),
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
        """Push a ``disk_sync`` disk (best effort, bounded), then terminate.

        Without the push, whatever the last ``after`` hook missed would die with the disk.
        """
        if self._sync_argv is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    self.exec(self._sync_argv, timeout=CLOSE_SYNC_TIMEOUT_S, keep_env=True),
                    CLOSE_SYNC_TIMEOUT_S + 30,
                )
        # ``terminate`` only requests the stop; wait so ``attach`` sees it finished.
        await self._sandbox.terminate.aio()
        # ``wait`` raises for a sandbox that ended by timeout; that is still closed.
        with contextlib.suppress(Exception):
            await self._sandbox.wait.aio(raise_on_termination=False)

    def __repr__(self) -> str:
        return f"ModalSandbox({shlex.quote(self.id)})"
