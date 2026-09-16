"""The Modal backend: a ``modal.Sandbox`` whose files are backed by a bucket prefix.

Files live in the product's object storage (S3, R2, or MinIO in development)
under a per-thread prefix, in one of two ways (``SandboxSpec.storage``):

``"mount"``
    The prefix is mounted with ``CloudBucketMount``. Whole-file writes only (no
    append, rename or seek). The bucket keys go to the mount, never the process.
``"disk_sync"``
    Files live on the container's own disk under :data:`DISK_PATH`, with normal
    POSIX semantics. ``open`` restores the prefix onto the disk with s5cmd (an empty
    prefix restores ``SandboxSpec.seed`` instead, copied into the prefix at the same
    time and then marked done) and :meth:`ModalSandbox.sync` pushes it back, deleting
    remote files removed locally; a service host also pushes after its calls and every
    ``sync_interval_s``, each push bounded by ``sync_timeout_s``, and ``close``
    pushes once more. Restored files get their objects' mtimes, so a push
    uploads only what changed. Images a service returns are uploaded at once under
    ``image_prefix`` and returned as durable asset references when
    ``SandboxSpec.upload_images`` is enabled. The tradeoff: s5cmd
    runs in the container, so the bucket keys are in the sandbox's environment;
    list them in ``scrub_env`` so agent-run code does not see them. The image
    needs s5cmd (:func:`with_s5cmd`).

Either way a dead sandbox reopens on the same prefix. ``secret_name`` names a
Modal secret holding ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY`` (plus
``AWS_REGION``: ``auto`` for R2, ``us-east-1`` for MinIO), or pass them inline as
``bucket_env``. The endpoint comes from ``endpoint_url`` (any https URL, a
cloudflared tunnel included); s5cmd uses path-style addressing for any custom
endpoint, which MinIO needs.

A spec's services are served by :mod:`actant.sandbox.host`, launched as the
sandbox entrypoint through :mod:`actant.sandbox.entry` (which restores
``disk_sync`` storage first). Readiness is a TCP probe on the host's port: the
host binds only after the restore and the service imports. The endpoint is a
Modal connect token for that port: Modal's proxy authenticates each request and
no port is exposed publicly. The token is minted with the worker's Modal client
on first use, cached on the handle, and re-minted only when the proxy rejects it,
so nothing secret is persisted. The provider keeps the handles it made, so the
per-call ``attach`` is one ``poll``. The image must have actant and
the services' package installed.

Requires the ``modal`` extra. Imported lazily so the package stays importable
without it.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any
from urllib.parse import urlsplit

import actant.sandbox.entry as entry
import actant.sandbox.host as host
from actant.sandbox.base import (
    Endpoint,
    Entry,
    ExecResult,
    ImageBucket,
    Sandbox,
    SandboxSpec,
    Storage,
)
from actant.sandbox.protocol import (
    EntryConfig,
    Header,
    HostConfig,
    PushConfig,
    RestoreConfig,
    Route,
    SeedConfig,
    StampConfig,
)

MOUNT_PATH = "/mnt/sandbox"
DISK_PATH = "/root/sandbox"
THREAD_TAG = "actant_thread"
_log = logging.getLogger(__name__)
#: How long the restore (and the readiness wait around it) may take.
SYNC_TIMEOUT_S = 1800
#: Slack over a command's own timeout for Modal's API round trips.
API_SLACK_S = 60
S5CMD_VERSION = "2.3.0"
#: Appended to a thread's key (not inside its prefix) for its seed marker object.
SEED_MARKER_SUFFIX = ".actant-seeded"
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
    """``app_name`` groups the sandboxes on Modal; the bucket settings describe storage.

    ``secret_name`` is a Modal secret holding the bucket's access keys. With
    ``storage="mount"`` the mount consumes it and the process never sees it;
    with ``"disk_sync"`` it is in the sandbox environment for s5cmd.
    """

    app_name: str
    bucket: str
    key_prefix: str = "sandboxes/"
    endpoint_url: str | None = None
    #: Where a ``disk_sync`` host uploads returned images: ``<image_prefix><thread>/``.
    image_prefix: str = "actant-images/"
    secret_name: str | None = None
    #: Inline bucket credentials (same keys as ``secret_name``), sent with
    #: ``modal.Secret.from_dict`` so a test needs no persisted Modal secret.
    #: ``AWS_REGION`` defaults to ``us-east-1``. Takes precedence over ``secret_name``.
    bucket_env: Mapping[str, str] | None = None
    client: Any = None
    # ponytail: handles of closed sandboxes stay until their id is attached again
    _live: dict[str, tuple[Any, ModalSandbox]] = field(
        default_factory=dict, init=False, repr=False
    )

    async def open(self, spec: SandboxSpec, *, agent_id: str, thread_id: str) -> Sandbox:
        del agent_id
        # Before any Modal call: a missing public endpoint fails here, not in a container.
        config = self.entry_config(spec, thread_id)
        modal = importlib.import_module("modal")
        app = await modal.App.lookup.aio(self.app_name, create_if_missing=True, client=self.client)
        if self.bucket_env is not None:
            bucket_secret = modal.Secret.from_dict({"AWS_REGION": "us-east-1", **self.bucket_env})
        elif self.secret_name:
            bucket_secret = modal.Secret.from_name(self.secret_name)
        else:
            bucket_secret = None
        secrets = [modal.Secret.from_name(name) for name in spec.secrets]
        disk_sync = spec.storage == Storage.DISK_SYNC
        if disk_sync:
            root, volumes = DISK_PATH, {}
            if bucket_secret is not None:
                secrets.append(bucket_secret)
        else:  # Storage.MOUNT; SandboxSpec rejects anything else
            root = MOUNT_PATH
            volumes = {
                MOUNT_PATH: modal.CloudBucketMount(
                    self.bucket,
                    key_prefix=f"{self.key_prefix}{thread_id}/",
                    bucket_endpoint_url=self.endpoint_url,
                    secret=bucket_secret,
                )
            }
        command: list[str] = []
        probe = None
        if disk_sync or spec.services:
            command = ["python", "-m", entry.__name__, config.model_dump_json()]
            if disk_sync:
                probe = modal.Probe.with_exec("test", "-f", entry.READY_FILE)
            if spec.services:
                probe = modal.Probe.with_tcp(spec.service_port)
        sandbox = await modal.Sandbox.create.aio(
            *command,
            app=app,
            image=spec.image,
            cpu=spec.cpu,
            memory=spec.memory_mb,
            gpu=spec.gpu,
            timeout=spec.timeout_s,
            idle_timeout=spec.idle_timeout_s,
            **self._network(spec),
            # The host and s5cmd need these; ``exec`` removes ``scrub_env`` per command.
            secrets=secrets,
            env=dict(spec.env) or None,
            # ``attach`` gets no thread id; the tag carries it for ``sync``.
            tags={THREAD_TAG: thread_id},
            volumes=volumes,
            workdir=root,
            readiness_probe=probe,
            client=self.client,
        )
        if probe is not None:
            try:
                await sandbox.wait_until_ready.aio(timeout=SYNC_TIMEOUT_S)
            except Exception as exc:
                detail = ""
                if await sandbox.poll.aio() is not None:  # exited: its stderr is complete
                    detail = (await sandbox.stderr.read.aio())[-4000:]
                with contextlib.suppress(Exception):
                    await sandbox.terminate.aio()
                raise RuntimeError(
                    f"sandbox for thread {thread_id} never became ready: {exc}\n{detail}"
                ) from exc
        return self._handle(sandbox, spec, thread_id)

    async def attach(self, spec: SandboxSpec, sandbox_id: str) -> Sandbox:
        live = self._live.get(sandbox_id)
        if live is None:
            modal = importlib.import_module("modal")
            try:
                sandbox = await modal.Sandbox.from_id.aio(sandbox_id, client=self.client)
            except Exception as exc:  # noqa: BLE001 -- any lookup failure means "gone"
                raise KeyError(sandbox_id) from exc
            if await sandbox.poll.aio() is not None:
                raise KeyError(sandbox_id)
            # A live sandbox still has its disk: nothing to restore.
            thread_id = (await sandbox.get_tags.aio())[THREAD_TAG]
            return self._handle(sandbox, spec, thread_id)
        sandbox, handle = live
        if await sandbox.poll.aio() is not None:
            del self._live[sandbox_id]
            raise KeyError(sandbox_id)
        return handle

    def _handle(self, sandbox: Any, spec: SandboxSpec, thread_id: str) -> ModalSandbox:
        disk_sync = spec.storage == Storage.DISK_SYNC
        handle = ModalSandbox(
            sandbox,
            spec.env,
            root=DISK_PATH if disk_sync else MOUNT_PATH,
            sync_argv=self.sync_argv(thread_id) if disk_sync else None,
            sync_timeout_s=spec.sync_timeout_s,
            scrub_env=spec.scrub_env,
            service_port=spec.service_port if spec.services else None,
        )
        self._live[handle.id] = (sandbox, handle)
        return handle

    def _network(self, spec: SandboxSpec) -> dict[str, list[str]]:
        """Outbound allowlists for ``spec.network=False``.

        Not ``block_network``: that also cuts the inbound connect-token proxy and
        rejects a TCP probe. An empty allowlist denies all outbound traffic; a
        ``disk_sync`` sandbox may still reach its bucket endpoint.
        """
        if spec.network:
            return {}
        if spec.storage != Storage.DISK_SYNC:
            return {"outbound_cidr_allowlist": []}
        bucket_host = urlsplit(self.endpoint_url or "").hostname
        if bucket_host is None:
            raise ValueError(
                "disk_sync without network needs endpoint_url: s5cmd may reach only that host"
            )
        return {"outbound_domain_allowlist": [bucket_host]}

    def entry_config(self, spec: SandboxSpec, thread_id: str) -> EntryConfig:
        """The entrypoint's config: restore a ``disk_sync`` disk, then serve the services."""
        disk_sync = spec.storage == Storage.DISK_SYNC
        restore = push = None
        if disk_sync:
            restore = RestoreConfig(
                argv=self.restore_argv(thread_id),
                timeout_s=SYNC_TIMEOUT_S,
                stamp=self.stamp_config(thread_id),
                seed=None if spec.seed is None else self.seed_config(thread_id, spec.seed),
            )
            push = PushConfig(
                argv=self.sync_argv(thread_id),
                interval_s=spec.sync_interval_s,
                timeout_s=spec.sync_timeout_s,
            )
        served = None
        if spec.services:
            served = HostConfig(
                services=dict(spec.services),
                port=spec.service_port,
                bind="0.0.0.0",
                scrub=list(spec.scrub_env),
                push=push,
                images=host.image_upload_config(self.image_bucket(), spec, thread_id)
                if disk_sync and spec.upload_images
                else None,
            )
        return EntryConfig(restore=restore, host=served)

    def image_bucket(self) -> ImageBucket:
        """Where a ``disk_sync`` host uploads the images its services return."""
        return ImageBucket(self.bucket, self.image_prefix, self.endpoint_url)

    def _remote(self, thread_id: str) -> str:
        return f"s3://{self.bucket}/{self.key_prefix}{thread_id}/"

    def _s5cmd(self, *args: str) -> list[str]:
        endpoint = ["--endpoint-url", self.endpoint_url] if self.endpoint_url else []
        return ["s5cmd", *endpoint, *args]

    def sync_argv(self, thread_id: str) -> list[str]:
        """Push a ``disk_sync`` disk to its prefix, mirroring: remote files removed locally go."""
        return self._s5cmd("sync", "--delete", f"{DISK_PATH}/", self._remote(thread_id))

    def restore_argv(self, thread_id: str) -> list[str]:
        """Pull a thread's prefix onto the disk (never deletes)."""
        return self._pull_argv(self._remote(thread_id))

    def stamp_config(self, thread_id: str) -> StampConfig:
        """List a thread's prefix alongside the pull, so pulled files keep their objects' mtimes."""
        return self._stamp_config(self._remote(thread_id))

    def seed_config(self, thread_id: str, seed: str) -> SeedConfig:
        """Pull ``seed`` onto a new thread's disk while copying it into the thread's prefix."""
        source = f"s3://{self.bucket}/{seed}"
        marker = self.seed_marker(thread_id)
        return SeedConfig(
            argv=self._pull_argv(source),
            copy_argv=self._s5cmd("cp", f"{source}*", self._remote(thread_id)),
            write_marker_argv=self._s5cmd("pipe", marker),
            check_marker_argv=self._s5cmd("ls", marker),
            stamp=self._stamp_config(source),
        )

    def seed_marker(self, thread_id: str) -> str:
        """The object that says a thread's seed copy finished: a sibling of its prefix
        (``sandboxes/t1`` + :data:`SEED_MARKER_SUFFIX`), so no pull or push touches it."""
        return f"s3://{self.bucket}/{self.key_prefix}{thread_id}{SEED_MARKER_SUFFIX}"

    def _pull_argv(self, source: str) -> list[str]:
        return self._s5cmd("sync", f"{source}*", f"{DISK_PATH}/")

    def _stamp_config(self, source: str) -> StampConfig:
        argv = ["s5cmd", "--json", *self._s5cmd("ls", f"{source}*")[1:]]
        return StampConfig(argv=argv, prefix=source, root=DISK_PATH)


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
        service_port: int | None = None,
        sync_timeout_s: float = 300.0,
    ) -> None:
        self._sandbox = sandbox
        self._sync_timeout = sync_timeout_s
        self._env = dict(env)
        self._scrub = tuple(scrub_env)
        self._root = root
        self._sync_argv = list(sync_argv) if sync_argv else None
        self.id = str(sandbox.object_id)
        self._service_port = service_port
        self._endpoint: Endpoint | None = None

    async def endpoint(self, *, refresh: bool = False) -> Endpoint | None:
        """A connect token for the service port, minted once and again on ``refresh``.

        Modal documents no expiry; a token is re-minted only after the proxy answers 401.
        """
        if self._service_port is None:
            return None
        if self._endpoint is None or refresh:
            creds = await self._sandbox.create_connect_token.aio(port=self._service_port)
            self._endpoint = Endpoint(creds.url, {Header.AUTHORIZATION: f"Bearer {creds.token}"})
        return self._endpoint

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
    ) -> ExecResult:
        # The SDK drops ``None`` values from ``env`` rather than unsetting them, and
        # secrets are container-wide, so ``env -u`` (argv, no shell) removes them here.
        unset = [f"-u{name}" for name in self._scrub if name not in (env or {})]
        prefix = ["env", *unset] if unset else []
        return await self._run([*prefix, *argv], cwd=cwd, timeout=timeout, env=env)

    async def _run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        timeout: float,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        workdir = f"{self._root}/{self._check(cwd)}" if cwd else self._root
        # ``timeout`` runs inside the container, so the process group dies
        # there; 124 is its exit code, the same one the local backend uses.
        process = await self._sandbox.exec.aio(
            "timeout", str(int(timeout)), *argv, workdir=workdir, env={**self._env, **(env or {})}
        )
        stdout, stderr = await process.stdout.read.aio(), await process.stderr.read.aio()
        code = await process.wait.aio()
        return ExecResult(code, stdout, stderr, timed_out=code == 124)

    async def sync(self) -> ExecResult:
        """Push the disk to the bucket prefix (``disk_sync``). A no-op for a mount."""
        if self._sync_argv is None:
            return ExecResult(0, "", "")
        # Unscrubbed: s5cmd needs the bucket keys that ``scrub_env`` usually lists.
        return await self._bounded_sync()

    async def _bounded_sync(self) -> ExecResult:
        """The push, killed in the container after ``sync_timeout_s``; a Modal API call that
        hangs past that plus :data:`API_SLACK_S` is reported as timed out, never awaited."""
        assert self._sync_argv is not None
        try:
            return await asyncio.wait_for(
                self._run(self._sync_argv, timeout=self._sync_timeout),
                self._sync_timeout + API_SLACK_S,
            )
        except TimeoutError:
            return ExecResult(124, "", "sync did not return in time\n", timed_out=True)

    async def close(self) -> None:
        """Stop the service host gracefully (its instances close and it pushes a ``disk_sync``
        disk), or push the disk here when there is no host; then terminate.

        Best effort, bounded, and never raises. Modal's ``terminate``, ``timeout`` and ``idle_timeout``
        kill the container outright, so this is the only path on which ``close`` runs.
        """
        if not await self._stop_host() and self._sync_argv is not None:
            try:
                pushed = await self._bounded_sync()
                if pushed.returncode:
                    tail = pushed.stderr[-500:]
                    _log.warning("%r: final push exited %d: %s", self, pushed.returncode, tail)
            except Exception:  # noqa: BLE001 -- close never raises
                _log.warning("%r: final push failed", self, exc_info=True)
        # ``terminate`` only requests the stop; wait so ``attach`` sees it finished.
        # ``wait`` raises for a sandbox that ended by timeout; that is still closed.
        try:
            await asyncio.wait_for(self._sandbox.terminate.aio(), API_SLACK_S)
        except Exception:  # noqa: BLE001 -- close never raises
            _log.warning("%r: terminate failed", self, exc_info=True)
            return
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._sandbox.wait.aio(raise_on_termination=False), API_SLACK_S)

    async def _stop_host(self) -> bool:
        """``POST /v1/shutdown``; whether the host confirmed it closed and pushed."""
        if self._service_port is None:
            return False
        with contextlib.suppress(Exception):
            for refresh in (False, True):
                endpoint = await asyncio.wait_for(self.endpoint(refresh=refresh), API_SLACK_S)
                assert endpoint is not None
                # The host's final push may wait for a running one: two timeouts.
                timeout = 2 * self._sync_timeout + API_SLACK_S
                status, _ = await asyncio.to_thread(
                    host.post, endpoint, Route.SHUTDOWN, b"{}", timeout
                )
                if status != HTTPStatus.UNAUTHORIZED:
                    return status == HTTPStatus.OK
        return False

    def __repr__(self) -> str:
        return f"ModalSandbox({self.id!r})"
