"""Sandboxes: where an agent's tools read, write and run code.

A sandbox is owned by a key (:mod:`actant.sandbox.registry`): a thread's own
id, or a key a product opened and the thread names. A tool that needs one
declares it (``needs_sandbox``, or a ``Sandbox`` parameter on a function tool)
and the runtime hands it the thread's sandbox through its ``CallContext``.
Tools that do not need one never see one.

The filesystem a sandbox exposes is its working directory. Every backend keys
it by the sandbox's key under the spec's ``mount``: a directory for the local
backend, a bucket prefix mounted into the container for cloud backends.
The worker holds no files; it reads results back through the same handle.

This module is a leaf: it imports nothing from the runtime, so agent
definitions and tools can name these types without cycles.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


@dataclass(frozen=True)
class ExecResult:
    """What a command left behind. ``timed_out`` implies ``returncode == 124``."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class Entry:
    """One file, as ``ls`` reports it: enough to tell a fresh file from an old one."""

    path: str
    size: int
    mtime: float


class Sandbox(Protocol):
    """A thread's working directory plus a way to run commands in it.

    Paths are relative to the sandbox root; implementations reject paths that
    escape it. ``write`` replaces the whole file: the backends that mount
    object storage cannot append or seek, so no tool should rely on either.

    ``exec`` removes the spec's ``scrub_env`` names from the command's environment
    (an explicit ``env`` entry still wins). Scrubbing is best effort, not a security
    boundary: code running as the same user can read another process's
    ``/proc/<pid>/environ`` or call the service host.

    """

    id: str

    async def read(self, path: str) -> bytes: ...

    async def write(self, path: str, data: bytes) -> None: ...

    async def ls(self, pattern: str) -> list[Entry]: ...

    async def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        timeout: float,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult: ...

    async def sync(self) -> ExecResult:
        """Push the sandbox's files to durable storage. A no-op where storage already is
        the filesystem (``mount``, ``local``). ``close`` also pushes, best effort, so
        calling this is only needed for a checkpoint mid-run."""
        ...

    async def endpoint(self, *, refresh: bool = False) -> Endpoint | None:
        """Where the spec's services are served, ``None`` when it has none.

        Backends that authenticate with short-lived credentials cache them per
        sandbox; ``refresh`` asks for new ones after the host rejected the old.
        """
        ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class Endpoint:
    """How to reach a sandbox's service host: a base URL plus the headers that authenticate."""

    url: str
    #: Kept out of ``repr``: they carry the bearer token.
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)


class Backend(StrEnum):
    """The backends actant ships. ``SandboxSpec.backend`` stays a ``str``: a product
    may register its own provider under any name."""

    LOCAL = "local"
    MODAL = "modal"


class Storage(StrEnum):
    #: The bucket prefix is the filesystem (whole-file writes only).
    MOUNT = "mount"
    #: A local disk, restored from the prefix on open and pushed back with
    #: :meth:`Sandbox.sync` -- ordinary file semantics, a few seconds behind the bucket.
    DISK_SYNC = "disk_sync"


@dataclass(frozen=True)
class Location:
    """A key prefix in a bucket: ``prefix`` is empty (the whole bucket) or ends in ``/``."""

    bucket: str
    prefix: str

    def __post_init__(self) -> None:
        if not self.bucket or (self.prefix and not self.prefix.endswith("/")):
            raise ValueError(f"a location is a bucket and a prefix ending in '/': {self!r}")

    @property
    def url(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"


def _check_path(path: str, what: str) -> None:
    if path and any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(f"a {what} path is relative and normalized: {path!r}")


def inside(path: str, parent: str) -> bool:
    """Whether ``path`` is ``parent`` or under it (both relative to the disk's root)."""
    return not parent or path == parent or path.startswith(f"{parent}/")


@dataclass(frozen=True)
class Restore:
    """One entry of a ``disk_sync`` restore plan: ``source`` pulled into ``path`` (relative
    to the disk's root, ``""`` for the root). Only the ``push`` entry is pushed back, and
    its push skips the other entries' paths."""

    source: Location
    path: str
    push: bool

    def __post_init__(self) -> None:
        _check_path(self.path, "restore")


@dataclass(frozen=True)
class Mount:
    """A read-only bucket prefix mounted at ``path`` (relative to the disk's root, never
    the root) of a ``disk_sync`` disk, with ``CloudBucketMount``: files are fetched when
    read, so opening costs nothing however large ``source`` is. No pull, push or mtime
    stamp touches ``path``."""

    source: Location
    path: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("a mount path is not the disk's root")
        _check_path(self.path, "mount")


@dataclass(frozen=True)
class SandboxSpec:
    """What an agent's tools need from their sandbox. Lives on the agent definition.

    ``backend`` names a provider registered on the worker. ``mount`` is the
    durable root the backend keys per thread: a directory for ``local``, a
    bucket location for cloud backends. ``image`` is backend-specific (a
    ``modal.Image`` for Modal) and ignored by ``local``.
    """

    backend: str = Backend.LOCAL
    mount: str | None = None
    image: object | None = None
    cpu: int = 2
    memory_mb: int = 8192
    timeout_s: int = 3600
    idle_timeout_s: int = 600
    env: Mapping[str, str] = field(default_factory=dict)
    #: A GPU type the backend understands (``"L4"``, ``"H100"``); ``None`` for CPU only.
    gpu: str | None = None
    #: Where the backend may place the sandbox: a region it understands (Modal's ``"us"``,
    #: ``"us-east"``) or several; ``None`` for anywhere. Modal's default also places
    #: sandboxes in Europe and Asia; a ``disk_sync`` sandbox belongs near its bucket.
    region: str | tuple[str, ...] | None = None
    #: Outbound network. Off by default: turn it on only for tools that call external APIs.
    network: bool = False
    #: Backend secret names (Modal secrets) injected into the sandbox's environment.
    secrets: tuple[str, ...] = ()
    #: Environment variables removed from commands the agent's own code runs, so a
    #: script never sees the service keys that ``secrets`` put in the sandbox.
    scrub_env: tuple[str, ...] = ()
    #: How a cloud backend keeps files; see :class:`Storage`.
    storage: Storage = Storage.MOUNT
    #: Name to ``"pkg.mod:Class"``, served by one service host inside the sandbox (see
    #: :mod:`actant.sandbox.service`). Fixed here, at launch; requests name a service
    #: and a method, never a module.
    services: Mapping[str, str] = field(default_factory=dict)
    #: The port the host listens on inside a container. ``local`` picks a free one.
    service_port: int = 8080
    #: ``disk_sync``: the service host pushes every this many seconds while calls have
    #: completed since its last successful push (and after calls).
    sync_interval_s: float = 60.0
    #: ``disk_sync``: a push running longer is killed; the next one still runs.
    #: :meth:`Sandbox.sync` and ``close`` are bounded by it too.
    sync_timeout_s: float = 300.0
    #: ``disk_sync``: how long the restore on open (and the readiness wait around it) may
    #: take. A tool that opens the sandbox needs at least this in ``ActivityTimeouts.tool_s``.
    restore_timeout_s: float = 1800.0
    #: ``disk_sync``: folders, relative to the pushed folder, that no push sends (``"renders"``,
    #: ``"cache/frames"``): files a run can make again, or ones restored from elsewhere.
    push_exclude: tuple[str, ...] = ()
    #: ``disk_sync``: a bucket key prefix (``"templates/base/"``) that starts a new thread.
    #: When the thread's prefix is empty, the sandbox pulls the seed while the bucket copies
    #: it into the thread's prefix; startup waits for both. A thread with files ignores it.
    seed: str | None = None
    #: ``disk_sync``: what the disk is made of, recorded with the sandbox: entries pulled at
    #: open, exactly one of them pushed back (the workspace; a ``seed`` fills it). Empty: the
    #: key's own prefix under the provider's ``key_prefix``, at the root.
    restore: tuple[Restore, ...] = ()
    #: ``disk_sync``: read-only prefixes mounted into the disk, for inputs too large to pull.
    #: A mount's path is inside no other mount's, and no restore entry's is at or under it.
    mounts: tuple[Mount, ...] = ()
    #: Upload returned images to configured storage and return durable references.
    #: False returns inline bytes; signing and retention belong to storage adapters.
    upload_images: bool = True
    #: Seconds one attempt to upload an image may take (its connect and read timeouts); boto
    #: retries it, and an image that never lands returns as inline bytes.
    image_upload_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        # Coerce (and validate) a plain string from an untyped config.
        object.__setattr__(self, "storage", Storage(self.storage))
        if self.sync_interval_s <= 0 or self.sync_timeout_s <= 0:
            raise ValueError("sync_interval_s and sync_timeout_s must be positive")
        if not (math.isfinite(self.restore_timeout_s) and self.restore_timeout_s > 0):
            raise ValueError("restore_timeout_s must be a positive, finite number")
        for folder in self.push_exclude:
            parts = folder.split("/")
            if not folder or folder.startswith("/") or any(p in ("", ".", "..") for p in parts):
                raise ValueError(f"push_exclude holds relative folders, not {folder!r}")
        if self.seed is not None and (
            self.storage != Storage.DISK_SYNC or not self.seed.endswith("/")
        ):
            raise ValueError("seed is a key prefix ending in '/', for disk_sync storage")
        timeout = self.image_upload_timeout_s
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not (math.isfinite(timeout) and timeout > 0)
        ):
            raise ValueError("image_upload_timeout_s must be a positive, finite number")


@dataclass(frozen=True)
class ImageBucket:
    """Storage destination for service images, separate from workspace sync."""

    bucket: str
    prefix: str = "actant-images/"
    endpoint_url: str | None = None

    def destination(self, thread_id: str) -> str:
        """The ``s3://`` prefix a thread's images upload to."""
        return f"s3://{self.bucket}/{self.prefix}{thread_id}/"


class SandboxProvider(Protocol):
    """Opens sandboxes for a backend and reattaches to ones that already exist."""

    async def open(self, spec: SandboxSpec, *, sandbox_id: str) -> Sandbox:
        """A new sandbox for ``sandbox_id`` (see :mod:`actant.sandbox.registry`): its files
        under the spec's ``mount`` by that id, unless the spec says where they are."""
        ...

    async def attach(self, spec: SandboxSpec, provider_id: str) -> Sandbox:
        """The live sandbox with this backend id. Raises ``KeyError`` when it is gone."""
        ...


@dataclass(frozen=True)
class ArtifactRef:
    """Where a deliverable ended up, as the product's artifact store names it."""

    name: str
    uri: str
    mime: str
    size: int

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "uri": self.uri, "mime": self.mime, "size": self.size}


class ArtifactSink(Protocol):
    """Where the runtime puts a run's deliverables. The product implements it."""

    async def save(self, thread_id: str, name: str, data: bytes, mime: str) -> ArtifactRef: ...
