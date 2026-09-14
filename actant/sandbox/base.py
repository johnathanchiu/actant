"""Sandboxes: where an agent's tools read, write and run code.

One sandbox per thread. A tool that needs one declares it (``needs_sandbox``,
or a ``Sandbox`` parameter on a function tool) and the runtime hands it the
thread's sandbox through its :class:`~actant.tools.base.CallContext`. Tools
that do not need one never see one.

The filesystem a sandbox exposes is the thread's working directory. Every
backend keys it by thread under the spec's ``mount``: a directory for the
local backend, a bucket prefix mounted into the container for cloud backends.
The worker holds no files; it reads results back through the same handle.

This module is a leaf: it imports nothing from the runtime, so agent
definitions and tools can name these types without cycles.
"""

from __future__ import annotations

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


#: The longest a SigV4 presigned URL may live (seven days).
MAX_PRESIGN_S = 7 * 24 * 3600
#: Appended to a thread's key (not inside its prefix) for the prefix its images upload to,
#: so no push or pull touches them.
IMAGES_SUFFIX = ".actant-images/"


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
    #: ``disk_sync``: a bucket key prefix (``"templates/base/"``) that starts a new thread.
    #: When the thread's prefix is empty, the sandbox pulls the seed while the bucket copies
    #: it into the thread's prefix; startup waits for both. A thread with files ignores it.
    seed: str | None = None
    #: How long a presigned image URL lives. A host whose files reach a bucket (``disk_sync``,
    #: or a provider given an :class:`ImageBucket`) uploads each image a service returns and
    #: sends its URL instead of its bytes. ``None`` always sends bytes. At most seven days.
    image_url_ttl_s: int | None = 6 * 3600

    def __post_init__(self) -> None:
        # Coerce (and validate) a plain string from an untyped config.
        object.__setattr__(self, "storage", Storage(self.storage))
        if self.sync_interval_s <= 0 or self.sync_timeout_s <= 0:
            raise ValueError("sync_interval_s and sync_timeout_s must be positive")
        if self.seed is not None and (
            self.storage != Storage.DISK_SYNC or not self.seed.endswith("/")
        ):
            raise ValueError("seed is a key prefix ending in '/', for disk_sync storage")
        if self.image_url_ttl_s is not None and not 0 < self.image_url_ttl_s <= MAX_PRESIGN_S:
            raise ValueError("image_url_ttl_s must be positive and at most seven days")


@dataclass(frozen=True)
class ImageBucket:
    """Where a backend whose files are not in a bucket uploads returned images to presign them.

    ``endpoint_url`` is where uploads go (``None`` is AWS S3); ``public_endpoint_url`` is the
    host the presigned URLs name, for a model provider that cannot reach ``endpoint_url``
    (``None`` is ``endpoint_url``). Bucket keys come from the environment.
    """

    bucket: str
    key_prefix: str = "sandboxes/"
    endpoint_url: str | None = None
    public_endpoint_url: str | None = None

    def destination(self, thread_id: str) -> str:
        """The ``s3://`` prefix a thread's images upload to: a sibling of its files' prefix."""
        return f"s3://{self.bucket}/{self.key_prefix}{thread_id}{IMAGES_SUFFIX}"


class SandboxProvider(Protocol):
    """Opens sandboxes for a backend and reattaches to ones that already exist."""

    async def open(self, spec: SandboxSpec, *, agent_id: str, thread_id: str) -> Sandbox: ...

    async def attach(self, spec: SandboxSpec, sandbox_id: str) -> Sandbox:
        """The live sandbox with this id. Raises ``KeyError`` when it is gone."""
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
