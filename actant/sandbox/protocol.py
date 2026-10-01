"""The typed messages between a worker, a sandbox's entrypoint and its service host.

HTTP bodies (:class:`CallRequest`, :class:`CallResponse`) are JSON on the wire;
the launch configuration (:class:`EntryConfig`) is one JSON document on the
entrypoint's command line. Both sides parse into these models, so a malformed
message fails validation at the boundary instead of deep inside a handler.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt


class Route(StrEnum):
    #: ``POST`` a :class:`CallRequest`; the reply is a :class:`CallResponse`.
    CALL = "/v1/call"
    #: Close every instance, push storage once more, answer, and exit.
    SHUTDOWN = "/v1/shutdown"


class Header(StrEnum):
    AUTHORIZATION = "Authorization"
    CONNECTION = "Connection"
    CONTENT_LENGTH = "Content-Length"
    CONTENT_TYPE = "Content-Type"


class _Message(BaseModel):
    model_config = ConfigDict(frozen=True)


class CallRequest(_Message):
    service: str
    key: str
    method: str
    init: dict[str, Any] = Field(default_factory=dict)
    args: dict[str, Any] = Field(default_factory=dict)


class ImageSourceKind(StrEnum):
    #: The bytes, base64-encoded in the response.
    INLINE = "inline"
    #: A storage key in the bucket; the model call resolves it.
    ASSET = "asset"


class InlineSource(_Message):
    kind: Literal[ImageSourceKind.INLINE] = ImageSourceKind.INLINE
    data_b64: str


class AssetSource(_Message):
    kind: Literal[ImageSourceKind.ASSET] = ImageSourceKind.ASSET
    storage_key: str


class Image(_Message):
    """One image a service method returned. ``name`` is its sandbox-relative path, or
    ``image-<n>`` for bytes. A host with :class:`ImageUploadConfig` sends a
    :class:`AssetSource`; without one, or when an upload fails, the bytes go inline."""

    name: str
    media_type: str
    source: Annotated[InlineSource | AssetSource, Field(discriminator="kind")]


class StorageStatus(_Message):
    """A pushing host's storage status, on every call response and on
    ``ToolResult.metadata[MetadataKey.STORAGE]`` (as JSON: read it back with
    ``StorageStatus.model_validate``)."""

    #: Unix time the latest push started.
    last_attempt_at: float | None = None
    #: Unix time the latest successful push started.
    last_success_at: float | None = None
    #: Short reason the latest push failed; ``None`` once one succeeds.
    last_error: str | None = None
    #: Failed pushes since the last success.
    consecutive_failures: int = 0
    #: Calls completed that no successful push has covered.
    pending: bool = False
    #: Why the first image in this response that went inline instead of as a URL did;
    #: ``None`` when every image uploaded (or there were none).
    image_error: str | None = None


class CallResponse(_Message):
    text: str = ""
    images: list[Image] = Field(default_factory=list)
    error: str | None = None
    #: Only from a host that pushes storage or uploads images; absent on the wire otherwise.
    storage: StorageStatus | None = None

    def to_json(self) -> bytes:
        exclude = {"storage"} if self.storage is None else None
        return self.model_dump_json(exclude=exclude).encode()


class PushConfig(_Message):
    """The host's storage push: run ``argv`` after calls and every ``interval_s`` while
    calls are unpushed; kill it after ``timeout_s``."""

    argv: list[str] = Field(min_length=1)
    interval_s: PositiveFloat = 60.0
    timeout_s: PositiveFloat = 300.0


class ImageUploadConfig(_Message):
    """Upload bytes under a durable S3 reference. Signing belongs to the reader."""

    destination: str = Field(pattern=r"^s3://[^/]+/(.*/)?$")
    endpoint_url: str | None = None
    # A failed or timed-out upload falls back to inline bytes.
    timeout_s: PositiveFloat = Field(default=10.0, allow_inf_nan=False)


class HostConfig(_Message):
    #: Service name to ``"pkg.mod:Class"``.
    services: dict[str, str] = Field(min_length=1)
    #: ``0`` picks a free port.
    port: int = Field(default=8080, ge=0, le=65535)
    bind: str = "0.0.0.0"
    #: Names :func:`actant.sandbox.host.script_env` removes.
    scrub: list[str] = Field(default_factory=list)
    push: PushConfig | None = None
    images: ImageUploadConfig | None = None


class StampConfig(_Message):
    """Alongside a pull, list ``prefix`` with ``argv`` (``s5cmd --json ls``) and give each
    restored file under ``root`` its object's mtime, except under the ``skip`` prefixes
    (read-only mounts)."""

    argv: list[str] = Field(min_length=1)
    prefix: str
    root: str
    skip: list[str] = Field(default_factory=list)


class SeedConfig(_Message):
    """A new run's files: when the run's prefix is empty, pull ``argv`` (the seed) onto the
    disk while ``copy_argv`` copies the seed into the run's prefix, then write the marker.
    Startup waits for all of it."""

    argv: list[str] = Field(min_length=1)
    copy_argv: list[str] = Field(min_length=1)
    #: Writes the marker (from empty stdin) once the copy finished.
    write_marker_argv: list[str] = Field(min_length=1)
    #: Lists the marker: a run's prefix with files and no marker is incompletely seeded.
    check_marker_argv: list[str] = Field(min_length=1)
    stamp: StampConfig | None = None


class PullConfig(_Message):
    """One more pull onto the disk: ``argv``, with ``stamp`` alongside it."""

    argv: list[str] = Field(min_length=1)
    stamp: StampConfig | None = None


class RestoreConfig(_Message):
    #: Pulls the run's prefix onto the disk.
    argv: list[str] = Field(min_length=1)
    #: The whole restore's bound: a pull, its retries and the seed's copy end by then.
    timeout_s: PositiveFloat = 1800.0
    #: A pull (the run's, the seed's or another input's) is killed once nothing arrives
    #: for this long (no output and no bytes on disk under its stamp's root), and run
    #: again, up to ``attempts`` runs in all. A slow pull that keeps moving is never
    #: killed before ``timeout_s``. A retry skips the files already on disk and resumes
    #: each large object still missing from where it stopped (see
    #: :mod:`actant.sandbox.ranged`).
    stall_s: PositiveFloat = 60.0
    attempts: PositiveInt = 1
    #: The bucket endpoint a retry's ranged downloads use (the pull's own ``argv`` has
    #: it too); ``None`` is AWS.
    endpoint_url: str | None = None
    #: Where the entrypoint writes the :class:`RestoreSummary` as JSON.
    summary_path: str | None = None
    stamp: StampConfig | None = None
    #: Used only when the run's prefix is empty.
    seed: SeedConfig | None = None
    #: Other inputs (read-only, never pushed), pulled alongside with the same retries;
    #: an empty one is not a failure.
    also: list[PullConfig] = []


class Pulled(StrEnum):
    """How a pull ended."""

    FILES = "files"
    #: The prefix has no objects: a new run.
    EMPTY = "empty"
    FAILED = "failed"


class PullSummary(_Message):
    """One pull of a restore, as it ended."""

    #: The pulled prefix (``s3://bucket/prefix/``), or the command's last argument.
    source: str
    outcome: Pulled
    #: Objects this pull fetched, across its attempts.
    objects: int = 0
    #: Objects its listing found; ``None`` without a listing.
    listed: int | None = None
    #: The listed objects' total size.
    bytes: int | None = None
    seconds: float = 0.0
    attempts: int = 0
    #: Attempts killed because nothing arrived for ``stall_s``.
    stalls: int = 0
    #: Large objects a retry finished with ranged downloads.
    resumed: int = 0
    error: str | None = None

    def line(self) -> str:
        listed = "?" if self.listed is None else str(self.listed)
        size = "?" if self.bytes is None else f"{self.bytes / 1e6:.1f} MB"
        text = (
            f"{self.source} {self.outcome}: {self.objects}/{listed} objects, {size}, "
            f"{self.seconds:.1f}s, {self.attempts} attempts ({max(0, self.attempts - 1)} "
            f"retries, {self.stalls} stalls, {self.resumed} resumed)"
        )
        # One line, so the tail of a failed sandbox's stderr holds the whole summary.
        return text if self.error is None else f"{text}: {' '.join(self.error[-500:].split())}"


class RestoreSummary(_Message):
    """What a sandbox's restore did: on the ``ModalSandbox`` it opened, and in the error
    of one whose restore failed."""

    ok: bool
    seconds: float
    pulls: list[PullSummary] = Field(default_factory=list)

    def line(self) -> str:
        pulls = "; ".join(pull.line() for pull in self.pulls)
        return f"restore {'ok' if self.ok else 'failed'} in {self.seconds:.1f}s: {pulls}"


class EntryConfig(_Message):
    """``python -m actant.sandbox.entry '<EntryConfig JSON>'``: restore, then serve (or idle)."""

    restore: RestoreConfig | None = None
    host: HostConfig | None = None
