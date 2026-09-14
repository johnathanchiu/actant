"""The typed messages between a worker, a sandbox's entrypoint and its service host.

HTTP bodies (:class:`CallRequest`, :class:`CallResponse`) are JSON on the wire;
the launch configuration (:class:`EntryConfig`) is one JSON document on the
entrypoint's command line. Both sides parse into these models, so a malformed
message fails validation at the boundary instead of deep inside a handler.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat

from actant.sandbox.base import MAX_PRESIGN_S


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
    #: A presigned URL the model provider fetches from the bucket.
    URL = "url"


class InlineSource(_Message):
    kind: Literal[ImageSourceKind.INLINE] = ImageSourceKind.INLINE
    data_b64: str


class UrlSource(_Message):
    kind: Literal[ImageSourceKind.URL] = ImageSourceKind.URL
    url: str
    #: Unix time the URL stops working.
    expires_at: float


class Image(_Message):
    """One image a service method returned. ``name`` is its sandbox-relative path, or
    ``image-<n>`` for bytes. A host with :class:`ImageUploadConfig` sends a
    :class:`UrlSource`; without one, or when an upload fails, the bytes go inline."""

    name: str
    media_type: str
    source: Annotated[InlineSource | UrlSource, Field(discriminator="kind")]


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
    """Upload each returned image (content-addressed, under ``destination``) with s5cmd
    against ``endpoint_url``, then presign it against ``public_endpoint_url``.

    The two endpoints differ when the bucket is reached one way from the sandbox and
    another from the model provider (a local MinIO behind a public tunnel). A presigned
    URL signs its host, so the public endpoint must forward requests with that host.
    Bucket keys come from the host's environment (``AWS_ACCESS_KEY_ID`` and friends).
    """

    #: ``s3://bucket/prefix/``.
    destination: str = Field(pattern=r"^s3://[^/]+/(.*/)?$")
    #: Where uploads go; ``None`` is AWS S3.
    endpoint_url: str | None = None
    #: The host presigned URLs name. A model provider fetches them, so it must reach it.
    public_endpoint_url: str = Field(pattern=r"^https?://[^/]+")
    expires_s: int = Field(gt=0, le=MAX_PRESIGN_S)
    #: Upload plus presign, per image; past it the commands are killed and the image
    #: goes inline.
    timeout_s: PositiveFloat = 10.0


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
    restored file under ``root`` its object's mtime."""

    argv: list[str] = Field(min_length=1)
    prefix: str
    root: str


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


class RestoreConfig(_Message):
    #: Pulls the run's prefix onto the disk; each command failing or exceeding
    #: ``timeout_s`` fails startup.
    argv: list[str] = Field(min_length=1)
    timeout_s: PositiveFloat = 1800.0
    stamp: StampConfig | None = None
    #: Used only when the run's prefix is empty.
    seed: SeedConfig | None = None


class EntryConfig(_Message):
    """``python -m actant.sandbox.entry '<EntryConfig JSON>'``: restore, then serve (or idle)."""

    restore: RestoreConfig | None = None
    host: HostConfig | None = None
