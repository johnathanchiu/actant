"""The typed messages between a worker, a sandbox's entrypoint and its service host.

HTTP bodies (:class:`CallRequest`, :class:`CallResponse`) are JSON on the wire;
the launch configuration (:class:`EntryConfig`) is one JSON document on the
entrypoint's command line. Both sides parse into these models, so a malformed
message fails validation at the boundary instead of deep inside a handler.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


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


class Image(_Message):
    name: str
    media_type: str
    data_b64: str


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


class CallResponse(_Message):
    text: str = ""
    images: list[Image] = Field(default_factory=list)
    error: str | None = None
    #: Only from a host that pushes storage; absent on the wire otherwise.
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


class HostConfig(_Message):
    #: Service name to ``"pkg.mod:Class"``.
    services: dict[str, str] = Field(min_length=1)
    #: ``0`` picks a free port.
    port: int = Field(default=8080, ge=0, le=65535)
    bind: str = "0.0.0.0"
    #: Names :func:`actant.sandbox.host.script_env` removes.
    scrub: list[str] = Field(default_factory=list)
    push: PushConfig | None = None


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
