"""Resumable ranged downloads for a restore's retry: one large object, part by part.

s5cmd starts an object over on every run, so a large object on a slow connection can
fail every attempt of a restore near its end. A retry fetches each large object still
missing here instead: ranged GETs of :data:`PART_BYTES`, appended to a partial file, so a
part that stalls is fetched again from the last byte written, never from zero. The
partial file is renamed over the target once complete. Needs the ``sandbox`` extra
(boto3) in the sandbox image; boto3 is imported only when a retry needs it, so a sandbox
without it still restores, with s5cmd alone.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict, cast

#: One ranged GET.
PART_BYTES = 16 * 1024 * 1024
#: One read from a part's body; progress is visible on disk after each.
CHUNK_BYTES = 1024 * 1024
#: Appended to a target's name while its bytes arrive.
PARTIAL_SUFFIX = ".actant-partial"


class Body(Protocol):
    def read(self, amt: int) -> bytes: ...

    def close(self) -> None: ...


class GetObjectOutput(TypedDict):
    Body: Body
    ETag: str
    ContentRange: NotRequired[str]


class RangedClient(Protocol):
    def get_object(self, **request: str) -> GetObjectOutput: ...


class FetchError(Exception):
    """An object that could not be fetched; its partial file is kept for the next try."""


def client(endpoint_url: str | None, stall_s: float) -> RangedClient:
    """An S3 client configured as s5cmd is (``AWS_*`` environment, path-style for a custom
    endpoint) whose reads fail after ``stall_s`` without a byte. Retries are the caller's,
    so they resume instead of starting a part over. Raises ``ImportError`` without boto3."""
    import boto3  # the optional ``sandbox`` extra; see the module docstring
    from botocore.config import Config

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    config = Config(
        connect_timeout=stall_s,
        read_timeout=stall_s,
        retries={"total_max_attempts": 1},
        s3={"addressing_style": "path"} if endpoint_url else None,
    )
    made = boto3.client("s3", endpoint_url=endpoint_url, region_name=region, config=config)
    # boto3 generates its clients' methods at runtime, so its types lack ``get_object``.
    return cast(RangedClient, made)


def partial_path(target: Path) -> Path:
    return target.with_name(f"{target.name}{PARTIAL_SUFFIX}")


def fetch(
    s3: RangedClient,
    url: str,
    size: int,
    target: Path,
    *,
    deadline: float,
    tries: int,
    etag: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Fetch the object at ``url`` (``s3://bucket/key``, ``size`` bytes) into ``target``,
    resuming any partial file left by an earlier call; how many bytes this call fetched.

    A failed part is fetched again from the last byte written. ``tries`` consecutive
    failures without a byte between them, or ``deadline`` (``clock`` time), raise
    :class:`FetchError`. The object's ETag is pinned on the first part, so a change
    mid-download starts it over rather than mixing two versions; pass the listing's
    ``etag`` to pin it across calls too."""
    from botocore.exceptions import BotoCoreError, ClientError  # with boto3, as in ``client``

    bucket, _, key = url.removeprefix("s3://").partition("/")
    if etag is not None:
        etag = f'"{etag.strip(chr(34))}"'  # s5cmd lists it bare; If-Match wants it quoted
    partial = partial_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fetched = 0
    failures = 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= size:
            break
        if clock() >= deadline:
            raise FetchError(f"{url}: out of time at {offset}/{size} bytes")
        if failures >= tries:
            raise FetchError(f"{url}: {failures} tries without progress at {offset}/{size} bytes")
        end = min(offset + PART_BYTES, size) - 1
        request = {"Bucket": bucket, "Key": key, "Range": f"bytes={offset}-{end}"}
        if etag is not None:
            request["IfMatch"] = etag
        try:
            etag = _part(s3, request, partial, deadline, clock)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in ("PreconditionFailed", "412"):
                partial.unlink(missing_ok=True)  # the object changed: start it over
                etag = None
        except (BotoCoreError, OSError):
            pass  # a stalled read keeps what it wrote before it; the next part resumes there
        now = partial.stat().st_size if partial.exists() else 0
        fetched += max(0, now - offset)
        failures = 0 if now > offset else failures + 1
    if partial.stat().st_size != size:
        raise FetchError(f"{url}: {partial.stat().st_size} bytes on disk, the object has {size}")
    partial.replace(target)
    return fetched


def _part(
    s3: RangedClient,
    request: dict[str, str],
    partial: Path,
    deadline: float,
    clock: Callable[[], float],
) -> str:
    """Append one ranged GET's bytes to ``partial`` as they arrive. A read that stalls
    raises from the client; what was written before it stays."""
    response = s3.get_object(**request)
    body = response["Body"]
    if "ContentRange" not in response:
        body.close()
        raise FetchError(f"{request['Key']}: the server ignored the range")
    try:
        with partial.open("ab") as out:
            while clock() < deadline:
                chunk = body.read(CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                out.flush()
    finally:
        body.close()
    return response["ETag"]
