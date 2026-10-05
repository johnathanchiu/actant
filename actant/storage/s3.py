"""S3 asset resolution: existence checks through an injected boto3-compatible client, and
presigned GET URLs that every process computes identically; or an asset's bytes in one GET,
for a caller that sends them inline (:class:`~actant.assets.InlineImages`).

Configure the SDK client with bounded SDK timeouts/retries; the signing endpoint must be the
one the model provider can reach. Credentials and client lifecycle remain caller-owned. The
resolver accepts only keys in its configured bucket/prefix. Applications with per-user
authorization must check access before delegating to this adapter.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from typing import Protocol
from urllib.parse import urlsplit

from actant.assets import AssetContext, AssetReference, MissingAsset, ResolvedImage
from actant.storage.sigv4 import (
    MAX_EXPIRES_S,
    AddressingStyle,
    SigningKeys,
    check_endpoint,
    presign_get,
)

#: URLs are signed at the start of a window this long, so every process and restart sends the
#: same URL for an object within it; the provider's prompt cache matches image URLs exactly.
WINDOW_S = 6 * 3600
#: How long a URL outlives its window, so one handed out always has at least this much left.
BUFFER_S = 1800
#: Margin on top of the model call's budget, as `actant.llm.providers._shared` uses.
EXPIRY_MARGIN_S = 120
MISSING_CODES = frozenset({"NoSuchKey", "NotFound", "404"})


class S3Body(Protocol):
    def read(self, amt: int | None = ..., /) -> bytes: ...

    def close(self) -> None: ...


class S3Client(Protocol):
    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, object]: ...

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, S3Body]: ...


class S3AssetResolver:
    """Resolves stable references to presigned URLs.

    A URL is signed at ``floor(now / window_s) * window_s`` and lives ``window_s + buffer_s``:
    a pure function of the key, the window and the credentials.

    A key found to exist is not checked again within its window, so retention must not
    delete a referenced object inside a window.
    """

    def __init__(
        self,
        client: S3Client,
        *,
        endpoint_url: str,
        region: str,
        keys: SigningKeys,
        bucket: str,
        prefix: str,
        window_s: int = WINDOW_S,
        buffer_s: int = BUFFER_S,
        addressing_style: AddressingStyle = "path",
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not bucket or not prefix or not prefix.endswith("/"):
            raise ValueError("bucket and a nonempty prefix ending in '/' are required")
        if window_s < 1 or buffer_s < 0 or window_s + buffer_s > MAX_EXPIRES_S:
            raise ValueError("invalid URL window or buffer")
        if not region or not keys.access_key_id or not keys.secret_access_key:
            raise ValueError("region and credentials are required")
        check_endpoint(endpoint_url)
        self.client, self.bucket, self.prefix = client, bucket, prefix
        self.endpoint_url, self.region, self.keys = endpoint_url, region, keys
        self.window_s, self.buffer_s = window_s, buffer_s
        self.addressing_style: AddressingStyle = addressing_style
        self.clock = clock
        # per-process; a miss re-checks
        self._window = -1
        self._found: set[str] = set()

    async def resolve(
        self, asset: AssetReference, context: AssetContext
    ) -> ResolvedImage | MissingAsset:
        key = self._key(asset.storage_key)
        if self.buffer_s <= context.minimum_validity_s + EXPIRY_MARGIN_S:
            raise ValueError("URL buffer must exceed the model budget plus the expiry margin")
        window = self._window_start()
        if window != self._window:
            self._window, self._found = window, set()
        if key not in self._found:
            if not await self._exists(key):
                return MissingAsset()
            if window == self._window:
                self._found.add(key)
        signed_at = self._window_start()
        expires_s = self.window_s + self.buffer_s
        url = presign_get(
            self.endpoint_url,
            self.region,
            self.keys,
            self.bucket,
            key,
            signed_at=signed_at,
            expires_s=expires_s,
            addressing_style=self.addressing_style,
        )
        return ResolvedImage(asset.mime, url=url, expires_at=signed_at + expires_s)

    async def read(self, asset: AssetReference, limit: int) -> bytes | MissingAsset | None:
        """The asset's bytes, read in one GET on the client's pooled connections, or None
        when they are more than `limit` (only `limit + 1` are read): no existence check and
        no URL, for a caller that sends the bytes themselves."""
        key = self._key(asset.storage_key)
        try:
            return await asyncio.to_thread(self._get, key, limit)
        except Exception as error:
            if _missing(error):
                return MissingAsset()
            raise

    def _get(self, key: str, limit: int) -> bytes | None:
        body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
        try:
            data = body.read(limit + 1)
        finally:
            body.close()
        return data if len(data) <= limit else None

    def _window_start(self) -> int:
        now = self.clock() if self.clock is not None else time.time()
        return int(now // self.window_s) * self.window_s

    def _key(self, storage_key: str) -> str:
        key = storage_key
        if key.startswith("s3://"):
            location = urlsplit(key)
            if location.netloc != self.bucket or location.query or location.fragment:
                raise PermissionError("asset is outside the configured bucket")
            key = location.path.lstrip("/")
        if not key.startswith(self.prefix) or any(p in {".", ".."} for p in key.split("/")):
            raise PermissionError("asset is outside the configured prefix")
        return key

    async def _exists(self, key: str) -> bool:
        try:
            await asyncio.to_thread(self.client.head_object, Bucket=self.bucket, Key=key)
        except Exception as error:
            if _missing(error):
                return False
            raise
        return True


def _missing(error: Exception) -> bool:
    """botocore's ClientError exposes a structured response. Only an explicit missing-object
    response means missing; 403 and transport errors propagate."""
    response = getattr(error, "response", None)
    detail = response.get("Error") if isinstance(response, dict) else None
    return isinstance(detail, dict) and detail.get("Code") in MISSING_CODES
