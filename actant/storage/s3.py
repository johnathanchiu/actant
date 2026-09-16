"""S3 asset resolution using an injected boto3-compatible client.

Configure the SDK client with the endpoint reachable by the model provider and
bounded SDK timeouts/retries. Credentials and client lifecycle remain caller-owned.
The resolver accepts only keys in its configured bucket/prefix. Applications with
per-user authorization must check access before delegating to this adapter.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import Protocol
from urllib.parse import urlsplit

from actant.assets import AssetContext, AssetReference, MissingAsset, ResolvedImage


class S3Client(Protocol):
    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, object]: ...
    def generate_presigned_url(
        self, ClientMethod: str, *, Params: dict[str, str], ExpiresIn: int
    ) -> str: ...


class S3AssetResolver:
    """Sign stable references once per validity window; never implement SigV4 here."""

    def __init__(
        self,
        client: S3Client,
        *,
        bucket: str,
        prefix: str,
        url_ttl_s: int = 3600,
        cache_entries: int = 1024,
    ) -> None:
        if not bucket or not prefix or not prefix.endswith("/"):
            raise ValueError("bucket and a nonempty prefix ending in '/' are required")
        if not 0 < url_ttl_s <= 604800 or cache_entries < 1:
            raise ValueError("invalid URL lifetime or cache size")
        self.client, self.bucket, self.prefix = client, bucket, prefix
        self.url_ttl_s, self.cache_entries = url_ttl_s, cache_entries
        self._cache: OrderedDict[tuple[str, str], ResolvedImage] = OrderedDict()

    async def resolve(
        self, asset: AssetReference, context: AssetContext
    ) -> ResolvedImage | MissingAsset:
        key = asset.storage_key
        if key.startswith("s3://"):
            location = urlsplit(key)
            if location.netloc != self.bucket or location.query or location.fragment:
                raise PermissionError("asset is outside the configured bucket")
            key = location.path.lstrip("/")
        if not key.startswith(self.prefix) or any(p in {".", ".."} for p in key.split("/")):
            raise PermissionError("asset is outside the configured prefix")
        if self.url_ttl_s <= context.minimum_validity_s + 120:
            raise ValueError("URL lifetime must exceed the model budget plus 120 seconds")
        cache_key = (key, asset.mime)
        cached = self._cache.get(cache_key)
        if (
            cached is not None
            and cached.expires_at is not None
            and cached.expires_at > time.time() + context.minimum_validity_s + 120
        ):
            self._cache.move_to_end(cache_key)
            return cached
        try:
            await asyncio.to_thread(self.client.head_object, Bucket=self.bucket, Key=key)
        except Exception as error:
            # botocore ClientError exposes a structured response. Only an explicit
            # missing-object response means missing; 403 and transport errors propagate.
            response = getattr(error, "response", None)
            detail = response.get("Error") if isinstance(response, dict) else None
            code = detail.get("Code") if isinstance(detail, dict) else None
            if code in {"NoSuchKey", "NotFound", "404"}:
                self._cache.pop(cache_key, None)
                return MissingAsset()
            raise
        signed_at = time.time()
        url = await asyncio.to_thread(
            self.client.generate_presigned_url,
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.url_ttl_s,
        )
        resolved = ResolvedImage(asset.mime, url=url, expires_at=signed_at + self.url_ttl_s)
        self._cache[cache_key] = resolved
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.cache_entries:
            self._cache.popitem(last=False)
        return resolved
