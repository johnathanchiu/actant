"""S3 asset resolution using an injected boto3-compatible client.

Configure the SDK client with the endpoint reachable by the model provider and
bounded SDK timeouts/retries. Credentials and client lifecycle remain caller-owned.
The resolver accepts only keys in its configured bucket/prefix. Applications with
per-user authorization must check access before delegating to this adapter.

Signed URLs live in a ``SignedUrlStore`` shared by every process, so a thread's image keeps one
URL across workers and restarts until it nears expiry: the provider's prompt cache matches
image URLs exactly, and SigV4 puts the signing time in the URL.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Protocol
from urllib.parse import urlsplit

from actant.assets import (
    AssetContext,
    AssetReference,
    MissingAsset,
    ResolvedImage,
    SignedUrl,
    SignedUrlStore,
)


class S3Client(Protocol):
    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, object]: ...
    def generate_presigned_url(
        self, ClientMethod: str, *, Params: dict[str, str], ExpiresIn: int
    ) -> str: ...


class S3AssetResolver:
    """Sign each object once per URL lifetime, shared through ``urls``; never implement SigV4 here."""

    def __init__(
        self,
        client: S3Client,
        *,
        bucket: str,
        prefix: str,
        urls: SignedUrlStore,
        url_ttl_s: int = 3600,
    ) -> None:
        if not bucket or not prefix or not prefix.endswith("/"):
            raise ValueError("bucket and a nonempty prefix ending in '/' are required")
        if not 0 < url_ttl_s <= 604800:
            raise ValueError("invalid URL lifetime")
        self.client, self.bucket, self.prefix = client, bucket, prefix
        self.urls, self.url_ttl_s = urls, url_ttl_s

    async def resolve(
        self, asset: AssetReference, context: AssetContext
    ) -> ResolvedImage | MissingAsset:
        key = asset.storage_key
        if key.startswith("s3://"):
            parsed = urlsplit(key)
            if parsed.netloc != self.bucket or parsed.query or parsed.fragment:
                raise PermissionError("asset is outside the configured bucket")
            key = parsed.path.lstrip("/")
        if not key.startswith(self.prefix) or any(p in {".", ".."} for p in key.split("/")):
            raise PermissionError("asset is outside the configured prefix")
        if self.url_ttl_s <= context.minimum_validity_s + 120:
            raise ValueError("URL lifetime must exceed the model budget plus 120 seconds")
        location = f"s3://{self.bucket}/{key}"
        replace_before = time.time() + context.minimum_validity_s + 120
        stored = await self.urls.get(location)
        if stored is not None and stored.expires_at > replace_before:
            return ResolvedImage(asset.mime, url=stored.url, expires_at=stored.expires_at)
        try:
            await asyncio.to_thread(self.client.head_object, Bucket=self.bucket, Key=key)
        except Exception as error:
            # botocore ClientError exposes a structured response. Only an explicit
            # missing-object response means missing; 403 and transport errors propagate.
            response = getattr(error, "response", None)
            detail = response.get("Error") if isinstance(response, dict) else None
            code = detail.get("Code") if isinstance(detail, dict) else None
            if code in {"NoSuchKey", "NotFound", "404"}:
                return MissingAsset()
            raise
        signed_at = time.time()
        url = await asyncio.to_thread(
            self.client.generate_presigned_url,
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.url_ttl_s,
        )
        signed = await self.urls.put(
            location, SignedUrl(url, signed_at + self.url_ttl_s), replace_before=replace_before
        )
        return ResolvedImage(asset.mime, url=signed.url, expires_at=signed.expires_at)
