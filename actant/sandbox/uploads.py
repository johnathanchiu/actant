"""A service host's image uploads: one long-lived boto3 client on its own threads.

One client per host reuses its pooled connections across calls, and its own thread pool
keeps uploads from queueing behind the worker threads that run service methods. Credentials,
region and endpoint are the ``AWS_*`` environment and ``ImageUploadConfig.endpoint_url``.
Needs the ``sandbox`` extra (boto3) in the sandbox image.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import mimetypes
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, cast

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from actant.sandbox.protocol import AssetSource, Image, ImageUploadConfig, InlineSource

#: Threads, and pooled S3 connections, one host's uploads share across all its calls.
UPLOAD_THREADS = 32
#: Requests one upload may send: boto's ``standard`` retries, each with its own
#: connect and read timeout of ``ImageUploadConfig.timeout_s``.
UPLOAD_ATTEMPTS = 3
#: boto's backoff between attempts, at most: ``standard`` waits up to 2**n s, jittered.
UPLOAD_BACKOFF_S = 4.0


def upload_budget(timeout_s: float) -> float:
    """How long one upload may run once its thread starts it: every attempt's connect and
    read timeouts, and the backoff between them."""
    return UPLOAD_ATTEMPTS * 2 * timeout_s + UPLOAD_BACKOFF_S


class S3Client(Protocol):
    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, ContentType: str
    ) -> Mapping[str, object]: ...


def s3_client(endpoint_url: str | None, timeout_s: float) -> S3Client:
    """An S3 client configured as s5cmd was: ``AWS_REGION`` (else ``AWS_DEFAULT_REGION``,
    else ``us-east-1``), path-style addressing for a custom endpoint, and ``timeout_s`` as
    each attempt's connect and read timeouts."""
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    config = Config(
        max_pool_connections=UPLOAD_THREADS,
        connect_timeout=timeout_s,
        read_timeout=timeout_s,
        retries={"mode": "standard", "total_max_attempts": UPLOAD_ATTEMPTS},
        s3={"addressing_style": "path"} if endpoint_url else None,
    )
    client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region, config=config)
    # boto3 generates its clients' methods at runtime, so its types lack ``put_object``.
    return cast(S3Client, client)


class ImageUploader:
    def __init__(self, config: ImageUploadConfig, client: S3Client | None = None) -> None:
        self.config = config
        self.client = client or s3_client(config.endpoint_url, config.timeout_s)
        self.executor = ThreadPoolExecutor(UPLOAD_THREADS, thread_name_prefix="actant-upload")

    async def upload(self, image: Image) -> tuple[Image, str | None]:
        """``image`` with a durable asset source, or unchanged with the reason it could not be.

        The key is ``destination`` plus the bytes' SHA-256 and the media type's extension. The
        upload is timed from when its thread starts it, not while it waits for a thread: a
        burst of images queues behind ``UPLOAD_THREADS``, and that wait is not the bucket's."""
        if not isinstance(image.source, InlineSource):
            return image, None
        data = base64.b64decode(image.source.data_b64)
        extension = mimetypes.guess_extension(image.media_type) or ""
        key = f"{self.config.destination}{hashlib.sha256(data).hexdigest()}{extension}"
        bucket, _, name = key.removeprefix("s3://").partition("/")
        put = functools.partial(
            self.client.put_object,
            Bucket=bucket,
            Key=name,
            Body=data,
            ContentType=image.media_type,
        )
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def start() -> Mapping[str, object]:
            loop.call_soon_threadsafe(started.set)
            return put()

        running = loop.run_in_executor(self.executor, start)
        budget = upload_budget(self.config.timeout_s)
        try:
            waiting = asyncio.ensure_future(started.wait())
            try:
                await asyncio.wait({running, waiting}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                waiting.cancel()
            async with asyncio.timeout(budget):
                await running
        except TimeoutError:
            return image, f"{image.name}: upload timed out after {budget:g}s"
        except (BotoCoreError, ClientError) as error:
            return image, f"{image.name}: upload failed: {error}"[:500]
        return image.model_copy(update={"source": AssetSource(storage_key=key)}), None
