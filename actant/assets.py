"""Durable media references and request-time resolution, independent of storage vendors."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from actant.blocks import (
    AssetBlock,
    Base64Source,
    InlineImageBlock,
    PromptBlock,
    TextBlock,
    UrlImageBlock,
)
from actant.llm.messages import Message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssetReference:
    storage_key: str
    mime: str

    def to_block(self) -> AssetBlock:
        return AssetBlock(storage_key=self.storage_key, mime=self.mime)


@dataclass(frozen=True)
class AssetContext:
    agent_id: str
    thread_id: str
    run_id: str
    turn_id: str
    # URLs must remain usable for the entire model activity, including provider retries.
    minimum_validity_s: float = 600.0


@dataclass(frozen=True)
class ResolvedImage:
    mime: str
    url: str | None = None
    expires_at: float | None = None
    data: bytes | None = None

    def to_block(self, minimum_validity_s: float) -> InlineImageBlock | UrlImageBlock:
        if (self.url is None) == (self.data is None):
            raise ValueError("resolved image requires exactly one of url or data")
        if self.url is not None:
            if not self.url.startswith(("https://", "http://")):
                raise ValueError("resolved image URL must use HTTP(S)")
            if self.expires_at is not None and self.expires_at <= time.time() + minimum_validity_s:
                raise ValueError("resolved image URL does not cover the model-call budget")
            return UrlImageBlock(url=self.url, media_type=self.mime)
        data = base64.b64encode(self.data or b"").decode()
        return InlineImageBlock(source=Base64Source(media_type=self.mime, data=data))


@dataclass(frozen=True)
class MissingAsset:
    reason: str = "asset no longer available"


class AssetResolver(Protocol):
    async def resolve(
        self, asset: AssetReference, context: AssetContext
    ) -> ResolvedImage | MissingAsset: ...


@runtime_checkable
class AssetReader(Protocol):
    """A resolver that also gives an asset's bytes directly (such as
    :class:`~actant.storage.s3.S3AssetResolver`): None when they are more than `limit`."""

    async def read(self, asset: AssetReference, limit: int) -> bytes | MissingAsset | None: ...


#: How many asset references one request resolves at once.
RESOLVE_CONCURRENCY = 16


async def prepare_messages(
    messages: Sequence[Message],
    resolver: AssetResolver | None,
    context: AssetContext,
) -> list[Message]:
    """Resolve each image :class:`AssetBlock` for this model call, and note any other file as
    text, without mutating the transcript or discarding message metadata. A request's references resolve concurrently, at most
    :data:`RESOLVE_CONCURRENCY` at a time. Missing bytes are visible; resolver failures
    propagate to execution.
    """
    assets = [
        block
        for message in messages
        if isinstance(message.content, list)
        for block in message.content
        if isinstance(block, AssetBlock) and block.mime.startswith("image/")
    ]
    limit = asyncio.Semaphore(RESOLVE_CONCURRENCY)

    async def resolve(block: AssetBlock) -> PromptBlock:
        if resolver is None:
            raise ValueError("asset references require an AssetResolver")
        async with limit:
            image = await resolver.resolve(AssetReference(block.storage_key, block.mime), context)
        if isinstance(image, MissingAsset):
            text = f"[Image unavailable: {image.reason}; asset_storage_key={block.storage_key}]"
            return TextBlock(text=text)
        return image.to_block(context.minimum_validity_s)

    resolved = iter(await asyncio.gather(*(resolve(block) for block in assets)))
    prepared: list[Message] = []
    for message in messages:
        if not isinstance(message.content, list):
            prepared.append(message)
            continue
        blocks: list[PromptBlock] = []
        for block in message.content:
            if not isinstance(block, AssetBlock):
                blocks.append(block)
            elif block.mime.startswith("image/"):
                blocks.append(next(resolved))
            else:
                text = f"[Attached file: mime={block.mime}, asset_storage_key={block.storage_key}]"
                blocks.append(TextBlock(text=text))
        prepared.append(replace(message, content=blocks))
    return prepared


#: The largest picture a whole request of pictures may hold, read before it is fitted: Azure
#: OpenAI's Responses API takes at most 50 MB of images in one request. Larger is missing.
READ_IMAGE_BYTES = 50_000_000
#: The largest picture sent inline: 50 of them (Azure's per-request image limit),
#: base64-encoded (4/3 the bytes), stay within 90% of :data:`READ_IMAGE_BYTES`.
INLINE_IMAGE_BYTES = READ_IMAGE_BYTES * 9 // 10 // 50 * 3 // 4
#: The longest side the model looks at: OpenAI scales a picture to fit 2048 x 2048 first.
INLINE_IMAGE_SIDE = 2048
#: A re-encoded picture's JPEG quality.
INLINE_JPEG_QUALITY = 85
#: The inline pictures one :class:`InlineImages` keeps, in bytes.
INLINE_CACHE_BYTES = 256 * 1024 * 1024


def read_image(url: str, limit: int, timeout_s: float = 30) -> bytes | None:
    """The picture at `url`, or None when it is larger than `limit` bytes. Blocks; only
    `limit + 1` bytes are ever read."""

    with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310
        data = response.read(limit + 1)
    return data if len(data) <= limit else None


def fit_image(
    data: bytes,
    side: int = INLINE_IMAGE_SIDE,
    max_bytes: int = INLINE_IMAGE_BYTES,
    quality: int = INLINE_JPEG_QUALITY,
) -> tuple[bytes, str]:
    """A picture as it goes inline, and its media type: at most `side` on its long side, as
    JPEG at `quality`, or as PNG when it has real transparency, which JPEG would flatten onto
    a background that is not there; within `max_bytes`, the quality (down to `quality - 30`),
    then the size, stepping down until it is. A JPEG already that small is sent as it is. The
    same picture always comes out the same, so a provider's prompt cache keeps matching.
    Needs Pillow (``actant[images]``); raises ``PIL.UnidentifiedImageError`` for bytes that
    are not a picture."""

    from PIL import Image

    with Image.open(io.BytesIO(data)) as opened:
        small = max(opened.size) <= side and len(data) <= max_bytes
        if small and opened.format == "JPEG":
            return data, "image/jpeg"
        image = opened.copy()
    image.thumbnail((side, side), Image.Resampling.LANCZOS)
    if image.mode == "P":  # a palette's transparency, as an alpha channel
        image = image.convert("RGBA")
    alpha = False
    if image.mode in ("RGBA", "LA", "PA"):
        low = image.getchannel("A").getextrema()[0]  # one band: (min, max)
        alpha = isinstance(low, (int, float)) and low < 255
    if not alpha:
        image = image.convert("RGB")
    floor, step = quality - 30, quality
    while True:
        out = io.BytesIO()
        if alpha:
            image.save(out, format="PNG", optimize=True)
        else:
            image.save(out, format="JPEG", quality=step)
        if out.tell() <= max_bytes:
            return out.getvalue(), "image/png" if alpha else "image/jpeg"
        if not alpha and step > floor:
            step -= 10
        else:  # smaller, a quarter each step
            size = (max(1, image.width * 3 // 4), max(1, image.height * 3 // 4))
            image = image.resize(size, Image.Resampling.LANCZOS)


class InlineImages:
    """Sends each picture in the request itself, for a provider that cannot fetch URLs.

    Azure OpenAI's fetcher times out on presigned URLs that OpenAI direct fetches fine
    ("Unable to download content from the provided URL before the timeout", 400). This wraps
    a URL resolver (such as :class:`~actant.storage.s3.S3AssetResolver`): each picture is
    read from its URL, `read_attempts` times at most, and resolved to bytes fitted by
    :func:`fit_image`. A resolver that reads bytes itself (:class:`AssetReader`) is read
    once instead, with no URL and no existence check: its client's own retries and timeouts
    apply, and a cold picture costs one GET on a pooled connection, not a HEAD and a GET on
    a new one. One the store will not give fails the turn rather than going as a URL; one
    that is no picture, or larger than `read_bytes`, is missing, said so. The choice depends
    on the picture alone, so every turn sends a picture the same way and the prompt cache
    keeps matching.

    A stored picture never changes (new bytes get a new key), so a fitted one is kept,
    `cache_bytes` of them, the least recently sent dropped first: a thread's next turn sends
    its history's pictures from memory, and turns asking for one picture at once share a
    read. Needs Pillow (``actant[images]``).
    """

    def __init__(
        self,
        resolver: AssetResolver,
        *,
        image_side: int = INLINE_IMAGE_SIDE,
        image_bytes: int = INLINE_IMAGE_BYTES,
        jpeg_quality: int = INLINE_JPEG_QUALITY,
        read_bytes: int = READ_IMAGE_BYTES,
        read_attempts: int = 3,
        read_timeout_s: float = 30,
        cache_bytes: int = INLINE_CACHE_BYTES,
    ) -> None:
        if read_attempts < 1:
            raise ValueError("read_attempts must be positive")
        self.resolver = resolver
        self.image_side, self.image_bytes, self.jpeg_quality = (
            image_side,
            image_bytes,
            jpeg_quality,
        )
        self.read_bytes, self.read_attempts = read_bytes, read_attempts
        self.read_timeout_s = read_timeout_s
        self.cache_bytes = cache_bytes
        self._kept: OrderedDict[str, ResolvedImage] = OrderedDict()
        self._kept_bytes = 0
        self._reading: dict[str, asyncio.Future[ResolvedImage | MissingAsset]] = {}

    async def resolve(
        self, asset: AssetReference, context: AssetContext
    ) -> ResolvedImage | MissingAsset:
        key = asset.storage_key
        if (kept := self._kept.get(key)) is not None:
            self._kept.move_to_end(key)
            return kept
        if (reading := self._reading.get(key)) is not None:
            return await asyncio.shield(reading)
        reading = self._reading[key] = asyncio.ensure_future(self._read(asset, context))
        try:
            found = await asyncio.shield(reading)
        finally:
            if reading.done():
                del self._reading[key]
            else:  # this caller was cancelled: whoever asks next shares the read under way
                reading.add_done_callback(lambda _: self._reading.pop(key, None))
        if isinstance(found, ResolvedImage) and found.data is not None:
            self._keep(key, found)
        return found

    def _keep(self, key: str, image: ResolvedImage) -> None:
        size = len(image.data or b"")
        if size > self.cache_bytes:
            return
        self._kept[key] = image
        self._kept_bytes += size
        while self._kept_bytes > self.cache_bytes:
            _, dropped = self._kept.popitem(last=False)
            self._kept_bytes -= len(dropped.data or b"")

    async def _read(
        self, asset: AssetReference, context: AssetContext
    ) -> ResolvedImage | MissingAsset:
        if isinstance(self.resolver, AssetReader):
            read = await self.resolver.read(asset, self.read_bytes)
            if isinstance(read, MissingAsset):
                return read
            return await self._fit(asset, read)
        found = await self.resolver.resolve(asset, context)
        if isinstance(found, MissingAsset) or found.url is None:
            return found
        data: bytes | None = None
        for attempt in range(self.read_attempts):
            try:
                data = await asyncio.to_thread(
                    read_image, found.url, self.read_bytes, self.read_timeout_s
                )
                break
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == self.read_attempts - 1:
                    raise
                logger.warning(
                    "image %s: read %d failed: %s", asset.storage_key, attempt + 1, error
                )
                await asyncio.sleep(2**attempt)
        return await self._fit(asset, data)

    async def _fit(
        self, asset: AssetReference, data: bytes | None
    ) -> ResolvedImage | MissingAsset:
        from PIL import UnidentifiedImageError

        if data is None:
            return MissingAsset(f"{asset.storage_key}: larger than a whole request's images")
        try:
            data, mime = await asyncio.to_thread(
                fit_image, data, self.image_side, self.image_bytes, self.jpeg_quality
            )
        except UnidentifiedImageError:
            return MissingAsset(f"{asset.storage_key}: not a picture")
        return ResolvedImage(mime, data=data)
