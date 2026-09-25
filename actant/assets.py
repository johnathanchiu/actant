"""Durable media references and request-time resolution, independent of storage vendors."""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from actant.blocks import (
    AssetBlock,
    Base64Source,
    InlineImageBlock,
    PromptBlock,
    TextBlock,
    UrlImageBlock,
)
from actant.llm.messages import Message


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


#: How many asset references one request resolves at once.
RESOLVE_CONCURRENCY = 16


async def prepare_messages(
    messages: Sequence[Message],
    resolver: AssetResolver | None,
    context: AssetContext,
) -> list[Message]:
    """Resolve each :class:`AssetBlock` for this model call, without mutating the transcript
    or discarding message metadata. A request's references resolve concurrently, at most
    :data:`RESOLVE_CONCURRENCY` at a time. Missing bytes are visible; resolver failures
    propagate to execution.
    """
    assets = [
        block
        for message in messages
        if isinstance(message.content, list)
        for block in message.content
        if isinstance(block, AssetBlock)
    ]
    if not assets:
        return list(messages)
    if resolver is None:
        raise ValueError("asset references require an AssetResolver")
    limit = asyncio.Semaphore(RESOLVE_CONCURRENCY)

    async def resolve(block: AssetBlock) -> PromptBlock:
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
        blocks = [next(resolved) if isinstance(b, AssetBlock) else b for b in message.content]
        prepared.append(replace(message, content=blocks))
    return prepared
