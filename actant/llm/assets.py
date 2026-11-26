"""Resolve a transcript's asset references into blocks a provider can send."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace

from actant.infra.assets import AssetContext, AssetReference, AssetResolver, MissingAsset
from actant.infra.blocks import AssetBlock, PromptBlock, TextBlock
from actant.llm.messages import Message


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
