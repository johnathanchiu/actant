"""Durable media references and request-time resolution, independent of storage vendors."""

from __future__ import annotations

import base64
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from actant.llm.messages import Message


@dataclass(frozen=True)
class AssetReference:
    storage_key: str
    mime: str

    def to_block(self) -> dict[str, object]:
        return {"type": "asset", "storage_key": self.storage_key, "mime": self.mime}


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

    def to_block(self, minimum_validity_s: float) -> dict[str, object]:
        if (self.url is None) == (self.data is None):
            raise ValueError("resolved image requires exactly one of url or data")
        if self.url is not None:
            if not self.url.startswith(("https://", "http://")):
                raise ValueError("resolved image URL must use HTTP(S)")
            if self.expires_at is not None and self.expires_at <= time.time() + minimum_validity_s:
                raise ValueError("resolved image URL does not cover the model-call budget")
            source: dict[str, object] = {"type": "url", "url": self.url}
        else:
            source = {
                "type": "base64",
                "media_type": self.mime,
                "data": base64.b64encode(self.data or b"").decode(),
            }
        return {"type": "image", "source": source}


@dataclass(frozen=True)
class MissingAsset:
    reason: str = "asset no longer available"


class AssetResolver(Protocol):
    async def resolve(
        self, asset: AssetReference, context: AssetContext
    ) -> ResolvedImage | MissingAsset: ...


async def prepare_messages(
    messages: Sequence[Message],
    resolver: AssetResolver | None,
    context: AssetContext,
) -> list[Message]:
    """Resolve stored blocks without mutating transcript or discarding message metadata.

    Legacy URL blocks can only be recovered when they carry a storage key. Never
    derive a key from an untrusted URL or guess an object's retention from its age.
    Missing bytes are visible; resolver failures propagate to execution.
    """
    prepared: list[Message] = []
    for message in messages:
        if not isinstance(message.content, list):
            prepared.append(message)
            continue
        blocks: list[dict[str, object]] = []
        for block in message.content:
            reference: AssetReference | None = None
            if block.get("type") == "asset":
                key, mime = block.get("storage_key"), block.get("mime")
                if not isinstance(key, str) or not key or not isinstance(mime, str) or not mime:
                    raise ValueError("asset requires nonempty storage_key and mime")
                reference = AssetReference(key, mime)
            elif block.get("type") == "image":
                source = block.get("source")
                if isinstance(source, dict) and source.get("type") == "url":
                    expiry = source.get("expires_at")
                    if (
                        isinstance(expiry, (int, float))
                        and expiry <= time.time() + context.minimum_validity_s
                    ):
                        key = source.get("key")
                        if isinstance(key, str) and key:
                            mime = source.get("media_type", "image/png")
                            reference = AssetReference(key, str(mime))
                        else:
                            blocks.append(
                                {
                                    "type": "text",
                                    "text": "[Image unavailable: legacy URL expired; no storage reference]",
                                }
                            )
                            continue
                    else:
                        # Storage metadata never reaches provider wire payloads.
                        blocks.append(
                            {
                                **block,
                                "source": {
                                    k: v
                                    for k, v in source.items()
                                    if k not in {"key", "expires_at"}
                                },
                            }
                        )
                        continue
            if reference is None:
                blocks.append(block)
            elif not reference.mime.startswith("image/"):
                blocks.append(
                    {
                        "type": "text",
                        "text": f"[Attached file: mime={reference.mime}, asset_storage_key={reference.storage_key}]",
                    }
                )
            else:
                if resolver is None:
                    raise ValueError("asset references require an AssetResolver")
                image = await resolver.resolve(reference, context)
                if isinstance(image, MissingAsset):
                    blocks.append(
                        {
                            "type": "text",
                            "text": f"[Image unavailable: {image.reason}; asset_storage_key={reference.storage_key}]",
                        }
                    )
                else:
                    blocks.append(image.to_block(context.minimum_validity_s))
        prepared.append(replace(message, content=blocks))
    return prepared
