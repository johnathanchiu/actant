"""The typed content blocks of a message.

:data:`Block` is what history stores: a user prompt's or a tool result's content, in
``actant_message_parts.content_blocks``. None of its kinds carries a URL; an uploaded
image is an :class:`AssetBlock` (a storage key), which :func:`actant.assets.prepare_messages`
resolves for each model call. :data:`PromptBlock` adds the one kind only that
preparation produces, :class:`UrlImageBlock`, and is what the provider adapters read.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

_IMAGE_MIME = r"^image/[\w.+-]+$"


class _Block(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TextBlock(_Block):
    type: Literal["text"] = "text"
    text: str


class AssetBlock(_Block):
    """An image in the product's bucket, by key; never a URL."""

    type: Literal["asset"] = "asset"
    storage_key: Annotated[str, Field(min_length=1)]
    mime: Annotated[str, Field(pattern=_IMAGE_MIME)]


class Base64Source(_Block):
    type: Literal["base64"] = "base64"
    media_type: Annotated[str, Field(pattern=_IMAGE_MIME)]
    data: str


class InlineImageBlock(_Block):
    """Image bytes a tool returned with no bucket to upload them to, as local runs do."""

    type: Literal["image"] = "image"
    source: Base64Source


class UrlImageBlock(_Block):
    """An image the provider fetches: made by :func:`actant.assets.prepare_messages` for one
    model call from an :class:`AssetBlock`, and never stored."""

    type: Literal["image_url"] = "image_url"
    url: str
    media_type: str


Block = Annotated[TextBlock | AssetBlock | InlineImageBlock, Field(discriminator="type")]
PromptBlock = Annotated[
    TextBlock | AssetBlock | InlineImageBlock | UrlImageBlock, Field(discriminator="type")
]

#: Validates and dumps stored content: ``BLOCKS.validate_python(rows)``,
#: ``BLOCKS.dump_python(blocks, mode="json")``.
BLOCKS: TypeAdapter[list[Block]] = TypeAdapter(list[Block])
PROMPT_BLOCKS: TypeAdapter[list[PromptBlock]] = TypeAdapter(list[PromptBlock])
