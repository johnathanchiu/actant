"""Shared helpers used by two or more provider adapters.

Single-provider helpers live in their owning provider module:
- ``REASONING_EFFORT``, ``content_to_openai_user_parts`` → ``openai.py``
- ``dereference_schema``, ``strip_unsupported_schema_keys`` → ``gemini.py``
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from typing import NoReturn

from actant.blocks import AssetBlock, InlineImageBlock, PromptBlock, TextBlock, UrlImageBlock
from actant.llm.messages import Message, ToolCall

ToolSchema = dict[str, object]
#: A content block in a provider's own request shape.
WireBlock = dict[str, object]


def env_api_key(name: str, explicit: str | None = None) -> str:
    key = explicit or os.environ.get(name)
    if not key:
        raise RuntimeError(f"{name} is required for this provider adapter.")
    return key


def normalize_json_schema(schema: object) -> object:
    if not isinstance(schema, dict):
        return schema

    result: dict[str, object] = {}
    for key, value in schema.items():
        if key == "prefixItems":
            if value and isinstance(value, list):
                result["items"] = normalize_json_schema(value[0])
            continue
        if key == "$defs" and isinstance(value, dict):
            result[key] = {k: normalize_json_schema(v) for k, v in value.items()}
        elif isinstance(value, dict):
            result[key] = normalize_json_schema(value)
        elif isinstance(value, list):
            result[key] = [normalize_json_schema(item) for item in value]
        else:
            result[key] = value
    return result


def unresolved(block: AssetBlock) -> NoReturn:
    raise ValueError(
        f"asset {block.storage_key!r} reached a provider unresolved: run prepare_messages first"
    )


def convert_image_source(block: InlineImageBlock | UrlImageBlock) -> WireBlock:
    """An image as an OpenAI Responses ``input_image``."""
    if isinstance(block, UrlImageBlock):
        url = block.url
    else:
        url = f"data:{block.source.media_type};base64,{block.source.data}"
    return {"type": "input_image", "image_url": url, "detail": "high"}


def split_tool_content(
    content: str | list[PromptBlock] | None,
) -> tuple[str, list[WireBlock]]:
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []

    text_parts: list[str] = []
    image_parts: list[WireBlock] = []
    for block in content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, AssetBlock):
            unresolved(block)
        else:
            image_parts.append(convert_image_source(block))
    return "\n".join(text_parts) if text_parts else "OK", image_parts


def sanitize_tool_messages(
    messages: Sequence[Message | dict[str, object]],
) -> list[Message]:
    """Messages every adapter can send: tool calls and results paired by id."""
    sanitized: list[Message] = []
    pending_ids: list[str] = []

    for raw_message in messages:
        message = Message.from_raw(raw_message)
        if message.role == "assistant" and message.tool_calls is not None:
            normalized_tool_calls: list[ToolCall] = []
            for raw_tool_call in message.tool_calls:
                tool_call = ToolCall.from_raw(raw_tool_call)
                if not tool_call.id:
                    tool_call.id = f"call_{uuid.uuid4().hex}"
                pending_ids.append(tool_call.id)
                normalized_tool_calls.append(tool_call)
            message.tool_calls = normalized_tool_calls
        elif message.role == "tool":
            tool_call_id = message.tool_call_id or ""
            if tool_call_id:
                if pending_ids and pending_ids[0] == tool_call_id:
                    pending_ids.pop(0)
                elif tool_call_id in pending_ids:
                    pending_ids.remove(tool_call_id)
            elif pending_ids:
                message.tool_call_id = pending_ids.pop(0)
            else:
                message.tool_call_id = f"call_{uuid.uuid4().hex}"
        sanitized.append(message)

    return sanitized
