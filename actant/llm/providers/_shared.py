"""Shared helpers used by two or more provider adapters.

Single-provider helpers live in their owning provider module:
- ``REASONING_EFFORT``, ``content_to_openai_user_parts`` → ``openai.py``
- ``dereference_schema``, ``strip_unsupported_schema_keys`` → ``gemini.py``
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Mapping, Sequence

from actant.llm.messages import Message, ToolCall

ToolSchema = dict[str, object]
ContentBlock = dict[str, object]
#: A URL image expiring within this long is replaced, so the provider's fetch never races it.
EXPIRY_MARGIN_S = 120.0
EXPIRED_IMAGE = "[image expired: its URL is no longer valid]"


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


def convert_image_source(source: Mapping[str, object]) -> ContentBlock | None:
    if source.get("type") == "base64":
        return {
            "type": "input_image",
            "image_url": f"data:{source['media_type']};base64,{source['data']}",
        }
    if source.get("type") == "url":
        return {"type": "input_image", "image_url": source["url"]}
    return None


def split_tool_content(
    content: str | list[ContentBlock] | None,
) -> tuple[str, list[ContentBlock]]:
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []

    text_parts: list[str] = []
    image_parts: list[ContentBlock] = []
    for block in content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
        elif block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
        elif block.get("type") == "image":
            source = block.get("source")
            image = convert_image_source(source) if isinstance(source, Mapping) else None
            if image:
                image_parts.append(image)
    return "\n".join(text_parts) if text_parts else "OK", image_parts


def live_image_urls(content: list[ContentBlock], now: float) -> list[ContentBlock]:
    """``content`` with each URL image source's ``expires_at`` and ``key`` removed (no provider
    accepts them), and an image expiring before ``now + EXPIRY_MARGIN_S`` replaced by a text note: a
    provider rejects the whole request when it cannot fetch one."""
    live: list[ContentBlock] = []
    for block in content:
        source = block.get("source") if isinstance(block, dict) else None
        if not isinstance(source, Mapping) or block.get("type") != "image":
            live.append(block)
            continue
        expires_at = source.get("expires_at")
        if source.get("type") != "url" or expires_at is None:
            live.append(block)
        elif isinstance(expires_at, int | float) and expires_at > now + EXPIRY_MARGIN_S:
            kept = {k: v for k, v in source.items() if k not in {"expires_at", "key"}}
            live.append({**block, "source": kept})
        else:
            live.append({"type": "text", "text": EXPIRED_IMAGE})
    return live


def sanitize_tool_messages(
    messages: Sequence[Message | dict[str, object]],
) -> list[Message]:
    """Messages every adapter can send: tool calls and results paired by id, and URL
    images that would expire before the request lands replaced (:func:`live_image_urls`)."""
    sanitized: list[Message] = []
    pending_ids: list[str] = []
    now = time.time()

    for raw_message in messages:
        message = Message.from_raw(raw_message)
        if isinstance(message.content, list):
            message.content = live_image_urls(message.content, now)
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
