"""Structured message persistence helpers."""

from __future__ import annotations

import json
from typing import cast

from actant.blocks import BLOCKS, Block, PromptBlock
from actant.core import JSONObject, JSONValue
from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.runtime.types.session import MessagePart, PartKind, WaitStatus


def message_to_parts(message: Message) -> list[MessagePart]:
    parts: list[MessagePart] = []
    if message.role == "user":
        if isinstance(message.content, list):
            parts.append(
                MessagePart(
                    kind=PartKind.USER_PROMPT,
                    content_blocks=BLOCKS.validate_python(message.content),
                )
            )
        else:
            parts.append(
                MessagePart(kind=PartKind.USER_PROMPT, content=str(message.content or ""))
            )
        return parts

    if message.thought_summary:
        parts.append(
            MessagePart(
                kind=PartKind.THINKING,
                content=message.thought_summary,
                signature=message.thinking_signature,
                reasoning_items=message.reasoning_items,
            )
        )
    if message.content:
        parts.append(MessagePart(kind=PartKind.TEXT, content=str(message.content)))
    for tool_call in message.tool_calls or []:
        parts.append(
            MessagePart(
                kind=PartKind.TOOL_CALL,
                tool_call_id=tool_call.id,
                tool_name=tool_call.function.name,
                args=_args_to_object(tool_call.function.arguments),
                signature=tool_call.thought_signature,
            )
        )
    return parts


def parts_to_messages(parts: list[MessagePart]) -> list[Message]:
    messages: list[Message] = []
    assistant_text: list[str] = []
    thought_summary: str | None = None
    thinking_signature: str | None = None
    reasoning_items: list[object] | None = None
    tool_calls: list[ToolCall] = []
    tool_results: list[Message] = []

    for part in parts:
        if part.kind == PartKind.USER_PROMPT:
            _flush_assistant(
                messages,
                assistant_text,
                tool_calls,
                tool_results,
                thought_summary,
                thinking_signature,
                reasoning_items,
            )
            thought_summary = None
            thinking_signature = None
            reasoning_items = None
            user_content: str | list[PromptBlock]
            if part.content_blocks:
                user_content = list(part.content_blocks)
            else:
                user_content = part.content or ""
            messages.append(Message(role="user", content=user_content))
        elif part.kind == PartKind.TEXT:
            assistant_text.append(part.content or "")
        elif part.kind == PartKind.THINKING:
            thought_summary = part.content
            thinking_signature = part.signature
            reasoning_items = part.reasoning_items
        elif part.kind == PartKind.TOOL_CALL and part.tool_call_id and part.tool_name:
            tool_calls.append(
                ToolCall(
                    id=part.tool_call_id,
                    function=ToolCallFunction(
                        name=part.tool_name,
                        arguments=json.dumps(part.args or {}),
                    ),
                    thought_signature=part.signature,
                    extra_content=(
                        {"google": {"thought_signature": part.signature}} if part.signature else {}
                    ),
                )
            )
            if part.result is not None:
                tool_results.append(
                    Message(
                        role="tool",
                        content=tool_result_content(part.result),
                        tool_call_id=part.tool_call_id,
                        name=part.tool_name,
                    )
                )
            elif part.wait_status == WaitStatus.DENIED:
                tool_results.append(
                    Message(
                        role="tool",
                        content=json.dumps({"error": "Waiting tool call denied"}),
                        tool_call_id=part.tool_call_id,
                        name=part.tool_name,
                    )
                )
    _flush_assistant(
        messages,
        assistant_text,
        tool_calls,
        tool_results,
        thought_summary,
        thinking_signature,
        reasoning_items,
    )
    return messages


def _flush_assistant(
    messages: list[Message],
    text_parts: list[str],
    tool_calls: list[ToolCall],
    tool_results: list[Message],
    thought_summary: str | None,
    thinking_signature: str | None,
    reasoning_items: list[object] | None,
) -> None:
    if text_parts or tool_calls or thought_summary or reasoning_items:
        messages.append(
            Message(
                role="assistant",
                content="".join(text_parts),
                tool_calls=list(tool_calls) or None,
                thought_summary=thought_summary,
                thinking_signature=thinking_signature,
                reasoning_items=reasoning_items,
            )
        )
        messages.extend(tool_results)
    text_parts.clear()
    tool_calls.clear()
    tool_results.clear()


def tool_result_blocks(result: object) -> list[Block] | None:
    """The blocks under a tool result's ``content_blocks`` key (:meth:`ToolResult.to_dict`),
    validated; ``None`` when it has none."""
    if not isinstance(result, dict):
        return None
    blocks = result.get("content_blocks")
    if blocks is None:
        return None
    return BLOCKS.validate_python(blocks) or None


def tool_result_content(result: object) -> str | list[PromptBlock]:
    """A persisted tool result as ``Message.content``: its blocks when it has any, which
    :func:`~actant.assets.prepare_messages` later resolves, otherwise its JSON."""
    blocks = tool_result_blocks(result)
    return list(blocks) if blocks else json.dumps(result)


def _args_to_object(arguments: str) -> JSONObject:
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return cast(JSONObject, _json_dict(parsed))
    return {}


def _json_dict(value: dict[str, object]) -> JSONObject:
    result: JSONObject = {}
    for key, item in value.items():
        if _is_json_value(item):
            result[key] = cast(JSONValue, item)
    return result


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False
