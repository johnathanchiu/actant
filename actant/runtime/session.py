"""Structured message persistence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import cast

from actant.core import JSONObject, JSONValue
from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.runtime.types.session import MessagePart, PartKind, WaitStatus


def message_to_parts(message: Message) -> list[MessagePart]:
    parts: list[MessagePart] = []
    if message.role == "user":
        if isinstance(message.content, list):
            # Multimodal user message — text + asset blocks. Persist as
            # content_blocks so the round-trip preserves the structure.
            parts.append(
                MessagePart(
                    kind=PartKind.USER_PROMPT,
                    content_blocks=_normalize_content_blocks(message.content),
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
            # Multimodal user message: rebuild Message.content as the
            # block list. Falls back to the text content for the
            # legacy / string-only case.
            user_content: str | list[dict[str, object]]
            if part.content_blocks:
                user_content = part.content_blocks
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
                        {"google": {"thought_signature": part.signature}}
                        if part.signature
                        else {}
                    ),
                )
            )
            if part.result is not None:
                tool_results.append(
                    Message(
                        role="tool",
                        content=_tool_result_content(part.result),
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


def _normalize_content_blocks(value: object) -> list[dict[str, object]]:
    """Coerce list-typed Message.content into the canonical block list shape.

    Each entry must be a dict (text or asset block); non-dict entries
    get dropped so a malformed payload can't poison persistence.
    """
    if not isinstance(value, list):
        return []
    return [block for block in value if isinstance(block, dict)]


def _tool_result_content(result: dict[str, object]) -> str | list[dict[str, object]]:
    """Pick the right ``Message.content`` shape for a persisted tool result.

    Tools that produce mixed text + asset output set a ``content_blocks``
    key on their ``ToolResult.metadata`` (or directly on the result dict).
    When present, surface that as a list — the LLM-layer asset resolver
    later expands the asset refs into provider-shaped image blocks.
    Falls back to JSON-stringifying the raw result for legacy text-only
    tool returns.
    """
    blocks = result.get("content_blocks")
    if isinstance(blocks, list):
        normalized = _normalize_content_blocks(blocks)
        if normalized:
            return normalized
    return json.dumps(result)


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


@dataclass
class InMemorySessionStore:
    _messages_by_thread: dict[str, list[list[MessagePart]]] = field(default_factory=dict)
    _message_counter: int = 0

    async def save_user_message(
        self, thread_id: str, content: str | list[dict[str, object]]
    ) -> str:
        if isinstance(content, list):
            blocks = [block for block in content if isinstance(block, dict)]
            part = MessagePart(kind=PartKind.USER_PROMPT, content_blocks=blocks or None)
        else:
            part = MessagePart(kind=PartKind.USER_PROMPT, content=content)
        return await self._append(thread_id, [part])

    async def save_assistant_message(self, thread_id: str, parts: list[MessagePart]) -> str:
        return await self._append(thread_id, parts)

    async def update_tool_result(
        self,
        thread_id: str,
        tool_call_id: str,
        result: dict[str, object],
    ) -> None:
        part = self._find_tool_call(thread_id, tool_call_id)
        if part is not None:
            part.result = result
            part.wait_status = None

    async def update_wait_status(
        self,
        thread_id: str,
        tool_call_id: str,
        status: WaitStatus,
    ) -> None:
        part = self._find_tool_call(thread_id, tool_call_id)
        if part is not None:
            part.wait_status = status

    async def get_conversation(self, thread_id: str) -> list[Message]:
        messages: list[Message] = []
        for parts in self._messages_by_thread.get(thread_id, []):
            messages.extend(parts_to_messages(list(parts)))
        return messages

    async def _append(self, thread_id: str, parts: list[MessagePart]) -> str:
        self._messages_by_thread.setdefault(thread_id, []).append(parts)
        self._message_counter += 1
        return f"msg_{self._message_counter}"

    def _find_tool_call(self, thread_id: str, tool_call_id: str) -> MessagePart | None:
        for parts in self._messages_by_thread.get(thread_id, []):
            for part in parts:
                if part.kind == PartKind.TOOL_CALL and part.tool_call_id == tool_call_id:
                    return part
        return None
