"""OpenAI Responses API adapter."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

import openai
from openai.types.responses.response_create_params import (
    ResponseCreateParamsBase,
)
from openai.types.responses.response_includable import ResponseIncludable
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses.tool_param import ToolParam
from openai.types.shared_params.reasoning import Reasoning

from actant.llm.errors import StreamCancelled
from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.llm.providers._shared import (
    ContentBlock,
    convert_image_source,
    env_api_key,
    normalize_json_schema,
    sanitize_tool_messages,
    split_tool_content,
)
from actant.llm.rate_limit import RateLimiter

if TYPE_CHECKING:
    from actant.runtime.events.streaming import StreamListener

logger = logging.getLogger(__name__)

REASONING_MODELS = ("gpt-5", "o1", "o3", "o4")
REASONING_EFFORT: dict[str, str] = {
    "low": "low",
    "med": "medium",
    "medium": "medium",
    "high": "high",
}
ToolSchema = dict[str, object]
RequestParams = ResponseCreateParamsBase


class _ToolStreamState(Protocol):
    ids_by_key: dict[str, str]
    names_by_id: dict[str, str]
    started: set[str]


def content_to_openai_user_parts(
    content: str | list[ContentBlock] | None,
) -> list[ContentBlock]:
    if content is None:
        return [{"type": "input_text", "text": ""}]
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]

    parts: list[ContentBlock] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append({"type": "input_text", "text": str(block)})
        elif block.get("type") == "text":
            parts.append({"type": "input_text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            source = block.get("source")
            image = convert_image_source(source) if isinstance(source, Mapping) else None
            if image:
                parts.append(image)
    return parts or [{"type": "input_text", "text": ""}]


class OpenAIProvider:
    """LLMClient implementation for OpenAI's Responses API."""

    def __init__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        thinking_level: str = "med",
        client: openai.AsyncOpenAI | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.model_id = model_id
        self.thinking_level = thinking_level
        self.client = client or openai.AsyncOpenAI(api_key=env_api_key("OPENAI_API_KEY", api_key))
        self._rate_limiter = rate_limiter

    def _is_reasoning_model(self) -> bool:
        return any(self.model_id.startswith(prefix) for prefix in REASONING_MODELS)

    @staticmethod
    def convert_tools(tools: list[dict]) -> list[ToolSchema]:
        responses_tools: list[ToolSchema] = []
        for tool in tools or []:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict):
                continue
            responses_tools.append(
                {
                    "type": "function",
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "parameters": normalize_json_schema(function.get("parameters", {})),
                    "strict": False,
                }
            )
        return responses_tools

    @classmethod
    def convert_messages(cls, messages: Sequence[Message]) -> list[ToolSchema]:
        items: list[ToolSchema] = []
        for message in messages:
            if message.role == "user":
                items.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": content_to_openai_user_parts(message.content),
                    }
                )
            elif message.role == "assistant":
                items.extend(cls._convert_assistant_message(message))
            elif message.role == "tool":
                items.extend(cls._convert_tool_message(message))
        return items

    @staticmethod
    def _convert_assistant_message(message: Message) -> list[ToolSchema]:
        items: list[ToolSchema] = []
        if message.reasoning_items is not None:
            items.extend(cast(list[ToolSchema], message.reasoning_items))
        for tool_call in message.tool_calls or []:
            call_id = tool_call.id or f"call_{uuid.uuid4().hex}"
            items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
            )
        if message.content:
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": message.content}],
                }
            )
        return items

    @staticmethod
    def _convert_tool_message(message: Message) -> list[ToolSchema]:
        items: list[ToolSchema] = []
        text, image_parts = split_tool_content(message.content)
        items.append(
            {
                "type": "function_call_output",
                "call_id": message.tool_call_id or f"call_{uuid.uuid4().hex}",
                "output": text,
            }
        )
        if image_parts:
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        *image_parts,
                        {
                            "type": "input_text",
                            "text": "[Tool result images for the above function call]",
                        },
                    ],
                }
            )
        return items

    def _request_params(
        self, system: str, messages: Sequence[Message], tools: list[dict]
    ) -> RequestParams:
        params: RequestParams = {
            "model": self.model_id,
            "instructions": system,
            "input": cast(
                ResponseInputParam,
                self.convert_messages(sanitize_tool_messages(messages)),
            ),
            "store": False,
        }
        converted_tools = self.convert_tools(tools)
        if converted_tools:
            params["tools"] = cast(list[ToolParam], converted_tools)
        if self._is_reasoning_model():
            params["reasoning"] = cast(
                Reasoning,
                {
                    "effort": REASONING_EFFORT.get(self.thinking_level, "medium"),
                    "summary": "auto",
                },
            )
            params["include"] = cast(
                list[ResponseIncludable],
                ["reasoning.encrypted_content"],
            )
        return params

    async def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: list[dict],
        listener: "StreamListener | None" = None,
    ) -> Message:
        params = self._request_params(system, messages, tools)
        if self._rate_limiter is None:
            message, _ = await self._stream(params, listener)
            return message
        estimated = self._estimate_tokens(messages, params)
        try:
            async with self._rate_limiter.reserve(estimated) as reservation:
                message, actual = await self._stream(params, listener)
                reservation.record_actual(actual)
                return message
        except openai.RateLimitError as exc:
            # The bucket's estimate was off (most often: a reasoning
            # model's hidden thinking tokens). Honor the server's
            # retry-after exactly once before re-reserving and trying
            # again. If we miss twice in a row the budget is
            # mis-configured and we re-raise so Actant's job retry can
            # take over (or fail loudly).
            wait = _parse_retry_after(exc) or 5.0
            logger.warning(
                "actant.openai.rate_limit_miss model=%s wait_secs=%.2f error=%s",
                self.model_id,
                wait,
                exc,
            )
            await asyncio.sleep(wait + 0.5)
            async with self._rate_limiter.reserve(estimated) as reservation:
                message, actual = await self._stream(params, listener)
                reservation.record_actual(actual)
                return message

    async def _stream(
        self,
        params: RequestParams,
        listener: "StreamListener | None",
    ) -> tuple[Message, int]:
        tool_stream_state = _OpenAIToolStreamState()
        async with self.client.responses.stream(**params) as stream:
            async for event in stream:
                if listener is not None and listener.cancel_requested():
                    raise StreamCancelled
                if listener is None:
                    continue
                await _forward_stream_event(event, listener, tool_stream_state)
            response = await stream.get_final_response()

        text = ""
        thought = ""
        tool_calls: list[ToolCall] = []
        reasoning_items: list[ToolSchema] = []

        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for block in getattr(item, "content", []) or []:
                    if getattr(block, "type", None) in ("output_text", "text"):
                        text += getattr(block, "text", "") or ""
            elif item_type == "function_call":
                tool_calls.append(_function_call_to_tool_call(item))
            elif item_type == "reasoning":
                for summary in getattr(item, "summary", []) or []:
                    thought += getattr(summary, "text", "") or ""
                if reasoning_item := _extract_reasoning_item(item):
                    reasoning_items.append(reasoning_item)

        message = Message(
            role="assistant",
            content=text or None,
            tool_calls=tool_calls or None,
            thought_summary=thought or None,
            reasoning_items=cast(list[object], reasoning_items) or None,
        )
        usage = getattr(response, "usage", None)
        actual_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return message, actual_tokens

    def _estimate_tokens(self, messages: Sequence[Message], params: RequestParams) -> int:
        # Conservative char-heuristic. ~4 chars/token is OpenAI's rule
        # of thumb; we use 3 to overestimate slightly so the bucket
        # rarely misses. Reasoning models burn additional tokens
        # internally so we double the input estimate for them.
        char_total = sum(_message_chars(m) for m in messages)
        input_estimate = (char_total // 3) + 200
        if self._is_reasoning_model():
            input_estimate *= 2
        output_ceiling = int(params.get("max_output_tokens") or 2048)
        return input_estimate + output_ceiling


def _extract_reasoning_item(item: object) -> ToolSchema | None:
    encrypted_content = getattr(item, "encrypted_content", None)
    if not encrypted_content:
        return None
    return {
        "id": getattr(item, "id", None),
        "type": "reasoning",
        "summary": [
            {"type": getattr(s, "type", "summary_text"), "text": getattr(s, "text", "")}
            for s in getattr(item, "summary", []) or []
        ],
        "encrypted_content": encrypted_content,
    }


class _OpenAIToolStreamState:
    def __init__(self) -> None:
        self.ids_by_key: dict[str, str] = {}
        self.names_by_id: dict[str, str] = {}
        self.started: set[str] = set()


async def _forward_stream_event(
    event: object,
    listener: "StreamListener",
    state: _ToolStreamState,
) -> None:
    event_type = getattr(event, "type", None)
    if event_type == "response.output_text.delta":
        await listener.on_text_delta(getattr(event, "delta", "") or "")
        return
    if event_type == "response.reasoning_summary_text.delta":
        await listener.on_thinking_delta(getattr(event, "delta", "") or "")
        return
    if event_type == "response.output_item.added":
        item = getattr(event, "item", None)
        if getattr(item, "type", None) != "function_call":
            return
        tool_call_id = _stream_tool_call_id(item)
        tool_name = getattr(item, "name", "") or ""
        _remember_stream_tool_call(event, item, tool_call_id, tool_name, state)
        await _emit_tool_call_start(listener, tool_call_id, tool_name, state)
        return
    if event_type == "response.function_call_arguments.delta":
        tool_call_id = _lookup_stream_tool_call_id(event, state)
        if tool_call_id is None:
            return
        tool_name = state.names_by_id.get(tool_call_id, "")
        await _emit_tool_call_start(listener, tool_call_id, tool_name, state)
        await listener.on_tool_call_args_delta(tool_call_id, getattr(event, "delta", "") or "")
        return
    if event_type == "response.function_call_arguments.done":
        tool_call_id = _lookup_stream_tool_call_id(event, state)
        if tool_call_id is not None:
            await listener.on_tool_call_args_complete(tool_call_id)


async def _emit_tool_call_start(
    listener: "StreamListener",
    tool_call_id: str,
    tool_name: str,
    state: _ToolStreamState,
) -> None:
    if tool_call_id in state.started:
        return
    state.started.add(tool_call_id)
    await listener.on_tool_call_start(tool_call_id, tool_name)


def _remember_stream_tool_call(
    event: object,
    item: object,
    tool_call_id: str,
    tool_name: str,
    state: _ToolStreamState,
) -> None:
    state.names_by_id[tool_call_id] = tool_name
    for key in _stream_keys(event):
        state.ids_by_key[key] = tool_call_id
    for key in _stream_keys(item):
        state.ids_by_key[key] = tool_call_id


def _lookup_stream_tool_call_id(event: object, state: _ToolStreamState) -> str | None:
    for key in _stream_keys(event):
        tool_call_id = state.ids_by_key.get(key)
        if tool_call_id is not None:
            return tool_call_id
    return None


def _stream_keys(obj: object) -> list[str]:
    keys: list[str] = []
    for attr in ("item_id", "id"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and value:
            keys.append(f"{attr}:{value}")
    output_index = getattr(obj, "output_index", None)
    if isinstance(output_index, int):
        keys.append(f"output_index:{output_index}")
    return keys


def _stream_tool_call_id(item: object) -> str:
    return (
        getattr(item, "call_id", None) or getattr(item, "id", None) or f"call_{uuid.uuid4().hex}"
    )


def _function_call_to_tool_call(item: object) -> ToolCall:
    call_id = (
        getattr(item, "call_id", None) or getattr(item, "id", None) or f"call_{uuid.uuid4().hex}"
    )
    return ToolCall(
        id=call_id,
        function=ToolCallFunction(
            name=getattr(item, "name", "") or "",
            arguments=getattr(item, "arguments", "") or "",
        ),
    )


def _message_chars(message: Message) -> int:
    """Rough character count for token estimation. Tool-call args are
    serialized but image bytes / encrypted reasoning blobs aren't —
    those tend to dominate when present and would overshoot the
    estimate by 10x+ if naively counted."""
    total = 0
    content = message.content
    if isinstance(content, str):
        total += len(content)
    elif isinstance(content, list):
        for block in content:
            text = block.get("text") if isinstance(block, dict) else None
            if isinstance(text, str):
                total += len(text)
    for tool_call in message.tool_calls or []:
        total += len(tool_call.function.name) + len(tool_call.function.arguments)
    return total


def _parse_retry_after(exc: openai.RateLimitError) -> float | None:
    """Extract retry-after / retry-after-ms from the response."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None) or {}
    ms = headers.get("retry-after-ms") if hasattr(headers, "get") else None
    if ms is not None:
        try:
            return float(ms) / 1000.0
        except (TypeError, ValueError):
            pass
    secs = headers.get("retry-after") if hasattr(headers, "get") else None
    if secs is not None:
        try:
            return float(secs)
        except (TypeError, ValueError):
            pass
    return None
