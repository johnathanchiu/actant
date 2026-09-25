"""Anthropic Messages API adapter."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import anthropic
from anthropic.types.cache_control_ephemeral_param import CacheControlEphemeralParam
from anthropic.types.message_create_params import MessageCreateParamsBase
from anthropic.types.message_param import MessageParam
from anthropic.types.output_config_param import OutputConfigParam
from anthropic.types.thinking_config_param import ThinkingConfigParam
from anthropic.types.tool_union_param import ToolUnionParam

from actant.core import JSONObject
from actant.blocks import AssetBlock, PromptBlock, UrlImageBlock
from actant.llm.errors import StreamCancelled
from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.llm.providers._shared import (
    WireBlock,
    env_api_key,
    sanitize_tool_messages,
    unresolved,
)
from actant.llm.rate_limit import RateLimiter

if TYPE_CHECKING:
    from actant.runtime.events.streaming import StreamListener

logger = logging.getLogger(__name__)

MAX_TOKENS = 64000
THINKING_BUDGETS: dict[str, int] = {
    "low": 1024,
    "med": 10000,
    "medium": 10000,
    "high": 32000,
}
# thinking_level -> adaptive effort (4.7+).
_ADAPTIVE_EFFORT: dict[str, str] = {
    "low": "low",
    "med": "medium",
    "medium": "medium",
    "high": "high",
    "none": "low",
}
ToolSchema = dict[str, object]
_CACHE: dict[str, object] = {"type": "ephemeral"}
# major, then a minor that is not a date snapshot.
_VERSION = re.compile(r"claude-(?:[a-z]+-)?(\d{1,2})(?:-(\d{1,2})(?!\d))?")


class AnthropicProvider:
    """LLMClient implementation for Anthropic Messages API."""

    # tool_choice cannot name a subset of tools.
    supports_allowed_tools = False

    def __init__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        thinking_level: str = "med",
        client: anthropic.AsyncAnthropic | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.model_id = model_id
        self.thinking_level = thinking_level
        self.client = client or anthropic.AsyncAnthropic(
            api_key=env_api_key("ANTHROPIC_API_KEY", api_key)
        )
        self._rate_limiter = rate_limiter

    def _is_reasoning_model(self) -> bool:
        """Claude 3.7 and later think."""
        version = _claude_version(self.model_id)
        return version is not None and version >= (3, 7)

    def _uses_adaptive_thinking(self) -> bool:
        """4.7+ takes adaptive thinking; older models take budget_tokens."""
        version = _claude_version(self.model_id)
        return version is not None and version >= (4, 7)

    @staticmethod
    def convert_tools(tools: list[dict]) -> list[ToolSchema]:
        anthropic_tools: list[ToolSchema] = []
        for tool in tools or []:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict):
                continue
            anthropic_tools.append(
                {
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "input_schema": function.get("parameters", {}),
                }
            )
        return anthropic_tools

    @staticmethod
    def convert_messages(messages: Sequence[Message]) -> list[ToolSchema]:
        converted: list[ToolSchema] = []
        for message in messages:
            if message.role == "user":
                converted.append({"role": "user", "content": _content(message.content)})
            elif message.role == "assistant":
                blocks: list[object] = []
                if message.thought_summary and message.thinking_signature:
                    thinking_block: ToolSchema = {
                        "type": "thinking",
                        "thinking": message.thought_summary,
                        "signature": message.thinking_signature,
                    }
                    blocks.append(thinking_block)
                if message.content:
                    blocks.append({"type": "text", "text": cast(object, message.content)})
                for tool_call in message.tool_calls or []:
                    try:
                        input_data = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        input_data = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "input": cast(object, input_data),
                        }
                    )
                if blocks:
                    converted.append({"role": "assistant", "content": blocks})
            elif message.role == "tool":
                tool_result: ToolSchema = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id or "",
                    "content": _content(message.content),
                }
                if (
                    converted
                    and converted[-1].get("role") == "user"
                    and isinstance(converted[-1].get("content"), list)
                ):
                    cast(list[object], converted[-1]["content"]).append(tool_result)
                else:
                    converted.append({"role": "user", "content": [tool_result]})
        return converted

    def _request_params(
        self, system: str, messages: Sequence[Message], tools: list[dict]
    ) -> MessageCreateParamsBase:
        # Breakpoints: tools, system, and (automatic) the last message block.
        params: MessageCreateParamsBase = {
            "model": self.model_id,
            "messages": cast(
                list[MessageParam],
                self.convert_messages(sanitize_tool_messages(messages)),
            ),
            "max_tokens": MAX_TOKENS,
            "cache_control": cast(CacheControlEphemeralParam, _CACHE),
        }
        if system:
            params["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": cast(CacheControlEphemeralParam, _CACHE),
                }
            ]
        converted_tools = self.convert_tools(tools)
        if converted_tools:
            converted_tools[-1] = {**converted_tools[-1], "cache_control": _CACHE}
            params["tools"] = cast(list[ToolUnionParam], converted_tools)
        if self._is_reasoning_model():
            if self._uses_adaptive_thinking():
                effort = _ADAPTIVE_EFFORT.get(self.thinking_level, "medium")
                params["thinking"] = cast(ThinkingConfigParam, {"type": "adaptive"})
                params["output_config"] = cast(OutputConfigParam, {"effort": effort})
            else:
                params["thinking"] = cast(
                    ThinkingConfigParam,
                    {
                        "type": "enabled",
                        "budget_tokens": THINKING_BUDGETS.get(self.thinking_level, 10000),
                    },
                )
        return params

    async def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: list[dict],
        listener: "StreamListener | None" = None,
        *,
        allowed_tools: tuple[str, ...] = (),
    ) -> Message:
        if allowed_tools:
            raise NotImplementedError(
                "Anthropic does not support allowed_tools with a stable full tool list"
            )
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
        except anthropic.RateLimitError as exc:
            # Estimate missed: honor retry-after once, then let a second miss raise.
            wait = _parse_retry_after(exc) or 5.0
            logger.warning(
                "actant.anthropic.rate_limit_miss model=%s wait_secs=%.2f error=%s",
                self.model_id,
                wait,
                exc,
            )
            await asyncio.sleep(wait + 0.5)
            estimated_retry = self._estimate_tokens(messages, params)
            async with self._rate_limiter.reserve(estimated_retry) as reservation:
                message, actual = await self._stream(params, listener)
                reservation.record_actual(actual)
                return message

    async def _stream(
        self,
        params: MessageCreateParamsBase,
        listener: "StreamListener | None",
    ) -> tuple[Message, int]:
        # Stream events tie tool-use deltas to their block only by index.
        async with self.client.messages.stream(**params) as stream:
            if listener is not None:
                tool_index_to_id: dict[int, str] = {}
                async for event in stream:
                    if listener.cancel_requested():
                        raise StreamCancelled
                    event_type = getattr(event, "type", None)
                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if getattr(block, "type", None) == "tool_use":
                            tool_call_id = getattr(block, "id", "") or ""
                            tool_name = getattr(block, "name", "") or ""
                            if tool_call_id:
                                tool_index_to_id[getattr(event, "index", -1)] = tool_call_id
                                await listener.on_tool_call_start(tool_call_id, tool_name)
                        continue
                    if event_type == "content_block_stop":
                        idx = getattr(event, "index", -1)
                        tool_call_id = tool_index_to_id.pop(idx, None)
                        if tool_call_id is not None:
                            await listener.on_tool_call_args_complete(tool_call_id)
                        continue
                    if event_type != "content_block_delta":
                        continue
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        await listener.on_text_delta(getattr(delta, "text", "") or "")
                    elif delta_type == "thinking_delta":
                        await listener.on_thinking_delta(getattr(delta, "thinking", "") or "")
                    elif delta_type == "input_json_delta":
                        idx = getattr(event, "index", -1)
                        tool_call_id = tool_index_to_id.get(idx)
                        if tool_call_id is not None:
                            partial = getattr(delta, "partial_json", "") or ""
                            if partial:
                                await listener.on_tool_call_args_delta(tool_call_id, partial)
            response = await stream.get_final_message()

        text = ""
        thought = ""
        thinking_signature: str | None = None
        tool_calls: list[ToolCall] = []

        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text += getattr(block, "text", "") or ""
            elif block_type == "thinking":
                thought += getattr(block, "thinking", "") or ""
                if thinking_signature is None:
                    thinking_signature = getattr(block, "signature", None)
            elif block_type == "tool_use":
                tool_calls.append(_tool_call_from_block(block))

        usage = getattr(response, "usage", None)
        report = _usage_report(usage)
        if listener is not None and usage is not None:
            await listener.on_usage(
                getattr(response, "id", "") or "",
                self.model_id,
                report,
                getattr(response, "stop_reason", None) or "unknown",
            )
        input_tokens = cast("int | None", report.get("input_tokens"))
        output_tokens = cast("int | None", report.get("output_tokens"))
        actual_tokens = cast(int, report.get("total_tokens", 0))

        return (
            Message(
                role="assistant",
                content=text or None,
                tool_calls=tool_calls or None,
                thought_summary=thought or None,
                thinking_signature=thinking_signature,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            actual_tokens,
        )

    def _estimate_tokens(
        self,
        messages: Sequence[Message],
        params: MessageCreateParamsBase,
    ) -> int:
        # ~3 chars/token, doubled for hidden reasoning tokens.
        char_total = sum(_message_chars(m) for m in messages)
        input_estimate = (char_total // 3) + 200
        if self._is_reasoning_model():
            input_estimate *= 2
        output_ceiling = int(params.get("max_tokens") or MAX_TOKENS)
        return input_estimate + output_ceiling


def _parse_retry_after(exc: anthropic.RateLimitError) -> float | None:
    """The server's retry-after seconds, or None."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", {}) or {}
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _message_chars(message: Message) -> int:
    """Rough character count for token estimation."""
    total = len(message.content or "") if isinstance(message.content, str) else 0
    if isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, dict):
                total += len(json.dumps(block, default=str))
    for tc in message.tool_calls or []:
        total += len(tc.function.arguments or "")
        total += len(tc.function.name or "")
    return total


def _tool_call_from_block(block: object) -> ToolCall:
    tool_input = getattr(block, "input", None) or {}
    try:
        arguments = json.dumps(tool_input)
    except (TypeError, ValueError):
        arguments = "{}"
    return ToolCall(
        id=getattr(block, "id", "") or "",
        function=ToolCallFunction(
            name=getattr(block, "name", "") or "",
            arguments=arguments,
        ),
    )


def _usage_int(usage: object, field: str) -> int | None:
    """One usage count, or None when not reported."""
    value = getattr(usage, field, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_report(usage: object) -> JSONObject:
    """Usage in the OpenAI Responses shape (input includes cache)."""
    uncached = _usage_int(usage, "input_tokens")
    read = _usage_int(usage, "cache_read_input_tokens")
    written = _usage_int(usage, "cache_creation_input_tokens")
    output = _usage_int(usage, "output_tokens")
    report: JSONObject = {}
    if uncached is not None or read is not None or written is not None:
        report["input_tokens"] = (uncached or 0) + (read or 0) + (written or 0)
        report["input_tokens_details"] = {
            "cached_tokens": read or 0,
            "cache_write_tokens": written or 0,
        }
    if output is not None:
        report["output_tokens"] = output
    report["total_tokens"] = cast(int, report.get("input_tokens", 0)) + (output or 0)
    return report


def _claude_version(model_id: str) -> tuple[int, int] | None:
    """(major, minor) from a Claude model id, or None."""
    match = _VERSION.search(model_id)
    if match is None:
        return None
    return int(match[1]), int(match[2] or 0)


def _content(content: str | list[PromptBlock] | None) -> str | list[WireBlock]:
    """A user or tool message's content as Anthropic takes it: text and base64 images are
    already its shape."""
    if not isinstance(content, list):
        return content or ""
    blocks: list[WireBlock] = []
    for block in content:
        if isinstance(block, AssetBlock):
            unresolved(block)
        elif isinstance(block, UrlImageBlock):
            blocks.append({"type": "image", "source": {"type": "url", "url": block.url}})
        else:
            blocks.append(block.model_dump(mode="json"))
    return blocks
