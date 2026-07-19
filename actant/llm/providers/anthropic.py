"""Anthropic Messages API adapter."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsBase
from anthropic.types.message_param import MessageParam
from anthropic.types.output_config_param import OutputConfigParam
from anthropic.types.thinking_config_param import ThinkingConfigParam
from anthropic.types.tool_union_param import ToolUnionParam

from actant.llm.errors import StreamCancelled
from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.llm.providers._shared import env_api_key, sanitize_tool_messages
from actant.llm.rate_limit import RateLimiter

if TYPE_CHECKING:
    from actant.runtime.hooks import StreamListener

logger = logging.getLogger(__name__)

MAX_TOKENS = 64000
THINKING_BUDGETS: dict[str, int] = {
    "low": 1024,
    "med": 10000,
    "medium": 10000,
    "high": 32000,
}
# 4.7+ ``adaptive`` thinking takes an effort label instead of a token
# budget. Map our existing thinking_level vocabulary onto Anthropic's
# OpenAI-aligned ``low``/``medium``/``high``.
_ADAPTIVE_EFFORT: dict[str, str] = {
    "low": "low",
    "med": "medium",
    "medium": "medium",
    "high": "high",
    "none": "low",
}
ToolSchema = dict[str, object]


class AnthropicProvider:
    """LLMClient implementation for Anthropic Messages API."""

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
        return any(
            prefix in self.model_id for prefix in ("claude-3", "claude-sonnet-4", "claude-opus-4")
        )

    def _uses_adaptive_thinking(self) -> bool:
        """``claude-opus-4-7`` (and presumably newer 4.7+ models)
        rejects the legacy ``thinking.type=enabled`` config with::

          "thinking.type.enabled" is not supported for this model. Use
          "thinking.type.adaptive" and "output_config.effort" to control
          thinking behavior.

        Detect the new format by version: anything ``-4-7`` or higher
        (4-8, 4-9, 5-0, ...) uses ``adaptive`` + ``output_config.effort``.
        Older 4-5/4-6 models keep the ``enabled`` shape.
        """
        for prefix in ("claude-opus-4-", "claude-sonnet-4-"):
            if prefix in self.model_id:
                tail = self.model_id.split(prefix, 1)[1]
                # Tail starts with the minor version: ``7`` for 4-7,
                # ``7-20251201`` for snapshot ids, ``10`` for 4-10, etc.
                # First ``-``-delimited segment is the minor version.
                # Snapshot-only ids like ``claude-sonnet-4-20250514`` have
                # no minor version — the 8-digit date sits where the minor
                # would be, so guard against that before int-parsing.
                head = tail.split("-", 1)[0]
                if len(head) >= 6:
                    return False
                try:
                    return int(head) >= 7
                except ValueError:
                    return False
        # Future Claude families (5-x, 6-x, ...) — assume new format.
        return any(
            self.model_id.startswith(prefix)
            for prefix in ("claude-opus-5", "claude-sonnet-5", "claude-opus-6", "claude-sonnet-6")
        )

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
                converted.append({"role": "user", "content": cast(object, message.content)})
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
                    "content": cast(object, message.content),
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
        params: MessageCreateParamsBase = {
            "model": self.model_id,
            "system": system,
            "messages": cast(
                list[MessageParam],
                self.convert_messages(sanitize_tool_messages(messages)),
            ),
            "max_tokens": MAX_TOKENS,
        }
        converted_tools = self.convert_tools(tools)
        if converted_tools:
            params["tools"] = cast(list[ToolUnionParam], converted_tools)
        if self._is_reasoning_model():
            if self._uses_adaptive_thinking():
                # 4.7+ models took ``thinking.type=enabled`` away in
                # favor of an ``adaptive`` mode driven by an effort
                # knob on ``output_config``. Map our low/med/high
                # thinking_level onto OpenAI-style effort labels;
                # ``budget_tokens`` is gone — the model decides.
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
        except anthropic.RateLimitError as exc:
            # Bucket estimate was off (Anthropic's thinking tokens
            # often diverge from our heuristic). Honor the server's
            # retry-after exactly once, then re-reserve and try again.
            # If we miss twice in a row the budget is mis-configured
            # and we re-raise so Actant's job retry layer can handle it.
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
        # Tool-use content blocks arrive as start → input_json_delta* → stop.
        # ``index`` on stream events is the only field that ties them
        # together, so we map index → tool_call_id at start time.
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
        actual_tokens = int(getattr(usage, "input_tokens", 0) or 0) + int(
            getattr(usage, "output_tokens", 0) or 0
        )

        return (
            Message(
                role="assistant",
                content=text or None,
                tool_calls=tool_calls or None,
                thought_summary=thought or None,
                thinking_signature=thinking_signature,
            ),
            actual_tokens,
        )

    def _estimate_tokens(
        self,
        messages: Sequence[Message],
        params: MessageCreateParamsBase,
    ) -> int:
        # Same conservative char-heuristic the OpenAI provider uses:
        # ~3 chars/token (overestimate slightly so the bucket rarely
        # misses). Reasoning models eat extra hidden tokens so double
        # the input estimate for them.
        char_total = sum(_message_chars(m) for m in messages)
        input_estimate = (char_total // 3) + 200
        if self._is_reasoning_model():
            input_estimate *= 2
        output_ceiling = int(params.get("max_tokens") or MAX_TOKENS)
        return input_estimate + output_ceiling


def _parse_retry_after(exc: anthropic.RateLimitError) -> float | None:
    """Pull the server's retry-after hint off the response. Returns
    None when the header isn't set (caller falls back to a default)."""
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
    """Rough character count for a Message — used for token-budget
    estimation. Counts content + tool_call argument JSON."""
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
