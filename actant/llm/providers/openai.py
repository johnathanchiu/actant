"""OpenAI Responses API adapter."""

from __future__ import annotations

import json
import asyncio
import logging
import random
import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

import httpx
import openai
from openai.types.responses import Response
from openai.types.responses.response_create_params import (
    ResponseCreateParamsBase,
)
from openai.types.responses.response_includable import ResponseIncludable
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses.tool_param import ToolParam
from openai.types.shared_params.reasoning import Reasoning

from actant.core import JSONObject
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

REASONING_MODELS = ("gpt-6", "gpt-5", "o1", "o3", "o4")
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
    """LLMClient implementation for OpenAI's Responses API.

    ``idle_s`` bounds opening a stream and silences while a message or tool call is
    streaming; ``reasoning_idle_s`` bounds silences outside an open output item, where
    a reasoning model legitimately emits nothing; ``turn_s`` bounds the whole call,
    including retries. Only transient failures retry: timeouts, connection errors,
    408/409/429/5xx, server or rate-limit error codes, and a stream that closes
    before a terminal event.
    """

    supports_allowed_tools = True

    def __init__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        thinking_level: str = "med",
        client: openai.AsyncOpenAI | None = None,
        rate_limiter: RateLimiter | None = None,
        idle_s: float = 60.0,
        reasoning_idle_s: float = 180.0,
        turn_s: float = 240.0,
        attempts: int = 3,
    ) -> None:
        self.model_id = model_id
        self.thinking_level = thinking_level
        # This provider owns retries; SDK retries would stack under each attempt.
        self.client = (
            client.with_options(max_retries=0)
            if client is not None
            else openai.AsyncOpenAI(api_key=env_api_key("OPENAI_API_KEY", api_key), max_retries=0)
        )
        self._rate_limiter = rate_limiter
        if idle_s <= 0 or reasoning_idle_s <= 0 or turn_s <= 0 or attempts < 1:
            raise ValueError("idle_s, reasoning_idle_s, turn_s and attempts must be positive")
        self.idle_s, self.reasoning_idle_s = idle_s, reasoning_idle_s
        self.turn_s, self.attempts = turn_s, attempts

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
        *,
        allowed_tools: tuple[str, ...] = (),
    ) -> Message:
        # One budget includes every retry, backoff, and rate-limiter wait.
        # An outer activity deadline must not silently multiply by attempts.
        #
        # `asyncio.timeout` would wait for the cancelled call to finish, and a stuck HTTP
        # stream never does: callers measured turns of 8 minutes on a 60 s budget. So the
        # call is abandoned instead -- cancelled, and left to die on its own time.
        call = asyncio.ensure_future(
            self._complete(system, messages, tools, listener, allowed_tools)
        )
        done, _ = await asyncio.wait({call}, timeout=self.turn_s)
        if done:
            return call.result()
        call.cancel()
        # Retrieved, so a late failure is not reported as an exception nobody consumed.
        call.add_done_callback(lambda task: task.cancelled() or task.exception())
        raise TimeoutError(f"no answer in {self.turn_s:.0f} s")

    async def _complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: list[dict],
        listener: "StreamListener | None",
        allowed_tools: tuple[str, ...],
    ) -> Message:
        params = self._request_params(system, messages, tools)
        if allowed_tools:
            params["tool_choice"] = {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": name} for name in allowed_tools],
            }
        estimated = self._estimate_tokens(messages, params)
        for attempt in range(self.attempts):
            if attempt and listener is not None:
                await listener.on_stream_reset()
            try:
                return await self._reserved_attempt(params, listener, estimated)
            except Exception as error:
                if not _is_transient(error) or attempt + 1 == self.attempts:
                    raise
                logger.warning(
                    "model attempt failed: %s; retry %d/%d",
                    error,
                    attempt + 1,
                    self.attempts,
                )
                retry_after = (
                    _parse_retry_after(error) if isinstance(error, openai.RateLimitError) else None
                )
                await asyncio.sleep(
                    max(0, retry_after or 0) + random.uniform(0, min(30, 2**attempt))
                )
        raise AssertionError("unreachable")

    async def _reserved_attempt(
        self,
        params: RequestParams,
        listener: "StreamListener | None",
        estimated: int,
    ) -> Message:
        # Every attempt is a request the server counts, so each one takes its
        # own reservation. A failed attempt keeps its estimate unless the
        # server reported what it actually consumed.
        if self._rate_limiter is None:
            return (await self._stream_attempt(params, listener))[0]
        async with self._rate_limiter.reserve(estimated) as reservation:
            try:
                message, actual = await self._stream_attempt(params, listener)
            except StreamInterrupted as error:
                if error.tokens is not None:
                    reservation.record_actual(error.tokens)
                raise
            reservation.record_actual(actual)
            return message

    async def _stream_attempt(
        self,
        params: RequestParams,
        listener: "StreamListener | None",
    ) -> tuple[Message, int]:
        tool_stream_state = _OpenAIToolStreamState()
        manager = self.client.responses.stream(**params)
        stream = await asyncio.wait_for(manager.__aenter__(), self.idle_s)
        try:
            events = stream.__aiter__()
            whitespace = 0
            arguments = ""
            open_call: tuple[str | None, str | None] = (None, None)
            emitting = False
            while True:
                # Idle bounds follow the Responses streaming contract: while a
                # message or function_call item is open the model is emitting
                # tokens, so a silence longer than ``idle_s`` is a dead stream.
                # Outside an open output item the server is queued or reasoning,
                # which emits no events until a summary or item arrives, so the
                # longer ``reasoning_idle_s`` applies.
                gap = self.idle_s if emitting else self.reasoning_idle_s
                try:
                    event = await asyncio.wait_for(anext(events), gap)
                except StopAsyncIteration:
                    break
                if listener is not None and listener.cancel_requested():
                    raise StreamCancelled
                if event.type == "response.output_item.added":
                    emitting = event.item.type in ("message", "function_call")
                    if event.item.type == "function_call":
                        # remembered here, not from the listener's state: a completion with
                        # no listener still has to be able to salvage a stalled call below
                        open_call = (
                            getattr(event.item, "call_id", None),
                            getattr(event.item, "name", None),
                        )
                        arguments = ""
                elif event.type == "response.output_item.done":
                    emitting = False
                elif event.type == "error":
                    raise StreamInterrupted(
                        f"stream error {event.code}: {event.message}",
                        retryable=event.code in _TRANSIENT_CODES,
                    )
                elif event.type == "response.failed" or event.type == "response.incomplete":
                    raise _unfinished(event.response)
                elif event.type == "response.function_call_arguments.delta":
                    delta = event.delta or ""
                    arguments += delta
                    whitespace = (
                        whitespace + len(delta)
                        if not delta.strip()
                        else len(delta) - len(delta.rstrip())
                    )
                    if whitespace >= 300:
                        # The model wrote every argument and then streamed whitespace instead
                        # of stopping. When what it wrote is already one complete object, that
                        # is the call: take it rather than spend a turn asking again. Arguments
                        # cut mid-string (a file's content) are genuinely unfinished and retry.
                        whole = _whole_arguments(arguments)
                        call_id, name = open_call
                        if whole is not None and call_id and name:
                            return (
                                Message(
                                    role="assistant",
                                    tool_calls=[
                                        ToolCall(
                                            id=call_id,
                                            function=ToolCallFunction(name=name, arguments=whole),
                                        )
                                    ],
                                ),
                                0,
                            )
                        raise StreamInterrupted(
                            "tool arguments contain 300 consecutive whitespace characters",
                            retryable=True,
                        )
                if listener is None:
                    continue
                await _forward_stream_event(event, listener, tool_stream_state)
            try:
                response = await asyncio.wait_for(stream.get_final_response(), self.idle_s)
            except RuntimeError as error:
                raise StreamInterrupted(
                    "stream closed without response.completed", retryable=True
                ) from error
            if response.status != "completed":
                raise _unfinished(response)
        finally:
            await manager.__aexit__(None, None, None)

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

        usage = getattr(response, "usage", None)
        if listener is not None and usage is not None:
            await listener.on_usage(
                response.id,
                self.model_id,
                cast(JSONObject, usage.model_dump(mode="json")),
                response.status or "unknown",
            )
        input_tokens = _usage_int(usage, "input_tokens")
        output_tokens = _usage_int(usage, "output_tokens")
        message = Message(
            role="assistant",
            content=text or None,
            tool_calls=tool_calls or None,
            thought_summary=thought or None,
            reasoning_items=cast(list[object], reasoning_items) or None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        # The limiter wants the server's own total, which includes
        # reasoning tokens the input/output split may not surface.
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


class StreamInterrupted(RuntimeError):
    """A response cannot be committed because its stream did not complete.

    ``retryable`` says whether another attempt can succeed; ``tokens`` is what the
    attempt consumed when the server reported usage.
    """

    def __init__(self, message: str, *, retryable: bool, tokens: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.tokens = tokens


# Response and stream error codes worth another attempt; the rest describe the request.
_TRANSIENT_CODES = frozenset({"server_error", "rate_limit_exceeded"})


def _unfinished(response: Response) -> StreamInterrupted:
    """A failed, incomplete, or cancelled response. Only a failure with a transient
    error code retries: ``incomplete`` (max_output_tokens, content_filter) and
    ``cancelled`` would end the same way again."""
    error = response.error
    reason = response.incomplete_details.reason if response.incomplete_details else None
    detail = error.code if error is not None else reason
    usage = response.usage
    return StreamInterrupted(
        f"response ended with status {response.status}" + (f" ({detail})" if detail else ""),
        retryable=response.status == "failed"
        and error is not None
        and error.code in _TRANSIENT_CODES,
        tokens=usage.total_tokens if usage is not None else None,
    )


def _is_transient(error: Exception) -> bool:
    if isinstance(error, StreamInterrupted):
        return error.retryable
    if isinstance(error, openai.APIStatusError):
        return error.status_code in (408, 409, 429) or error.status_code >= 500
    if isinstance(error, (TimeoutError, openai.APIConnectionError, httpx.TransportError)):
        return True
    if isinstance(error, openai.APIError):
        # The SDK raises a bare APIError for an SSE payload carrying an ``error`` object.
        body = error.body
        return isinstance(body, Mapping) and (
            body.get("code") in _TRANSIENT_CODES or body.get("type") in _TRANSIENT_CODES
        )
    return False


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


def _usage_int(usage: object, field: str) -> int | None:
    """Read one token count off a provider usage object.

    None when the provider did not report the field at all, so callers
    can tell "not reported" from a genuine zero.
    """
    value = getattr(usage, field, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _whole_arguments(text: str) -> str | None:
    """`text` as one compact JSON object when it already is one, else None.

    gpt-6-astra writes every argument of a call and then streams whitespace instead of
    stopping, several times per authoring round. What it wrote is the call, so taking it
    saves a turn. Only the closing brace may be missing: arguments cut inside a string (a
    file's content in a write) are genuinely unfinished and must be asked for again.
    """

    body = text.strip().rstrip(",")
    if not body.startswith("{"):
        return None
    for candidate in (body, body + "}"):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return json.dumps(parsed, separators=(",", ":"))
    return None
