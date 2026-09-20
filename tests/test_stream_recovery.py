"""OpenAI stream recovery, exercised through the real SDK stream over a fake transport."""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import openai
import pytest

from actant.agents import AgentDefinition
from actant.llm.providers.fake import FakeLLM
from actant.llm.providers.openai import OpenAIProvider, StreamInterrupted
from actant.llm.rate_limit import RateLimitConfig, RateLimiter
from actant.runtime.events.streaming import StreamListener
from actant.tools import tool
from actant.tools.registry import ToolRegistry

Event = dict[str, object] | float  # an SSE event, or a pause in seconds before the next


def _response(status: str, output: list | None = None, **extra: object) -> dict[str, object]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-6-astra",
        "status": status,
        "output": output or [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        **extra,
    }


def _events(*body: Event, terminal: dict[str, object] | None = None) -> list[Event]:
    """response.created, ``body``, then ``terminal`` (response.completed by default)."""
    done = terminal or {
        "type": "response.completed",
        "response": _response(
            "completed",
            [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                }
            ],
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        ),
    }
    return [{"type": "response.created", "response": _response("in_progress")}, *body, done]


def _text(delta: str, index: int = 0) -> list[Event]:
    return [
        {
            "type": "response.output_item.added",
            "output_index": index,
            "item": {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        },
        {
            "type": "response.content_part.added",
            "output_index": index,
            "content_index": 0,
            "item_id": "msg_1",
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        {
            "type": "response.output_text.delta",
            "output_index": index,
            "content_index": 0,
            "item_id": "msg_1",
            "delta": delta,
            "logprobs": [],
        },
    ]


class _SSE(httpx.AsyncByteStream):
    def __init__(self, events: list[Event]) -> None:
        self.events = events

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for number, event in enumerate(self.events):
            if not isinstance(event, dict):
                await asyncio.sleep(event)
                continue
            payload = {"sequence_number": number, **event}
            yield f"event: {event['type']}\ndata: {json.dumps(payload)}\n\n".encode()


def _provider(
    replies: list[httpx.Response | list[Event]], **kwargs: object
) -> tuple[OpenAIProvider, list[httpx.Request]]:
    """A provider whose client answers each request with the next reply."""
    requests: list[httpx.Request] = []
    queue = iter(replies)

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        reply = next(queue)
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=_SSE(reply)
        )

    client = openai.AsyncOpenAI(
        api_key="test",
        base_url="http://openai.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    return OpenAIProvider("gpt-6-astra", client=client, **kwargs), requests  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("actant.llm.providers.openai.random.uniform", lambda *args: 0)


def _failed(code: str) -> dict[str, object]:
    return {
        "type": "response.failed",
        "response": _response(
            "failed",
            error={"code": code, "message": code},
            usage={"input_tokens": 700, "output_tokens": 0, "total_tokens": 700},
        ),
    }


def _incomplete(reason: str) -> dict[str, object]:
    return {
        "type": "response.incomplete",
        "response": _response("incomplete", incomplete_details={"reason": reason}),
    }


async def test_completed_stream_returns_message() -> None:
    provider, requests = _provider([_events(*_text("hello"))])
    answer = await provider.complete("system", [], [])
    assert answer.content == "hello"
    assert answer.output_tokens == 5
    assert len(requests) == 1


@pytest.mark.parametrize(
    "terminal",
    [
        _incomplete("max_output_tokens"),
        _incomplete("content_filter"),
        _failed("invalid_prompt"),
        {"type": "error", "code": "invalid_request_error", "message": "bad"},
    ],
    ids=["max_output_tokens", "content_filter", "failed-invalid", "error-event"],
)
async def test_permanent_stream_outcomes_are_not_retried(terminal: dict[str, object]) -> None:
    provider, requests = _provider([_events(terminal=terminal)] * 3, attempts=3)
    with pytest.raises(StreamInterrupted):
        await provider.complete("system", [], [])
    assert len(requests) == 1


async def test_sse_error_payload_is_not_retried_unless_transient() -> None:
    bad = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
    provider, requests = _provider([_events(terminal=bad)] * 3, attempts=3)
    with pytest.raises(openai.APIError):
        await provider.complete("system", [], [])
    assert len(requests) == 1

    busy = {"type": "error", "error": {"type": "server_error", "message": "busy"}}
    provider, requests = _provider([_events(terminal=busy), _events(*_text("hello"))], attempts=3)
    assert (await provider.complete("system", [], [])).content == "hello"
    assert len(requests) == 2


@pytest.mark.parametrize(
    "first",
    [
        _events(terminal=_failed("server_error")),
        _events(terminal={"type": "error", "code": "rate_limit_exceeded", "message": "slow"}),
        _events(*_text("hel"))[:-1],  # closed before any terminal event
        httpx.Response(503),
        httpx.Response(429, headers={"retry-after-ms": "0"}),
    ],
    ids=["failed-server", "error-rate", "truncated", "503", "429"],
)
async def test_transient_failures_retry(first: httpx.Response | list[Event]) -> None:
    provider, requests = _provider([first, _events(*_text("hello"))], attempts=2)
    assert (await provider.complete("system", [], [])).content == "hello"
    assert len(requests) == 2


async def test_sdk_retries_do_not_stack_under_provider_attempts() -> None:
    provider, requests = _provider([httpx.Response(500)] * 9, attempts=2)
    with pytest.raises(openai.InternalServerError):
        await provider.complete("system", [], [])
    assert len(requests) == 2


async def test_reasoning_silence_is_not_an_idle_cut() -> None:
    # Scaled 1000x: a 70 s reasoning gap against a 60 s token idle bound. A reasoning
    # model with summary="auto" opens its reasoning item, then emits nothing until the
    # summary or the next output item.
    reasoning = [
        {"type": "response.in_progress", "response": _response("in_progress")},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "reasoning", "id": "rs_1", "summary": []},
        },
        0.07,
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "reasoning", "id": "rs_1", "summary": []},
        },
    ]
    provider, requests = _provider(
        [_events(*reasoning, *_text("hello", index=1))],
        idle_s=0.06,
        reasoning_idle_s=0.2,
        attempts=1,
    )
    assert (await provider.complete("system", [], [])).content == "hello"
    assert len(requests) == 1


async def test_silence_inside_streaming_output_is_cut_and_retried() -> None:
    stalled = _events(*_text("hel"), 0.2)
    provider, requests = _provider(
        [stalled, _events(*_text("hello"))], idle_s=0.05, reasoning_idle_s=1.0, attempts=2
    )
    assert (await provider.complete("system", [], [])).content == "hello"
    assert len(requests) == 2


async def test_listener_is_reset_before_a_retry() -> None:
    seen: list[str] = []

    class Listener(StreamListener):
        async def on_text_delta(self, delta: str) -> None:
            seen.append(delta)

        async def on_stream_reset(self) -> None:
            seen.append("<reset>")

    provider, _ = _provider(
        [_events(*_text("hel"), terminal=_failed("server_error")), _events(*_text("hello"))],
        attempts=2,
    )
    await provider.complete("system", [], [], Listener())
    assert seen == ["hel", "<reset>", "hello"]


async def test_each_attempt_reserves_and_records_failed_usage() -> None:
    limiter = RateLimiter(RateLimitConfig(tokens_per_minute=100000, requests_per_minute=100))
    provider, _ = _provider(
        [_events(terminal=_failed("server_error")), _events(*_text("hello"))],
        attempts=2,
        rate_limiter=limiter,
    )
    await provider.complete("system", [], [])
    assert len(limiter._request_window) == 2
    assert [tokens for _, tokens in limiter._token_window] == [700, 15]


async def test_turn_budget_includes_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("actant.llm.providers.openai.random.uniform", lambda *args: 10)
    provider, requests = _provider([httpx.Response(503)] * 3, turn_s=0.05, attempts=3)
    with pytest.raises(TimeoutError):
        await provider.complete("system", [], [])
    assert len(requests) == 1


async def test_continuously_streaming_attempt_has_a_total_deadline() -> None:
    trickle: list[Event] = [0.01] * 50
    provider, _ = _provider([_events(*trickle)], turn_s=0.05, idle_s=1.0, attempts=1)
    with pytest.raises(TimeoutError):
        await provider.complete("system", [], [])


async def test_whitespace_loop_is_interrupted() -> None:
    call = {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "edit",
            "arguments": "",
        },
    }
    spaces = {
        "type": "response.function_call_arguments.delta",
        "output_index": 0,
        "item_id": "fc_1",
        "delta": " " * 150,
    }
    provider, _ = _provider([_events(call, spaces, spaces)], attempts=1)
    with pytest.raises(StreamInterrupted, match="whitespace"):
        await provider.complete("system", [], [])


async def test_arguments_written_whole_before_the_stall_are_taken_as_the_call() -> None:
    """gpt-6-astra writes every argument of a call and then streams whitespace instead of
    stopping, several times per authoring round. What it wrote is the call, so it is taken
    rather than spending a turn asking again."""

    call = {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "fit",
            "arguments": "",
        },
    }
    written = {
        "type": "response.function_call_arguments.delta",
        "output_index": 0,
        "item_id": "fc_1",
        "delta": '{"object_id": "chair-1", "yaw": 149.0',  # no closing brace
    }
    spaces = {
        "type": "response.function_call_arguments.delta",
        "output_index": 0,
        "item_id": "fc_1",
        "delta": " " * 150,
    }
    provider, _ = _provider([_events(call, written, spaces, spaces)], attempts=1)
    message = await provider.complete("system", [], [])
    assert message.tool_calls is not None
    (only,) = message.tool_calls
    assert only.id == "call_1"
    assert only.function.name == "fit"
    assert json.loads(only.function.arguments) == {"object_id": "chair-1", "yaw": 149.0}


async def test_arguments_cut_inside_a_string_still_ask_again() -> None:
    """A write's file content cut mid-string is genuinely unfinished, not a stalled call."""

    call = {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "write",
            "arguments": "",
        },
    }
    partial = {
        "type": "response.function_call_arguments.delta",
        "output_index": 0,
        "item_id": "fc_1",
        "delta": '{"path": "room.py", "text": "room.place(',  # cut inside the string
    }
    spaces = {
        "type": "response.function_call_arguments.delta",
        "output_index": 0,
        "item_id": "fc_1",
        "delta": " " * 150,
    }
    provider, _ = _provider([_events(call, partial, spaces, spaces)], attempts=1)
    with pytest.raises(StreamInterrupted, match="whitespace"):
        await provider.complete("system", [], [])


async def test_final_tools_preserve_full_request_schema() -> None:
    provider, requests = _provider([_events(*_text("done"))])
    tools = [
        {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
        for name in ("read", "edit")
    ]
    await provider.complete("system", [], tools, allowed_tools=("edit",))
    params = json.loads(requests[0].content)
    assert [tool["name"] for tool in params["tools"]] == ["read", "edit"]
    assert params["tool_choice"] == {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "function", "name": "edit"}],
    }


def test_final_tools_are_validated_at_definition() -> None:
    @tool
    async def edit() -> str:
        """Edit."""
        return "ok"

    def define(llm: FakeLLM, final_tools: tuple[str, ...], allow: set[str]) -> AgentDefinition:
        return AgentDefinition(
            id="a",
            name="a",
            persona="p",
            llm=llm,
            tools=ToolRegistry([edit]),
            tool_allowlist=allow,
            final_tools=final_tools,
        )

    class NoAllowedTools(FakeLLM):
        supports_allowed_tools = False

    assert define(FakeLLM([]), ("edit",), set()).final_tools == ("edit",)
    assert define(NoAllowedTools([]), (), set()).final_tools == ()
    with pytest.raises(ValueError, match="not registered"):
        define(FakeLLM([]), ("missing",), set())
    with pytest.raises(ValueError, match="not registered"):
        define(FakeLLM([]), ("edit",), {"other"})
    with pytest.raises(ValueError, match="cannot honour"):
        define(NoAllowedTools([]), ("edit",), set())
