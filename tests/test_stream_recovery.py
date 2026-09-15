import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import httpx
import openai

from actant.llm.messages import Message
from actant.llm.providers.openai import OpenAIProvider, StreamInterrupted
from actant.llm.rate_limit import RateLimitConfig, RateLimiter


async def test_rate_limiter_does_not_multiply_retry_attempts(monkeypatch):
    limiter = RateLimiter(RateLimitConfig(tokens_per_minute=100000, requests_per_minute=100))
    provider = OpenAIProvider("gpt-test", api_key="test", attempts=2, rate_limiter=limiter)
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.test"))
    attempt = AsyncMock(side_effect=openai.RateLimitError("limited", response=response, body=None))
    monkeypatch.setattr(provider, "_stream_attempt", attempt)
    monkeypatch.setattr("actant.llm.providers.openai.random.uniform", lambda *args: 0)
    with pytest.raises(openai.RateLimitError):
        await provider.complete("system", [], [])
    assert attempt.await_count == 2
    await provider.client.close()


async def test_idle_attempt_retries_without_committing_partial_answer(monkeypatch):
    provider = OpenAIProvider("gpt-6-astra", api_key="test", idle_s=0.01, attempts=2)
    closed = []

    class Stream:
        def __init__(self, stall):
            self.stall = stall

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append(self.stall)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.stall:
                await asyncio.Event().wait()
            raise StopAsyncIteration

        async def get_final_response(self):
            return SimpleNamespace(status="completed", output=[], usage=None)

    streams = iter([Stream(True), Stream(False)])
    monkeypatch.setattr(provider.client.responses, "stream", lambda **kw: next(streams))
    monkeypatch.setattr("actant.llm.providers.openai.random.uniform", lambda *args: 0)
    answer = await provider.complete("system", [], [])
    assert answer.role == "assistant"
    assert closed == [True, False]
    await provider.client.close()


async def test_turn_budget_includes_retry_backoff(monkeypatch):
    provider = OpenAIProvider("gpt-test", api_key="test", turn_s=0.02, attempts=3)
    attempt = AsyncMock(side_effect=StreamInterrupted("incomplete"))
    monkeypatch.setattr(provider, "_stream_attempt", attempt)
    monkeypatch.setattr("actant.llm.providers.openai.random.uniform", lambda *args: 10)
    with pytest.raises(TimeoutError):
        await provider.complete("system", [], [])
    assert attempt.await_count == 1
    await provider.client.close()


@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled"])
async def test_noncompleted_response_is_never_returned(monkeypatch, status):
    provider = OpenAIProvider("gpt-test", api_key="test", attempts=1)
    stream = AsyncMock()
    stream.__aiter__.return_value = []
    stream.get_final_response.return_value = SimpleNamespace(status=status)
    manager = AsyncMock()
    manager.__aenter__.return_value = stream
    monkeypatch.setattr(provider.client.responses, "stream", lambda **kw: manager)
    with pytest.raises(StreamInterrupted, match=status):
        await provider.complete("system", [], [])
    manager.__aexit__.assert_awaited_once()
    await provider.client.close()


async def test_whitespace_loop_closes_stream(monkeypatch):
    provider = OpenAIProvider("gpt-test", api_key="test", attempts=1)
    stream = AsyncMock()
    stream.__aiter__.return_value = [
        SimpleNamespace(type="response.function_call_arguments.delta", delta=" " * 150),
        SimpleNamespace(type="response.function_call_arguments.delta", delta=" " * 150),
    ]
    manager = AsyncMock()
    manager.__aenter__.return_value = stream
    monkeypatch.setattr(provider.client.responses, "stream", lambda **kw: manager)
    with pytest.raises(StreamInterrupted, match="whitespace"):
        await provider.complete("system", [], [])
    manager.__aexit__.assert_awaited_once()
    await provider.client.close()


async def test_final_tools_preserve_full_request_schema(monkeypatch):
    provider = OpenAIProvider("gpt-6-astra", api_key="test")
    stream = AsyncMock(return_value=(Message(role="assistant", content="done"), 0))
    monkeypatch.setattr(provider, "_stream", stream)
    tools = [
        {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
        for name in ("read", "edit")
    ]
    await provider.complete("system", [], tools, allowed_tools=("edit",))
    params = stream.call_args.args[0]
    assert [tool["name"] for tool in params["tools"]] == ["read", "edit"]
    assert params["tool_choice"] == {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "function", "name": "edit"}],
    }
    await provider.client.close()


async def test_continuously_streaming_attempt_has_a_total_deadline(monkeypatch):
    provider = OpenAIProvider("gpt-6-astra", api_key="test", turn_s=0.01, attempts=1)

    async def stuck(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(provider, "_stream_attempt", stuck)
    with pytest.raises(TimeoutError):
        await provider.complete("system", [], [])
    await provider.client.close()
