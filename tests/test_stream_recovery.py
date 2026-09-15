import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from actant.llm.messages import Message
from actant.llm.providers.openai import OpenAIProvider


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
