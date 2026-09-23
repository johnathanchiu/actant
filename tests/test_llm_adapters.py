from __future__ import annotations

import time

import json
from types import SimpleNamespace

import pytest

from actant.llm import Message, ToolCall, ToolCallFunction, provider_for_model
from actant.llm.providers._shared import sanitize_tool_messages
from actant.llm.providers.anthropic import AnthropicProvider
from actant.llm.providers.gemini import GeminiProvider
from actant.llm.providers.openai import (
    OpenAIProvider,
    _OpenAIToolStreamState,
    _forward_stream_event,
)
from actant.runtime.events.streaming import StreamListener
from actant.tools import make_tool_schema


class _RecordingListener(StreamListener):
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str]] = []

    def cancel_requested(self) -> bool:
        return False

    async def on_text_delta(self, delta: str) -> None:
        self.events.append(("text", delta, ""))

    async def on_thinking_delta(self, delta: str) -> None:
        self.events.append(("thinking", delta, ""))

    async def on_tool_call_start(self, tool_call_id: str, name: str) -> None:
        self.events.append(("tool_start", tool_call_id, name))

    async def on_tool_call_args_delta(self, tool_call_id: str, delta: str) -> None:
        self.events.append(("tool_args", tool_call_id, delta))

    async def on_tool_call_args_complete(self, tool_call_id: str) -> None:
        self.events.append(("tool_complete", tool_call_id, ""))


def test_provider_for_model_routes_known_prefixes() -> None:
    assert provider_for_model("gpt-example") == "openai"
    assert provider_for_model("o4-mini") == "openai"
    assert provider_for_model("claude-example") == "anthropic"
    assert provider_for_model("gemini/example-model") == "gemini"
    assert provider_for_model("qwen-example") == "qwen"


def test_provider_for_model_rejects_unknown_prefix() -> None:
    with pytest.raises(ValueError, match="Cannot determine provider"):
        provider_for_model("unknown-model")


def test_openai_requests_are_never_stored_by_provider() -> None:
    provider = OpenAIProvider(model_id="gpt-5.4-nano", api_key="test")

    params = provider._request_params(
        "You are helpful.",
        [Message(role="user", content="hello")],
        [],
    )

    assert params.get("store") is False


def test_tool_call_from_raw_normalizes_null_fields() -> None:
    tool_call = ToolCall.from_raw(
        {
            "id": None,
            "type": None,
            "function": {
                "name": None,
                "arguments": None,
            },
            "thought_signature": None,
            "extra_content": None,
        }
    )

    assert tool_call.id == ""
    assert tool_call.type == "function"
    assert tool_call.function.name == ""
    assert tool_call.function.arguments == ""
    assert tool_call.extra_content == {}


def test_sanitize_tool_messages_assigns_missing_tool_call_id() -> None:
    assistant = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                id="",
                function=ToolCallFunction(name="echo", arguments='{"message":"hi"}'),
            )
        ],
    )
    tool = Message(role="tool", content=json.dumps({"result": "hi"}))

    sanitized = sanitize_tool_messages(
        [json.loads(json.dumps(assistant.to_dict())), json.loads(json.dumps(tool.to_dict()))]
    )

    assert sanitized[0].tool_calls is not None
    assert sanitized[0].tool_calls[0].id
    assert sanitized[1].tool_call_id == sanitized[0].tool_calls[0].id


def test_openai_converts_tool_result_with_images_after_function_output() -> None:
    message = Message(
        role="tool",
        tool_call_id="call_1",
        content=[
            {"type": "text", "text": "done"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "abc",
                },
            },
        ],
    )

    items = OpenAIProvider._convert_tool_message(message)

    assert items[0] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "done",
    }
    assert items[1]["role"] == "user"


def test_anthropic_omits_unsigned_thinking_from_history() -> None:
    converted = AnthropicProvider.convert_messages(
        [
            Message(
                role="assistant",
                thought_summary="private chain of thought",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=ToolCallFunction(name="ask_user", arguments='{"question":"Q"}'),
                    )
                ],
            )
        ]
    )

    assert converted == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "ask_user",
                    "input": {"question": "Q"},
                }
            ],
        }
    ]


def test_anthropic_preserves_signed_thinking_in_history() -> None:
    converted = AnthropicProvider.convert_messages(
        [
            Message(
                role="assistant",
                thought_summary="signed thought",
                thinking_signature="sig_123",
            )
        ]
    )

    assert converted == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "signed thought",
                    "signature": "sig_123",
                }
            ],
        }
    ]


@pytest.mark.parametrize(
    ("model_id", "reasoning", "adaptive"),
    [
        ("claude-3-5-sonnet-20241022", False, False),
        ("claude-3-7-sonnet-20250219", True, False),
        ("claude-sonnet-4-20250514", True, False),
        ("claude-haiku-4-5", True, False),
        ("claude-opus-4-6", True, False),
        ("claude-opus-4-7", True, True),
        ("claude-opus-5-5", True, True),
        ("claude-fable-5-1", True, True),
        ("claude-sonnet-6", True, True),
        ("claude-example", False, False),
    ],
)
def test_anthropic_thinking_follows_model_version(
    model_id: str, reasoning: bool, adaptive: bool
) -> None:
    provider = AnthropicProvider(model_id=model_id, api_key="test", thinking_level="high")
    params = provider._request_params("S", [Message(role="user", content="hi")], [])
    thinking = params.get("thinking")
    if not reasoning:
        assert thinking is None
    elif adaptive:
        assert thinking == {"type": "adaptive", "display": "summarized"}
        assert params.get("output_config") == {"effort": "high"}
    else:
        assert thinking == {"type": "enabled", "budget_tokens": 32000}


def test_anthropic_caches_tools_system_and_history() -> None:
    provider = AnthropicProvider(model_id="claude-opus-5-5", api_key="test")
    tools = [make_tool_schema(name, "d", {}) for name in ("a", "b")]
    params = provider._request_params("System", [Message(role="user", content="hi")], tools)
    ephemeral = {"type": "ephemeral"}
    assert params.get("cache_control") == ephemeral  # automatic: the last message block
    assert params.get("system") == [{"type": "text", "text": "System", "cache_control": ephemeral}]
    sent = list(params.get("tools", []))
    assert [tool for tool in sent if "cache_control" in tool] == [sent[-1]]
    assert "cache_control" not in AnthropicProvider.convert_tools(tools)[-1]


def test_anthropic_sends_tool_result_images_inside_tool_result() -> None:
    image: dict[str, object] = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "iVBO"},
    }
    text: dict[str, object] = {"type": "text", "text": "render"}
    provider = AnthropicProvider(model_id="claude-opus-5-5", api_key="test")
    params = provider._request_params(
        "S",
        [
            Message(role="user", content="look"),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id="t1", function=ToolCallFunction(name="see", arguments="{}"))
                ],
            ),
            Message(role="tool", tool_call_id="t1", content=[text, image]),
        ],
        [],
    )
    assert list(params["messages"])[-1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [text, image]}],
    }


@pytest.mark.asyncio
async def test_anthropic_reports_usage_in_the_openai_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        id="msg_1",
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(
            input_tokens=40,
            cache_read_input_tokens=5000,
            cache_creation_input_tokens=300,
            output_tokens=90,
        ),
    )

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def get_final_message(self):
            return response

    received = []

    class Listener(StreamListener):
        async def on_usage(self, response_id, model, usage, status):
            received.append((response_id, model, usage, status))

    provider = AnthropicProvider(model_id="claude-opus-5-5", api_key="test")
    monkeypatch.setattr(provider.client.messages, "stream", lambda **kw: Stream())
    message, total = await provider._stream({}, Listener())  # pyright: ignore[reportArgumentType]
    usage = {
        "input_tokens": 5340,
        "input_tokens_details": {"cached_tokens": 5000, "cache_write_tokens": 300},
        "output_tokens": 90,
        "total_tokens": 5430,
    }
    assert received == [("msg_1", "claude-opus-5-5", usage, "end_turn")]
    assert (message.input_tokens, message.output_tokens, total) == (5340, 90, 5430)
    await provider.client.close()


def test_openai_converts_chat_tool_schema_to_responses_schema() -> None:
    schema = make_tool_schema(
        "echo",
        "Echo",
        parameters={
            "coords": {
                "type": "array",
                "prefixItems": [{"type": "number"}, {"type": "number"}],
            }
        },
    )

    [converted] = OpenAIProvider.convert_tools([schema])

    assert converted["type"] == "function"
    assert converted["name"] == "echo"
    parameters = converted["parameters"]
    assert isinstance(parameters, dict)
    coords = parameters["properties"]["coords"]
    assert isinstance(coords, dict)
    assert "prefixItems" not in coords
    assert coords["items"] == {"type": "number"}


async def test_openai_forwards_streamed_function_call_events() -> None:
    listener = _RecordingListener()
    state = _OpenAIToolStreamState()

    await _forward_stream_event(
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="fetch_url",
            ),
        ),
        listener,
        state,
    )
    await _forward_stream_event(
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            output_index=0,
            delta='{"url"',
        ),
        listener,
        state,
    )
    await _forward_stream_event(
        SimpleNamespace(
            type="response.function_call_arguments.done",
            output_index=0,
        ),
        listener,
        state,
    )

    assert listener.events == [
        ("tool_start", "call_1", "fetch_url"),
        ("tool_args", "call_1", '{"url"'),
        ("tool_complete", "call_1", ""),
    ]


def test_gemini_argument_conversion_handles_json_string() -> None:
    assert GeminiProvider.convert_arguments('{"x": 1}') == {"x": 1}
    assert GeminiProvider.convert_arguments("[1, 2]") == {}


def test_gemini_replays_tool_call_thought_signature() -> None:
    provider = GeminiProvider(
        model_id="gemini-example",
        api_key="test",
        check_thinking_support=False,
    )
    message = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="tc_1",
                function=ToolCallFunction(name="plot_points", arguments='{"points": []}'),
                thought_signature=GeminiProvider.encode_signature(b"sig"),
            )
        ],
    )

    content = provider.convert_message(message)

    assert content.role == "model"
    assert content.parts
    assert content.parts[0].thought_signature == b"sig"


def test_gemini_sends_url_image_sources_as_file_data() -> None:
    provider = GeminiProvider(
        model_id="gemini-example", api_key="test", check_thinking_support=False
    )
    [part] = provider.content_blocks_to_parts(
        [
            {
                "type": "image",
                "source": {"type": "url", "url": "https://b.example/k/a.jpg?X-Amz-Signature=s"},
            }
        ]
    )
    assert part.file_data is not None
    assert part.file_data.file_uri == "https://b.example/k/a.jpg?X-Amz-Signature=s"
    assert part.file_data.mime_type == "image/jpeg"


def test_astra_request_preserves_requested_reasoning_and_encrypted_state() -> None:
    provider = OpenAIProvider(model_id="gpt-6-astra", api_key="test", thinking_level="low")
    params = provider._request_params("System", [Message(role="user", content="hello")], [])
    reasoning = params.get("reasoning")
    assert reasoning is not None
    assert reasoning.get("effort") == "low"
    include = params.get("include")
    assert include is not None
    assert "reasoning.encrypted_content" in include


@pytest.mark.asyncio
async def test_openai_usage_callback_retains_cache_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from actant.runtime.events.streaming import StreamListener

    usage = {
        "input_tokens": 5040,
        "output_tokens": 1090,
        "total_tokens": 6130,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 4912},
    }
    response = SimpleNamespace(
        id="resp-test",
        status="completed",
        output=[],
        usage=SimpleNamespace(**usage, model_dump=lambda **kw: usage),
    )

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def get_final_response(self):
            return response

    received = []

    class Listener(StreamListener):
        async def on_usage(self, response_id, model, usage, status):
            received.append((response_id, model, usage, status))

    provider = OpenAIProvider(model_id="gpt-6-astra", api_key="test")
    monkeypatch.setattr(provider.client.responses, "stream", lambda **kw: Stream())
    message, total = await provider._stream_attempt({}, Listener())
    assert total == 6130
    assert message.input_tokens == 5040
    assert received == [("resp-test", "gpt-6-astra", usage, "completed")]
    await provider.client.close()


def test_replayed_url_images_drop_expires_at_and_expired_ones_become_a_note() -> None:
    from actant.llm.providers._shared import EXPIRED_IMAGE, sanitize_tool_messages

    live = {
        "type": "image",
        "source": {"type": "url", "url": "https://l", "expires_at": time.time() + 3600},
    }
    dead = {
        "type": "image",
        "source": {"type": "url", "url": "https://d", "expires_at": time.time() + 30},
    }
    stored = Message(role="tool", tool_call_id="t", content=[live, dead])
    [sent] = sanitize_tool_messages([stored])
    assert sent.content == [
        {"type": "image", "source": {"type": "url", "url": "https://l"}},
        {"type": "text", "text": EXPIRED_IMAGE},
    ]
    assert "expires_at" in live["source"]  # pyright: ignore[reportOperatorIssue] -- the stored message is untouched
