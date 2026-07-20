from __future__ import annotations

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
