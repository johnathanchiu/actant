"""Multimodal round-trip through the parts ↔ messages helpers.

Exercises the ``content_blocks`` carrier on ``MessagePart`` —
specifically that user messages with mixed text + asset blocks survive
``message_to_parts`` → ``parts_to_messages`` unchanged, and that tool
results carrying ``content_blocks`` reconstruct as multimodal
``Message.content`` lists for the LLM.
"""

from __future__ import annotations

import json

from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.runtime.session import message_to_parts, parts_to_messages
from actant.runtime.types.session import MessagePart, PartKind


def test_user_text_only_round_trip_preserves_string_shape() -> None:
    parts = message_to_parts(Message(role="user", content="hello"))
    assert len(parts) == 1
    assert parts[0].kind is PartKind.USER_PROMPT
    assert parts[0].content == "hello"
    assert parts[0].content_blocks is None

    messages = parts_to_messages(parts)
    assert messages == [Message(role="user", content="hello")]


def test_user_multimodal_round_trip_preserves_block_list() -> None:
    blocks: list[dict[str, object]] = [
        {"type": "text", "text": "describe this"},
        {
            "type": "asset",
            "storage_key": "user_uploads/abc/xyz",
            "mime": "image/png",
            "asset_public_id": "asset_xyz",
        },
    ]

    parts = message_to_parts(Message(role="user", content=blocks))
    assert len(parts) == 1
    assert parts[0].kind is PartKind.USER_PROMPT
    assert parts[0].content_blocks == blocks
    assert parts[0].content is None

    [message] = parts_to_messages(parts)
    assert message.role == "user"
    assert message.content == blocks


def test_user_multimodal_drops_non_dict_entries() -> None:
    """Malformed payloads (e.g. a stray string mixed into the block list)
    shouldn't poison persistence — the bad entry is silently dropped."""
    parts = message_to_parts(
        Message(
            role="user",
            content=[
                {"type": "text", "text": "hi"},
                "stray-string",  # type: ignore[list-item]
                {"type": "asset", "storage_key": "k", "mime": "image/png"},
            ],
        )
    )
    assert parts[0].content_blocks == [
        {"type": "text", "text": "hi"},
        {"type": "asset", "storage_key": "k", "mime": "image/png"},
    ]


def test_tool_result_with_content_blocks_reconstructs_as_list() -> None:
    """When a tool's ``result`` dict carries a ``content_blocks`` key,
    the reconstructed tool message's content is a list (multimodal),
    not a JSON-stringified blob."""
    blocks: list[dict[str, object]] = [
        {"type": "text", "text": "rendered the floorplan"},
        {
            "type": "asset",
            "storage_key": "agent_artifacts/render-1.png",
            "mime": "image/png",
        },
    ]
    parts = [
        MessagePart(kind=PartKind.USER_PROMPT, content="render"),
        MessagePart(
            kind=PartKind.TOOL_CALL,
            tool_call_id="tc_1",
            tool_name="render",
            args={"layout": "grid"},
            result={"content_blocks": blocks, "metadata": {"width": 1024}},
        ),
    ]

    messages = parts_to_messages(parts)

    tool_message = next(m for m in messages if m.role == "tool")
    assert tool_message.tool_call_id == "tc_1"
    assert tool_message.content == blocks


def test_tool_result_without_content_blocks_keeps_json_string() -> None:
    """Legacy tool returns (no content_blocks) preserve the existing
    JSON-stringified shape — back-compat for every tool that doesn't
    opt into multimodal yet."""
    result = {"output": "ok", "metadata": {"exit_code": 0}}
    parts = [
        MessagePart(
            kind=PartKind.TOOL_CALL,
            tool_call_id="tc_legacy",
            tool_name="run",
            args={},
            result=result,
        ),
    ]

    messages = parts_to_messages(parts)
    tool_message = next(m for m in messages if m.role == "tool")
    assert tool_message.content == json.dumps(result)


def test_tool_call_thought_signature_round_trips() -> None:
    """Gemini requires the function-call thought_signature on replay;
    losing it makes the next post-tool turn fail before the model can
    continue."""
    message = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="tc_signed",
                function=ToolCallFunction(name="plot_points", arguments='{"points": []}'),
                thought_signature="signature_b64",
            )
        ],
    )

    parts = message_to_parts(message)
    tool_part = next(p for p in parts if p.kind is PartKind.TOOL_CALL)
    assert tool_part.signature == "signature_b64"

    [reconstructed] = parts_to_messages(parts)
    assert reconstructed.tool_calls is not None
    tool_call = reconstructed.tool_calls[0]
    assert tool_call.thought_signature == "signature_b64"
    assert tool_call.extra_content == {"google": {"thought_signature": "signature_b64"}}


def test_assistant_text_unchanged() -> None:
    """The slice 2a change is scoped to USER_PROMPT and TOOL_RESULT
    only — assistant TEXT parts stay text-only since the LLM protocols
    don't carry multimodal in assistant text."""
    message = Message(role="assistant", content="the answer is 42")
    parts = message_to_parts(message)
    assert any(p.kind is PartKind.TEXT and p.content == "the answer is 42" for p in parts)

    [reconstructed] = parts_to_messages(parts)
    assert reconstructed.role == "assistant"
    assert reconstructed.content == "the answer is 42"
