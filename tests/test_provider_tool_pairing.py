"""Provider snapshots pair orphan calls truthfully, without changing stored history."""

import json

from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.llm.providers._shared import TOOL_OUTPUT_UNAVAILABLE, sanitize_tool_messages
from actant.llm.providers.openai import OpenAIProvider


def call(call_id, name="write"):
    return ToolCall(id=call_id, function=ToolCallFunction(name=name, arguments="{}"))


def assistant(*ids):
    return Message(role="assistant", content="", tool_calls=[call(id) for id in ids])


def output(call_id, text="actual result"):
    return Message(role="tool", tool_call_id=call_id, name="write", content=text)


def _text(message: Message) -> str:
    assert isinstance(message.content, str)
    return message.content


def _body(message: Message) -> dict[str, str]:
    return json.loads(_text(message))


def test_saved_re8ad81_orphan_gets_uncertain_output_only_in_request(caplog):
    call_id = "call_loHJN40ZI9cYNTMIZvrn9lGG"
    messages = [assistant(call_id)]
    before = [m.to_dict() for m in messages]
    paired = sanitize_tool_messages(messages)
    assert [m.to_dict() for m in messages] == before
    assert len(paired) == 2
    assert paired[1].tool_call_id == call_id
    assert _body(paired[1]) == {
        "status": "interrupted",
        "error": TOOL_OUTPUT_UNAVAILABLE,
    }
    assert "may or may not have taken effect" in _text(paired[1])
    assert "Check its effects before retrying" in _text(paired[1])
    assert call_id in caplog.text
    wire = OpenAIProvider.convert_messages(paired)
    assert wire[1]["type"] == "function_call_output"
    assert wire[1]["call_id"] == call_id


def test_real_later_output_wins():
    messages = [assistant("a"), Message(role="user", content="follow-up"), output("a")]
    assert [m.to_dict() for m in sanitize_tool_messages(messages)] == [
        m.to_dict() for m in messages
    ]


def test_parallel_calls_only_fill_missing_results():
    messages = [assistant("a", "b", "c"), output("c"), output("a")]
    paired = sanitize_tool_messages(messages)
    results = [m for m in paired if m.role == "tool"]
    assert [m.tool_call_id for m in results] == ["b", "c", "a"]
    assert _body(results[0])["status"] == "interrupted"
    assert [m.content for m in results[1:]] == ["actual result", "actual result"]


def test_duplicate_results_sent_once_first_result_preserved():
    messages = [assistant("a"), output("a", "first"), output("a", "second")]
    paired = sanitize_tool_messages(messages)
    assert len(paired) == 2 and paired[1].content == "first"
    assert len(messages) == 3


def test_cancelled_history_reopened_with_new_user_request():
    messages = [
        assistant("old"),
        Message(role="user", content="new assignment"),
        assistant("new"),
        output("new"),
    ]
    paired = sanitize_tool_messages(messages)
    assert paired[1].tool_call_id == "old"
    assert _body(paired[1])["status"] == "interrupted"
    assert paired[2].content == "new assignment"
    assert paired[-1].content == "actual result"


def test_well_formed_trace_and_second_sanitization_unchanged():
    messages = [
        Message(role="user", content="request"),
        assistant("a", "b"),
        output("b"),
        output("a"),
        Message(role="assistant", content="done"),
    ]
    paired = sanitize_tool_messages(messages)
    assert [m.to_dict() for m in paired] == [m.to_dict() for m in messages]
    orphan = sanitize_tool_messages([assistant("missing")])
    assert [m.to_dict() for m in sanitize_tool_messages(orphan)] == [m.to_dict() for m in orphan]
