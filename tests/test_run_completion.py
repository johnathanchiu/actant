"""A run's last word and its thread's inbound tags, as its completion carries them."""

from actant.blocks import TextBlock
from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.runtime.completion import LastCall, inbound_tags, last_word


def _call(name: str, arguments: str) -> ToolCall:
    return ToolCall(id=name, function=ToolCallFunction(name=name, arguments=arguments))


def test_the_last_word_is_after_the_last_inbound_message() -> None:
    messages = [
        Message(role="user", content="first", tag="brief"),
        Message(role="assistant", content="old answer"),
        Message(role="user", content="again", tag="brief"),
        Message(role="assistant", content=[TextBlock(text="checking")]),
        Message(
            role="assistant",
            content="",
            tool_calls=[_call("done", '{"report": "built"}'), _call("bad", "{not json")],
        ),
    ]
    assert last_word(messages) == ("checking", LastCall("done", {"report": "built"}))
    assert inbound_tags(messages) == ("brief", "brief")


def test_a_run_that_said_nothing_since_its_message_has_no_last_word() -> None:
    messages = [Message(role="assistant", content="before"), Message(role="user", content="hi")]
    assert last_word(messages) == ("", None)
    assert inbound_tags(messages) == ()
