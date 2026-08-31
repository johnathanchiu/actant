"""Provider-reported token usage rides on the message it describes.

The counts exist so a caller can bill (or budget) per token. They come
off the provider response, ride the ``Message``, and land on the
message header row, so the transcript itself is the usage record and no
separate accounting stream can drift from it.

``None`` means the provider did not report usage. That is deliberately
distinct from ``0``: a biller must treat unknown as unknown, never as
free.
"""

from __future__ import annotations

import pytest

from actant.llm.messages import Message
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.stores.postgres import ActantMessageModel
from actant.runtime.stores.postgres.conversion import message_from_header

_AGENT = "a"
_THREAD = "t"


def test_total_tokens_is_none_when_the_provider_reported_nothing() -> None:
    assert Message(role="assistant").total_tokens is None


def test_total_tokens_keeps_a_reported_zero() -> None:
    assert Message(role="assistant", input_tokens=0, output_tokens=0).total_tokens == 0


def test_total_tokens_sums_both_halves() -> None:
    assert Message(role="assistant", input_tokens=7, output_tokens=3).total_tokens == 10


def test_total_tokens_tolerates_a_missing_half() -> None:
    """A provider reporting only output still yields a usable total."""
    assert Message(role="assistant", output_tokens=3).total_tokens == 3


def test_usage_round_trips_through_the_wire_form() -> None:
    original = Message(role="assistant", content="hi", input_tokens=11, output_tokens=5)
    restored = Message.from_raw(original.to_dict())
    assert (restored.input_tokens, restored.output_tokens) == (11, 5)


def test_usage_round_trips_through_a_message_copy() -> None:
    original = Message(role="assistant", content="hi", input_tokens=11, output_tokens=5)
    assert Message.from_raw(original).input_tokens == 11


def test_a_message_without_usage_stays_clean_on_the_wire() -> None:
    """Absent usage adds no keys, so existing consumers see no change."""
    plain = Message(role="user", content="hi").to_dict()
    assert "input_tokens" not in plain
    assert "output_tokens" not in plain


@pytest.mark.asyncio
async def test_usage_survives_the_message_log() -> None:
    stores = InMemoryRuntimeStores()
    await stores.messages.append_assistant(
        _AGENT,
        _THREAD,
        "turn",
        Message(role="assistant", content="hello", input_tokens=120, output_tokens=8),
    )

    (logged,) = await stores.messages.list_for_thread(_AGENT, _THREAD)
    assert (logged.input_tokens, logged.output_tokens) == (120, 8)


def test_usage_is_hydrated_from_the_message_header() -> None:
    """The counts live on the header row, not in a part."""
    row = ActantMessageModel(
        message_id="msg_1",
        agent_id=_AGENT,
        thread_id=_THREAD,
        turn_id="turn",
        role="assistant",
        input_tokens=120,
        output_tokens=8,
    )
    row.parts = []

    message = message_from_header(row)

    assert (message.input_tokens, message.output_tokens) == (120, 8)
    assert message.total_tokens == 128
