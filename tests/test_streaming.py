"""Streaming primitives — fake LLM token deltas + cancel.

End-to-end runtime streaming is covered in
``tests/test_workflow_thread.py`` (the ``listener_factory`` runs inside
``run_turn``). Here we exercise the LLM/listener contract without
spinning up a full Temporal worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from actant.llm.errors import StreamCancelled
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime.events.streaming import StreamListener


@dataclass
class RecordingListener(StreamListener):
    text_deltas: list[str] = field(default_factory=list)
    thinking_deltas: list[str] = field(default_factory=list)

    async def on_text_delta(self, delta: str) -> None:
        self.text_deltas.append(delta)

    async def on_thinking_delta(self, delta: str) -> None:
        self.thinking_deltas.append(delta)


@dataclass
class CancelAfterFirstListener(RecordingListener):
    cancel_after: int = 1

    def cancel_requested(self) -> bool:
        return len(self.text_deltas) >= self.cancel_after


@pytest.mark.asyncio
async def test_fake_llm_emits_text_deltas_in_order() -> None:
    llm = FakeLLM(
        [FakeResponse(text="Hello", text_chunks=["He", "llo"], thinking_chunks=["thinking"])]
    )
    listener = RecordingListener()

    message = await llm.complete("sys", [], [], listener)

    assert listener.text_deltas == ["He", "llo"]
    assert listener.thinking_deltas == ["thinking"]
    assert message.content == "Hello"


@pytest.mark.asyncio
async def test_fake_llm_no_listener_returns_identical_message() -> None:
    llm = FakeLLM([FakeResponse(text="Hello", text_chunks=["He", "llo"])])
    message = await llm.complete("sys", [], [])
    assert message.content == "Hello"
    assert message.tool_calls is None


@pytest.mark.asyncio
async def test_fake_llm_cancel_raises_stream_cancelled() -> None:
    llm = FakeLLM([FakeResponse(text="Hello", text_chunks=["He", "ll", "o"])])
    listener = CancelAfterFirstListener()

    with pytest.raises(StreamCancelled):
        await llm.complete("sys", [], [], listener)

    assert listener.text_deltas == ["He"]
