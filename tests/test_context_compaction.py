"""Model-context compaction: a summarizing turn near the real limits, and nothing else.

The rules these hold: context is never truncated or rewritten; the only reducer is a
summary; it fires only when the next request would cross the token fraction or the image
limit; the fresh context is exactly system + summary + pinned + pending; every stored
message is left as it was; and a history recorded without compaction replays unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import replace
from pathlib import Path

from temporalio.client import WorkflowHistory
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

from actant.agents import AgentDefinition
from actant.blocks import Base64Source, Block, InlineImageBlock, TextBlock
from actant.core import JSONObject
from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime import AgentRuntime
from actant.runtime.compaction import (
    COMPACTION_PROMPT,
    Compaction,
    CompactionHook,
    PinProvider,
    pinned_item,
    summary_message,
)
from actant.runtime.local import LocalThreadRuntime
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.activities import TemporalRuntimeActivities
from actant.runtime.temporal.activities.context import ActivityContext
from actant.runtime.temporal.types import (
    CompactionConfig,
    InboundMessage,
    RunOutcome,
    TemporalRuntimeConfig,
    ThreadInput,
)
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from actant.tools import PinNoteTool, ToolRegistry, tool
from runtime_fixtures import static_agents

_HISTORIES = Path(__file__).parent / "histories"
_AGENT = "compacting"
_THREAD = "t1"
_PERSONA = "You are careful."
_IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}
_PINNED_IMAGE = InlineImageBlock(source=Base64Source(media_type="image/png", data="BB=="))


@tool
async def note(text: str) -> str:
    """Write something down."""

    return f"noted: {text}"


def _call(text: str, *, input_tokens: int | None = None) -> FakeResponse:
    return FakeResponse(
        tool_calls=[
            ToolCall(
                id=f"tc_{uuid.uuid4().hex[:8]}",
                function=ToolCallFunction("note", f'{{"text": "{text}"}}'),
            )
        ],
        input_tokens=input_tokens,
        output_tokens=10 if input_tokens is not None else None,
    )


def _says(text: str, *, input_tokens: int | None = None) -> FakeResponse:
    return FakeResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=10 if input_tokens is not None else None,
    )


class _Events:
    def __init__(self) -> None:
        self.compacted: list[JSONObject] = []

    async def publish(self, channel: str, event: JSONObject) -> None:
        data = event["data"]
        if event["type"] == "context_compacted" and isinstance(data, dict):
            self.compacted.append(data)


class _Setup:
    def __init__(
        self,
        replies: list[FakeResponse],
        *,
        on_compact: CompactionHook | None = None,
        pin_providers: dict[str, PinProvider] | None = None,
        context_window_tokens: int | None = None,
        max_images_per_request: int | None = None,
    ) -> None:
        self.llm = FakeLLM(
            replies,
            context_window_tokens=context_window_tokens,
            max_images_per_request=max_images_per_request,
        )
        self.agent = AgentDefinition(
            id=_AGENT,
            name="Compacting",
            persona=_PERSONA,
            llm=self.llm,
            tools=ToolRegistry([note]),
        )
        self.stores = InMemoryRuntimeStores()
        self.agent = replace(
            self.agent, tools=ToolRegistry([note, PinNoteTool(self.stores.pinned_notes)])
        )
        self.events = _Events()
        self.context = ActivityContext(
            stores=self.stores,
            resolve_agent=static_agents({_AGENT: self.agent}),
            event_sink=self.events,
            on_compact=on_compact,
            pin_providers=pin_providers,
        )
        self.runtime = LocalThreadRuntime(self.context)

    async def run(
        self, content: str | list[dict[str, object]], compaction: CompactionConfig | None
    ) -> RunOutcome:
        outcome = await self.runtime.run(
            ThreadInput(
                agent_id=_AGENT,
                thread_id=_THREAD,
                max_turns_per_run=6,
                context_compaction=compaction,
            ),
            [InboundMessage(content=content)],
        )
        return outcome.outcome

    async def stored(self) -> list[Message]:
        return await self.stores.messages.list_for_thread(_AGENT, _THREAD)


async def _pin(compaction: Compaction) -> list[Block]:
    return [TextBlock(text=f"checklist after {compaction.reason}"), _PINNED_IMAGE]


# === when it fires ===


async def test_the_token_fraction_triggers_one_summary_turn_without_tools() -> None:
    s = _Setup(
        [
            _call("a", input_tokens=950),  # 960 of a 1,000 window, plus the tool result
            _says("SUMMARY: noted a; next, answer."),
            _says("done", input_tokens=120),
        ]
    )
    window = CompactionConfig(context_window_tokens=1_000)
    assert await s.run("first", window) is RunOutcome.COMPLETED

    assert len(s.llm.calls) == 3
    system, request, tools = s.llm.calls[1]
    assert tools == []
    assert system == _PERSONA
    # The summary replaces what came before the pending call; the prompt is last.
    assert [m.content for m in request] == ["first", COMPACTION_PROMPT]

    [record] = await s.stores.compactions.list_for_thread(_AGENT, _THREAD)
    assert record.reason == "tokens"
    assert record.boundary == 1
    assert record.tokens_before > 900
    assert record.tokens_after < record.tokens_before
    assert record.summary == "SUMMARY: noted a; next, answer."
    assert s.events.compacted[0]["summary"] == record.summary
    assert s.events.compacted[0]["tokens_before"] == record.tokens_before


async def test_the_image_limit_triggers_before_the_token_fraction() -> None:
    s = _Setup([_says("two seen", input_tokens=10), _says("SUMMARY"), _says("three")])
    limits = CompactionConfig(context_window_tokens=1_000_000, max_images_per_request=2)
    await s.run([{"type": "text", "text": "look"}, _IMAGE, _IMAGE], limits)
    assert await s.stores.compactions.list_for_thread(_AGENT, _THREAD) == []

    await s.run([{"type": "text", "text": "and this"}, _IMAGE], limits)

    [record] = await s.stores.compactions.list_for_thread(_AGENT, _THREAD)
    assert record.reason == "images"
    assert (record.images_before, record.images_after) == (3, 1)
    assert record.boundary == 2
    _, request, tools = s.llm.calls[1]
    assert tools == []
    assert request[-1].content == COMPACTION_PROMPT


async def test_the_provider_supplies_limits_the_config_leaves_unset() -> None:
    s = _Setup(
        [_call("a", input_tokens=950), _says("SUMMARY"), _says("done")],
        context_window_tokens=1_000,
    )
    await s.run("first", CompactionConfig())
    assert len(await s.stores.compactions.list_for_thread(_AGENT, _THREAD)) == 1


async def test_nothing_fires_below_both_limits() -> None:
    s = _Setup([_call("a", input_tokens=800), _says("done", input_tokens=850)])
    limits = CompactionConfig(context_window_tokens=1_000, max_images_per_request=3)
    await s.run([{"type": "text", "text": "look"}, _IMAGE, _IMAGE, _IMAGE], limits)

    assert len(s.llm.calls) == 2
    assert all(tools for _, _, tools in s.llm.calls)
    assert await s.stores.compactions.list_for_thread(_AGENT, _THREAD) == []
    assert s.events.compacted == []


async def test_without_the_setting_nothing_is_measured() -> None:
    s = _Setup(
        [_call("a", input_tokens=950), _says("done")],
        context_window_tokens=1_000,
        max_images_per_request=1,
    )
    await s.run([{"type": "text", "text": "look"}, _IMAGE, _IMAGE], None)
    assert len(s.llm.calls) == 2
    assert await s.stores.compactions.list_for_thread(_AGENT, _THREAD) == []


# === what the model sees afterwards, and what the store keeps ===


async def test_the_fresh_context_is_system_summary_pinned_then_pending() -> None:
    s = _Setup(
        [_call("a", input_tokens=950), _says("SUMMARY"), _says("done", input_tokens=40)],
        on_compact=_pin,
    )
    await s.run("first", CompactionConfig(context_window_tokens=1_000))

    system, fresh, tools = s.llm.calls[2]
    stored = await s.stored()
    assert system == _PERSONA
    assert tools  # the continued turn has its tools back
    assert fresh[0] == summary_message("SUMMARY")
    assert fresh[1] == Message(
        role="user", content=[TextBlock(text="checklist after tokens"), _PINNED_IMAGE]
    )
    # Pending: the call whose result had not been seen, and that result.
    assert fresh[2:] == stored[1:3]
    assert [m.role for m in fresh[2:]] == ["assistant", "tool"]


async def test_old_messages_stay_in_the_store_untouched() -> None:
    s = _Setup([_says("one", input_tokens=950), _says("SUMMARY"), _says("two")])
    window = CompactionConfig(context_window_tokens=1_000)
    await s.run("first", window)
    before = [Message.from_raw(m) for m in await s.stored()]

    await s.run("second", window)

    after = await s.stored()
    assert after[: len(before)] == before
    assert [(m.role, m.content) for m in after] == [
        ("user", "first"),
        ("assistant", "one"),
        ("user", "second"),
        ("assistant", "two"),
    ]
    _, fresh, _ = s.llm.calls[2]
    assert fresh == [summary_message("SUMMARY"), after[2]]


async def test_a_second_compaction_summarizes_the_first_summary_onward() -> None:
    s = _Setup(
        [
            _says("one", input_tokens=950),
            _says("SUMMARY 1"),
            _says("two", input_tokens=960),
            _says("SUMMARY 2"),
            _says("three"),
        ]
    )
    window = CompactionConfig(context_window_tokens=1_000)
    for text in ("first", "second", "third"):
        await s.run(text, window)

    first, second = await s.stores.compactions.list_for_thread(_AGENT, _THREAD)
    assert (first.boundary, second.boundary) == (2, 4)
    _, request, _ = s.llm.calls[3]
    assert [m.content for m in request] == [
        summary_message("SUMMARY 1").content,
        "second",
        "two",
        COMPACTION_PROMPT,
    ]
    _, fresh, _ = s.llm.calls[4]
    assert [m.content for m in fresh] == [summary_message("SUMMARY 2").content, "third"]
    assert len(await s.stored()) == 6


# === state that must not depend on the summary ===

_CHECKLIST = (
    "- [x] measure the room\n- [ ] place the sofa (open)\n- [ ] check the door swing (open)"
)


def _pin_call(key: str, text: str, *, input_tokens: int) -> FakeResponse:
    arguments = json.dumps({"key": key, "text": text})
    return FakeResponse(
        tool_calls=[ToolCall(id="tc_pin", function=ToolCallFunction("pin_note", arguments))],
        input_tokens=input_tokens,
        output_tokens=10,
    )


async def test_a_pinned_checklist_follows_the_summary_verbatim_whatever_it_says() -> None:
    s = _Setup(
        [
            _pin_call("checklist", _CHECKLIST, input_tokens=950),
            _says("The user likes blue."),  # a summary that forgets every open item
            _says("continuing"),
        ]
    )
    await s.run("plan the room", CompactionConfig(context_window_tokens=1_000))

    _, fresh, _ = s.llm.calls[2]
    assert fresh[0] == summary_message("The user likes blue.")
    assert fresh[1] == Message(
        role="user", content=[*pinned_item("checklist", [TextBlock(text=_CHECKLIST)])]
    )
    [record] = await s.stores.compactions.list_for_thread(_AGENT, _THREAD)
    assert record.pinned == pinned_item("checklist", [TextBlock(text=_CHECKLIST)])


async def test_pin_orders_providers_and_notes_then_the_hook_runs_last() -> None:
    async def room_file(compaction: Compaction) -> list[Block]:
        return [TextBlock(text=f"room.py for {compaction.thread_id}")]

    async def hook(compaction: Compaction) -> list[Block]:
        return [TextBlock(text="from the hook")]

    s = _Setup(
        [_call("a", input_tokens=950), _says("SUMMARY"), _says("done")],
        on_compact=hook,
        pin_providers={"room_file": room_file},
    )
    await s.stores.pinned_notes.pin(_AGENT, _THREAD, "zeta", "unnamed note")
    await s.stores.pinned_notes.pin(_AGENT, _THREAD, "checklist", _CHECKLIST)
    config = CompactionConfig(context_window_tokens=1_000, pin=["checklist", "room_file"])
    await s.run("first", config)

    _, fresh, _ = s.llm.calls[2]
    assert fresh[1].content == [
        *pinned_item("checklist", [TextBlock(text=_CHECKLIST)]),
        *pinned_item("room_file", [TextBlock(text=f"room.py for {_THREAD}")]),
        *pinned_item("zeta", [TextBlock(text="unnamed note")]),
        TextBlock(text="from the hook"),
    ]


async def test_pinned_notes_survive_every_compaction(caplog) -> None:  # type: ignore[no-untyped-def]
    s = _Setup(
        [
            _says("one", input_tokens=950),
            _says("SUMMARY 1"),
            _says("two", input_tokens=960),
            _says("SUMMARY 2"),
            _says("three"),
        ]
    )
    await s.stores.pinned_notes.pin(_AGENT, _THREAD, "checklist", _CHECKLIST)
    window = CompactionConfig(context_window_tokens=1_000, pin=["checklist", "missing"])
    with caplog.at_level(logging.WARNING):
        for text in ("first", "second", "third"):
            await s.run(text, window)

    pinned = Message(
        role="user", content=[*pinned_item("checklist", [TextBlock(text=_CHECKLIST)])]
    )
    assert s.llm.calls[2][1][1] == pinned
    assert s.llm.calls[4][1][1] == pinned
    assert "pin_missing" in caplog.text and "name=missing" in caplog.text
    assert "no on_compact hook" in caplog.text


async def test_without_a_hook_or_pins_the_log_says_nothing_is_pinned(caplog) -> None:  # type: ignore[no-untyped-def]
    s = _Setup([_says("one", input_tokens=950), _says("SUMMARY"), _says("two")])
    window = CompactionConfig(context_window_tokens=1_000)
    with caplog.at_level(logging.WARNING):
        await s.run("first", window)
        await s.run("second", window)
    assert "nothing is pinned after the summary" in caplog.text


# === the workflow ===


async def _through_the_workflow(
    compaction: CompactionConfig | None,
) -> tuple[FakeLLM, InMemoryRuntimeStores, WorkflowHistory]:
    summary = [_says("SUMMARY")] if compaction is not None else []
    llm = FakeLLM([_call("a", input_tokens=950), *summary, _says("done")])
    agent = AgentDefinition(
        id=_AGENT, name="Compacting", persona=_PERSONA, llm=llm, tools=ToolRegistry([note])
    )
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({_AGENT: agent}))
    )
    task_queue = f"test-actant-{uuid.uuid4().hex[:8]}"
    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=activities.all,
        ):
            config = TemporalRuntimeConfig(task_queue=task_queue, context_compaction=compaction)
            runtime = AgentRuntime(client=env.client, stores=stores, config=config)
            workflow_id = await runtime.send_message(_AGENT, _THREAD, "first")
            handle = env.client.get_workflow_handle(workflow_id)
            await asyncio.wait_for(handle.result(), timeout=20.0)
            history = await handle.fetch_history()
    return llm, stores, history


async def test_the_workflow_compacts_then_runs_the_same_turn_and_replays() -> None:
    llm, stores, history = await _through_the_workflow(
        CompactionConfig(context_window_tokens=1_000)
    )
    assert len(llm.calls) == 3
    assert llm.calls[1][2] == []
    [record] = await stores.compactions.list_for_thread(_AGENT, _THREAD)
    assert [run.turn_count for run in await stores.runs.list_for_thread(_AGENT, _THREAD)] == [2]
    assert (await stores.threads.get(_AGENT, _THREAD)).turn_count == 2
    assert record.summary == "SUMMARY"
    await Replayer(workflows=[AgentThreadWorkflow]).replay_workflow(history)


async def test_the_workflow_without_compaction_replays() -> None:
    llm, stores, history = await _through_the_workflow(None)
    assert len(llm.calls) == 2
    assert await stores.compactions.list_for_thread(_AGENT, _THREAD) == []
    await Replayer(workflows=[AgentThreadWorkflow]).replay_workflow(history)


async def test_a_history_recorded_before_compaction_existed_replays() -> None:
    """Recorded by the workflow on main before compaction: two tool turns, then an answer."""
    history = WorkflowHistory.from_json(
        "thread-recorded",
        (_HISTORIES / "tool_turns_before_compaction.json").read_text(),
    )
    await Replayer(workflows=[AgentThreadWorkflow]).replay_workflow(history)
