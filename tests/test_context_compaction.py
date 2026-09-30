"""Model-context compaction: one summary call near the real limits, and nothing else.

The rules these hold: no stored message is truncated, rewritten or deleted; the only
reducer is a summary; it fires only when the next request would cross the token fraction
or the image limit; the model then sees system + summary + kept messages + the rows after
the compaction row; and a history recorded without compaction replays unchanged.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from pathlib import Path

from temporalio.client import WorkflowHistory
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

from actant.agents import AgentDefinition
from actant.blocks import AssetBlock, CompactionBlock, TextBlock
from actant.core import JSONObject
from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime import AgentRuntime
from actant.runtime.compaction import (
    COMPACTION_PROMPT,
    compaction_of,
    count_images,
    estimate_tokens,
    retained,
    summary_message,
)
from actant.runtime.events.streaming import StreamListener
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
from actant.tools import RecallImageTool, ToolRegistry, tool
from actant.tools.base import CallContext
from runtime_fixtures import static_agents

_HISTORIES = Path(__file__).parent / "histories"
_AGENT = "compacting"
_THREAD = "t1"
_PERSONA = "You are careful."
_IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}
_WINDOW = CompactionConfig(context_window_tokens=10_000)


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
        self.events = _Events()
        self.runtime = LocalThreadRuntime(
            ActivityContext(
                stores=self.stores,
                resolve_agent=static_agents({_AGENT: self.agent}),
                event_sink=self.events,
            )
        )

    async def run(
        self,
        content: str | list[dict[str, object]],
        compaction: CompactionConfig | None,
        tag: str | None = None,
    ) -> RunOutcome:
        outcome = await self.runtime.run(
            ThreadInput(
                agent_id=_AGENT,
                thread_id=_THREAD,
                max_turns_per_run=6,
                context_compaction=compaction,
            ),
            [InboundMessage(content=content, tag=tag)],
        )
        return outcome.outcome

    async def stored(self) -> list[Message]:
        return await self.stores.messages.list_for_thread(_AGENT, _THREAD)

    async def compactions(self) -> list[tuple[int, CompactionBlock]]:
        return [
            (i, block)
            for i, m in enumerate(await self.stored())
            if (block := compaction_of(m)) is not None
        ]


def _texts(messages: Sequence[Message]) -> list[object]:
    return [
        m.content if isinstance(m.content, str) and m.role != "tool" else m.role for m in messages
    ]


# === when it fires ===


async def test_the_token_fraction_triggers_one_summary_call_within_the_margin() -> None:
    s = _Setup(
        [
            _call("a", input_tokens=9_500),  # 9,510 of a 10,000 window, plus the tool result
            _says("SUMMARY: noted a; next, answer."),
            _says("done", input_tokens=120),
        ]
    )
    assert await s.run("first", _WINDOW) is RunOutcome.COMPLETED

    system, request, tools = s.llm.calls[1]
    assert (system, tools) == (_PERSONA, [])
    assert _texts(request) == ["first", COMPACTION_PROMPT]
    # The summary's input and its capped output fit the window together.
    max_output = s.llm.max_output_tokens[1]
    assert max_output is not None
    assert estimate_tokens(request, _PERSONA) + max_output <= 10_000

    [(index, block)] = await s.compactions()
    assert (index, block.reason) == (3, "tokens")
    assert block.summary == "SUMMARY: noted a; next, answer."
    assert block.tokens_before > 9_000 > block.tokens_after
    assert s.events.compacted[0]["summary"] == block.summary


async def test_the_image_limit_triggers_and_the_summary_call_stays_under_it() -> None:
    s = _Setup([_says("two seen", input_tokens=10), _says("SUMMARY"), _says("three")])
    limits = CompactionConfig(context_window_tokens=1_000_000, max_images_per_request=2)
    await s.run([{"type": "text", "text": "look"}, _IMAGE, _IMAGE], limits)
    assert await s.compactions() == []

    await s.run([{"type": "text", "text": "and this"}, _IMAGE], limits)

    [(index, block)] = await s.compactions()
    assert block.reason == "images"
    assert (block.images_before, block.images_after) == (3, 0)
    _, request, tools = s.llm.calls[1]
    assert tools == [] and count_images(request) <= 2
    # The message that crossed the limit is stored after the compaction row.
    stored = await s.stored()
    assert index == 2 and count_images(stored[3:]) == 1
    _, fresh, _ = s.llm.calls[2]
    assert fresh == [summary_message("SUMMARY"), stored[3]]


async def test_the_provider_supplies_limits_the_config_leaves_unset() -> None:
    s = _Setup(
        [_call("a", input_tokens=9_500), _says("SUMMARY"), _says("done")],
        context_window_tokens=10_000,
    )
    await s.run("first", CompactionConfig())
    assert len(await s.compactions()) == 1


async def test_nothing_fires_below_both_limits_or_without_the_setting() -> None:
    s = _Setup([_call("a", input_tokens=8_000), _says("done", input_tokens=8_500)])
    limits = CompactionConfig(context_window_tokens=10_000, max_images_per_request=3)
    await s.run([{"type": "text", "text": "look"}, _IMAGE, _IMAGE, _IMAGE], limits)
    assert len(s.llm.calls) == 2 and all(tools for _, _, tools in s.llm.calls)

    off = _Setup(
        [_call("a", input_tokens=9_500), _says("done")],
        context_window_tokens=10_000,
        max_images_per_request=1,
    )
    await off.run([{"type": "text", "text": "look"}, _IMAGE, _IMAGE], None)
    assert len(off.llm.calls) == 2
    assert await s.compactions() == [] and await off.compactions() == []


# === what the model sees, and what the store keeps ===


async def test_the_view_is_summary_then_the_latest_of_each_kept_tag_then_the_rest() -> None:
    s = _Setup(
        [
            _call("a", input_tokens=100),
            _call("b", input_tokens=200),
            _says("ok", input_tokens=9_500),
            _says("SUMMARY"),
            _says("next done"),
        ]
    )
    keep = CompactionConfig(context_window_tokens=10_000, keep=["brief", "tool:note"])
    await s.run("the brief", keep, tag="brief")
    await s.run("next", keep)

    stored = await s.stored()
    assert [m.tag for m in stored[:6]] == ["brief", None, "tool:note", None, "tool:note", None]
    [(index, block)] = await s.compactions()
    # tool:note is kept at its latest version (note b), not the first.
    assert (index, block.kept) == (6, [stored[0].id, stored[4].id])
    system, fresh, _ = s.llm.calls[4]
    assert system == _PERSONA
    assert fresh == [summary_message("SUMMARY"), stored[0], retained(stored[4]), stored[7]]
    kept_result = fresh[2]
    assert kept_result.role == "user" and isinstance(kept_result.content, list)
    assert kept_result.content[1:] == [TextBlock(text=str(stored[4].content))]


async def test_the_boundary_never_splits_a_tool_call_from_its_result() -> None:
    s = _Setup([_call("a", input_tokens=9_500), _says("SUMMARY"), _says("done")])
    await s.run("first", _WINDOW)

    stored = await s.stored()
    _, request, _ = s.llm.calls[1]
    assert not any(m.tool_calls for m in request)
    [(index, block)] = await s.compactions()
    assert (index, block.kept) == (3, [stored[1].id, stored[2].id])
    _, fresh, _ = s.llm.calls[2]
    assert fresh == [summary_message("SUMMARY"), stored[1], stored[2]]
    assert [m.role for m in fresh[1:]] == ["assistant", "tool"]


async def test_old_messages_stay_untouched_and_a_second_compaction_chains() -> None:
    s = _Setup(
        [
            _says("one", input_tokens=9_500),
            _says("SUMMARY 1"),
            _says("two", input_tokens=9_600),
            _says("SUMMARY 2"),
            _says("three"),
        ]
    )
    await s.run("first", _WINDOW)
    before = [Message.from_raw(m) for m in await s.stored()]
    await s.run("second", _WINDOW)
    await s.run("third", _WINDOW)

    stored = await s.stored()
    assert stored[: len(before)] == before
    assert _texts(stored) == ["first", "one", "user", "second", "two", "user", "third", "three"]
    _, request, _ = s.llm.calls[3]
    assert _texts(request) == [
        summary_message("SUMMARY 1").content,
        "second",
        "two",
        COMPACTION_PROMPT,
    ]
    _, fresh, _ = s.llm.calls[4]
    assert _texts(fresh) == [summary_message("SUMMARY 2").content, "third"]


async def test_list_for_model_reads_from_the_latest_compaction_row_plus_its_kept_rows() -> None:
    s = _Setup(
        [
            _says("one", input_tokens=9_500),
            _says("SUMMARY 1"),
            _says("two", input_tokens=9_600),
            _says("SUMMARY 2"),
            _says("three"),
        ]
    )
    keep = CompactionConfig(context_window_tokens=10_000, keep=["brief"])
    await s.run("the brief", keep, tag="brief")
    await s.run("second", keep)
    await s.run("third", keep)

    rows = await s.stores.messages.list_for_model(_AGENT, _THREAD)
    stored = await s.stored()
    assert [m.id for m in rows] == [stored[i].id for i in (0, 5, 6, 7)]
    assert compaction_of(rows[1]) == (await s.compactions())[1][1]


async def test_a_rejected_summary_call_fails_the_run_and_drops_nothing() -> None:
    s = _Setup([_call("a", input_tokens=9_500)])  # no reply queued: the summary call raises
    assert await s.run("first", _WINDOW) is RunOutcome.FAILED
    assert await s.compactions() == []
    assert [m.role for m in await s.stored()] == ["user", "assistant", "tool"]


async def test_recall_image_returns_the_stored_asset_block() -> None:
    stores = InMemoryRuntimeStores()
    image = AssetBlock(storage_key="k/1.png", mime="image/png", asset_public_id="img_1")
    await stores.messages.append_user(_AGENT, _THREAD, [TextBlock(text="look"), image])
    ctx = CallContext(
        agent_id=_AGENT, thread_id=_THREAD, run_id="r", tool_call_id="c", turn_id="t"
    )
    tool_ = RecallImageTool(stores.messages)
    assert (await (await tool_.build({"id": "img_1"}, ctx)).execute()).content_blocks == [image]
    assert (await (await tool_.build({"id": "nope"}, ctx)).execute()).error is not None


# === the workflow ===


class _GatedLLM(FakeLLM):
    """Holds the summary call until the test releases it."""

    def __init__(self, replies: list[FakeResponse]) -> None:
        super().__init__(replies)
        self.summarizing = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: list[dict],
        listener: StreamListener | None = None,
        *,
        allowed_tools: tuple[str, ...] = (),
        max_output_tokens: int | None = None,
    ) -> Message:
        if not tools:
            self.summarizing.set()
            await self.release.wait()
        return await super().complete(
            system,
            messages,
            tools,
            listener,
            allowed_tools=allowed_tools,
            max_output_tokens=max_output_tokens,
        )


async def _through_the_workflow(
    compaction: CompactionConfig | None, *, message_during_compaction: bool = False
) -> tuple[FakeLLM, InMemoryRuntimeStores, WorkflowHistory]:
    summary = [_says("SUMMARY")] if compaction is not None else []
    llm = _GatedLLM([_call("a", input_tokens=9_500), *summary, _says("done")])
    if not message_during_compaction:
        llm.release.set()
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
            config = TemporalRuntimeConfig(
                task_queue=task_queue,
                context_compaction=compaction,
                interleave_inbox=message_during_compaction,
            )
            runtime = AgentRuntime(client=env.client, stores=stores, config=config)
            workflow_id = await runtime.send_message(_AGENT, _THREAD, "first")
            if message_during_compaction:
                await asyncio.wait_for(llm.summarizing.wait(), timeout=20.0)
                await runtime.send_message(_AGENT, _THREAD, "meanwhile")
                await asyncio.sleep(0.5)  # let the signal land before the summary returns
                llm.release.set()
            handle = env.client.get_workflow_handle(workflow_id)
            await asyncio.wait_for(handle.result(), timeout=20.0)
            history = await handle.fetch_history()
    return llm, stores, history


async def test_the_workflow_compacts_then_runs_the_same_turn_and_replays() -> None:
    llm, stores, history = await _through_the_workflow(_WINDOW)
    assert len(llm.calls) == 3 and llm.calls[1][2] == []
    stored = await stores.messages.list_for_thread(_AGENT, _THREAD)
    assert [m.kind for m in stored].count("compaction") == 1
    assert [run.turn_count for run in await stores.runs.list_for_thread(_AGENT, _THREAD)] == [2]
    await Replayer(workflows=[AgentThreadWorkflow]).replay_workflow(history)


async def test_a_message_arriving_during_compaction_lands_after_the_compaction_row() -> None:
    llm, stores, history = await _through_the_workflow(_WINDOW, message_during_compaction=True)
    stored = await stores.messages.list_for_thread(_AGENT, _THREAD)
    assert [m.kind for m in stored] == ["message"] * 3 + ["compaction"] + ["message"] * 2
    assert _texts([stored[0], *stored[4:]]) == ["first", "meanwhile", "done"]
    _, fresh, _ = llm.calls[2]
    assert _texts(fresh) == [
        summary_message("SUMMARY").content,
        "assistant",
        "tool",
        "meanwhile",
    ]
    await Replayer(workflows=[AgentThreadWorkflow]).replay_workflow(history)


async def test_the_workflow_without_compaction_replays() -> None:
    llm, stores, history = await _through_the_workflow(None)
    assert len(llm.calls) == 2
    stored = await stores.messages.list_for_thread(_AGENT, _THREAD)
    assert all(m.kind == "message" for m in stored)
    await Replayer(workflows=[AgentThreadWorkflow]).replay_workflow(history)


async def test_a_history_recorded_before_compaction_existed_replays() -> None:
    """Recorded by the workflow on main before compaction: two tool turns, then an answer."""
    history = WorkflowHistory.from_json(
        "thread-recorded",
        (_HISTORIES / "tool_turns_before_compaction.json").read_text(),
    )
    await Replayer(workflows=[AgentThreadWorkflow]).replay_workflow(history)
