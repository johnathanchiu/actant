"""Model-context compaction: one summary call near the real limits, and nothing else.

The rules these hold: no stored message is truncated, rewritten or deleted; the only
reducer is a summary; it fires only when the next request would cross the token fraction
or the image limit; the model then sees system + summary + kept messages + the rows after
the compaction row; and a history recorded without compaction replays unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence
from pathlib import Path

import pytest
from temporalio.client import WorkflowFailureError, WorkflowHistory
from temporalio.exceptions import ActivityError, ApplicationError
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
    compaction_request,
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
    UNREGISTERED_SUMMARIZER,
)
from actant.runtime.types.threads import RunStatus
from actant.runtime.temporal.workflow import AgentThreadWorkflow, ContextSummaryWorkflow
from actant.tools import RecallImageTool, ToolRegistry, tool
from actant.tools.base import CallContext
from runtime_fixtures import static_agents

_HISTORIES = Path(__file__).parent / "histories"
_AGENT = "compacting"
_THREAD = "t1"
_PERSONA = "You are careful."
_IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}
_WINDOW = CompactionConfig()


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
        context_window_tokens: int | None = 10_000,
        max_images_per_request: int | None = None,
        llm: FakeLLM | None = None,
        summarizers: dict[str, FakeLLM] | None = None,
    ) -> None:
        self.llm = llm or FakeLLM(
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
                summarizers=summarizers,
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
    s = _Setup(
        [_says("two seen", input_tokens=10), _says("SUMMARY"), _says("three")],
        context_window_tokens=1_000_000,
        max_images_per_request=2,
    )
    limits = CompactionConfig()
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


async def test_a_summary_without_images_is_sent_their_labels_alone() -> None:
    s = _Setup(
        [_says("two seen", input_tokens=10), _says("SUMMARY"), _says("three")],
        context_window_tokens=1_000_000,
        max_images_per_request=2,
    )
    text_only = CompactionConfig(images=False)
    await s.run([{"type": "text", "text": "look"}, _IMAGE, _IMAGE], text_only)
    await s.run([{"type": "text", "text": "and this"}, _IMAGE], text_only)

    assert len(await s.compactions()) == 1
    _, request, _ = s.llm.calls[1]
    assert count_images(request) == 0
    content = request[0].content
    assert isinstance(content, list)
    assert [b.text for b in content if isinstance(b, TextBlock)] == [
        "look",
        "[image]",
        "[image]",
    ]
    # the model's own context keeps its images: only the summary call goes without
    stored = await s.stored()
    assert count_images(stored) == 3


def test_a_stored_image_without_images_keeps_its_id_label() -> None:
    asset = AssetBlock(storage_key="s3://bucket/a.png", mime="image/png")
    [message, _] = compaction_request(
        [Message(role="user", content=[TextBlock(text="see"), asset])], images=False
    )
    assert message.content == [
        TextBlock(text="see"),
        TextBlock(text="[image id=s3://bucket/a.png]"),
    ]


async def test_the_provider_declares_the_limits_and_none_is_not_checked() -> None:
    s = _Setup([_call("a", input_tokens=9_500), _says("SUMMARY"), _says("done")])
    await s.run("first", CompactionConfig())
    assert len(await s.compactions()) == 1

    unlimited = _Setup([_call("a", input_tokens=9_500), _says("done")], context_window_tokens=None)
    await unlimited.run("first", CompactionConfig())
    assert await unlimited.compactions() == []


async def test_nothing_fires_below_both_limits_or_without_the_setting() -> None:
    s = _Setup(
        [_call("a", input_tokens=8_000), _says("done", input_tokens=8_500)],
        max_images_per_request=3,
    )
    limits = CompactionConfig()
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
    keep = CompactionConfig(keep=["brief", "tool:note"])
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
    keep = CompactionConfig(keep=["brief"])
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
        super().__init__(replies, context_window_tokens=10_000)
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


# === written ahead, in the background ===

_AHEAD = CompactionConfig(background=0.5)


class _AheadLLM(FakeLLM):
    """Turns answer from the queue; a summary call (no tools) answers ``SUMMARY``
    after ``summary_s``, beside the turns."""

    def __init__(self, replies: list[FakeResponse], *, summary_s: float = 0.0) -> None:
        super().__init__(replies, context_window_tokens=10_000)
        self.summary_s = summary_s
        self.summaries: list[list[Message]] = []
        self.turns: list[list[Message]] = []
        self.release = asyncio.Event()
        self.release.set()
        # Hold summaries until this many turns have started, then let one finish.
        self.release_at: int | None = None
        self.fail = False

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
            self.summaries.append(list(messages))
            await self.release.wait()
            await asyncio.sleep(self.summary_s)
            if self.fail:
                raise RuntimeError("summarizer down")
            return Message(role="assistant", content="SUMMARY")
        self.turns.append(list(messages))
        if len(self.turns) == self.release_at:
            self.release.set()
            await asyncio.sleep(0.05)
        return await super().complete(
            system,
            messages,
            tools,
            listener,
            allowed_tools=allowed_tools,
            max_output_tokens=max_output_tokens,
        )


def _ahead(replies: list[FakeResponse], *, summary_s: float = 0.0) -> _Setup:
    return _Setup([], llm=_AheadLLM(replies, summary_s=summary_s))


def test_background_sits_below_the_threshold() -> None:
    for bad in (0.0, 0.9, 0.95):
        try:
            CompactionConfig(background=bad)
        except ValueError:
            continue
        raise AssertionError(f"background={bad} was accepted")


async def test_a_summary_written_ahead_never_holds_a_turn_and_lands_at_a_boundary() -> None:
    s = _ahead(
        [
            _call("a", input_tokens=6_000),  # past 50%: the next turn starts a summary
            _call("b", input_tokens=6_100),
            _call("c", input_tokens=6_200),
            _says("done", input_tokens=500),
        ]
    )
    llm = s.llm
    assert isinstance(llm, _AheadLLM)
    llm.release.clear()
    llm.release_at = 3
    assert await s.run("first", _AHEAD) is RunOutcome.COMPLETED

    # One summary, of what preceded the turn open when it started.
    [request] = llm.summaries
    assert _texts(request) == ["first", COMPACTION_PROMPT]
    # Turns 2 and 3 ran on the full context while it was written.
    assert all(_texts(turn)[0] == "first" for turn in llm.turns[:3])
    stored = await s.stored()
    [(index, block)] = await s.compactions()
    # Stored at the first boundary after it was ready; everything after "first" kept.
    assert index == 7 and block.kept == [m.id for m in stored[1:7]]
    assert llm.turns[3] == [summary_message("SUMMARY"), *stored[1:7]]
    assert block.tokens_before > 5_000


async def test_the_hard_limit_waits_for_the_summary_in_flight_instead_of_a_second() -> None:
    s = _ahead(
        [
            _call("a", input_tokens=6_000),
            _call("b", input_tokens=9_500),  # the next request crosses 90%
            _says("done", input_tokens=500),
        ],
        summary_s=0.2,
    )
    llm = s.llm
    assert isinstance(llm, _AheadLLM)
    assert await s.run("first", _AHEAD) is RunOutcome.COMPLETED

    assert len(llm.summaries) == 1 and len(await s.compactions()) == 1
    stored = await s.stored()
    _, block = (await s.compactions())[0]
    assert block.kept == [m.id for m in stored[1:5]]
    assert llm.turns[2][0] == summary_message("SUMMARY")


async def test_a_run_ends_without_waiting_and_the_summary_stores_itself() -> None:
    s = _ahead([_call("a", input_tokens=6_000), _says("done", input_tokens=6_100)], summary_s=1.0)
    started = asyncio.get_running_loop().time()
    assert await s.run("first", _AHEAD) is RunOutcome.COMPLETED

    assert asyncio.get_running_loop().time() - started < 0.5
    assert await s.compactions() == []
    await asyncio.sleep(1.2)
    stored = await s.stored()
    [(index, block)] = await s.compactions()
    assert index == 4 and block.kept == [m.id for m in stored[1:4]]


async def _summary_rows(stores: InMemoryRuntimeStores) -> list[str]:
    """The thread's message kinds once its summary is stored (or after 5 s)."""
    kinds: list[str] = []
    for _ in range(100):
        kinds = [m.kind for m in await stores.messages.list_for_thread(_AGENT, _THREAD)]
        if "compaction" in kinds:
            break
        await asyncio.sleep(0.05)
    return kinds


async def test_a_run_never_waits_on_a_summary_and_the_summary_lands_on_its_own() -> None:
    # One turn per run, its caller waiting for the thread to close before the next:
    # the summary started ahead outlives the run, which closes at once.
    llm = _AheadLLM(
        [_call("a", input_tokens=6_000), _call("b", input_tokens=6_100), _says("done")],
        summary_s=2.0,
    )
    agent = AgentDefinition(
        id=_AGENT, name="Compacting", persona=_PERSONA, llm=llm, tools=ToolRegistry([note])
    )
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({_AGENT: agent}))
    )
    task_queue = f"test-actant-{uuid.uuid4().hex[:8]}"
    workflows = [AgentThreadWorkflow, ContextSummaryWorkflow]
    histories = []
    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client, task_queue=task_queue, workflows=workflows, activities=activities.all
        ):
            config = TemporalRuntimeConfig(
                task_queue=task_queue, context_compaction=_AHEAD, max_turns_per_run=1
            )
            runtime = AgentRuntime(client=env.client, stores=stores, config=config)
            for text in ("first", "second"):
                started = asyncio.get_running_loop().time()
                workflow_id = await runtime.send_message(_AGENT, _THREAD, text)
                handle = env.client.get_workflow_handle(workflow_id)
                await asyncio.wait_for(handle.result(), timeout=20.0)
                assert asyncio.get_running_loop().time() - started < 1.5
                histories.append(await handle.fetch_history())
            # the second run found the first's summary in flight and started none
            assert len(llm.summaries) == 1
            kinds = await _summary_rows(stores)
            summary = env.client.get_workflow_handle(f"{workflow_id}-summary")
            await asyncio.wait_for(summary.result(), timeout=20.0)
            histories.append(await summary.fetch_history())

    # the thread had closed: the summary's workflow stored it
    assert kinds == ["message"] * 6 + ["compaction"]
    scheduled = [
        e.activity_task_scheduled_event_attributes.activity_type.name
        for e in histories[-1].events
        if e.HasField("activity_task_scheduled_event_attributes")
    ]
    assert scheduled == ["summarize_context", "store_summary"]
    for history in histories:
        await Replayer(workflows=workflows).replay_workflow(history)


async def test_a_summary_ready_mid_run_lands_at_the_next_turn() -> None:
    llm = _AheadLLM(
        [
            _call("a", input_tokens=6_000),  # past 50%: the next turn starts a summary
            _call("b", input_tokens=6_100),
            _call("c", input_tokens=6_200),
            _says("done", input_tokens=500),
        ]
    )
    llm.release.clear()
    llm.release_at = 3
    agent = AgentDefinition(
        id=_AGENT, name="Compacting", persona=_PERSONA, llm=llm, tools=ToolRegistry([note])
    )
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({_AGENT: agent}))
    )
    task_queue = f"test-actant-{uuid.uuid4().hex[:8]}"
    workflows = [AgentThreadWorkflow, ContextSummaryWorkflow]
    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client, task_queue=task_queue, workflows=workflows, activities=activities.all
        ):
            config = TemporalRuntimeConfig(task_queue=task_queue, context_compaction=_AHEAD)
            runtime = AgentRuntime(client=env.client, stores=stores, config=config)
            workflow_id = await runtime.send_message(_AGENT, _THREAD, "first")
            handle = env.client.get_workflow_handle(workflow_id)
            await asyncio.wait_for(handle.result(), timeout=20.0)
            history = await handle.fetch_history()

    [request] = llm.summaries
    assert _texts(request) == ["first", COMPACTION_PROMPT]
    stored = await stores.messages.list_for_thread(_AGENT, _THREAD)
    [index] = [i for i, m in enumerate(stored) if m.kind == "compaction"]
    # sent to the thread, which stored it at a boundary: the last turn saw it
    assert index < len(stored) - 1
    assert llm.turns[-1][0] == summary_message("SUMMARY")
    await Replayer(workflows=workflows).replay_workflow(history)


async def test_a_failed_background_summary_is_logged_and_changes_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    s = _ahead([_call("a", input_tokens=6_000), _says("done", input_tokens=6_100)])
    llm = s.llm
    assert isinstance(llm, _AheadLLM)
    llm.fail = True
    with caplog.at_level(logging.WARNING):
        assert await s.run("first", _AHEAD) is RunOutcome.COMPLETED
        await asyncio.sleep(0.1)  # the run did not wait on it

    assert await s.compactions() == []
    [record] = [r for r in caplog.records if "background_failed" in r.getMessage()]
    assert f"thread={_THREAD}" in record.getMessage()
    assert "summarizer down" in record.getMessage()


# === on an app's own prompt ===


async def test_an_apps_prompt_replaces_the_generic_one() -> None:
    s = _Setup([_call("a", input_tokens=9_500), _says("SUMMARY"), _says("done")])
    await s.run("first", CompactionConfig(prompt="Keep the plan."))

    _, request, _ = s.llm.calls[1]
    assert _texts(request) == ["first", "Keep the plan."]


async def test_an_apps_prompt_is_kept_when_the_capped_summary_is_written_again() -> None:
    fast = FakeLLM([FakeResponse(text="half a summ", output_tokens=1_000)])
    s = _Setup(
        [_call("a", input_tokens=9_500), _says("WHOLE SUMMARY"), _says("done")],
        summarizers={"fast": fast},
    )
    config = CompactionConfig(summarizer="fast", summary_tokens=1_000, prompt="Keep the plan.")
    await s.run("first", config)

    [(_, capped, _)] = fast.calls
    assert str(capped[-1].content).startswith("Keep the plan.\n\nFit the summary")
    _, whole, _ = s.llm.calls[1]
    assert _texts(whole) == ["first", "Keep the plan."]


def test_an_empty_prompt_is_refused() -> None:
    with pytest.raises(ValueError, match="prompt"):
        CompactionConfig(prompt="  ")


# === on a summarizer of its own ===


async def test_a_named_summarizer_writes_the_summary_capped_and_told_its_length() -> None:
    fast = FakeLLM([_says("FAST SUMMARY", input_tokens=9_000)], context_window_tokens=400_000)
    s = _Setup([_call("a", input_tokens=9_500), _says("done")], summarizers={"fast": fast})
    config = CompactionConfig(summarizer="fast", summary_tokens=2_000)
    assert await s.run("first", config) is RunOutcome.COMPLETED

    [(_, block)] = await s.compactions()
    assert block.summary == "FAST SUMMARY"
    assert len(s.llm.calls) == 2  # the two turns; the summary was not the agent's
    [(system, request, tools)] = fast.calls
    assert (system, tools) == (_PERSONA, [])
    assert fast.max_output_tokens == [2_000]
    assert "about 1500 words" in str(request[-1].content)


async def test_a_request_the_summarizer_cannot_take_is_summarized_on_the_agents_model() -> None:
    small = FakeLLM([], context_window_tokens=1_000)  # under the 2,000-token cap alone
    s = _Setup(
        [_call("a", input_tokens=9_500), _says("SUMMARY"), _says("done")],
        summarizers={"small": small},
    )
    await s.run("first", CompactionConfig(summarizer="small", summary_tokens=2_000))

    assert small.calls == []
    [(_, block)] = await s.compactions()
    assert block.summary == "SUMMARY"


async def test_a_summary_cut_short_at_the_cap_is_written_again_whole(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fast = FakeLLM([FakeResponse(text="half a summ", output_tokens=1_000)])
    s = _Setup(
        [_call("a", input_tokens=9_500), _says("WHOLE SUMMARY"), _says("done")],
        summarizers={"fast": fast},
    )
    with caplog.at_level(logging.WARNING):
        await s.run("first", CompactionConfig(summarizer="fast", summary_tokens=1_000))

    [(_, block)] = await s.compactions()
    assert block.summary == "WHOLE SUMMARY"
    assert s.llm.max_output_tokens[1] is not None and s.llm.max_output_tokens[1] > 1_000
    assert any("summary_retried" in r.getMessage() for r in caplog.records)


async def test_an_unregistered_summarizer_fails_before_a_run_opens() -> None:
    s = _Setup([_call("a", input_tokens=9_500)])
    with pytest.raises(ApplicationError, match="no summarizer named 'nope'") as raised:
        await s.run("first", CompactionConfig(summarizer="nope"))
    assert raised.value.type == UNREGISTERED_SUMMARIZER and raised.value.non_retryable
    assert s.llm.calls == [] and await s.stores.runs.list_for_thread(_AGENT, _THREAD) == []


async def _in_a_workflow(
    llm: FakeLLM,
    compaction: CompactionConfig,
    summarizers: dict[str, FakeLLM] | None = None,
) -> tuple[InMemoryRuntimeStores, BaseException | None]:
    """One message through the workflow; the stores, and what the thread failed with."""

    agent = AgentDefinition(
        id=_AGENT, name="Compacting", persona=_PERSONA, llm=llm, tools=ToolRegistry([note])
    )
    stores = InMemoryRuntimeStores()
    context = ActivityContext(
        stores=stores, resolve_agent=static_agents({_AGENT: agent}), summarizers=summarizers
    )
    task_queue = f"test-actant-{uuid.uuid4().hex[:8]}"
    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=TemporalRuntimeActivities(context).all,
        ):
            config = TemporalRuntimeConfig(task_queue=task_queue, context_compaction=compaction)
            runtime = AgentRuntime(client=env.client, stores=stores, config=config)
            workflow_id = await runtime.send_message(_AGENT, _THREAD, "first")
            try:
                await asyncio.wait_for(
                    env.client.get_workflow_handle(workflow_id).result(), timeout=30.0
                )
            except WorkflowFailureError as error:
                return stores, error.cause
    return stores, None


async def test_the_workflow_fails_once_on_an_unregistered_summarizer_and_opens_no_run() -> None:
    llm = FakeLLM([_call("a", input_tokens=9_500)], context_window_tokens=10_000)
    stores, failed = await _in_a_workflow(llm, CompactionConfig(summarizer="nope"))
    assert isinstance(failed, ActivityError)
    cause = failed.cause
    assert isinstance(cause, ApplicationError) and cause.type == UNREGISTERED_SUMMARIZER
    assert "no summarizer named 'nope'" in str(cause)
    assert llm.calls == [] and await stores.runs.list_for_thread(_AGENT, _THREAD) == []


class _FlakySummaryLLM(FakeLLM):
    """Its first summary call (a call without tools) fails, as a provider outage does."""

    def __init__(self, replies: list[FakeResponse]) -> None:
        super().__init__(replies, context_window_tokens=10_000)
        self.failed = False

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
        if not tools and not self.failed:
            self.failed = True
            raise RuntimeError("provider unavailable")
        return await super().complete(
            system,
            messages,
            tools,
            listener,
            allowed_tools=allowed_tools,
            max_output_tokens=max_output_tokens,
        )


async def test_a_failed_summary_call_backs_off_and_compacts_within_the_same_run() -> None:
    llm = _FlakySummaryLLM([_call("a", input_tokens=9_500), _says("SUMMARY"), _says("done")])
    stores, failed = await _in_a_workflow(llm, _WINDOW)
    assert failed is None and llm.failed
    [run] = await stores.runs.list_for_thread(_AGENT, _THREAD)
    assert run.status is RunStatus.IDLE and run.turn_count == 2
    stored = await stores.messages.list_for_thread(_AGENT, _THREAD)
    assert [m.kind for m in stored].count("compaction") == 1


async def test_a_summarizer_that_fails_falls_back_to_the_agents_model() -> None:
    down = FakeLLM([])  # no reply queued: the call raises, as a refused capped answer does
    s = _Setup(
        [_call("a", input_tokens=9_500), _says("SUMMARY"), _says("done")],
        summarizers={"down": down},
    )
    assert await s.run("first", CompactionConfig(summarizer="down")) is RunOutcome.COMPLETED
    [(_, block)] = await s.compactions()
    assert block.summary == "SUMMARY" and len(down.calls) == 1
