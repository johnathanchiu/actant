"""In-memory stores for tests and local demos.

These back the projection-only ``RuntimeStores`` surface used by the
Temporal runtime: threads, runs, messages, and tool calls.
Coordination stores (events, turn_jobs, tool_jobs, mailbox) are gone —
Temporal owns durable inbox delivery and work scheduling.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field, replace

from actant.agents import Agent
from actant.core import JSONObject, new_id
from actant.llm.messages import Message
from actant.runtime.types.threads import (
    AgentRun,
    AgentThread,
    MessageRecord,
    RunStatus,
)
from actant.tools.calls import ToolCallRecord, ToolCallStatus


class InMemoryAgentStore:
    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    async def save(self, agent: Agent) -> None:
        self._agents[agent.id] = agent

    async def get(self, agent_id: str) -> Agent:
        return self._agents[agent_id]


class InMemoryThreadStore:
    """In-memory ``ThreadStore`` that snapshots on read/write so callers
    can't mutate the store's record by mutating a returned thread.
    """

    def __init__(self) -> None:
        self._threads: dict[tuple[str, str], AgentThread] = {}

    async def get_or_create(self, agent_id: str, thread_id: str) -> AgentThread:
        key = (agent_id, thread_id)
        thread = self._threads.get(key)
        if thread is None:
            thread = AgentThread(id=thread_id, agent_id=agent_id)
            self._threads[key] = thread
        return replace(thread)

    async def get(self, agent_id: str, thread_id: str) -> AgentThread:
        return replace(self._threads[(agent_id, thread_id)])

    async def update(self, thread: AgentThread) -> None:
        self._threads[(thread.agent_id, thread.id)] = replace(thread)

    async def list_for_agent(self, agent_id: str) -> list[AgentThread]:
        return [replace(t) for (a, _id), t in self._threads.items() if a == agent_id]


class InMemoryRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, AgentRun] = {}

    async def create(
        self,
        agent_id: str,
        thread_id: str,
        *,
        run_id: str,
        max_turns: int,
    ) -> AgentRun:
        run = AgentRun(
            id=run_id,
            agent_id=agent_id,
            thread_id=thread_id,
            max_turns=max(1, max_turns),
        )
        self._runs[run_id] = run
        return run

    async def get(self, run_id: str) -> AgentRun:
        return self._runs[run_id]

    async def update(self, run: AgentRun) -> None:
        self._runs[run.id] = run

    async def finish(self, run_id: str, status: RunStatus) -> None:
        # Idempotent: missing run = nothing to finalize. See
        # SQLAlchemyRunStore.finish for rationale.
        run = self._runs.get(run_id)
        if run is None:
            return
        run.status = status
        self._runs[run_id] = run


class InMemoryToolCallStore:
    def __init__(self) -> None:
        self._records: dict[str, ToolCallRecord] = {}

    async def save(self, tc: ToolCallRecord) -> None:
        self._records[tc.id] = tc

    async def update_status(
        self,
        tc_id: str,
        status: ToolCallStatus,
        *,
        result: object = None,
        prompt: str | None = None,
        wait_request: JSONObject | None = None,
    ) -> None:
        tc = self._records[tc_id]
        tc.status = status
        if result is not None:
            tc.result = result
        if prompt is not None:
            tc.prompt = prompt
        if wait_request is not None:
            tc.wait_request = wait_request

    async def finish_waiting(
        self,
        tc_id: str,
        status: ToolCallStatus,
        *,
        result: object,
    ) -> bool:
        tc = self._records[tc_id]
        if tc.status is not ToolCallStatus.WAITING:
            return False
        tc.status = status
        tc.result = result
        return True

    async def get(self, tc_id: str) -> ToolCallRecord:
        return self._records[tc_id]

    async def get_group(self, group_id: str) -> list[ToolCallRecord]:
        return [tc for tc in self._records.values() if tc.group_id == group_id]

    async def get_by_run(self, run_id: str) -> list[ToolCallRecord]:
        return [tc for tc in self._records.values() if tc.run_id == run_id]

    async def get_by_thread_and_turn(self, thread_id: str, turn_id: str) -> list[ToolCallRecord]:
        return [
            tc
            for tc in self._records.values()
            if tc.thread_id == thread_id and tc.turn_id == turn_id
        ]

    async def get_open_for_thread(self, agent_id: str, thread_id: str) -> list[ToolCallRecord]:
        open_states = {
            ToolCallStatus.REQUESTED,
            ToolCallStatus.RUNNING,
            ToolCallStatus.WAITING,
        }
        return [
            tc
            for tc in self._records.values()
            if tc.agent_id == agent_id and tc.thread_id == thread_id and tc.status in open_states
        ]


class InMemoryMessageStore:
    def __init__(self) -> None:
        self._messages: dict[tuple[str, str], list[Message]] = {}
        self._counter = 0
        self._tool_call_store: "InMemoryToolCallStore | None" = None

    async def append_user(
        self,
        agent_id: str,
        thread_id: str,
        content: str | list[dict[str, object]],
    ) -> MessageRecord:
        return await self._append(agent_id, thread_id, Message(role="user", content=content))

    async def append_assistant(
        self,
        agent_id: str,
        thread_id: str,
        turn_id: str,
        message: Message,
    ) -> MessageRecord:
        del turn_id
        return await self._append(agent_id, thread_id, message)

    async def append_assistant_with_tool_calls(
        self,
        agent_id: str,
        thread_id: str,
        turn_id: str,
        message: Message,
        tool_calls: "Sequence[ToolCallRecord]",
    ) -> MessageRecord:
        record = await self.append_assistant(agent_id, thread_id, turn_id, message)
        if self._tool_call_store is not None:
            for tc in tool_calls:
                await self._tool_call_store.save(tc)
        return record

    async def append_tool_result(
        self,
        agent_id: str,
        thread_id: str,
        turn_id: str,
        tool_call_id: str,
        name: str,
        result: object,
    ) -> MessageRecord:
        del turn_id
        existing = next(
            (
                m
                for m in self._messages.get((agent_id, thread_id), [])
                if m.role == "tool" and m.tool_call_id == tool_call_id
            ),
            None,
        )
        if existing is not None:
            return MessageRecord(new_id("msg"), agent_id, thread_id, existing)
        content: str | list[dict[str, object]]
        if isinstance(result, dict):
            blocks = result.get("content_blocks")
            if isinstance(blocks, list):
                normalized = [b for b in blocks if isinstance(b, dict)]
                content = normalized if normalized else _json_text(result)
            else:
                content = _json_text(result)
        else:
            content = _json_text(result)
        return await self._append(
            agent_id,
            thread_id,
            Message(
                role="tool",
                content=content,
                tool_call_id=tool_call_id,
                name=name,
            ),
        )

    async def list_for_thread(self, agent_id: str, thread_id: str) -> list[Message]:
        return list(self._messages.get((agent_id, thread_id), []))

    async def _append(self, agent_id: str, thread_id: str, message: Message) -> MessageRecord:
        self._messages.setdefault((agent_id, thread_id), []).append(message)
        self._counter += 1
        return MessageRecord(
            id=f"msg_{self._counter}",
            agent_id=agent_id,
            thread_id=thread_id,
            message=message,
        )


class InMemoryEventPublisher:
    def __init__(self) -> None:
        self.events: dict[str, list[JSONObject]] = {}
        self._subscribers: dict[str, list[asyncio.Queue[JSONObject]]] = {}

    async def publish(self, channel: str, event: JSONObject) -> None:
        self.events.setdefault(channel, []).append(event)
        for queue in self._subscribers.get(channel, []):
            await queue.put(event)

    async def subscribe(self, channel: str) -> AsyncIterator[JSONObject]:
        queue: asyncio.Queue[JSONObject] = asyncio.Queue()
        self._subscribers.setdefault(channel, []).append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers[channel].remove(queue)


@dataclass
class InMemoryRuntimeStores:
    agents: InMemoryAgentStore = field(default_factory=InMemoryAgentStore)
    threads: InMemoryThreadStore = field(default_factory=InMemoryThreadStore)
    runs: InMemoryRunStore = field(default_factory=InMemoryRunStore)
    messages: InMemoryMessageStore = field(default_factory=InMemoryMessageStore)
    tool_calls: InMemoryToolCallStore = field(default_factory=InMemoryToolCallStore)
    publisher: InMemoryEventPublisher = field(default_factory=InMemoryEventPublisher)

    def __post_init__(self) -> None:
        # Lets the message store fan tool-call writes into the
        # tool-call store inside ``append_assistant_with_tool_calls``.
        self.messages._tool_call_store = self.tool_calls


def _json_text(value: object) -> str:
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return json.dumps({"result": str(value)})
