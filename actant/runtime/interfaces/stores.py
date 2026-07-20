"""Runtime store interfaces.

The Temporal runtime owns durable execution (inbox, leases, scheduling).
Stores hold projections — threads, runs, messages, and tool calls —
written from inside activities and read by application code.
"""

from __future__ import annotations

from typing import Protocol

from actant.agents import Agent
from actant.core import JSONObject
from actant.llm.messages import Message
from actant.runtime.events.publisher import EventPublisher
from actant.runtime.types.threads import (
    AgentRun,
    AgentThread,
    MessageRecord,
    RunStatus,
)
from actant.tools.calls import ToolCallRecord, ToolCallStatus
from collections.abc import Sequence


class AgentStore(Protocol):
    async def save(self, agent: Agent) -> None: ...

    async def get(self, agent_id: str) -> Agent: ...


class ThreadStore(Protocol):
    async def get_or_create(self, agent_id: str, thread_id: str) -> AgentThread: ...

    async def get(self, agent_id: str, thread_id: str) -> AgentThread: ...

    async def update(self, thread: AgentThread) -> None: ...

    async def list_for_agent(self, agent_id: str) -> list[AgentThread]: ...


class RunStore(Protocol):
    async def create(
        self,
        agent_id: str,
        thread_id: str,
        *,
        run_id: str,
        max_turns: int,
    ) -> AgentRun: ...

    async def get(self, run_id: str) -> AgentRun: ...

    async def update(self, run: AgentRun) -> None: ...

    async def finish(self, run_id: str, status: RunStatus) -> None: ...


class MessageStore(Protocol):
    async def append_user(
        self,
        agent_id: str,
        thread_id: str,
        content: str | list[dict[str, object]],
    ) -> MessageRecord: ...

    async def append_assistant(
        self, agent_id: str, thread_id: str, turn_id: str, message: Message
    ) -> MessageRecord: ...

    async def append_assistant_with_tool_calls(
        self,
        agent_id: str,
        thread_id: str,
        turn_id: str,
        message: Message,
        tool_calls: Sequence[ToolCallRecord],
    ) -> MessageRecord:
        """Atomically persist an agent turn's assistant output and tool calls.

        Writes the assistant message AND its tool-call records in one
        transaction. Either both commit or neither — prevents the state
        where the message claims a tool call that has no corresponding
        ToolCallRecord (which produces a 400 from OpenAI on the next
        turn).
        """
        ...

    async def append_tool_result(
        self,
        agent_id: str,
        thread_id: str,
        turn_id: str,
        tool_call_id: str,
        name: str,
        result: object,
    ) -> MessageRecord: ...

    async def list_for_thread(self, agent_id: str, thread_id: str) -> list[Message]: ...


class ToolCallStore(Protocol):
    async def save(self, tc: ToolCallRecord) -> None: ...

    async def update_status(
        self,
        tc_id: str,
        status: ToolCallStatus,
        *,
        result: object = None,
        prompt: str | None = None,
        wait_request: JSONObject | None = None,
    ) -> None: ...

    async def set_temporal_handle(
        self,
        tc_id: str,
        *,
        workflow_id: str,
        activity_id: str,
    ) -> None:
        """Stamp the Temporal ``(workflow_id, activity_id)`` onto the
        record so external callers (the runtime client's ``resolve_deferred_tool_call``
        path) can complete this activity asynchronously via
        ``client.complete_activity_by_id``. Set by
        ``await_external_resolution`` activity for WAIT-decision tools."""
        ...

    async def get(self, tc_id: str) -> ToolCallRecord: ...

    async def get_group(self, group_id: str) -> list[ToolCallRecord]: ...

    async def get_by_run(self, run_id: str) -> list[ToolCallRecord]: ...

    async def get_by_thread_and_turn(
        self, thread_id: str, turn_id: str
    ) -> list[ToolCallRecord]: ...

    async def get_open_for_thread(
        self, agent_id: str, thread_id: str
    ) -> list[ToolCallRecord]: ...


class RuntimeStores(Protocol):
    @property
    def threads(self) -> ThreadStore: ...

    @property
    def runs(self) -> RunStore: ...

    @property
    def messages(self) -> MessageStore: ...

    @property
    def tool_calls(self) -> ToolCallStore: ...

    @property
    def publisher(self) -> EventPublisher: ...
