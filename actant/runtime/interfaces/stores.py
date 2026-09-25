"""Runtime store interfaces.

The Temporal runtime owns durable execution (inbox, leases, scheduling).
Stores hold projections — threads, runs, messages, and tool calls —
written from inside activities and read by application code.
"""

from __future__ import annotations

from typing import Protocol

from actant.blocks import Block
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

    async def update(self, thread: AgentThread) -> None:
        """Write the thread's fields, except ``sandbox_id``: only ``claim_sandbox`` sets it."""
        ...

    async def claim_sandbox(
        self, agent_id: str, thread_id: str, *, expected: str | None, sandbox_id: str | None
    ) -> str | None:
        """Set the thread's sandbox id if it is still ``expected``; return the id it holds.

        A compare-and-set, so two workers that each opened a sandbox for the
        thread agree on one: the caller whose id comes back won, the other
        closes its sandbox and attaches the winner's.
        """
        ...

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

    async def finish(
        self, run_id: str, status: RunStatus, *, stop_reason: str | None = None
    ) -> None: ...

    async def list_for_thread(self, agent_id: str, thread_id: str) -> list[AgentRun]:
        """A thread's runs, newest first: how a product finds what the last run delivered."""
        ...


class MessageStore(Protocol):
    async def append_user(
        self,
        agent_id: str,
        thread_id: str,
        content: str | list[Block],
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
    ) -> bool:
        """Atomically update a nonterminal call; terminal outcomes are immutable."""
        ...

    async def finish_waiting(
        self,
        tc_id: str,
        status: ToolCallStatus,
        *,
        result: object,
    ) -> bool:
        """Atomically finish a WAITING call; return whether this caller won."""
        ...

    async def get(self, tc_id: str) -> ToolCallRecord: ...

    async def get_group(self, group_id: str) -> list[ToolCallRecord]: ...

    async def get_by_run(self, run_id: str) -> list[ToolCallRecord]: ...

    async def get_by_thread_and_turn(
        self, thread_id: str, turn_id: str
    ) -> list[ToolCallRecord]: ...

    async def get_open_for_thread(self, agent_id: str, thread_id: str) -> list[ToolCallRecord]: ...


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
