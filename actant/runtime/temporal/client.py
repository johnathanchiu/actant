"""Client-side commands for the Temporal runtime.

One workflow per ``(agent_id, thread_id)``. ``send_message`` issues a
``signal_with_start`` so the workflow is created on first contact and
signalled on every subsequent message — Temporal owns durable inbox
delivery, ordering, and single-writer semantics.

Deferred tool resolutions are durable workflow signals. Temporal records a
resolution even if the workflow has not reached its wait condition yet, so
the client never polls or coordinates activity handles.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import temporalio.client
import temporalio.exceptions  # re-exported for callers that catch typed errors

from actant.agents import AgentDefinition
from actant.runtime.exceptions import ToolCallNotFoundError, ToolCallNotWaitingError
from actant.runtime.temporal.activities import (
    HookFactory,
    ListenerFactory,
    MessagePreprocessor,
)
from actant.runtime.temporal.types import (
    DeferredToolResolution,
    InboundMessage,
    SignalName,
    TemporalRuntimeConfig,
    ThreadInput,
    ThreadStateView,
)
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from actant.runtime.interfaces.stores import RuntimeStores
from actant.tools.calls import ToolCallStatus


class TemporalRuntimeClient:
    """Send commands to Actant thread workflows through Temporal."""

    def __init__(
        self,
        *,
        stores: RuntimeStores,
        agents: Mapping[str, AgentDefinition],
        hooks_factory: HookFactory | None = None,
        listener_factory: ListenerFactory | None = None,
        message_preprocessor: MessagePreprocessor | None = None,
        config: object | None = None,
    ) -> None:
        # Stores validate public commands; agents provide per-agent turn limits.
        # Hooks and listeners execute only in the worker process.
        self.stores = stores
        self.agents = agents
        self.hooks_factory = hooks_factory
        self.listener_factory = listener_factory
        self.message_preprocessor = message_preprocessor
        self.config = _coerce_config(config)
        self._client: temporalio.client.Client | None = None

    async def send_message(
        self,
        agent_id: str,
        thread_id: str,
        content: str | list[dict[str, object]],
    ) -> str:
        """Signal the thread workflow with a new inbound message.

        Uses ``signal_with_start`` so the workflow is created on first
        contact and signalled on every subsequent call. Idempotent:
        re-sending starts no new execution if one is already running.
        """
        client = await self._get_client()
        wf_id = self._workflow_id(agent_id, thread_id)
        msg = InboundMessage(content=content)
        agent_max_turns = self._max_turns_for_agent(agent_id)
        thread_input = ThreadInput(
            agent_id=agent_id,
            thread_id=thread_id,
            max_turns_per_run=agent_max_turns,
            external_resolution_timeout_seconds=(self.config.external_resolution_timeout_seconds),
            history_size_threshold=self.config.history_size_threshold,
        )
        await client.start_workflow(
            AgentThreadWorkflow.run,
            thread_input,
            id=wf_id,
            task_queue=self.config.task_queue,
            start_signal=SignalName.INBOUND,
            start_signal_args=[msg],
        )
        # Signals don't have ids in Temporal; return the workflow id as
        # a stable handle the caller can correlate against.
        return wf_id

    def _max_turns_for_agent(self, agent_id: str) -> int:
        agent = self.agents.get(agent_id)
        if agent is None:
            return self.config.max_turns_per_run
        return max(1, agent.max_turns_per_thread)

    async def resolve_tool_call(
        self,
        agent_id: str,
        thread_id: str,
        tool_call_id: str,
        *,
        approved: bool | None = None,
        answer: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Signal a waiting thread workflow with an external tool result."""
        try:
            record = await self.stores.tool_calls.get(tool_call_id)
        except KeyError:
            raise ToolCallNotFoundError(tool_call_id) from None
        if record.agent_id != agent_id or record.thread_id != thread_id:
            raise ToolCallNotFoundError(tool_call_id)
        if record.status in {
            ToolCallStatus.COMPLETED,
            ToolCallStatus.BLOCKED,
            ToolCallStatus.FAILED,
        }:
            return
        if record.status is not ToolCallStatus.WAITING:
            raise ToolCallNotWaitingError(tool_call_id, record.status)
        client = await self._get_client()
        handle = client.get_workflow_handle(self._workflow_id(agent_id, thread_id))
        await handle.signal(
            AgentThreadWorkflow.resolve_tool,
            DeferredToolResolution(
                tool_call_id=tool_call_id,
                approved=approved,
                answer=answer,
                payload=payload or {},
            ),
        )

    async def cancel_thread(self, agent_id: str, thread_id: str) -> None:
        client = await self._get_client()
        handle = client.get_workflow_handle(self._workflow_id(agent_id, thread_id))
        await handle.cancel()

    async def get_state(self, agent_id: str, thread_id: str) -> ThreadStateView:
        client = await self._get_client()
        handle = client.get_workflow_handle(self._workflow_id(agent_id, thread_id))
        result = await handle.query(AgentThreadWorkflow.get_state)
        return result

    async def _get_client(self) -> temporalio.client.Client:
        if self._client is None:
            self._client = await temporalio.client.Client.connect(
                self.config.address,
                namespace=self.config.namespace,
            )
        return self._client

    def _workflow_id(self, agent_id: str, thread_id: str) -> str:
        return f"{self.config.workflow_id_prefix}-{agent_id}-{thread_id}"


def _coerce_config(config: object | None) -> TemporalRuntimeConfig:
    if config is None:
        return TemporalRuntimeConfig()
    if isinstance(config, TemporalRuntimeConfig):
        return config
    raise TypeError("temporal config must be a TemporalRuntimeConfig")
