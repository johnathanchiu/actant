"""Client-side commands for the Temporal runtime.

One workflow per ``(agent_id, thread_id)``. ``send_message`` issues a
``signal_with_start`` so the workflow is created on first contact and
signalled on every subsequent message — Temporal owns durable inbox
delivery, ordering, and single-writer semantics.

Deferred tool resolution does NOT use a workflow signal. Instead, the
workflow fires an ``await_external_resolution`` activity for any tool
that returned WAIT from its admission decision; that activity stamps
``(workflow_id, activity_id)`` onto the tool_call record and parks via
``activity.raise_complete_async``. ``resolve_deferred_tool_call`` looks up that
handle from the record and delivers the result via
``client.complete_activity_by_id`` — the workflow's ``await`` on the
activity handle unblocks naturally with the result.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, cast

import temporalio.client
import temporalio.exceptions  # re-exported for callers that catch typed errors
import temporalio.service

from actant.runtime.exceptions import ToolResolutionStaleError

from actant.agents import AgentDefinition
from actant.runtime.temporal.activities import (
    HookFactory,
    ListenerFactory,
    MessagePreprocessor,
)
from actant.runtime.temporal.types import (
    ExecuteOutcome,
    ExecuteStatus,
    InboundMessage,
    SignalName,
    TemporalRuntimeConfig,
    ThreadInput,
    ThreadStateView,
)
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from actant.runtime.interfaces.stores import RuntimeStores
from actant.tools.admission import ToolResolution, ToolResolve
from actant.tools.base import ToolResult
from actant.tools.calls import ToolCallRecord, ToolCallStatus

# admit_tool fires on_tool_waiting (which emits the FE-visible
# ``deferred`` SSE event) BEFORE the workflow dispatches
# await_external_resolution, which is what stamps the temporal handle.
# A scripted or fast-clicking caller can race that gap and call
# resolve_deferred_tool_call while temporal_workflow_id is still null. We poll briefly
# for the handle to land before failing — the handle write happens within
# tens of ms in practice, so a short bounded wait avoids the surprising
# 500 without papering over a real misconfiguration.
_HANDLE_WAIT_BACKOFF_S: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)


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
        # Stores + agents are used here for deferred resolution so the
        # client can look up the tool's on_resolve transform and persist
        # the resolved result before completing the activity. The hook
        # factories are unused on the client side; they live on the
        # worker process via TemporalRuntimeWorker. AgentRuntime wires
        # the same dependencies into both ends without branching.
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
            external_resolution_timeout_seconds=(
                self.config.external_resolution_timeout_seconds
            ),
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

    async def resolve_deferred_tool_call(
        self,
        agent_id: str,
        thread_id: str,
        tool_call_id: str,
        *,
        approved: bool | None = None,
        answer: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Deliver an external resolution for a deferred tool call.

        The deferred tool's ``await_external_resolution`` activity is
        currently parked in async-completion state. This method:

        1. Looks up the tool from the agent registry, runs ``on_resolve``
           (if defined) to transform the resolution payload into a
           ``ToolResult``. Tools without ``on_resolve`` get the default
           ``{"approved", "answer", **payload}`` shape.
        2. Persists the result onto the tool_call record (status +
           result) so ``finalize_tool_group`` later reads a terminal
           record with a real result.
        3. Completes the parked activity via ``complete_activity_by_id``
           with an ``ExecuteOutcome``. The workflow's ``await`` unblocks.

        Used by external integrations (HTTP APIs, approval UIs) when an
        out-of-band resolution arrives.
        """
        del thread_id  # unused — record carries the temporal handle directly
        record = await self._await_temporal_handle(tool_call_id)

        resolution = ToolResolution(
            approved=approved, answer=answer, payload=payload or {}
        )
        result = await self._apply_on_resolve(agent_id, record, resolution)
        result.tool_call_id = tool_call_id

        status = (
            ToolCallStatus.COMPLETED if result.is_success() else ToolCallStatus.FAILED
        )
        await self.stores.tool_calls.update_status(
            tool_call_id, status, result=result.to_dict()
        )

        client = await self._get_client()
        if record.temporal_workflow_id is None or record.temporal_activity_id is None:
            raise RuntimeError(f"tool call {tool_call_id} is missing Temporal activity handle")
        handle = client.get_async_activity_handle(
            workflow_id=record.temporal_workflow_id,
            run_id=None,
            activity_id=record.temporal_activity_id,
        )
        outcome = ExecuteOutcome(
            tool_call_id=tool_call_id,
            status=(
                ExecuteStatus.COMPLETED if result.is_success() else ExecuteStatus.FAILED
            ).value,
            terminal=bool(result.metadata.get("terminal")),
        )
        try:
            await handle.complete(outcome)
        except temporalio.service.RPCError as exc:
            # NotFound = the workflow / activity is gone (Temporal volume
            # reset, workflow terminated, activity timed out). The store
            # still reports WAITING — that's stale state. Reconcile by
            # marking the tool call FAILED with a diagnostic reason, then
            # surface a typed error so callers can distinguish this from
            # a generic Temporal hiccup.
            is_not_found = (
                exc.status == temporalio.service.RPCStatusCode.NOT_FOUND
                or "cannot find pending activity" in str(exc).lower()
            )
            if not is_not_found:
                raise
            reason = f"Temporal lost the pending activity: {exc}"
            stale_result = {
                "error": "stale_activity",
                "tool_call_id": tool_call_id,
                "detail": reason,
            }
            await self.stores.tool_calls.update_status(
                tool_call_id, ToolCallStatus.FAILED, result=stale_result
            )
            raise ToolResolutionStaleError(tool_call_id, reason) from exc

    async def cancel_thread(self, agent_id: str, thread_id: str) -> None:
        client = await self._get_client()
        handle = client.get_workflow_handle(self._workflow_id(agent_id, thread_id))
        await handle.cancel()

    async def get_state(self, agent_id: str, thread_id: str) -> ThreadStateView:
        client = await self._get_client()
        handle = client.get_workflow_handle(self._workflow_id(agent_id, thread_id))
        result = await handle.query(AgentThreadWorkflow.get_state)
        return result

    # === internal ===

    async def _await_temporal_handle(self, tool_call_id: str) -> ToolCallRecord:
        """Read the tool_call record, waiting briefly for the temporal
        handle to land if the admission emitted the deferred event but
        await_external_resolution hasn't stamped the activity_id yet.

        Bounded by ``_HANDLE_WAIT_BACKOFF_S``; if the handle never
        lands (e.g. the admission was actually BLOCK / ALLOW, or the
        workflow died) we raise the same RuntimeError as before so
        callers get a clear failure rather than hanging.
        """
        record = await self.stores.tool_calls.get(tool_call_id)
        if record.temporal_workflow_id is not None and record.temporal_activity_id is not None:
            return record
        # Only wait if the admission was WAIT. ALLOW / BLOCK / COMPLETED
        # never get a handle and never will, so retrying just hides bugs.
        if record.status != ToolCallStatus.WAITING:
            raise RuntimeError(
                f"resolve_deferred_tool_call: tool_call {tool_call_id!r} has no temporal "
                "activity handle (was the admission decision actually WAIT?)"
            )
        for delay in _HANDLE_WAIT_BACKOFF_S:
            await asyncio.sleep(delay)
            record = await self.stores.tool_calls.get(tool_call_id)
            if (
                record.temporal_workflow_id is not None
                and record.temporal_activity_id is not None
            ):
                return record
            if record.status != ToolCallStatus.WAITING:
                # Status changed under us (cancelled, completed) —
                # surface the same error rather than racing a stale
                # complete_activity_by_id call.
                break
        raise RuntimeError(
            f"resolve_deferred_tool_call: tool_call {tool_call_id!r} has no temporal "
            "activity handle (was the admission decision actually WAIT?)"
        )

    async def _apply_on_resolve(
        self,
        agent_id: str,
        record: ToolCallRecord,
        resolution: ToolResolution,
    ) -> ToolResult:
        agent = self.agents.get(agent_id)
        if agent is not None:
            tool = agent.tools.get(record.name)
            if tool is not None and callable(getattr(tool, "on_resolve", None)):
                try:
                    return await cast(ToolResolve, tool).on_resolve(record, resolution)
                except Exception as exc:  # noqa: BLE001
                    return ToolResult.fail(f"on_resolve failed: {exc}")
        # Default deferred-resolve shape — record the raw resolution.
        payload_dict: dict[str, object] = {
            "approved": resolution.approved,
            "answer": resolution.answer,
        }
        if resolution.payload:
            payload_dict.update(resolution.payload)
        return ToolResult.ok(payload_dict)

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
