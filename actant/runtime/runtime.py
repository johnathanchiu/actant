"""One Temporal runtime: commands, queries, and activity hosting."""

from __future__ import annotations

from datetime import timedelta
from uuid import UUID
import temporalio.client
import temporalio.worker
import temporalio.service

from actant.core import JSONObject
from actant.runtime.completion import RunCompletionHandler
from actant.runtime.events.publisher import EventSink, EventSource
from actant.runtime.gate import TurnGate
from actant.runtime.exceptions import (
    ThreadNotFoundError,
    ToolCallNotFoundError,
    ToolCallNotWaitingError,
)
from actant.runtime.interfaces.stores import RuntimeStores
from actant.runtime.temporal.activities import TemporalRuntimeActivities
from actant.runtime.temporal.activities.context import (
    ActivityContext,
    AgentResolver,
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
from actant.runtime.thread import ThreadHandle
from actant.runtime.types.threads import ThreadStatus
from actant.tools.calls import ToolCallStatus
from actant.sandbox.base import ArtifactSink
from actant.sandbox.registry import SandboxRegistry
from actant.assets import AssetResolver


class AgentRuntime:
    """Submit and observe threads, and optionally execute them through Temporal.

    The caller owns the injected Temporal connection and service adapters. An API
    process needs only the client and stores. Execution also requires resolve_agent.
    """

    def __init__(
        self,
        *,
        client: temporalio.client.Client,
        stores: RuntimeStores,
        config: TemporalRuntimeConfig | None = None,
        resolve_agent: AgentResolver | None = None,
        sandboxes: SandboxRegistry | None = None,
        assets: AssetResolver | None = None,
        artifact_sink: ArtifactSink | None = None,
        event_source: EventSource | None = None,
        event_sink: EventSink | None = None,
        turn_gate: TurnGate | None = None,
        run_completion_handler: RunCompletionHandler | None = None,
        message_preprocessor: MessagePreprocessor | None = None,
    ) -> None:
        self.client = client
        self.stores = stores
        self.config = config or TemporalRuntimeConfig()
        self.event_source = event_source
        self._context = ActivityContext(
            stores=stores,
            resolve_agent=resolve_agent,
            sandboxes=sandboxes,
            assets=assets,
            artifact_sink=artifact_sink,
            event_sink=event_sink,
            turn_gate=turn_gate,
            run_completion_handler=run_completion_handler,
            message_preprocessor=message_preprocessor,
        )
        self._running = False
        self._worker: temporalio.worker.Worker | None = None

    async def run_worker(self) -> None:
        """Poll until shutdown or cancellation. Injected resources remain caller-owned."""
        if self._context.resolve_agent is None:
            raise ValueError("run_worker requires resolve_agent")
        if self._running:
            raise RuntimeError("this runtime is already polling")
        self._running = True
        try:
            activities = TemporalRuntimeActivities(self._context)
            self._worker = temporalio.worker.Worker(
                self.client,
                task_queue=self.config.task_queue,
                workflows=[AgentThreadWorkflow],
                activities=activities.all,
                max_concurrent_activities=self.config.max_concurrent_activities,
                graceful_shutdown_timeout=timedelta(
                    seconds=self.config.graceful_shutdown_timeout_seconds
                ),
            )
            await self._worker.run()
        finally:
            self._worker = None
            self._running = False

    async def shutdown(self) -> None:
        """Stop polling and await activity shutdown; does not drain whole workflows.

        In-flight activities receive the configured grace period before cancellation.
        Both this call and run_worker return after worker shutdown completes.
        Calling without an active worker is harmless.
        """
        worker = self._worker
        if worker is not None:
            await worker.shutdown()

    def thread(self, agent_id: str, thread_id: str | UUID) -> ThreadHandle:
        return ThreadHandle(self, agent_id=agent_id, thread_id=str(thread_id))

    def _workflow_id(self, agent_id: str, thread_id: str) -> str:
        return f"{self.config.workflow_id_prefix}-{agent_id}-{thread_id}"

    async def send_message(
        self,
        agent_id: str,
        thread_id: str,
        content: str | list[dict[str, object]],
        *,
        parent_thread_id: str | None = None,
    ) -> str:
        """Signal the thread workflow with a new inbound message.

        ``parent_thread_id`` marks the thread as a subagent's; pass it when
        starting a sub-thread so the runtime records the link and its tools
        see ``CallContext.parent_thread_id``.

        Uses ``signal_with_start`` so the workflow is created on first
        contact and signalled on every subsequent call. Idempotent:
        re-sending starts no new execution if one is already running.
        """
        client = self.client
        wf_id = self._workflow_id(agent_id, thread_id)
        msg = InboundMessage(content=content)
        agent_max_turns = self.config.max_turns_per_run
        thread_input = ThreadInput(
            agent_id=agent_id,
            thread_id=thread_id,
            max_turns_per_run=agent_max_turns,
            external_resolution_timeout_seconds=(self.config.external_resolution_timeout_seconds),
            history_size_threshold=self.config.history_size_threshold,
            parent_thread_id=parent_thread_id,
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

    async def resolve_tool_call(
        self,
        agent_id: str,
        thread_id: str,
        tool_call_id: str,
        *,
        approved: bool | None = None,
        answer: str = "",
        payload: JSONObject | None = None,
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
        client = self.client
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
        """Stop a running thread. A finished one is already stopped.

        Threads end when their work is done, so cancelling is routinely
        aimed at a workflow that has already closed -- Temporal raises for
        that, and it is not an error worth propagating: the caller asked for
        the thread not to be running, and it is not running.
        """
        client = self.client
        handle = client.get_workflow_handle(self._workflow_id(agent_id, thread_id))
        try:
            await handle.cancel()
        except temporalio.service.RPCError as error:
            if error.status is not temporalio.service.RPCStatusCode.NOT_FOUND:
                raise
            # NOT_FOUND covers "already finished" and "never existed", and
            # only the first is fine. Ask the stores which it was, so a
            # typo'd id or a misconfigured namespace still surfaces instead
            # of every cancel silently succeeding forever.
            try:
                await self.stores.threads.get(agent_id, thread_id)
            except KeyError:
                raise ThreadNotFoundError(thread_id) from error

    async def get_state(self, agent_id: str, thread_id: str) -> ThreadStateView:
        """What the stores say about this thread.

        Read from the stores rather than queried from the workflow. A thread
        ends when its work is done, and a closed workflow is only queryable
        while Temporal retains its history -- so the query answers for a
        while and then starts failing, which is worse than not working at
        all. The stores are the durable record and answer either way.
        """
        # ``get`` rather than ``get_or_create``: asking about a thread that
        # does not exist is a caller's mistake, and creating a row to answer
        # it turns a typo into a plausible-looking idle thread.
        thread = await self.stores.threads.get(agent_id, thread_id)
        return ThreadStateView(
            agent_id=agent_id,
            thread_id=thread_id,
            # Only a running workflow knows its queue depth, and this no
            # longer asks one. Reported as unknown rather than zero, because
            # a message arriving mid-run does queue and zero would be a lie.
            inbox_size=None,
            turn_count_total=thread.turn_count,
            current_run_id=thread.active_run_id,
            cancelled=thread.status is ThreadStatus.CANCELLED,
        )
