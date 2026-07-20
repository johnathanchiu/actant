"""Temporal activities for the AgentThreadWorkflow.

All canonical store writes happen in these activity bodies. Hooks are
fired only after the corresponding canonical write commits, so a crash
between the two never publishes events for state that didn't persist.

Each tool-related activity has a single, focused job:

- ``admit_tool``: classify (allow/block/wait). Persists the decision.
  Returns ``AdmitOutcome``. Infallible — any unexpected exception is
  mapped to ``decision=block, reason="admission_error: ..."``.

- ``execute_tool``: run an ALLOW-decision tool inline. Persists the
  result. Returns ``ExecuteOutcome``. Infallible — any unexpected
  exception is mapped to ``status=failed, result={"error": "..."}``.

- ``await_external_resolution``: handles WAIT-decision tools via
  Temporal's async-activity-completion. Stamps ``(workflow_id,
  activity_id)`` onto the tool_call record so an external caller can
  later complete this activity via ``client.complete_activity_by_id``;
  then returns by raising ``raise_complete_async``. The activity stays
  "running" (consuming zero compute) until the external completion
  lands with the result.

The activity boundary is where exceptions get converted to structured
outcomes — workflow code never has to wrap activity calls in
try/except. This keeps the workflow deterministic and small.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import cast

from temporalio import activity
from temporalio.exceptions import ApplicationError

from actant.agents import AgentDefinition
from actant.core import JSONObject, new_id
from actant.llm.errors import StreamCancelled
from actant.llm.messages import Message  # for type hint in MessagePreprocessor
from actant.llm.messages import ToolCall as LLMToolCall
from actant.runtime.executors.temporal_types import (
    ActivityName,
    AdmitDecision,
    AdmitInput,
    AdmitOutcome,
    ApplyThreadCancellationInput,
    AwaitExternalResolutionInput,
    ExecuteInput,
    ExecuteOutcome,
    ExecuteStatus,
    FinalizeRunInput,
    RunOutcome,
    RunTurnInput,
    StartRunInput,
    ToolCallSpec,
    TurnResult,
)
from actant.runtime.hooks import AgentThreadHooks, StreamListener
from actant.runtime.interfaces.stores import RuntimeStores
from actant.runtime.types.context import TurnContext
from actant.runtime.types.threads import AgentThread, RunStatus, ThreadStatus
from actant.tools.admission import (
    ToolCallView,
    ToolCanExecute,
    ToolDecision,
    ToolDecisionKind,
)
from actant.tools.base import Tool, ToolInvocation, ToolResult
from actant.tools.calls import ToolCallRecord, ToolCallStatus

HookFactory = Callable[[AgentThread], AgentThreadHooks]
ListenerFactory = Callable[[AgentThread], StreamListener]
MessagePreprocessor = Callable[[list[Message]], Awaitable[list[Message]]]


class TemporalRuntimeActivities:
    """Worker-bound activities that compose the AgentThreadWorkflow.

    Constructed once per worker process. Activity callables are bound
    methods, so the worker registers ``activities=instance.all`` and
    each method has access to ``stores`` / ``agents`` / hook factories
    via ``self``.
    """

    def __init__(
        self,
        *,
        stores: RuntimeStores,
        agents: Mapping[str, AgentDefinition],
        hooks_factory: HookFactory | None = None,
        listener_factory: ListenerFactory | None = None,
        message_preprocessor: MessagePreprocessor | None = None,
    ) -> None:
        self.stores = stores
        self.agents = agents
        self.hooks_factory = hooks_factory
        self.listener_factory = listener_factory
        self.message_preprocessor = message_preprocessor

    @property
    def all(self) -> list[Callable[..., object]]:
        """Activity callables for ``Worker(activities=...)`` registration."""
        return [
            self.start_run,
            self.run_turn,
            self.admit_tool,
            self.execute_tool,
            self.await_external_resolution,
            self.finalize_tool_group,
            self.finalize_run,
            self.apply_thread_cancellation,
        ]

    # === start_run ===

    @activity.defn(name=ActivityName.START_RUN)
    async def start_run(self, payload: StartRunInput) -> None:
        """Create the projection row for a new run.

        Idempotent on retry: if a run row already exists for ``run_id``
        we no-op. The thread row is upserted by ``get_or_create``.
        """
        try:
            await self.stores.runs.create(
                payload.agent_id,
                payload.thread_id,
                run_id=payload.run_id,
                max_turns=payload.max_turns,
            )
        except Exception:
            # InMemory and Postgres stores raise on duplicate run_id; the
            # workflow only calls start_run once per run, but Temporal can
            # retry the activity on transient failures. Treat duplicate as
            # success so retries are safe.
            try:
                await self.stores.runs.get(payload.run_id)
            except Exception:
                raise
        thread = await self.stores.threads.get_or_create(payload.agent_id, payload.thread_id)
        thread.active_run_id = payload.run_id
        thread.status = ThreadStatus.ACTIVE
        await self.stores.threads.update(thread)

    # === run_turn ===

    @activity.defn(name=ActivityName.RUN_TURN)
    async def run_turn(self, payload: RunTurnInput) -> TurnResult:
        """Run one LLM turn.

        Persists any ``new_messages`` (only on the first turn of a run),
        builds the turn context, calls ``agent.complete``, and atomically
        commits the assistant message + its tool-call records. Hooks
        fire after each canonical write.
        """
        agent = self._require_agent(payload.agent_id)
        thread = await self.stores.threads.get_or_create(payload.agent_id, payload.thread_id)
        run = await self.stores.runs.get(payload.run_id)
        hooks = self._hooks(thread)
        listener = self._listener(thread)

        for msg in payload.new_messages:
            await self.stores.messages.append_user(
                payload.agent_id, payload.thread_id, msg.content
            )
            await hooks.on_user_message(msg.content)

        messages = await self.stores.messages.list_for_thread(
            payload.agent_id, payload.thread_id
        )
        if self.message_preprocessor is not None:
            messages = await self.message_preprocessor(messages)
        context = TurnContext(
            agent=agent,
            system_prompt=agent.persona,
            messages=messages,
            thread_id=payload.thread_id,
            turn_id=payload.turn_id,
            turn_index=payload.turn_index,
        )

        await hooks.on_turn_start(payload.turn_index, payload.turn_id)
        try:
            assistant = await agent.complete(context.messages, listener)
        except StreamCancelled as exc:
            # Cancellation aborts this turn; surface as non-retryable so
            # the workflow can decide to fail the run cleanly.
            raise ApplicationError("turn cancelled", non_retryable=True) from exc

        # Build tool-call records up front so the assistant message AND
        # its tool-call rows commit in one DB transaction. Splitting
        # them risks a state where the message claims a tool_call with
        # no matching ToolCallRecord, which 400s the next provider call.
        group_id = new_id("group")[:12]
        records: list[ToolCallRecord] = [
            ToolCallRecord(
                id=tc.id,
                group_id=group_id,
                run_id=payload.run_id,
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                turn_id=payload.turn_id,
                turn_index=payload.turn_index,
                name=tc.function.name,
                args=_parse_tool_args(tc),
            )
            for tc in (assistant.tool_calls or [])
        ]
        await self.stores.messages.append_assistant_with_tool_calls(
            payload.agent_id,
            payload.thread_id,
            payload.turn_id,
            assistant,
            records,
        )

        await hooks.on_assistant_message(assistant)
        for record in records:
            await hooks.on_tool_call(record.id, record.name, record.args)

        thread.turn_count += 1
        run.turn_count += 1
        run.status = RunStatus.ACTIVE
        await self.stores.threads.update(thread)
        await self.stores.runs.update(run)

        return TurnResult(
            turn_id=payload.turn_id,
            turn_index=payload.turn_index,
            tool_calls=[
                ToolCallSpec(
                    id=r.id,
                    group_id=r.group_id,
                    run_id=r.run_id,
                    turn_id=r.turn_id,
                    turn_index=r.turn_index,
                    name=r.name,
                )
                for r in records
            ],
        )

    # === admit_tool ===

    @activity.defn(name=ActivityName.ADMIT_TOOL)
    async def admit_tool(self, payload: AdmitInput) -> AdmitOutcome:
        """Classify one tool call: allow / block / wait.

        Persists the decision onto the tool_call record:
        - BLOCK: status=BLOCKED, result={"error": reason}
        - WAIT: status=WAITING, prompt + wait_request
        - ALLOW: status=RUNNING (status reflects "this is being executed");
          ``execute_tool`` overwrites with COMPLETED/FAILED + result.

        Fires hooks: ``on_tool_result`` for BLOCK, ``on_tool_waiting`` for
        WAIT. ALLOW fires no hook here — ``execute_tool`` fires
        ``on_tool_result`` once it has a real result.

        Infallible. Any unexpected exception (tool registry lookup
        failure, ``can_execute`` raising, hook failure) is converted to
        ``decision=BLOCK, reason="admission_error: <exc>"``. The
        transcript invariant downstream (``finalize_tool_group`` writing
        a tool_result message) is unaffected because the record always
        ends in a terminal status with a result attached.
        """
        try:
            return await self._admit_tool_inner(payload)
        except Exception as exc:  # noqa: BLE001 — boundary catch
            return await self._admit_failed(payload.tool_call_id, f"admission_error: {exc}")

    async def _admit_tool_inner(self, payload: AdmitInput) -> AdmitOutcome:
        agent = self._require_agent(payload.agent_id)
        record = await self.stores.tool_calls.get(payload.tool_call_id)
        thread = await self.stores.threads.get_or_create(payload.agent_id, payload.thread_id)
        hooks = self._hooks(thread)

        tool = agent.tools.get(record.name)
        if tool is None:
            return await self._admit_block(record, hooks, f"Tool {record.name} not found")

        try:
            invocation = await tool.build(record.args)
        except Exception as exc:  # noqa: BLE001
            return await self._admit_block(record, hooks, f"Tool build error: {exc}")

        context = TurnContext(
            agent=agent,
            system_prompt=agent.persona,
            messages=await self.stores.messages.list_for_thread(
                payload.agent_id, payload.thread_id
            ),
            thread_id=payload.thread_id,
            turn_id=record.turn_id,
            turn_index=record.turn_index,
        )
        decision = await _tool_decision(tool, record, invocation, context)

        if decision.kind == ToolDecisionKind.BLOCK:
            return await self._admit_block(record, hooks, decision.reason or "Tool call blocked")

        if decision.kind == ToolDecisionKind.WAIT:
            wait_request = decision.wait_request
            wait_request_dict = wait_request.to_dict() if wait_request is not None else None
            prompt = decision.reason or invocation.get_description()
            await self.stores.tool_calls.update_status(
                record.id,
                ToolCallStatus.WAITING,
                prompt=prompt,
                wait_request=wait_request_dict,
            )
            await hooks.on_tool_waiting(
                record.id,
                prompt,
                record.turn_id,
                wait_request=wait_request_dict,
            )
            return AdmitOutcome(
                tool_call_id=record.id,
                decision=AdmitDecision.WAIT.value,
                reason=decision.reason,
                wait_request=wait_request_dict,
            )

        # ALLOW
        await self.stores.tool_calls.update_status(record.id, ToolCallStatus.RUNNING)
        return AdmitOutcome(tool_call_id=record.id, decision=AdmitDecision.ALLOW.value)

    async def _admit_block(
        self, record: ToolCallRecord, hooks: AgentThreadHooks, reason: str
    ) -> AdmitOutcome:
        result = ToolResult.fail(reason)
        result.tool_call_id = record.id
        await self.stores.tool_calls.update_status(
            record.id, ToolCallStatus.BLOCKED, result=result.to_dict()
        )
        await hooks.on_tool_result(record.id, result, record.turn_id)
        return AdmitOutcome(
            tool_call_id=record.id, decision=AdmitDecision.BLOCK.value, reason=reason
        )

    async def _admit_failed(self, tool_call_id: str, reason: str) -> AdmitOutcome:
        """Fallback for unexpected exceptions in admit. Persists a
        BLOCKED status with the error captured so finalize_tool_group
        sees a terminal record. We don't have a thread reference at the
        outer catch, so the on_tool_result hook is best-effort: try to
        load thread and fire the hook; swallow if even that fails."""
        result = ToolResult.fail(reason)
        result.tool_call_id = tool_call_id
        try:
            await self.stores.tool_calls.update_status(
                tool_call_id, ToolCallStatus.BLOCKED, result=result.to_dict()
            )
        except Exception:  # noqa: BLE001
            # If even the persist fails, finalize_tool_group's missing-
            # result fallback (``{"error": "No result"}``) covers the
            # transcript invariant. The activity still returns a
            # structured outcome so the workflow stays clean.
            pass
        return AdmitOutcome(
            tool_call_id=tool_call_id, decision=AdmitDecision.BLOCK.value, reason=reason
        )

    # === execute_tool ===

    @activity.defn(name=ActivityName.EXECUTE_TOOL)
    async def execute_tool(self, payload: ExecuteInput) -> ExecuteOutcome:
        """Run an ALLOW-decision tool inline.

        Builds the invocation again (admit's invocation lived in admit's
        process and isn't serializable), executes it, persists the
        result. Fires ``on_tool_result``.

        Infallible. Any unexpected exception is mapped to
        ``status=FAILED`` with ``result={"error": "execute_error: ..."}``.
        """
        try:
            return await self._execute_tool_inner(payload)
        except Exception as exc:  # noqa: BLE001
            return await self._execute_failed(payload.tool_call_id, f"execute_error: {exc}")

    async def _execute_tool_inner(self, payload: ExecuteInput) -> ExecuteOutcome:
        agent = self._require_agent(payload.agent_id)
        record = await self.stores.tool_calls.get(payload.tool_call_id)
        thread = await self.stores.threads.get_or_create(payload.agent_id, payload.thread_id)
        hooks = self._hooks(thread)

        tool = agent.tools.get(record.name)
        if tool is None:
            return await self._execute_failed(record.id, f"Tool {record.name} not found")

        try:
            invocation = await tool.build(record.args)
        except Exception as exc:  # noqa: BLE001
            return await self._execute_failed(record.id, f"Tool build error: {exc}")

        try:
            result = await invocation.execute()
        except Exception as exc:  # noqa: BLE001
            result = ToolResult.fail(f"Tool execution error: {exc}")

        status = ToolCallStatus.COMPLETED if result.error is None else ToolCallStatus.FAILED
        result.tool_call_id = record.id
        await self.stores.tool_calls.update_status(record.id, status, result=result.to_dict())
        await hooks.on_tool_result(record.id, result, record.turn_id)
        execute_status = ExecuteStatus.COMPLETED if result.is_success() else ExecuteStatus.FAILED
        return ExecuteOutcome(
            tool_call_id=record.id,
            status=execute_status.value,
            terminal=bool(result.metadata.get("terminal")),
        )

    async def _execute_failed(self, tool_call_id: str, reason: str) -> ExecuteOutcome:
        result = ToolResult.fail(reason)
        result.tool_call_id = tool_call_id
        try:
            await self.stores.tool_calls.update_status(
                tool_call_id, ToolCallStatus.FAILED, result=result.to_dict()
            )
        except Exception:  # noqa: BLE001
            pass
        return ExecuteOutcome(tool_call_id=tool_call_id, status=ExecuteStatus.FAILED.value)

    # === await_external_resolution ===

    @activity.defn(name=ActivityName.AWAIT_EXTERNAL_RESOLUTION)
    async def await_external_resolution(
        self, payload: AwaitExternalResolutionInput
    ) -> ExecuteOutcome:
        """Handle a WAIT-decision tool via async activity completion.

        The activity body:
        1. Reads the tool_call record + ``activity.info()``.
        2. Stamps ``(workflow_id, activity_id)`` onto the record so the
           runtime client's ``resolve_tool`` path can find this activity.
        3. Calls ``activity.raise_complete_async()`` — Temporal SDK
           catches the sentinel and records the activity as "running,
           pending external completion." The activity DOES NOT return
           normally here; the workflow's ``await`` on this activity's
           handle remains suspended.

        When an external caller invokes
        ``client.complete_activity_by_id(workflow_id, activity_id, result)``,
        Temporal delivers the result and the workflow's await unblocks.

        Note: this function's declared return type is ``ExecuteOutcome``
        for documentation purposes — the actual result delivered to the
        workflow comes from the external completion call, not from this
        function body. The body never returns normally; it always raises
        ``CompleteAsyncError``.
        """
        info = activity.info()
        if info.workflow_id is None:
            raise ApplicationError("missing workflow_id for async activity completion")
        await self.stores.tool_calls.set_temporal_handle(
            payload.tool_call_id,
            workflow_id=info.workflow_id,
            activity_id=info.activity_id,
        )
        # Hand off to async completion. Temporal SDK catches the raised
        # sentinel and records the activity as still running.
        activity.raise_complete_async()
        # Unreachable — raise_complete_async always raises. Annotated
        # for type-checkers and to satisfy the return type.
        raise AssertionError("raise_complete_async() did not raise")

    # === finalize_tool_group ===

    @activity.defn(name=ActivityName.FINALIZE_TOOL_GROUP)
    async def finalize_tool_group(self, group_id: str) -> None:
        """Append tool_result messages for a fully-terminal tool group.

        Ordering is sorted by tool_call id so every replica/replay
        produces the same canonical transcript. Fires
        ``on_tool_resolved`` for tools that originally returned WAIT
        (signals "the deferred tool's external resolution has landed").
        ``on_tool_result`` is NOT fired here — that already fired in
        ``admit_tool`` (BLOCK path) or ``execute_tool`` (ALLOW path) or
        was the result of the external completion of
        ``await_external_resolution``.

        Defense in depth: any record without a ``result`` (which would
        violate the activity-boundary contract) gets a fallback
        ``{"error": "No result"}`` so the transcript invariant always
        holds — ``every tool_call has a matching tool_result``.
        """
        records = await self.stores.tool_calls.get_group(group_id)
        if not records:
            return
        first = records[0]
        thread = await self.stores.threads.get_or_create(first.agent_id, first.thread_id)
        hooks = self._hooks(thread)

        for record in sorted(records, key=lambda r: r.id):
            result_dict = (
                record.result if isinstance(record.result, dict) else {"error": "No result"}
            )
            await self.stores.messages.append_tool_result(
                record.agent_id,
                record.thread_id,
                record.turn_id,
                record.id,
                record.name,
                result_dict,
            )
            if record.wait_request is not None:
                await hooks.on_tool_resolved(
                    record.id, _result_from_record(record), record.turn_id
                )

    # === finalize_run ===

    @activity.defn(name=ActivityName.FINALIZE_RUN)
    async def finalize_run(self, payload: FinalizeRunInput) -> None:
        """Close out a run and clear thread.active_run_id.

        Cancellation needs special handling: any tool_calls still in
        REQUESTED / RUNNING / WAITING get a cancellation placeholder
        so the LLM transcript invariant holds (every tool_call has a
        matching tool_result). Without this, a continue-as-new or a
        post-mortem read of the transcript would 400 the next provider
        call on the dangling tool_call.
        """
        if payload.outcome == RunOutcome.CANCELLED.value:
            open_records = await self.stores.tool_calls.get_open_for_thread(
                payload.agent_id, payload.thread_id
            )
            for record in open_records:
                await self.stores.tool_calls.update_status(
                    record.id,
                    ToolCallStatus.COMPLETED,
                    result={"status": "cancelled", "reason": "session_cancelled"},
                )

        run_status = _run_status_for_outcome(payload.outcome)
        await self.stores.runs.finish(payload.run_id, run_status)
        thread = await self.stores.threads.get_or_create(payload.agent_id, payload.thread_id)
        thread.active_run_id = None
        thread.status = _thread_status_for_outcome(payload.outcome)
        await self.stores.threads.update(thread)
        hooks = self._hooks(thread)
        success = payload.outcome == RunOutcome.COMPLETED.value
        await hooks.on_complete(success=success, reason=payload.outcome, message="")

    # === apply_thread_cancellation ===

    @activity.defn(name=ActivityName.APPLY_THREAD_CANCELLATION)
    async def apply_thread_cancellation(self, payload: ApplyThreadCancellationInput) -> None:
        """Thread-level cancel cleanup. Always idempotent.

        Two repair jobs:

        1. **Projection repair**: walk open tool_call records (REQUESTED
           / RUNNING / WAITING) and stamp ``status=COMPLETED`` with the
           ``session_cancelled`` placeholder result.

        2. **Transcript repair**: walk every tool_call in this thread
           and ensure a matching ``tool_result`` message exists in the
           message log. If a tool_call doesn't have a tool_result
           (because cancel landed mid-group and ``finalize_tool_group``
           never ran), append a placeholder so the next provider call
           doesn't 400 on the orphan. The LLM transcript invariant —
           "every tool_call has a matching tool_result" — must hold
           even after cancel.

        Both writes are idempotent. If the projection record already has
        a result, ``get_open_for_thread`` won't return it. If the
        message log already has a tool_result for the call_id,
        ``append_tool_result`` is idempotent on (agent, thread,
        tool_call_id) and no-ops.

        Called in the workflow's ``CancelledError`` handler regardless
        of whether a run was active — covers the "cancel between runs"
        case where ``finalize_run`` doesn't fire AND the "cancel raced
        with new tool_call commits" case where ``finalize_tool_group``
        was skipped.
        """
        # 1. Projection repair: stamp open tool_call records terminal.
        open_records = await self.stores.tool_calls.get_open_for_thread(
            payload.agent_id, payload.thread_id
        )
        for record in open_records:
            await self.stores.tool_calls.update_status(
                record.id,
                ToolCallStatus.COMPLETED,
                result={"status": "cancelled", "reason": "session_cancelled"},
            )

        # 2. Transcript repair: append tool_result messages for any
        # tool_call without one. Walk the message log, diff
        # tool_call ids against tool_result ids, and append placeholders
        # for orphans. ``append_tool_result`` is idempotent on
        # (agent, thread, tool_call_id) so re-running this is safe.
        messages = await self.stores.messages.list_for_thread(payload.agent_id, payload.thread_id)
        tool_call_ids: set[str] = set()
        tool_result_ids: set[str] = set()
        for msg in messages:
            if msg.role == "assistant" and msg.tool_calls:
                tool_call_ids.update(tc.id for tc in msg.tool_calls)
            elif msg.role == "tool" and msg.tool_call_id is not None:
                tool_result_ids.add(msg.tool_call_id)

        orphans = tool_call_ids - tool_result_ids
        for tool_call_id in orphans:
            try:
                record = await self.stores.tool_calls.get(tool_call_id)
            except KeyError:
                continue
            result = (
                record.result
                if isinstance(record.result, dict)
                else {"status": "cancelled", "reason": "session_cancelled"}
            )
            await self.stores.messages.append_tool_result(
                record.agent_id,
                record.thread_id,
                record.turn_id,
                record.id,
                record.name,
                result,
            )

        thread = await self.stores.threads.get_or_create(payload.agent_id, payload.thread_id)
        thread.active_run_id = None
        thread.status = ThreadStatus.CANCELLED
        await self.stores.threads.update(thread)

    # === helpers ===

    def _require_agent(self, agent_id: str) -> AgentDefinition:
        agent = self.agents.get(agent_id)
        if agent is None:
            raise ApplicationError(f"Unknown agent: {agent_id}", non_retryable=True)
        return agent

    def _hooks(self, thread: AgentThread) -> AgentThreadHooks:
        if self.hooks_factory is None:
            return AgentThreadHooks()
        return self.hooks_factory(thread)

    def _listener(self, thread: AgentThread) -> StreamListener:
        if self.listener_factory is None:
            return StreamListener()
        return self.listener_factory(thread)


def _parse_tool_args(tool_call: LLMToolCall) -> JSONObject:
    try:
        parsed = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return cast(JSONObject, parsed)
    return {}


async def _tool_decision(
    tool: Tool,
    call: object,
    invocation: ToolInvocation,
    context: TurnContext,
) -> ToolDecision:
    if callable(getattr(tool, "can_execute", None)):
        return await cast(ToolCanExecute, tool).can_execute(
            cast(ToolCallView, call), invocation, context
        )
    return ToolDecision.allow()


def _result_from_record(record: ToolCallRecord) -> ToolResult:
    raw = record.result if isinstance(record.result, dict) else {}
    error = raw.get("error")
    metadata_raw = raw.get("metadata")
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    if isinstance(error, str):
        return ToolResult(tool_call_id=record.id, error=error, metadata=metadata)
    return ToolResult(tool_call_id=record.id, output=raw.get("result"), metadata=metadata)


def _run_status_for_outcome(outcome: str) -> RunStatus:
    if outcome == RunOutcome.COMPLETED.value:
        return RunStatus.IDLE  # current schema uses IDLE for "ran to completion"
    if outcome == RunOutcome.EXHAUSTED.value:
        return RunStatus.EXHAUSTED
    if outcome == RunOutcome.FAILED.value:
        return RunStatus.FAILED
    if outcome == RunOutcome.CANCELLED.value:
        return RunStatus.CANCELLED
    return RunStatus.IDLE


def _thread_status_for_outcome(outcome: str) -> ThreadStatus:
    """Thread status after a run finalizes.

    COMPLETED / EXHAUSTED leave the thread alive — the next user
    message starts a fresh run. FAILED / CANCELLED are terminal
    so callers can distinguish "between turns" from "this thread
    is done" (UI run-card uses this).
    """
    if outcome == RunOutcome.FAILED.value:
        return ThreadStatus.FAILED
    if outcome == RunOutcome.CANCELLED.value:
        return ThreadStatus.CANCELLED
    return ThreadStatus.IDLE
