"""Activities that admit, execute, resolve, and finalize tool calls."""

from __future__ import annotations

import asyncio
import inspect
import mimetypes
from typing import cast

from temporalio import activity

from actant.agents import AgentDefinition
from actant.runtime.events import AgentThreadHooks
from actant.runtime.temporal.activities.context import ActivityContext
from actant.runtime.temporal.types import (
    ActivityName,
    AdmitDecision,
    AdmitInput,
    AdmitOutcome,
    ExecuteInput,
    ExecuteOutcome,
    ExecuteStatus,
    ResolveToolInput,
)
from actant.runtime.types.context import TurnContext
from actant.tools.admission import (
    ToolCallView,
    ToolCanExecute,
    ToolDecision,
    ToolDecisionKind,
    ToolResolution,
    ToolResolve,
)
from actant.tools.base import CallContext, Tool, ToolInvocation, ToolResult
from actant.tools.calls import ToolCallRecord, ToolCallStatus

HEARTBEAT_SECONDS = 30.0


class ToolActivities(ActivityContext):
    """Activities for one parallel tool group."""

    @activity.defn(name=ActivityName.ADMIT_TOOL)
    async def admit_tool(self, payload: AdmitInput) -> AdmitOutcome:
        """Classify one tool as allowed, blocked, or waiting."""
        try:
            return await self._admit_tool(payload)
        except Exception as exc:  # noqa: BLE001 -- activity boundary
            return await self._admit_failed(payload.tool_call_id, f"admission_error: {exc}")

    async def _admit_tool(self, payload: AdmitInput) -> AdmitOutcome:
        agent = self._require_agent(payload.agent_id)
        record = await self.stores.tool_calls.get(payload.tool_call_id)
        thread = await self.stores.threads.get_or_create(payload.agent_id, payload.thread_id)
        hooks = self._hooks(thread)
        tool = agent.tools.get(record.name)
        if tool is None:
            return await self._deny(record, hooks, f"Tool {record.name} not found")

        try:
            # The same builder execution uses, so a tool that needs its call
            # gets the same invocation in both places. Building one way here
            # and another there is how can_execute ends up inspecting an
            # object unlike the one that will actually run.
            invocation = await tool.build(
                record.args, await self._call_context(agent, tool, record)
            )
        except Exception as exc:  # noqa: BLE001
            return await self._deny(record, hooks, f"Tool build error: {exc}")

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
        if decision.kind == ToolDecisionKind.DENY:
            return await self._deny(record, hooks, decision.reason or "Tool call denied")
        if decision.kind == ToolDecisionKind.AWAIT_HUMAN:
            request = decision.wait_request
            request_data = request.to_dict() if request is not None else None
            prompt = decision.reason or invocation.get_description()
            await self.stores.tool_calls.update_status(
                record.id,
                ToolCallStatus.WAITING,
                prompt=prompt,
                wait_request=request_data,
            )
            await hooks.on_tool_waiting(
                record.id, prompt, record.turn_id, wait_request=request_data
            )
            return AdmitOutcome(
                tool_call_id=record.id,
                decision=AdmitDecision.AWAIT_HUMAN.value,
                reason=decision.reason,
                wait_request=request_data,
            )

        await self.stores.tool_calls.update_status(record.id, ToolCallStatus.RUNNING)
        return AdmitOutcome(tool_call_id=record.id, decision=AdmitDecision.EXECUTE.value)

    async def _deny(
        self, record: ToolCallRecord, hooks: AgentThreadHooks, reason: str
    ) -> AdmitOutcome:
        result = ToolResult.fail(reason)
        result.tool_call_id = record.id
        await self.stores.tool_calls.update_status(
            record.id, ToolCallStatus.BLOCKED, result=result.to_dict()
        )
        await hooks.on_tool_result(record.id, result, record.turn_id)
        return AdmitOutcome(
            tool_call_id=record.id,
            decision=AdmitDecision.DENY.value,
            reason=reason,
        )

    async def _admit_failed(self, tool_call_id: str, reason: str) -> AdmitOutcome:
        result = ToolResult.fail(reason)
        result.tool_call_id = tool_call_id
        try:
            await self.stores.tool_calls.update_status(
                tool_call_id, ToolCallStatus.BLOCKED, result=result.to_dict()
            )
        except Exception:  # noqa: BLE001 -- preserve structured boundary
            pass
        return AdmitOutcome(
            tool_call_id=tool_call_id,
            decision=AdmitDecision.DENY.value,
            reason=reason,
        )

    @activity.defn(name=ActivityName.EXECUTE_TOOL)
    async def execute_tool(self, payload: ExecuteInput) -> ExecuteOutcome:
        """Execute and persist one previously admitted tool call."""
        try:
            return await self._execute_tool(payload)
        except Exception as exc:  # noqa: BLE001 -- activity boundary
            return await self._execute_failed(payload.tool_call_id, f"execute_error: {exc}")

    async def _execute_tool(self, payload: ExecuteInput) -> ExecuteOutcome:
        agent = self._require_agent(payload.agent_id)
        record = await self.stores.tool_calls.get(payload.tool_call_id)
        tool = agent.tools.get(record.name)
        if tool is None:
            return await self._execute_failed(record.id, f"Tool {record.name} not found")
        # A tool that runs code can take minutes, and opening its sandbox can
        # too (an image builds on first use). Heartbeats keep Temporal from
        # mistaking a long healthy activity for a dead worker, and let a
        # cancellation reach it; they start before anything slow.
        beat = asyncio.create_task(_heartbeat())
        try:
            try:
                ctx = await self._call_context(agent, tool, record)
                invocation = await tool.build(record.args, ctx)
            except Exception as exc:  # noqa: BLE001
                return await self._execute_failed(record.id, f"Tool build error: {exc}")
            result = await invocation.execute()
        except Exception as exc:  # noqa: BLE001
            result = ToolResult.fail(f"Tool execution error: {exc}")
        else:
            result = await self._materialise(agent, record, ctx, result)
        finally:
            beat.cancel()

        result.tool_call_id = record.id
        status = ToolCallStatus.COMPLETED if result.error is None else ToolCallStatus.FAILED
        await self.stores.tool_calls.update_status(record.id, status, result=result.to_dict())
        thread = await self.stores.threads.get_or_create(payload.agent_id, payload.thread_id)
        await self._hooks(thread).on_tool_result(record.id, result, record.turn_id)
        return _outcome(record.id, result)

    async def _call_context(
        self, agent: AgentDefinition, tool: object, record: ToolCallRecord
    ) -> CallContext:
        """What the tool is being built for. The sandbox is opened only for tools
        that declared they need it, so a plain tool never waits on one."""
        sandbox = None
        if getattr(tool, "needs_sandbox", False):
            sandbox = await self._sandbox_for(agent, record.thread_id)
        thread = await self.stores.threads.get(record.agent_id, record.thread_id)
        return CallContext(
            agent_id=record.agent_id,
            thread_id=record.thread_id,
            run_id=record.run_id,
            tool_call_id=record.id,
            turn_id=record.turn_id,
            sandbox=sandbox,
            parent_thread_id=thread.parent_thread_id,
        )

    async def _materialise(
        self,
        agent: AgentDefinition,
        record: ToolCallRecord,
        ctx: CallContext,
        result: ToolResult,
    ) -> ToolResult:
        """Store the deliverables a terminal result names, and attach their refs.

        Any terminal result may list ``deliverables`` (workspace paths); the
        ``finish`` tool is the explicit way to do it. A missing file or sink is
        the tool's failure, and not terminal, so the model can correct it.
        """
        raw = result.metadata.get("deliverables")
        if not result.metadata.get("terminal") or not raw:
            return result
        paths = [str(item) for item in raw] if isinstance(raw, list) else []
        if not paths:
            return result
        if self.artifact_sink is None:
            failed = ToolResult.fail(
                "deliverables were listed but the worker has no artifact sink"
            )
            failed.metadata = {k: v for k, v in result.metadata.items() if k != "terminal"}
            return failed
        sandbox = ctx.sandbox
        if sandbox is None:
            sandbox = await self._sandbox_for(agent, record.thread_id)
        refs: list[dict[str, object]] = []
        for path in paths:
            try:
                data = await sandbox.read(path)
            except Exception as exc:  # noqa: BLE001 -- the path is the model's claim
                failed = ToolResult.fail(f"deliverable {path!r} could not be read: {exc}")
                failed.metadata = {k: v for k, v in result.metadata.items() if k != "terminal"}
                return failed
            mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
            name = path.rsplit("/", 1)[-1]
            ref = await self.artifact_sink.save(record.thread_id, name, data, mime)
            refs.append(ref.to_dict())
        result.metadata["artifacts"] = refs
        return result

    async def _execute_failed(self, tool_call_id: str, reason: str) -> ExecuteOutcome:
        result = ToolResult.fail(reason)
        result.tool_call_id = tool_call_id
        try:
            await self.stores.tool_calls.update_status(
                tool_call_id, ToolCallStatus.FAILED, result=result.to_dict()
            )
        except Exception:  # noqa: BLE001 -- preserve structured boundary
            pass
        return ExecuteOutcome(tool_call_id=tool_call_id, status=ExecuteStatus.FAILED.value)

    @activity.defn(name=ActivityName.RESOLVE_TOOL)
    async def resolve_tool(self, payload: ResolveToolInput) -> ExecuteOutcome:
        """Persist a resolution delivered after the workflow's durable wait."""
        record = await self.stores.tool_calls.get(payload.tool_call_id)
        if record.status in {
            ToolCallStatus.COMPLETED,
            ToolCallStatus.BLOCKED,
            ToolCallStatus.FAILED,
        }:
            return _outcome_from_record(record)
        if record.status is not ToolCallStatus.WAITING:
            return await self._execute_failed(
                record.id, f"Tool call is {record.status.value}, not waiting"
            )

        if payload.resolution is None:
            result = ToolResult.fail("Deferred tool resolution timed out")
        else:
            resolution = ToolResolution(
                approved=payload.resolution.approved,
                answer=payload.resolution.answer,
                payload=payload.resolution.payload,
            )
            result = await self._apply_resolution(payload.agent_id, record, resolution)

        result.tool_call_id = record.id
        status = ToolCallStatus.COMPLETED if result.is_success() else ToolCallStatus.FAILED
        if not await self.stores.tool_calls.finish_waiting(
            record.id, status, result=result.to_dict()
        ):
            return _outcome_from_record(await self.stores.tool_calls.get(record.id))
        thread = await self.stores.threads.get_or_create(payload.agent_id, payload.thread_id)
        await self._hooks(thread).on_tool_resolved(record.id, result, record.turn_id)
        return _outcome(record.id, result)

    async def _apply_resolution(
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
                    resolve = cast(ToolResolve, tool).on_resolve
                    # A tool that runs after approval needs the same context it
                    # would have had at execution (its sandbox, its ids).
                    if "ctx" in inspect.signature(resolve).parameters:
                        ctx = await self._call_context(agent, tool, record)
                        return await resolve(record, resolution, ctx=ctx)
                    return await resolve(record, resolution)
                except Exception as exc:  # noqa: BLE001
                    return ToolResult.fail(f"on_resolve failed: {exc}")
        output: dict[str, object] = {
            "approved": resolution.approved,
            "answer": resolution.answer,
        }
        output.update(resolution.payload)
        return ToolResult.ok(output)

    @activity.defn(name=ActivityName.FINALIZE_TOOL_GROUP)
    async def finalize_tool_group(self, group_id: str) -> None:
        """Append one canonical tool-result message for every group member."""
        records = await self.stores.tool_calls.get_group(group_id)
        for record in sorted(records, key=lambda item: item.id):
            result = record.result if isinstance(record.result, dict) else {"error": "No result"}
            await self.stores.messages.append_tool_result(
                record.agent_id,
                record.thread_id,
                record.turn_id,
                record.id,
                record.name,
                result,
            )


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
    return ToolDecision.execute()


def _result_from_record(record: ToolCallRecord) -> ToolResult:
    raw = record.result if isinstance(record.result, dict) else {}
    error = raw.get("error")
    raw_metadata = raw.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    if isinstance(error, str):
        return ToolResult(tool_call_id=record.id, error=error, metadata=metadata)
    return ToolResult(tool_call_id=record.id, output=raw.get("result"), metadata=metadata)


def _outcome(tool_call_id: str, result: ToolResult) -> ExecuteOutcome:
    return ExecuteOutcome(
        tool_call_id=tool_call_id,
        status=(ExecuteStatus.COMPLETED if result.is_success() else ExecuteStatus.FAILED).value,
        terminal=bool(result.metadata.get("terminal")),
    )


def _outcome_from_record(record: ToolCallRecord) -> ExecuteOutcome:
    return _outcome(record.id, _result_from_record(record))


async def _heartbeat() -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        try:
            activity.heartbeat()
        except RuntimeError:
            # Not inside an activity (a direct call from a test); nothing to report to.
            return
