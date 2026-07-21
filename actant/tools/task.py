"""Generic subagent invocation tool.

Supports two modes:

- **Synchronous** — provide an ``invoker``. ``execute()`` calls
  ``invoker.invoke(name, message, context)`` which returns a
  complete ``ToolResult``. Right for in-process registries
  (echo, deterministic transforms, fan-out aggregation that
  finishes in one call).

- **Deferred** — provide a ``spawner`` plus ``parent_thread_id``.
  ``can_execute`` returns ``ToolDecision.wait`` immediately and
  schedules ``spawner.spawn(...)`` via ``asyncio.create_task`` so
  the executor's ``status=WAITING`` write lands first. The
  spawner kicks off a sub-thread on the host coordinator;
  the coordinator must arrange that the sub-thread's terminal
  event calls ``resolve_tool_call(parent_tool_call_id, ...)``
  with a JSON-encoded result envelope. ``on_resolve`` then
  parses that envelope into the ``ToolResult`` the parent
  agent ultimately sees. Right for sub-thread delegations
  that span many turns and don't fit a single async call.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from actant.core import JSONObject, JSONValue
from actant.tools.admission import (
    ToolDecision,
    ToolResolution,
    ToolWaitRequest,
)
from actant.tools.base import (
    BaseToolInvocation,
    ToolResult,
    ToolSchema,
    make_tool_schema,
)


class SubagentInvoker(Protocol):
    """Sync mode: ``invoke`` returns a complete ``ToolResult``."""

    async def invoke(
        self, name: str, message: str, context: JSONObject
    ) -> ToolResult: ...


class SubagentSpawner(Protocol):
    """Deferred mode: ``spawn`` kicks off a sub-thread and returns.

    The host coordinator is responsible for ensuring that the
    sub-thread's terminal event eventually calls
    ``resolve_tool_call(parent_tool_call_id, approved, answer)``
    with ``answer`` being a JSON-encoded result envelope the
    parent's ``on_resolve`` can parse.
    """

    async def spawn(
        self,
        *,
        name: str,
        message: str,
        context: JSONObject,
        parent_thread_id: str,
        parent_tool_call_id: str,
    ) -> None: ...


@dataclass
class InMemorySubagentRegistry:
    """Trivial sync-mode registry that maps names to invokers and
    delegates ``invoke`` to the matching one."""

    invokers: dict[str, SubagentInvoker]

    async def invoke(self, name: str, message: str, context: JSONObject) -> ToolResult:
        invoker = self.invokers.get(name)
        if invoker is None:
            return ToolResult.fail(f"Subagent {name!r} not found")
        return await invoker.invoke(name, message, context)


@dataclass
class TaskTool:
    """Delegate to a registered subagent.

    Pass exactly one of ``invoker`` (sync) or ``spawner`` (deferred).

    **Parent-thread resolution (deferred mode):** ``parent_thread_id``
    is optional — if unset, the tool reads ``call.thread_id``
    from each invocation (which the runtime always stamps on
    ``ToolCallView``). This means a single ``TaskTool`` instance can be
    shared across many threads in one ``AgentDefinition``, instead of
    requiring per-thread agent construction just to pin a different
    ``parent_thread_id`` on each TaskTool.

    Setting ``parent_thread_id`` at construction time still works and
    overrides the per-call value — useful when an app builds a fresh
    AgentDefinition per thread and wants the agent's TaskTool tied to
    that thread's id verbatim.

    ``subagent_choices`` populates the schema's ``subagent`` enum so
    the LLM picks from valid names; descriptions appear in the schema
    description block.
    """

    invoker: SubagentInvoker | None = None
    spawner: SubagentSpawner | None = None
    parent_thread_id: str | None = None
    subagent_choices: Sequence[str] = ()
    subagent_descriptions: dict[str, str] = field(default_factory=dict)
    name: str = "task"

    def __post_init__(self) -> None:
        if (self.invoker is None) == (self.spawner is None):
            raise ValueError("TaskTool requires exactly one of `invoker` or `spawner`")
        # parent_thread_id is no longer required for deferred mode —
        # ``can_execute`` falls back to ``call.thread_id`` when it's
        # unset. The old check is dropped intentionally.

    @property
    def deferred(self) -> bool:
        return self.spawner is not None

    @property
    def schema(self) -> ToolSchema:
        if self.subagent_choices:
            choice_lines = "\n".join(
                f"  - {n}: {self.subagent_descriptions.get(n, '')}"
                for n in self.subagent_choices
            )
            description = (
                "Delegate a focused, well-scoped task to a specialist subagent. "
                "The subagent runs on its own and returns its outputs back as "
                "a structured result.\n"
                f"Available subagents:\n{choice_lines}"
            )
        else:
            description = "Delegate a task to a named subagent."
        subagent_param: dict[str, object] = {
            "type": "string",
            "description": "Name of the subagent.",
        }
        if self.subagent_choices:
            subagent_param["enum"] = list(self.subagent_choices)
        return make_tool_schema(
            self.name,
            description,
            parameters={
                "subagent": subagent_param,
                "message": {
                    "type": "string",
                    "description": "Self-contained instruction for the subagent.",
                },
                "context": {
                    "type": "object",
                    "description": "Optional structured inputs for the subagent.",
                },
            },
            required=["subagent", "message"],
        )

    async def build(self, params: JSONObject) -> "TaskInvocation":
        return TaskInvocation(params, self.invoker)

    # Deferred mode hooks. The runtime discovers ``can_execute`` and
    # ``on_resolve`` structurally; expose both and gate their behavior on
    # ``self.deferred`` so synchronous TaskTool instances use ALLOW.

    async def can_execute(
        self, call: Any, invocation: Any, context: Any
    ) -> ToolDecision:
        del invocation, context
        if not self.deferred:
            return ToolDecision.allow()
        args: JSONObject = call.args if isinstance(call.args, dict) else {}
        subagent = args.get("subagent")
        message = args.get("message")
        if not isinstance(subagent, str) or not subagent:
            return ToolDecision.block(reason="`subagent` is required")
        if self.subagent_choices and subagent not in self.subagent_choices:
            valid = ", ".join(self.subagent_choices)
            return ToolDecision.block(
                reason=f"Unknown subagent {subagent!r}; valid: {valid}"
            )
        if not isinstance(message, str) or not message.strip():
            return ToolDecision.block(reason="`message` is required")
        ctx: JSONObject = args.get("context") if isinstance(args.get("context"), dict) else {}  # type: ignore[assignment]

        # Spawn synchronously: the sub-thread's earliest possible
        # terminal event is at least one LLM round trip away, so the
        # executor's subsequent ``status=WAITING`` write trivially
        # lands first. Using ``asyncio.create_task`` here would be
        # unsafe — the loop only keeps weak references to tasks, so a
        # fire-and-forget task can be GC'd before the spawn coroutine
        # ever runs. ``await`` keeps the reference alive for the
        # duration of can_execute and surfaces spawn failures as
        # exceptions instead of silently dropped tasks.
        spawner = self.spawner
        assert spawner is not None  # invariant: deferred mode
        # Prefer the construction-time parent_thread_id (apps that build
        # one TaskTool per thread); fall back to the per-call thread_id
        # the runtime stamps on every ToolCallView. The fallback lets
        # apps share one TaskTool across many threads.
        parent_thread_id = self.parent_thread_id or getattr(call, "thread_id", None)
        if not parent_thread_id:
            return ToolDecision.block(
                reason=(
                    "TaskTool has no parent_thread_id: neither set at "
                    "construction nor present on the tool call."
                )
            )
        try:
            await spawner.spawn(
                name=subagent,
                message=message,
                context=ctx,
                parent_thread_id=parent_thread_id,
                parent_tool_call_id=call.id,
            )
        except Exception as exc:
            # Surface spawn failures inline rather than silently
            # parking the parent forever. Returning BLOCK rolls back
            # to a normal failed-tool flow.
            return ToolDecision.block(reason=f"Subagent spawn failed: {exc}")

        prompt = f"Delegating to {subagent}: {message[:80]}"
        return ToolDecision.wait(
            ToolWaitRequest(
                kind="subagent_task",
                prompt=prompt,
                payload={"subagent": subagent},
            )
        )

    async def on_resolve(self, call: Any, resolution: ToolResolution) -> ToolResult:
        del call
        if not self.deferred:
            # Shouldn't happen in sync mode (no WAIT was returned), but
            # be defensive: just echo the resolution back.
            return ToolResult.ok(
                {"approved": resolution.approved, "answer": resolution.answer}
            )
        if resolution.approved is False:
            return ToolResult.fail(resolution.answer or "Subagent task failed")
        if not resolution.answer:
            return ToolResult.ok({})
        try:
            payload = json.loads(resolution.answer)
        except (ValueError, TypeError) as exc:
            return ToolResult.fail(f"Subagent returned malformed result: {exc}")
        if not isinstance(payload, dict):
            return ToolResult.fail("Subagent returned non-object result")
        return ToolResult.ok(payload)


class TaskInvocation(BaseToolInvocation[JSONObject, object]):
    """Sync-mode invocation. Deferred-mode TaskTools never reach
    ``execute`` because ``can_execute`` returns WAIT first."""

    def __init__(self, params: JSONObject, invoker: SubagentInvoker | None) -> None:
        super().__init__(params)
        self._invoker = invoker

    def get_description(self) -> str:
        subagent = self.params.get("subagent")
        return (
            f"Delegate task to {subagent}"
            if isinstance(subagent, str)
            else "Delegate task"
        )

    async def execute(self) -> ToolResult:
        if self._invoker is None:
            # Deferred-mode safety: if we somehow get here, fail
            # rather than silently returning empty.
            return ToolResult.fail(
                "TaskTool is in deferred mode; execute() should not be reached."
            )
        subagent = self.params.get("subagent")
        message = self.params.get("message")
        context = self.params.get("context")
        if not isinstance(subagent, str) or not subagent:
            return ToolResult.fail("subagent is required")
        if not isinstance(message, str) or not message:
            return ToolResult.fail("message is required")
        return await self._invoker.invoke(subagent, message, _context_payload(context))


def _context_payload(value: JSONValue | None) -> JSONObject:
    if isinstance(value, dict):
        return value
    return {}
