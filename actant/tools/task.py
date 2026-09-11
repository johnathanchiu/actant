"""Generic subagent invocation tool.

Supports two modes:

- **Synchronous** — provide an ``invoker``. ``execute()`` calls
  ``invoker.invoke(name, message, context)`` which returns a
  complete ``ToolResult``. Right for in-process registries
  (echo, deterministic transforms, fan-out aggregation that
  finishes in one call).

- **Background** — provide a ``spawner``. ``execute()`` starts the
  sub-thread and returns its id straight away. Right for delegations
  that span many turns.

  Background does not mean a different kind of tool call. Every tool
  call is blocking: the agent calls it and gets a result. This one
  returns a handle rather than an answer, which is what lets a parent
  start several subagents and supervise them instead of stopping on
  the first. Use ``check``/``message``/``stop`` (``supervise.py``) to
  work with what it returns.

  The parent does not have to poll for the ending: a finished
  sub-thread messages its parent, which wakes it whether it is parked
  or already closed.

This tool used to park the parent on a ``WAIT`` until the subagent
finished, resolved by the host through ``resolve_tool_call``. That
overloaded ``WAIT`` -- which otherwise always means "a person has to
answer" -- onto a machine finishing its work, and it meant a parent
could supervise exactly one child, badly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from actant.core import JSONObject, JSONValue
from actant.tools.admission import (
    ToolCallView,
    ToolDecision,
    TurnContextView,
)
from actant.tools.base import (
    BaseToolInvocation,
    CallContext,
    ToolInvocation,
    ToolResult,
    ToolSchema,
    make_tool_schema,
)


class SubagentInvoker(Protocol):
    """Sync mode: ``invoke`` returns a complete ``ToolResult``."""

    async def invoke(self, name: str, message: str, context: JSONObject) -> ToolResult: ...


class SubagentSpawner(Protocol):
    """Background mode: ``spawn`` starts a sub-thread and returns its id.

    The id is what the parent gets back as the tool result, and what it
    passes to ``check``/``message``/``stop``. Raising is how a spawner
    reports that it could not start -- the caller turns that into a
    failed tool result rather than a parent waiting on nothing.
    """

    async def spawn(
        self,
        *,
        name: str,
        message: str,
        context: JSONObject,
        parent_thread_id: str,
    ) -> str: ...


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

    Pass exactly one of ``invoker`` (sync) or ``spawner`` (background).

    **Parent-thread resolution (background mode):** ``parent_thread_id``
    is optional — if unset, the tool reads the thread from the
    :class:`CallContext` every ``build`` receives. This means a single
    ``TaskTool`` instance can be shared across many threads in one
    ``AgentDefinition``, instead of requiring per-thread agent construction
    just to pin a different ``parent_thread_id`` on each TaskTool.

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
        # parent_thread_id is no longer required for background mode --
        # ``can_execute`` falls back to ``call.thread_id`` when it's
        # unset. The old check is dropped intentionally.

    @property
    def background(self) -> bool:
        return self.spawner is not None

    @property
    def schema(self) -> ToolSchema:
        if self.subagent_choices:
            choice_lines = "\n".join(
                f"  - {n}: {self.subagent_descriptions.get(n, '')}" for n in self.subagent_choices
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

    async def build(self, params: JSONObject, ctx: CallContext) -> "TaskInvocation":
        """The context carries the thread doing the delegating.

        Plain arguments cannot serve background mode: the parent thread id is
        not in them, and it is the whole point of the call.
        """
        return TaskInvocation(
            params,
            invoker=self.invoker,
            spawner=self.spawner,
            parent_thread_id=self.parent_thread_id or ctx.thread_id or None,
        )

    # ``can_execute`` validates and nothing more. Starting the subagent is
    # work, and work belongs in ``execute`` -- admission runs on every call
    # including ones that get denied, so a side effect here fires for calls
    # that never execute.

    async def can_execute(
        self,
        call: ToolCallView,
        invocation: ToolInvocation | None,
        context: TurnContextView | None,
    ) -> ToolDecision:
        del invocation, context
        if not self.background:
            return ToolDecision.execute()
        args: JSONObject = call.args if isinstance(call.args, dict) else {}
        subagent = args.get("subagent")
        message = args.get("message")
        if not isinstance(subagent, str) or not subagent:
            return ToolDecision.deny(reason="`subagent` is required")
        if self.subagent_choices and subagent not in self.subagent_choices:
            valid = ", ".join(self.subagent_choices)
            return ToolDecision.deny(reason=f"Unknown subagent {subagent!r}; valid: {valid}")
        if not isinstance(message, str) or not message.strip():
            return ToolDecision.deny(reason="`message` is required")
        if self._parent_thread_id(call) is None:
            return ToolDecision.deny(
                reason=(
                    "TaskTool has no parent_thread_id: neither set at "
                    "construction nor present on the tool call."
                )
            )
        return ToolDecision.execute()

    def _parent_thread_id(self, call: ToolCallView) -> str | None:
        """Prefer the construction-time id, fall back to the call's.

        Apps that build one TaskTool per thread set it at construction;
        apps that share one across threads rely on the id the runtime
        stamps on every ToolCallView.

        Empty is missing, not a thread: a spawn against "" would parent the
        child to nothing, and the notification on completion would have
        nowhere to go.
        """
        return self.parent_thread_id or getattr(call, "thread_id", None) or None


class TaskInvocation(BaseToolInvocation[JSONObject, object]):
    """Delegate, either inline or in the background.

    Background mode returns the sub-thread's id rather than its answer. That
    is deliberate: the parent gets a handle it can supervise, instead of
    stopping until the subagent is done. The work itself arrives later --
    the sub-thread messages its parent when it finishes.
    """

    def __init__(
        self,
        params: JSONObject,
        *,
        invoker: SubagentInvoker | None = None,
        spawner: SubagentSpawner | None = None,
        parent_thread_id: str | None = None,
    ) -> None:
        super().__init__(params)
        self._invoker = invoker
        self._spawner = spawner
        self._parent_thread_id = parent_thread_id

    def get_description(self) -> str:
        subagent = self.params.get("subagent")
        return f"Delegate task to {subagent}" if isinstance(subagent, str) else "Delegate task"

    async def execute(self) -> ToolResult:
        subagent = self.params.get("subagent")
        message = self.params.get("message")
        if not isinstance(subagent, str) or not subagent:
            return ToolResult.fail("subagent is required")
        if not isinstance(message, str) or not message:
            return ToolResult.fail("message is required")
        context = _context_payload(self.params.get("context"))

        if self._spawner is not None:
            return await self._start(subagent, message, context)
        if self._invoker is None:
            return ToolResult.fail("TaskTool has neither an invoker nor a spawner.")
        return await self._invoker.invoke(subagent, message, context)

    async def _start(self, subagent: str, message: str, context: JSONObject) -> ToolResult:
        spawner = self._spawner
        assert spawner is not None
        if self._parent_thread_id is None:
            return ToolResult.fail("TaskTool has no parent_thread_id.")
        try:
            thread_id = await spawner.spawn(
                name=subagent,
                message=message,
                context=context,
                parent_thread_id=self._parent_thread_id,
            )
        except Exception as exc:  # noqa: BLE001 -- a failed spawn is a failed tool
            return ToolResult.fail(f"Subagent spawn failed: {exc}")

        # ``sub_thread_id`` is in the output, not in metadata, even though
        # it is bookkeeping rather than something the model needs: the tool
        # result event carries only ``output`` and ``error``
        # (``PublishingThreadHooks.on_tool_result``), so metadata never
        # reaches a viewer. Putting it there hides a running subagent from
        # the UI until someone reloads the page.
        return ToolResult.ok(
            {
                "subagent": subagent,
                "thread_id": thread_id,
                "sub_thread_id": thread_id,
                "status": "running",
            }
        )


def _context_payload(value: JSONValue | None) -> JSONObject:
    if isinstance(value, dict):
        return value
    return {}
