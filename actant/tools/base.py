"""Base classes for tools."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, Protocol, TypeVar

from actant.blocks import BLOCKS, Block
from actant.core import JSONObject
from actant.sandbox.access import SandboxAccess
from actant.sandbox.base import Sandbox

ToolSchema = dict[str, object]
ParamsT = TypeVar("ParamsT")
OutputT = TypeVar("OutputT")


def make_tool_schema(
    name: str,
    description: str,
    parameters: dict[str, object] | None = None,
    required: list[str] | None = None,
) -> ToolSchema:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


class MetadataKey(StrEnum):
    """``ToolResult.metadata`` keys with a meaning to the runtime or a runner."""

    #: ``True`` ends the run after this tool group.
    TERMINAL = "terminal"
    #: Workspace paths a terminal result hands to the ``ArtifactSink``.
    DELIVERABLES = "deliverables"
    #: The stored refs of ``deliverables``; written by the runtime, never by a tool.
    ARTIFACTS = "artifacts"
    #: A service host's :class:`~actant.sandbox.protocol.StorageStatus`, as JSON.
    STORAGE = "storage"


@dataclass
class ToolResult:
    output: object = None
    error: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    content_blocks: list[Block] | None = None

    @classmethod
    def ok(cls, output: object = None, **metadata: object) -> "ToolResult":
        return cls(output=output, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: object) -> "ToolResult":
        return cls(error=error, metadata=metadata)

    def is_success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {}
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.error:
            result["error"] = self.error
        else:
            result["result"] = self.output
        if self.metadata:
            result["metadata"] = self.metadata
        if self.content_blocks:
            result["content_blocks"] = BLOCKS.dump_python(self.content_blocks, mode="json")
        return result


class ToolInvocation(Protocol):
    def get_description(self) -> str: ...

    async def execute(self) -> ToolResult: ...


@dataclass(frozen=True)
class CallContext:
    """The call a tool is being built for: who is calling, from which thread and run.

    Every ``build`` receives one, so a tool never has to learn its thread from
    its arguments. ``sandbox`` is set only for tools that declared
    ``needs_sandbox``; for everything else it is ``None``.
    """

    agent_id: str
    thread_id: str
    run_id: str
    tool_call_id: str
    turn_id: str
    sandbox: Sandbox | None = None
    #: The agent's ``sandbox_access``, beside ``sandbox``: pass it to ``sandbox.exec`` to run
    #: as the agent's own user.
    sandbox_access: SandboxAccess | None = None
    #: Set when the calling thread is itself a subagent. One level of delegation
    #: only: a tool that spawns threads refuses when this is set.
    parent_thread_id: str | None = None
    #: The thread's ``ThreadInput.context``, given by whoever started it; empty for
    #: admission checks.
    context: Mapping[str, Any] = field(default_factory=dict)


_CURRENT: ContextVar[CallContext | None] = ContextVar("actant_current_call", default=None)


def current_call() -> CallContext | None:
    """The call the tool running here was built for, for code a tool calls that is not
    handed it (a service runner); ``None`` outside a tool's execution."""
    return _CURRENT.get()


@contextmanager
def calling(ctx: CallContext) -> Iterator[None]:
    """:func:`current_call` is ``ctx`` inside."""
    token = _CURRENT.set(ctx)
    try:
        yield
    finally:
        _CURRENT.reset(token)


class Tool(Protocol):
    """A tool. Optional attributes the runtime reads: ``needs_sandbox`` (see
    :class:`SandboxedTool`) and ``retry_safe``, ``True`` when a call whose worker was
    lost may simply run again; without it such a call ends as interrupted."""

    name: str

    @property
    def schema(self) -> ToolSchema: ...

    async def build(self, params: JSONObject, ctx: CallContext) -> ToolInvocation: ...


class SandboxedTool(Tool, Protocol):
    """A tool that runs in the thread's sandbox: ``needs_sandbox`` is ``True`` and
    ``ctx.sandbox`` is set when ``build`` is called."""

    needs_sandbox: bool


class BaseToolInvocation(Generic[ParamsT, OutputT]):
    def __init__(self, params: ParamsT) -> None:
        self.params = params

    def get_description(self) -> str:
        return "Running tool"

    async def execute(self) -> ToolResult:
        raise NotImplementedError


class BaseDeclarativeTool:
    def __init__(self, name: str, schema: ToolSchema) -> None:
        self.name = name
        self._schema = schema

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    async def build(self, params: JSONObject, ctx: CallContext) -> ToolInvocation:
        raise NotImplementedError
