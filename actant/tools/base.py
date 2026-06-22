"""Base classes for tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

from actant.core import JSONObject

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


@dataclass
class ToolResult:
    output: object = None
    error: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    content_blocks: list[dict[str, object]] | None = None

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
            result["content_blocks"] = self.content_blocks
        return result


class ToolInvocation(Protocol):
    def get_description(self) -> str: ...

    async def execute(self) -> ToolResult: ...


class Tool(Protocol):
    name: str

    @property
    def schema(self) -> ToolSchema: ...

    async def build(self, params: JSONObject) -> ToolInvocation: ...


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

    async def build(self, params: JSONObject) -> ToolInvocation:
        raise NotImplementedError
