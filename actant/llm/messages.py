"""Provider-neutral message and streaming models."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, cast

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCallFunction:
    name: str
    arguments: str

    @classmethod
    def from_raw(cls, value: "ToolCallFunction | dict[str, object]") -> "ToolCallFunction":
        if isinstance(value, cls):
            return cls(name=value.name, arguments=value.arguments)
        if not isinstance(value, dict):
            raise TypeError(f"Expected ToolCallFunction or dict, got {type(value).__name__}")

        raw_arguments = value.get("arguments", "")
        if raw_arguments is None:
            arguments = ""
        elif isinstance(raw_arguments, str):
            arguments = raw_arguments
        else:
            arguments = json.dumps(raw_arguments)

        raw_name = value.get("name", "")
        name = "" if raw_name is None else str(raw_name)
        return cls(name=name, arguments=arguments)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ToolCall:
    id: str
    function: ToolCallFunction
    type: str = "function"
    thought_signature: str | None = None
    extra_content: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, value: "ToolCall | dict[str, object]") -> "ToolCall":
        if isinstance(value, cls):
            return cls(
                id=value.id,
                function=ToolCallFunction.from_raw(value.function),
                type=value.type,
                thought_signature=value.thought_signature,
                extra_content=deepcopy(value.extra_content),
            )
        if not isinstance(value, dict):
            raise TypeError(f"Expected ToolCall or dict, got {type(value).__name__}")

        raw_id = value.get("id", "")
        raw_type = value.get("type", "function")
        raw_extra = value.get("extra_content", {})
        raw_signature = value.get("thought_signature")
        return cls(
            id="" if raw_id is None else str(raw_id),
            function=ToolCallFunction.from_raw(cast(dict[str, object], value.get("function", {}))),
            type="function" if raw_type is None else str(raw_type),
            thought_signature=str(raw_signature) if raw_signature is not None else None,
            extra_content=deepcopy(raw_extra) if isinstance(raw_extra, dict) else {},
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "id": self.id,
            "function": self.function.to_dict(),
        }
        if self.type != "function":
            data["type"] = self.type
        if self.thought_signature is not None:
            data["thought_signature"] = self.thought_signature
        if self.extra_content:
            data["extra_content"] = deepcopy(self.extra_content)
        return data


@dataclass
class Message:
    role: Role
    content: str | list[dict[str, object]] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    thought_summary: str | None = None
    thinking_signature: str | None = None
    reasoning_items: list[object] | None = None

    @classmethod
    def from_raw(cls, value: "Message | dict[str, object]") -> "Message":
        if isinstance(value, cls):
            return cls(
                role=value.role,
                content=deepcopy(value.content),
                tool_calls=(
                    [ToolCall.from_raw(tc) for tc in value.tool_calls]
                    if value.tool_calls is not None
                    else None
                ),
                tool_call_id=value.tool_call_id,
                name=value.name,
                thought_summary=value.thought_summary,
                thinking_signature=value.thinking_signature,
                reasoning_items=deepcopy(value.reasoning_items),
            )
        if not isinstance(value, dict):
            raise TypeError(f"Expected Message or dict, got {type(value).__name__}")

        raw_tool_calls = value.get("tool_calls")
        content = value.get("content", "")
        raw_reasoning_items = value.get("reasoning_items")
        return cls(
            role=cast(Role, value.get("role", "user")),
            content=deepcopy(content) if isinstance(content, list) else str(content),
            tool_calls=(
                [ToolCall.from_raw(tc) for tc in raw_tool_calls]
                if isinstance(raw_tool_calls, list)
                else None
            ),
            tool_call_id=cast(str | None, value.get("tool_call_id")),
            name=cast(str | None, value.get("name")),
            thought_summary=cast(str | None, value.get("thought_summary")),
            thinking_signature=cast(str | None, value.get("thinking_signature")),
            reasoning_items=(
                deepcopy(raw_reasoning_items) if isinstance(raw_reasoning_items, list) else None
            ),
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "role": self.role,
            "content": deepcopy(self.content),
        }
        if self.tool_calls is not None:
            data["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            data["name"] = self.name
        if self.thought_summary is not None:
            data["thought_summary"] = self.thought_summary
        if self.thinking_signature is not None:
            data["thinking_signature"] = self.thinking_signature
        if self.reasoning_items is not None:
            data["reasoning_items"] = deepcopy(self.reasoning_items)
        return data
