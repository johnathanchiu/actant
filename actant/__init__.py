"""Actant runtime package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from actant.agents import AgentDefinition
    from actant.runtime import AgentRuntime
    from actant.tools import FunctionTool, tool

__all__ = [
    "AgentDefinition",
    "AgentRuntime",
    "FunctionTool",
    "tool",
]


def __getattr__(name: str) -> Any:
    if name == "AgentDefinition":
        from actant import agents

        return agents.AgentDefinition
    if name == "AgentRuntime":
        from actant import runtime

        return runtime.AgentRuntime
    if name in {"FunctionTool", "tool"}:
        from actant import tools

        return getattr(tools, name)
    raise AttributeError(f"module 'actant' has no attribute {name!r}")
