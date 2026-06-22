"""Actant runtime package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from actant.agents import Agent, AgentDefinition, ContextPolicy, ModelConfig
    from actant.runtime import AgentRuntime

__all__ = [
    "Agent",
    "AgentDefinition",
    "AgentRuntime",
    "ContextPolicy",
    "ModelConfig",
]


def __getattr__(name: str) -> Any:
    if name in {"Agent", "AgentDefinition", "ContextPolicy", "ModelConfig"}:
        from actant import agents

        return getattr(agents, name)
    if name == "AgentRuntime":
        from actant import runtime

        return runtime.AgentRuntime
    raise AttributeError(f"module 'actant' has no attribute {name!r}")
