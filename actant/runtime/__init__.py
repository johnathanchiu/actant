"""Public Temporal runtime, thread handles, and execution contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from actant.runtime.temporal.activities.context import AgentResolver
    from actant.runtime.completion import RunCompletion, RunCompletionHandler
    from actant.runtime.gate import TurnGate, TurnStart
    from actant.runtime.temporal.types import TemporalRuntimeConfig
    from actant.runtime.thread import ThreadHandle
    from actant.runtime.runtime import AgentRuntime

__all__ = [
    "AgentRuntime",
    "AgentResolver",
    "RunCompletion",
    "RunCompletionHandler",
    "TemporalRuntimeConfig",
    "ThreadHandle",
    "TurnGate",
    "TurnStart",
]


def __getattr__(name: str) -> Any:
    if name == "AgentResolver":
        from actant.runtime.temporal.activities.context import AgentResolver

        return AgentResolver
    if name == "AgentRuntime":
        from actant.runtime.runtime import AgentRuntime

        return AgentRuntime
    if name in {"RunCompletion", "RunCompletionHandler"}:
        from actant.runtime import completion

        return getattr(completion, name)
    if name in {"TurnGate", "TurnStart"}:
        from actant.runtime import gate

        return getattr(gate, name)
    if name == "TemporalRuntimeConfig":
        from actant.runtime.temporal.types import TemporalRuntimeConfig

        return TemporalRuntimeConfig
    if name == "ThreadHandle":
        from actant.runtime.thread import ThreadHandle

        return ThreadHandle
    raise AttributeError(f"module 'actant.runtime' has no attribute {name!r}")
