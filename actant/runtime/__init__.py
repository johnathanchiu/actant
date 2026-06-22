"""Runtime execution primitives.

Public surface:

- ``AgentRuntime`` — friendly facade. Wires stores + agents + an
  executor (Temporal by default).
- ``TemporalRuntimeWorker`` — worker process that polls a Temporal
  task queue and executes ``AgentThreadWorkflow`` + activities.
- ``TemporalRuntimeConfig`` — connection/task-queue configuration.
- ``StepResult`` / ``StepStatus`` — return type of executor stepping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from actant.runtime.executors.temporal import TemporalRuntimeWorker
    from actant.runtime.executors.temporal_types import TemporalRuntimeConfig
    from actant.runtime.runtime import AgentRuntime
    from actant.runtime.types.orchestration import StepResult, StepStatus

__all__ = [
    "AgentRuntime",
    "StepResult",
    "StepStatus",
    "TemporalRuntimeConfig",
    "TemporalRuntimeWorker",
]


def __getattr__(name: str) -> Any:
    if name == "AgentRuntime":
        from actant.runtime.runtime import AgentRuntime

        return AgentRuntime
    if name in {"StepResult", "StepStatus"}:
        from actant.runtime.types import orchestration

        return getattr(orchestration, name)
    if name == "TemporalRuntimeConfig":
        from actant.runtime.executors.temporal_types import TemporalRuntimeConfig

        return TemporalRuntimeConfig
    if name == "TemporalRuntimeWorker":
        from actant.runtime.executors import TemporalRuntimeWorker

        return TemporalRuntimeWorker
    raise AttributeError(f"module 'actant.runtime' has no attribute {name!r}")
