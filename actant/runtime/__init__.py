"""Public runtime entry points.

The runtime package is organized by responsibility:

- ``runtime.py``: application-facing client facade.
- ``temporal/workflow.py``: durable orchestration algorithm.
- ``temporal/activities.py``: side effects scheduled by the workflow.
- ``temporal/client.py`` and ``temporal/worker.py``: deployment roles.
- ``stores/``: readable execution projections.
- ``hooks.py``: optional lifecycle and streaming observers.

See ``docs/architecture.md`` in the source distribution for the detailed
execution path and activity contracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from actant.runtime.temporal.types import TemporalRuntimeConfig
    from actant.runtime.temporal.worker import TemporalRuntimeWorker
    from actant.runtime.runtime import AgentRuntime

__all__ = [
    "AgentRuntime",
    "TemporalRuntimeConfig",
    "TemporalRuntimeWorker",
]


def __getattr__(name: str) -> Any:
    if name == "AgentRuntime":
        from actant.runtime.runtime import AgentRuntime

        return AgentRuntime
    if name == "TemporalRuntimeConfig":
        from actant.runtime.temporal.types import TemporalRuntimeConfig

        return TemporalRuntimeConfig
    if name == "TemporalRuntimeWorker":
        from actant.runtime.temporal.worker import TemporalRuntimeWorker

        return TemporalRuntimeWorker
    raise AttributeError(f"module 'actant.runtime' has no attribute {name!r}")
