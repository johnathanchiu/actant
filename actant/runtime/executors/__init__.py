"""Runtime executor implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from actant.runtime.executors.base import RuntimeExecutor
    from actant.runtime.executors.temporal import TemporalExecutor, TemporalRuntimeWorker
    from actant.runtime.executors.temporal_types import TemporalRuntimeConfig

__all__ = [
    "RuntimeExecutor",
    "TemporalExecutor",
    "TemporalRuntimeConfig",
    "TemporalRuntimeWorker",
]


def __getattr__(name: str) -> Any:
    if name == "RuntimeExecutor":
        from actant.runtime.executors.base import RuntimeExecutor

        return RuntimeExecutor
    if name in {"TemporalExecutor", "TemporalRuntimeWorker"}:
        from actant.runtime.executors import temporal

        return getattr(temporal, name)
    if name == "TemporalRuntimeConfig":
        from actant.runtime.executors.temporal_types import TemporalRuntimeConfig

        return TemporalRuntimeConfig
    raise AttributeError(f"module 'actant.runtime.executors' has no attribute {name!r}")
