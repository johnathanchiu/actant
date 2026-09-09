"""Worker-bound Temporal activity groups."""

from __future__ import annotations

from collections.abc import Callable

from actant.runtime.temporal.activities.context import (
    HookFactory,
    ListenerFactory,
    MessagePreprocessor,
)
from actant.runtime.temporal.activities.runs import RunActivities
from actant.runtime.temporal.activities.threads import ThreadActivities
from actant.runtime.temporal.activities.tools import ToolActivities


class TemporalRuntimeActivities(RunActivities, ToolActivities, ThreadActivities):
    """Complete activity set hosted by an Actant worker."""

    @property
    def all(self) -> list[Callable[..., object]]:
        """Activity callables for Temporal worker registration."""
        return [
            self.start_run,
            self.run_turn,
            self.admit_tool,
            self.execute_tool,
            self.resolve_tool,
            self.finalize_tool_group,
            self.finalize_run,
            self.apply_thread_cancellation,
        ]


__all__ = [
    "HookFactory",
    "ListenerFactory",
    "MessagePreprocessor",
    "RunActivities",
    "TemporalRuntimeActivities",
    "ThreadActivities",
    "ToolActivities",
]
