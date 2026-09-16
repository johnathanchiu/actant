"""Compose activity groups without an inherited runtime or forwarding facade."""

from collections.abc import Callable
from actant.runtime.temporal.activities.context import (
    ActivityContext,
    MessagePreprocessor,
)
from actant.runtime.temporal.activities.runs import RunActivities
from actant.runtime.temporal.activities.tools import ToolActivities
from actant.runtime.temporal.activities.threads import ThreadActivities


class TemporalRuntimeActivities:
    def __init__(self, context: ActivityContext) -> None:
        self.runs = RunActivities(context)
        self.tools = ToolActivities(context)
        self.threads = ThreadActivities(context)

    @property
    def all(self) -> list[Callable[..., object]]:
        return [
            self.runs.start_run,
            self.runs.run_turn,
            self.tools.admit_tool,
            self.tools.execute_tool,
            self.tools.resolve_tool,
            self.tools.finalize_tool_group,
            self.runs.finalize_run,
            self.threads.apply_thread_cancellation,
        ]


__all__ = [
    "TemporalRuntimeActivities",
    "RunActivities",
    "ToolActivities",
    "ThreadActivities",
    "ActivityContext",
    "MessagePreprocessor",
]
