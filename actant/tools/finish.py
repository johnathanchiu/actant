"""The explicit end of a task run.

A chat agent is done when it answers. A task agent has to say so: ``finish``
ends the run (its result is terminal) and names the deliverables, files in
the thread's sandbox that the runtime stores as artifacts and reports on the
run's completion. Agents without a sandbox can still call it with no paths.
"""

from __future__ import annotations

from actant.core import JSONObject
from actant.tools.base import (
    BaseDeclarativeTool,
    BaseToolInvocation,
    CallContext,
    ToolInvocation,
    ToolResult,
    make_tool_schema,
)

DESCRIPTION = (
    "Call once, when the work is complete. `summary` is your final answer; `paths` are "
    "files in your workspace to hand back as deliverables. This ends the run."
)


class _FinishInvocation(BaseToolInvocation[JSONObject, JSONObject]):
    def get_description(self) -> str:
        return "Finishing"

    async def execute(self) -> ToolResult:
        paths = self.params.get("paths") or []
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            return ToolResult.fail("`paths` must be a list of workspace paths")
        return ToolResult.ok(
            {"status": "done", "summary": self.params.get("summary", "")},
            terminal=True,
            deliverables=paths,
        )


class FinishTool(BaseDeclarativeTool):
    def __init__(self, description: str = DESCRIPTION) -> None:
        super().__init__(
            "finish",
            make_tool_schema(
                "finish",
                description,
                parameters={
                    "summary": {"type": "string", "description": "The final answer, briefly."},
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Workspace paths of the deliverables.",
                    },
                },
                required=["summary"],
            ),
        )

    async def build(self, params: JSONObject, ctx: CallContext) -> ToolInvocation:
        del ctx
        return _FinishInvocation(params)
