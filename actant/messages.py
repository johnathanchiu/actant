"""Text the model reads: tool errors and the runtime's reminders.

Each is a named constant, formatted with ``str.format`` where it takes values, so the
wording lives in one place and an app can see exactly what its agents are told.
"""

from __future__ import annotations

#: Appended when a ``terminal`` agent answers in prose instead of calling ``finish``.
FINISH_REMINDER = (
    "<finish_required>\n"
    "This run ends only when you call `finish` with your summary and deliverable paths, "
    "or when a tool result is terminal. Call `finish` now, or keep working with your tools.\n"
    "</finish_required>"
)

# === the tool activities ===
TOOL_NOT_FOUND = "Tool {name} not found"
TOOL_BUILD_ERROR = "Tool build error: {error}"
TOOL_CALL_DENIED = "Tool call denied"
TOOL_EXECUTION_ERROR = "Tool execution error: {error}"
NO_ARTIFACT_SINK = "deliverables were listed but the worker has no artifact sink"
DELIVERABLE_UNREADABLE = "deliverable {path!r} could not be read: {error}"
TOOL_CALL_NOT_WAITING = "Tool call is {status}, not waiting"
RESOLUTION_TIMED_OUT = "Deferred tool resolution timed out"
ON_RESOLVE_FAILED = "on_resolve failed: {error}"

# === built-in tools ===
TOOL_NOT_APPROVED = "Tool call was not approved"
TOOL_NEEDS_SANDBOX = "Tool {name!r} needs a sandbox and none was given"
TOOL_NEEDS_CONTEXT = "Tool {name!r} needs its call context"
NO_IMAGE_WITH_ID = "no image with id {id!r} in this thread"
FINISH_PATHS_NOT_A_LIST = "`paths` must be a list of workspace paths"

# === the task tool ===
SUBAGENT_REQUIRED = "`subagent` is required"
MESSAGE_REQUIRED = "`message` is required"
UNKNOWN_SUBAGENT = "Unknown subagent {name!r}; valid: {valid}"
SUBAGENT_NOT_FOUND = "Subagent {name!r} not found"
SUBAGENT_SPAWN_FAILED = "Subagent spawn failed: {error}"
TASK_NO_INVOKER = "TaskTool has neither an invoker nor a spawner."
TASK_NO_PARENT = (
    "TaskTool has no parent_thread_id: neither set at construction nor present on the tool call."
)
