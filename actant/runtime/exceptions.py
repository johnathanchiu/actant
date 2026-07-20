"""Typed exceptions raised by the runtime.

These exist so application code can catch specific failure modes
(e.g. "the workflow lost the activity, store needs cleanup") without
matching on raw Temporal SDK error strings.
"""

from __future__ import annotations


class RuntimeError_(Exception):
    """Base class for actant runtime exceptions."""


class ToolResolutionStaleError(RuntimeError_):
    """Raised by ``runtime.resolve_deferred_tool_call`` when the parked Temporal
    activity for a deferred tool call no longer exists on the worker.

    This happens when:
    - Temporal's persistent state was reset (volume nuke in dev) but
      the actant tool_calls store still records the call as WAITING.
    - The workflow that owned the activity was terminated or timed
      out independently.
    - A race condition between the activity completing and a stale
      ``resolve_deferred_tool_call`` call beating its successor to delivery.

    The runtime updates the store to ``ToolCallStatus.FAILED`` with a
    diagnostic ``stale_activity`` result before raising so the
    application's next read of the tool-call record sees a terminal
    state instead of perpetually offering to resolve.

    Catch this in application code (DemoCoordinator.resolve_deferred_tool_call,
    or whatever your equivalent is) and surface a useful message to
    the user: the parked operation is gone, they should start a new
    run instead of waiting.
    """

    def __init__(self, tool_call_id: str, reason: str) -> None:
        super().__init__(
            f"Deferred tool call {tool_call_id!r} cannot be resolved: {reason}"
        )
        self.tool_call_id = tool_call_id
        self.reason = reason


__all__ = ["RuntimeError_", "ToolResolutionStaleError"]
