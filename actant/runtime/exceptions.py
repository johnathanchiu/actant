"""Typed errors from application-facing runtime commands."""

from __future__ import annotations

from actant.tools.calls import ToolCallStatus


class ToolCallResolutionError(Exception):
    """Base class for invalid tool-resolution commands."""


class ToolCallNotFoundError(ToolCallResolutionError, LookupError):
    """The call does not exist or does not belong to the addressed thread."""

    def __init__(self, tool_call_id: str) -> None:
        super().__init__(f"Tool call {tool_call_id!r} was not found")
        self.tool_call_id = tool_call_id


class ToolCallNotWaitingError(ToolCallResolutionError):
    """The call exists but cannot accept an external resolution."""

    def __init__(self, tool_call_id: str, status: ToolCallStatus) -> None:
        super().__init__(f"Tool call {tool_call_id!r} is {status.value}, not waiting")
        self.tool_call_id = tool_call_id
        self.status = status


__all__ = [
    "ToolCallNotFoundError",
    "ToolCallNotWaitingError",
    "ToolCallResolutionError",
]
