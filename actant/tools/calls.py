"""Tool call lifecycle records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from actant.core import JSONObject


class ToolCallStatus(StrEnum):
    REQUESTED = "requested"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass
class ToolCallRecord:
    id: str
    group_id: str
    run_id: str
    agent_id: str
    thread_id: str
    turn_id: str
    turn_index: int
    name: str
    args: JSONObject
    status: ToolCallStatus = ToolCallStatus.REQUESTED
    prompt: str | None = None
    wait_request: JSONObject | None = None
    result: object | None = None
