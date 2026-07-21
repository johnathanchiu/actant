"""Typed events consumed from an Actant thread's live event stream."""

from __future__ import annotations

from dataclasses import dataclass, field

from actant.core import JSONObject


@dataclass(frozen=True)
class ThreadEvent:
    """Transport-neutral event emitted for one agent thread."""

    type: str
    thread_id: str
    data: JSONObject = field(default_factory=dict)
    parent_thread_id: str | None = None
    parent_tool_call_id: str | None = None
    subagent: str | None = None

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "ThreadEvent":
        event_type = payload.get("type")
        thread_id = payload.get("thread_id")
        data = payload.get("data")
        if not isinstance(event_type, str):
            raise ValueError("runtime event is missing a string `type`")
        if not isinstance(thread_id, str):
            raise ValueError("runtime event is missing a string `thread_id`")
        if not isinstance(data, dict):
            data = {}
        return cls(
            type=event_type,
            thread_id=thread_id,
            data=data,
            parent_thread_id=_optional_string(payload.get("parent_thread_id")),
            parent_tool_call_id=_optional_string(payload.get("parent_tool_call_id")),
            subagent=_optional_string(payload.get("subagent")),
        )

    @property
    def text(self) -> str | None:
        """Text delta or completed assistant text, when present."""
        value = self.data.get("delta")
        if isinstance(value, str):
            return value
        value = self.data.get("content")
        return value if isinstance(value, str) else None

    @property
    def tool_call_id(self) -> str | None:
        return _optional_string(self.data.get("tool_call_id"))

    @property
    def is_terminal(self) -> bool:
        return self.type in {"complete", "error"}


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None
