"""Core utilities and structural types."""

from __future__ import annotations

import uuid
from typing import TypeAlias

JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
JSONObject: TypeAlias = dict[str, JSONValue]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"
