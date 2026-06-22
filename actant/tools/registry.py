"""Tool registry."""

from __future__ import annotations

from collections.abc import Sequence

from actant.tools.base import Tool


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas_for(self, allowlist: set[str] | None = None) -> list[dict[str, object]]:
        tools = list(self._tools.values())
        if allowlist:
            tools = [tool for tool in tools if tool.name in allowlist]
        return [tool.schema for tool in tools]

    def __contains__(self, name: str) -> bool:
        return name in self._tools
