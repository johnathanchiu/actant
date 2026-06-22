"""Tool wrappers for memory stores."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from actant.core import JSONObject, new_id
from actant.memory.cards import MemoryCard
from actant.memory.store import MemoryStore
from actant.tools import BaseToolInvocation, Tool, ToolResult, ToolSchema, make_tool_schema


class _MemoryInvocation(BaseToolInvocation[JSONObject, object]):
    def __init__(
        self,
        params: JSONObject,
        description: str,
        execute: Callable[[JSONObject], Awaitable[ToolResult]],
    ) -> None:
        super().__init__(params)
        self._description = description
        self._execute = execute

    def get_description(self) -> str:
        return self._description

    async def execute(self) -> ToolResult:
        return await self._execute(self.params)


def _string(args: JSONObject, key: str) -> str | None:
    value = args.get(key)
    return value if isinstance(value, str) and value else None


def _string_tuple(args: JSONObject, key: str) -> tuple[str, ...]:
    value = args.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _metadata(args: JSONObject) -> JSONObject:
    value = args.get("metadata")
    return value if isinstance(value, dict) else {}


@dataclass
class ListMemoryCardsTool:
    store: MemoryStore
    namespace: str
    name: str = "list_memory_cards"

    @property
    def schema(self) -> ToolSchema:
        return make_tool_schema(self.name, "List available memory cards.")

    async def build(self, params: JSONObject) -> _MemoryInvocation:
        return _MemoryInvocation(params, "List memory cards", self._execute)

    async def _execute(self, args: JSONObject) -> ToolResult:
        del args
        refs = await self.store.list(self.namespace)
        return ToolResult.ok({"cards": [ref.to_dict() for ref in refs]})


@dataclass
class ReadMemoryCardTool:
    store: MemoryStore
    namespace: str
    name: str = "read_memory_card"

    @property
    def schema(self) -> ToolSchema:
        return make_tool_schema(
            self.name,
            "Read a memory card by id.",
            parameters={"card_id": {"type": "string"}},
            required=["card_id"],
        )

    async def build(self, params: JSONObject) -> _MemoryInvocation:
        return _MemoryInvocation(params, "Read memory card", self._execute)

    async def _execute(self, args: JSONObject) -> ToolResult:
        card_id = _string(args, "card_id")
        if card_id is None:
            return ToolResult.fail("card_id is required")
        card = await self.store.get(self.namespace, card_id)
        if card is None:
            return ToolResult.fail(f"Memory card {card_id!r} not found")
        return ToolResult.ok(card.to_dict())


@dataclass
class SearchMemoryTool:
    store: MemoryStore
    namespace: str
    name: str = "search_memory"

    @property
    def schema(self) -> ToolSchema:
        return make_tool_schema(
            self.name,
            "Search memory cards by keyword.",
            parameters={
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            required=["query"],
        )

    async def build(self, params: JSONObject) -> _MemoryInvocation:
        return _MemoryInvocation(params, "Search memory", self._execute)

    async def _execute(self, args: JSONObject) -> ToolResult:
        query = _string(args, "query")
        if query is None:
            return ToolResult.fail("query is required")
        raw_limit = args.get("limit")
        limit = raw_limit if isinstance(raw_limit, int) and raw_limit > 0 else 10
        results = await self.store.search(self.namespace, query, limit=limit)
        return ToolResult.ok({"results": [result.to_dict() for result in results]})


@dataclass
class WriteMemoryCardTool:
    store: MemoryStore
    namespace: str
    name: str = "write_memory_card"

    @property
    def schema(self) -> ToolSchema:
        return make_tool_schema(
            self.name,
            "Create or replace a memory card.",
            parameters={
                "card_id": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"},
            },
            required=["title", "body"],
        )

    async def build(self, params: JSONObject) -> _MemoryInvocation:
        return _MemoryInvocation(params, "Write memory card", self._execute)

    async def _execute(self, args: JSONObject) -> ToolResult:
        title = _string(args, "title")
        body = _string(args, "body")
        if title is None:
            return ToolResult.fail("title is required")
        if body is None:
            return ToolResult.fail("body is required")

        card_id = _string(args, "card_id") or new_id("mem")
        existing = await self.store.get(self.namespace, card_id)
        if existing is None:
            card = MemoryCard(
                id=card_id,
                namespace=self.namespace,
                title=title,
                body=body,
                tags=_string_tuple(args, "tags"),
                metadata=_metadata(args),
            )
            await self.store.put(card)
        else:
            existing.title = title
            existing.body = body
            existing.tags = _string_tuple(args, "tags")
            existing.metadata = _metadata(args)
            existing.updated_at = datetime.now(UTC)
            existing.version += 1
            card = existing
        return ToolResult.ok(card.to_dict())


@dataclass
class AppendMemoryCardTool:
    store: MemoryStore
    namespace: str
    name: str = "append_memory_card"

    @property
    def schema(self) -> ToolSchema:
        return make_tool_schema(
            self.name,
            "Append text to an existing memory card.",
            parameters={
                "card_id": {"type": "string"},
                "body": {"type": "string"},
            },
            required=["card_id", "body"],
        )

    async def build(self, params: JSONObject) -> _MemoryInvocation:
        return _MemoryInvocation(params, "Append memory card", self._execute)

    async def _execute(self, args: JSONObject) -> ToolResult:
        card_id = _string(args, "card_id")
        body = _string(args, "body")
        if card_id is None:
            return ToolResult.fail("card_id is required")
        if body is None:
            return ToolResult.fail("body is required")
        card = await self.store.append(self.namespace, card_id, body)
        if card is None:
            return ToolResult.fail(f"Memory card {card_id!r} not found")
        return ToolResult.ok(card.to_dict())


def memory_tools(store: MemoryStore, namespace: str) -> list[Tool]:
    return [
        ListMemoryCardsTool(store=store, namespace=namespace),
        ReadMemoryCardTool(store=store, namespace=namespace),
        SearchMemoryTool(store=store, namespace=namespace),
        WriteMemoryCardTool(store=store, namespace=namespace),
        AppendMemoryCardTool(store=store, namespace=namespace),
    ]
