"""Agent configuration models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from actant.llm.base import LLMClient
from actant.llm.messages import Message
from actant.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from actant.runtime.hooks import StreamListener


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None


@dataclass(frozen=True)
class Agent:
    id: str
    name: str
    persona: str
    persona_version: str
    model: ModelConfig
    tool_allowlist: set[str] = field(default_factory=set)
    max_turns_per_thread: int = 25


@dataclass(frozen=True)
class ContextPolicy:
    max_input_tokens: int | None = None
    reserve_output_tokens: int = 4096


@dataclass(frozen=True)
class AgentDefinition:
    id: str
    name: str
    persona: str
    llm: LLMClient
    tools: ToolRegistry
    tool_allowlist: set[str] = field(default_factory=set)
    context_policy: ContextPolicy = field(default_factory=ContextPolicy)
    persona_version: str = "v1"
    max_turns_per_thread: int = 25

    async def complete(
        self,
        messages: Sequence[Message],
        listener: "StreamListener | None" = None,
    ) -> Message:
        return await self.llm.complete(
            self.persona,
            list(messages),
            self.tools.schemas_for(self.tool_allowlist),
            listener,
        )
