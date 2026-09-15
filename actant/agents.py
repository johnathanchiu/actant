"""Agent configuration models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from actant.llm.base import LLMClient
from actant.llm.messages import Message
from actant.sandbox.base import SandboxSpec
from actant.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from actant.runtime.events.streaming import StreamListener


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
    #: Where this agent's sandboxed tools run. ``None`` means no tool may
    #: declare ``needs_sandbox``.
    sandbox: SandboxSpec | None = None
    #: How a run ends. ``reply``: a turn with no tool calls completes it,
    #: which is right for chat. ``terminal``: only a terminal tool result
    #: (``finish``, or any result with ``metadata["terminal"]``) completes
    #: it; a text-only turn gets one reminder, a second ends the run as
    #: exhausted. Right for task agents, which otherwise end by silence.
    completion: Literal["reply", "terminal"] = "reply"
    final_tools: tuple[str, ...] = ()

    async def complete(
        self,
        messages: Sequence[Message],
        listener: "StreamListener | None" = None,
        *,
        final_turn: bool = False,
    ) -> Message:
        if final_turn and self.final_tools:
            if any(
                name not in self.tools or (self.tool_allowlist and name not in self.tool_allowlist)
                for name in self.final_tools
            ):
                raise ValueError("final_tools must be registered and allowed tools")
            return await self.llm.complete(
                self.persona,
                [
                    *messages,
                    Message(
                        role="user",
                        content=f"Last turn: use {', '.join(self.final_tools)} to finish your remaining changes.",
                    ),
                ],
                self.tools.schemas_for(self.tool_allowlist),
                listener,
                allowed_tools=self.final_tools,
            )
        return await self.llm.complete(
            self.persona,
            list(messages),
            self.tools.schemas_for(self.tool_allowlist),
            listener,
        )
