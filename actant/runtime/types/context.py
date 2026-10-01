"""Turn context models."""

from __future__ import annotations

from dataclasses import dataclass

from actant.agents import AgentDefinition
from actant.llm.messages import Message


@dataclass
class TurnContext:
    agent: AgentDefinition
    system_prompt: str
    messages: list[Message]
    thread_id: str
    turn_id: str
    turn_index: int
