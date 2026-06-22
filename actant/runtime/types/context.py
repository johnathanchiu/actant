"""Turn context models."""

from __future__ import annotations

from dataclasses import dataclass

from actant.agents import Agent, AgentDefinition
from actant.llm.messages import Message


@dataclass
class TurnContext:
    agent: Agent | AgentDefinition
    system_prompt: str
    messages: list[Message]
    thread_id: str
    turn_id: str
    turn_index: int
