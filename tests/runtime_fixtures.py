"""Static definitions for tests; production can resolve persisted per-thread context."""

from collections.abc import Mapping
from actant.agents import AgentDefinition
from actant.runtime.temporal.activities.context import AgentResolver


def static_agents(agents: Mapping[str, AgentDefinition]) -> AgentResolver:
    async def resolve(agent_id: str, thread_id: str) -> AgentDefinition:
        return agents[agent_id]

    return resolve
