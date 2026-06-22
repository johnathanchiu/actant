"""Runtime executor boundary.

Executors own how runtime work is driven. The default implementation uses
Actant's existing SQL/in-memory queues; alternate implementations can place a
durable workflow engine below the same public runtime API.
"""

from __future__ import annotations

from typing import Protocol

from actant.runtime.types.orchestration import StepResult


class RuntimeExecutor(Protocol):
    async def send_message(
        self,
        agent_id: str,
        thread_id: str,
        content: str | list[dict[str, object]],
    ) -> str: ...

    async def run_one(self) -> StepResult: ...

    async def run_forever(self, *, idle_sleep: float = 0.1) -> None: ...

    async def run_until_idle(
        self, agent_id: str, thread_id: str, max_steps: int = 25
    ) -> StepResult: ...

    def stop(self) -> None: ...
