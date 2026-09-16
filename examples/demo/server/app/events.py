"""Route descendant events using durable thread parent links."""

from actant.core import JSONObject
from actant.runtime.events.publisher import EventSink
from actant.runtime.interfaces.stores import ThreadStore
from actant.runtime.types.threads import AgentThread


class DemoEvents:
    def __init__(self, threads: ThreadStore, publisher: EventSink, agent_ids: list[str]) -> None:
        self.threads = threads
        self.publisher = publisher
        self.agent_ids = agent_ids
        self._parents: dict[str, tuple[str, JSONObject] | None] = {}

    async def _thread(self, thread_id: str) -> AgentThread:
        for agent_id in self.agent_ids:
            try:
                return await self.threads.get(agent_id, thread_id)
            except KeyError:
                continue
        raise KeyError(thread_id)

    async def publish(self, channel: str, event: JSONObject) -> None:
        await self.publisher.publish(channel, event)
        thread_id = event.get("thread_id")
        if not isinstance(thread_id, str):
            return
        if thread_id not in self._parents:
            thread = await self._thread(thread_id)
            route: tuple[str, JSONObject] | None = None
            if thread.parent_thread_id is not None:
                metadata: JSONObject = {
                    "parent_thread_id": thread.parent_thread_id,
                    "subagent": thread.agent_id,
                }
                seen = {thread_id}
                while thread.parent_thread_id is not None:
                    if thread.parent_thread_id in seen:
                        raise ValueError("cyclic demo parent links")
                    seen.add(thread.parent_thread_id)
                    thread = await self._thread(thread.parent_thread_id)
                route = (f"thread:{thread.id}", metadata)
            self._parents[thread_id] = route
        route = self._parents[thread_id]
        if route is not None:
            parent_channel, metadata = route
            await self.publisher.publish(parent_channel, {**event, **metadata})
