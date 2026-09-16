"""Demo routing recovers from persisted links without a coordinator registry."""

from examples.demo.server.app.events import DemoEvents
from actant.runtime.stores import InMemoryRuntimeStores


async def test_fresh_demo_router_publishes_grandchild_to_root() -> None:
    stores = InMemoryRuntimeStores()
    await stores.threads.get_or_create("root", "root")
    for agent_id, thread_id, parent in [
        ("researcher", "child", "root"),
        ("summarizer", "leaf", "child"),
    ]:
        thread = await stores.threads.get_or_create(agent_id, thread_id)
        thread.parent_thread_id = parent
        await stores.threads.update(thread)
    router = DemoEvents(stores.threads, stores.publisher, ["root", "researcher", "summarizer"])
    await router.publish(
        "thread:leaf", {"type": "text_delta", "thread_id": "leaf", "data": {"delta": "hi"}}
    )
    assert stores.publisher.events["thread:leaf"][0]["data"] == {"delta": "hi"}
    event = stores.publisher.events["thread:root"][0]
    assert event["parent_thread_id"] == "child" and event["subagent"] == "summarizer"
    assert "thread:child" not in stores.publisher.events
