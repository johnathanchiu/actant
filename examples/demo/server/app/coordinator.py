"""Application-owned agent construction, delegation, and durable completion.

DemoEvents routes live events using persisted parent links. Completion is a
separate retryable callback, independent of the process that spawned the child.
"""

from __future__ import annotations

import asyncio
from temporalio.client import Client
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from actant.core import JSONObject
from actant.agents import AgentDefinition
from actant.runtime import AgentRuntime, TemporalRuntimeConfig
from actant.runtime.completion import RunCompletion
from actant.runtime.stores.in_memory import InMemoryEventPublisher
from actant.runtime.stores.postgres import (
    SQLAlchemyMessageStore,
    SQLAlchemyRunStore,
    SQLAlchemyThreadStore,
    SQLAlchemyToolCallStore,
    create_schema,
)
from actant.runtime.types.threads import AgentThread
from actant.tools.base import Tool
from actant.tools.supervise import supervision_tools
from actant.tools.task import TaskTool

from app.agents import (
    AGENT_ID,
    RESEARCHER_AGENT_ID,
    SUMMARIZER_AGENT_ID,
    build_main_agent,
    build_researcher_agent,
    build_summarizer_agent,
)
from app.llm import build_llm
from app.events import DemoEvents


# Subagent names the demo recognizes, mapped to their agent IDs in the
# runtime. The TaskTool's `subagent_choices` constrain which subset
# each parent can actually call.
_SUBAGENT_IDS = {
    RESEARCHER_AGENT_ID: RESEARCHER_AGENT_ID,
    SUMMARIZER_AGENT_ID: SUMMARIZER_AGENT_ID,
}


DEFAULT_DATABASE_URL = "postgresql+asyncpg://actant:actant@localhost:55435/actant_demo"


@dataclass
class _DemoStores:
    """Composite of actant stores keyed off one shared Postgres
    session factory, plus an in-process event publisher (the demo
    runs server + worker in one process). For multi-process deploys
    you'd swap in a Redis or NATS publisher."""

    threads: SQLAlchemyThreadStore
    runs: SQLAlchemyRunStore
    messages: SQLAlchemyMessageStore
    tool_calls: SQLAlchemyToolCallStore
    publisher: InMemoryEventPublisher


class DemoCoordinator:
    """Composes actant primitives + demo policy into one object."""

    def __init__(
        self,
        stores: _DemoStores,
        runtime: AgentRuntime,
        main_agent: AgentDefinition,
        worker_task: asyncio.Task[None],
        engine: object,
        model_id: str,
    ) -> None:
        self.stores = stores
        self.runtime = runtime
        self.main_agent = main_agent
        self.worker_task = worker_task
        self.engine = engine
        self.model_id = model_id

    # ─── SubagentSpawner protocol (TaskTool.spawner) ────────────────

    async def spawn(
        self,
        *,
        name: str,
        message: str,
        context: JSONObject,
        parent_thread_id: str,
    ) -> str:
        """Start a sub-thread and hand its id back to the parent.

        The parent is NOT parked: the id it gets is the whole tool
        result, and it is what `check_subagent` / `message_subagent` /
        `stop_subagent` take. The parent hears about the ending when
        `handle_run_completion` messages it.
        """
        sub_agent_id = _SUBAGENT_IDS.get(name)
        if sub_agent_id is None:
            raise ValueError(f"unknown subagent name: {name!r}")
        sub_thread_id = f"sub_{uuid.uuid4().hex[:10]}"
        thread = await self.stores.threads.get_or_create(sub_agent_id, sub_thread_id)
        await self.stores.threads.update(
            AgentThread(
                id=thread.id,
                agent_id=thread.agent_id,
                status=thread.status,
                turn_count=thread.turn_count,
                active_run_id=thread.active_run_id,
                parent_thread_id=parent_thread_id,
            )
        )
        composed = message
        if context:
            composed = (
                f"{message}\n\nContext from caller:\n```json\n{json.dumps(context, indent=2)}\n```"
            )
        await self.runtime.send_message(sub_agent_id, sub_thread_id, composed)
        return sub_thread_id

    # ─── SubagentSupervisor protocol (supervision_tools) ────────────

    async def status(self, thread_id: str) -> JSONObject:
        """What the demo knows about a sub-thread, read by the model."""
        thread = await self._thread_row(thread_id)
        if thread is None:
            return {"thread_id": thread_id, "status": "unknown"}
        return {
            "thread_id": thread_id,
            "subagent": thread.agent_id,
            "status": str(thread.status),
            "turn_count": thread.turn_count,
        }

    async def send(self, thread_id: str, message: str) -> None:
        await self.runtime.send_message(await self.agent_id_for(thread_id), thread_id, message)

    async def stop(self, thread_id: str) -> None:
        # Must be safe on a sub-thread that already finished — the model
        # has no way to know it did.
        try:
            await self.runtime.cancel_thread(await self.agent_id_for(thread_id), thread_id)
        except Exception:  # noqa: BLE001 -- already gone is not an error here
            pass

    async def _thread_row(self, thread_id: str) -> AgentThread | None:
        """The durable row for a thread, whichever demo agent owns it.

        The stores are keyed by (agent_id, thread_id) and the demo has
        three agents, so "which agent owns this thread" is three misses
        at worst."""
        for agent_id in (AGENT_ID, *_SUBAGENT_IDS.values()):
            try:
                return await self.stores.threads.get(agent_id, thread_id)
            except KeyError:
                continue
        return None

    async def agent_id_for(self, thread_id: str) -> str:
        """Which agent owns a thread, read from the durable projection.

        NOT from the registry: that is an in-memory event-routing index,
        it is empty in a worker that did not spawn the thread, and a
        finished sub-thread is dropped from it. Guessing the top-level
        agent for a sub-thread starts a phantom workflow under the main
        agent bearing the sub-thread's id."""
        thread = await self._thread_row(thread_id)
        return thread.agent_id if thread is not None else AGENT_ID

    # ─── Resolve flows ──────────────────────────────────────────────

    async def resolve_user_deferred(
        self,
        *,
        thread_id: str,
        tool_call_id: str,
        approved: bool | None = None,
        answer: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """User-driven resolve (DeferredPanel POST). Single entry point.

        Derives ``agent_id`` from the thread row: if ``thread_id`` is a
        sub-thread, the wait belongs to the sub-agent (e.g. researcher's
        ask_user); otherwise it's a main-thread wait.

        Funneled through `resolve_tool_call`, which durably signals the
        owning thread workflow."""
        # The resolve restarts the thread, so its events need somewhere
        # to route, same as `send`.
        await self.runtime.resolve_tool_call(
            await self.agent_id_for(thread_id),
            thread_id,
            tool_call_id,
            approved=approved,
            answer=answer,
            payload=payload,
        )

    async def handle_run_completion(self, completion: RunCompletion) -> None:
        """Tell the parent its child finished, once the child's run is
        persisted.

        Nothing of the parent's is parked — its ``task()`` call completed the
        moment the child was spawned. So completion is a MESSAGE on the
        parent's inbox, the same one a person's message arrives on: it wakes
        the parent whether it is parked or already closed.

        This handler runs inside Temporal's retryable ``finalize_run``
        activity, so it derives linkage and output from stores rather than
        hooks or process memory. Harvest semantics are unchanged: the child's
        last assistant message, tagged with which subagent produced it.
        """
        thread = await self.stores.threads.get(completion.agent_id, completion.thread_id)
        if thread.parent_thread_id is None:
            return
        parent_agent_id = await self.agent_id_for(thread.parent_thread_id)
        # Completion handlers retry and signals are not deduplicated, so the
        # parent's own transcript is the "already delivered" record: the
        # envelope names the run that finished, and every run of a child
        # gets its own delivery. Registry presence cannot serve here — the
        # handler runs in whichever worker finalized the child's run, which
        # is not necessarily the one that spawned it. The window between the
        # signal and the workflow persisting the message is not covered; a
        # host that cannot tolerate a repeat wants its own delivery table.
        delivered = await self.stores.messages.list_for_thread(
            parent_agent_id, thread.parent_thread_id
        )
        if any(
            message.role == "user"
            and isinstance(message.content, str)
            and completion.run_id in message.content
            for message in delivered
        ):
            return

        messages = await self.stores.messages.list_for_thread(
            completion.agent_id, completion.thread_id
        )
        text = next(
            (
                message.content
                for message in reversed(messages)
                if message.role == "assistant"
                and isinstance(message.content, str)
                and message.content.strip()
            ),
            completion.outcome,
        )
        # The demo's subagent names and agent ids are the same strings, so the
        # completing run identifies the subagent without a tool-call lookup.
        envelope = {
            "subagent": completion.agent_id,
            "thread_id": completion.thread_id,
            "run_id": completion.run_id,
            "succeeded": completion.succeeded,
            "text": text,
        }
        # The parent may itself be a sub-thread whose link this worker has
        # never seen, and the message is about to start a run on it.
        await self.runtime.send_message(
            parent_agent_id,
            thread.parent_thread_id,
            f"Subagent finished:\n```json\n{json.dumps(envelope, indent=2)}\n```",
        )
        # The child's run is over, so its routing entry is dead weight until
        # something wakes it again — and whatever does re-links it first.

    # ─── Shutdown ───────────────────────────────────────────────────

    async def shutdown(self) -> None:
        self.worker_task.cancel()
        try:
            await self.worker_task
        except (asyncio.CancelledError, Exception):
            pass
        dispose = getattr(self.engine, "dispose", None)
        if dispose is not None:
            await dispose()


# ─── Event routing ──────────────────────────────────────────────────


async def build_coordinator() -> DemoCoordinator:
    """Wires everything together. Call once at server startup."""
    database_url = os.getenv("ACTANT_DEMO_DATABASE_URL", DEFAULT_DATABASE_URL)
    engine = create_async_engine(database_url, future=True)
    await create_schema(engine)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    stores = _DemoStores(
        threads=SQLAlchemyThreadStore(session_factory),
        runs=SQLAlchemyRunStore(session_factory),
        messages=SQLAlchemyMessageStore(session_factory),
        tool_calls=SQLAlchemyToolCallStore(session_factory),
        publisher=InMemoryEventPublisher(),
    )

    llm, model_id = build_llm()

    # TaskTool's spawner needs a reference back to the coordinator.
    # The coordinator instance doesn't exist yet, so we use a
    # forward-reference and patch it after construction. Real apps
    # could use a property-based pattern; this is the simplest.
    coordinator_ref: list[DemoCoordinator] = []

    class _CoordinatorProxy:
        async def spawn(self, **kwargs) -> str:
            return await coordinator_ref[0].spawn(**kwargs)

        async def status(self, thread_id: str) -> JSONObject:
            return await coordinator_ref[0].status(thread_id)

        async def send(self, thread_id: str, message: str) -> None:
            await coordinator_ref[0].send(thread_id, message)

        async def stop(self, thread_id: str) -> None:
            await coordinator_ref[0].stop(thread_id)

    proxy = _CoordinatorProxy()
    spawner = proxy
    # The three supervision tools ride alongside task(): task() hands the
    # parent a running sub-thread, these are what it does with one.
    supervision: list[Tool] = supervision_tools(proxy)

    # Main's task tool: can only delegate to researcher. The enum
    # constraint stops the model from inventing other subagent names.
    main_task_tool = TaskTool(
        spawner=spawner,
        subagent_choices=[RESEARCHER_AGENT_ID],
        subagent_descriptions={
            RESEARCHER_AGENT_ID: (
                "Fetches URLs, asks clarifying questions, and produces "
                "summaries. Can further delegate to the summarizer "
                "subagent. Best for multi-step research delegations."
            ),
        },
    )

    # Researcher's task tool: can only delegate to summarizer (a leaf
    # agent with no tools). Demonstrates 2-level nesting and the
    # publish-to-root chain.
    researcher_task_tool = TaskTool(
        spawner=spawner,
        subagent_choices=[SUMMARIZER_AGENT_ID],
        subagent_descriptions={
            SUMMARIZER_AGENT_ID: (
                "Condenses text or research notes into a compact "
                "structured summary. No tools — pure rewrite."
            ),
        },
    )

    main_agent = build_main_agent(llm, main_task_tool, supervision)
    researcher = build_researcher_agent(llm, researcher_task_tool, supervision)
    summarizer = build_summarizer_agent(llm)
    agents = {
        main_agent.id: main_agent,
        researcher.id: researcher,
        summarizer.id: summarizer,
    }
    temporal_address = os.getenv("ACTANT_TEMPORAL_ADDRESS", "localhost:27233")
    temporal_config = TemporalRuntimeConfig(address=temporal_address)
    client = await Client.connect(temporal_address, namespace=temporal_config.namespace)

    async def resolve_agent(agent_id: str, thread_id: str) -> AgentDefinition:
        return agents[agent_id]

    runtime = AgentRuntime(
        client=client,
        stores=stores,
        config=temporal_config,
        resolve_agent=resolve_agent,
        event_sink=DemoEvents(stores.threads, stores.publisher, list(agents)),
        event_source=stores.publisher,
        run_completion_handler=lambda completion: coordinator_ref[0].handle_run_completion(
            completion
        ),
    )
    worker_task = asyncio.create_task(runtime.run_worker(), name="actant-demo-worker")

    coordinator = DemoCoordinator(
        stores=stores,
        runtime=runtime,
        main_agent=agents[AGENT_ID],
        worker_task=worker_task,
        engine=engine,
        model_id=model_id,
    )
    coordinator_ref.append(coordinator)
    return coordinator
