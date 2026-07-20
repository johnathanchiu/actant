"""DemoCoordinator — composes actant's coordinator primitives with
demo-specific policy.

This is the canonical "how to build on actant" reference example
that `docs/coordinator-guide.md` points at. The shape is:

- Own the runtime stores + publisher + sub-thread registry.
- Build one main agent plus researcher and summarizer subagents globally.
- Wire AgentRuntime with the registry-aware factories from
  `actant.runtime.coordinator`.
- Implement `SubagentSpawner` for `TaskTool` to delegate work.
- Funnel ALL deferred resolutions (user-driven AND
  sub-thread-completion driven) through one `resolve_deferred_tool_call` call
  with state-divergence handling.

NO subclassing of any actant base class. Pure composition.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from actant.core import JSONObject
from actant.runtime import AgentRuntime, TemporalRuntimeConfig, TemporalRuntimeWorker
from actant.runtime.completion import RunCompletion
from actant.runtime.coordinator import (
    SubThreadLink,
    SubThreadRegistry,
    publishing_hooks_factory,
    resolve_deferred_tool_call,
)
from actant.runtime.exceptions import ToolResolutionStaleError
from actant.runtime.events import (
    AgentThreadHooks,
    PublishingStreamListener,
    PublishingThreadHooks,
    StreamListener,
)
from actant.runtime.stores.in_memory import InMemoryEventPublisher
from actant.runtime.stores.postgres import (
    SQLAlchemyMessageStore,
    SQLAlchemyRunStore,
    SQLAlchemyThreadStore,
    SQLAlchemyToolCallStore,
    create_schema,
)
from actant.runtime.types.threads import AgentThread
from actant.tools.calls import ToolCallStatus
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


# Subagent names the demo recognizes, mapped to their agent IDs in the
# runtime. The TaskTool's `subagent_choices` constrain which subset
# each parent can actually call.
_SUBAGENT_IDS = {
    RESEARCHER_AGENT_ID: RESEARCHER_AGENT_ID,
    SUMMARIZER_AGENT_ID: SUMMARIZER_AGENT_ID,
}


DEFAULT_DATABASE_URL = (
    "postgresql+asyncpg://actant:actant@localhost:55435/actant_demo"
)


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
        worker: TemporalRuntimeWorker,
        worker_task: asyncio.Task[None],
        engine: object,
        model_id: str,
        registry: SubThreadRegistry,
    ) -> None:
        self.stores = stores
        self.runtime = runtime
        self.worker = worker
        self.worker_task = worker_task
        self.engine = engine
        self.model_id = model_id
        self.registry = registry

    # ─── SubagentSpawner protocol (TaskTool.spawner) ────────────────

    async def spawn(
        self,
        *,
        name: str,
        message: str,
        context: JSONObject,
        parent_thread_id: str,
        parent_tool_call_id: str,
    ) -> None:
        sub_agent_id = _SUBAGENT_IDS.get(name)
        if sub_agent_id is None:
            raise ValueError(f"unknown subagent name: {name!r}")
        # The parent's agent_id is derivable without a store lookup:
        # if the parent thread is a registered sub-thread, the registry
        # knows its agent; otherwise it's the one top-level agent
        # (AGENT_ID). Same single-source-of-truth shape a production
        # coordinator would use, backed by the in-memory registry
        # instead of a store query.
        parent_link = self.registry.get(parent_thread_id)
        parent_agent_id = (
            parent_link.sub_agent_id if parent_link is not None else AGENT_ID
        )
        sub_thread_id = f"sub_{uuid.uuid4().hex[:10]}"
        link = SubThreadLink(
            sub_thread_id=sub_thread_id,
            parent_thread_id=parent_thread_id,
            parent_tool_call_id=parent_tool_call_id,
            sub_agent_id=sub_agent_id,
            subagent_name=name,
            metadata={"parent_agent_id": parent_agent_id},
        )
        # Register BEFORE send_message so the hooks_factory sees the
        # link synchronously and wires dual-publish from the very
        # first event.
        self.registry.register(link)
        # Persist parent metadata onto the sub-thread row too so
        # /api/threads/:id/sub_threads can report the mapping after
        # process restart.
        thread = await self.stores.threads.get_or_create(
            sub_agent_id, sub_thread_id
        )
        await self.stores.threads.update(
            AgentThread(
                id=thread.id,
                agent_id=thread.agent_id,
                status=thread.status,
                turn_count=thread.turn_count,
                active_run_id=thread.active_run_id,
                parent_thread_id=parent_thread_id,
                parent_tool_call_id=parent_tool_call_id,
            )
        )
        composed = message
        if context:
            composed = (
                f"{message}\n\nContext from caller:\n"
                f"```json\n{json.dumps(context, indent=2)}\n```"
            )
        await self.runtime.send_message(sub_agent_id, sub_thread_id, composed)

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

        Derives ``agent_id`` from the registry: if ``thread_id`` is a
        registered sub-thread, the wait belongs to the sub-agent (e.g.
        researcher's ask_user); otherwise it's a main-thread wait.

        Funneled through `resolve_deferred_tool_call` for state reconciliation —
        if Temporal lost the activity, `ToolResolutionStaleError`
        bubbles up (route layer turns it into 409 Conflict)."""
        link = self.registry.get(thread_id)
        agent_id = link.sub_agent_id if link is not None else AGENT_ID
        await resolve_deferred_tool_call(
            self.runtime,
            agent_id=agent_id,
            thread_id=thread_id,
            tool_call_id=tool_call_id,
            approved=approved,
            answer=answer,
            payload=payload,
        )

    async def handle_run_completion(self, completion: RunCompletion) -> None:
        """Resolve a parent task after a persisted child run completes.

        This handler runs inside Temporal's retryable ``finalize_run`` activity.
        It derives linkage and output from stores rather than hooks or process
        memory, so worker restart does not orphan the parent's parked call.
        """
        thread = await self.stores.threads.get(
            completion.agent_id, completion.thread_id
        )
        if thread.parent_tool_call_id is None or thread.parent_thread_id is None:
            return

        parent_call = await self.stores.tool_calls.get(thread.parent_tool_call_id)
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
        subagent = parent_call.args.get("subagent")
        envelope = {
            "text": text,
            "subagent": subagent if isinstance(subagent, str) else completion.agent_id,
        }
        try:
            await resolve_deferred_tool_call(
                self.runtime,
                agent_id=parent_call.agent_id,
                thread_id=thread.parent_thread_id,
                tool_call_id=parent_call.id,
                approved=completion.succeeded,
                answer=json.dumps(envelope),
            )
        except ToolResolutionStaleError:
            # The helper already reconciled the projection. Treat stale parent
            # activity identity as a completed repair, not a retryable failure.
            pass
        finally:
            self.registry.pop(completion.thread_id)

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


def _find_root_thread_id(registry: SubThreadRegistry, link: SubThreadLink) -> str:
    """Walk the sub-thread chain up to the root (a thread NOT in the
    registry). For 1-level nesting the root is the immediate parent;
    for N-level it's the topmost ancestor."""
    current = link
    while True:
        parent = registry.get(current.parent_thread_id)
        if parent is None:
            return current.parent_thread_id
        current = parent


async def _restore_subthread_registry(
    stores: _DemoStores,
    registry: SubThreadRegistry,
    agent_ids: list[str],
) -> None:
    """Rebuild live parent/child links from durable projections."""
    for agent_id in agent_ids:
        for thread in await stores.threads.list_for_agent(agent_id):
            if thread.parent_thread_id is None or thread.parent_tool_call_id is None:
                continue
            try:
                parent_call = await stores.tool_calls.get(thread.parent_tool_call_id)
            except KeyError:
                continue
            if parent_call.status is not ToolCallStatus.WAITING:
                continue
            subagent = parent_call.args.get("subagent")
            registry.register(
                SubThreadLink(
                    sub_thread_id=thread.id,
                    parent_thread_id=thread.parent_thread_id,
                    parent_tool_call_id=parent_call.id,
                    sub_agent_id=thread.agent_id,
                    subagent_name=(
                        subagent if isinstance(subagent, str) else thread.agent_id
                    ),
                    metadata={"parent_agent_id": parent_call.agent_id},
                )
            )


# ─── Build the coordinator ──────────────────────────────────────────


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
    registry = SubThreadRegistry()

    # TaskTool's spawner needs a reference back to the coordinator.
    # The coordinator instance doesn't exist yet, so we use a
    # forward-reference and patch it after construction. Real apps
    # could use a property-based pattern; this is the simplest.
    coordinator_ref: list[DemoCoordinator] = []

    class _CoordinatorProxy:
        async def spawn(self, **kwargs):
            return await coordinator_ref[0].spawn(**kwargs)

    spawner = _CoordinatorProxy()

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

    main_agent = build_main_agent(llm, main_task_tool)
    researcher = build_researcher_agent(llm, researcher_task_tool)
    summarizer = build_summarizer_agent(llm)
    agents = {
        main_agent.id: main_agent,
        researcher.id: researcher,
        summarizer.id: summarizer,
    }
    await _restore_subthread_registry(stores, registry, list(agents))

    # Hook + listener factories. For sub-threads, both dual-publish to
    # the ROOT thread (not just the immediate parent) so the main
    # thread's SSE subscriber sees every descendant event regardless
    # of nesting depth. Durable parent resolution is handled separately
    # by the retryable run_completion_handler below.
    base_hooks = publishing_hooks_factory(stores.publisher, registry=registry)

    def hooks_factory(thread: AgentThread) -> AgentThreadHooks:
        link = registry.get(thread.id)
        if link is None:
            return base_hooks(thread)
        root_id = _find_root_thread_id(registry, link)
        return PublishingThreadHooks(
            thread.id,
            publisher=stores.publisher,
            parent_channel=f"thread:{root_id}",
            parent_metadata={
                "parent_thread_id": link.parent_thread_id,
                "parent_tool_call_id": link.parent_tool_call_id,
                "subagent": link.subagent_name,
            },
        )

    def listener_factory(thread: AgentThread) -> StreamListener:
        link = registry.get(thread.id)
        if link is None:
            return PublishingStreamListener(thread.id, publisher=stores.publisher)
        root_id = _find_root_thread_id(registry, link)
        return PublishingStreamListener(
            thread.id,
            publisher=stores.publisher,
            parent_channel=f"thread:{root_id}",
            parent_metadata={
                "parent_thread_id": link.parent_thread_id,
                "parent_tool_call_id": link.parent_tool_call_id,
                "subagent": link.subagent_name,
            },
        )

    temporal_address = os.getenv("ACTANT_TEMPORAL_ADDRESS", "localhost:27233")
    temporal_config = TemporalRuntimeConfig(address=temporal_address)
    runtime = AgentRuntime(
        stores=stores,
        agents=agents,
        hooks_factory=hooks_factory,
        listener_factory=listener_factory,
        temporal=temporal_config,
    )
    worker = TemporalRuntimeWorker(
        stores=stores,
        agents=agents,
        config=temporal_config,
        hooks_factory=hooks_factory,
        listener_factory=listener_factory,
        run_completion_handler=lambda completion: coordinator_ref[0].handle_run_completion(
            completion
        ),
    )
    worker_task = asyncio.create_task(worker.run(), name="actant-demo-worker")

    coordinator = DemoCoordinator(
        stores=stores,
        runtime=runtime,
        worker=worker,
        worker_task=worker_task,
        engine=engine,
        model_id=model_id,
        registry=registry,
    )
    coordinator_ref.append(coordinator)
    return coordinator
