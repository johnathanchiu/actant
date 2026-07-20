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
from actant.llm.messages import Message
from actant.runtime import AgentRuntime, TemporalRuntimeConfig, TemporalRuntimeWorker
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
from actant.tools.base import ToolResult
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
        sub_thread_text: dict[str, str],
    ) -> None:
        self.stores = stores
        self.runtime = runtime
        self.worker = worker
        self.worker_task = worker_task
        self.engine = engine
        self.model_id = model_id
        self.registry = registry
        # Tracks the most-recent assistant text per sub-thread so we
        # can use it as the resolved `answer` when the sub completes.
        # In-memory only — survives within a single process lifetime.
        self._sub_thread_text = sub_thread_text

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

    async def _on_sub_thread_complete(
        self,
        sub_thread_id: str,
        *,
        success: bool,
        reason: str,
        message: str,
    ) -> None:
        """Fired from a sub-thread's hooks on_complete. Resolves the
        parent's parked `task()` tool call with the sub-thread's final
        assistant text wrapped in a JSON envelope (TaskTool.on_resolve
        does json.loads on the answer)."""
        link = self.registry.pop(sub_thread_id)
        if link is None:
            return
        text = self._sub_thread_text.pop(sub_thread_id, "") or message or reason
        if not text.strip():
            text = "(sub-agent returned no text)"
        envelope = {"text": text, "subagent": link.subagent_name}
        # Parent might be the main agent OR another sub-agent (when
        # researcher delegates to summarizer). The link's metadata
        # records which (stamped at spawn time).
        parent_agent_id = str(link.metadata.get("parent_agent_id") or AGENT_ID)
        try:
            await resolve_deferred_tool_call(
                self.runtime,
                agent_id=parent_agent_id,
                thread_id=link.parent_thread_id,
                tool_call_id=link.parent_tool_call_id,
                approved=success,
                answer=json.dumps(envelope),
            )
        except ToolResolutionStaleError:
            # Parent's activity is gone (workflow cancelled / Temporal
            # reset). resolve_deferred_tool_call already marked the store FAILED;
            # nothing more we can do.
            pass

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


# ─── Hooks: thin subclass that notifies the coordinator ─────────────


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


class _SubThreadHooks(PublishingThreadHooks):
    """PublishingThreadHooks for a sub-thread that ALSO notifies the
    DemoCoordinator when (a) an assistant message lands (to remember
    the last text) and (b) the run completes (to resolve the parent's
    deferred tool call).

    Dual-publishes to the ROOT thread (not just the immediate parent)
    so the main thread's SSE subscriber sees every descendant event,
    no matter how deep the sub-agent nesting goes. The event's
    ``parent_thread_id`` metadata still names the IMMEDIATE parent —
    that's what drives UI nesting via the sub_threads map.
    """

    def __init__(
        self,
        thread_id: str,
        publisher: InMemoryEventPublisher,
        link: SubThreadLink,
        root_thread_id: str,
        coordinator: DemoCoordinator,
    ) -> None:
        super().__init__(
            thread_id,
            publisher=publisher,
            parent_channel=f"thread:{root_thread_id}",
            parent_metadata={
                "parent_thread_id": link.parent_thread_id,
                "parent_tool_call_id": link.parent_tool_call_id,
                "subagent": link.subagent_name,
            },
        )
        self._link = link
        self._coordinator = coordinator

    async def on_assistant_message(self, message: Message) -> None:
        await super().on_assistant_message(message)
        if isinstance(message.content, str) and message.content.strip():
            self._coordinator._sub_thread_text[self._link.sub_thread_id] = message.content

    async def on_complete(self, success: bool, reason: str, message: str) -> None:
        await super().on_complete(success, reason, message)
        await self._coordinator._on_sub_thread_complete(
            self._link.sub_thread_id, success=success, reason=reason, message=message
        )

    async def on_error(self, error: Exception) -> None:
        await super().on_error(error)
        # A failed sub-thread must still resolve the parent's deferred
        # task() call (with failure) so the parent doesn't hang in
        # WAITING.
        await self._coordinator._on_sub_thread_complete(
            self._link.sub_thread_id,
            success=False,
            reason="error",
            message=str(error),
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
    sub_thread_text: dict[str, str] = {}

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

    # Hook + listener factories. For sub-threads, both dual-publish to
    # the ROOT thread (not just the immediate parent) so the main
    # thread's SSE subscriber sees every descendant event regardless
    # of nesting depth. Hooks additionally drive the resolve flow on
    # completion via _SubThreadHooks.
    base_hooks = publishing_hooks_factory(stores.publisher, registry=registry)

    def hooks_factory(thread: AgentThread) -> AgentThreadHooks:
        link = registry.get(thread.id)
        if link is None:
            return base_hooks(thread)
        root_id = _find_root_thread_id(registry, link)
        return _SubThreadHooks(
            thread.id, stores.publisher, link, root_id, coordinator_ref[0]
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
        sub_thread_text=sub_thread_text,
    )
    coordinator_ref.append(coordinator)
    return coordinator


# Silence unused-import noise. These exist for the protocol surface
# only — the type names are exported to callers via re-export.
_ToolResult = ToolResult
_StreamListener = StreamListener
