"""SQLAlchemy models for Actant runtime storage.

These models mirror the canonical Postgres runtime tables. Applications
can use them directly in Alembic metadata while still implementing the
store protocols with their preferred session/transaction wiring.

Coordination lives in Temporal. The models here are projection-only.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, selectinload

from actant.core import JSONObject, new_id
from actant.llm.messages import Message, Role
from actant.runtime.session import message_to_parts, parts_to_messages
from actant.runtime.types.session import MessagePart, PartKind, WaitStatus
from actant.runtime.types.threads import (
    AgentRun,
    AgentThread,
    MessageRecord,
    RunStatus,
    ThreadStatus,
)
from actant.tools.calls import ToolCallRecord, ToolCallStatus

logger = logging.getLogger(__name__)


class ActantRuntimeBase(DeclarativeBase):
    pass


ACTANT_RUNTIME_METADATA = ActantRuntimeBase.metadata


async def create_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(ACTANT_RUNTIME_METADATA.create_all)


class ActantThreadModel(ActantRuntimeBase):
    __tablename__ = "actant_threads"

    agent_id: Mapped[str] = mapped_column(Text, primary_key=True)
    thread_id: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    active_run_id: Mapped[str | None] = mapped_column(Text)
    parent_thread_id: Mapped[str | None] = mapped_column(Text)
    parent_turn_id: Mapped[str | None] = mapped_column(Text)
    parent_tool_call_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ActantRunModel(ActantRuntimeBase):
    __tablename__ = "actant_runs"

    run_id: Mapped[str] = mapped_column(Text, primary_key=True)
    agent_id: Mapped[str] = mapped_column(Text, nullable=False)
    thread_id: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    max_turns: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("25"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ActantMessageModel(ActantRuntimeBase):
    """Header row for a logical message (one turn-of-conversation).

    The body — text, thinking trace, tool calls, tool results, asset
    blocks — lives in :class:`ActantMessagePartModel` rows linked by
    ``message_id``. Pydantic-ai aligned: each part is one logical
    section of the message.
    """

    __tablename__ = "actant_messages"

    message_id: Mapped[str] = mapped_column(Text, primary_key=True)
    agent_id: Mapped[str] = mapped_column(Text, nullable=False)
    thread_id: Mapped[str] = mapped_column(Text, nullable=False)
    turn_id: Mapped[str | None] = mapped_column(Text)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    parts: Mapped[list["ActantMessagePartModel"]] = relationship(
        "ActantMessagePartModel",
        cascade="all, delete-orphan",
        order_by="ActantMessagePartModel.part_index",
        lazy="selectin",
    )


class ActantMessagePartModel(ActantRuntimeBase):
    """Body row — one part within a message.

    Columns split by which kinds use them:
      * ``content``: text-bearing kinds (text, thinking, system_prompt,
        and user_prompt when string-only).
      * ``content_blocks``: user_prompt + tool_result when multimodal.
      * ``reasoning_items``: thinking only.
      * ``signature``: thinking continuation OR Gemini tool_call thought signature.
      * ``tool_call_id`` / ``tool_name`` / ``args``: tool_call.
      * ``tool_call_id`` / ``result``: tool_result.
    """

    __tablename__ = "actant_message_parts"

    message_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("actant_messages.message_id", ondelete="CASCADE"),
        primary_key=True,
    )
    part_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str | None] = mapped_column(Text)
    content_blocks: Mapped[list[dict[str, object]] | None] = mapped_column(JSONB)
    signature: Mapped[str | None] = mapped_column(Text)
    reasoning_items: Mapped[list[object] | None] = mapped_column(JSONB)
    tool_call_id: Mapped[str | None] = mapped_column(Text)
    tool_name: Mapped[str | None] = mapped_column(Text)
    args: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    result: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    wait_status: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("ix_actant_message_parts_tool_call_id", "tool_call_id"),)


class ActantToolCallModel(ActantRuntimeBase):
    __tablename__ = "actant_tool_calls"

    tool_call_id: Mapped[str] = mapped_column(Text, primary_key=True)
    group_id: Mapped[str] = mapped_column(Text, nullable=False)
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    agent_id: Mapped[str] = mapped_column(Text, nullable=False)
    thread_id: Mapped[str] = mapped_column(Text, nullable=False)
    turn_id: Mapped[str] = mapped_column(Text, nullable=False)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    args: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    status: Mapped[str] = mapped_column(Text, nullable=False)
    prompt: Mapped[str | None] = mapped_column(Text)
    wait_request: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    result: Mapped[object | None] = mapped_column(JSONB)
    # Stamped by ``await_external_resolution`` so external callers can
    # complete the deferred tool's activity via
    # ``client.complete_activity_by_id``. Null when the tool didn't
    # take the WAIT path.
    temporal_workflow_id: Mapped[str | None] = mapped_column(Text)
    temporal_activity_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class SQLAlchemyThreadStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def get_or_create(self, agent_id: str, thread_id: str) -> AgentThread:
        async with self.session_factory() as session:
            async with session.begin():
                thread = await _get_or_create_thread(session, agent_id, thread_id)
                return _thread(thread)

    async def get(self, agent_id: str, thread_id: str) -> AgentThread:
        async with self.session_factory() as session:
            thread = await _get_thread(session, agent_id, thread_id)
            if thread is None:
                raise KeyError(thread_id)
            return _thread(thread)

    async def update(self, thread: AgentThread) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                model = await _get_or_create_thread(session, thread.agent_id, thread.id)
                model.status = thread.status.value
                model.turn_count = thread.turn_count
                model.active_run_id = thread.active_run_id
                model.parent_thread_id = thread.parent_thread_id
                model.parent_turn_id = thread.parent_turn_id
                model.parent_tool_call_id = thread.parent_tool_call_id
                model.updated_at = datetime.now(UTC)

    async def list_for_agent(self, agent_id: str) -> list[AgentThread]:
        async with self.session_factory() as session:
            rows = await session.execute(
                select(ActantThreadModel)
                .where(ActantThreadModel.agent_id == agent_id)
                .order_by(ActantThreadModel.updated_at.desc())
            )
            return [_thread(row) for row in rows.scalars().all()]


class SQLAlchemyRunStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def create(
        self, agent_id: str, thread_id: str, *, run_id: str, max_turns: int
    ) -> AgentRun:
        run = AgentRun(id=run_id, agent_id=agent_id, thread_id=thread_id, max_turns=max_turns)
        async with self.session_factory() as session:
            async with session.begin():
                session.add(
                    ActantRunModel(
                        run_id=run.id,
                        agent_id=run.agent_id,
                        thread_id=run.thread_id,
                        status=run.status.value,
                        turn_count=run.turn_count,
                        max_turns=run.max_turns,
                    )
                )
        return run

    async def get(self, run_id: str) -> AgentRun:
        async with self.session_factory() as session:
            run = await session.get(ActantRunModel, run_id)
            if run is None:
                raise KeyError(run_id)
            return _run(run)

    async def update(self, run: AgentRun) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                model = await session.get(ActantRunModel, run.id)
                if model is None:
                    raise KeyError(run.id)
                model.status = run.status.value
                model.turn_count = run.turn_count
                model.max_turns = run.max_turns
                model.updated_at = datetime.now(UTC)

    async def finish(self, run_id: str, status: RunStatus) -> None:
        # Idempotent: missing run = nothing to finalize. Temporal can
        # redeliver ``finalize_run`` after the row has been cleaned up
        # (test teardown, manual cleanup, retention policy); raising
        # KeyError would push the activity into endless retries.
        async with self.session_factory() as session:
            async with session.begin():
                model = await session.get(ActantRunModel, run_id)
                if model is None:
                    return
                model.status = status.value
                model.updated_at = datetime.now(UTC)


class SQLAlchemyMessageStore:
    """Parts-based message persistence.

    Each ``append_*`` call writes one ``ActantMessageModel`` header
    plus N ``ActantMessagePartModel`` rows, decomposed via
    :func:`message_to_parts`. ``list_for_thread`` reads the headers
    with their parts eagerly loaded and assembles ``Message`` objects
    via :func:`parts_to_messages`. Multimodal content (text + asset
    blocks) round-trips through the part rows' ``content_blocks``
    JSONB column.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def append_user(
        self,
        agent_id: str,
        thread_id: str,
        content: str | list[dict[str, object]],
    ) -> MessageRecord:
        return await self._append(agent_id, thread_id, None, Message(role="user", content=content))

    async def append_assistant(
        self, agent_id: str, thread_id: str, turn_id: str, message: Message
    ) -> MessageRecord:
        return await self._append(agent_id, thread_id, turn_id, message)

    async def append_assistant_with_tool_calls(
        self,
        agent_id: str,
        thread_id: str,
        turn_id: str,
        message: Message,
        tool_calls: Sequence[ToolCallRecord],
    ) -> MessageRecord:
        message_id = new_id("msg")
        parts = message_to_parts(message)
        async with self.session_factory() as session:
            async with session.begin():
                session.add(
                    ActantMessageModel(
                        message_id=message_id,
                        agent_id=agent_id,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        role=message.role,
                    )
                )
                for index, part in enumerate(parts):
                    session.add(_part_row(message_id, index, part))
                for tc in tool_calls:
                    session.add(
                        ActantToolCallModel(
                            tool_call_id=tc.id,
                            group_id=tc.group_id,
                            run_id=tc.run_id,
                            agent_id=tc.agent_id,
                            thread_id=tc.thread_id,
                            turn_id=tc.turn_id,
                            turn_index=tc.turn_index,
                            name=tc.name,
                            args=tc.args,
                            status=tc.status.value,
                            prompt=tc.prompt,
                            wait_request=tc.wait_request,
                            result=tc.result,
                        )
                    )
        return MessageRecord(message_id, agent_id, thread_id, message)

    async def append_tool_result(
        self,
        agent_id: str,
        thread_id: str,
        turn_id: str,
        tool_call_id: str,
        name: str,
        result: object,
    ) -> MessageRecord:
        # Idempotent on (agent_id, thread_id, tool_call_id). The
        # transcript should never have two tool messages for the same
        # call. We look up by the tool_result part's tool_call_id since
        # that's where the cross-message link lives.
        async with self.session_factory() as session:
            existing_message_id = (
                await session.scalars(
                    select(ActantMessagePartModel.message_id)
                    .join(
                        ActantMessageModel,
                        ActantMessageModel.message_id == ActantMessagePartModel.message_id,
                    )
                    .where(
                        ActantMessageModel.agent_id == agent_id,
                        ActantMessageModel.thread_id == thread_id,
                        ActantMessageModel.role == "tool",
                        ActantMessagePartModel.kind == PartKind.TOOL_RESULT.value,
                        ActantMessagePartModel.tool_call_id == tool_call_id,
                    )
                    .limit(1)
                )
            ).first()
            if existing_message_id is not None:
                logger.warning(
                    "actant.message_store.duplicate_tool_result_skipped "
                    "agent=%s thread=%s tool_call_id=%s",
                    agent_id,
                    thread_id,
                    tool_call_id,
                )
                row = (
                    await session.scalars(
                        select(ActantMessageModel)
                        .options(selectinload(ActantMessageModel.parts))
                        .where(ActantMessageModel.message_id == existing_message_id)
                    )
                ).one()
                return MessageRecord(
                    row.message_id,
                    agent_id,
                    thread_id,
                    _message_from_header(row),
                )

        message_id = new_id("msg")
        async with self.session_factory() as session:
            async with session.begin():
                session.add(
                    ActantMessageModel(
                        message_id=message_id,
                        agent_id=agent_id,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        role="tool",
                    )
                )
                session.add(_tool_result_part(message_id, tool_call_id, name, result))
        return MessageRecord(
            message_id,
            agent_id,
            thread_id,
            Message(
                role="tool",
                content=_tool_result_content_for_message(result),
                tool_call_id=tool_call_id,
                name=name,
            ),
        )

    async def list_for_thread(self, agent_id: str, thread_id: str) -> list[Message]:
        async with self.session_factory() as session:
            rows = (
                await session.scalars(
                    select(ActantMessageModel)
                    .options(selectinload(ActantMessageModel.parts))
                    .where(
                        ActantMessageModel.agent_id == agent_id,
                        ActantMessageModel.thread_id == thread_id,
                    )
                    .order_by(ActantMessageModel.created_at, ActantMessageModel.message_id)
                )
            ).all()
            return [_message_from_header(row) for row in rows]

    async def _append(
        self, agent_id: str, thread_id: str, turn_id: str | None, message: Message
    ) -> MessageRecord:
        message_id = new_id("msg")
        parts = message_to_parts(message)
        async with self.session_factory() as session:
            async with session.begin():
                session.add(
                    ActantMessageModel(
                        message_id=message_id,
                        agent_id=agent_id,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        role=message.role,
                    )
                )
                for index, part in enumerate(parts):
                    session.add(_part_row(message_id, index, part))
        return MessageRecord(message_id, agent_id, thread_id, message)


class SQLAlchemyToolCallStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def save(self, tc: ToolCallRecord) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                session.add(
                    ActantToolCallModel(
                        tool_call_id=tc.id,
                        group_id=tc.group_id,
                        run_id=tc.run_id,
                        agent_id=tc.agent_id,
                        thread_id=tc.thread_id,
                        turn_id=tc.turn_id,
                        turn_index=tc.turn_index,
                        name=tc.name,
                        args=tc.args,
                        status=tc.status.value,
                        prompt=tc.prompt,
                        wait_request=tc.wait_request,
                        result=tc.result,
                    )
                )

    async def update_status(
        self,
        tc_id: str,
        status: ToolCallStatus,
        *,
        result: object = None,
        prompt: str | None = None,
        wait_request: JSONObject | None = None,
    ) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                tc = await session.get(ActantToolCallModel, tc_id)
                if tc is None:
                    raise KeyError(tc_id)
                tc.status = status.value
                if result is not None:
                    tc.result = result
                if prompt is not None:
                    tc.prompt = prompt
                if wait_request is not None:
                    tc.wait_request = cast(dict[str, object], wait_request)
                tc.updated_at = datetime.now(UTC)

    async def set_temporal_handle(
        self,
        tc_id: str,
        *,
        workflow_id: str,
        activity_id: str,
    ) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                tc = await session.get(ActantToolCallModel, tc_id)
                if tc is None:
                    raise KeyError(tc_id)
                tc.temporal_workflow_id = workflow_id
                tc.temporal_activity_id = activity_id
                tc.updated_at = datetime.now(UTC)

    async def get(self, tc_id: str) -> ToolCallRecord:
        async with self.session_factory() as session:
            tc = await session.get(ActantToolCallModel, tc_id)
            if tc is None:
                raise KeyError(tc_id)
            return _tool_call(tc)

    async def get_group(self, group_id: str) -> list[ToolCallRecord]:
        async with self.session_factory() as session:
            rows = (
                await session.scalars(
                    select(ActantToolCallModel)
                    .where(ActantToolCallModel.group_id == group_id)
                    .order_by(ActantToolCallModel.tool_call_id)
                )
            ).all()
            return [_tool_call(row) for row in rows]

    async def get_by_run(self, run_id: str) -> list[ToolCallRecord]:
        async with self.session_factory() as session:
            rows = (
                await session.scalars(
                    select(ActantToolCallModel)
                    .where(ActantToolCallModel.run_id == run_id)
                    .order_by(ActantToolCallModel.created_at)
                )
            ).all()
            return [_tool_call(row) for row in rows]

    async def get_by_thread_and_turn(self, thread_id: str, turn_id: str) -> list[ToolCallRecord]:
        async with self.session_factory() as session:
            rows = (
                await session.scalars(
                    select(ActantToolCallModel)
                    .where(
                        ActantToolCallModel.thread_id == thread_id,
                        ActantToolCallModel.turn_id == turn_id,
                    )
                    .order_by(ActantToolCallModel.created_at)
                )
            ).all()
            return [_tool_call(row) for row in rows]

    async def get_open_for_thread(self, agent_id: str, thread_id: str) -> list[ToolCallRecord]:
        async with self.session_factory() as session:
            rows = (
                await session.scalars(
                    select(ActantToolCallModel)
                    .where(
                        ActantToolCallModel.agent_id == agent_id,
                        ActantToolCallModel.thread_id == thread_id,
                        ActantToolCallModel.status.in_(
                            [
                                ToolCallStatus.REQUESTED.value,
                                ToolCallStatus.RUNNING.value,
                                ToolCallStatus.WAITING.value,
                            ]
                        ),
                    )
                    .order_by(ActantToolCallModel.created_at)
                )
            ).all()
            return [_tool_call(row) for row in rows]


class SQLAlchemyEventPublisher:
    async def publish(self, channel: str, event: JSONObject) -> None:
        del channel, event

    async def subscribe(self, channel: str) -> AsyncIterator[JSONObject]:
        del channel
        if False:
            yield {}


class SQLAlchemyRuntimeStores:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.threads = SQLAlchemyThreadStore(session_factory)
        self.runs = SQLAlchemyRunStore(session_factory)
        self.messages = SQLAlchemyMessageStore(session_factory)
        self.tool_calls = SQLAlchemyToolCallStore(session_factory)
        self.publisher = SQLAlchemyEventPublisher()


async def _get_thread(
    session: AsyncSession, agent_id: str, thread_id: str
) -> ActantThreadModel | None:
    return await session.get(
        ActantThreadModel,
        {"agent_id": agent_id, "thread_id": thread_id},
    )


async def _get_or_create_thread(
    session: AsyncSession, agent_id: str, thread_id: str
) -> ActantThreadModel:
    thread = await _get_thread(session, agent_id, thread_id)
    if thread is not None:
        return thread
    thread = ActantThreadModel(
        agent_id=agent_id,
        thread_id=thread_id,
        status=ThreadStatus.IDLE.value,
        turn_count=0,
    )
    session.add(thread)
    await session.flush()
    return thread


def _thread(row: ActantThreadModel) -> AgentThread:
    return AgentThread(
        id=row.thread_id,
        agent_id=row.agent_id,
        status=ThreadStatus(row.status),
        turn_count=row.turn_count,
        active_run_id=row.active_run_id,
        parent_thread_id=row.parent_thread_id,
        parent_turn_id=row.parent_turn_id,
        parent_tool_call_id=row.parent_tool_call_id,
    )


def _run(row: ActantRunModel) -> AgentRun:
    return AgentRun(
        id=row.run_id,
        agent_id=row.agent_id,
        thread_id=row.thread_id,
        status=RunStatus(row.status),
        turn_count=row.turn_count,
        max_turns=row.max_turns,
    )


def _message_from_header(row: ActantMessageModel) -> Message:
    """Reassemble a single ``Message`` from a header row + its parts."""
    role = row.role
    parts = [_message_part_from_row(p) for p in row.parts]

    if role == "tool":
        for part in parts:
            if part.kind is PartKind.TOOL_RESULT:
                content: str | list[dict[str, object]]
                if part.content_blocks:
                    content = part.content_blocks
                elif part.result is not None:
                    content = json.dumps(part.result)
                else:
                    content = part.content or ""
                return Message(
                    role="tool",
                    content=content,
                    tool_call_id=part.tool_call_id,
                    name=part.tool_name,
                )
        return Message(role=cast(Role, role), content="")

    messages = parts_to_messages(parts)
    if messages:
        return messages[0]
    return Message(role=cast(Role, role), content="")


def _part_row(message_id: str, part_index: int, part: MessagePart) -> ActantMessagePartModel:
    return ActantMessagePartModel(
        message_id=message_id,
        part_index=part_index,
        kind=part.kind.value,
        content=part.content,
        content_blocks=part.content_blocks,
        signature=part.signature,
        reasoning_items=part.reasoning_items,
        tool_call_id=part.tool_call_id,
        tool_name=part.tool_name,
        args=part.args,
        result=part.result,
        wait_status=part.wait_status.value if part.wait_status is not None else None,
    )


def _message_part_from_row(row: ActantMessagePartModel) -> MessagePart:
    return MessagePart(
        kind=PartKind(row.kind),
        content=row.content,
        content_blocks=row.content_blocks,
        signature=row.signature,
        reasoning_items=row.reasoning_items,
        tool_call_id=row.tool_call_id,
        tool_name=row.tool_name,
        args=cast(JSONObject, row.args) if row.args is not None else None,
        result=row.result,
        wait_status=WaitStatus(row.wait_status) if row.wait_status is not None else None,
    )


def _tool_result_part(
    message_id: str, tool_call_id: str, name: str, result: object
) -> ActantMessagePartModel:
    """Build a TOOL_RESULT part row from the raw result a tool returned."""
    blocks: list[dict[str, object]] | None = None
    if isinstance(result, dict):
        candidate = result.get("content_blocks")
        if isinstance(candidate, list):
            normalized = [b for b in candidate if isinstance(b, dict)]
            blocks = normalized or None
    return ActantMessagePartModel(
        message_id=message_id,
        part_index=0,
        kind=PartKind.TOOL_RESULT.value,
        content_blocks=blocks,
        result=result if isinstance(result, dict) else {"value": result},
        tool_call_id=tool_call_id,
        tool_name=name,
    )


def _tool_result_content_for_message(result: object) -> str | list[dict[str, object]]:
    if isinstance(result, dict):
        candidate = result.get("content_blocks")
        if isinstance(candidate, list):
            normalized = [b for b in candidate if isinstance(b, dict)]
            if normalized:
                return normalized
        return json.dumps(result)
    return json.dumps(result)


def _tool_call(row: ActantToolCallModel) -> ToolCallRecord:
    return ToolCallRecord(
        id=row.tool_call_id,
        group_id=row.group_id,
        run_id=row.run_id,
        agent_id=row.agent_id,
        thread_id=row.thread_id,
        turn_id=row.turn_id,
        turn_index=row.turn_index,
        name=row.name,
        args=cast(JSONObject, row.args),
        status=ToolCallStatus(row.status),
        prompt=row.prompt,
        wait_request=cast(JSONObject | None, row.wait_request),
        result=row.result,
        temporal_workflow_id=row.temporal_workflow_id,
        temporal_activity_id=row.temporal_activity_id,
    )


__all__ = [
    "ACTANT_RUNTIME_METADATA",
    "ActantMessageModel",
    "ActantMessagePartModel",
    "ActantRunModel",
    "ActantRuntimeBase",
    "ActantThreadModel",
    "ActantToolCallModel",
    "SQLAlchemyEventPublisher",
    "SQLAlchemyMessageStore",
    "SQLAlchemyRunStore",
    "SQLAlchemyRuntimeStores",
    "SQLAlchemyThreadStore",
    "SQLAlchemyToolCallStore",
    "create_schema",
]
