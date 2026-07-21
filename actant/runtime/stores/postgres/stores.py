"""SQLAlchemy implementations of Actant's projection-store protocols."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from actant.core import JSONObject, new_id
from actant.llm.messages import Message
from actant.runtime.session import message_to_parts
from actant.runtime.stores.postgres.conversion import (
    message_from_header,
    message_part_row,
    run_from_row,
    thread_from_row,
    tool_call_from_row,
    tool_result_content,
    tool_result_part_row,
)
from actant.runtime.stores.postgres.models import (
    ActantMessageModel,
    ActantMessagePartModel,
    ActantRunModel,
    ActantThreadModel,
    ActantToolCallModel,
)
from actant.runtime.types.session import PartKind
from actant.runtime.types.threads import (
    AgentRun,
    AgentThread,
    MessageRecord,
    RunStatus,
    ThreadStatus,
)
from actant.tools.calls import ToolCallRecord, ToolCallStatus

logger = logging.getLogger(__name__)


class SQLAlchemyThreadStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def get_or_create(self, agent_id: str, thread_id: str) -> AgentThread:
        async with self.session_factory() as session:
            async with session.begin():
                thread = await _get_or_create_thread(session, agent_id, thread_id)
                return thread_from_row(thread)

    async def get(self, agent_id: str, thread_id: str) -> AgentThread:
        async with self.session_factory() as session:
            thread = await _get_thread(session, agent_id, thread_id)
            if thread is None:
                raise KeyError(thread_id)
            return thread_from_row(thread)

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
            return [thread_from_row(row) for row in rows.scalars().all()]


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
            return run_from_row(run)

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
                    session.add(message_part_row(message_id, index, part))
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
                    message_from_header(row),
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
                session.add(tool_result_part_row(message_id, tool_call_id, name, result))
        return MessageRecord(
            message_id,
            agent_id,
            thread_id,
            Message(
                role="tool",
                content=tool_result_content(result),
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
            return [message_from_header(row) for row in rows]

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
                    session.add(message_part_row(message_id, index, part))
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

    async def finish_waiting(
        self,
        tc_id: str,
        status: ToolCallStatus,
        *,
        result: object,
    ) -> bool:
        async with self.session_factory() as session:
            async with session.begin():
                outcome = await session.execute(
                    update(ActantToolCallModel)
                    .where(
                        ActantToolCallModel.tool_call_id == tc_id,
                        ActantToolCallModel.status == ToolCallStatus.WAITING.value,
                    )
                    .values(
                        status=status.value,
                        result=result,
                        updated_at=datetime.now(UTC),
                    )
                )
                return cast(CursorResult[object], outcome).rowcount == 1

    async def get(self, tc_id: str) -> ToolCallRecord:
        async with self.session_factory() as session:
            tc = await session.get(ActantToolCallModel, tc_id)
            if tc is None:
                raise KeyError(tc_id)
            return tool_call_from_row(tc)

    async def get_group(self, group_id: str) -> list[ToolCallRecord]:
        async with self.session_factory() as session:
            rows = (
                await session.scalars(
                    select(ActantToolCallModel)
                    .where(ActantToolCallModel.group_id == group_id)
                    .order_by(ActantToolCallModel.tool_call_id)
                )
            ).all()
            return [tool_call_from_row(row) for row in rows]

    async def get_by_run(self, run_id: str) -> list[ToolCallRecord]:
        async with self.session_factory() as session:
            rows = (
                await session.scalars(
                    select(ActantToolCallModel)
                    .where(ActantToolCallModel.run_id == run_id)
                    .order_by(ActantToolCallModel.created_at)
                )
            ).all()
            return [tool_call_from_row(row) for row in rows]

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
            return [tool_call_from_row(row) for row in rows]

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
            return [tool_call_from_row(row) for row in rows]


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


__all__ = [
    "SQLAlchemyEventPublisher",
    "SQLAlchemyMessageStore",
    "SQLAlchemyRunStore",
    "SQLAlchemyRuntimeStores",
    "SQLAlchemyThreadStore",
    "SQLAlchemyToolCallStore",
]
