"""SQLAlchemy schema for Actant's queryable runtime projections.

Temporal owns coordination. These tables expose thread, run, transcript, and
tool-call state to applications without replaying workflow history.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class ActantRuntimeBase(DeclarativeBase):
    pass


ACTANT_RUNTIME_METADATA = ActantRuntimeBase.metadata


async def create_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(ACTANT_RUNTIME_METADATA.create_all)


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
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
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
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ActantMessageModel(ActantRuntimeBase):
    """Header for one logical message; its body lives in ordered parts."""

    __tablename__ = "actant_messages"

    message_id: Mapped[str] = mapped_column(Text, primary_key=True)
    agent_id: Mapped[str] = mapped_column(Text, nullable=False)
    thread_id: Mapped[str] = mapped_column(Text, nullable=False)
    turn_id: Mapped[str | None] = mapped_column(Text)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    parts: Mapped[list["ActantMessagePartModel"]] = relationship(
        "ActantMessagePartModel",
        cascade="all, delete-orphan",
        order_by="ActantMessagePartModel.part_index",
        lazy="selectin",
    )


class ActantMessagePartModel(ActantRuntimeBase):
    """One structured text, thinking, tool-call, or tool-result message part."""

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
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    status: Mapped[str] = mapped_column(Text, nullable=False)
    prompt: Mapped[str | None] = mapped_column(Text)
    wait_request: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    result: Mapped[object | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


__all__ = [
    "ACTANT_RUNTIME_METADATA",
    "ActantMessageModel",
    "ActantMessagePartModel",
    "ActantRunModel",
    "ActantRuntimeBase",
    "ActantThreadModel",
    "ActantToolCallModel",
    "create_schema",
]
