"""Postgres-backed :class:`~actant.infra.sandbox.store.SandboxStore`: the ``actant_sandboxes``
table and its store. The table sits in the sandbox layer's own metadata; actant's migrations
create it alongside the runtime's tables."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, MetaData, Text, delete, func, select, update
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from actant.infra.sandbox.base import SandboxSpec
from actant.infra.sandbox.store import SandboxRecord, spec_from_json, spec_to_json


class SandboxBase(DeclarativeBase):
    pass


SANDBOX_METADATA: MetaData = SandboxBase.metadata


class ActantSandboxModel(SandboxBase):
    """A sandbox (:mod:`actant.infra.sandbox.store`): its backend's live id and its spec."""

    __tablename__ = "actant_sandboxes"

    sandbox_id: Mapped[str] = mapped_column(Text, primary_key=True)
    provider_id: Mapped[str] = mapped_column(Text, nullable=False)
    spec: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SQLAlchemySandboxStore:
    """``actant.infra.sandbox.store.SandboxStore`` over ``actant_sandboxes``."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def get(self, sandbox_id: str) -> SandboxRecord | None:
        async with self.session_factory() as session:
            return _sandbox_record(await session.get(ActantSandboxModel, sandbox_id))

    async def claim(
        self, sandbox_id: str, *, expected: str | None, provider_id: str, spec: SandboxSpec
    ) -> SandboxRecord | None:
        values = {"provider_id": provider_id, "spec": spec_to_json(spec)}
        row = ActantSandboxModel.sandbox_id == sandbox_id
        async with self.session_factory() as session:
            async with session.begin():
                if expected is None:
                    await session.execute(
                        insert(ActantSandboxModel)
                        .values(sandbox_id=sandbox_id, **values)
                        .on_conflict_do_nothing(index_elements=[ActantSandboxModel.sandbox_id])
                    )
                else:
                    # The row lock makes a racing claim re-check ``expected`` after this commits.
                    await session.execute(
                        update(ActantSandboxModel)
                        .where(row, ActantSandboxModel.provider_id == expected)
                        .values(**values, updated_at=datetime.now(UTC))
                    )
                found = await session.execute(select(ActantSandboxModel).where(row))
                return _sandbox_record(found.scalar_one_or_none())

    async def forget(self, sandbox_id: str) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(
                    delete(ActantSandboxModel).where(ActantSandboxModel.sandbox_id == sandbox_id)
                )


def _sandbox_record(row: ActantSandboxModel | None) -> SandboxRecord | None:
    if row is None:
        return None
    return SandboxRecord(row.sandbox_id, row.provider_id, spec_from_json(row.spec))


__all__ = ["SANDBOX_METADATA", "ActantSandboxModel", "SQLAlchemySandboxStore", "SandboxBase"]
