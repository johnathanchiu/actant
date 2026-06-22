from __future__ import annotations

from typing import cast

from sqlalchemy import Table
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from actant.runtime.stores.postgres.sqlalchemy import (
    ACTANT_RUNTIME_METADATA,
    ActantRunModel,
    ActantThreadModel,
    ActantToolCallModel,
    SQLAlchemyRuntimeStores,
)


def test_sqlalchemy_metadata_declares_runtime_tables() -> None:
    """Schema is projection-only — coordination tables removed."""
    assert set(ACTANT_RUNTIME_METADATA.tables) == {
        "actant_threads",
        "actant_runs",
        "actant_messages",
        "actant_message_parts",
        "actant_tool_calls",
        "actant_memory_cards",
    }


def test_sqlalchemy_models_match_key_runtime_columns() -> None:
    thread_table = cast(Table, ActantThreadModel.__table__)
    run_table = cast(Table, ActantRunModel.__table__)
    tool_call_table = cast(Table, ActantToolCallModel.__table__)

    assert [column.name for column in thread_table.primary_key] == [
        "agent_id",
        "thread_id",
    ]
    assert [column.name for column in run_table.primary_key] == ["run_id"]
    assert [column.name for column in tool_call_table.primary_key] == ["tool_call_id"]

    thread_columns = set(thread_table.columns.keys())
    assert {
        "agent_id",
        "thread_id",
        "status",
        "turn_count",
        "active_run_id",
        "parent_thread_id",
        "parent_turn_id",
        "parent_tool_call_id",
        "created_at",
        "updated_at",
    } == thread_columns
    # Coordination columns are gone.
    assert "claim_lease_until" not in thread_columns

    run_columns = set(run_table.columns.keys())
    assert "waiting_group_id" not in run_columns
    assert "continued_groups" not in run_columns


def test_sqlalchemy_models_compile_for_postgres() -> None:
    thread_table = cast(Table, ActantThreadModel.__table__)
    ddl = str(CreateTable(thread_table).compile(dialect=postgresql.dialect()))

    assert "CREATE TABLE actant_threads" in ddl
    assert "TIMESTAMP WITH TIME ZONE" in ddl


def test_sqlalchemy_runtime_stores_wire_concrete_store_implementations() -> None:
    session_factory = async_sessionmaker[AsyncSession]()
    stores = SQLAlchemyRuntimeStores(session_factory)

    assert stores.threads.__class__.__name__ == "SQLAlchemyThreadStore"
    assert stores.runs.__class__.__name__ == "SQLAlchemyRunStore"
    assert stores.messages.__class__.__name__ == "SQLAlchemyMessageStore"
    assert stores.tool_calls.__class__.__name__ == "SQLAlchemyToolCallStore"
    assert stores.memory.__class__.__name__ == "SQLAlchemyMemoryStore"
    # Coordination stores no longer exist on the runtime stores facade.
    assert not hasattr(stores, "events")
    assert not hasattr(stores, "mailbox")
    assert not hasattr(stores, "turn_jobs")
    assert not hasattr(stores, "tool_jobs")
