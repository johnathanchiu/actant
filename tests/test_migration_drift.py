"""Actant's revisions build exactly the schema its SQLAlchemy models describe.

Opt in with ACTANT_TEST_POSTGRES_URL pointing to the disposable local
actant_core_test database. The test applies every revision in a schema of its
own, compares the result with ACTANT_RUNTIME_METADATA, and drops the schema.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from actant.migrations import versions_path
from actant.runtime.stores.postgres import ACTANT_RUNTIME_METADATA


def test_applied_revisions_match_the_models() -> None:
    raw = os.environ.get("ACTANT_TEST_POSTGRES_URL")
    if not raw:
        pytest.skip("requires disposable ACTANT_TEST_POSTGRES_URL")
    url = make_url(raw)
    if url.host not in {"localhost", "127.0.0.1"} or url.database != "actant_core_test":
        raise ValueError("test requires local actant_core_test database")
    schema = "drift_" + uuid4().hex
    script = ScriptDirectory(str(versions_path().parent), version_locations=[str(versions_path())])
    engine = create_engine(url.set(drivername="postgresql+psycopg2"))
    try:
        with engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            connection.execute(text(f'SET search_path TO "{schema}"'))
            context = MigrationContext.configure(connection)
            with Operations.context(context):
                for revision in reversed(list(script.walk_revisions("base", "heads"))):
                    revision.module.upgrade()
            diff = compare_metadata(context, ACTANT_RUNTIME_METADATA)
            assert diff == []
    finally:
        with engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        engine.dispose()
