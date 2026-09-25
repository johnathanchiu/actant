"""0003_typed_blocks rewrites every legacy block shape into a typed one, once.

Opt in with ACTANT_TEST_POSTGRES_URL pointing to the disposable local
actant_core_test database, as test_migration_drift.py does.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, create_engine, text
from sqlalchemy.engine import make_url

from actant.blocks import BLOCKS
from actant.migrations import versions_path

UNAVAILABLE = "[Image unavailable: stored before images were kept by key]"
URL = "https://bucket.example/images/a.png?X-Amz-Signature=abc"

LEGACY: dict[str, list[object]] = {
    "url_with_key": [
        {"type": "text", "text": "look"},
        {
            "type": "image",
            "source": {
                "type": "url",
                "url": URL,
                "expires_at": 1.0,
                "key": "s3://bucket/images/t/a.jpg",
                "media_type": "image/jpeg",
            },
        },
    ],
    "url_with_key_no_media_type": [
        {"type": "image", "source": {"type": "url", "url": URL, "key": "images/b"}}
    ],
    "url_without_key": [
        {"type": "image", "source": {"type": "url", "url": URL, "expires_at": 1.0}}
    ],
    "input_image_data": [
        {"type": "input_image", "image_url": "data:image/webp;base64,UklGRg==", "detail": "high"}
    ],
    "input_image_https": [{"type": "input_image", "image_url": URL}],
    "typed": [{"type": "asset", "storage_key": "k", "mime": "application/pdf"}],
}

EXPECTED: dict[str, list[object]] = {
    "url_with_key": [
        {"type": "text", "text": "look"},
        {"type": "asset", "storage_key": "s3://bucket/images/t/a.jpg", "mime": "image/jpeg"},
    ],
    "url_with_key_no_media_type": [
        {"type": "asset", "storage_key": "images/b", "mime": "image/png"}
    ],
    "url_without_key": [{"type": "text", "text": UNAVAILABLE}],
    "input_image_data": [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/webp", "data": "UklGRg=="},
        }
    ],
    "input_image_https": [{"type": "text", "text": UNAVAILABLE}],
    "typed": [{"type": "asset", "storage_key": "k", "mime": "application/pdf"}],
}


@pytest.fixture
def connection() -> Iterator[Connection]:
    raw = os.environ.get("ACTANT_TEST_POSTGRES_URL")
    if not raw:
        pytest.skip("requires disposable ACTANT_TEST_POSTGRES_URL")
    url = make_url(raw)
    if url.host not in {"localhost", "127.0.0.1"} or url.database != "actant_core_test":
        raise ValueError("test requires local actant_core_test database")
    schema = "typed_blocks_" + uuid4().hex
    engine = create_engine(url.set(drivername="postgresql+psycopg2"))
    try:
        with engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            connection.execute(text(f'SET search_path TO "{schema}"'))
            yield connection
    finally:
        with engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        engine.dispose()


SCRIPT = ScriptDirectory(str(versions_path().parent), version_locations=[str(versions_path())])


def _upgrade(connection: Connection, lower: str, upper: str) -> None:
    """Apply the revisions after ``lower`` through ``upper``, in order."""
    with Operations.context(MigrationContext.configure(connection)):
        for revision in reversed(list(SCRIPT.walk_revisions(lower, upper))):
            if revision.revision != lower:
                revision.module.upgrade()


def _seed(connection: Connection, rows: dict[str, list[object] | None]) -> None:
    for message_id, blocks in rows.items():
        connection.execute(
            text(
                "INSERT INTO actant_messages (message_id, agent_id, thread_id, role) "
                "VALUES (:m, 'a', 't', 'user')"
            ),
            {"m": message_id},
        )
        connection.execute(
            text(
                "INSERT INTO actant_message_parts (message_id, part_index, kind, content_blocks) "
                "VALUES (:m, 0, 'user_prompt', CAST(:b AS JSONB))"
            ),
            {"m": message_id, "b": None if blocks is None else json.dumps(blocks)},
        )


def _stored(connection: Connection) -> dict[str, list[object] | None]:
    rows = connection.execute(text("SELECT message_id, content_blocks FROM actant_message_parts"))
    return {message_id: blocks for message_id, blocks in rows}


def test_legacy_blocks_become_typed_and_a_rerun_changes_nothing(connection: Connection) -> None:
    _upgrade(connection, "base", "0002_sandbox_and_run_reason")
    _seed(connection, {**LEGACY, "no_blocks": None})

    _upgrade(connection, "0002_sandbox_and_run_reason", "0003_typed_blocks")
    stored = _stored(connection)
    assert stored == {**EXPECTED, "no_blocks": None}
    for blocks in EXPECTED.values():
        BLOCKS.validate_python(blocks)

    _upgrade(connection, "0002_sandbox_and_run_reason", "0003_typed_blocks")
    assert _stored(connection) == stored


def test_an_unknown_shape_fails_the_upgrade(connection: Connection) -> None:
    _upgrade(connection, "base", "0002_sandbox_and_run_reason")
    _seed(connection, {"odd": [{"type": "video", "url": URL}]})

    with pytest.raises(RuntimeError, match="1 content_blocks rows do not validate"):
        _upgrade(connection, "0002_sandbox_and_run_reason", "0003_typed_blocks")
    assert _stored(connection) == {"odd": [{"type": "video", "url": URL}]}


def test_downgrade_refuses() -> None:
    revision = SCRIPT.get_revision("0003_typed_blocks")
    assert revision is not None
    with pytest.raises(NotImplementedError, match="irreversible"):
        revision.module.downgrade()
