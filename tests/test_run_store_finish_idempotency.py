"""``RunStore.finish`` must no-op on a missing run.

Temporal can redeliver the ``finalize_run`` activity after the run
row has been cleaned up (test teardown, manual cleanup, retention
policy). Pre-fix, ``finish`` called ``get`` which raised KeyError,
the activity entered exponential-backoff retry, and a doomed
finalize spammed the worker log forever. Post-fix, all three
``RunStore`` implementations (in-memory and SQLAlchemy) treat a
missing run as a successful no-op.

The sqlalchemy + raw impls are exercised at the protocol level by
``test_postgres_persistence.py`` against a real Postgres; this
module pins the in-memory contract directly.
"""

from __future__ import annotations

import pytest

from actant.runtime.stores.in_memory import InMemoryRunStore
from actant.runtime.types.threads import RunStatus


@pytest.mark.asyncio
async def test_finish_on_missing_run_is_a_noop() -> None:
    store = InMemoryRunStore()
    # No create() — the run does not exist in the store.
    await store.finish("never-created", RunStatus.CANCELLED)
    # No exception means contract held; verify the row is still absent.
    with pytest.raises(KeyError):
        await store.get("never-created")


@pytest.mark.asyncio
async def test_finish_writes_status_when_run_exists() -> None:
    store = InMemoryRunStore()
    await store.create("a", "t", run_id="r1", max_turns=5)
    await store.finish("r1", RunStatus.EXHAUSTED)
    run = await store.get("r1")
    assert run.status == RunStatus.EXHAUSTED


@pytest.mark.asyncio
async def test_finish_is_idempotent_on_repeat_calls() -> None:
    """A retried ``finalize_run`` activity may call ``finish`` twice
    for the same row. Both calls must succeed."""
    store = InMemoryRunStore()
    await store.create("a", "t", run_id="r1", max_turns=5)
    await store.finish("r1", RunStatus.CANCELLED)
    await store.finish("r1", RunStatus.CANCELLED)
    run = await store.get("r1")
    assert run.status == RunStatus.CANCELLED
