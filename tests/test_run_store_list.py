"""A thread's runs are listable, newest first, so a product can find what the last run delivered."""

from __future__ import annotations

import pytest

from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.types.threads import RunStatus


@pytest.mark.asyncio
async def test_runs_list_for_a_thread_newest_first() -> None:
    stores = InMemoryRuntimeStores()
    await stores.runs.create("a", "t", run_id="r1", max_turns=5)
    await stores.runs.create("a", "t", run_id="r2", max_turns=5)
    await stores.runs.create("a", "other", run_id="r3", max_turns=5)
    await stores.runs.finish("r2", RunStatus.EXHAUSTED, stop_reason="stopped without finishing")

    runs = await stores.runs.list_for_thread("a", "t")
    assert [r.id for r in runs] == ["r2", "r1"]
    assert runs[0].stop_reason == "stopped without finishing"


@pytest.mark.asyncio
async def test_creating_a_run_again_returns_the_first_unchanged() -> None:
    """A retried ``start_run`` creates its run again; the store keeps the first."""
    stores = InMemoryRuntimeStores()
    first = await stores.runs.create("a", "t", run_id="r1", max_turns=5)
    first.turn_count = 2
    await stores.runs.update(first)
    again = await stores.runs.create("a", "t", run_id="r1", max_turns=9)
    assert (again.max_turns, again.turn_count) == (5, 2)
