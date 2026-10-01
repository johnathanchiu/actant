"""Sandbox records: the spec round trip and the in-memory store's compare-and-set."""

from __future__ import annotations

import pytest

from actant.sandbox import SandboxSpec, Storage
from actant.sandbox.store import InMemorySandboxStore, spec_from_json, spec_to_json


def test_a_spec_round_trips_through_json_without_its_image() -> None:
    spec = SandboxSpec(
        backend="modal",
        image=object(),
        env={"A": "1"},
        region=("us-east", "us-west"),
        secrets=("bucket",),
        storage=Storage.DISK_SYNC,
        services={"room": "pkg.mod:Room"},
        push_exclude=("renders",),
        gpu="L4",
    )
    data = spec_to_json(spec)
    assert "image" not in data and data["storage"] == "disk_sync"
    back = spec_from_json(data)
    assert back.image is None
    assert back == SandboxSpec(**{**spec.__dict__, "image": None})
    # A field a newer actant wrote is dropped, not refused.
    assert spec_from_json({**data, "later": 1}) == back


@pytest.mark.asyncio
async def test_the_in_memory_store_sets_an_id_only_over_the_one_expected() -> None:
    store, spec = InMemorySandboxStore(), SandboxSpec(backend="local", image=object())
    first = await store.claim("s", expected=None, provider_id="a", spec=spec)
    assert first is not None and first.provider_id == "a" and first.spec.image is None
    lost = await store.claim("s", expected=None, provider_id="b", spec=spec)
    assert lost == first
    moved = await store.claim("s", expected="a", provider_id="c", spec=spec)
    assert moved is not None and moved.provider_id == "c"
    await store.forget("s")
    assert await store.get("s") is None
    assert await store.claim("s", expected="c", provider_id="d", spec=spec) is None
