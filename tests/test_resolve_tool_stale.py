"""Tests for deferred tool-call state reconciliation.

When Temporal can't find the parked activity (workflow terminated, volume
nuked in dev, activity timed out), the runtime should:

1. Update the tool_calls store from WAITING → FAILED with a diagnostic
   ``stale_activity`` reason.
2. Raise ``ToolResolutionStaleError`` (typed) instead of letting the
   raw ``temporalio.service.RPCError`` propagate.

This isolates the failure mode so applications can handle it
gracefully — and prevents the WAITING state from outliving the
underlying workflow.
"""

from __future__ import annotations

from typing import Any

import pytest
import temporalio.service

from actant.runtime.exceptions import ToolResolutionStaleError
from actant.runtime.temporal.client import TemporalRuntimeClient
from actant.runtime.temporal.types import TemporalRuntimeConfig
from actant.runtime.stores import InMemoryRuntimeStores
from actant.tools.calls import ToolCallRecord, ToolCallStatus


_AGENT = "test_agent"
_THREAD = "test_thread"
_TOOL_CALL_ID = "tc_stale_1"


def _make_record() -> ToolCallRecord:
    return ToolCallRecord(
        id=_TOOL_CALL_ID,
        group_id="g_1",
        run_id="r_1",
        agent_id=_AGENT,
        thread_id=_THREAD,
        turn_id="turn_1",
        turn_index=0,
        name="ask_user",
        args={"question": "anything"},
        status=ToolCallStatus.WAITING,
        prompt="anything",
        wait_request={"kind": "question", "prompt": "anything"},
        temporal_workflow_id="wf_stale",
        temporal_activity_id="33",  # the activity Temporal "lost"
    )


class _NotFoundActivityHandle:
    """Stand-in for ``client.get_async_activity_handle(...)``. Raises
    NOT_FOUND on ``complete`` to mimic Temporal having lost the
    pending activity."""

    async def complete(self, _outcome: Any) -> None:
        raise temporalio.service.RPCError(
            "cannot find pending activity with ActivityID 33",
            temporalio.service.RPCStatusCode.NOT_FOUND,
            b"",
        )


class _FakeTemporalClient:
    def get_async_activity_handle(
        self, *, workflow_id: str, run_id: str | None, activity_id: str
    ) -> _NotFoundActivityHandle:
        del workflow_id, run_id, activity_id
        return _NotFoundActivityHandle()


@pytest.fixture
def stores() -> InMemoryRuntimeStores:
    return InMemoryRuntimeStores()


@pytest.fixture
async def runtime_client(stores: InMemoryRuntimeStores) -> TemporalRuntimeClient:
    ex = TemporalRuntimeClient(
        stores=stores,
        agents={},  # deferred resolution doesn't need agent definitions
        config=TemporalRuntimeConfig(),
    )
    # Bypass real Temporal connection; inject the fake client.
    ex._client = _FakeTemporalClient()  # type: ignore[assignment]
    return ex


async def test_deferred_resolution_reconciles_when_activity_not_found(
    runtime_client: TemporalRuntimeClient, stores: InMemoryRuntimeStores
) -> None:
    """The store should flip WAITING → FAILED with stale_activity reason,
    and the caller should see ``ToolResolutionStaleError``."""
    record = _make_record()
    await stores.tool_calls.save(record)

    # Sanity: record is WAITING before resolve.
    pre = await stores.tool_calls.get(_TOOL_CALL_ID)
    assert pre.status == ToolCallStatus.WAITING

    with pytest.raises(ToolResolutionStaleError) as excinfo:
        await runtime_client.resolve_deferred_tool_call(
            _AGENT, _THREAD, _TOOL_CALL_ID, approved=True, answer="ok"
        )

    assert excinfo.value.tool_call_id == _TOOL_CALL_ID
    assert "lost the pending activity" in excinfo.value.reason.lower()

    # Store reconciled — no more WAITING.
    post = await stores.tool_calls.get(_TOOL_CALL_ID)
    assert post.status == ToolCallStatus.FAILED
    assert isinstance(post.result, dict)
    assert post.result.get("error") == "stale_activity"
    assert post.result.get("tool_call_id") == _TOOL_CALL_ID
    assert "lost the pending activity" in str(post.result.get("detail", "")).lower()


async def test_deferred_resolution_non_notfound_rpc_error_propagates(
    runtime_client: TemporalRuntimeClient, stores: InMemoryRuntimeStores
) -> None:
    """Other Temporal errors (e.g. INTERNAL) shouldn't be silently swallowed
    by the reconciliation path. They should bubble up as-is."""

    class _InternalErrorHandle:
        async def complete(self, _outcome: Any) -> None:
            raise temporalio.service.RPCError(
                "something else broke",
                temporalio.service.RPCStatusCode.INTERNAL,
                b"",
            )

    class _FakeInternal:
        def get_async_activity_handle(self, **_: Any) -> _InternalErrorHandle:
            return _InternalErrorHandle()

    runtime_client._client = _FakeInternal()  # type: ignore[assignment]

    record = _make_record()
    record.id = "tc_internal_1"
    await stores.tool_calls.save(record)

    with pytest.raises(temporalio.service.RPCError):
        await runtime_client.resolve_deferred_tool_call(
            _AGENT, _THREAD, "tc_internal_1", approved=True, answer="ok"
        )

    # Store should NOT be marked stale — non-NOT_FOUND errors aren't
    # diagnostic of "the workflow lost the activity."
    post = await stores.tool_calls.get("tc_internal_1")
    assert post.status != ToolCallStatus.FAILED or post.result is None or (
        isinstance(post.result, dict) and post.result.get("error") != "stale_activity"
    )
