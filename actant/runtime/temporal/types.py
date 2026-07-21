"""Serializable payloads crossing Temporal workflow/activity boundaries.

All types here cross the workflow/activity boundary, so they are frozen
dataclasses with JSON-friendly fields (primitives, dicts, lists, nested
dataclasses) and no live runtime objects (stores, hooks, agents).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


# === Configuration ===


@dataclass(frozen=True)
class TemporalRuntimeConfig:
    """Connection and lifecycle configuration for Actant's Temporal runtime."""

    address: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "actant-runtime"
    workflow_id_prefix: str = "actant-thread"
    max_turns_per_run: int = 25
    # Soft threshold for triggering continue_as_new at the run boundary.
    # Replay walks every event so very long histories slow workflow tasks
    # down. 5_000 is a starting point; tune via load test.
    history_size_threshold: int = 5_000
    # How long a workflow may remain durably suspended for an external
    # tool resolution. The workflow consumes no worker compute while waiting.
    external_resolution_timeout_seconds: int = 7 * 24 * 60 * 60  # 7 days


# === Names ===
#
# Stable identifiers for the Temporal APIs that still require names.


class ActivityName(StrEnum):
    """Activity names registered on the worker.

    Workflows dispatch through typed method references. Explicit registered
    names keep the Temporal wire contract stable if Python symbols move.
    """

    START_RUN = "start_run"
    RUN_TURN = "run_turn"
    ADMIT_TOOL = "admit_tool"
    EXECUTE_TOOL = "execute_tool"
    RESOLVE_TOOL = "resolve_tool"
    FINALIZE_TOOL_GROUP = "finalize_tool_group"
    FINALIZE_RUN = "finalize_run"
    APPLY_THREAD_CANCELLATION = "apply_thread_cancellation"


class SignalName(StrEnum):
    """Workflow signal names. Used by ``signal_with_start`` and
    ``handle.signal``. The strings match the ``@workflow.signal``
    method names on ``AgentThreadWorkflow``.

    Deferred tool resolutions arrive as durable signals. Temporal records
    them even when the workflow has not reached its wait condition yet.
    """

    INBOUND = "inbound"
    CANCEL = "cancel"
    RESOLVE_TOOL = "resolve_tool"


# === Outcomes ===


class RunOutcome(StrEnum):
    COMPLETED = "completed"
    EXHAUSTED = "exhausted"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ThreadOutcome(StrEnum):
    STOPPED = "stopped"
    CANCELLED = "cancelled"


class AdmitDecision(StrEnum):
    """Output of ``admit_tool``.

    ``ALLOW`` → workflow fires ``execute_tool``.
    ``BLOCK`` → terminal; admit already persisted the failed result.
    ``WAIT`` → workflow suspends until a durable resolution signal arrives.
    """

    ALLOW = "allow"
    BLOCK = "block"
    WAIT = "wait"


class ExecuteStatus(StrEnum):
    """Output of ``execute_tool`` and ``resolve_tool``."""

    COMPLETED = "completed"
    FAILED = "failed"


# === Workflow input ===


@dataclass(frozen=True)
class InboundMessage:
    """Payload for the ``inbound`` workflow signal.

    ``content`` mirrors the existing ``send_message`` API: either a plain
    string or a list of multimodal content blocks.
    """

    content: str | list[dict[str, Any]]
    source: str = "user"
    correlation_id: str | None = None


@dataclass(frozen=True)
class ThreadInput:
    agent_id: str
    thread_id: str
    max_turns_per_run: int = 25
    external_resolution_timeout_seconds: int = 7 * 24 * 60 * 60
    # Carry-forward state for continue_as_new. Empty on initial start.
    carry_inbox: list[InboundMessage] = field(default_factory=list)
    # Appended after carry_inbox to preserve the positional shape of the
    # pre-0.1 payload while still carrying client configuration into the
    # deterministic workflow.
    history_size_threshold: int = 5_000
    # Thread-level workflow state that must survive continue-as-new. The
    # per-agent-run turn budget intentionally does not carry forward.
    turn_count_total: int = 0


# === Activity I/O ===


@dataclass(frozen=True)
class StartRunInput:
    agent_id: str
    thread_id: str
    run_id: str
    max_turns: int


@dataclass(frozen=True)
class FinalizeRunInput:
    agent_id: str
    thread_id: str
    run_id: str
    outcome: str  # RunOutcome value
    turn_count: int


@dataclass(frozen=True)
class RunTurnInput:
    agent_id: str
    thread_id: str
    run_id: str
    turn_id: str
    turn_index: int
    # Inbox messages to apply before the turn. Only non-empty on the first
    # turn of a run; subsequent turns of the same run pass [].
    new_messages: list[InboundMessage] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCallSpec:
    """Subset of ``ToolCallRecord`` that the workflow needs to fan out tools.

    Activities reload the full ``ToolCallRecord`` from the store via ``id``.
    Keeping the workflow payload small bounds the per-event history size.
    """

    id: str
    group_id: str
    run_id: str
    turn_id: str
    turn_index: int
    name: str


@dataclass(frozen=True)
class TurnResult:
    turn_id: str
    turn_index: int
    tool_calls: list[ToolCallSpec] = field(default_factory=list)


@dataclass(frozen=True)
class AdmitInput:
    agent_id: str
    thread_id: str
    run_id: str
    tool_call_id: str


@dataclass(frozen=True)
class AdmitOutcome:
    """Structured output of ``admit_tool``. Activity is infallible —
    any unexpected exception is mapped to ``decision=BLOCK`` with the
    exception text in ``reason``."""

    tool_call_id: str
    decision: str  # AdmitDecision value
    reason: str | None = None
    # Set when ``decision == WAIT``; the prompt the external resolver
    # should display / use. Surfaces to ``on_tool_waiting`` hook.
    wait_request: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExecuteInput:
    agent_id: str
    thread_id: str
    run_id: str
    tool_call_id: str


@dataclass(frozen=True)
class ExecuteOutcome:
    """Structured output of ``execute_tool`` and ``resolve_tool``.
    Activities are infallible — any
    unexpected exception is mapped to ``status=FAILED`` with the
    error captured in ``result``."""

    tool_call_id: str
    status: str  # ExecuteStatus value
    terminal: bool = False


@dataclass(frozen=True)
class DeferredToolResolution:
    """External input delivered durably to a thread workflow."""

    tool_call_id: str
    approved: bool | None = None
    answer: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolveToolInput:
    """Persist a deferred tool result after its workflow is awakened.

    ``resolution=None`` represents expiration of the durable workflow wait.
    """

    agent_id: str
    thread_id: str
    run_id: str
    tool_call_id: str
    resolution: DeferredToolResolution | None


@dataclass(frozen=True)
class ApplyThreadCancellationInput:
    """Thread-level cancel cleanup.

    Idempotent. Walks open ``tool_calls`` and writes the
    ``session_cancelled`` placeholder so the LLM transcript invariant
    holds. Sets ``thread.status = CANCELLED`` and clears
    ``active_run_id``. Always called on workflow cancel — even when
    there's no active run for ``finalize_run`` to handle.
    """

    agent_id: str
    thread_id: str


# === Query views ===


@dataclass(frozen=True)
class ThreadStateView:
    agent_id: str
    thread_id: str
    inbox_size: int
    turn_count_total: int
    current_run_id: str | None
    cancelled: bool
