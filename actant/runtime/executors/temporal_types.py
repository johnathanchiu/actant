"""Serializable Temporal runtime payloads for the AgentThreadWorkflow.

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
    """Configuration for Actant's Temporal executor."""

    address: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "actant-runtime"
    workflow_id_prefix: str = "actant-thread"
    max_turns_per_run: int = 25
    # Soft threshold for triggering continue_as_new at the run boundary.
    # Replay walks every event so very long histories slow workflow tasks
    # down. 5_000 is a starting point; tune via load test.
    history_size_threshold: int = 5_000
    # How long ``await_external_resolution`` activities are allowed to
    # remain pending before Temporal considers them dead. Deferred tools
    # (human approval, async coordination) need a generous bound — set
    # to whatever the longest legitimate human-in-the-loop wait could
    # take. The workflow consumes zero compute during the wait so this
    # is purely a "mark dead and surface to operator" timeout.
    external_resolution_timeout_seconds: int = 7 * 24 * 60 * 60  # 7 days


# === Names ===
#
# Public identifiers for everything addressable by string. Downstream
# users can import these to start workflows, send signals, query state,
# or register custom activities without copying magic strings.


class WorkflowName(StrEnum):
    AGENT_THREAD = "AgentThreadWorkflow"


class ActivityName(StrEnum):
    """Activity names registered on the worker.

    Workflow code references activities by name (rather than by direct
    function reference) because activity callables are bound methods on
    a per-worker class. The strings here are the source of truth for
    both ``@activity.defn(name=...)`` and ``execute_activity(...)``.
    """

    START_RUN = "start_run"
    RUN_TURN = "run_turn"
    ADMIT_TOOL = "admit_tool"
    EXECUTE_TOOL = "execute_tool"
    AWAIT_EXTERNAL_RESOLUTION = "await_external_resolution"
    FINALIZE_TOOL_GROUP = "finalize_tool_group"
    FINALIZE_RUN = "finalize_run"
    APPLY_THREAD_CANCELLATION = "apply_thread_cancellation"


class SignalName(StrEnum):
    """Workflow signal names. Used by ``signal_with_start`` and
    ``handle.signal``. The strings match the ``@workflow.signal``
    method names on ``AgentThreadWorkflow``.

    Deferred tool resolution does NOT go through a workflow signal —
    it lands directly on the ``await_external_resolution`` activity via
    Temporal's async-activity-completion path
    (``client.complete_activity_by_id``). The workflow doesn't even
    know about pending resolutions; it just awaits its activity handles.
    """

    INBOUND = "inbound"
    CANCEL = "cancel"


class QueryName(StrEnum):
    """Workflow query names. Used by ``handle.query``."""

    GET_STATE = "get_state"


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
    ``WAIT`` → workflow fires ``await_external_resolution``; an external
    caller will deliver the result via ``complete_activity_by_id``.
    """

    ALLOW = "allow"
    BLOCK = "block"
    WAIT = "wait"


class ExecuteStatus(StrEnum):
    """Output of ``execute_tool`` and ``await_external_resolution``."""

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
    """Structured output of ``execute_tool`` and
    ``await_external_resolution``. Activity is infallible — any
    unexpected exception is mapped to ``status=FAILED`` with the
    error captured in ``result``."""

    tool_call_id: str
    status: str  # ExecuteStatus value
    terminal: bool = False


@dataclass(frozen=True)
class AwaitExternalResolutionInput:
    """Input for the ``await_external_resolution`` activity.

    The activity persists ``(workflow_id, activity_id)`` from
    ``activity.info()`` onto the tool_call record so external callers
    (HTTP APIs, approval UIs) can later complete this activity with a
    result via ``client.complete_activity_by_id``. The activity body
    calls ``raise_complete_async`` and returns; the activity remains
    "running" in Temporal until the external completion lands.
    """

    agent_id: str
    thread_id: str
    run_id: str
    tool_call_id: str


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
