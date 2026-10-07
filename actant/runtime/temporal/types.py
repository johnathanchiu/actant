"""Serializable payloads crossing Temporal workflow/activity boundaries.

All types here cross the workflow/activity boundary, so they are frozen
dataclasses with JSON-friendly fields (primitives, dicts, lists, nested
dataclasses) and no live runtime objects (stores, hooks, agents).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from actant.sandbox.access import SandboxAccess


# === Configuration ===


@dataclass(frozen=True)
class ActivityTimeouts:
    """How long one model turn, one compaction, and one tool call may run, in seconds.

    Each is the activity's ``start_to_close_timeout``: a tool that opens a sandbox
    needs at least the sandbox's own open budget here. A tool call that stops
    heartbeating for ``tool_heartbeat_s`` is treated as lost (its worker died), and a
    model turn for ``turn_heartbeat_s``; set each longer than the activity's own beat,
    every ``actant.heartbeat.HEARTBEAT_EVERY_S``.
    """

    turn_s: float = 600.0
    compact_s: float = 600.0
    tool_s: float = 600.0
    tool_heartbeat_s: float = 120.0
    turn_heartbeat_s: float = 120.0

    def __post_init__(self) -> None:
        limits = (self.turn_s, self.compact_s, self.tool_s)
        if min(*limits, self.tool_heartbeat_s, self.turn_heartbeat_s) <= 0:
            raise ValueError("activity timeouts must be positive")


#: The ``ApplicationError`` type a thread fails with when ``CompactionConfig.summarizer``
#: names a client its worker never registered.
UNREGISTERED_SUMMARIZER = "UnregisteredSummarizer"


@dataclass(frozen=True)
class CompactionConfig:
    """When to compact the model's context, and what survives it verbatim.

    Compaction is one summarizing call on the agent's own model, with no tools,
    made when the next request would pass ``threshold`` of the context window
    or carry more images than allowed. It never truncates, rewrites or deletes
    a stored message. The model then sees the system prompt, the summary, the
    latest message of each tag in ``keep`` (in transcript order), and the
    messages from the boundary on. Tool results are tagged ``tool:<name>``;
    an app tags what it sends (``send_message(..., tag="brief")``)::

        CompactionConfig(threshold=0.9, keep=["brief", "tool:read_checklist"])

    With ``background``, the summary is written ahead of need: once a request
    passes that fraction of the window (or of the image limit), a summary of
    everything before the open turn starts beside the turns, which keep
    running on the full context. It is swapped in at the next turn boundary
    after it is ready, with every message stored since kept after it. It is
    written in a workflow of its own, so a run that ends never waits on it. A
    turn waits on a summary only when it would cross ``threshold`` itself::

        CompactionConfig(threshold=0.9, background=0.5, keep=["tool:checklist"])

    ``summarizer`` names a model client the worker registers
    (``AgentRuntime(summarizers={"fast": client})``) to write the summary on
    instead of the agent's own model: a smaller model, configured with little or
    no reasoning, makes the call faster. A request that client cannot take (past
    its declared window or image limit) is summarized on the agent's model.
    ``summary_tokens`` caps the summary's output (the default is 16,000, less
    when the window's margin is smaller); output tokens are most of the call's
    time. A summary that runs into the cap was cut short, so it is written again
    on the agent's own model with the default cap: a summary is never truncated::

        CompactionConfig(summarizer="fast", summary_tokens=4_000)

    ``prompt`` replaces actant's generic summary instruction (``COMPACTION_PROMPT``)
    for an app that knows what its agents must carry across a compaction; the
    runtime's ``compaction_instructions`` and the length line ``summary_tokens``
    adds still follow it. ``None`` keeps the generic one::

        CompactionConfig(prompt="<compaction_request>...</compaction_request>")

    ``images=False`` sends the summary call no images, each as its
    ``[image id=...]`` label alone: the summary is text, and on a context
    heavy with pictures they are much of the call's input and time. The image
    index is then written from what the conversation said about each one::

        CompactionConfig(images=False)

    ``image_limit="drop_oldest"`` stops the image limit from triggering a summary:
    a request past it is sent with its oldest images replaced by their
    ``[image id=...]`` label (``recall_image`` shows one again), in chunks of
    half the limit so the cached prefix stays stable. Only the context window
    then summarizes, and no text is lost. ``"summarize"`` (the default)
    summarizes at either limit::

        CompactionConfig(image_limit="drop_oldest")

    The limits are the model client's: its ``context_window_tokens`` and
    ``max_images_per_request`` (``OpenAIProvider`` takes both); a limit the
    client does not declare is not checked. Unrelated to
    ``history_size_threshold``, which rotates Temporal's event history.
    """

    threshold: float = 0.9
    keep: list[str] = field(default_factory=list)
    background: float | None = None
    summarizer: str | None = None
    summary_tokens: int | None = None
    prompt: str | None = None
    images: bool = True
    image_limit: Literal["summarize", "drop_oldest"] = "summarize"

    def __post_init__(self) -> None:
        if self.background is not None and not 0 < self.background < self.threshold:
            raise ValueError("background must be above 0 and below threshold")
        if self.summary_tokens is not None and self.summary_tokens < 256:
            raise ValueError("summary_tokens must be at least 256")
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("prompt must not be empty; None keeps the generic one")


@dataclass(frozen=True)
class TemporalRuntimeConfig:
    """Connection and lifecycle configuration for Actant's Temporal runtime."""

    address: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "actant-runtime"
    workflow_id_prefix: str = "actant-thread"
    max_turns_per_run: int | None = None
    # Allow active activities to finish before worker shutdown requests cancellation.
    graceful_shutdown_timeout_seconds: float = 0.0
    # Activities (model turns, tool calls) one worker runs at once. None keeps
    # the Temporal SDK's default of 100, which no single process should size by.
    max_concurrent_activities: int | None = None
    # Soft threshold for triggering continue_as_new at the run boundary.
    # Replay walks every event so very long histories slow workflow tasks
    # down. 5_000 is a starting point; tune via load test.
    history_size_threshold: int = 5_000
    # How long a workflow may remain durably suspended for an external
    # tool resolution. The workflow consumes no worker compute while waiting.
    external_resolution_timeout_seconds: int = 7 * 24 * 60 * 60  # 7 days
    # Hand a message sent to a running thread to the model on the run's next
    # turn rather than at the next run. Copied into ``ThreadInput`` when a
    # thread starts, so it applies to executions started after it changes.
    interleave_inbox: bool = False
    # Summarize the model's context near its real limits. ``None`` (the
    # default) never compacts. Copied into ``ThreadInput`` like
    # ``interleave_inbox``.
    context_compaction: CompactionConfig | None = None
    # How long a model turn, a compaction and a tool call may each run. Copied
    # into ``ThreadInput`` like ``interleave_inbox``.
    activity_timeouts: ActivityTimeouts = field(default_factory=ActivityTimeouts)


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
    COMPACT_CONTEXT = "compact_context"
    SUMMARIZE_CONTEXT = "summarize_context"
    STORE_SUMMARY = "store_summary"
    ADMIT_TOOL = "admit_tool"
    EXECUTE_TOOL = "execute_tool"
    RESOLVE_TOOL = "resolve_tool"
    SUMMARY_READY = "summary_ready"
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
    """Output of ``admit_tool``: who produces this call's result.

    ``EXECUTE`` → workflow fires ``execute_tool``; the tool produces it.
    ``DENY`` → terminal; admission already persisted the refusal as the
    result, so the workflow does nothing.
    ``AWAIT_HUMAN`` → workflow suspends until a person answers.

    Only ``AWAIT_HUMAN`` suspends. That is the whole of what the workflow
    needs to know from this; which of the other two produced the result is
    the activity's business.
    """

    EXECUTE = "execute"
    DENY = "deny"
    AWAIT_HUMAN = "await_human"


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
    # Stored on the message; ``CompactionConfig.keep`` names tags to keep.
    tag: str | None = None


@dataclass(frozen=True)
class ThreadInput:
    agent_id: str
    thread_id: str
    max_turns_per_run: int | None = None
    #: Set when this thread is a subagent of another; recorded on the thread
    #: row so its tools see ``CallContext.parent_thread_id``.
    parent_thread_id: str | None = None
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
    # Deliver messages that arrive mid-run on the run's next turn, after the
    # previous turn's tool results, instead of holding them for the next run.
    # Last and defaulted, like every field after carry_inbox: a history
    # recorded before it existed decodes to False and replays the old path.
    interleave_inbox: bool = False
    # Model-context compaction for this thread; ``None`` never compacts. Last
    # and defaulted for the same reason as ``interleave_inbox``.
    context_compaction: CompactionConfig | None = None
    # Last and defaulted likewise: a history recorded before it existed decodes
    # to the defaults, which are the timeouts it ran with.
    activity_timeouts: ActivityTimeouts = field(default_factory=ActivityTimeouts)
    #: The id of a sandbox a product opened that this thread works in (recorded on the
    #: thread as ``AgentThread.sandbox_id``); ``None`` gives it a sandbox of its own.
    sandbox_id: str | None = None
    #: What this thread may write in its sandbox (recorded on the thread as
    #: ``AgentThread.sandbox_access``); ``None``: the agent definition's.
    sandbox_access: SandboxAccess | None = None
    #: JSON its starter gives each of this run's tool calls (``CallContext.context``): what a
    #: tool would otherwise query its starter's workflow for, which a busy workflow refuses
    #: once its query buffer is full. Kept across ``continue_as_new``.
    context: dict[str, Any] = field(default_factory=dict)


# === Activity I/O ===


@dataclass(frozen=True)
class StartRunInput:
    agent_id: str
    thread_id: str
    run_id: str
    max_turns: int | None
    parent_thread_id: str | None = None
    sandbox_id: str | None = None
    sandbox_access: SandboxAccess | None = None
    #: ``CompactionConfig.summarizer``, checked before the run opens: a name the worker
    #: never registered fails the thread once, rather than every run that compacts.
    summarizer: str | None = None


@dataclass(frozen=True)
class StartedRun:
    turn_count: int
    max_turns: int
    error: str | None = None


@dataclass(frozen=True)
class FinalizeRunInput:
    agent_id: str
    thread_id: str
    run_id: str
    outcome: str  # RunOutcome value
    turn_count: int
    stop_reason: str | None = None


@dataclass(frozen=True)
class RunTurnInput:
    agent_id: str
    thread_id: str
    run_id: str
    turn_id: str
    turn_index: int
    # Inbox messages to apply before the turn. Only non-empty on the first
    # turn of a run, unless the thread sets ``interleave_inbox``: then any
    # turn carries whatever arrived since the previous one.
    new_messages: list[InboundMessage] = field(default_factory=list)
    # How many text-only turns this run has already answered with a
    # reminder. Only meaningful for ``completion="terminal"`` agents.
    text_only_turns: int = 0
    # The thread's compaction limits; ``None`` never measures the request.
    context_compaction: CompactionConfig | None = None
    # Set when the workflow calls this turn again after compacting for it:
    # the turn gate already admitted it, and it must not ask to compact twice.
    compacted: bool = False
    # A background summary that is ready: stored as the compaction row before
    # anything else this turn does (``ContextSummary``).
    summary: ContextSummary | None = None
    # The turn gate already admitted this turn (it is run again after waiting
    # on a background summary); it is measured again.
    admitted: bool = False


@dataclass(frozen=True)
class CompactionTrigger:
    """``run_turn``'s measurement of a request that would cross a limit.

    ``tokens`` is the last turn's reported usage plus an estimate of what was
    added since; ``images`` is exact. ``carried`` names the stored messages (by id) of
    the turn still open (an assistant tool call and its results), which the
    summary does not replace and the compaction keeps whole.
    """

    reason: str
    tokens: int
    images: int
    carried: list[str] = field(default_factory=list)
    # A background trigger's boundary: the last stored message the summary
    # replaces. Later messages are all kept when it is stored.
    through: str | None = None


@dataclass(frozen=True)
class CompactContextInput:
    agent_id: str
    thread_id: str
    run_id: str
    turn_id: str
    turn_index: int
    trigger: CompactionTrigger
    config: CompactionConfig = field(default_factory=CompactionConfig)


@dataclass(frozen=True)
class ContextSummary:
    """A background summary, written but not yet stored.

    ``base`` is the compaction row the summarized view started from; a summary
    whose base is no longer the latest is stale and dropped unstored.
    """

    summary: str
    through: str
    base: str | None
    trigger: CompactionTrigger


@dataclass(frozen=True)
class SummaryJob:
    """A background summary's own workflow (``ContextSummaryWorkflow``): what to
    summarize, and how long the summary call may take."""

    compact: CompactContextInput
    compact_s: float


@dataclass(frozen=True)
class StoreSummaryInput:
    agent_id: str
    thread_id: str
    run_id: str
    summary: ContextSummary
    config: CompactionConfig = field(default_factory=CompactionConfig)


@dataclass(frozen=True)
class CompactionOutcome:
    message_id: str
    tokens_after: int
    images_after: int


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
    # ``reminded``: a ``completion="terminal"`` agent answered without tool
    # calls, the activity appended the reminder, and the run continues.
    # ``stop_reason``: the run ends as exhausted -- that agent answered without
    # tool calls twice, or the worker's turn gate refused the turn.
    reminded: bool = False
    stop_reason: str | None = None
    # The request this turn would send crosses a compaction limit. Nothing
    # was sent: the workflow compacts, then runs the turn again.
    compaction: CompactionTrigger | None = None
    # The request passed ``CompactionConfig.background``: the turn ran, and the
    # workflow starts a summary beside the next turns if none is running.
    summarize: CompactionTrigger | None = None


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
    #: The thread's ``ThreadInput.context``.
    context: dict[str, Any] = field(default_factory=dict)


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
    #: None when read from the stores, which cannot see a running
    #: workflow's queue. Only a live workflow knows its own depth.
    inbox_size: int | None
    turn_count_total: int
    current_run_id: str | None
    cancelled: bool
