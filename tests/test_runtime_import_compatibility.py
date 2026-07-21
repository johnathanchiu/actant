"""Public runtime imports resolve to the reorganized implementation."""

from actant.runtime import (
    AgentRuntime,
    RunCompletion,
    RunCompletionHandler,
    TemporalRuntimeConfig,
    TemporalRuntimeWorker,
    ThreadHandle,
)
from actant.runtime.completion import RunCompletion as CanonicalCompletion
from actant.runtime.completion import RunCompletionHandler as CanonicalCompletionHandler
from actant.runtime.runtime import AgentRuntime as RuntimeModuleAgentRuntime
from actant.runtime.temporal import TemporalRuntimeClient
from actant.runtime.temporal.client import TemporalRuntimeClient as CanonicalClient
from actant.runtime.temporal.types import TemporalRuntimeConfig as CanonicalConfig
from actant.runtime.temporal.worker import TemporalRuntimeWorker as CanonicalWorker
from actant.runtime.thread import ThreadHandle as CanonicalThreadHandle


def test_public_runtime_imports_resolve_to_canonical_types() -> None:
    assert AgentRuntime is RuntimeModuleAgentRuntime
    assert TemporalRuntimeConfig is CanonicalConfig
    assert TemporalRuntimeWorker is CanonicalWorker
    assert TemporalRuntimeClient is CanonicalClient
    assert RunCompletion is CanonicalCompletion
    assert RunCompletionHandler is CanonicalCompletionHandler
    assert ThreadHandle is CanonicalThreadHandle
