"""Public runtime imports resolve to the reorganized implementation."""

from actant.runtime import AgentRuntime, TemporalRuntimeConfig, TemporalRuntimeWorker
from actant.runtime.runtime import AgentRuntime as RuntimeModuleAgentRuntime
from actant.runtime.temporal import TemporalRuntimeClient
from actant.runtime.temporal.client import TemporalRuntimeClient as CanonicalClient
from actant.runtime.temporal.types import TemporalRuntimeConfig as CanonicalConfig
from actant.runtime.temporal.worker import TemporalRuntimeWorker as CanonicalWorker


def test_public_runtime_imports_resolve_to_canonical_types() -> None:
    assert AgentRuntime is RuntimeModuleAgentRuntime
    assert TemporalRuntimeConfig is CanonicalConfig
    assert TemporalRuntimeWorker is CanonicalWorker
    assert TemporalRuntimeClient is CanonicalClient
