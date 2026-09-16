"""Public runtime imports resolve to the reorganized implementation."""

from actant import FunctionTool, tool
from actant.runtime import (
    AgentRuntime,
    RunCompletion,
    RunCompletionHandler,
    TemporalRuntimeConfig,
    ThreadHandle,
    TurnGate,
    TurnStart,
)
from actant.runtime import gate
from actant.runtime.completion import RunCompletion as CanonicalCompletion
from actant.runtime.completion import RunCompletionHandler as CanonicalCompletionHandler
from actant.runtime.runtime import AgentRuntime as RuntimeModuleAgentRuntime
from actant.runtime.temporal.types import TemporalRuntimeConfig as CanonicalConfig
from actant.runtime.thread import ThreadHandle as CanonicalThreadHandle
from actant.tools import FunctionTool as CanonicalFunctionTool
from actant.tools import tool as canonical_tool


def test_public_runtime_imports_resolve_to_canonical_types() -> None:
    assert AgentRuntime is RuntimeModuleAgentRuntime
    assert TemporalRuntimeConfig is CanonicalConfig
    assert RunCompletion is CanonicalCompletion
    assert RunCompletionHandler is CanonicalCompletionHandler
    assert ThreadHandle is CanonicalThreadHandle
    assert (TurnGate, TurnStart) == (gate.TurnGate, gate.TurnStart)
    assert FunctionTool is CanonicalFunctionTool
    assert tool is canonical_tool
