"""Tool interfaces."""

from actant.tools.admission import (
    ToolCallView,
    ToolDecision,
    ToolResolution,
    ToolWaitRequest,
)
from actant.tools.base import (
    BaseDeclarativeTool,
    BaseToolInvocation,
    Tool,
    ToolInvocation,
    ToolResult,
    ToolSchema,
    make_tool_schema,
)
from actant.tools.calls import ToolCallRecord, ToolCallStatus
from actant.tools.function import FunctionTool, FunctionToolInvocation, ToolArguments, tool
from actant.tools.registry import ToolRegistry
from actant.tools.supervise import SubagentSupervisor, supervision_tools
from actant.tools.task import (
    InMemorySubagentRegistry,
    SubagentInvoker,
    SubagentSpawner,
    TaskTool,
)

__all__ = [
    "InMemorySubagentRegistry",
    "BaseDeclarativeTool",
    "BaseToolInvocation",
    "FunctionTool",
    "FunctionToolInvocation",
    "SubagentInvoker",
    "SubagentSpawner",
    "SubagentSupervisor",
    "TaskTool",
    "Tool",
    "ToolArguments",
    "ToolInvocation",
    "ToolCallView",
    "ToolCallRecord",
    "ToolCallStatus",
    "ToolDecision",
    "ToolRegistry",
    "ToolResolution",
    "ToolResult",
    "ToolSchema",
    "ToolWaitRequest",
    "make_tool_schema",
    "supervision_tools",
    "tool",
]
