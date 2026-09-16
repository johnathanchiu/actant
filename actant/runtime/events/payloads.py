"""Shared payload contracts for structured runtime events.

Tool outputs and provider metadata remain open JSON by design. Transport
validation belongs inside the observer failure boundary, never an activity's
side-effect path.
"""

from pydantic import BaseModel, ConfigDict, Field, JsonValue

JSONObject = dict[str, JsonValue]


class ActivityPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    agent_id: str | None = None
    run_id: str | None = None
    turn_id: str | None = None
    turn_uid: str | None = None
    turn_index: int | None = None


class AssistantMessagePayload(ActivityPayload):
    content: str | list[JSONObject] | None = None
    thought_summary: str | None = None
    tool_calls: list[JSONObject] = Field(default_factory=list)


class ToolResultData(BaseModel):
    tool_call_id: str | None = None
    result: JsonValue = None
    error: str | None = None
    metadata: JSONObject = Field(default_factory=dict)
    content_blocks: list[JSONObject] | None = None


class ToolResultPayload(ActivityPayload):
    tool_call_id: str
    result: ToolResultData
    output: str | None = None
    error: str | None = None


class WaitRequestData(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str
    payload: JSONObject = Field(default_factory=dict)


class ToolWaitingPayload(ActivityPayload):
    tool_call_id: str
    prompt: str
    wait_request: WaitRequestData | None = None
    wait_kind: str | None = None
    wait_payload: JSONObject = Field(default_factory=dict)


class ModelUsagePayload(ActivityPayload):
    response_id: str
    model: str
    usage: JSONObject
    status: str


PAYLOAD_MODELS: dict[str, type[ActivityPayload]] = {
    "assistant_message": AssistantMessagePayload,
    "tool_result": ToolResultPayload,
    "tool_resolved": ToolResultPayload,
    "tool_waiting": ToolWaitingPayload,
    "model_usage": ModelUsagePayload,
}
