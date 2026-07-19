from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

import httpx

from actant.core import JSONObject, JSONValue
from actant.tools.admission import (
    ToolDecision,
    ToolResolution,
    ToolWaitRequest,
)
from actant.tools.base import (
    BaseDeclarativeTool,
    BaseToolInvocation,
    ToolInvocation,
    ToolResult,
    make_tool_schema,
)


class _GetCurrentTimeInvocation(BaseToolInvocation[None, str]):
    def get_description(self) -> str:
        return "Reading the current UTC time"

    async def execute(self) -> ToolResult:
        return ToolResult.ok(datetime.now(timezone.utc).isoformat())


class GetCurrentTimeTool(BaseDeclarativeTool):
    def __init__(self) -> None:
        super().__init__(
            name="get_current_time",
            schema=make_tool_schema(
                name="get_current_time",
                description="Return the current UTC time as an ISO-8601 string.",
            ),
        )

    async def build(self, params: JSONObject) -> ToolInvocation:
        return _GetCurrentTimeInvocation(None)


class _FetchUrlInvocation(BaseToolInvocation[dict[str, Any], str]):
    def get_description(self) -> str:
        url = self.params.get("url", "")
        return f"Fetching {url}"

    async def execute(self) -> ToolResult:
        url = str(self.params.get("url", ""))
        if not url:
            return ToolResult.fail("url is required")
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                body = response.text
                if len(body) > 8000:
                    body = body[:8000] + "\n... [truncated]"
                return ToolResult.ok(body)
        except httpx.HTTPError as exc:
            return ToolResult.fail(f"fetch failed: {exc}")


class FetchUrlTool(BaseDeclarativeTool):
    def __init__(self) -> None:
        super().__init__(
            name="fetch_url",
            schema=make_tool_schema(
                name="fetch_url",
                description="GET a URL and return the response body as text (truncated to 8KB).",
                parameters={
                    "url": {
                        "type": "string",
                        "description": "The fully-qualified URL to fetch.",
                    },
                },
                required=["url"],
            ),
        )

    async def build(self, params: JSONObject) -> ToolInvocation:
        return _FetchUrlInvocation(dict(params))


class _DeferredInvocation(BaseToolInvocation[dict[str, Any], str]):
    """Placeholder invocation for deferred tools — never actually executed.

    When ``can_execute`` returns WAIT, the runtime parks the call and
    invokes ``on_resolve`` later instead of ``execute``. We still need
    a buildable invocation so the admission pipeline has something to
    inspect (``get_description`` shows up in the SSE wait_request).
    """

    def __init__(self, params: dict[str, Any], description: str) -> None:
        super().__init__(params)
        self._description = description

    def get_description(self) -> str:
        return self._description

    async def execute(self) -> ToolResult:
        # Unreachable in practice — WAIT decisions skip execute() entirely.
        return ToolResult.fail("deferred tool was executed without resolution")


class RequestApprovalTool(BaseDeclarativeTool):
    """Demonstrates the admission/resolve flow with a yes-or-no gate.

    The agent calls this to request human approval for a sensitive action.
    The runtime parks the tool call as a Temporal async activity (zero
    compute while waiting) and emits ``on_tool_waiting`` over SSE. The
    UI's deferred panel renders Approve / Deny buttons; clicking either
    posts to ``resolve_tool`` with ``approved=true/false``, and
    ``on_resolve`` here turns that into a real ToolResult.
    """

    def __init__(self) -> None:
        super().__init__(
            name="request_approval",
            schema=make_tool_schema(
                name="request_approval",
                description=(
                    "Ask the human for explicit approval before performing a "
                    "sensitive or irreversible action. Returns whether the "
                    "user approved. Use this when the user has asked you to "
                    "do something destructive, costly, or out-of-policy."
                ),
                parameters={
                    "action": {
                        "type": "string",
                        "description": (
                            "Short, concrete description of the action you "
                            "want approved (e.g. 'delete the file foo.txt')."
                        ),
                    },
                },
                required=["action"],
            ),
        )

    async def build(self, params: JSONObject) -> ToolInvocation:
        action = str(params.get("action", ""))
        return _DeferredInvocation(dict(params), f"Awaiting approval: {action}")

    async def can_execute(self, call, invocation, context):  # type: ignore[no-untyped-def]
        action = str(call.args.get("action", ""))
        return ToolDecision.wait(
            ToolWaitRequest(
                kind="approval",
                prompt=f"Approve action: {action}",
                payload={"action": action},
            )
        )

    async def on_resolve(self, call, resolution: ToolResolution) -> ToolResult:
        if resolution.approved is True:
            return ToolResult.ok({"approved": True, "action": call.args.get("action")})
        return ToolResult.fail("user denied the action")


class AskUserTool(BaseDeclarativeTool):
    """Pause and ask the human a multiple-choice question.

    The agent provides a question + 2 to 5 plausible options. The UI
    renders one button per option; the user clicks one. ``answer``
    carries the chosen option text verbatim.

    Multiple-choice (rather than free-form text) gives the demo a
    tighter, more deterministic UX — the agent has to think about
    what answers it actually needs, and the user doesn't have to type.
    """

    def __init__(self) -> None:
        super().__init__(
            name="ask_user",
            schema=make_tool_schema(
                name="ask_user",
                description=(
                    "Pause and ask the human a multiple-choice clarifying "
                    "question when you need information that isn't in the "
                    "conversation. Provide 2-5 plausible options as "
                    "concise strings; the user picks one. The chosen "
                    "option text is returned as the result. Use this when "
                    "you'd otherwise be guessing among a small set of "
                    "possibilities."
                ),
                parameters={
                    "question": {
                        "type": "string",
                        "description": "The question to ask the user.",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "2-5 short, concrete options for the user to "
                            "pick from. Each option should be a self-"
                            "contained answer phrase, not a sentence."
                        ),
                        "minItems": 2,
                        "maxItems": 5,
                    },
                },
                required=["question", "options"],
            ),
        )

    async def build(self, params: JSONObject) -> ToolInvocation:
        question = str(params.get("question", ""))
        return _DeferredInvocation(dict(params), f"Awaiting choice: {question}")

    async def can_execute(self, call, invocation, context):  # type: ignore[no-untyped-def]
        question = str(call.args.get("question", ""))
        raw_options = call.args.get("options", [])
        options = [str(o) for o in raw_options if isinstance(o, str) and o.strip()]
        if len(options) < 2:
            return ToolDecision.block(
                reason="`ask_user` requires at least 2 options for the user to choose from."
            )
        return ToolDecision.wait(
            ToolWaitRequest(
                kind="multiple_choice",
                prompt=question,
                payload={
                    "question": question,
                    "options": cast(list[JSONValue], options),
                },
            )
        )

    async def on_resolve(self, call, resolution: ToolResolution) -> ToolResult:
        answer = resolution.answer.strip()
        if not answer:
            return ToolResult.fail("user did not pick an option")
        return ToolResult.ok(answer)


def demo_tools() -> list[BaseDeclarativeTool]:
    return [GetCurrentTimeTool(), FetchUrlTool(), RequestApprovalTool(), AskUserTool()]
