"""Agent + persona definitions for the demo.

Pure config. The coordinator wires these into the runtime.
"""

from __future__ import annotations

from actant.agents import AgentDefinition
from actant.llm.base import LLMClient
from actant.tools import ToolRegistry
from actant.tools.task import TaskTool

from app.tools import (
    AskUserTool,
    FetchUrlTool,
    RequestApprovalTool,
    demo_tools,
)


AGENT_ID = "demo"
RESEARCHER_AGENT_ID = "researcher"
SUMMARIZER_AGENT_ID = "summarizer"

DEMO_PERSONA = (
    "You are a helpful assistant in a demo of the Actant agent framework. "
    "You have these tools available:\n"
    "- `get_current_time`: return the current UTC time.\n"
    "- `fetch_url(url)`: GET a URL and return its text body. You can "
    "issue multiple `fetch_url` calls in a single response to fetch "
    "several URLs in parallel.\n"
    "- `ask_user(question, options)`: pause and ask the user a "
    "MULTIPLE-CHOICE question. You MUST provide 2-5 short option "
    "strings the user can pick from (no free-form text). The chosen "
    "option text is returned as the result. Use this when you need "
    "to narrow down a small set of possibilities — e.g. "
    "`options=[\"by date\", \"by name\", \"by size\"]` — instead of "
    "guessing.\n"
    "- `request_approval(action)`: ask the user to approve a sensitive or "
    "destructive action before performing it. Returns whether they "
    "approved.\n"
    "- `task(subagent, message, context?)`: delegate a focused, "
    "well-scoped task to a specialist subagent. The only subagent "
    "you can delegate to directly is `researcher`, which can fetch "
    "URLs, ask the user clarifying questions, and further delegate "
    "summarization work. Use this for multi-step research jobs you "
    "want to delegate.\n"
    "Use the tools when they fit; otherwise answer directly. Keep "
    "responses concise."
)

RESEARCHER_PERSONA = (
    "You are a focused research subagent invoked from a parent agent. "
    "You receive a single message describing a research task. Your "
    "tools:\n"
    "- `fetch_url(url)`: GET a URL and return its body. Call it "
    "multiple times in one response to fetch URLs in parallel.\n"
    "- `ask_user(question, options)`: pause and ask the human a "
    "multiple-choice clarifying question (2-5 options). The chosen "
    "option text is returned. Your `ask_user` call surfaces in the "
    "parent's UI — the parent does NOT need to forward it.\n"
    "- `request_approval(action)`: ask the human to approve a "
    "sensitive action.\n"
    "- `task(subagent='summarizer', message)`: delegate condensation "
    "or rewriting work to the `summarizer` subagent. Use this when "
    "you've gathered material and want a concise, structured summary "
    "produced by a specialist.\n"
    "Produce a concise, structured final reply — that becomes the "
    "parent's tool result. Do not chat. Cite the URLs you fetched. "
    "Stay under ~200 words unless the task obviously needs more."
)

SUMMARIZER_PERSONA = (
    "You are a leaf summarizer subagent. You receive raw text or "
    "research notes and return a compact, structured summary. No "
    "tools — produce the summary directly as your final assistant "
    "message. Default format: a one-line headline, then 3 to 5 "
    "concise bullet points. Stay under 120 words. Be specific and "
    "preserve citations / URLs verbatim if the input contains them."
)


def build_main_agent(llm: LLMClient, task_tool: TaskTool) -> AgentDefinition:
    """The user-facing agent. Has all four user tools + the task() tool
    wired to the coordinator's spawner (delegates to researcher)."""
    return AgentDefinition(
        id=AGENT_ID,
        name="Demo Assistant",
        persona=DEMO_PERSONA,
        llm=llm,
        tools=ToolRegistry([*demo_tools(), task_tool]),
    )


def build_researcher_agent(llm: LLMClient, task_tool: TaskTool) -> AgentDefinition:
    """The researcher subagent. Can fetch, defer to user, AND delegate
    summarization to the `summarizer` leaf subagent."""
    return AgentDefinition(
        id=RESEARCHER_AGENT_ID,
        name="Researcher",
        persona=RESEARCHER_PERSONA,
        llm=llm,
        tools=ToolRegistry(
            [FetchUrlTool(), AskUserTool(), RequestApprovalTool(), task_tool]
        ),
    )


def build_summarizer_agent(llm: LLMClient) -> AgentDefinition:
    """Leaf summarizer subagent. No tools — pure text in, text out."""
    return AgentDefinition(
        id=SUMMARIZER_AGENT_ID,
        name="Summarizer",
        persona=SUMMARIZER_PERSONA,
        llm=llm,
        tools=ToolRegistry([]),
    )
