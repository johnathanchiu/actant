"""``recall_image``: show an image again after context compaction dropped it from view.

A compaction's summary lists notable images by id (an ``AssetBlock``'s
``asset_public_id``, else its storage key). This tool finds that block in the thread's
transcript and returns it, so the image is attached again from the asset store it is
already in: nothing is uploaded. Give it the runtime's ``stores.messages``.
"""

from __future__ import annotations

from typing import Protocol

from actant.messages import NO_IMAGE_WITH_ID
from actant.blocks import AssetBlock
from actant.core import JSONObject
from actant.llm.messages import Message
from actant.tools.base import (
    BaseDeclarativeTool,
    BaseToolInvocation,
    CallContext,
    ToolInvocation,
    ToolResult,
    make_tool_schema,
)


class TranscriptReader(Protocol):
    async def list_for_thread(self, agent_id: str, thread_id: str) -> list[Message]: ...


class _RecallImageInvocation(BaseToolInvocation[JSONObject, object]):
    def __init__(self, params: JSONObject, messages: TranscriptReader, ctx: CallContext) -> None:
        super().__init__(params)
        self._messages = messages
        self._ctx = ctx

    def get_description(self) -> str:
        return "Recalling an image"

    async def execute(self) -> ToolResult:
        wanted = self.params.get("id")
        history = await self._messages.list_for_thread(self._ctx.agent_id, self._ctx.thread_id)
        for message in reversed(history):
            for block in message.content if isinstance(message.content, list) else []:
                if isinstance(block, AssetBlock) and wanted in (
                    block.asset_public_id,
                    block.storage_key,
                ):
                    return ToolResult(output={"recalled": wanted}, content_blocks=[block])
        return ToolResult.fail(NO_IMAGE_WITH_ID.format(id=wanted))


class RecallImageTool(BaseDeclarativeTool):
    def __init__(self, messages: TranscriptReader) -> None:
        super().__init__(
            "recall_image",
            make_tool_schema(
                "recall_image",
                "Show an earlier image of this conversation again, by the id the "
                "compaction summary's image index gives for it.",
                parameters={"id": {"type": "string", "description": "The image's id."}},
                required=["id"],
            ),
        )
        self._messages = messages

    async def build(self, params: JSONObject, ctx: CallContext) -> ToolInvocation:
        return _RecallImageInvocation(params, self._messages, ctx)
