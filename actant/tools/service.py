"""Expose a service (:mod:`actant.sandbox.service`) to a model as tools.

Each public method becomes one tool: the schema comes from its signature without
``self``, the description from its docstring. :func:`tools` runs the methods
through any :class:`~actant.sandbox.service.Runner`; every runner encodes results
the same way, so switching between them changes where the code runs and nothing
the model sees. Images arrive as bytes and are stored in the worker's own
:class:`~actant.assets.AssetStore` before the result goes anywhere, so a result carries
storage keys, never bytes (:func:`to_tool_result`).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import mimetypes

from pydantic import ValidationError

from actant.assets import AssetStore
from actant.blocks import AssetBlock, Base64Source, Block, InlineImageBlock, TextBlock
from actant.core import JSONObject
import actant.sandbox.host as host
from actant.sandbox.protocol import CallResponse
from actant.sandbox.service import LocalRunner, Runner
from actant.tools.base import BaseToolInvocation, CallContext, MetadataKey, ToolResult, ToolSchema

#: The most image bytes (decoded) one service response may carry.
MAX_IMAGE_BYTES = 16 * 1024 * 1024


async def to_tool_result(
    response: CallResponse,
    *,
    thread_id: str,
    store: AssetStore | None,
    max_image_bytes: int = MAX_IMAGE_BYTES,
) -> ToolResult:
    """A host response as a :class:`ToolResult`.

    Each image is stored in ``store`` under ``<thread_id>/images/<sha256><ext>`` (the same
    bytes once) and becomes an :class:`AssetBlock` with the store's key; without a store it
    stays inline. A response whose images total more than ``max_image_bytes`` fails.

    A response the method marked ``terminal`` is a terminal result: the run ends after
    this tool group, with no further model turn. A host that pushes storage reports a :class:`~actant.sandbox.protocol.StorageStatus`;
    its JSON lands on ``metadata[MetadataKey.STORAGE]`` so a product can warn when
    pushes fail, without the call itself failing.
    """
    result = await _result(response, thread_id, store, max_image_bytes)
    if response.terminal and response.error is None:
        result.metadata[MetadataKey.TERMINAL] = True
    if response.storage is not None:
        result.metadata[MetadataKey.STORAGE] = response.storage.model_dump(mode="json")
    return result


async def _result(
    response: CallResponse, thread_id: str, store: AssetStore | None, max_image_bytes: int
) -> ToolResult:
    text = response.text
    if response.error is not None:
        return ToolResult.fail("\n".join(part for part in (response.error, text) if part))
    if not response.images:
        return ToolResult.ok(text)
    # The decoded size, from the base64 length, before decoding any of it.
    size = sum(_decoded_size(image.source.data_b64) for image in response.images)
    if size > max_image_bytes:
        return ToolResult.fail(
            f"the service returned {len(response.images)} images of {size} bytes, "
            f"over the {max_image_bytes}-byte limit for one response; return fewer or "
            "smaller images"
        )
    images = (
        await _store(response, thread_id, store)
        if store is not None
        else [
            InlineImageBlock(
                source=Base64Source(media_type=image.media_type, data=image.source.data_b64)
            )
            for image in response.images
        ]
    )
    # LLM APIs reject empty text blocks.
    blocks: list[Block] = [TextBlock(text=text)] if text else []
    for image, block in zip(response.images, images, strict=True):
        blocks.append(TextBlock(text=image.caption or f"Image {image.name}:"))
        blocks.append(block)
    return ToolResult(output=text, content_blocks=blocks)


def _decoded_size(data_b64: str) -> int:
    return len(data_b64) * 3 // 4 - data_b64[-2:].count("=")


async def _store(response: CallResponse, thread_id: str, store: AssetStore) -> list[Block]:
    """Each image in ``store``, content-addressed: the same bytes are written once."""
    keys: dict[str, tuple[bytes, str]] = {}
    named: list[tuple[str, str]] = []
    for image in response.images:
        data = base64.b64decode(image.source.data_b64, validate=True)
        ext = mimetypes.guess_extension(image.media_type) or ""
        key = f"{thread_id}/images/{hashlib.sha256(data).hexdigest()}{ext}"
        keys.setdefault(key, (data, image.media_type))
        named.append((key, image.media_type))
    stored = await asyncio.gather(*(store.put(k, d, m) for k, (d, m) in keys.items()))
    full = dict(zip(keys, stored, strict=True))
    return [AssetBlock(storage_key=full[key], mime=mime) for key, mime in named]


class ServiceInvocation(BaseToolInvocation[dict[str, object], ToolResult]):
    def __init__(self, tool: ServiceTool, params: dict[str, object], ctx: CallContext) -> None:
        super().__init__(params)
        self._tool = tool
        self._ctx = ctx

    def get_description(self) -> str:
        return f"Running {self._tool.name}"

    async def execute(self) -> ToolResult:
        ctx = self._ctx
        response = await self._tool.runner.call(
            self._tool.name,
            self.params,
            key=ctx.thread_id,
            sandbox=ctx.sandbox,
            call_id=f"{ctx.thread_id}:{ctx.tool_call_id}",
        )
        return await to_tool_result(
            response,
            thread_id=ctx.thread_id,
            store=ctx.asset_store,
            max_image_bytes=self._tool.max_image_bytes,
        )


class ServiceTool:
    """One service method as an actant :class:`~actant.tools.base.Tool`.

    A call's id is its thread and tool call, so a call whose worker was lost is sent again
    under the same id and a host waits for the run it has instead of starting another:
    ``retry_safe`` for every runner but :class:`LocalRunner`, which died with the worker."""

    def __init__(
        self, cls: type, method: str, runner: Runner, max_image_bytes: int = MAX_IMAGE_BYTES
    ) -> None:
        self.name = method
        self.runner = runner
        self.max_image_bytes = max_image_bytes
        self.needs_sandbox = runner.needs_sandbox
        self.retry_safe = not isinstance(runner, LocalRunner)
        self._model = host.parameters_model(cls, method)
        description = inspect.getdoc(getattr(cls, method)) or f"Run {method}."
        self._schema: ToolSchema = {
            "type": "function",
            "function": {
                "name": method,
                "description": description,
                "parameters": self._model.model_json_schema(),
            },
        }

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    async def build(self, params: JSONObject, ctx: CallContext) -> ServiceInvocation:
        try:
            validated = self._model.model_validate(params).model_dump(mode="json")
        except ValidationError as exc:
            raise ValueError(f"Invalid arguments for {self.name}: {exc}") from exc
        return ServiceInvocation(self, validated, ctx)


def tools(
    cls: type, runner: Runner, *, max_image_bytes: int = MAX_IMAGE_BYTES
) -> list[ServiceTool]:
    """The service ``cls``'s methods as tools, each run through ``runner``; one call's
    images may total ``max_image_bytes``."""
    return [
        ServiceTool(cls, method, runner, max_image_bytes) for method in host.service_methods(cls)
    ]


def tool_schemas(cls: type) -> list[ToolSchema]:
    """The tool schemas of the service ``cls``, without choosing where they run."""
    return [tool.schema for tool in tools(cls, LocalRunner(None))]
