"""Expose a service (:mod:`actant.sandbox.service`) to a model as tools.

Each public method becomes one tool: the schema comes from its signature without
``self``, the description from its docstring. :func:`tools` runs the methods
through any :class:`~actant.sandbox.service.Runner`; every runner encodes results
the same way, so switching between them changes where the code runs and nothing
the model sees. Images become the image content blocks the LLM adapters accept
(:func:`image_block`): a URL where the host presigned one, else base64 bytes.
"""

from __future__ import annotations

import inspect

from pydantic import ValidationError

from actant.core import JSONObject
import actant.sandbox.host as host
from actant.sandbox.protocol import CallResponse, Image, InlineSource
from actant.sandbox.service import LocalRunner, Runner
from actant.tools.base import BaseToolInvocation, CallContext, MetadataKey, ToolResult, ToolSchema


def image_block(image: Image) -> dict[str, object]:
    """An image as the content block a model API takes: a ``url`` source when the host
    presigned one, else ``base64`` bytes."""
    if isinstance(image.source, InlineSource):
        source: dict[str, object] = {
            "type": "base64",
            "media_type": image.media_type,
            "data": image.source.data_b64,
        }
    else:
        source = {"type": "url", "url": image.source.url}
    return {"type": "image", "source": source}


def to_tool_result(response: CallResponse) -> ToolResult:
    """A host response as a :class:`ToolResult`.

    A host that pushes storage reports a :class:`~actant.sandbox.protocol.StorageStatus`;
    its JSON lands on ``metadata[MetadataKey.STORAGE]`` so a product can warn when
    pushes fail, without the call itself failing.
    """
    result = _result(response)
    if response.storage is not None:
        result.metadata[MetadataKey.STORAGE] = response.storage.model_dump(mode="json")
    return result


def _result(response: CallResponse) -> ToolResult:
    text = response.text
    if response.error is not None:
        return ToolResult.fail("\n".join(part for part in (response.error, text) if part))
    if not response.images:
        return ToolResult.ok(text)
    # LLM APIs reject empty text blocks.
    blocks: list[dict[str, object]] = [{"type": "text", "text": text}] if text else []
    for image in response.images:
        blocks.append({"type": "text", "text": f"Image {image.name}:"})
        blocks.append(image_block(image))
    return ToolResult(output=text, content_blocks=blocks)


class ServiceInvocation(BaseToolInvocation[dict[str, object], ToolResult]):
    def __init__(self, tool: ServiceTool, params: dict[str, object], ctx: CallContext) -> None:
        super().__init__(params)
        self._tool = tool
        self._ctx = ctx

    def get_description(self) -> str:
        return f"Running {self._tool.name}"

    async def execute(self) -> ToolResult:
        response = await self._tool.runner.call(
            self._tool.name, self.params, key=self._ctx.thread_id, sandbox=self._ctx.sandbox
        )
        return to_tool_result(response)


class ServiceTool:
    """One service method as an actant :class:`~actant.tools.base.Tool`."""

    def __init__(self, cls: type, method: str, runner: Runner) -> None:
        self.name = method
        self.runner = runner
        self.needs_sandbox = runner.needs_sandbox
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


def tools(cls: type, runner: Runner) -> list[ServiceTool]:
    """The service ``cls``'s methods as tools, each run through ``runner``."""
    return [ServiceTool(cls, method, runner) for method in host.service_methods(cls)]


def tool_schemas(cls: type) -> list[ToolSchema]:
    """The tool schemas of the service ``cls``, without choosing where they run."""
    return [tool.schema for tool in tools(cls, LocalRunner(None))]
