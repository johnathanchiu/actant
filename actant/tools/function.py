"""Function-backed tools for the common case."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from string import Formatter
from typing import TypeAlias, TypeVar, cast, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from actant.core import JSONObject
from actant.tools.admission import (
    ToolCallView,
    ToolDecision,
    ToolResolution,
    ToolWaitRequest,
    TurnContextView,
)
from actant.tools.base import BaseToolInvocation, ToolInvocation, ToolResult, ToolSchema

ToolFunction: TypeAlias = Callable[..., object]
ToolArguments: TypeAlias = dict[str, object]
AdmissionCallback: TypeAlias = Callable[[ToolArguments], ToolDecision | Awaitable[ToolDecision]]
ApprovalPrompt: TypeAlias = str | Callable[[ToolArguments], str | Awaitable[str]]
ResolutionCallback: TypeAlias = Callable[
    [ToolArguments, ToolResolution], object | Awaitable[object]
]
AwaitedT = TypeVar("AwaitedT")


class FunctionToolInvocation(BaseToolInvocation[ToolArguments, object]):
    """One validated invocation of a :class:`FunctionTool`."""

    def __init__(self, tool: FunctionTool, params: ToolArguments) -> None:
        super().__init__(params)
        self._tool = tool

    def get_description(self) -> str:
        return f"Running {self._tool.name}"

    async def execute(self) -> ToolResult:
        return await self._tool._execute(self.params)


class FunctionTool:
    """Adapt an annotated Python function to Actant's tool protocol."""

    def __init__(
        self,
        function: ToolFunction,
        *,
        name: str | None = None,
        description: str | None = None,
        approval: ApprovalPrompt | None = None,
        admission: AdmissionCallback | None = None,
        resolve: ResolutionCallback | None = None,
    ) -> None:
        if approval is not None and admission is not None:
            raise TypeError("Pass either `approval` or `admission`, not both")
        self.function = function
        self.name = name or function.__name__
        self.description = description or inspect.getdoc(function) or f"Run {self.name}."
        self.approval = approval
        self.admission = admission
        self.resolve = resolve
        self._params_model = _parameter_model(function, self.name)
        if isinstance(approval, str):
            _validate_approval_template(approval, set(self._params_model.model_fields))
        self._schema: ToolSchema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._params_model.model_json_schema(),
            },
        }

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    async def build(self, params: JSONObject) -> FunctionToolInvocation:
        return FunctionToolInvocation(self, self._validated_params(params))

    def _validated_params(self, params: JSONObject) -> ToolArguments:
        try:
            validated = self._params_model.model_validate(params)
        except ValidationError as exc:
            raise ValueError(f"Invalid arguments for {self.name}: {exc}") from exc
        return validated.model_dump()

    async def can_execute(
        self,
        call: ToolCallView,
        invocation: ToolInvocation,
        context: TurnContextView,
    ) -> ToolDecision:
        del context
        params = (
            invocation.params
            if isinstance(invocation, FunctionToolInvocation)
            else self._validated_params(call.args)
        )
        if self.approval is not None:
            prompt = (
                await _maybe_await(self.approval(params))
                if callable(self.approval)
                else self.approval.format_map(params)
            )
            return ToolDecision.wait(
                ToolWaitRequest(kind="approval", prompt=prompt, payload={"args": call.args})
            )
        if self.admission is not None:
            return await _maybe_await(self.admission(params))
        return ToolDecision.allow()

    async def on_resolve(
        self,
        call: ToolCallView,
        resolution: ToolResolution,
    ) -> ToolResult:
        if self.approval is not None:
            if resolution.approved is not True:
                return ToolResult.fail("Tool call was not approved")
            return await self._execute(self._validated_params(call.args))
        if self.resolve is not None:
            return _as_result(
                await _invoke_resolution(
                    self.resolve,
                    self._validated_params(call.args),
                    resolution,
                )
            )
        output: dict[str, object] = {
            "approved": resolution.approved,
            "answer": resolution.answer,
        }
        output.update(resolution.payload)
        return ToolResult.ok(output)

    async def _execute(self, params: ToolArguments) -> ToolResult:
        if inspect.iscoroutinefunction(self.function):
            output = await self.function(**params)
        else:
            output = await asyncio.to_thread(self.function, **params)
        return _as_result(output)


@overload
def tool(function: ToolFunction, /) -> FunctionTool: ...


@overload
def tool(
    function: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    approval: ApprovalPrompt | None = None,
    admission: AdmissionCallback | None = None,
    resolve: ResolutionCallback | None = None,
) -> Callable[[ToolFunction], FunctionTool]: ...


def tool(
    function: ToolFunction | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    approval: ApprovalPrompt | None = None,
    admission: AdmissionCallback | None = None,
    resolve: ResolutionCallback | None = None,
) -> FunctionTool | Callable[[ToolFunction], FunctionTool]:
    """Create an Actant tool from an annotated sync or async function."""

    def create(candidate: ToolFunction) -> FunctionTool:
        return FunctionTool(
            candidate,
            name=name,
            description=description,
            approval=approval,
            admission=admission,
            resolve=resolve,
        )

    return create(function) if function is not None else create


def _parameter_model(function: ToolFunction, tool_name: str) -> type[BaseModel]:
    signature = inspect.signature(function)
    hints = get_type_hints(function, include_extras=True)
    fields: dict[str, tuple[object, object]] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise TypeError(
                f"Tool {tool_name!r} must use named parameters without *args or **kwargs"
            )
        annotation = hints.get(parameter.name, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"Tool parameter {parameter.name!r} requires a type annotation")
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (annotation, default)
    # Pydantic intentionally accepts dynamic field definitions through
    # ``**fields``; its static overloads cannot express a runtime signature.
    return create_model(  # pyright: ignore[reportCallIssue, reportArgumentType]
        f"{tool_name.title().replace('_', '')}Params",
        __config__=ConfigDict(extra="forbid"),
        **fields,  # pyright: ignore[reportArgumentType]
    )


def _validate_approval_template(template: str, parameter_names: set[str]) -> None:
    for _literal, field_name, _format_spec, _conversion in Formatter().parse(template):
        if field_name is None:
            continue
        if not field_name:
            raise ValueError("Approval templates must use named fields such as `{title}`")
        if "." in field_name or "[" in field_name:
            raise ValueError("Approval templates may reference only direct tool parameters")
        if field_name not in parameter_names:
            raise ValueError(f"Approval template references unknown tool parameter {field_name!r}")


async def _invoke_resolution(
    callback: ResolutionCallback,
    args: ToolArguments,
    resolution: ToolResolution,
) -> object:
    return await _maybe_await(callback(args, resolution))


async def _maybe_await(value: AwaitedT | Awaitable[AwaitedT]) -> AwaitedT:
    if inspect.isawaitable(value):
        return await cast(Awaitable[AwaitedT], value)
    return value


def _as_result(output: object) -> ToolResult:
    return output if isinstance(output, ToolResult) else ToolResult.ok(output)
