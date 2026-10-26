"""A backend that talks to real MCP servers through the official Python SDK.

Plain ``tools/call`` (including ``InputRequiredResult`` round trips) is tested
end to end against the SDK. MCP Tasks are experimental: ``tasks/get``,
``tasks/result`` and ``tasks/cancel`` follow the SDK types, and ``tasks/update``
(sending input to a paused task) has not been tested against a real server.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Mapping
from typing import Any, Literal

from mcp import Client
from mcp.shared.exceptions import MCPError
from mcp_types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    CancelTaskRequest,
    CancelTaskRequestParams,
    CreateTaskResult,
    ElicitResult,
    GetTaskPayloadRequest,
    GetTaskPayloadRequestParams,
    GetTaskRequest,
    GetTaskRequestParams,
    InputRequiredResult,
    Request,
    RequestParams,
    TaskMetadata,
    TextContent,
)
from mcp_types.jsonrpc import CONNECTION_CLOSED, INTERNAL_ERROR, INVALID_PARAMS, REQUEST_TIMEOUT
from pydantic import TypeAdapter, ValidationError

from .backend import (
    CallOutcome,
    Done,
    NeedsInput,
    Question,
    Rejected,
    Started,
    TaskLost,
    TaskStatus,
    ToolFailed,
    Uncertain,
)
from .scheduler import KEY_META

# Errors after which the server may or may not have acted on the request.
UNCERTAIN_CODES = frozenset({CONNECTION_CLOSED, REQUEST_TIMEOUT, INTERNAL_ERROR})

_RAW = TypeAdapter(dict[str, Any])
_CREATE_OR_RESULT = TypeAdapter(CreateTaskResult | CallToolResult)


class UpdateTaskRequestParams(RequestParams):
    task_id: str
    input_responses: dict[str, Any]


class UpdateTaskRequest(Request[UpdateTaskRequestParams, Literal["tasks/update"]]):
    """Send input to a task that is waiting for it (tasks extension). Untested against a real server."""

    method: Literal["tasks/update"] = "tasks/update"
    params: UpdateTaskRequestParams
    name_param = "taskId"


def result_value(result: CallToolResult) -> Any:
    """The JSON value a node's result is taken to be (what refs index into)."""
    if result.structured_content is not None:
        sc = result.structured_content
        # MCPServer wraps non-object return values as {"result": ...}.
        if set(sc) == {"result"}:
            return sc["result"]
        return sc
    texts = [c.text for c in result.content if isinstance(c, TextContent)]
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except ValueError:
            return texts[0]
    return texts


def _error_text(result: CallToolResult) -> str:
    texts = [c.text for c in result.content if isinstance(c, TextContent)]
    return "; ".join(texts) or "tool error"


def _question(result: InputRequiredResult) -> Question:
    requests = {k: v.model_dump(mode="json", by_alias=True, exclude_none=True) for k, v in (result.input_requests or {}).items()}
    prompts = [r.get("params", {}).get("message", "") for r in requests.values()]
    schema = None
    if len(requests) == 1:
        schema = next(iter(requests.values())).get("params", {}).get("requestedSchema")
    return Question(
        prompt="\n".join(p for p in prompts if p) or "input required",
        schema=schema,
        state={"input_requests": requests, "request_state": result.request_state},
    )


def _responses(question: Question, answer: Any) -> dict[str, ElicitResult]:
    """Turn a person's answer into one response per pending input request.

    ``answer`` is either the form content for a single request, or a mapping
    from request key to content. Use ``{"action": "decline"}`` to decline.
    """
    keys = list(question.state.get("input_requests", {}))
    if len(keys) == 1 or not (isinstance(answer, Mapping) and set(answer) <= set(keys)):
        per_key = {k: answer for k in keys}
    else:
        per_key = dict(answer)
    out = {}
    for k, v in per_key.items():
        if isinstance(v, Mapping) and v.get("action") in ("decline", "cancel"):
            out[k] = ElicitResult(action=v["action"])
        else:
            out[k] = ElicitResult(action="accept", content=v if isinstance(v, Mapping) else {"value": v})
    return out


def _wrap(exc: BaseException) -> Exception:
    if isinstance(exc, MCPError):
        if exc.code in UNCERTAIN_CODES:
            return Uncertain(f"MCP error {exc.code}: {exc.error.message}")
        return Rejected(f"MCP error {exc.code}: {exc.error.message}")
    return Uncertain(f"{type(exc).__name__}: {exc}")


_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (OSError, ValidationError, EOFError)
try:  # anyio stream errors when a connection drops
    import anyio

    _TRANSPORT_ERRORS += (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream)
except ImportError:  # pragma: no cover
    pass


class MCPBackend:
    """Route toolgraph calls to one or more connected ``mcp.Client`` instances.

    With several servers, tools are named ``"<server>.<tool>"``. Tools listed in
    ``task_tools`` are called as MCP Tasks (task-augmented ``tools/call``).
    """

    def __init__(
        self,
        servers: Client | Mapping[str, Client],
        *,
        separator: str = ".",
        task_tools: Collection[str] = (),
        task_ttl_ms: int | None = None,
        route: Callable[[str], tuple[Client, str]] | None = None,
    ) -> None:
        self.servers = servers
        self.separator = separator
        self.task_tools = set(task_tools)
        self.task_ttl_ms = task_ttl_ms
        self._route = route

    def route(self, tool: str) -> tuple[Client, str]:
        if self._route is not None:
            return self._route(tool)
        if isinstance(self.servers, Client):
            return self.servers, tool
        server, _, name = tool.partition(self.separator)
        if server not in self.servers or not name:
            raise Rejected(f"no server for tool {tool!r}")
        return self.servers[server], name

    async def call(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        idempotency_key: str | None,
        answer: Any = None,
        question: Question | None = None,
    ) -> CallOutcome:
        client, name = self.route(tool)
        meta = {KEY_META: idempotency_key} if idempotency_key else None
        input_responses = request_state = None
        if question is not None:
            input_responses = _responses(question, answer)
            request_state = question.state.get("request_state")
        try:
            if tool in self.task_tools:
                params = CallToolRequestParams(
                    name=name, arguments=args, task=TaskMetadata(ttl=self.task_ttl_ms), _meta=meta,
                    input_responses=input_responses, request_state=request_state,
                )
                result: Any = await client.session.send_request(CallToolRequest(params=params), _CREATE_OR_RESULT)
            else:
                result = await client.session.call_tool(
                    name, args, meta=meta, input_responses=input_responses,  # type: ignore[arg-type]
                    request_state=request_state, allow_input_required=True,
                )
        except MCPError as e:
            raise _wrap(e) from e
        except _TRANSPORT_ERRORS as e:
            raise _wrap(e) from e
        if isinstance(result, CreateTaskResult):
            poll = result.task.poll_interval
            return Started(result.task.task_id, poll_interval=poll / 1000 if poll else None)
        if isinstance(result, InputRequiredResult):
            return NeedsInput(_question(result))
        if isinstance(result, CallToolResult):
            if result.is_error:
                return ToolFailed(_error_text(result), result_value(result))
            return Done(result_value(result))
        raise Uncertain(f"unexpected result type {type(result).__name__}")

    async def _task_request(self, tool: str, request: Request[Any, Any]) -> dict[str, Any]:
        client, _ = self.route(tool)
        try:
            return await client.session.send_request(request, _RAW)
        except MCPError as e:
            if e.code == INVALID_PARAMS:
                raise TaskLost(e.error.message) from e
            raise _wrap(e) from e
        except _TRANSPORT_ERRORS as e:
            raise _wrap(e) from e

    async def get_task(self, tool: str, task_id: str) -> TaskStatus:
        raw = await self._task_request(tool, GetTaskRequest(params=GetTaskRequestParams(task_id=task_id)))
        status = raw.get("status", "working")
        question = None
        if status == "input_required" and raw.get("inputRequests"):
            ir = InputRequiredResult.model_validate({"inputRequests": raw["inputRequests"]})
            question = _question(ir)
        poll = raw.get("pollInterval")
        return TaskStatus(status, raw.get("statusMessage"), question, poll / 1000 if poll else None)

    async def task_result(self, tool: str, task_id: str) -> Done | ToolFailed:
        raw = await self._task_request(tool, GetTaskPayloadRequest(params=GetTaskPayloadRequestParams(task_id=task_id)))
        result = CallToolResult.model_validate(raw)
        if result.is_error:
            return ToolFailed(_error_text(result), result_value(result))
        return Done(result_value(result))

    async def answer_task(self, tool: str, task_id: str, question: Question, answer: Any) -> None:
        responses = {k: v.model_dump(mode="json", by_alias=True, exclude_none=True) for k, v in _responses(question, answer).items()}
        await self._task_request(
            tool, UpdateTaskRequest(params=UpdateTaskRequestParams(task_id=task_id, input_responses=responses))
        )

    async def cancel_task(self, tool: str, task_id: str) -> None:
        await self._task_request(tool, CancelTaskRequest(params=CancelTaskRequestParams(task_id=task_id)))
