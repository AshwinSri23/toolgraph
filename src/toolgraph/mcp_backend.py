"""A backend that talks to real MCP servers through the official Python SDK.

Plain ``tools/call`` (including ``InputRequiredResult`` round trips) and the
MCP Tasks extension (``io.modelcontextprotocol/tasks``, protocol 2026-07-28)
are both tested against real servers: in-process SDK servers, and the official
"everything" reference server over Streamable HTTP.

Clients must declare the tasks extension for servers to hand out task handles::

    Client(url, **client_options())
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Literal

from mcp import Client
from mcp.client.extension import ClaimContext, ClientExtension, ResultClaim
from mcp.shared.exceptions import MCPError
from mcp_types import (
    CallToolResult,
    ElicitResult,
    ErrorData,
    InputRequiredResult,
    Request,
    RequestParams,
    Result,
    TextContent,
)
from mcp_types.jsonrpc import CONNECTION_CLOSED, INTERNAL_ERROR, INVALID_PARAMS, REQUEST_TIMEOUT
from pydantic import ConfigDict, TypeAdapter

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
from .policy import PolicyRegistry, SideEffect, ToolPolicy
from .scheduler import KEY_META

TASKS_EXTENSION = "io.modelcontextprotocol/tasks"

# Errors after which the server may or may not have acted on the request.
UNCERTAIN_CODES = frozenset({CONNECTION_CLOSED, REQUEST_TIMEOUT, INTERNAL_ERROR})

_RAW = TypeAdapter(dict[str, Any])


class TaskHandle(Result):
    """A ``tools/call`` answered with a task (``resultType: "task"``) instead of a result."""

    model_config = ConfigDict(extra="allow")

    result_type: Literal["task"] = "task"
    task_id: str
    status: str = "working"
    poll_interval_ms: int | None = None


async def _unresolved(handle: TaskHandle, ctx: ClaimContext) -> CallToolResult:
    # toolgraph records the handle and polls it itself, durably; it never asks the SDK to.
    raise RuntimeError("toolgraph task handles are not resolved by the SDK")


class TasksExtension(ClientExtension):
    """Declares the tasks extension and lets ``tools/call`` return task handles."""

    identifier = TASKS_EXTENSION

    def claims(self) -> list[ResultClaim[Any]]:
        return [ResultClaim(result_type="task", model=TaskHandle, resolve=_unresolved)]


async def _input_arrives_in_band(context: Any, params: Any) -> ErrorData:
    """Elicitation callback that declares form support but refuses mid-call pushes.

    Questions reach toolgraph in band (``InputRequiredResult`` or a task's
    ``input_required`` status) so that they survive restarts.
    """
    return ErrorData(code=INVALID_PARAMS, message="toolgraph answers input requests in band only")


def client_options() -> dict[str, Any]:
    """Keyword arguments for ``mcp.Client`` that toolgraph's MCP backend expects."""
    return {"extensions": [TasksExtension()], "elicitation_callback": _input_arrives_in_band}


class _TaskIdParams(RequestParams):
    task_id: str


class GetTask(Request[_TaskIdParams, Literal["tasks/get"]]):
    method: Literal["tasks/get"] = "tasks/get"
    params: _TaskIdParams
    name_param = "taskId"


class CancelTask(Request[_TaskIdParams, Literal["tasks/cancel"]]):
    method: Literal["tasks/cancel"] = "tasks/cancel"
    params: _TaskIdParams
    name_param = "taskId"


class _UpdateParams(RequestParams):
    task_id: str
    input_responses: dict[str, Any]


class UpdateTask(Request[_UpdateParams, Literal["tasks/update"]]):
    """Answer the input requests of a task that is ``input_required``."""

    method: Literal["tasks/update"] = "tasks/update"
    params: _UpdateParams
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


class Connection:
    """One MCP client that is reconnected after its transport fails.

    The SDK's HTTP transport tears down the whole client when a single POST
    loses its connection. Each client therefore lives in its own supervisor
    task: a transport failure ends that task (the call in flight sees an error
    or times out, and its outcome is treated as unknown) and the next call
    connects again.
    """

    def __init__(self, factory: Callable[[], Client]) -> None:
        self.factory = factory
        self.connects = 0
        self.failures: list[str] = []
        self._client: Client | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()

    async def client(self) -> Client:
        async with self._lock:
            if self._client is None or self._task is None or self._task.done():
                await self._open()
            assert self._client is not None
            return self._client

    async def _open(self) -> None:
        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._stop = asyncio.Event()
        self._task = asyncio.get_running_loop().create_task(self._supervise(ready))
        await ready

    async def _supervise(self, ready: asyncio.Future[None]) -> None:
        try:
            async with self.factory() as client:
                self._client = client
                self.connects += 1
                ready.set_result(None)
                await self._stop.wait()
        except Exception as e:  # the transport died, or we never connected
            self.failures.append(f"{type(e).__name__}: {e}")
            if not ready.done():
                # Nothing was sent, so this is a definite failure.
                ready.set_exception(Rejected(f"could not connect: {e}"))
        finally:
            self._client = None

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            except BaseException:
                pass


@asynccontextmanager
async def connect(
    factories: Mapping[str, Callable[[], Client]], **kwargs: Any
) -> AsyncIterator[MCPBackend]:
    """An ``MCPBackend`` over reconnecting connections, one per server::

        async with connect({"shop": lambda: Client(url, **client_options())}) as backend:
            ...
    """
    connections = {name: Connection(f) for name, f in factories.items()}
    try:
        yield MCPBackend(connections, **kwargs)
    finally:
        for c in connections.values():
            await c.close()


Target = Client | Connection


class MCPBackend:
    """Route toolgraph calls to one or more MCP servers.

    ``servers`` maps a server name to a connected ``mcp.Client`` or a
    :class:`Connection`; with several servers, tools are named
    ``"<server>.<tool>"``. Create clients with ``Client(..., **client_options())``
    so servers may answer with task handles; whether a call becomes a task is
    the server's decision. Prefer :func:`connect`, which reconnects.
    """

    def __init__(
        self,
        servers: Target | Mapping[str, Target],
        *,
        separator: str = ".",
        route: Callable[[str], tuple[Any, str]] | None = None,
    ) -> None:
        self.servers = servers
        self.separator = separator
        self._route = route

    def route(self, tool: str) -> tuple[Any, str]:
        if self._route is not None:
            return self._route(tool)
        if isinstance(self.servers, (Client, Connection)):
            return self.servers, tool
        server, _, name = tool.partition(self.separator)
        if server not in self.servers or not name:
            raise Rejected(f"no server for tool {tool!r}")
        return self.servers[server], name

    @staticmethod
    async def _session(target: Any) -> Any:
        if isinstance(target, Connection):
            target = await target.client()
        return target.session

    async def call(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        idempotency_key: str | None,
        answer: Any = None,
        question: Question | None = None,
    ) -> CallOutcome:
        target, name = self.route(tool)
        session = await self._session(target)
        meta = {KEY_META: idempotency_key} if idempotency_key else None
        input_responses = request_state = None
        if question is not None:
            input_responses = _responses(question, answer)
            request_state = question.state.get("request_state")
        try:
            result = await session.call_tool(
                name, args, meta=meta, input_responses=input_responses,  # type: ignore[arg-type]
                request_state=request_state, allow_input_required=True, allow_claimed=True,
            )
        except Exception as e:  # MCP errors, dropped connections, malformed replies
            raise _wrap(e) from e
        if isinstance(result, TaskHandle):
            poll = result.poll_interval_ms
            return Started(result.task_id, poll_interval=poll / 1000 if poll else None)
        if isinstance(result, InputRequiredResult):
            return NeedsInput(_question(result))
        if isinstance(result, CallToolResult):
            return _outcome(result)
        raise Uncertain(f"unexpected result type {type(result).__name__}")

    async def _task_request(self, tool: str, request: Request[Any, Any]) -> dict[str, Any]:
        target, _ = self.route(tool)
        session = await self._session(target)
        try:
            return await session.send_request(request, _RAW)
        except MCPError as e:
            if e.code == INVALID_PARAMS:
                raise TaskLost(e.error.message) from e
            raise _wrap(e) from e
        except Exception as e:
            raise _wrap(e) from e

    async def _get(self, tool: str, task_id: str) -> dict[str, Any]:
        return await self._task_request(tool, GetTask(params=_TaskIdParams(task_id=task_id)))

    async def get_task(self, tool: str, task_id: str) -> TaskStatus:
        raw = await self._get(tool, task_id)
        status = raw.get("status", "working")
        question = None
        if status == "input_required" and raw.get("inputRequests"):
            question = _question(InputRequiredResult.model_validate({"inputRequests": raw["inputRequests"]}))
        poll = raw.get("pollIntervalMs")
        return TaskStatus(status, raw.get("statusMessage"), question, poll / 1000 if poll else None)

    async def task_result(self, tool: str, task_id: str) -> Done | ToolFailed:
        # In the tasks extension the result rides on the completed status itself.
        raw = await self._get(tool, task_id)
        if raw.get("status") != "completed" or "result" not in raw:
            raise Rejected(f"task {task_id} has no result (status {raw.get('status')!r})")
        return _outcome(CallToolResult.model_validate(raw["result"]))

    async def answer_task(self, tool: str, task_id: str, question: Question, answer: Any) -> None:
        responses = {
            k: v.model_dump(mode="json", by_alias=True, exclude_none=True)
            for k, v in _responses(question, answer).items()
        }
        await self._task_request(tool, UpdateTask(params=_UpdateParams(task_id=task_id, input_responses=responses)))

    async def cancel_task(self, tool: str, task_id: str) -> None:
        await self._task_request(tool, CancelTask(params=_TaskIdParams(task_id=task_id)))


def _outcome(result: CallToolResult) -> Done | ToolFailed:
    if result.is_error:
        return ToolFailed(_error_text(result), result_value(result))
    return Done(result_value(result))


def policies_from_annotations(
    tools: Any, *, prefix: str = "", registry: PolicyRegistry | None = None, **policy: Any
) -> PolicyRegistry:
    """Register a policy per tool from its MCP annotations (``tools`` is a ``list_tools`` result).

    ``readOnlyHint`` maps to READ and ``idempotentHint`` to IDEMPOTENT_WRITE; any
    other tool is an UNKEYED_WRITE. Annotations are hints from the server, so
    only trust them from servers you trust, and register keyed writes yourself.
    Tools already in ``registry`` are left alone.
    """
    reg = registry or PolicyRegistry()
    for tool in getattr(tools, "tools", tools):
        name = prefix + tool.name
        if name in reg.policies:
            continue
        hints = tool.annotations
        if hints is not None and hints.read_only_hint:
            effect = SideEffect.READ
        elif hints is not None and hints.idempotent_hint:
            effect = SideEffect.IDEMPOTENT_WRITE
        else:
            effect = SideEffect.UNKEYED_WRITE
        reg.register(name, ToolPolicy(side_effect=effect, **policy))
    return reg
