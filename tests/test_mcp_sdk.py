"""End-to-end tests against the official MCP Python SDK (in-process servers)."""

import asyncio
import time

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402
from mcp.server.mcpserver import Context, MCPServer  # noqa: E402
from mcp.shared.exceptions import MCPError  # noqa: E402
from mcp_types import (  # noqa: E402
    CallToolResult,
    ElicitRequest,
    ElicitRequestFormParams,
    InputRequiredResult,
    TextContent,
)
from mcp_types.jsonrpc import CONNECTION_CLOSED, INVALID_PARAMS, METHOD_NOT_FOUND, REQUEST_TIMEOUT  # noqa: E402

from toolgraph import Node, PolicyRegistry, Rejected, Scheduler, SideEffect, Store, Uncertain, ref  # noqa: E402
from toolgraph import store as st  # noqa: E402
from toolgraph.backend import Question  # noqa: E402
from toolgraph.mcp_backend import MCPBackend, _wrap, result_value  # noqa: E402
from toolgraph.sim.harness import SimulatedCrash  # noqa: E402


def travel_servers():
    flights = MCPServer("flights")
    mail = MCPServer("mail")
    log = {"searches": 0, "sent": [], "booked": {}, "slow": []}

    @flights.tool()
    def search(city: str) -> dict:
        log["searches"] += 1
        return {"flight": f"TG{len(city)}", "city": city}

    @flights.tool()
    def book(flight: str, idempotency_key: str) -> dict:
        log["booked"].setdefault(idempotency_key, flight)
        return {"confirmation": f"C-{idempotency_key[-6:]}", "flight": log["booked"][idempotency_key]}

    @flights.tool()
    async def slow(n: int) -> int:
        log["slow"].append(n)
        await asyncio.sleep(0.3)
        return n

    @flights.tool()
    async def pay(amount: int, ctx: Context) -> CallToolResult | InputRequiredResult:
        if not ctx.input_responses:
            return InputRequiredResult(
                input_requests={
                    "otp": ElicitRequest(
                        params=ElicitRequestFormParams(
                            message="Enter the code we texted you",
                            requested_schema={"type": "object", "properties": {"code": {"type": "string"}}},
                        )
                    )
                },
                request_state="pay-1",
            )
        code = ctx.input_responses["otp"].content["code"]
        return CallToolResult(content=[TextContent(type="text", text=f'{{"paid": {amount}, "code": "{code}"}}')])

    @flights.tool()
    def flaky() -> str:
        raise RuntimeError("upstream down")

    @mail.tool()
    def send(to: str, body: str) -> str:
        log["sent"].append((to, body))
        return "sent"

    return flights, mail, log


def registry():
    reg = PolicyRegistry()
    reg.register("flights.search", side_effect=SideEffect.READ, backoff_base=0.01)
    reg.register("flights.slow", side_effect=SideEffect.READ)
    reg.register("flights.book", side_effect=SideEffect.KEYED_WRITE, backoff_base=0.01)
    reg.register("flights.pay", side_effect=SideEffect.KEYED_WRITE, key_arg=None)
    reg.register("flights.flaky", side_effect=SideEffect.READ, max_attempts=2, backoff_base=0.01)
    reg.register("mail.send", side_effect=SideEffect.UNKEYED_WRITE, writes=("inbox:{to}",))
    return reg


async def run_graph(path, nodes, *, hook=None, steps=None, run_id="r"):
    flights, mail, log = travel_servers() if not isinstance(nodes, tuple) else nodes[1]
    if isinstance(nodes, tuple):
        nodes = nodes[0]
    async with Client(flights) as fc, Client(mail) as mc:
        store = Store(path, hook=hook)
        sched = Scheduler(store, MCPBackend({"flights": fc, "mail": mc}), registry(), run_id=run_id)
        sched.start(nodes)
        if steps:
            await steps(sched)
        try:
            report = await sched.run()
        except SimulatedCrash:
            # Caught here: leaving the SDK's task groups would wrap it in an ExceptionGroup.
            report = None
        finally:
            store.close()
        return report, log


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "mcp.db")


def test_plain_calls_end_to_end(db):
    nodes = [
        Node("f", "flights.search", {"city": "Lisbon"}),
        Node("b", "flights.book", {"flight": ref("f", "flight")}),
        Node("e", "mail.send", {"to": "me", "body": ref("b", "confirmation")}),
    ]
    report, log = asyncio.run(run_graph(db, nodes))
    assert report.status == "completed"
    assert report.result("f") == {"flight": "TG6", "city": "Lisbon"}
    assert report.result("e") == "sent"
    assert log["sent"] == [("me", report.result("b")["confirmation"])]
    key = report.nodes["b"].idem_key
    assert log["booked"] == {key: "TG6"}


def test_independent_calls_overlap(db):
    nodes = [Node(f"s{i}", "flights.slow", {"n": i}) for i in range(4)]
    t0 = time.perf_counter()
    report, log = asyncio.run(run_graph(db, nodes))
    assert report.status == "completed"
    assert sorted(log["slow"]) == [0, 1, 2, 3]
    assert time.perf_counter() - t0 < 1.0  # 4 x 0.3s run concurrently


def test_tool_error_is_retried_for_reads(db):
    report, _ = asyncio.run(run_graph(db, [Node("x", "flights.flaky")]))
    assert report.nodes["x"].status == st.FAILED
    assert report.nodes["x"].attempts == 2
    assert "tool error" in report.nodes["x"].error


def test_input_required_round_trip(db):
    servers = travel_servers()

    async def main():
        report, _ = await run_graph(db, ([Node("p", "flights.pay", {"amount": 20})], servers))
        assert report.status == "blocked"
        q = report.nodes["p"].question
        assert q["prompt"] == "Enter the code we texted you"
        assert q["schema"]["properties"]["code"] == {"type": "string"}

        async def answer(sched):
            sched.answer("p", {"code": "314159"})

        return await run_graph(db, ([], servers), steps=answer)

    report, _ = asyncio.run(main())
    assert report.status == "completed"
    assert report.result("p") == {"paid": 20, "code": "314159"}


def test_unknown_server_is_rejected_without_side_effects(db):
    report, _ = asyncio.run(run_graph(db, [Node("x", "nowhere.tool")]))
    assert report.nodes["x"].status == st.FAILED
    assert "rejected" in report.nodes["x"].error


def test_crash_after_send_does_not_resend_unkeyed_write(db):
    servers = travel_servers()

    def hook(point):
        if point == "received":
            raise SimulatedCrash(point)

    async def main():
        crashed, _ = await run_graph(db, ([Node("e", "mail.send", {"to": "a", "body": "hi"})], servers), hook=hook)
        assert crashed is None
        return await run_graph(db, ([], servers))

    report, log = asyncio.run(main())
    assert report.nodes["e"].status == st.NEEDS_VERIFICATION
    assert log["sent"] == [("a", "hi")]


def test_crash_after_send_resends_keyed_write_with_same_key(db):
    servers = travel_servers()
    fired = []

    def hook(point):
        if point == "received" and not fired:
            fired.append(point)
            raise SimulatedCrash(point)

    async def main():
        crashed, _ = await run_graph(db, ([Node("b", "flights.book", {"flight": "TG1"})], servers), hook=hook)
        assert crashed is None
        return await run_graph(db, ([], servers))

    report, log = asyncio.run(main())
    assert report.nodes["b"].status == st.SUCCEEDED
    assert report.nodes["b"].attempts == 2
    assert list(log["booked"]) == [report.nodes["b"].idem_key]


# -- adapter details ---------------------------------------------------------------


def test_error_mapping():
    assert isinstance(_wrap(MCPError(CONNECTION_CLOSED, "closed")), Uncertain)
    assert isinstance(_wrap(MCPError(REQUEST_TIMEOUT, "slow")), Uncertain)
    assert isinstance(_wrap(MCPError(METHOD_NOT_FOUND, "no")), Rejected)
    assert isinstance(_wrap(MCPError(INVALID_PARAMS, "bad")), Rejected)
    assert isinstance(_wrap(OSError("reset")), Uncertain)


def test_result_value():
    text = lambda s: CallToolResult(content=[TextContent(type="text", text=s)])  # noqa: E731
    assert result_value(text('{"a": 1}')) == {"a": 1}
    assert result_value(text("plain")) == "plain"
    assert result_value(CallToolResult(content=[], structured_content={"result": 3})) == 3
    assert result_value(CallToolResult(content=[], structured_content={"x": 1, "y": 2})) == {"x": 1, "y": 2}


class FakeSession:
    def __init__(self, replies):
        self.replies = replies
        self.sent = []

    async def send_request(self, request, result_type):
        data = request.model_dump(by_alias=True, mode="json", exclude_none=True)
        self.sent.append(data)
        reply = self.replies[data["method"]]
        if isinstance(reply, Exception):
            raise reply
        return result_type.validate_python(reply) if hasattr(result_type, "validate_python") else result_type.model_validate(reply)


class FakeClient:
    def __init__(self, replies):
        self.session = FakeSession(replies)


TASK = {"taskId": "t-1", "status": "working", "createdAt": "2026-01-01T00:00:00Z",
        "lastUpdatedAt": "2026-01-01T00:00:00Z", "ttl": None, "pollInterval": 1500}


def test_task_requests_follow_sdk_types():
    fake = FakeClient({
        "tools/call": {"task": TASK},
        "tasks/get": {**TASK, "status": "input_required", "inputRequests": {
            "q": {"method": "elicitation/create", "params": {"mode": "form", "message": "Seat?", "requestedSchema": {"type": "object", "properties": {}}}}}},
        "tasks/result": {"content": [{"type": "text", "text": '{"ok": true}'}]},
        "tasks/update": {},
        "tasks/cancel": {**TASK, "status": "cancelled"},
    })
    backend = MCPBackend({}, task_tools={"svc.render"}, route=lambda tool: (fake, tool.split(".")[1]))

    async def main():
        started = await backend.call("svc.render", {"x": 1}, idempotency_key="k1")
        assert started.task_id == "t-1" and started.poll_interval == 1.5
        status = await backend.get_task("svc.render", "t-1")
        assert status.status == "input_required" and status.question.prompt == "Seat?"
        await backend.answer_task("svc.render", "t-1", status.question, {"seat": "1A"})
        assert (await backend.task_result("svc.render", "t-1")).value == {"ok": True}
        await backend.cancel_task("svc.render", "t-1")

    asyncio.run(main())
    call, get, update = fake.session.sent[0], fake.session.sent[1], fake.session.sent[2]
    assert call["method"] == "tools/call" and call["params"]["task"] == {}
    assert call["params"]["_meta"] == {"io.toolgraph/idempotency-key": "k1"}
    assert get == {"method": "tasks/get", "params": {"taskId": "t-1"}}
    assert update["params"] == {"taskId": "t-1", "inputResponses": {"q": {"action": "accept", "content": {"seat": "1A"}}}}
    assert [s["method"] for s in fake.session.sent[3:]] == ["tasks/result", "tasks/cancel"]


def test_unknown_task_is_reported_lost():
    from toolgraph import TaskLost

    fake = FakeClient({"tasks/get": MCPError(INVALID_PARAMS, "no such task")})
    backend = MCPBackend({}, route=lambda tool: (fake, tool))
    with pytest.raises(TaskLost):
        asyncio.run(backend.get_task("x", "t-9"))


def test_answers_map_to_each_input_request():
    from toolgraph.mcp_backend import _responses

    q = Question("?", state={"input_requests": {"a": {}, "b": {}}})
    out = _responses(q, {"a": {"code": "1"}, "b": {"action": "decline"}})
    assert out["a"].action == "accept" and out["a"].content == {"code": "1"}
    assert out["b"].action == "decline"
    single = _responses(Question("?", state={"input_requests": {"only": {}}}), "yes")
    assert single["only"].content == {"value": "yes"}
