"""Run scenarios against the simulated world, crash them on purpose, and check invariants."""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .. import store as st
from ..graph import Node, ref
from ..policy import PolicyRegistry, SideEffect, ToolPolicy
from ..scheduler import Mutations, PlannerView, Scheduler
from ..store import Store, StoreDead
from .vloop import run_virtual
from .world import SimBackend, SimTool, World, result_value

RUN = "sim-run"
EFFECTS = ("read", "keyed_write", "idempotent_write", "unkeyed_write")
PROTOCOLS = ("plain", "tasks")
JOB_GROUP = "jobs"


class SimulatedCrash(BaseException):
    """The client process died here."""


def catalog() -> dict[str, SimTool]:
    return {
        f"{p}.{e}": SimTool(f"{p}.{e}", e, p)  # type: ignore[arg-type]
        for p in PROTOCOLS
        for e in EFFECTS
    }


def policies(group_limit: int) -> PolicyRegistry:
    reg = PolicyRegistry()
    for name, tool in catalog().items():
        is_read = tool.effect == "read"
        reg.register(
            name,
            ToolPolicy(
                side_effect=SideEffect(tool.effect),
                max_attempts=3,
                backoff_base=1.0,
                timeout=10.0,
                poll_interval=2.0,
                concurrency_group=JOB_GROUP if tool.protocol == "tasks" else None,
                reads=("res:{res}",) if is_read else (),
                writes=() if is_read else ("res:{res}",),
            ),
        )
    reg.limit(JOB_GROUP, group_limit)
    return reg


@dataclass
class NodeSpec:
    id: str
    tool: str
    deps: list[str] = field(default_factory=list)
    resource: str | None = None
    faults: list[str] = field(default_factory=list)

    def node(self) -> Node:
        args: dict[str, Any] = {"call": self.id}
        if self.resource is not None:
            args["res"] = self.resource
        for d in self.deps:
            args[f"in_{d}"] = ref(d, "v")
        return Node(self.id, self.tool, args)

    @property
    def effect(self) -> str:
        return self.tool.split(".", 1)[1]


@dataclass
class Scenario:
    nodes: list[NodeSpec]
    waves: list[list[NodeSpec]] = field(default_factory=list)
    """Further waves handed out by the planner, one each time the graph drains."""
    max_concurrency: int = 4
    group_limit: int = 2

    def all_nodes(self) -> list[NodeSpec]:
        return self.nodes + [n for w in self.waves for n in w]


@dataclass
class SimResult:
    scenario: Scenario
    world: World
    nodes: dict[str, st.NodeRow]
    events: list[dict[str, Any]]
    trace: list[str]
    crashes: int
    status: str


class CrashHook:
    """Counts crash points; kills the process at the chosen (point, occurrence)."""

    def __init__(self, crash: tuple[str, int] | None, trace: list[str] | None) -> None:
        self.crash = crash
        self.trace = trace
        self.counts: dict[str, int] = {}
        self.store: Store | None = None
        self.fired = False

    def __call__(self, point: str) -> None:
        n = self.counts.get(point, 0) + 1
        self.counts[point] = n
        if self.trace is not None:
            self.trace.append(point)
        if self.crash is not None and not self.fired and self.crash == (point, n):
            self.fired = True
            assert self.store is not None
            self.store.kill()
            raise SimulatedCrash(f"{point}#{n}")


async def _human(sched: Scheduler, world: World, store: Store) -> None:
    """Answers questions and verifies unknown outcomes by reading the world back."""
    tools = catalog()
    try:
        while True:
            await asyncio.sleep(5.0)
            world.sync()
            for row in store.nodes(sched.run_id):
                if row.status == st.WAITING_INPUT and not row.has_answer:
                    sched.answer(row.id, world.answers.get(row.id, "42"))
                elif row.status == st.NEEDS_VERIFICATION:
                    if not world.settled(row.id):
                        continue
                    applied = [a for a in world.ledger if a.call == row.id]
                    if applied:
                        tool = tools[row.tool]
                        sched.resolve_verification(
                            row.id, applied=True, result=result_value(tool.name, row.id, applied[-1].args)
                        )
                    else:
                        sched.resolve_verification(row.id, applied=False)
    except StoreDead:
        # The scheduler died first; nothing left to do.
        return


def run_scenario(
    sc: Scenario,
    *,
    crash: tuple[str, int] | None = None,
    mutations: Mutations = Mutations(),
    trace: list[str] | None = None,
    db_dir: str | None = None,
    deadline: float = 100_000.0,
) -> SimResult:
    world = World(catalog(), faults={n.id: list(n.faults) for n in sc.all_nodes()})
    reg = policies(sc.group_limit)
    tmp = tempfile.mkdtemp(prefix="toolgraph-sim-", dir=db_dir)
    path = os.path.join(tmp, "run.db")
    hook = CrashHook(crash, trace)
    crashes = 0
    status = "?"

    async def planner(view: PlannerView) -> list[Node] | None:
        i = view.wave - 1
        return [n.node() for n in sc.waves[i]] if i < len(sc.waves) else None

    async def session() -> str:
        loop = asyncio.get_running_loop()
        store = Store(path, clock=loop.time, durable=False, hook=hook)
        hook.store = store
        sched = Scheduler(
            store, SimBackend(world), reg, run_id=RUN, planner=planner,
            max_concurrency=sc.max_concurrency, mutations=mutations,
        )
        tasks: list[asyncio.Task[Any]] = []
        try:
            sched.start([n.node() for n in sc.nodes])
            # The scheduler and the person answering it share one process: if
            # either hits a crash point, both die.
            main = loop.create_task(sched.run(wait_for_humans=True))
            helper = loop.create_task(_human(sched, world, store))
            tasks = [main, helper]
            async with asyncio.timeout(max(0.0, deadline - loop.time())):
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if helper.done() and not main.done():
                    helper.result()  # re-raises the crash
                return (await main).status
        except (SimulatedCrash, StoreDead):
            return "crashed"
        except TimeoutError:
            return "stuck"
        finally:
            for t in tasks:
                t.cancel()
                try:
                    await t
                except BaseException:
                    pass
            world.now = max(world.now, loop.time())
            store.close()

    for _ in range(10):
        status = run_virtual(session, start=world.now)
        if status != "crashed":
            break
        crashes += 1
    world.finish()
    store = Store(path, durable=False)
    try:
        nodes = {r.id: r for r in store.nodes(RUN)}
        events = store.events(RUN)
    finally:
        store.close()
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    return SimResult(sc, world, nodes, events, trace or [], crashes, status)


# -- invariants -------------------------------------------------------------------


def required_edges(sc: Scenario) -> set[tuple[str, str]]:
    """Every edge the scheduler must have, recomputed independently from the scenario."""
    edges: set[tuple[str, str]] = set()
    order = sc.all_nodes()
    for j, b in enumerate(order):
        for d in b.deps:
            edges.add((d, b.id))
        for a in order[:j]:
            if a.resource is None or a.resource != b.resource:
                continue
            if a.effect != "read" or b.effect != "read":
                edges.add((a.id, b.id))
    return edges


def check(result: SimResult) -> list[str]:
    sc, world, nodes, events = result.scenario, result.world, result.nodes, result.events
    v: list[str] = []
    specs = {n.id: n for n in sc.all_nodes()}

    if result.status in ("crashed", "stuck"):
        v.append(f"run never finished ({result.status})")

    # 1. Side effects: a non-idempotent write is applied at most once, exactly once if it succeeded.
    for nid, spec in specs.items():
        row = nodes.get(nid)
        n = world.applications(nid)
        if spec.effect in ("keyed_write", "unkeyed_write"):
            if n > 1:
                v.append(f"{nid}: {spec.effect} applied {n} times")
            if row is not None and row.status == st.SUCCEEDED and n != 1:
                v.append(f"{nid}: succeeded but applied {n} times")
        if spec.effect == "idempotent_write" and row is not None and row.status == st.SUCCEEDED and n < 1:
            v.append(f"{nid}: succeeded but never applied")

    # 2. Every required data and hazard edge exists.
    required = required_edges(sc)
    for a, b in sorted(required):
        if b in nodes and a not in nodes[b].deps:
            v.append(f"missing edge {a} -> {b}")

    # 3. Nothing starts before everything it depends on has finished.
    done_at: dict[str, int] = {}
    for e in events:
        node = e["node"]
        if node is None:
            continue
        if e["kind"] in ("succeeded", "verified_applied") and node not in done_at:
            done_at[node] = e["seq"]
        if e["kind"] == "sending":
            for a, b in required:
                if b == node and (a not in done_at or done_at[a] > e["seq"]):
                    v.append(f"{node} started before {a} finished")

    # 4. Concurrency limits hold.
    reg = policies(sc.group_limit)
    state: dict[str, str] = {}
    has_task: dict[str, bool] = {}
    for e in events:
        node = e["node"]
        if node is None or not e["data"] or "status" not in e["data"]:
            continue
        state[node] = e["data"]["status"]
        if e["kind"] == "task_started":
            has_task[node] = True
        elif state[node] not in (st.RUNNING, st.WAITING_INPUT):
            has_task[node] = False
        busy = [
            n for n, s in state.items()
            if s in (st.SENDING, st.RUNNING) or (s == st.WAITING_INPUT and has_task.get(n))
        ]
        if len(busy) > sc.max_concurrency:
            v.append(f"{len(busy)} calls in flight at event {e['seq']} (limit {sc.max_concurrency})")
        in_group = [n for n in busy if reg.get(nodes[n].tool).concurrency_group == JOB_GROUP]
        if len(in_group) > sc.group_limit:
            v.append(f"{len(in_group)} jobs in flight at event {e['seq']} (limit {sc.group_limit})")

    # 5. Final statuses are consistent.
    for nid, row in nodes.items():
        if row.status not in st.TERMINAL:
            v.append(f"{nid}: left in {row.status}")
        if row.status == st.SUCCEEDED and any(nodes[d].status != st.SUCCEEDED for d in row.deps):
            v.append(f"{nid}: succeeded although a dependency did not")
        if row.status == st.SKIPPED and not any(
            nodes[d].status in (st.FAILED, st.SKIPPED, st.CANCELLED) for d in row.deps
        ):
            v.append(f"{nid}: skipped without a failed dependency")
        if row.status == st.SUCCEEDED and specs[nid].deps:
            seen = [r for r in world.received if r[2] == nid]
            last = seen[-1][3] if seen else {}
            for d in specs[nid].deps:
                if last.get(f"in_{d}") != nodes[d].result["v"]:
                    v.append(f"{nid}: received wrong value for {d}")
    missing = set(specs) - set(nodes)
    if missing and result.status == "completed":
        v.append(f"nodes never added: {sorted(missing)}")
    return v


def crash_points(sc: Scenario) -> list[tuple[str, int]]:
    """Every (point, occurrence) a clean run of ``sc`` passes through."""
    trace: list[str] = []
    run_scenario(sc, trace=trace)
    counts: dict[str, int] = {}
    out = []
    for p in trace:
        counts[p] = counts.get(p, 0) + 1
        out.append((p, counts[p]))
    return out


Checker = Callable[[SimResult], list[str]]
