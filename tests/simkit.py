"""Small helpers for driving a scheduler against the simulated world in tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from toolgraph import PolicyRegistry, Scheduler, Store
from toolgraph.scheduler import RunReport
from toolgraph.sim import SimBackend, SimTool, World, run_virtual


@dataclass
class Outcome:
    report: RunReport
    world: World
    events: list[dict[str, Any]]
    elapsed: float

    def kinds(self, node: str) -> list[str]:
        return [e["kind"] for e in self.events if e["node"] == node]

    def sends(self, node: str) -> list[dict[str, Any]]:
        return [e["data"] for e in self.events if e["node"] == node and e["kind"] == "sending"]

    def status(self, node: str) -> str:
        return self.report.nodes[node].status


def world_of(*tools: SimTool, faults=None) -> World:
    return World({t.name: t for t in tools}, faults=faults or {})


def run(
    path: str,
    world: World,
    policies: PolicyRegistry,
    nodes=None,
    *,
    steps=None,
    run_id: str = "r",
    **kwargs: Any,
) -> Outcome:
    """Start (if ``nodes``) and run once. ``steps(sched)`` may be a coroutine run alongside."""

    async def main() -> Outcome:
        loop = asyncio.get_running_loop()
        store = Store(path, clock=loop.time, durable=False)
        wait = kwargs.pop("wait_for_humans", False)
        sched = Scheduler(store, SimBackend(world), policies, run_id=run_id, **kwargs)
        if nodes is not None:
            sched.start(nodes)
        t0 = loop.time()
        helper = None
        if steps is not None:
            if wait:
                helper = loop.create_task(steps(sched))
            else:
                await steps(sched)
        try:
            report = await sched.run(wait_for_humans=wait)
        finally:
            if helper:
                helper.cancel()
            world.now = max(world.now, loop.time())
        events = store.events(run_id)
        store.close()
        return Outcome(report, world, events, loop.time() - t0)

    return run_virtual(main, start=world.now)
