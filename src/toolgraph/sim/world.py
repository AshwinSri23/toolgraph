"""A simulated world of MCP and MCP Tasks servers.

Servers live outside the client "process": when the client crashes the world
keeps its state, its half-finished jobs and its ledger of every write that was
really applied. Everything runs on the virtual clock, and the world only
advances its own state when asked (lazily, at the current virtual time), so it
does not depend on any task that dies with the client.

Faults are scripted per call and per attempt, so every run is reproducible.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from ..backend import (
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

Effect = Literal["read", "keyed_write", "idempotent_write", "unkeyed_write"]
Protocol = Literal["plain", "tasks"]

# Faults for one attempt (one tools/call):
#   ok            normal success
#   reject        refused before anything happened (e.g. 429); safe to retry
#   lost          applied, but the response never arrives (connection reset)
#   hang          applied, but the response never arrives and the client times out
#   error_applied tool error after the write went through (partial failure)
#   error_clean   tool error and nothing was applied
#   ask           the tool needs an answer before it can go on
#   task_fail     (tasks) the job reports failure after applying
#   task_lost     (tasks) the server forgets the job before it finishes
FAULTS = ("ok", "reject", "lost", "hang", "error_applied", "error_clean", "ask", "task_fail", "task_lost")


@dataclass(frozen=True)
class SimTool:
    name: str
    effect: Effect
    protocol: Protocol = "plain"
    latency: float = 0.5
    duration: float = 30.0
    """How long a task job takes."""


@dataclass
class Applied:
    tool: str
    call: str
    at: float
    key: str | None
    args: dict[str, Any]


@dataclass
class Job:
    id: str
    tool: SimTool
    call: str
    key: str | None
    args: dict[str, Any]
    created: float
    remaining: float
    fault: str
    resumed_at: float
    status: str = "working"
    question: Question | None = None
    asked: bool = False
    applied: bool = False
    expires: float | None = None


def result_value(tool: str, call: str, args: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256(json.dumps([tool, call, args], sort_keys=True, default=str).encode()).hexdigest()
    return {"tool": tool, "call": call, "v": digest[:12]}


@dataclass
class World:
    tools: dict[str, SimTool]
    faults: dict[str, list[str]] = field(default_factory=dict)
    """call id -> fault per attempt (missing attempts are "ok")."""

    answers: dict[str, Any] = field(default_factory=dict)
    """call id -> the answer the tool expects when it asks."""

    input_ttl: float = 600.0
    """Seconds a job waits for an answer before the server discards it."""

    now: float = 0.0
    ledger: list[Applied] = field(default_factory=list)
    received: list[tuple[float, str, str, dict[str, Any]]] = field(default_factory=list)
    """(time, tool, call, args) for every request that reached a server."""

    _attempts: dict[str, int] = field(default_factory=dict)
    _pending: list[tuple[float, int, Applied]] = field(default_factory=list)
    _keyed: dict[tuple[str, str], tuple[str, Any]] = field(default_factory=dict)
    """(tool, key) -> ("result", value) | ("task", task_id)"""
    _jobs: dict[str, Job] = field(default_factory=dict)
    _seq: int = 0

    # -- time ------------------------------------------------------------------

    def sync(self) -> float:
        """Bring the world up to the current virtual time."""
        try:
            t = asyncio.get_running_loop().time()
        except RuntimeError:
            t = self.now
        self.now = max(self.now, t)
        self._apply_due(self.now)
        for job in self._jobs.values():
            self._advance(job, self.now)
        return self.now

    def finish(self) -> None:
        """Let the servers finish everything they can (after the client is gone)."""
        self._apply_due(float("inf"))
        for job in self._jobs.values():
            if job.status == "working":
                self._advance(job, float("inf"))

    def _apply_due(self, t: float) -> None:
        due = [p for p in self._pending if p[0] <= t]
        self._pending = [p for p in self._pending if p[0] > t]
        for _at, _seq, applied in sorted(due, key=lambda p: (p[0], p[1])):
            self.ledger.append(applied)

    def _schedule_apply(self, at: float, tool: SimTool, call: str, key: str | None, args: dict[str, Any]) -> None:
        if tool.effect == "read":
            return
        self._seq += 1
        self._pending.append((at, self._seq, Applied(tool.name, call, at, key, args)))

    def _advance(self, job: Job, t: float) -> None:
        if job.status == "input_required" and t >= job.resumed_at + self.input_ttl:
            # Nobody answered: the server drops the job without applying it.
            job.status = "gone"
        if job.status != "working":
            return
        done_at = job.resumed_at + job.remaining
        if job.fault == "ask" and not job.asked:
            ask_at = job.resumed_at + job.remaining / 2
            if t >= ask_at:
                job.asked = True
                job.remaining -= ask_at - job.resumed_at
                job.resumed_at = ask_at
                job.status = "input_required"
                job.question = Question(prompt=f"answer for {job.call}", state={"key": job.call})
                return
        if job.fault == "task_lost" and t >= job.resumed_at + job.remaining / 2:
            job.status = "gone"
            return
        if t >= done_at:
            if job.tool.effect != "read" and not job.applied:
                job.applied = True
                self.ledger.append(Applied(job.tool.name, job.call, done_at, job.key, job.args))
            job.status = "failed" if job.fault == "task_fail" else "completed"

    # -- queries for checkers and verifiers --------------------------------------

    def applications(self, call: str) -> int:
        return sum(1 for a in self.ledger if a.call == call)

    def settled(self, call: str) -> bool:
        """True when no server has anything in progress that could still apply ``call``."""
        if any(p[2].call == call for p in self._pending):
            return False
        return not any(j.call == call and j.status in ("working", "input_required") for j in self._jobs.values())

    # -- server behaviour ----------------------------------------------------------

    def _fault(self, call: str) -> str:
        n = self._attempts.get(call, 0)
        self._attempts[call] = n + 1
        script = self.faults.get(call, [])
        return script[n] if n < len(script) else "ok"

    @staticmethod
    def call_id(tool: str, args: dict[str, Any]) -> str:
        if "call" in args:
            return str(args["call"])
        return tool + ":" + json.dumps(args, sort_keys=True, default=str)

    async def call(
        self, tool_name: str, args: dict[str, Any], key: str | None, answer: Any, question: Question | None
    ) -> CallOutcome:
        tool = self.tools[tool_name]
        clean = {k: v for k, v in args.items() if k != "idempotency_key"}
        call = self.call_id(tool_name, clean)
        self.sync()
        fault = self._fault(call)
        self.received.append((self.now, tool_name, call, clean))
        if fault == "reject":
            await asyncio.sleep(tool.latency)
            raise Rejected("server busy")
        if tool.protocol == "tasks":
            return await self._create_task(tool, call, key, clean, fault)
        return await self._plain(tool, call, key, clean, fault, answer, question)

    async def _plain(
        self, tool: SimTool, call: str, key: str | None, args: dict[str, Any], fault: str, answer: Any,
        question: Question | None,
    ) -> CallOutcome:
        value = result_value(tool.name, call, args)
        if fault == "ask" and question is None:
            await asyncio.sleep(tool.latency)
            return NeedsInput(Question(prompt=f"answer for {call}", state={"key": call}))
        if key is not None and tool.effect == "keyed_write" and (tool.name, key) in self._keyed:
            # Seen this key before: replay the stored response, apply nothing.
            await asyncio.sleep(tool.latency)
            self.sync()
            kind, stored = self._keyed[(tool.name, key)]
            return Done(stored)
        applies = fault != "error_clean"
        if applies:
            self._schedule_apply(self.now + tool.latency / 2, tool, call, key, args)
            if key is not None and tool.effect == "keyed_write":
                self._keyed[(tool.name, key)] = ("result", value)
        await asyncio.sleep(tool.latency)
        self.sync()
        if fault == "lost":
            raise Uncertain("connection reset")
        if fault == "hang":
            await asyncio.Event().wait()
        if fault in ("error_applied", "error_clean"):
            return ToolFailed("tool failed")
        return Done(value)

    async def _create_task(self, tool: SimTool, call: str, key: str | None, args: dict[str, Any], fault: str) -> CallOutcome:
        known = self._keyed.get((tool.name, key)) if key is not None and tool.effect == "keyed_write" else None
        if known is not None and self._jobs[known[1]].status != "gone":
            # Same key: rejoin the original job instead of starting another.
            await asyncio.sleep(tool.latency)
            return Started(known[1], poll_interval=2.0)
        self._seq += 1
        task_id = f"task-{self._seq}"
        job = Job(task_id, tool, call, key, args, self.now, tool.duration, fault, resumed_at=self.now)
        self._jobs[task_id] = job
        if key is not None and tool.effect == "keyed_write":
            self._keyed[(tool.name, key)] = ("task", task_id)
        await asyncio.sleep(tool.latency)
        self.sync()
        if fault == "lost":
            raise Uncertain("connection reset")
        if fault == "hang":
            await asyncio.Event().wait()
        return Started(task_id, poll_interval=2.0)

    def _job(self, task_id: str) -> Job:
        job = self._jobs.get(task_id)
        if job is None or job.status == "gone":
            raise TaskLost(task_id)
        return job

    async def get_task(self, task_id: str) -> TaskStatus:
        self.sync()
        job = self._job(task_id)
        await asyncio.sleep(job.tool.latency / 5)
        self.sync()
        job = self._job(task_id)
        return TaskStatus(job.status, question=job.question, poll_interval=2.0)  # type: ignore[arg-type]

    async def task_result(self, task_id: str) -> Done | ToolFailed:
        self.sync()
        job = self._job(task_id)
        await asyncio.sleep(job.tool.latency / 5)
        if job.status == "failed":
            return ToolFailed("job failed")
        if job.status != "completed":
            raise Rejected("not finished")
        return Done(result_value(job.tool.name, job.call, job.args))

    async def answer_task(self, task_id: str, question: Question, answer: Any) -> None:
        self.sync()
        job = self._job(task_id)
        await asyncio.sleep(job.tool.latency / 5)
        self.sync()
        if job.status == "input_required":
            job.status = "working"
            job.question = None
            job.resumed_at = self.now

    async def cancel_task(self, task_id: str) -> None:
        self.sync()
        job = self._job(task_id)
        if job.status in ("working", "input_required"):
            job.status = "cancelled"


class SimBackend:
    """The client side of the simulated servers: implements :class:`toolgraph.backend.Backend`."""

    def __init__(self, world: World) -> None:
        self.world = world

    async def call(
        self, tool: str, args: dict[str, Any], *, idempotency_key: str | None, answer: Any = None,
        question: Question | None = None,
    ) -> CallOutcome:
        return await self.world.call(tool, args, idempotency_key, answer, question)

    async def get_task(self, tool: str, task_id: str) -> TaskStatus:
        return await self.world.get_task(task_id)

    async def task_result(self, tool: str, task_id: str) -> Done | ToolFailed:
        return await self.world.task_result(task_id)

    async def answer_task(self, tool: str, task_id: str, question: Question, answer: Any) -> None:
        await self.world.answer_task(task_id, question, answer)

    async def cancel_task(self, tool: str, task_id: str) -> None:
        await self.world.cancel_task(task_id)
