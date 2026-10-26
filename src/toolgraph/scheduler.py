"""The scheduler: run a call graph in parallel, durably, and never repeat a side effect blindly.

Every state change is committed to the store before the scheduler acts on it:
"about to send" is written before the request goes out, and "server returned
job #5" is written before the job is polled. After a crash, :meth:`Scheduler.run`
reads the store back and decides what to do with each call whose outcome is
unknown, according to its tool's policy.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from . import store as st
from .backend import (
    Backend,
    Done,
    NeedsInput,
    Question,
    Rejected,
    Started,
    TaskLost,
    ToolFailed,
    Uncertain,
)
from .graph import GraphError, Node, PlannedNode, plan, resolve_args
from .policy import PolicyRegistry, SideEffect, ToolPolicy
from .store import NodeRow, Store

KEY_META = "io.toolgraph/idempotency-key"

Planner = Callable[["PlannerView"], Awaitable[list[Node] | None]]
"""Called when the graph drains. Returns the next wave of calls, or nothing to finish."""


@dataclass
class PlannerView:
    run_id: str
    wave: int
    nodes: dict[str, NodeRow]

    def result(self, node_id: str) -> Any:
        return self.nodes[node_id].result


@dataclass
class RunReport:
    run_id: str
    status: Literal["completed", "failed", "blocked"]
    nodes: dict[str, NodeRow]

    def result(self, node_id: str) -> Any:
        return self.nodes[node_id].result

    def by_status(self, status: str) -> list[str]:
        return [n.id for n in self.nodes.values() if n.status == status]


@dataclass(frozen=True)
class Mutations:
    """Deliberately broken behaviour, used only to prove the test suite catches bugs."""

    unsafe_replay: bool = False
    fresh_keys: bool = False
    no_hazards: bool = False


@dataclass
class _Outcome:
    """What the attempt told us, before the policy turns it into a status."""

    kind: Literal["definite", "uncertain"]
    message: str


class Scheduler:
    def __init__(
        self,
        store: Store,
        backend: Backend,
        policies: PolicyRegistry | None = None,
        *,
        run_id: str,
        planner: Planner | None = None,
        max_concurrency: int = 16,
        max_waves: int = 100,
        clock: Callable[[], float] | None = None,
        mutations: Mutations = Mutations(),
    ) -> None:
        self.store = store
        self.backend = backend
        self.policies = policies or PolicyRegistry()
        self.run_id = run_id
        self.planner = planner
        self.max_concurrency = max_concurrency
        self.max_waves = max_waves
        self.clock = clock or store.clock
        self.mutations = mutations
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._wake: asyncio.Event | None = None
        self._fatal: BaseException | None = None

    # -- building the graph ----------------------------------------------------

    def start(self, nodes: Iterable[Node]) -> bool:
        """Create the run with its first wave, unless it already exists (a restart).

        Returns True if the run was created.
        """
        nodes = list(nodes)
        if self.store.run_status(self.run_id) is not None:
            return False
        planned = self._plan(nodes)
        return self.store.create_run(self.run_id, planned, self._keys(planned))

    def add(self, nodes: Iterable[Node]) -> int:
        """Append a wave of nodes to an existing run."""
        planned = self._plan(list(nodes))
        return self.store.add_nodes(self.run_id, planned, self._keys(planned))

    def _plan(self, nodes: list[Node]) -> list[PlannedNode]:
        existing = [
            PlannedNode(r.id, r.tool, r.args, r.seq, r.deps) for r in self.store.nodes(self.run_id)
        ]
        return plan(nodes, self.policies.get, existing, hazards=not self.mutations.no_hazards)

    def _keys(self, planned: list[PlannedNode]) -> dict[str, str | None]:
        keys: dict[str, str | None] = {}
        for p in planned:
            if self.policies.get(p.tool).side_effect is SideEffect.KEYED_WRITE:
                digest = hashlib.sha256(f"{self.run_id}\x00{p.id}".encode()).hexdigest()
                keys[p.id] = "tg-" + digest[:32]
            else:
                keys[p.id] = None
        return keys

    # -- human input -------------------------------------------------------------

    def answer(self, node_id: str, answer: Any) -> None:
        """Answer the question a paused node asked."""
        row = self.store.node(self.run_id, node_id)
        if row.status != st.WAITING_INPUT:
            raise ValueError(f"node {node_id!r} is {row.status}, not waiting for input")
        self.store.update(
            self.run_id, node_id, st.WAITING_INPUT, "answered", has_answer=True, answer=answer
        )
        self._poke()

    def resolve_verification(
        self, node_id: str, *, applied: bool, result: Any = None, retry: bool = True
    ) -> None:
        """Settle a call whose outcome was unknown.

        ``applied=True`` marks it succeeded with ``result``. ``applied=False``
        means it definitely did not happen: it is sent again if ``retry``,
        otherwise marked failed.
        """
        row = self.store.node(self.run_id, node_id)
        if row.status != st.NEEDS_VERIFICATION:
            raise ValueError(f"node {node_id!r} is {row.status}, not awaiting verification")
        if applied:
            self.store.update(self.run_id, node_id, st.SUCCEEDED, "verified_applied", result=result)
        elif retry:
            self.store.update(
                self.run_id, node_id, st.RETRY_WAIT, "verified_not_applied",
                attempts=0, maybe_applied=False, task_id=None, not_before=self.clock(),
            )
        else:
            self.store.update(
                self.run_id, node_id, st.FAILED, "verified_not_applied",
                error="verified not applied",
            )
        self._poke()

    def resolved_args(self, node_id: str) -> dict[str, Any]:
        """A node's arguments with refs replaced by results, e.g. to read back an unknown outcome."""
        return self._args(self.store.node(self.run_id, node_id))

    def _poke(self) -> None:
        if self._wake is not None:
            self._wake.set()

    # -- the loop ------------------------------------------------------------------

    async def run(self, *, wait_for_humans: bool = False) -> RunReport:
        """Run until every call is finished, or until only people can make progress.

        With ``wait_for_humans`` the scheduler keeps waiting for :meth:`answer`
        and :meth:`resolve_verification` calls (from other tasks on this loop).
        """
        if self.store.run_status(self.run_id) is None:
            raise GraphError(f"run {self.run_id!r} has not been started")
        self._wake = asyncio.Event()
        self._fatal = None
        self.store.log(self.run_id, "run_resumed")
        self._recover()
        try:
            return await self._loop(wait_for_humans)
        finally:
            await self._stop_tasks()
            self._wake = None

    def _recover(self) -> None:
        """Decide the fate of calls that were in flight when the process died."""
        for row in self.store.nodes(self.run_id):
            if row.status != st.SENDING:
                continue
            policy = self.policies.get(row.tool)
            if self._replayable(policy):
                self.store.update(
                    self.run_id, row.id, st.RETRY_WAIT, "recovered_resend",
                    not_before=self.clock(), maybe_applied=True,
                )
            else:
                self.store.update(
                    self.run_id, row.id, st.NEEDS_VERIFICATION, "recovered_unknown",
                    maybe_applied=True, error="process stopped with the call in flight",
                )

    def _replayable(self, policy: ToolPolicy) -> bool:
        return policy.side_effect.safe_to_replay or self.mutations.unsafe_replay

    async def _loop(self, wait_for_humans: bool) -> RunReport:
        assert self._wake is not None
        while True:
            self._wake.clear()
            if self._fatal is not None:
                raise self._fatal
            rows = {r.id: r for r in self.store.nodes(self.run_id)}
            if self._skip_dead_branches(rows):
                continue
            now = self.clock()
            next_timer = self._dispatch(rows, now)
            if self._tasks:
                await self._sleep(next_timer, now)
                continue
            if next_timer is not None:
                await self._sleep(next_timer, now)
                continue
            statuses = {r.status for r in rows.values()}
            if statuses & st.BLOCKED_ON_HUMAN:
                if wait_for_humans:
                    await self._wake.wait()
                    continue
                return self._report(rows, "blocked")
            # Everything is terminal: the graph has drained.
            if await self._replan(rows):
                continue
            status = "completed" if statuses <= {st.SUCCEEDED} else "failed"
            self.store.set_run_status(self.run_id, status)
            return self._report(rows, status)  # type: ignore[arg-type]

    async def _sleep(self, until: float | None, now: float) -> None:
        assert self._wake is not None
        timeout = None if until is None else max(0.0, until - now)
        try:
            async with asyncio.timeout(timeout):
                await self._wake.wait()
        except TimeoutError:
            pass

    def _report(self, rows: dict[str, NodeRow], status: str) -> RunReport:
        return RunReport(self.run_id, status, rows)  # type: ignore[arg-type]

    async def _replan(self, rows: dict[str, NodeRow]) -> bool:
        if self.planner is None:
            return False
        waves = 1 + max((r.wave for r in rows.values()), default=-1)
        if waves >= self.max_waves:
            return False
        new = await self.planner(PlannerView(self.run_id, waves, rows))
        if not new:
            return False
        self.add(new)
        return True

    def _skip_dead_branches(self, rows: dict[str, NodeRow]) -> bool:
        changed = False
        for row in rows.values():
            if row.status != st.PENDING:
                continue
            dead = [d for d in row.deps if rows[d].status in (st.FAILED, st.SKIPPED, st.CANCELLED)]
            if dead:
                self.store.update(
                    self.run_id, row.id, st.SKIPPED, "skipped",
                    error=f"dependency {dead[0]!r} did not succeed",
                )
                changed = True
        return changed

    def _occupancy(self, rows: dict[str, NodeRow]) -> tuple[int, dict[str, int]]:
        busy = set(self._tasks)
        busy |= {r.id for r in rows.values() if r.status == st.RUNNING}
        busy |= {r.id for r in rows.values() if r.status == st.WAITING_INPUT and r.task_id}
        groups: dict[str, int] = {}
        for nid in busy:
            g = self.policies.get(rows[nid].tool).concurrency_group
            if g is not None:
                groups[g] = groups.get(g, 0) + 1
        return len(busy), groups

    def _dispatch(self, rows: dict[str, NodeRow], now: float) -> float | None:
        """Start every call that may start. Returns the earliest future wake-up time."""
        total, groups = self._occupancy(rows)
        next_timer: float | None = None
        for row in rows.values():  # ordered by listing order
            if row.id in self._tasks:
                continue
            policy = self.policies.get(row.tool)
            if row.status == st.RUNNING:
                # A job we were polling before a restart: it already holds its slot.
                self._spawn(row.id)
                continue
            if row.status == st.WAITING_INPUT:
                if not row.has_answer:
                    continue
                if row.task_id is not None:
                    # The paused job already holds its slot.
                    self._spawn(row.id)
                    continue
            elif row.status == st.RETRY_WAIT:
                if row.not_before is not None and row.not_before > now:
                    next_timer = row.not_before if next_timer is None else min(next_timer, row.not_before)
                    continue
            elif row.status != st.PENDING or any(rows[d].status != st.SUCCEEDED for d in row.deps):
                continue
            g = policy.concurrency_group
            if total >= self.max_concurrency:
                continue
            if g is not None and groups.get(g, 0) >= self.policies.group_limits.get(g, 1 << 30):
                continue
            total += 1
            if g is not None:
                groups[g] = groups.get(g, 0) + 1
            self._spawn(row.id)
        return next_timer

    def _spawn(self, node_id: str) -> None:
        task = asyncio.get_running_loop().create_task(self._drive(node_id), name=f"toolgraph:{node_id}")
        self._tasks[node_id] = task
        task.add_done_callback(lambda t, nid=node_id: self._finished(nid, t))

    def _finished(self, node_id: str, task: asyncio.Task[None]) -> None:
        self._tasks.pop(node_id, None)
        if not task.cancelled() and task.exception() is not None and self._fatal is None:
            self._fatal = task.exception()
        self._poke()

    async def _stop_tasks(self) -> None:
        tasks = list(self._tasks.values())
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except BaseException:
                pass
        self._tasks.clear()

    # -- one node ------------------------------------------------------------------

    async def _drive(self, node_id: str) -> None:
        row = self.store.node(self.run_id, node_id)
        policy = self.policies.get(row.tool)
        if row.status == st.RUNNING:
            await self._poll(row, policy)
        elif row.status == st.WAITING_INPUT and row.task_id:
            await self._answer_task(row, policy)
        else:
            await self._send(row, policy)

    def _args(self, row: NodeRow) -> dict[str, Any]:
        def result_of(dep: str) -> Any:
            return self.store.node(self.run_id, dep).result

        return resolve_args(row.args, result_of)

    def _key(self, row: NodeRow, attempt: int) -> str | None:
        if row.idem_key is None:
            return None
        if self.mutations.fresh_keys:
            return f"{row.idem_key}-{attempt}"
        return row.idem_key

    async def _send(self, row: NodeRow, policy: ToolPolicy) -> None:
        try:
            args = self._args(row)
        except (KeyError, IndexError, ValueError, TypeError) as e:
            self.store.update(self.run_id, row.id, st.FAILED, "bad_ref", error=f"could not resolve args: {e!r}")
            return
        attempt = row.attempts + 1
        key = self._key(row, attempt)
        if key is not None and policy.key_arg:
            args = {**args, policy.key_arg: key}
        question = answer = None
        if row.status == st.WAITING_INPUT and row.has_answer and row.question is not None:
            question, answer = Question(**row.question), row.answer
        now = self.clock()
        # The point of no return: once this commit lands the request may be out.
        self.store.update(
            self.run_id, row.id, st.SENDING, "sending",
            data={"attempt": attempt, "key": key},
            attempts=attempt, started=row.started if row.started is not None else now,
        )
        try:
            async with asyncio.timeout(policy.timeout):
                outcome = await self.backend.call(
                    row.tool, args, idempotency_key=key, answer=answer, question=question
                )
        except Rejected as e:
            self.store.point("received")
            return self._settle(row.id, policy, attempt, _Outcome("definite", f"rejected: {e}"))
        except (Uncertain, TimeoutError, ConnectionError, OSError) as e:
            self.store.point("received")
            return self._settle(row.id, policy, attempt, _Outcome("uncertain", f"{type(e).__name__}: {e}"))
        self.store.point("received")
        if isinstance(outcome, Done):
            self.store.update(self.run_id, row.id, st.SUCCEEDED, "succeeded", result=outcome.value)
        elif isinstance(outcome, ToolFailed):
            self._tool_failed(row.id, policy, attempt, outcome)
        elif isinstance(outcome, NeedsInput):
            self.store.update(
                self.run_id, row.id, st.WAITING_INPUT, "asked",
                question=outcome.question.__dict__, has_answer=False,
            )
        elif isinstance(outcome, Started):
            self.store.update(
                self.run_id, row.id, st.RUNNING, "task_started",
                data={"task_id": outcome.task_id}, task_id=outcome.task_id,
                task_started=row.task_started if row.task_id == outcome.task_id else self.clock(),
                question=None, has_answer=False,
            )
            fresh = self.store.node(self.run_id, row.id)
            await self._poll(fresh, policy, interval=outcome.poll_interval)
        else:  # pragma: no cover
            raise TypeError(f"backend returned {outcome!r}")

    def _tool_failed(self, node_id: str, policy: ToolPolicy, attempt: int, failure: ToolFailed) -> None:
        kind = "uncertain" if policy.tool_error_uncertain else "definite"
        self._settle(node_id, policy, attempt, _Outcome(kind, f"tool error: {failure.message}"))

    def _settle(self, node_id: str, policy: ToolPolicy, attempt: int, outcome: _Outcome) -> None:
        """Turn a failed attempt into retry, verify or fail, according to the policy."""
        row = self.store.node(self.run_id, node_id)
        maybe_applied = row.maybe_applied or (outcome.kind == "uncertain" and policy.side_effect.is_write)
        retry_allowed = attempt < policy.max_attempts
        if outcome.kind == "uncertain" and not self._replayable(policy):
            self.store.update(
                self.run_id, node_id, st.NEEDS_VERIFICATION, "needs_verification",
                error=outcome.message, maybe_applied=True, task_id=None,
            )
        elif retry_allowed:
            self.store.update(
                self.run_id, node_id, st.RETRY_WAIT, "retry_wait",
                data={"reason": outcome.message},
                error=outcome.message, maybe_applied=maybe_applied, task_id=None,
                not_before=self.clock() + policy.backoff(attempt),
            )
        elif maybe_applied and policy.side_effect is SideEffect.KEYED_WRITE:
            # Out of attempts, and an earlier one may have gone through.
            self.store.update(
                self.run_id, node_id, st.NEEDS_VERIFICATION, "needs_verification",
                error=outcome.message, maybe_applied=True, task_id=None,
            )
        else:
            self.store.update(
                self.run_id, node_id, st.FAILED, "failed",
                error=outcome.message, maybe_applied=maybe_applied, task_id=None,
            )

    def _job_unknown(self, row: NodeRow, policy: ToolPolicy, message: str) -> None:
        """A job ended without a usable result (lost, cancelled, timed out)."""
        if policy.side_effect in (SideEffect.READ, SideEffect.IDEMPOTENT_WRITE):
            self._settle(row.id, policy, row.attempts, _Outcome("uncertain", message))
        else:
            # Re-sending a keyed write would only rejoin the same dead job.
            self.store.update(
                self.run_id, row.id, st.NEEDS_VERIFICATION, "needs_verification",
                error=message, maybe_applied=True, task_id=None,
            )

    async def _poll(self, row: NodeRow, policy: ToolPolicy, interval: float | None = None) -> None:
        assert row.task_id is not None
        task_id = row.task_id
        failures = 0
        while True:
            await asyncio.sleep(interval if interval is not None else policy.poll_interval)
            if policy.job_timeout is not None and row.task_started is not None:
                if self.clock() - row.task_started > policy.job_timeout:
                    try:
                        await self.backend.cancel_task(row.tool, task_id)
                    except (Rejected, Uncertain, TaskLost, TimeoutError, OSError):
                        pass
                    self.store.point("received")
                    return self._job_unknown(row, policy, f"job exceeded {policy.job_timeout}s")
            try:
                async with asyncio.timeout(policy.timeout):
                    status = await self.backend.get_task(row.tool, task_id)
            except TaskLost as e:
                self.store.point("received")
                return self._job_unknown(row, policy, f"task lost: {e}")
            except (Rejected, Uncertain, TimeoutError, OSError):
                # Polling is a read: just try again, backing off.
                failures += 1
                interval = policy.backoff(failures)
                continue
            failures = 0
            self.store.point("received")
            interval = status.poll_interval
            if status.status == "working":
                continue
            if status.status == "input_required":
                if status.question is None:
                    continue
                self.store.update(
                    self.run_id, row.id, st.WAITING_INPUT, "asked",
                    question=status.question.__dict__, has_answer=False,
                )
                return
            if status.status == "cancelled":
                return self._job_unknown(row, policy, "task was cancelled")
            if status.status == "failed":
                return self._tool_failed(row.id, policy, row.attempts, ToolFailed(status.message or "task failed"))
            # completed
            try:
                async with asyncio.timeout(policy.timeout):
                    result = await self.backend.task_result(row.tool, task_id)
            except TaskLost as e:
                self.store.point("received")
                return self._job_unknown(row, policy, f"task lost: {e}")
            except (Rejected, Uncertain, TimeoutError, OSError):
                failures += 1
                interval = policy.backoff(failures)
                continue
            self.store.point("received")
            if isinstance(result, Done):
                self.store.update(self.run_id, row.id, st.SUCCEEDED, "succeeded", result=result.value)
            else:
                self._tool_failed(row.id, policy, row.attempts, result)
            return

    async def _answer_task(self, row: NodeRow, policy: ToolPolicy) -> None:
        """Send a person's answer to a paused job (tasks/update), then keep polling."""
        assert row.task_id is not None and row.question is not None
        question = Question(**row.question)
        failures = 0
        while True:
            try:
                async with asyncio.timeout(policy.timeout):
                    await self.backend.answer_task(row.tool, row.task_id, question, row.answer)
                break
            except TaskLost as e:
                self.store.point("received")
                return self._job_unknown(row, policy, f"task lost: {e}")
            except (Rejected, Uncertain, TimeoutError, OSError):
                # Answers are keyed by the question, so re-sending one is harmless.
                failures += 1
                await asyncio.sleep(policy.backoff(failures))
        self.store.point("received")
        self.store.update(
            self.run_id, row.id, st.RUNNING, "answer_sent", question=None, has_answer=False
        )
        await self._poll(self.store.node(self.run_id, row.id), policy)


__all__ = [
    "KEY_META",
    "Mutations",
    "Planner",
    "PlannerView",
    "RunReport",
    "Scheduler",
]
