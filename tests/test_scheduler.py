import pytest
from simkit import run, world_of

from toolgraph import GraphError, Node, PolicyRegistry, Scheduler, SideEffect, Store, ToolPolicy, ref
from toolgraph import store as st
from toolgraph.sim import SimBackend, SimTool

READ = SimTool("read", "read")
JOB = SimTool("job", "read", "tasks", duration=30)
KEYED = SimTool("pay", "keyed_write")
KEYED_JOB = SimTool("book", "keyed_write", "tasks", duration=30)
UNKEYED = SimTool("email", "unkeyed_write")
UNKEYED_JOB = SimTool("deploy", "unkeyed_write", "tasks", duration=30)
IDEM = SimTool("set_status", "idempotent_write")
ALL = (READ, JOB, KEYED, KEYED_JOB, UNKEYED, UNKEYED_JOB, IDEM)


def registry(**overrides) -> PolicyRegistry:
    reg = PolicyRegistry()
    for t in ALL:
        kw = dict(side_effect=SideEffect(t.effect), max_attempts=3, timeout=10, poll_interval=2)
        kw.update(overrides.get(t.name, {}))
        reg.register(t.name, ToolPolicy(**kw))
    return reg


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "run.db")


def go(db, nodes, faults=None, reg=None, **kw):
    return run(db, world_of(*ALL, faults=faults), reg or registry(), nodes, **kw)


# -- parallelism and ordering -------------------------------------------------------


def test_independent_jobs_run_in_parallel(db):
    out = go(db, [Node("a", "job", {"call": "a"}), Node("b", "job", {"call": "b"})])
    assert out.report.status == "completed"
    assert out.elapsed < 40, out.elapsed  # one 30s job duration, not two


def test_dependent_calls_run_in_order_and_pass_values(db):
    out = go(db, [Node("a", "read", {"call": "a"}), Node("b", "read", {"call": "b", "x": ref("a", "v")})])
    assert out.report.status == "completed"
    received = {c: args for _t, _tool, c, args in out.world.received}
    assert received["b"]["x"] == out.report.result("a")["v"]
    a_done = next(e["seq"] for e in out.events if e["node"] == "a" and e["kind"] == "succeeded")
    b_start = next(e["seq"] for e in out.events if e["node"] == "b" and e["kind"] == "sending")
    assert a_done < b_start


def test_hazard_orders_write_before_read(db):
    reg = registry(email={"writes": ("inbox:{to}",)}, read={"reads": ("inbox:{to}",)})
    out = go(db, [Node("w", "email", {"call": "w", "to": "x"}), Node("r", "read", {"call": "r", "to": "x"})], reg=reg)
    assert out.report.nodes["r"].deps == {"w": "raw"}
    order = [c for _t, _tool, c, _a in out.world.received]
    assert order == ["w", "r"]


def test_concurrency_group_serializes_jobs(db):
    reg = registry(job={"concurrency_group": "gpu"})
    reg.limit("gpu", 1)
    out = go(db, [Node("a", "job", {"call": "a"}), Node("b", "job", {"call": "b"})], reg=reg)
    assert out.report.status == "completed"
    assert out.elapsed >= 60


def test_max_concurrency(db):
    out = go(db, [Node(n, "job", {"call": n}) for n in "abc"], max_concurrency=1)
    assert out.report.status == "completed"
    assert out.elapsed >= 90


# -- retries ------------------------------------------------------------------------


def test_read_retries_after_rejection(db):
    out = go(db, [Node("a", "read", {"call": "a"})], faults={"a": ["reject"]})
    assert out.status("a") == st.SUCCEEDED
    assert out.report.nodes["a"].attempts == 2


def test_read_retries_after_lost_response(db):
    out = go(db, [Node("a", "read", {"call": "a"})], faults={"a": ["lost", "hang"]})
    assert out.status("a") == st.SUCCEEDED
    assert out.report.nodes["a"].attempts == 3


def test_retries_back_off(db):
    out = go(db, [Node("a", "read", {"call": "a"})], faults={"a": ["reject", "reject"]})
    times = [e["at"] for e in out.events if e["node"] == "a" and e["kind"] == "sending"]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert gaps[1] > gaps[0]


def test_exhausted_read_fails_and_skips_dependents(db):
    out = go(
        db,
        [Node("a", "read", {"call": "a"}), Node("b", "read", {"call": "b", "x": ref("a")}), Node("c", "read", {"call": "c"})],
        faults={"a": ["reject"] * 3},
    )
    assert out.report.status == "failed"
    assert (out.status("a"), out.status("b"), out.status("c")) == (st.FAILED, st.SKIPPED, st.SUCCEEDED)


def test_unkeyed_write_is_never_repeated_on_unknown_outcome(db):
    out = go(db, [Node("e", "email", {"call": "e"}), Node("f", "read", {"call": "f", "x": ref("e")})], faults={"e": ["lost"]})
    assert out.report.status == "blocked"
    assert out.status("e") == st.NEEDS_VERIFICATION
    assert out.status("f") == st.PENDING
    assert len(out.sends("e")) == 1
    assert out.world.applications("e") == 1


def test_unkeyed_write_retried_after_definite_rejection(db):
    out = go(db, [Node("e", "email", {"call": "e"})], faults={"e": ["reject"]})
    assert out.status("e") == st.SUCCEEDED
    assert out.world.applications("e") == 1


def test_unkeyed_tool_error_might_have_applied(db):
    out = go(db, [Node("e", "email", {"call": "e"})], faults={"e": ["error_clean"]})
    assert out.status("e") == st.NEEDS_VERIFICATION


def test_policy_can_declare_tool_errors_clean(db):
    reg = registry(email={"error_may_have_applied": False})
    out = go(db, [Node("e", "email", {"call": "e"})], faults={"e": ["error_clean"]}, reg=reg)
    assert out.status("e") == st.SUCCEEDED
    assert out.world.applications("e") == 1


def test_keyed_write_resends_with_same_key(db):
    out = go(db, [Node("p", "pay", {"call": "p"})], faults={"p": ["lost"]})
    assert out.status("p") == st.SUCCEEDED
    keys = {s["key"] for s in out.sends("p")}
    # The server recognises the repeated key and replays its stored response.
    assert len(out.sends("p")) == 2 and len(keys) == 1
    assert out.world.applications("p") == 1


def test_keys_are_sent_as_an_argument_and_distinct_per_node(db):
    out = go(db, [Node("p", "pay", {"call": "p"}), Node("q", "pay", {"call": "q"})])
    k = {n: out.sends(n)[0]["key"] for n in "pq"}
    assert k["p"] != k["q"]
    assert [a.key for a in sorted(out.world.ledger, key=lambda a: a.call)] == [k["p"], k["q"]]


def test_keyed_job_rejoins_original_job(db):
    out = go(db, [Node("b", "book", {"call": "b"})], faults={"b": ["lost"]})
    assert out.status("b") == st.SUCCEEDED
    assert out.world.applications("b") == 1
    task_ids = [e["data"]["task_id"] for e in out.events if e["kind"] == "task_started"]
    assert len(task_ids) == 1


def test_keyed_write_out_of_attempts_after_unknown_outcome_needs_verification(db):
    out = go(db, [Node("p", "pay", {"call": "p"})], faults={"p": ["lost", "reject", "reject"]})
    assert out.status("p") == st.NEEDS_VERIFICATION


def test_idempotent_write_is_repeated(db):
    out = go(db, [Node("s", "set_status", {"call": "s"})], faults={"s": ["lost"]})
    assert out.status("s") == st.SUCCEEDED
    assert out.world.applications("s") == 2


def test_read_job_lost_is_resent(db):
    out = go(db, [Node("j", "job", {"call": "j"})], faults={"j": ["task_lost"]})
    assert out.status("j") == st.SUCCEEDED
    assert len(out.sends("j")) == 2


def test_keyed_job_lost_needs_verification(db):
    out = go(db, [Node("b", "book", {"call": "b"})], faults={"b": ["task_lost"]})
    assert out.status("b") == st.NEEDS_VERIFICATION


def test_job_failure_on_unkeyed_write_needs_verification(db):
    out = go(db, [Node("d", "deploy", {"call": "d"})], faults={"d": ["task_fail"]})
    assert out.status("d") == st.NEEDS_VERIFICATION
    assert out.world.applications("d") == 1


def test_job_timeout_cancels(db):
    reg = registry(job={"job_timeout": 10}, deploy={"job_timeout": 10})
    out = go(db, [Node("j", "job", {"call": "j"}), Node("d", "deploy", {"call": "d"})], reg=reg)
    assert out.status("j") == st.FAILED  # every attempt times out
    assert out.status("d") == st.NEEDS_VERIFICATION
    assert out.world.applications("d") == 0


def test_unregistered_tool_is_treated_as_unkeyed_write(db):
    world = world_of(SimTool("mystery", "unkeyed_write"), faults={"m": ["reject"]})
    out = run(db, world, PolicyRegistry(), [Node("m", "mystery", {"call": "m"})])
    assert out.status("m") == st.FAILED  # one attempt only
    assert len(out.sends("m")) == 1


def test_bad_ref_path_fails_node(db):
    out = go(db, [Node("a", "read", {"call": "a"}), Node("b", "read", {"call": "b", "x": ref("a", "no.such")})])
    assert out.status("b") == st.FAILED
    assert "could not resolve" in out.report.nodes["b"].error


# -- people -------------------------------------------------------------------------


def test_question_pauses_node_until_answered(db):
    world = world_of(*ALL, faults={"p": ["ask"]})
    out = run(db, world, registry(), [Node("p", "pay", {"call": "p"}), Node("r", "read", {"call": "r"})])
    assert out.report.status == "blocked"
    assert out.status("p") == st.WAITING_INPUT and out.status("r") == st.SUCCEEDED
    assert out.report.nodes["p"].question["prompt"] == "answer for p"

    async def answer(sched):
        sched.answer("p", "123456")

    out = run(db, world, registry(), steps=answer)
    assert out.status("p") == st.SUCCEEDED
    assert world.applications("p") == 1


def test_question_answered_while_running(db):
    async def human(sched):
        import asyncio

        while sched.store.node("r", "p").status != st.WAITING_INPUT:
            await asyncio.sleep(1)
        sched.answer("p", "42")

    out = go(db, [Node("p", "pay", {"call": "p"})], faults={"p": ["ask"]}, steps=human, wait_for_humans=True)
    assert out.report.status == "completed"


def test_job_asks_question_and_resumes(db):
    async def human(sched):
        import asyncio

        while sched.store.node("r", "b").status != st.WAITING_INPUT:
            await asyncio.sleep(1)
        assert sched.store.node("r", "b").task_id
        sched.answer("b", {"seat": "12A"})

    out = go(db, [Node("b", "book", {"call": "b"})], faults={"b": ["ask"]}, steps=human, wait_for_humans=True)
    assert out.status("b") == st.SUCCEEDED
    assert "answer_sent" in out.kinds("b")
    assert len(out.sends("b")) == 1


def test_resolve_verification_applied(db):
    world = world_of(*ALL, faults={"e": ["lost"]})
    nodes = [Node("e", "email", {"call": "e"}), Node("f", "read", {"call": "f", "x": ref("e", "ok")})]
    run(db, world, registry(), nodes)

    async def verify(sched):
        sched.resolve_verification("e", applied=True, result={"ok": True})

    out = run(db, world, registry(), steps=verify)
    assert out.report.status == "completed"
    assert out.report.result("e") == {"ok": True}
    assert world.applications("e") == 1


def test_resolve_verification_not_applied_resends(db):
    world = world_of(*ALL, faults={"e": ["error_clean"]})
    run(db, world, registry(), [Node("e", "email", {"call": "e"})])

    async def verify(sched):
        sched.resolve_verification("e", applied=False)

    out = run(db, world, registry(), steps=verify)
    assert out.status("e") == st.SUCCEEDED
    assert world.applications("e") == 1


def test_resolve_verification_not_applied_without_retry_fails(db):
    world = world_of(*ALL, faults={"e": ["lost"]})
    run(db, world, registry(), [Node("e", "email", {"call": "e"})])

    async def verify(sched):
        sched.resolve_verification("e", applied=False, retry=False)

    out = run(db, world, registry(), steps=verify)
    assert out.status("e") == st.FAILED


def test_answer_and_verify_reject_wrong_state(db):
    go(db, [Node("a", "read", {"call": "a"})])
    s = Store(db)
    sched = Scheduler(s, SimBackend(world_of(*ALL)), registry(), run_id="r")
    with pytest.raises(ValueError):
        sched.answer("a", 1)
    with pytest.raises(ValueError):
        sched.resolve_verification("a", applied=True)
    s.close()


# -- the run ------------------------------------------------------------------------


def test_dynamic_replanning(db):
    seen = []

    async def planner(view):
        seen.append(view.wave)
        if view.wave == 1:
            return [Node("b", "read", {"call": "b", "x": ref("a", "v")})]
        if view.wave == 2:
            return [Node("c", "pay", {"call": "c", "y": view.result("b")["v"]})]
        return None

    out = go(db, [Node("a", "read", {"call": "a"})], planner=planner)
    assert out.report.status == "completed"
    assert seen == [1, 2, 3]
    assert [out.report.nodes[n].wave for n in "abc"] == [0, 1, 2]


def test_max_waves_stops_replanning(db):
    n = 0

    async def planner(view):
        nonlocal n
        n += 1
        return [Node(f"x{n}", "read", {"call": f"x{n}"})]

    out = go(db, [Node("a", "read", {"call": "a"})], planner=planner, max_waves=3)
    assert len(out.report.nodes) == 3


def test_start_twice_does_not_duplicate(db):
    s = Store(db)
    sched = Scheduler(s, SimBackend(world_of(*ALL)), registry(), run_id="r")
    assert sched.start([Node("a", "read")])
    assert not sched.start([Node("a", "read"), Node("b", "read")])
    assert [r.id for r in s.nodes("r")] == ["a"]
    s.close()


def test_run_requires_start(db):
    import asyncio

    s = Store(db)
    sched = Scheduler(s, SimBackend(world_of(*ALL)), registry(), run_id="nope")
    with pytest.raises(GraphError):
        asyncio.run(sched.run())
    s.close()


def test_completed_run_is_recorded(db):
    go(db, [Node("a", "read", {"call": "a"})])
    s = Store(db)
    assert s.run_status("r") == "completed"
    s.close()
