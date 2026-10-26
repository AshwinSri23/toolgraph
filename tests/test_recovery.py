"""Crash recovery: kill the process at every commit point and restart from the database."""

import pytest

from toolgraph import store as st
from toolgraph.sim import NodeSpec, Scenario, check, crash_points, run_scenario


def kinds(result, node):
    return [e["kind"] for e in result.events if e["node"] == node]


def sends(result, node):
    return sum(1 for e in result.events if e["node"] == node and e["kind"] == "sending")


@pytest.mark.parametrize(
    "tool, expected",
    [
        ("plain.read", "recovered_resend"),
        ("plain.keyed_write", "recovered_resend"),
        ("plain.idempotent_write", "recovered_resend"),
        ("plain.unkeyed_write", "recovered_unknown"),
        ("tasks.read", "recovered_resend"),
        ("tasks.keyed_write", "recovered_resend"),
        ("tasks.unkeyed_write", "recovered_unknown"),
    ],
)
def test_call_in_flight_at_crash_is_handled_by_policy(tool, expected):
    sc = Scenario([NodeSpec("a", tool)])
    r = run_scenario(sc, crash=("sending", 1))
    assert r.crashes == 1
    assert expected in kinds(r, "a")
    assert check(r) == []
    assert r.nodes["a"].status == st.SUCCEEDED


def test_crash_after_send_before_result_keyed_write_applies_once():
    sc = Scenario([NodeSpec("a", "plain.keyed_write")])
    r = run_scenario(sc, crash=("received", 1))
    assert r.world.applications("a") == 1
    assert sends(r, "a") == 2
    assert len({e["data"]["key"] for e in r.events if e["kind"] == "sending"}) == 1


def test_crash_after_send_before_result_unkeyed_write_is_not_resent():
    sc = Scenario([NodeSpec("a", "plain.unkeyed_write")])
    r = run_scenario(sc, crash=("received", 1))
    assert "recovered_unknown" in kinds(r, "a")
    assert "verified_applied" in kinds(r, "a")  # the read-back found it
    assert sends(r, "a") == 1
    assert r.world.applications("a") == 1


def test_running_job_is_polled_again_not_resent():
    sc = Scenario([NodeSpec("a", "tasks.unkeyed_write")])
    r = run_scenario(sc, crash=("task_started", 1))
    assert r.crashes == 1
    assert sends(r, "a") == 1
    assert r.nodes["a"].status == st.SUCCEEDED
    assert r.world.applications("a") == 1


def test_question_survives_restart():
    sc = Scenario([NodeSpec("a", "plain.keyed_write", faults=["ask"])])
    r = run_scenario(sc, crash=("asked", 1))
    assert r.nodes["a"].status == st.SUCCEEDED
    assert "answered" in kinds(r, "a")


def test_answer_survives_restart():
    sc = Scenario([NodeSpec("a", "tasks.keyed_write", faults=["ask"])])
    r = run_scenario(sc, crash=("answered", 1))
    assert r.nodes["a"].status == st.SUCCEEDED
    assert check(r) == []


def test_retry_timer_survives_restart():
    sc = Scenario([NodeSpec("a", "plain.read", faults=["reject"])])
    r = run_scenario(sc, crash=("retry_wait", 1))
    assert r.nodes["a"].status == st.SUCCEEDED
    assert sends(r, "a") == 2


def test_crash_while_adding_a_wave():
    sc = Scenario([NodeSpec("a", "plain.read")], waves=[[NodeSpec("b", "plain.keyed_write", ["a"])]])
    r = run_scenario(sc, crash=("nodes_added", 1))
    assert r.nodes["b"].status == st.SUCCEEDED
    assert check(r) == []


SCENARIOS = {
    "travel": Scenario(
        [
            NodeSpec("flights", "plain.read"),
            NodeSpec("hotels", "tasks.read"),
            NodeSpec("details", "plain.read", ["flights", "hotels"]),
            NodeSpec("book", "tasks.keyed_write", ["details"], resource="trip", faults=["lost"]),
            NodeSpec("email", "plain.unkeyed_write", ["book", "hotels"], resource="trip"),
        ]
    ),
    "mixed_faults": Scenario(
        [
            NodeSpec("a", "plain.read", resource="r0", faults=["hang"]),
            NodeSpec("b", "tasks.read", faults=["task_lost"]),
            NodeSpec("c", "plain.keyed_write", ["a", "b"], resource="r0", faults=["lost"]),
            NodeSpec("d", "tasks.unkeyed_write", ["c"], faults=["ask"]),
            NodeSpec("e", "plain.unkeyed_write", ["a"], faults=["lost"]),
        ],
        waves=[[NodeSpec("f", "tasks.keyed_write", ["d"], faults=["lost"])]],
    ),
    "hazards": Scenario(
        [
            NodeSpec("w1", "plain.idempotent_write", resource="file"),
            NodeSpec("r1", "plain.read", resource="file", faults=["reject"]),
            NodeSpec("w2", "tasks.keyed_write", resource="file", faults=["ask"]),
            NodeSpec("r2", "tasks.read", resource="file"),
        ],
        max_concurrency=2,
        group_limit=1,
    ),
    "partial_failures": Scenario(
        [
            NodeSpec("p", "plain.unkeyed_write", faults=["error_applied"]),
            NodeSpec("q", "plain.unkeyed_write", faults=["error_clean"]),
            NodeSpec("j", "tasks.unkeyed_write", faults=["task_fail"]),
            NodeSpec("k", "tasks.keyed_write", faults=["task_fail"]),
            NodeSpec("z", "plain.read", ["p", "q", "j", "k"]),
        ],
        max_concurrency=3,
    ),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_crash_at_every_commit_point(name):
    sc = SCENARIOS[name]
    points = crash_points(sc)
    assert len(points) > 20
    failures = {}
    for point in points:
        r = run_scenario(sc, crash=point)
        problems = check(r)
        if r.crashes != 1:
            problems.append(f"expected one crash, saw {r.crashes}")
        if problems:
            failures[point] = problems
    assert not failures, failures


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_clean_run_satisfies_invariants(name):
    r = run_scenario(SCENARIOS[name])
    assert r.status in ("completed", "failed")
    assert check(r) == []


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_replays_identically(name):
    a = run_scenario(SCENARIOS[name], crash=("received", 3))
    b = run_scenario(SCENARIOS[name], crash=("received", 3))
    assert a.events == b.events
    assert [(x.call, x.at) for x in a.world.ledger] == [(x.call, x.at) for x in b.world.ledger]
