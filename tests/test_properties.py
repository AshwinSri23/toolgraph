"""Property-based tests: random graphs, faults, questions and crashes must keep every invariant."""

from hypothesis import given, note
from hypothesis import strategies as hs

from toolgraph.sim import NodeSpec, Scenario, check, crash_points, run_scenario
from toolgraph.sim.harness import EFFECTS, PROTOCOLS

PLAIN_FAULTS = ("ok", "reject", "lost", "hang", "error_applied", "error_clean", "ask")
TASK_FAULTS = ("ok", "reject", "lost", "hang", "ask", "task_fail", "task_lost")


@hs.composite
def node_specs(draw, ident: str, earlier: list[str]) -> NodeSpec:
    protocol = draw(hs.sampled_from(PROTOCOLS))
    effect = draw(hs.sampled_from(EFFECTS))
    deps = draw(hs.lists(hs.sampled_from(earlier), unique=True, max_size=2)) if earlier else []
    resource = draw(hs.none() | hs.sampled_from(["r0", "r1"]))
    faults = draw(hs.lists(hs.sampled_from(PLAIN_FAULTS if protocol == "plain" else TASK_FAULTS), max_size=3))
    return NodeSpec(ident, f"{protocol}.{effect}", deps, resource, faults)


@hs.composite
def scenarios(draw, max_nodes: int = 7) -> Scenario:
    ids: list[str] = []
    first = []
    for i in range(draw(hs.integers(1, max_nodes))):
        first.append(draw(node_specs(f"n{i}", list(ids))))
        ids.append(f"n{i}")
    waves = []
    for w in range(draw(hs.integers(0, 2))):
        wave = []
        for i in range(draw(hs.integers(1, 3))):
            ident = f"w{w}_{i}"
            wave.append(draw(node_specs(ident, list(ids))))
            ids.append(ident)
        waves.append(wave)
    return Scenario(first, waves, max_concurrency=draw(hs.integers(1, 4)), group_limit=draw(hs.integers(1, 2)))


def assert_ok(result):
    problems = check(result)
    note(f"status={result.status} crashes={result.crashes}")
    note(f"nodes={ {k: (v.status, v.attempts) for k, v in result.nodes.items()} }")
    assert problems == [], problems


@given(scenarios())
def test_invariants_hold(sc):
    assert_ok(run_scenario(sc))


@given(scenarios(), hs.floats(0, 1, exclude_max=True))
def test_invariants_hold_across_a_crash(sc, where):
    points = crash_points(sc)
    point = points[int(where * len(points))]
    note(f"crash at {point}")
    r = run_scenario(sc, crash=point)
    assert r.crashes == 1
    assert_ok(r)


@given(scenarios(max_nodes=4), hs.floats(0, 1, exclude_max=True))
def test_runs_are_deterministic(sc, where):
    points = crash_points(sc)
    point = points[int(where * len(points))]
    a, b = run_scenario(sc, crash=point), run_scenario(sc, crash=point)
    assert a.events == b.events
    assert [(x.call, x.at) for x in a.world.ledger] == [(x.call, x.at) for x in b.world.ledger]
