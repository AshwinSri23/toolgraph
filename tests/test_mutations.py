"""Mutation checks: break one safety mechanism and the invariants must notice.

Each test asks Hypothesis to find a scenario where the broken scheduler violates
an invariant. If it cannot, the suite is too weak to protect that mechanism.
"""

import pytest
from hypothesis import find, settings
from test_properties import scenarios

from toolgraph.scheduler import Mutations
from toolgraph.sim import check, crash_points, run_scenario

SEARCH = settings(max_examples=500, deadline=None, database=None)


def violates(mutation: Mutations, crash: bool):
    def condition(sc) -> bool:
        if crash:
            for point in crash_points(sc)[::4]:
                if check(run_scenario(sc, crash=point, mutations=mutation)):
                    return True
            return False
        return bool(check(run_scenario(sc, mutations=mutation)))

    return condition


@pytest.mark.parametrize(
    "mutation, crash",
    [
        (Mutations(unsafe_replay=True), False),
        (Mutations(unsafe_replay=True), True),
        (Mutations(fresh_keys=True), False),
        (Mutations(no_hazards=True), False),
    ],
    ids=["unsafe-replay", "unsafe-replay-on-recovery", "fresh-keys", "no-hazard-edges"],
)
def test_suite_catches_mutation(mutation, crash):
    example = find(scenarios(max_nodes=5), violates(mutation, crash), settings=SEARCH)
    assert check(run_scenario(example, mutations=mutation)) or crash


def test_unmutated_scheduler_passes_the_same_search():
    # Sanity check for the test above: the real scheduler is not flagged.
    sc = find(scenarios(max_nodes=5), lambda sc: len(sc.all_nodes()) >= 5, settings=SEARCH)
    assert check(run_scenario(sc)) == []
