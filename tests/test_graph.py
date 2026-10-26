import pytest

from toolgraph.graph import (
    DATA,
    RAW,
    WAR,
    WAW,
    GraphError,
    Node,
    encode_args,
    iter_refs,
    lookup,
    overlap,
    plan,
    ref,
    resolve_args,
    resources,
)
from toolgraph.policy import PolicyRegistry, SideEffect, ToolPolicy


@pytest.fixture
def reg():
    r = PolicyRegistry()
    r.register("read_file", side_effect=SideEffect.READ, reads=("file:{path}",))
    r.register("edit_file", side_effect=SideEffect.IDEMPOTENT_WRITE, writes=("file:{path}",))
    r.register("run_tests", side_effect=SideEffect.READ, reads=("file:*",))
    r.register("search", side_effect=SideEffect.READ)
    return r


def deps(planned):
    return {p.id: p.deps for p in planned}


def test_ref_creates_data_edge(reg):
    p = plan([Node("a", "search"), Node("b", "search", {"q": ref("a", "items.0")})], reg.get)
    assert deps(p) == {"a": {}, "b": {"a": DATA}}


def test_independent_calls_have_no_edges(reg):
    p = plan([Node("a", "search"), Node("b", "search"), Node("c", "search")], reg.get)
    assert all(not d for d in deps(p).values())


def test_refs_nested_in_lists_and_dicts(reg):
    args = {"x": [1, {"y": ref("a")}], "z": ref("b", "k")}
    p = plan([Node("a", "search"), Node("b", "search"), Node("c", "search", args)], reg.get)
    assert deps(p)["c"] == {"a": DATA, "b": DATA}


def test_write_after_read_hazard(reg):
    p = plan([Node("r", "read_file", {"path": "x"}), Node("w", "edit_file", {"path": "x"})], reg.get)
    assert deps(p)["w"] == {"r": WAR}


def test_read_after_write_hazard(reg):
    p = plan([Node("w", "edit_file", {"path": "x"}), Node("r", "read_file", {"path": "x"})], reg.get)
    assert deps(p)["r"] == {"w": RAW}


def test_write_after_write_hazard(reg):
    p = plan([Node("w1", "edit_file", {"path": "x"}), Node("w2", "edit_file", {"path": "x"})], reg.get)
    assert deps(p)["w2"] == {"w1": WAW}


def test_reads_do_not_conflict(reg):
    p = plan([Node("r1", "read_file", {"path": "x"}), Node("r2", "read_file", {"path": "x"})], reg.get)
    assert deps(p)["r2"] == {}


def test_different_resources_do_not_conflict(reg):
    p = plan([Node("w1", "edit_file", {"path": "x"}), Node("w2", "edit_file", {"path": "y"})], reg.get)
    assert deps(p)["w2"] == {}


def test_edit_then_run_tests_is_ordered_by_wildcard(reg):
    p = plan([Node("e", "edit_file", {"path": "src/a.py"}), Node("t", "run_tests")], reg.get)
    assert deps(p)["t"] == {"e": RAW}


def test_ref_in_resource_arg_widens_to_wildcard(reg):
    nodes = [
        Node("s", "search"),
        Node("w", "edit_file", {"path": ref("s", "path")}),
        Node("r", "read_file", {"path": "anything"}),
    ]
    assert resources(("file:{path}",), encode_args(nodes[1].args)) == ["file:*"]
    assert deps(plan(nodes, reg.get))["r"] == {"w": RAW}


def test_missing_resource_arg_names_no_resource():
    assert resources(("file:{path}",), {}) == []


def test_data_edge_takes_precedence_over_hazard(reg):
    p = plan([Node("w", "edit_file", {"path": "x"}), Node("r", "read_file", {"path": "x", "v": ref("w")})], reg.get)
    assert deps(p)["r"] == {"w": DATA}


def test_unknown_ref_rejected(reg):
    with pytest.raises(GraphError, match="unknown"):
        plan([Node("a", "search", {"q": ref("nope")})], reg.get)


def test_self_ref_rejected(reg):
    with pytest.raises(GraphError, match="itself"):
        plan([Node("a", "search", {"q": ref("a")})], reg.get)


def test_duplicate_id_rejected(reg):
    with pytest.raises(GraphError, match="duplicate"):
        plan([Node("a", "search"), Node("a", "search")], reg.get)


def test_cycle_rejected(reg):
    with pytest.raises(GraphError, match="cycle"):
        plan([Node("a", "search", {"q": ref("b")}), Node("b", "search", {"q": ref("a")})], reg.get)


def test_forward_ref_within_wave_is_allowed(reg):
    p = plan([Node("b", "search", {"q": ref("a")}), Node("a", "search")], reg.get)
    assert deps(p)["b"] == {"a": DATA}


def test_later_wave_sees_existing_nodes(reg):
    first = plan([Node("w", "edit_file", {"path": "x"})], reg.get)
    second = plan([Node("r", "read_file", {"path": "x"})], reg.get, existing=first)
    assert second[0].seq == 1
    assert second[0].deps == {"w": RAW}


def test_hazards_can_be_disabled(reg):
    p = plan([Node("w", "edit_file", {"path": "x"}), Node("r", "read_file", {"path": "x"})], reg.get, hazards=False)
    assert deps(p)["r"] == {}


def test_lookup_paths():
    v = {"items": [{"id": 7}], "n": 1}
    assert lookup(v, "") == v
    assert lookup(v, "items.0.id") == 7
    with pytest.raises(KeyError):
        lookup(v, "missing")


def test_resolve_args_replaces_refs():
    args = encode_args({"a": ref("x", "id"), "b": [ref("y")], "c": 3})
    assert sorted(iter_refs(args)) == [("x", "id"), ("y", "")]
    results = {"x": {"id": 5}, "y": "hello"}
    assert resolve_args(args, results.__getitem__) == {"a": 5, "b": ["hello"], "c": 3}


def test_overlap():
    assert overlap("file:a", "file:a")
    assert overlap("file:*", "file:a")
    assert overlap("file:a", "file:*")
    assert not overlap("file:a", "file:b")
    assert not overlap("db:a", "file:*")


def test_policy_registry_glob_and_default():
    r = PolicyRegistry()
    r.register("github.*", side_effect=SideEffect.READ)
    r.register("github.create_issue", side_effect=SideEffect.KEYED_WRITE)
    assert r.get("github.list").side_effect is SideEffect.READ
    assert r.get("github.create_issue").side_effect is SideEffect.KEYED_WRITE
    unknown = r.get("email.send")
    assert unknown.side_effect is SideEffect.UNKEYED_WRITE
    assert unknown.max_attempts == 1


def test_policy_backoff_is_capped():
    p = ToolPolicy(backoff_base=1.0, backoff_max=5.0)
    assert [p.backoff(i) for i in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 5.0]


def test_tool_error_uncertainty_defaults():
    assert ToolPolicy(side_effect=SideEffect.UNKEYED_WRITE).tool_error_uncertain
    assert not ToolPolicy(side_effect=SideEffect.READ).tool_error_uncertain
    assert not ToolPolicy(side_effect=SideEffect.KEYED_WRITE, error_may_have_applied=False).tool_error_uncertain


def test_side_effect_replay_rules():
    assert SideEffect.READ.safe_to_replay and not SideEffect.READ.is_write
    assert SideEffect.KEYED_WRITE.safe_to_replay
    assert SideEffect.IDEMPOTENT_WRITE.safe_to_replay
    assert not SideEffect.UNKEYED_WRITE.safe_to_replay
