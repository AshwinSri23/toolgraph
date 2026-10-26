"""The call graph: nodes, data edges from refs, and hazard edges from resources."""

from __future__ import annotations

import fnmatch
import string
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .policy import ToolPolicy


class GraphError(ValueError):
    pass


@dataclass(frozen=True)
class Ref:
    """A placeholder for (part of) another node's result."""

    node: str
    path: str = ""

    def to_json(self) -> dict[str, str]:
        return {"$ref": self.node, "path": self.path}


def ref(node: str, path: str = "") -> Ref:
    """Use the result of ``node`` (or the part at dotted ``path``) as an argument."""
    return Ref(node, path)


@dataclass
class Node:
    id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


# Edge kinds. Data edges come from refs; the others order calls that touch the
# same resource, exactly as a compiler orders memory operations that may alias.
DATA = "data"
RAW = "raw"  # read after write
WAR = "war"  # write after read
WAW = "waw"  # write after write


def encode_args(value: Any) -> Any:
    """Turn ``Ref`` objects into their JSON form so args can be stored."""
    if isinstance(value, Ref):
        return value.to_json()
    if isinstance(value, dict):
        return {k: encode_args(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_args(v) for v in value]
    return value


def _is_ref(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"$ref", "path"}


def iter_refs(value: Any) -> Iterator[tuple[str, str]]:
    """Yield ``(node, path)`` for every ref inside encoded args."""
    if _is_ref(value):
        yield value["$ref"], value["path"]
    elif isinstance(value, dict):
        for v in value.values():
            yield from iter_refs(v)
    elif isinstance(value, list):
        for v in value:
            yield from iter_refs(v)


def lookup(value: Any, path: str) -> Any:
    for part in path.split(".") if path else []:
        if isinstance(value, list):
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            raise KeyError(f"cannot index {type(value).__name__} with {part!r}")
    return value


def resolve_args(value: Any, results: Callable[[str], Any]) -> Any:
    """Replace encoded refs with values from ``results(node_id)``."""
    if _is_ref(value):
        return lookup(results(value["$ref"]), value["path"])
    if isinstance(value, dict):
        return {k: resolve_args(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_args(v, results) for v in value]
    return value


def resources(templates: Iterable[str], args: dict[str, Any]) -> list[str]:
    """Format resource templates with a call's (encoded) arguments.

    A field whose value is a ref is not known until run time, so the resource
    is widened to a wildcard that conflicts with everything of that shape. A
    template whose field is missing from the arguments names no resource.
    """
    out = []
    for template in templates:
        parts: list[str] | None = []
        for literal, name, _spec, _conv in string.Formatter().parse(template):
            parts.append(literal)
            if name is None:
                continue
            try:
                value = lookup(args, name)
            except (KeyError, IndexError, ValueError):
                parts = None
                break
            if any(True for _ in iter_refs(value)):
                parts.append("*")
            else:
                parts.append(str(value))
        if parts is not None:
            out.append("".join(parts))
    return out


def overlap(a: str, b: str) -> bool:
    return a == b or fnmatch.fnmatchcase(a, b) or fnmatch.fnmatchcase(b, a)


@dataclass
class Access:
    reads: list[str]
    writes: list[str]


def hazard(earlier: Access, later: Access) -> str | None:
    """The kind of hazard edge needed from ``earlier`` to ``later``, if any."""
    if any(overlap(w, r) for w in earlier.writes for r in later.reads):
        return RAW
    if any(overlap(w1, w2) for w1 in earlier.writes for w2 in later.writes):
        return WAW
    if any(overlap(r, w) for r in earlier.reads for w in later.writes):
        return WAR
    return None


def access_for(policy: ToolPolicy, args: dict[str, Any]) -> Access:
    return Access(resources(policy.reads, args), resources(policy.writes, args))


@dataclass
class PlannedNode:
    id: str
    tool: str
    args: dict[str, Any]
    seq: int
    deps: dict[str, str]
    """dependency id -> edge kind"""


def plan(
    new: list[Node],
    policy_for: Callable[[str], ToolPolicy],
    existing: list[PlannedNode] | None = None,
    hazards: bool = True,
) -> list[PlannedNode]:
    """Compute edges for ``new`` nodes, appended after ``existing`` ones.

    Hazard edges respect listing order: a later call that conflicts with an
    earlier one waits for it.
    """
    existing = existing or []
    known = {n.id: n for n in existing}
    out: list[PlannedNode] = []
    seq = max((n.seq for n in existing), default=-1) + 1
    all_access = [(n, access_for(policy_for(n.tool), n.args)) for n in existing]
    for node in new:
        if not node.id or node.id in known:
            raise GraphError(f"duplicate or empty node id {node.id!r}")
        args = encode_args(node.args)
        p = PlannedNode(node.id, node.tool, args, seq, {})
        seq += 1
        known[node.id] = p
        out.append(p)
    new_ids = {p.id for p in out}
    for p in out:
        for dep, _path in iter_refs(p.args):
            if dep not in known:
                raise GraphError(f"node {p.id!r} refers to unknown node {dep!r}")
            if dep == p.id:
                raise GraphError(f"node {p.id!r} refers to itself")
            p.deps[dep] = DATA
        if hazards:
            mine = access_for(policy_for(p.tool), p.args)
            for other, theirs in all_access:
                kind = hazard(theirs, mine)
                if kind and other.id not in p.deps:
                    p.deps[other.id] = kind
            all_access.append((p, mine))
    _check_acyclic(known, new_ids)
    return out


def _check_acyclic(nodes: dict[str, PlannedNode], new_ids: set[str]) -> None:
    state: dict[str, int] = {}

    def visit(nid: str, stack: list[str]) -> None:
        mark = state.get(nid)
        if mark == 2:
            return
        if mark == 1:
            raise GraphError("dependency cycle: " + " -> ".join(stack + [nid]))
        state[nid] = 1
        for dep in nodes[nid].deps:
            visit(dep, stack + [nid])
        state[nid] = 2

    for nid in sorted(new_ids):
        visit(nid, [])
