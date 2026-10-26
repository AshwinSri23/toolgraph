"""SQLite persistence. Every state change is committed before the scheduler acts on it."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .graph import PlannedNode

# Node statuses.
PENDING = "pending"
SENDING = "sending"  # "about to send": committed before the request goes out
RUNNING = "running"  # the server returned a task handle; we are polling it
WAITING_INPUT = "waiting_input"
RETRY_WAIT = "retry_wait"
NEEDS_VERIFICATION = "needs_verification"
SUCCEEDED = "succeeded"
FAILED = "failed"
SKIPPED = "skipped"  # a dependency failed
CANCELLED = "cancelled"

TERMINAL = frozenset({SUCCEEDED, FAILED, SKIPPED, CANCELLED})
BLOCKED_ON_HUMAN = frozenset({NEEDS_VERIFICATION, WAITING_INPUT})

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created REAL NOT NULL,
    status TEXT NOT NULL,
    waves INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS nodes (
    run_id TEXT NOT NULL,
    id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    wave INTEGER NOT NULL,
    tool TEXT NOT NULL,
    args TEXT NOT NULL,
    deps TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    idem_key TEXT,
    task_id TEXT,
    question TEXT,
    answer TEXT,
    result TEXT,
    error TEXT,
    not_before REAL,
    started REAL,
    task_started REAL,
    maybe_applied INTEGER NOT NULL DEFAULT 0,
    updated REAL NOT NULL,
    PRIMARY KEY (run_id, id)
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    node_id TEXT,
    kind TEXT NOT NULL,
    at REAL NOT NULL,
    data TEXT
);
CREATE INDEX IF NOT EXISTS events_run ON events (run_id, seq);
"""


class StoreDead(BaseException):
    """Raised on any write after a simulated crash: a dead process commits nothing."""


@dataclass
class NodeRow:
    run_id: str
    id: str
    seq: int
    wave: int
    tool: str
    args: dict[str, Any]
    deps: dict[str, str]
    status: str
    attempts: int
    idem_key: str | None
    task_id: str | None
    question: dict[str, Any] | None
    answer: Any
    has_answer: bool
    result: Any
    error: str | None
    not_before: float | None
    started: float | None
    task_started: float | None
    maybe_applied: bool
    updated: float


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


class Store:
    def __init__(
        self,
        path: str,
        *,
        clock: Callable[[], float] | None = None,
        durable: bool = True,
        hook: Callable[[str], None] | None = None,
    ) -> None:
        import time

        self.path = path
        self.clock = clock or time.time
        self.hook = hook
        self.dead = False
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(f"PRAGMA synchronous={'FULL' if durable else 'OFF'}")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # -- crash points ---------------------------------------------------------

    def point(self, name: str) -> None:
        """A place where the process may die. Tests inject crashes here."""
        if self.dead:
            raise StoreDead(name)
        if self.hook is not None:
            self.hook(name)

    def kill(self) -> None:
        self.dead = True

    # -- writes -----------------------------------------------------------------

    def _tx(self, point: str, fn: Callable[[sqlite3.Connection], None]) -> None:
        if self.dead:
            raise StoreDead(point)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            fn(self.db)
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")
        self.point(point)

    def _event(self, db: sqlite3.Connection, run_id: str, node_id: str | None, kind: str, data: Any) -> None:
        db.execute(
            "INSERT INTO events (run_id, node_id, kind, at, data) VALUES (?, ?, ?, ?, ?)",
            (run_id, node_id, kind, self.clock(), json.dumps(data, sort_keys=True) if data is not None else None),
        )

    def create_run(self, run_id: str, nodes: list[PlannedNode], keys: dict[str, str | None]) -> bool:
        """Create the run with its first wave, in one transaction.

        Returns False (and changes nothing) if the run already exists.
        """
        if self.db.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone():
            return False

        def fn(db: sqlite3.Connection) -> None:
            db.execute("INSERT INTO runs (id, created, status) VALUES (?, ?, 'active')", (run_id, self.clock()))
            self._event(db, run_id, None, "run_created", None)
            self._insert_wave(db, run_id, nodes, keys)

        self._tx("run_created", fn)
        return True

    def add_nodes(self, run_id: str, nodes: list[PlannedNode], keys: dict[str, str | None]) -> int:
        """Add a wave of nodes atomically and return its wave number."""
        wave_box: list[int] = []
        self._tx("nodes_added", lambda db: wave_box.append(self._insert_wave(db, run_id, nodes, keys)))
        return wave_box[0]

    def _insert_wave(
        self, db: sqlite3.Connection, run_id: str, nodes: list[PlannedNode], keys: dict[str, str | None]
    ) -> int:
        (wave,) = db.execute("SELECT waves FROM runs WHERE id = ?", (run_id,)).fetchone()
        now = self.clock()
        for n in nodes:
            db.execute(
                "INSERT INTO nodes (run_id, id, seq, wave, tool, args, deps, status, idem_key, updated)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, n.id, n.seq, wave, n.tool, _dump(n.args), _dump(n.deps), PENDING, keys.get(n.id), now),
            )
            self._event(db, run_id, n.id, "added", {"deps": n.deps, "wave": wave})
        db.execute("UPDATE runs SET waves = waves + 1 WHERE id = ?", (run_id,))
        return wave

    def update(
        self,
        run_id: str,
        node_id: str,
        status: str,
        event: str,
        *,
        data: Any = None,
        **fields: Any,
    ) -> None:
        """Change a node's status (and other columns) and log the event, in one transaction."""
        cols = {"status": status, "updated": self.clock()}
        for k, v in fields.items():
            if k in ("question", "result"):
                cols[k] = None if v is None else json.dumps(v, sort_keys=True)
            elif k == "answer":
                continue  # written together with has_answer, since None is a valid answer
            elif k == "has_answer":
                cols["answer"] = json.dumps(fields.get("answer"), sort_keys=True) if v else None
            else:
                cols[k] = v

        def fn(db: sqlite3.Connection) -> None:
            sets = ", ".join(f"{k} = ?" for k in cols)
            cur = db.execute(f"UPDATE nodes SET {sets} WHERE run_id = ? AND id = ?", (*cols.values(), run_id, node_id))
            if cur.rowcount != 1:
                raise KeyError(f"no node {node_id!r} in run {run_id!r}")
            self._event(db, run_id, node_id, event, {"status": status, **(data or {})})

        self._tx(event, fn)

    def log(self, run_id: str, kind: str, data: Any = None, node_id: str | None = None) -> None:
        self._tx(kind, lambda db: self._event(db, run_id, node_id, kind, data))

    def set_run_status(self, run_id: str, status: str) -> None:
        def fn(db: sqlite3.Connection) -> None:
            db.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))
            self._event(db, run_id, None, "run_" + status, None)

        self._tx("run_" + status, fn)

    # -- reads ------------------------------------------------------------------

    def nodes(self, run_id: str) -> list[NodeRow]:
        rows = self.db.execute("SELECT * FROM nodes WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall()
        return [self._row(r) for r in rows]

    def node(self, run_id: str, node_id: str) -> NodeRow:
        r = self.db.execute("SELECT * FROM nodes WHERE run_id = ? AND id = ?", (run_id, node_id)).fetchone()
        if r is None:
            raise KeyError(f"no node {node_id!r} in run {run_id!r}")
        return self._row(r)

    def run_status(self, run_id: str) -> str | None:
        r = self.db.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
        return r[0] if r else None

    def events(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM events WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall()
        return [
            {
                "seq": r["seq"],
                "node": r["node_id"],
                "kind": r["kind"],
                "at": r["at"],
                "data": json.loads(r["data"]) if r["data"] else None,
            }
            for r in rows
        ]

    @staticmethod
    def _row(r: sqlite3.Row) -> NodeRow:
        def load(v: str | None) -> Any:
            return None if v is None else json.loads(v)

        return NodeRow(
            run_id=r["run_id"],
            id=r["id"],
            seq=r["seq"],
            wave=r["wave"],
            tool=r["tool"],
            args=load(r["args"]),
            deps=load(r["deps"]),
            status=r["status"],
            attempts=r["attempts"],
            idem_key=r["idem_key"],
            task_id=r["task_id"],
            question=load(r["question"]),
            answer=load(r["answer"]),
            has_answer=r["answer"] is not None,
            result=load(r["result"]),
            error=r["error"],
            not_before=r["not_before"],
            started=r["started"],
            task_started=r["task_started"],
            maybe_applied=bool(r["maybe_applied"]),
            updated=r["updated"],
        )
