import pytest

from toolgraph import store as st
from toolgraph.graph import PlannedNode
from toolgraph.store import Store, StoreDead


@pytest.fixture
def db(tmp_path):
    s = Store(str(tmp_path / "t.db"), clock=lambda: 1.0)
    yield s
    s.close()


def nodes():
    return [PlannedNode("a", "t", {"x": 1}, 0, {}), PlannedNode("b", "t", {"y": {"$ref": "a", "path": ""}}, 1, {"a": "data"})]


def test_create_run_is_atomic_and_idempotent(db):
    assert db.create_run("r", nodes(), {"a": "k1"})
    assert not db.create_run("r", [], {})
    rows = db.nodes("r")
    assert [r.id for r in rows] == ["a", "b"]
    assert rows[0].idem_key == "k1" and rows[1].idem_key is None
    assert rows[1].deps == {"a": "data"}
    assert db.run_status("r") == "active"


def test_update_round_trips_fields(db):
    db.create_run("r", nodes(), {})
    db.update("r", "a", st.WAITING_INPUT, "asked", question={"prompt": "?"}, has_answer=False)
    row = db.node("r", "a")
    assert row.status == st.WAITING_INPUT and row.question == {"prompt": "?"} and not row.has_answer
    db.update("r", "a", st.WAITING_INPUT, "answered", has_answer=True, answer=None)
    row = db.node("r", "a")
    assert row.has_answer and row.answer is None
    db.update("r", "a", st.SUCCEEDED, "succeeded", result={"v": [1, 2]})
    assert db.node("r", "a").result == {"v": [1, 2]}


def test_events_record_status(db):
    db.create_run("r", nodes(), {})
    db.update("r", "a", st.SENDING, "sending", data={"attempt": 1}, attempts=1)
    kinds = [(e["kind"], e["node"]) for e in db.events("r")]
    assert kinds == [("run_created", None), ("added", "a"), ("added", "b"), ("sending", "a")]
    assert db.events("r")[-1]["data"] == {"attempt": 1, "status": "sending"}


def test_add_nodes_increments_wave(db):
    db.create_run("r", nodes()[:1], {})
    wave = db.add_nodes("r", [PlannedNode("c", "t", {}, 2, {})], {})
    assert wave == 1
    assert db.node("r", "c").wave == 1


def test_unknown_node_update_rolls_back(db):
    db.create_run("r", nodes(), {})
    n = len(db.events("r"))
    with pytest.raises(KeyError):
        db.update("r", "zzz", st.FAILED, "failed")
    assert len(db.events("r")) == n


def test_hook_runs_after_commit(tmp_path):
    seen = []
    s = Store(str(tmp_path / "h.db"), hook=seen.append)
    s.create_run("r", nodes(), {})
    s.update("r", "a", st.SENDING, "sending")
    assert seen == ["run_created", "sending"]
    s.close()


def test_dead_store_refuses_writes(db):
    db.create_run("r", nodes(), {})
    db.kill()
    with pytest.raises(StoreDead):
        db.update("r", "a", st.SENDING, "sending")
    with pytest.raises(StoreDead):
        db.point("received")


def test_state_survives_reopen(tmp_path):
    path = str(tmp_path / "p.db")
    s = Store(path)
    s.create_run("r", nodes(), {})
    s.update("r", "a", st.RUNNING, "task_started", task_id="job-5")
    s.close()
    s2 = Store(path)
    assert s2.node("r", "a").task_id == "job-5"
    s2.close()
