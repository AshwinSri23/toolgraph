"""Real-world scenarios: separate processes, real transports, real SIGKILLs, real reference servers.

These are slower than the simulation and need Node.js plus the pinned reference
servers (``cd scenarios && npm install``). Run them with ``pytest -m realworld``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = os.path.join(os.path.dirname(HERE), "scenarios")
NODE_MODULES = os.path.join(SCENARIOS, "node_modules", "@modelcontextprotocol")
sys.path.insert(0, SCENARIOS)

pytestmark = pytest.mark.realworld

needs_node = pytest.mark.skipif(
    shutil.which("node") is None or not os.path.isdir(NODE_MODULES),
    reason="needs node and `npm install` in scenarios/",
)


def run_agent(*args: str, env: dict[str, str] | None = None) -> tuple[int, dict]:
    out = subprocess.run(
        [sys.executable, os.path.join(SCENARIOS, "agent.py"), *args],
        capture_output=True, text=True, timeout=300, env={**os.environ, **(env or {})},
    )
    line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
    return out.returncode, json.loads(line[7:]) if line else {"status": "no result", "stderr": out.stderr[-2000:]}


def test_shop_survives_random_sigkills(tmp_path):
    from kill_storm import storm

    report = storm(kills=12, orders=20, seed=7, workdir=str(tmp_path))
    assert report["problems"] == []
    assert report["killed"] >= 5


def test_shop_survives_lost_responses(tmp_path):
    from flaky_network import flaky

    report = flaky(orders=12, drop=0.5, seed=3, workdir=str(tmp_path))
    assert report["problems"] == []
    assert report["dropped_responses"] > 0
    assert report["verified_by_read_back"] or report["retried"]


@pytest.fixture(scope="module")
def everything_server():
    from kill_storm import free_port, wait_for_port

    port = free_port()
    proc = subprocess.Popen(
        ["node", os.path.join(NODE_MODULES, "server-everything/dist/index.js"), "streamableHttp"],
        env={**os.environ, "PORT": str(port)}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(10)


@needs_node
def test_research_tasks_on_reference_server(tmp_path, everything_server):
    code, summary = run_agent("research", "--db", str(tmp_path / "r.db"), "--url", everything_server)
    assert code == 0, summary
    nodes = summary["nodes"]
    assert "python (programming)" in nodes["ambiguous"]["result"]  # answered through tasks/update
    assert nodes["echo"]["result"] == "Echo: " + nodes["ambiguous"]["result"]
    assert summary["task_ids"]["plain"] and summary["task_ids"]["ambiguous"]


@needs_node
@pytest.mark.parametrize("point", ["task_started:1", "task_started:2", "asked:1", "answered:1", "answer_sent:1"])
def test_research_resumes_tasks_after_sigkill(tmp_path, everything_server, point):
    db = str(tmp_path / "r.db")
    code, _ = run_agent("research", "--db", db, "--url", everything_server, env={"TOOLGRAPH_KILL_AT": point})
    assert code == -signal.SIGKILL
    code, summary = run_agent("research", "--db", db, "--url", everything_server)
    assert code == 0, summary
    assert "python (programming)" in summary["nodes"]["ambiguous"]["result"]
    # The job was picked up again, not started a second time.
    assert all(len(ids) == 1 for ids in summary["task_ids"].values())


@needs_node
def test_coding_agent_sigkilled_at_every_commit_point(tmp_path):
    import files_sweep

    points = files_sweep.trace_points(str(tmp_path))
    assert len(points) > 20
    failures = {}
    for i, (name, n) in enumerate(points):
        root, db = tmp_path / f"root{i}", str(tmp_path / f"run{i}.db")
        root.mkdir()
        first = files_sweep.agent(db, str(root), {"TOOLGRAPH_KILL_AT": f"{name}:{n}"})
        problems = files_sweep.audit(str(root), files_sweep.summary_of(files_sweep.agent(db, str(root))))
        if first.returncode != -signal.SIGKILL:
            problems.append(f"exit {first.returncode}")
        if problems:
            failures[f"{name}:{n}"] = problems
    assert not failures, failures
