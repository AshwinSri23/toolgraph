"""SIGKILL a coding agent at every commit point while it edits real files, then audit the files.

The agent drives the official MCP filesystem server. One of its edits is not
idempotent (applied twice it reads ``retries = 1 + 1 + 1``), so a blind replay
after a crash would show up in the file itself.

    python scenarios/files_sweep.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "node_modules/@modelcontextprotocol/server-filesystem/dist/index.js")
EXPECTED = "retries = 1 + 1\nname = 'demo'\n"


def agent(db: str, root: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, os.path.join(HERE, "agent.py"), "files", "--db", db, "--root", root, "--server", SERVER],
        capture_output=True, text=True, timeout=120, env={**os.environ, **(env or {})},
    )


def summary_of(out: subprocess.CompletedProcess) -> dict:
    line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
    return json.loads(line[7:]) if line else {"status": "no result", "stderr": out.stderr[-1500:]}


def audit(root: str, summary: dict) -> list[str]:
    problems = []
    app = os.path.join(root, "src/app.py")
    content = open(app).read() if os.path.exists(app) else None
    if content != EXPECTED:
        problems.append(f"app.py is {content!r}")
    if not os.path.exists(os.path.join(root, "src/test_app.py")):
        problems.append("test_app.py missing")
    if summary.get("status") != "completed":
        problems.append(f"run ended {summary.get('status')}: {summary.get('stderr', '')}")
    else:
        if summary["nodes"]["read_app"]["result"] != {"content": EXPECTED}:
            problems.append("read_app ran before the edit (hazard ordering broken)")
        if "app.py" not in json.dumps(summary["nodes"]["search"]["result"]):
            problems.append("search ran before the files existed")
    return problems


def trace_points(workdir: str) -> list[tuple[str, int]]:
    root, db, trace = (os.path.join(workdir, n) for n in ("trace_root", "trace.db", "trace.txt"))
    os.makedirs(root)
    out = agent(db, root, {"TOOLGRAPH_TRACE": trace})
    problems = audit(root, summary_of(out))
    assert not problems, problems
    counts: Counter[str] = Counter()
    points = []
    for name in open(trace).read().split():
        counts[name] += 1
        points.append((name, counts[name]))
    return points


def main() -> None:
    results = {"points": 0, "killed": 0, "verified_by_read_back": 0, "problems": {}}
    with tempfile.TemporaryDirectory() as d:
        points = trace_points(d)
        for i, (name, n) in enumerate(points):
            root, db = os.path.join(d, f"root{i}"), os.path.join(d, f"run{i}.db")
            os.makedirs(root)
            first = agent(db, root, {"TOOLGRAPH_KILL_AT": f"{name}:{n}"})
            results["points"] += 1
            results["killed"] += first.returncode == -9
            summary = summary_of(agent(db, root))
            results["verified_by_read_back"] += bool(summary.get("verified"))
            problems = audit(root, summary)
            if first.returncode != -9:
                problems.append(f"expected SIGKILL, exit {first.returncode}")
            if problems:
                results["problems"][f"{name}:{n}"] = problems
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
