"""Kill the shop agent with SIGKILL at random moments, restart it each time, then audit the shop's ledger.

    python scenarios/kill_storm.py --kills 15 --orders 20 --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"nothing listening on {port}")


def start_shop(db: str, port: int, delay: float = 0.4) -> subprocess.Popen:
    proc = subprocess.Popen(
        [PY, os.path.join(HERE, "shop_server.py"), "--port", str(port), "--db", db, "--delay", str(delay)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    wait_for_port(port)
    return proc


def agent_cmd(db: str, url: str, orders: int) -> list[str]:
    return [PY, os.path.join(HERE, "agent.py"), "shop", "--db", db, "--url", url, "--orders", str(orders)]


def audit(shop_db: str, orders: int, summary: dict) -> list[str]:
    """Check the shop's own records: what really happened in the world."""
    db = sqlite3.connect(shop_db)
    problems = []
    charges = dict(db.execute("SELECT order_id, COUNT(*) FROM applied WHERE kind = 'charge' GROUP BY order_id"))
    emails = db.execute("SELECT body, COUNT(*) FROM emails GROUP BY body").fetchall()
    for i in range(orders):
        n = charges.get(f"o{i}", 0)
        if n != 1:
            problems.append(f"order o{i} charged {n} times")
    for body, n in emails:
        if n != 1:
            problems.append(f"email {body!r} sent {n} times")
    if len(emails) != orders:
        problems.append(f"{len(emails)} distinct emails for {orders} orders")
    statuses = dict(db.execute("SELECT order_id, value FROM status"))
    if len(statuses) != orders or set(statuses.values()) != {"paid"}:
        problems.append(f"statuses: {statuses}")
    if summary.get("status") != "completed":
        problems.append(f"run ended {summary.get('status')}")
    # The receipts must quote the charge that was actually made.
    charge_ids = {r[0] for r in db.execute("SELECT charge_id FROM charges")}
    for body, _ in emails:
        if body not in charge_ids:
            problems.append(f"email quotes unknown charge {body!r}")
    return problems


def storm(kills: int, orders: int, seed: int, workdir: str, min_s: float = 0.3, max_s: float = 2.5) -> dict:
    rng = random.Random(seed)
    port = free_port()
    shop_db, agent_db = os.path.join(workdir, "shop.db"), os.path.join(workdir, "agent.db")
    shop = start_shop(shop_db, port)
    url = f"http://127.0.0.1:{port}/mcp"
    killed = 0
    try:
        for _ in range(kills):
            proc = subprocess.Popen(agent_cmd(agent_db, url, orders), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            time.sleep(rng.uniform(min_s, max_s))
            if proc.poll() is None:
                proc.send_signal(signal.SIGKILL)
                killed += 1
            proc.wait()
        out = subprocess.run(agent_cmd(agent_db, url, orders), capture_output=True, text=True, timeout=300)
        line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
        summary = json.loads(line[7:]) if line else {"status": "no result", "stderr": out.stderr[-2000:]}
    finally:
        shop.send_signal(signal.SIGTERM)
        shop.wait(10)
    problems = audit(shop_db, orders, summary)
    return {
        "killed": killed,
        "resumes": summary.get("resumes"),
        "recovered": len(summary.get("recovered", [])),
        "verified_by_read_back": summary.get("verified", []),
        "retried": sum(1 for n in summary.get("nodes", {}).values() if n["attempts"] > 1),
        "problems": problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kills", type=int, default=15)
    parser.add_argument("--orders", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as d:
        print(json.dumps(storm(args.kills, args.orders, args.seed, d), indent=2))


if __name__ == "__main__":
    main()
