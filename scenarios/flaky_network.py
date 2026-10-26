"""Run the shop agent through a proxy that loses responses, then audit the shop's ledger.

    python scenarios/flaky_network.py --orders 20 --drop 0.5 --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile

from kill_storm import HERE, PY, agent_cmd, audit, free_port, start_shop, wait_for_port


def flaky(orders: int, drop: float, seed: int, workdir: str) -> dict:
    shop_port, proxy_port = free_port(), free_port()
    shop_db, agent_db = os.path.join(workdir, "shop.db"), os.path.join(workdir, "agent.db")
    proxy_log = os.path.join(workdir, "proxy.log")
    shop = start_shop(shop_db, shop_port)
    with open(proxy_log, "w") as log:
        proxy = subprocess.Popen(
            [PY, os.path.join(HERE, "chaos_proxy.py"), "--listen", str(proxy_port), "--target", str(shop_port),
             "--tools", "charge,send_email", "--drop", str(drop), "--seed", str(seed)],
            stdout=log, stderr=log,
        )
    try:
        wait_for_port(proxy_port)
        out = subprocess.run(
            agent_cmd(agent_db, f"http://127.0.0.1:{proxy_port}/mcp", orders), capture_output=True, text=True, timeout=600
        )
    finally:
        for p in (proxy, shop):
            p.send_signal(signal.SIGTERM)
            p.wait(10)
    line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
    summary = json.loads(line[7:]) if line else {"status": "no result", "stderr": out.stderr[-3000:]}
    with open(proxy_log) as f:
        dropped = sum(1 for l in f if l.startswith("dropped"))
    return {
        "dropped_responses": dropped,
        "reconnects": summary.get("reconnects"),
        "verified_by_read_back": summary.get("verified", []),
        "retried": {k: v["attempts"] for k, v in summary.get("nodes", {}).items() if v["attempts"] > 1},
        "problems": audit(shop_db, orders, summary) + ([summary["stderr"]] if "stderr" in summary else []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", type=int, default=20)
    parser.add_argument("--drop", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as d:
        print(json.dumps(flaky(args.orders, args.drop, args.seed, d), indent=2))


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
