"""A toolgraph agent for the real-world scenarios. Run as its own process so it can be killed.

    python scenarios/agent.py shop     --db run.db --url http://127.0.0.1:8811/mcp --orders 20
    python scenarios/agent.py files    --db run.db --root /tmp/project --server path/to/filesystem/index.js
    python scenarios/agent.py research --db run.db --url http://127.0.0.1:3001/mcp

Each run resumes from ``--db``; run it again after killing it. With
``TOOLGRAPH_KILL_AT=<point>:<n>`` the process sends itself SIGKILL right after
the n-th time it passes that commit point, which is a real process death at a precise
place. It prints ``RESULT <json>`` when the run is finished.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from mcp import Client, StdioServerParameters

from toolgraph import Node, PolicyRegistry, Scheduler, SideEffect, Store, ref
from toolgraph import store as st
from toolgraph.mcp_backend import MCPBackend, client_options, connect, policies_from_annotations, result_value

Verifier = Callable[[Scheduler, str], Awaitable[tuple[bool, Any]]]


def kill_hook() -> Callable[[str], None] | None:
    """SIGKILL this process at ``TOOLGRAPH_KILL_AT=<point>:<n>``; log points to ``TOOLGRAPH_TRACE``."""
    spec = os.environ.get("TOOLGRAPH_KILL_AT")
    trace = os.environ.get("TOOLGRAPH_TRACE")
    if not spec and not trace:
        return None
    point, _, n = (spec or ":0").rpartition(":")
    target, seen = int(n), [0]

    def hook(name: str) -> None:
        if trace:
            with open(trace, "a") as f:
                f.write(name + "\n")
        if name == point:
            seen[0] += 1
            if seen[0] == target:
                os.kill(os.getpid(), signal.SIGKILL)

    return hook


async def drive(sched: Scheduler, verifiers: dict[str, Verifier], answers: dict[str, Any]) -> dict[str, Any]:
    """Run until finished, answering questions and reading back unknown outcomes."""
    rounds = 0
    while True:
        report = await sched.run()
        if report.status != "blocked":
            break
        rounds += 1
        for nid in report.by_status(st.WAITING_INPUT):
            if not report.nodes[nid].has_answer:
                sched.answer(nid, answers[nid])
        for nid in report.by_status(st.NEEDS_VERIFICATION):
            verify = verifiers.get(report.nodes[nid].tool)
            if verify is None:
                raise SystemExit(f"no way to verify {nid}")
            applied, result = await verify(sched, nid)
            sched.resolve_verification(nid, applied=applied, result=result)
    events = sched.store.events(sched.run_id)
    return {
        "status": report.status,
        "nodes": {n: {"status": r.status, "attempts": r.attempts, "result": r.result} for n, r in report.nodes.items()},
        "verified": sorted({e["node"] for e in events if e["kind"].startswith("verified")}),
        "recovered": sorted({e["node"] for e in events if e["kind"].startswith("recovered")}),
        "resumes": sum(1 for e in events if e["kind"] == "run_resumed"),
        "human_rounds": rounds,
    }


# -- shop -------------------------------------------------------------------------


def shop_graph(orders: int) -> list[Node]:
    nodes = []
    for i in range(orders):
        o = f"o{i}"
        nodes += [
            Node(f"get_{o}", "shop.get_order", {"order": o}),
            Node(f"charge_{o}", "shop.charge", {"order": o, "amount": ref(f"get_{o}", "amount")}),
            Node(f"status_{o}", "shop.set_status", {"order": o, "value": "paid"}),
            Node(f"email_{o}", "shop.send_email", {"to": ref(f"get_{o}", "email"), "body": ref(f"charge_{o}", "charge_id")}),
        ]
    return nodes


def shop_policies() -> PolicyRegistry:
    common = {"timeout": 5.0, "max_attempts": 6, "backoff_base": 0.2, "backoff_max": 2.0}
    reg = PolicyRegistry()
    reg.register("shop.get_order", side_effect=SideEffect.READ, **common)
    reg.register("shop.email_sent", side_effect=SideEffect.READ, **common)
    reg.register("shop.charge", side_effect=SideEffect.KEYED_WRITE, writes=("order:{order}",), **common)
    reg.register("shop.set_status", side_effect=SideEffect.IDEMPOTENT_WRITE, writes=("order:{order}",), **common)
    reg.register("shop.send_email", side_effect=SideEffect.UNKEYED_WRITE, writes=("inbox:{to}",), **common)
    return reg


async def run_shop(args: argparse.Namespace) -> dict[str, Any]:
    # A reconnecting connection: one lost HTTP response must not take the agent down.
    async with connect({"shop": lambda: Client(args.url, **client_options())}) as backend:
        shop = backend.servers["shop"]

        async def read(tool: str, args: dict[str, Any]) -> Any:
            # Reads are safe to retry while the connection comes back.
            for attempt in range(10):
                try:
                    client = await shop.client()
                    return result_value(await client.call_tool(tool, args))
                except Exception:
                    await asyncio.sleep(0.2 * (attempt + 1))
            raise SystemExit(f"could not call {tool}")

        async def email_read_back(sched: Scheduler, nid: str) -> tuple[bool, Any]:
            return bool(await read("email_sent", sched.resolved_args(nid))), "sent"

        async def charge_read_back(sched: Scheduler, nid: str) -> tuple[bool, Any]:
            found = await read("find_charge", {"idempotency_key": sched.store.node(sched.run_id, nid).idem_key})
            return found["found"], {"charge_id": found["charge_id"], "amount": found["amount"]}

        store = Store(args.db, hook=kill_hook())
        sched = Scheduler(store, backend, shop_policies(), run_id="shop", max_concurrency=args.concurrency)
        sched.start(shop_graph(args.orders))
        summary = await drive(
            sched, {"shop.send_email": email_read_back, "shop.charge": charge_read_back}, {}
        )
        summary["reconnects"] = shop.connects - 1
        return summary


# -- files (a coding agent on the reference filesystem server) -----------------------


def file_text(value: Any) -> str:
    """read_text_file answers {"content": "<text>"}."""
    return value["content"] if isinstance(value, dict) else value


def files_graph(root: str) -> list[Node]:
    src = f"{root}/src"
    return [
        Node("mkdir", "fs.create_directory", {"path": src}),
        Node("write_app", "fs.write_file", {"path": f"{src}/app.py", "content": "retries = 1\nname = 'demo'\n"}),
        Node("write_test", "fs.write_file", {"path": f"{src}/test_app.py", "content": "from app import retries\n"}),
        # Not idempotent: applied twice, this would read "retries = 1 + 1 + 1".
        Node("bump", "fs.edit_file", {"path": f"{src}/app.py", "edits": [{"oldText": "retries = 1", "newText": "retries = 1 + 1"}]}),
        Node("read_app", "fs.read_text_file", {"path": f"{src}/app.py"}),
        Node("tree", "fs.directory_tree", {"path": root}),
        Node("search", "fs.search_files", {"path": root, "pattern": "**/*.py"}),
    ]


async def run_files(args: argparse.Namespace) -> dict[str, Any]:
    params = StdioServerParameters(command="node", args=[args.server, args.root])
    async with Client(params, **client_options()) as fs:
        reg = PolicyRegistry()
        # Resources: a file is "file:<path>"; directory-wide tools touch "file:<dir>*".
        common = {"timeout": 10.0, "max_attempts": 3, "backoff_base": 0.2}
        reg.register("fs.create_directory", side_effect=SideEffect.IDEMPOTENT_WRITE, writes=("file:{path}*",), **common)
        reg.register("fs.write_file", side_effect=SideEffect.IDEMPOTENT_WRITE, writes=("file:{path}",), **common)
        reg.register("fs.edit_file", side_effect=SideEffect.UNKEYED_WRITE, writes=("file:{path}",), **common)
        reg.register("fs.read_text_file", side_effect=SideEffect.READ, reads=("file:{path}",), **common)
        reg.register("fs.directory_tree", side_effect=SideEffect.READ, reads=("file:{path}*",), **common)
        reg.register("fs.search_files", side_effect=SideEffect.READ, reads=("file:{path}*",), **common)
        policies_from_annotations(await fs.list_tools(), prefix="fs.", registry=reg, **common)

        async def edit_read_back(sched: Scheduler, nid: str) -> tuple[bool, Any]:
            a = sched.resolved_args(nid)
            text = file_text(result_value(await fs.call_tool("read_text_file", {"path": a["path"]})))
            edit = a["edits"][0]
            return edit["newText"] in text, "verified by read-back"

        store = Store(args.db, hook=kill_hook())
        sched = Scheduler(store, MCPBackend({"fs": fs}), reg, run_id="files", max_concurrency=4)
        sched.start(files_graph(args.root))
        return await drive(sched, {"fs.edit_file": edit_read_back}, {})


# -- research (MCP Tasks on the reference "everything" server) ------------------------


async def run_research(args: argparse.Namespace) -> dict[str, Any]:
    async with Client(args.url, **client_options()) as ev:
        reg = PolicyRegistry()
        common = {"timeout": 10.0, "max_attempts": 3, "backoff_base": 0.2, "poll_interval": 0.5}
        # A research job only reads, whatever the server's conservative annotations say.
        reg.register("ev.simulate-research-query", side_effect=SideEffect.READ, job_timeout=120, **common)
        policies_from_annotations(await ev.list_tools(), prefix="ev.", registry=reg, **common)
        graph = [
            Node("sum", "ev.get-sum", {"a": 2, "b": 3}),
            Node("plain", "ev.simulate-research-query", {"topic": "durable execution"}),
            Node("ambiguous", "ev.simulate-research-query", {"topic": "python", "ambiguous": True}),
            Node("echo", "ev.echo", {"message": ref("ambiguous")}),
        ]
        store = Store(args.db, hook=kill_hook())
        sched = Scheduler(store, MCPBackend({"ev": ev}), reg, run_id="research", max_concurrency=4)
        sched.start(graph)
        summary = await drive(sched, {}, {"ambiguous": {"interpretation": "programming"}})
        summary["task_ids"] = {
            n: [e["data"]["task_id"] for e in store.events("research") if e["node"] == n and e["kind"] == "task_started"]
            for n in ("plain", "ambiguous")
        }
        return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("shop")
    p.add_argument("--db", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--orders", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=8)
    p = sub.add_parser("files")
    p.add_argument("--db", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--server", required=True)
    p = sub.add_parser("research")
    p.add_argument("--db", required=True)
    p.add_argument("--url", required=True)
    args = parser.parse_args()
    run = {"shop": run_shop, "files": run_files, "research": run_research}[args.cmd]
    summary = asyncio.run(run(args))
    print("RESULT " + json.dumps(summary, default=str), flush=True)
    sys.exit(0 if summary["status"] == "completed" else 2)


if __name__ == "__main__":
    main()
