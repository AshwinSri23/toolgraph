"""The travel example from the README, against real (in-process) MCP servers.

    search flights ─┐
                    ├─► fetch details ─► book flight (write) ───┐
    search hotels ──┴───────────────────────────────────────────┴─► send email (write)

Run it twice with --crash to see recovery: the first run dies right after the
booking request is sent, the second resumes from the database and re-sends the
booking with the same idempotency key, so the airline books exactly once.

    python examples/travel.py --crash
"""

from __future__ import annotations

import argparse
import asyncio
import os
import tempfile

from mcp import Client
from mcp.server.mcpserver import MCPServer

from toolgraph import Node, PolicyRegistry, Scheduler, SideEffect, Store, ref
from toolgraph.mcp_backend import MCPBackend

travel = MCPServer("travel")
mail = MCPServer("mail")
bookings: dict[str, str] = {}
outbox: list[str] = []


@travel.tool()
async def search_flights(dest: str) -> dict:
    await asyncio.sleep(0.5)
    return {"flight": "TP1234", "dest": dest}


@travel.tool()
async def search_hotels(dest: str) -> dict:
    await asyncio.sleep(0.5)
    return {"hotel": "Casa Azul", "dest": dest}


@travel.tool()
def fetch_details(flight: str, hotel: str) -> dict:
    return {"flight": flight, "hotel": hotel, "price": 412}


@travel.tool()
def book_flight(flight: str, idempotency_key: str) -> dict:
    # A real booking API deduplicates on the key; so does this one.
    bookings.setdefault(idempotency_key, flight)
    return {"confirmation": "PNR-" + idempotency_key[-5:].upper()}


@mail.tool()
def send_email(to: str, body: str) -> str:
    outbox.append(body)
    return "sent"


def policies() -> PolicyRegistry:
    reg = PolicyRegistry()
    reg.register("travel.search_*", side_effect=SideEffect.READ)
    reg.register("travel.fetch_details", side_effect=SideEffect.READ)
    reg.register("travel.book_flight", side_effect=SideEffect.KEYED_WRITE, writes=("trip:{flight}",))
    reg.register("mail.send_email", side_effect=SideEffect.UNKEYED_WRITE, writes=("inbox:{to}",))
    return reg


GRAPH = [
    Node("flights", "travel.search_flights", {"dest": "Lisbon"}),
    Node("hotels", "travel.search_hotels", {"dest": "Lisbon"}),
    Node("details", "travel.fetch_details", {"flight": ref("flights", "flight"), "hotel": ref("hotels", "hotel")}),
    Node("book", "travel.book_flight", {"flight": ref("details", "flight")}),
    Node(
        "email",
        "mail.send_email",
        {"to": "me@example.com", "body": ref("book", "confirmation"), "hotel": ref("hotels", "hotel")},
    ),
]


class Crash(BaseException):
    pass


async def run(db: str, crash: bool) -> None:
    def hook(point: str) -> None:
        if crash and point == "received" and bookings:
            raise Crash(point)

    async with Client(travel) as t, Client(mail) as m:
        store = Store(db, hook=hook)
        sched = Scheduler(store, MCPBackend({"travel": t, "mail": m}), policies(), run_id="trip-1")
        if sched.start(GRAPH):
            print("started run trip-1")
        else:
            print("resuming run trip-1 from", db)
        try:
            report = await sched.run()
        except Crash:
            print("  ** process died right after the booking request was sent **")
            return
        finally:
            store.close()
    print("status:", report.status)
    for nid, row in report.nodes.items():
        print(f"  {nid:8} {row.status:10} attempts={row.attempts} result={row.result}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crash", action="store_true", help="die after sending the booking, then resume")
    args = parser.parse_args()
    db = os.path.join(tempfile.mkdtemp(), "travel.db")
    asyncio.run(run(db, crash=args.crash))
    if args.crash:
        asyncio.run(run(db, crash=False))
    print("bookings made:", len(bookings), "| emails sent:", len(outbox))


if __name__ == "__main__":
    main()
