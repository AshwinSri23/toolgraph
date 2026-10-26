"""A small online shop exposed as an MCP server over Streamable HTTP.

It runs as its own process and keeps a ledger in SQLite of every side effect it
really applied, so tests can check for double charges and duplicate emails after
the agent process has been killed.

    python scenarios/shop_server.py --port 8811 --db /tmp/shop.db
"""

from __future__ import annotations

import argparse
import sqlite3

import anyio
from mcp.server.mcpserver import MCPServer

SCHEMA = """
CREATE TABLE IF NOT EXISTS charges (key TEXT PRIMARY KEY, order_id TEXT, amount INTEGER, charge_id TEXT);
CREATE TABLE IF NOT EXISTS emails (id INTEGER PRIMARY KEY AUTOINCREMENT, recipient TEXT, body TEXT);
CREATE TABLE IF NOT EXISTS status (order_id TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS applied (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, order_id TEXT, detail TEXT);
"""


def build(db_path: str, delay: float) -> MCPServer:
    db = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    db.executescript(SCHEMA)
    shop = MCPServer("shop")

    @shop.tool()
    async def get_order(order: str) -> dict:
        """Look up an order (read-only)."""
        await anyio.sleep(delay / 2)
        n = int(order.removeprefix("o"))
        return {"order": order, "amount": 1000 + 25 * n, "email": f"customer{n % 3}@example.com"}

    @shop.tool()
    async def charge(order: str, amount: int, idempotency_key: str) -> dict:
        """Charge a card. Deduplicated on idempotency_key, like a real payments API."""
        row = db.execute("SELECT charge_id FROM charges WHERE key = ?", (idempotency_key,)).fetchone()
        if row is None:
            charge_id = f"ch_{order}_{idempotency_key[-6:]}"
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO charges VALUES (?, ?, ?, ?)", (idempotency_key, order, amount, charge_id))
            db.execute("INSERT INTO applied (kind, order_id, detail) VALUES ('charge', ?, ?)", (order, str(amount)))
            db.execute("COMMIT")
        else:
            charge_id = row[0]
        # The money has moved; the response can still be lost after this point.
        await anyio.sleep(delay)
        return {"charge_id": charge_id, "amount": amount}

    @shop.tool()
    async def set_status(order: str, value: str) -> str:
        """Set an order's status (naturally idempotent)."""
        db.execute("INSERT INTO status VALUES (?, ?) ON CONFLICT(order_id) DO UPDATE SET value = excluded.value", (order, value))
        db.execute("INSERT INTO applied (kind, order_id, detail) VALUES ('status', ?, ?)", (order, value))
        await anyio.sleep(delay / 2)
        return value

    @shop.tool()
    async def send_email(to: str, body: str) -> str:
        """Send an email. No idempotency key: sending twice sends two emails."""
        db.execute("INSERT INTO emails (recipient, body) VALUES (?, ?)", (to, body))
        db.execute("INSERT INTO applied (kind, order_id, detail) VALUES ('email', ?, ?)", (body.split()[-1], to))
        await anyio.sleep(delay)
        return "sent"

    @shop.tool()
    def find_charge(idempotency_key: str) -> dict:
        """Look a charge up by its idempotency key, as payments APIs allow."""
        row = db.execute("SELECT charge_id, amount FROM charges WHERE key = ?", (idempotency_key,)).fetchone()
        return {"found": row is not None, "charge_id": row[0] if row else None, "amount": row[1] if row else None}

    @shop.tool()
    def email_sent(to: str, body: str) -> bool:
        """Read back whether an email was sent (used to verify unknown outcomes)."""
        return db.execute("SELECT 1 FROM emails WHERE recipient = ? AND body = ?", (to, body)).fetchone() is not None

    return shop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()
    build(args.db, args.delay).run("streamable-http", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
