"""
Portfolio State Persistence.

SCOPE HONESTY: uses SQLite (Python stdlib, zero extra infrastructure) so
this actually runs anywhere without requiring a database server to be
provisioned. For a real multi-user production deployment you'd point this
at Postgres/similar (the SQL here is intentionally vanilla enough to port
directly), and you'd add connection pooling, replication, and backups —
none of which a single-process library should assume about your
infrastructure. What's provided here is a correct, durable, crash-safe
(via SQLite's own WAL/transaction guarantees) persistence layer for
positions, NAV history, and order records.
"""
from __future__ import annotations
import sqlite3
import json
from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import dataclass


SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    portfolio_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    avg_cost REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (portfolio_id, symbol)
);

CREATE TABLE IF NOT EXISTS nav_history (
    portfolio_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    nav REAL NOT NULL,
    cash REAL NOT NULL,
    metadata TEXT,
    PRIMARY KEY (portfolio_id, timestamp)
);

CREATE TABLE IF NOT EXISTS orders (
    client_order_id TEXT PRIMARY KEY,
    portfolio_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    status TEXT NOT NULL,
    filled_quantity REAL NOT NULL DEFAULT 0,
    avg_fill_price REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class PortfolioStateStore:
    def __init__(self, db_path: str = "portfolio_state.db"):
        self.db_path = db_path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- positions --------------------------------------------------- #
    def upsert_position(self, portfolio_id: str, symbol: str, quantity: float, avg_cost: float):
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO positions (portfolio_id, symbol, quantity, avg_cost, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(portfolio_id, symbol) DO UPDATE SET "
                "quantity=excluded.quantity, avg_cost=excluded.avg_cost, updated_at=excluded.updated_at",
                (portfolio_id, symbol, quantity, avg_cost, now),
            )

    def get_positions(self, portfolio_id: str) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT symbol, quantity, avg_cost, updated_at FROM positions WHERE portfolio_id = ?",
                (portfolio_id,),
            ).fetchall()
        return {r["symbol"]: {"quantity": r["quantity"], "avg_cost": r["avg_cost"],
                               "updated_at": r["updated_at"]} for r in rows}

    # -- NAV history --------------------------------------------------- #
    def record_nav(self, portfolio_id: str, nav: float, cash: float, metadata: dict | None = None):
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO nav_history (portfolio_id, timestamp, nav, cash, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (portfolio_id, datetime.now(timezone.utc).isoformat(), nav, cash,
                 json.dumps(metadata or {})),
            )

    def get_nav_history(self, portfolio_id: str) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, nav, cash, metadata FROM nav_history "
                "WHERE portfolio_id = ? ORDER BY timestamp", (portfolio_id,),
            ).fetchall()
        return [{"timestamp": r["timestamp"], "nav": r["nav"], "cash": r["cash"],
                 "metadata": json.loads(r["metadata"])} for r in rows]

    # -- orders --------------------------------------------------------- #
    def save_order(self, portfolio_id: str, order) -> None:
        """Persist an `infra.oms.Order` object's current state."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO orders (client_order_id, portfolio_id, symbol, side, quantity, "
                "status, filled_quantity, avg_fill_price, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(client_order_id) DO UPDATE SET "
                "status=excluded.status, filled_quantity=excluded.filled_quantity, "
                "avg_fill_price=excluded.avg_fill_price, updated_at=excluded.updated_at",
                (order.client_order_id, portfolio_id, order.symbol, order.side, order.quantity,
                 order.status.value, order.filled_quantity, order.average_fill_price,
                 order.created_at.isoformat(), order.updated_at.isoformat()),
            )

    def get_orders(self, portfolio_id: str, status: str | None = None) -> list:
        query = "SELECT * FROM orders WHERE portfolio_id = ?"
        params = [portfolio_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def portfolio_snapshot(self, portfolio_id: str) -> dict:
        """Full point-in-time state: positions + latest NAV + open orders --
        what you'd load on restart after a crash to resume operations.
        """
        positions = self.get_positions(portfolio_id)
        nav_history = self.get_nav_history(portfolio_id)
        latest_nav = nav_history[-1] if nav_history else None
        open_orders = [o for o in self.get_orders(portfolio_id)
                       if o["status"] in ("new", "partially_filled", "pending_new")]
        return {"positions": positions, "latest_nav": latest_nav, "open_orders": open_orders}
