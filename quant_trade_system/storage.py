from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Storage:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connect(self) -> Iterable[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS strategies (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS backtests (
                    id TEXT PRIMARY KEY,
                    strategy_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    strategy_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    order_type TEXT NOT NULL,
                    limit_price REAL,
                    status TEXT NOT NULL,
                    broker_mode TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    filled_at TEXT,
                    fill_price REAL,
                    reason TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS risk_events (
                    id TEXT PRIMARY KEY,
                    strategy_id TEXT,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS portfolio_state (
                    account_id TEXT PRIMARY KEY,
                    cash REAL NOT NULL,
                    starting_cash REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    timestamp TEXT PRIMARY KEY,
                    equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    gross_exposure REAL NOT NULL,
                    net_exposure REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    drawdown REAL NOT NULL
                );
                """
            )
            row = conn.execute(
                "SELECT account_id FROM portfolio_state WHERE account_id = ?",
                ("default",),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO portfolio_state (account_id, cash, starting_cash, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("default", 1_000_000.0, 1_000_000.0, utc_now()),
                )

    def upsert_strategy(self, strategy_id: Optional[str], name: str, dataset: str, spec: Dict[str, Any], status: str) -> Dict[str, Any]:
        strategy_id = strategy_id or str(uuid.uuid4())
        now = utc_now()
        with self.connect() as conn:
            exists = conn.execute("SELECT id FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
            if exists:
                conn.execute(
                    """
                    UPDATE strategies
                    SET name = ?, dataset = ?, spec_json = ?, status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (name, dataset, json.dumps(spec), status, now, strategy_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO strategies (id, name, status, dataset, spec_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (strategy_id, name, status, dataset, json.dumps(spec), now, now),
                )
        return self.get_strategy(strategy_id)

    def get_strategy(self, strategy_id: str) -> Dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
            if row is None:
                raise KeyError(f"Strategy {strategy_id} not found")
            item = dict(row)
            item["spec"] = json.loads(item.pop("spec_json"))
            return item

    def list_strategies(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM strategies ORDER BY updated_at DESC").fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["spec"] = json.loads(item.pop("spec_json"))
            items.append(item)
        return items

    def save_backtest(self, strategy_id: str, report: Dict[str, Any]) -> str:
        record_id = str(uuid.uuid4())
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO backtests (id, strategy_id, report_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (record_id, strategy_id, json.dumps(report), utc_now()),
            )
        return record_id

    def list_backtests(self, strategy_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM backtests"
        params: tuple[Any, ...] = ()
        if strategy_id:
            query += " WHERE strategy_id = ?"
            params = (strategy_id,)
        query += " ORDER BY created_at DESC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["report"] = json.loads(item.pop("report_json"))
            items.append(item)
        return items

    def add_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        order_id = payload.get("id") or str(uuid.uuid4())
        metadata = json.dumps(payload.get("metadata", {}))
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO orders (
                    id, strategy_id, symbol, side, quantity, order_type, limit_price,
                    status, broker_mode, requested_at, filled_at, fill_price, reason, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    payload.get("strategy_id"),
                    payload["symbol"],
                    payload["side"],
                    payload["quantity"],
                    payload["order_type"],
                    payload.get("limit_price"),
                    payload["status"],
                    payload["broker_mode"],
                    payload["requested_at"],
                    payload.get("filled_at"),
                    payload.get("fill_price"),
                    payload["reason"],
                    metadata,
                ),
            )
        return self.get_order(order_id)

    def update_order(self, order_id: str, **fields: Any) -> Dict[str, Any]:
        if not fields:
            return self.get_order(order_id)
        allowed = {
            "status",
            "filled_at",
            "fill_price",
            "limit_price",
            "metadata_json",
            "broker_mode",
            "reason",
        }
        updates = []
        values: List[Any] = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            updates.append(f"{key} = ?")
            values.append(value)
        values.append(order_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE orders SET {', '.join(updates)} WHERE id = ?", tuple(values))
        return self.get_order(order_id)

    def get_order(self, order_id: str) -> Dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
            if row is None:
                raise KeyError(f"Order {order_id} not found")
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json"))
        return item

    def list_orders(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM orders ORDER BY requested_at DESC LIMIT ?", (limit,)).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            items.append(item)
        return items

    def get_positions(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM positions ORDER BY symbol ASC").fetchall()
        return [dict(row) for row in rows]

    def upsert_position(self, symbol: str, quantity: float, avg_price: float, realized_pnl: float) -> None:
        with self.connect() as conn:
            existing = conn.execute("SELECT symbol FROM positions WHERE symbol = ?", (symbol,)).fetchone()
            if abs(quantity) < 1e-9:
                conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
                return
            if existing:
                conn.execute(
                    """
                    UPDATE positions
                    SET quantity = ?, avg_price = ?, realized_pnl = ?, updated_at = ?
                    WHERE symbol = ?
                    """,
                    (quantity, avg_price, realized_pnl, utc_now(), symbol),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO positions (symbol, quantity, avg_price, realized_pnl, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (symbol, quantity, avg_price, realized_pnl, utc_now()),
                )

    def get_portfolio_state(self) -> Dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM portfolio_state WHERE account_id = ?", ("default",)).fetchone()
            return dict(row)

    def update_cash(self, cash: float) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE portfolio_state SET cash = ?, updated_at = ? WHERE account_id = ?",
                (cash, utc_now(), "default"),
            )

    def add_risk_event(
        self,
        strategy_id: Optional[str],
        event_type: str,
        severity: str,
        message: str,
        metadata: Dict[str, Any],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO risk_events (id, strategy_id, event_type, severity, message, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), strategy_id, event_type, severity, message, json.dumps(metadata), utc_now()),
            )

    def list_risk_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM risk_events ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            items.append(item)
        return items

    def add_portfolio_snapshot(self, snapshot: Dict[str, float]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO portfolio_snapshots (
                    timestamp, equity, cash, gross_exposure, net_exposure, realized_pnl, unrealized_pnl, drawdown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    snapshot["equity"],
                    snapshot["cash"],
                    snapshot["gross_exposure"],
                    snapshot["net_exposure"],
                    snapshot["realized_pnl"],
                    snapshot["unrealized_pnl"],
                    snapshot["drawdown"],
                ),
            )

    def list_portfolio_snapshots(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
