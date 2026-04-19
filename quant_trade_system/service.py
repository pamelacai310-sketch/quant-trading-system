from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .backtest import backtest_strategy, serialize_backtest
from .broker import PaperBroker, WebhookBroker
from .causal_ai import CausalTradingSystemV4
from .demo_data import ensure_demo_data
from .models import OrderRequest
from .risk import RiskManager
from .storage import Storage
from .strategy_engine import run_strategy_once


class QuantTradingService:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.state_dir = self.base_dir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        ensure_demo_data(str(self.data_dir))
        self.storage = Storage(str(self.state_dir / "quant.db"))
        self.risk_manager = RiskManager()
        self.paper_broker = PaperBroker(self.storage)
        self.live_broker = WebhookBroker(self.storage)
        self._seed_default_strategies()
        self.causal_system = CausalTradingSystemV4(
            str(self.base_dir),
            initial_capital=float(self.storage.get_portfolio_state()["starting_cash"]),
        )

    def _seed_default_strategies(self) -> None:
        if self.storage.list_strategies():
            return
        samples = [
            {
                "name": "Gold Momentum Crossover",
                "dataset": "gold_daily",
                "status": "active",
                "spec": {
                    "symbol": "XAUUSD",
                    "direction": "long_only",
                    "indicators": [
                        {"name": "fast_ma", "type": "sma", "window": 10},
                        {"name": "slow_ma", "type": "sma", "window": 30},
                        {"name": "vol_20", "type": "volatility", "window": 20},
                    ],
                    "entry_rules": [
                        {"left": "fast_ma", "op": "crosses_above", "right": "slow_ma"},
                        {"left": "close", "op": ">", "right": "slow_ma"},
                    ],
                    "exit_rules": [
                        {"left": "fast_ma", "op": "crosses_below", "right": "slow_ma"},
                    ],
                    "position_sizing": {"mode": "fixed_fraction", "risk_fraction": 0.12, "max_units": 180},
                    "risk_limits": {
                        "max_order_notional": 60_000,
                        "max_position_per_symbol": 180,
                        "max_gross_exposure": 250_000,
                        "stop_loss_pct": 0.05,
                        "take_profit_pct": 0.14,
                    },
                },
            },
            {
                "name": "QQQ Mean Reversion",
                "dataset": "nasdaq_daily",
                "status": "draft",
                "spec": {
                    "symbol": "QQQ",
                    "direction": "long_short",
                    "indicators": [
                        {"name": "ema_8", "type": "ema", "window": 8},
                        {"name": "z_20", "type": "zscore", "window": 20},
                        {"name": "vol_sma_20", "type": "volume_sma", "window": 20},
                    ],
                    "entry_rules": [
                        {"left": "z_20", "op": "<", "right": -1.3},
                        {"left": "volume", "op": ">", "right": "vol_sma_20"},
                    ],
                    "exit_rules": [
                        {"left": "z_20", "op": ">", "right": -0.1},
                    ],
                    "short_entry_rules": [
                        {"left": "z_20", "op": ">", "right": 1.3},
                        {"left": "close", "op": ">", "right": "ema_8"},
                    ],
                    "short_exit_rules": [
                        {"left": "z_20", "op": "<", "right": 0.1},
                    ],
                    "position_sizing": {"mode": "fixed_fraction", "risk_fraction": 0.08, "max_units": 450},
                    "risk_limits": {
                        "max_order_notional": 80_000,
                        "max_position_per_symbol": 450,
                        "max_gross_exposure": 300_000,
                        "stop_loss_pct": 0.04,
                        "take_profit_pct": 0.08,
                    },
                },
            },
            {
                "name": "Copper Causal Demand Trend",
                "dataset": "copper_daily",
                "status": "active",
                "spec": {
                    "symbol": "LME_Copper",
                    "direction": "long_only",
                    "indicators": [
                        {"name": "fast_ema", "type": "ema", "window": 8},
                        {"name": "slow_ema", "type": "ema", "window": 21},
                        {"name": "mom_20", "type": "momentum", "window": 20},
                    ],
                    "entry_rules": [
                        {"left": "fast_ema", "op": "crosses_above", "right": "slow_ema"},
                        {"left": "mom_20", "op": ">", "right": 0.03},
                    ],
                    "exit_rules": [
                        {"left": "fast_ema", "op": "crosses_below", "right": "slow_ema"},
                    ],
                    "position_sizing": {"mode": "fixed_fraction", "risk_fraction": 0.09, "max_units": 1200},
                    "risk_limits": {
                        "max_order_notional": 45_000,
                        "max_position_per_symbol": 1200,
                        "max_gross_exposure": 280_000,
                        "stop_loss_pct": 0.05,
                        "take_profit_pct": 0.1,
                    },
                },
            },
        ]
        for item in samples:
            self.storage.upsert_strategy(None, item["name"], item["dataset"], item["spec"], item["status"])

    def list_datasets(self) -> List[Dict[str, Any]]:
        items = []
        for path in sorted(self.data_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            items.append(
                {
                    "name": path.stem,
                    "path": str(path),
                    "rows": int(len(frame)),
                    "start": str(frame["timestamp"].iloc[0]),
                    "end": str(frame["timestamp"].iloc[-1]),
                    "columns": list(frame.columns),
                }
            )
        return items

    def load_dataset(self, name: str) -> pd.DataFrame:
        path = self.data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Dataset {name} not found")
        frame = pd.read_csv(path)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset {name} missing columns: {sorted(missing)}")
        return frame

    def save_strategy(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "name" not in payload or "spec" not in payload or "dataset" not in payload:
            raise ValueError("Strategy payload must include name, dataset and spec")
        return self.storage.upsert_strategy(
            payload.get("id"),
            payload["name"],
            payload["dataset"],
            payload["spec"],
            payload.get("status", "draft"),
        )

    def list_strategies(self) -> List[Dict[str, Any]]:
        return self.storage.list_strategies()

    def backtest_strategy(self, strategy_id: str) -> Dict[str, Any]:
        strategy = self.storage.get_strategy(strategy_id)
        frame = self.load_dataset(strategy["dataset"])
        result = backtest_strategy(strategy["id"], strategy["name"], frame, strategy["spec"])
        payload = serialize_backtest(result)
        self.storage.save_backtest(strategy_id, payload)
        return payload

    def list_backtests(self, strategy_id: str | None = None) -> List[Dict[str, Any]]:
        return self.storage.list_backtests(strategy_id)

    def _latest_drawdown(self) -> float:
        snapshots = self.storage.list_portfolio_snapshots(limit=1)
        return float(snapshots[0]["drawdown"]) if snapshots else 0.0

    def _last_price(self, dataset_name: str, symbol: str) -> float:
        frame = self.load_dataset(dataset_name)
        return float(frame.iloc[-1]["close"])

    def _portfolio_snapshot(self) -> Dict[str, Any]:
        positions = self.storage.get_positions()
        cash_state = self.storage.get_portfolio_state()
        latest_prices = {}
        dataset_by_symbol = {
            item["spec"]["symbol"]: item["dataset"] for item in self.storage.list_strategies()
        }
        for position in positions:
            dataset_name = dataset_by_symbol.get(position["symbol"])
            if dataset_name:
                latest_prices[position["symbol"]] = self._last_price(dataset_name, position["symbol"])
            else:
                latest_prices[position["symbol"]] = float(position["avg_price"])
        unrealized = 0.0
        gross = 0.0
        net = 0.0
        realized = 0.0
        for position in positions:
            price = latest_prices[position["symbol"]]
            qty = float(position["quantity"])
            avg = float(position["avg_price"])
            gross += abs(qty * price)
            net += qty * price
            realized += float(position["realized_pnl"])
            unrealized += qty * (price - avg)
        equity = float(cash_state["cash"]) + net
        drawdown = 0.0
        history = self.storage.list_portfolio_snapshots(limit=200)
        highs = [item["equity"] for item in history] + [equity]
        peak = max(highs) if highs else equity
        if peak:
            drawdown = equity / peak - 1
        snapshot = {
            "equity": round(equity, 2),
            "cash": round(float(cash_state["cash"]), 2),
            "gross_exposure": round(gross, 2),
            "net_exposure": round(net, 2),
            "realized_pnl": round(realized, 2),
            "unrealized_pnl": round(unrealized, 2),
            "drawdown": round(drawdown, 6),
        }
        self.storage.add_portfolio_snapshot(snapshot)
        return snapshot

    def _sync_causal_account(self) -> Dict[str, Any]:
        snapshot = self._portfolio_snapshot()
        self.causal_system.sync_account_snapshot(snapshot)
        return snapshot

    def submit_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        order = OrderRequest(
            symbol=payload["symbol"],
            side=payload["side"],
            quantity=float(payload["quantity"]),
            order_type=payload.get("order_type", "market"),
            limit_price=payload.get("limit_price"),
            strategy_id=payload.get("strategy_id"),
            broker_mode=payload.get("broker_mode", "paper"),
            reason=payload.get("reason", "manual"),
            metadata=payload.get("metadata", {}),
        )
        dataset_name = payload.get("dataset")
        if not dataset_name and order.strategy_id:
            strategy = self.storage.get_strategy(order.strategy_id)
            dataset_name = strategy["dataset"]
        if not dataset_name:
            dataset_name = next(
                (
                    item["dataset"]
                    for item in self.storage.list_strategies()
                    if item["spec"]["symbol"] == order.symbol
                ),
                None,
            )
        if not dataset_name:
            raise ValueError("Please provide dataset for the order")
        last_price = self._last_price(dataset_name, order.symbol)
        strategy_limits = None
        if order.strategy_id:
            strategy_limits = self.storage.get_strategy(order.strategy_id)["spec"].get("risk_limits", {})
        portfolio = self.storage.get_portfolio_state()
        positions = self.storage.get_positions()
        risk = self.risk_manager.check_order(
            order,
            portfolio,
            positions,
            last_price,
            strategy_limits=strategy_limits,
            latest_drawdown=self._latest_drawdown(),
        )
        if not risk.passed:
            self.storage.add_risk_event(order.strategy_id, "pre_trade", "high", "; ".join(risk.violations), risk.context)
            return {"status": "rejected", "risk": asdict(risk)}
        broker = self.paper_broker if order.broker_mode == "paper" else self.live_broker
        result = broker.execute(order, last_price)
        snapshot = self._portfolio_snapshot()
        return {"status": "accepted", "order": result, "portfolio": snapshot, "risk": asdict(risk)}

    def execute_strategy(self, strategy_id: str, broker_mode: str = "paper") -> Dict[str, Any]:
        strategy = self.storage.get_strategy(strategy_id)
        frame = self.load_dataset(strategy["dataset"])
        positions = self.storage.get_positions()
        current = next((item for item in positions if item["symbol"] == strategy["spec"]["symbol"]), None)
        current_position = float(current["quantity"]) if current else 0.0
        capital = float(self.storage.get_portfolio_state()["cash"])
        run = run_strategy_once(frame, strategy_id, strategy["name"], strategy["spec"], capital, current_position)
        if run.side is None or run.quantity <= 0:
            snapshot = self._portfolio_snapshot()
            return {"status": "no_action", "run": asdict(run), "portfolio": snapshot}
        order_payload = {
            "strategy_id": strategy_id,
            "dataset": strategy["dataset"],
            "symbol": strategy["spec"]["symbol"],
            "side": run.side,
            "quantity": round(run.quantity, 4),
            "order_type": "market",
            "broker_mode": broker_mode,
            "reason": run.reason,
            "metadata": {"indicators": run.indicators},
        }
        execution = self.submit_order(order_payload)
        return {"status": execution["status"], "run": asdict(run), "execution": execution}

    def research_summary(self) -> Dict[str, Any]:
        backtests = self.storage.list_backtests()
        p_values = [float(item["report"]["p_value"]) for item in backtests if "p_value" in item["report"]]
        adjusted = self._bh_adjustment(p_values)
        return {
            "backtest_count": len(backtests),
            "latest_backtests": backtests[:10],
            "fdr_adjusted_p_values": adjusted,
        }

    def _bh_adjustment(self, p_values: List[float]) -> List[float]:
        if not p_values:
            return []
        indexed = sorted(enumerate(p_values), key=lambda item: item[1])
        count = len(indexed)
        adjusted = [1.0] * count
        running = 1.0
        for rank, (idx, p_value) in reversed(list(enumerate(indexed, start=1))):
            value = min(running, p_value * count / rank)
            adjusted[idx] = round(value, 6)
            running = value
        return adjusted

    def dashboard(self) -> Dict[str, Any]:
        snapshot = self._sync_causal_account()
        strategies = self.storage.list_strategies()
        orders = self.storage.list_orders(limit=20)
        risk_events = self.storage.list_risk_events(limit=20)
        positions = self.storage.get_positions()
        return {
            "portfolio": snapshot,
            "strategies": strategies,
            "orders": orders,
            "positions": positions,
            "risk_events": risk_events,
            "datasets": self.list_datasets(),
            "causal_status": self.causal_system.get_system_status(),
        }

    def causal_status(self) -> Dict[str, Any]:
        self._sync_causal_account()
        return self.causal_system.get_system_status()

    def _dataset_bundle(self) -> Dict[str, pd.DataFrame]:
        bundle = {}
        for item in self.list_datasets():
            bundle[item["name"]] = self.load_dataset(item["name"])
        return bundle

    def run_causal_pipeline(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        self._sync_causal_account()
        symbols = symbols or ["AAPL", "MSFT", "GOOGL", "600000"]
        return self.causal_system.full_analysis_pipeline_v4(symbols=symbols, raw_datasets=self._dataset_bundle())

    def ecosystem_status(self) -> Dict[str, Any]:
        self._sync_causal_account()
        return self.causal_system.ecosystem.status()

    def advanced_backtest(self, strategy_id: str) -> Dict[str, Any]:
        strategy = self.storage.get_strategy(strategy_id)
        frame = self.load_dataset(strategy["dataset"])
        return self.causal_system.ecosystem.run_backtrader_backtest(strategy, frame)

    def export_strategy(self, strategy_id: str, target: str) -> Dict[str, Any]:
        strategy = self.storage.get_strategy(strategy_id)
        return self.causal_system.ecosystem.export_strategy(strategy, target)

    def price_option(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.causal_system.ecosystem.price_option(payload)

    def dataset_series(self, dataset: str) -> Dict[str, Any]:
        frame = self.load_dataset(dataset)
        series = [
            {
                "time": str(row["timestamp"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }
            for _, row in frame.tail(240).iterrows()
        ]
        markers = []
        symbol = next(
            (item["spec"]["symbol"] for item in self.storage.list_strategies() if item["dataset"] == dataset),
            None,
        )
        if symbol:
            for order in self.storage.list_orders(limit=200):
                if order["symbol"] != symbol or order["filled_at"] is None:
                    continue
                timestamp = str(order["filled_at"])[:10]
                markers.append(
                    {
                        "time": timestamp,
                        "position": "belowBar" if order["side"] == "buy" else "aboveBar",
                        "color": "#6ed3a7" if order["side"] == "buy" else "#ff8f82",
                        "shape": "arrowUp" if order["side"] == "buy" else "arrowDown",
                        "text": f"{order['side']} {order['quantity']}",
                    }
                )
        return {"dataset": dataset, "series": series, "markers": markers}

    def get_market_regime_snapshot(self) -> Dict[str, Any]:
        self._sync_causal_account()
        return self.causal_system.build_market_data(self._dataset_bundle())

    def get_causal_decision(self, market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._sync_causal_account()
        if market_data is None:
            market_data = self.get_market_regime_snapshot()
        return self.causal_system.trading_agent.execute_decision(
            current_date=pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
            market_data=market_data,
        )

    def execute_causal_decision(self, broker_mode: str = "paper", market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._sync_causal_account()
        decision = self.get_causal_decision(market_data)
        positions = self.storage.get_positions()
        position_map = {item["symbol"]: item for item in positions}
        action_results: List[Dict[str, Any]] = []
        symbol_map = {
            "COMEX_Gold": ("XAUUSD", "gold_daily"),
            "LME_Copper": ("LME_Copper", "copper_daily"),
        }
        for action in decision.get("actions", []):
            action_name = action.get("action")
            target_symbol = action.get("symbol")
            if action_name == "HOLD":
                action_results.append({"status": "skipped", "reason": action.get("reason"), "action": action})
                continue
            if action_name == "FORCE_LIQUIDATE":
                liquidations = []
                for position in positions:
                    qty = abs(float(position["quantity"]))
                    if qty <= 0:
                        continue
                    liquidation = self.submit_order(
                        {
                            "symbol": position["symbol"],
                            "side": "sell" if float(position["quantity"]) > 0 else "buy",
                            "quantity": qty,
                            "dataset": next((item["dataset"] for item in self.storage.list_strategies() if item["spec"]["symbol"] == position["symbol"]), None),
                            "broker_mode": broker_mode,
                            "reason": "causal_force_liquidate",
                            "metadata": {"cause": action.get("reason")},
                        }
                    )
                    liquidations.append(liquidation)
                action_results.append({"status": "accepted", "action": action, "executions": liquidations})
                continue
            if action_name == "LONG" and target_symbol in symbol_map:
                mapped_symbol, dataset = symbol_map[target_symbol]
                existing_qty = abs(float(position_map.get(mapped_symbol, {}).get("quantity", 0.0)))
                quantity = 10 if mapped_symbol == "XAUUSD" else 100
                if existing_qty > 0:
                    action_results.append({"status": "skipped", "reason": "existing_position", "action": action})
                    continue
                execution = self.submit_order(
                    {
                        "symbol": mapped_symbol,
                        "dataset": dataset,
                        "side": "buy",
                        "quantity": quantity,
                        "broker_mode": broker_mode,
                        "reason": "causal_agent_signal",
                        "metadata": {"cause": action},
                    }
                )
                action_results.append({"status": "accepted", "action": action, "execution": execution})
                continue
            action_results.append({"status": "ignored", "reason": "unsupported_action", "action": action})
        return {
            "decision": decision,
            "results": action_results,
            "portfolio": self._sync_causal_account(),
        }
