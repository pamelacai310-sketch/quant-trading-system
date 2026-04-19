from __future__ import annotations

import math
from dataclasses import asdict
from math import erfc, sqrt
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .models import BacktestResult
from .strategy_engine import prepare_frame


def _annualized_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = max(len(equity) / 252, 1 / 252)
    return (1 + total_return) ** (1 / years) - 1


def _sharpe(returns: pd.Series) -> float:
    std = returns.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(252) * returns.mean() / std)


def _sortino(returns: pd.Series) -> float:
    downside = returns[returns < 0]
    std = downside.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(252) * returns.mean() / std)


def _max_drawdown(equity: pd.Series) -> float:
    peaks = equity.cummax()
    drawdowns = equity / peaks - 1
    return float(drawdowns.min())


def _p_value(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) < 5:
        return 1.0
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 1.0
    t_stat = returns.mean() / (std / math.sqrt(len(returns)))
    return float(erfc(abs(t_stat) / sqrt(2)))


def backtest_strategy(
    strategy_id: str,
    strategy_name: str,
    frame: pd.DataFrame,
    spec: Dict[str, Any],
    starting_cash: float = 1_000_000.0,
    fee_bps: float = 3.0,
    slippage_bps: float = 5.0,
) -> BacktestResult:
    enriched = prepare_frame(frame, spec)
    if enriched.empty:
        raise ValueError("No data available after indicator warmup")

    direction = spec.get("direction", "long_only")
    current_qty = 0.0
    cash = starting_cash
    entry_price = 0.0
    trades: List[Dict[str, Any]] = []
    equity_curve: List[Dict[str, Any]] = []
    max_units = float(spec.get("position_sizing", {}).get("max_units", 100))
    fixed_fraction = float(spec.get("position_sizing", {}).get("risk_fraction", 0.15))
    stop_loss = spec.get("risk_limits", {}).get("stop_loss_pct")
    take_profit = spec.get("risk_limits", {}).get("take_profit_pct")

    for idx in range(1, len(enriched)):
        row = enriched.iloc[idx]
        prev = enriched.iloc[idx - 1]
        price = float(row["close"])
        signal = 0
        long_enter = spec.get("entry_rules", [])
        long_exit = spec.get("exit_rules", [])
        short_enter = spec.get("short_entry_rules", [])
        short_exit = spec.get("short_exit_rules", [])

        if current_qty > 0:
            signal = 1
            pnl_pct = price / entry_price - 1 if entry_price else 0
            if long_exit and all(_condition(enriched, idx, c) for c in long_exit):
                signal = 0
            if stop_loss is not None and pnl_pct <= -abs(float(stop_loss)):
                signal = 0
            if take_profit is not None and pnl_pct >= abs(float(take_profit)):
                signal = 0
        elif current_qty < 0:
            signal = -1
            pnl_pct = entry_price / price - 1 if entry_price else 0
            if short_exit and all(_condition(enriched, idx, c) for c in short_exit):
                signal = 0
            if stop_loss is not None and pnl_pct <= -abs(float(stop_loss)):
                signal = 0
            if take_profit is not None and pnl_pct >= abs(float(take_profit)):
                signal = 0
        else:
            if long_enter and all(_condition(enriched, idx, c) for c in long_enter):
                signal = 1
            elif direction == "long_short" and short_enter and all(_condition(enriched, idx, c) for c in short_enter):
                signal = -1

        target_qty = 0.0
        if signal != 0:
            target_qty = min(max_units, (cash + abs(current_qty) * price) * fixed_fraction / max(price, 0.01))
            target_qty *= signal

        delta = target_qty - current_qty
        if abs(delta) > 1e-9:
            execution_price = price * (1 + slippage_bps / 10_000 * np.sign(delta))
            fee = abs(delta) * execution_price * fee_bps / 10_000
            cash -= delta * execution_price + fee
            current_qty = target_qty
            if current_qty != 0:
                entry_price = execution_price
            trades.append(
                {
                    "timestamp": row["timestamp"],
                    "side": "buy" if delta > 0 else "sell",
                    "quantity": round(abs(delta), 4),
                    "price": round(float(execution_price), 4),
                    "fee": round(float(fee), 4),
                    "signal": int(signal),
                }
            )

        equity = cash + current_qty * price
        equity_curve.append(
            {
                "timestamp": row["timestamp"],
                "equity": round(float(equity), 2),
                "cash": round(float(cash), 2),
                "position_qty": round(float(current_qty), 4),
                "close": round(float(price), 4),
            }
        )

    equity_series = pd.Series([item["equity"] for item in equity_curve], dtype=float)
    returns = equity_series.pct_change().fillna(0.0)
    winning_trades = 0
    for i in range(1, len(trades)):
        prev_trade = trades[i - 1]
        trade = trades[i]
        if prev_trade["side"] == "buy" and trade["side"] == "sell" and trade["price"] > prev_trade["price"]:
            winning_trades += 1
        if prev_trade["side"] == "sell" and trade["side"] == "buy" and trade["price"] < prev_trade["price"]:
            winning_trades += 1

    result = BacktestResult(
        strategy_id=strategy_id,
        total_return=float(equity_series.iloc[-1] / starting_cash - 1) if not equity_series.empty else 0.0,
        annual_return=_annualized_return(equity_series) if not equity_series.empty else 0.0,
        sharpe=_sharpe(returns),
        sortino=_sortino(returns),
        max_drawdown=_max_drawdown(equity_series) if not equity_series.empty else 0.0,
        win_rate=float(winning_trades / max(len(trades) // 2, 1)) if trades else 0.0,
        p_value=_p_value(returns),
        trade_count=len(trades),
        equity_curve=equity_curve[-180:],
        trades=trades[-50:],
        stats={
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "starting_cash": starting_cash,
            "ending_equity": float(equity_series.iloc[-1]) if not equity_series.empty else starting_cash,
        },
    )
    return result


def _condition(frame: pd.DataFrame, index: int, condition: Dict[str, Any]) -> bool:
    from .strategy_engine import _evaluate_condition

    return _evaluate_condition(frame, index, condition)


def serialize_backtest(result: BacktestResult) -> Dict[str, Any]:
    payload = asdict(result)
    for key in ("total_return", "annual_return", "sharpe", "sortino", "max_drawdown", "win_rate", "p_value"):
        payload[key] = round(float(payload[key]), 6)
    return payload
