from __future__ import annotations

import math
from dataclasses import asdict
from math import erfc, sqrt
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .models import BacktestResult
from .core.robustness import deflated_sharpe_ratio, evaluate_cpcv_returns
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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _resolve_adaptive_risk_limits(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve fixed risk limits plus optional MAE/MFE feedback overrides."""

    risk_limits = dict(spec.get("risk_limits", {}) or {})
    feedback = risk_limits.get("mae_mfe_feedback", {}) if isinstance(risk_limits.get("mae_mfe_feedback", {}), dict) else {}
    recommended = feedback.get("recommended_risk_limits", feedback) if isinstance(feedback, dict) else {}
    enabled = bool(feedback.get("enabled", recommended.get("enabled", False))) if isinstance(feedback, dict) else False
    sample_size = int(_as_float(feedback.get("sample_size", recommended.get("sample_size", 0)), 0)) if isinstance(feedback, dict) else 0
    use_feedback = enabled and sample_size >= int(_as_float(risk_limits.get("mae_mfe_min_samples", 2), 2))

    stop_loss = risk_limits.get("stop_loss_pct")
    take_profit = risk_limits.get("take_profit_pct")
    trailing_activation = risk_limits.get("trailing_activation_pct")
    trailing_stop = risk_limits.get("trailing_stop_pct")

    if use_feedback and isinstance(recommended, dict):
        stop_loss = recommended.get("stop_loss_pct", stop_loss)
        take_profit = recommended.get("take_profit_pct", take_profit)
        trailing_activation = recommended.get("trailing_activation_pct", trailing_activation)
        trailing_stop = recommended.get("trailing_stop_pct", trailing_stop)

    return {
        "stop_loss_pct": None if stop_loss is None else abs(_as_float(stop_loss)),
        "take_profit_pct": None if take_profit is None else abs(_as_float(take_profit)),
        "trailing_activation_pct": None if trailing_activation is None else abs(_as_float(trailing_activation)),
        "trailing_stop_pct": None if trailing_stop is None else abs(_as_float(trailing_stop)),
        "feedback_applied": bool(use_feedback),
        "feedback_sample_size": sample_size,
    }


def _new_open_trade(row: pd.Series, side: str, quantity: float, execution_price: float) -> Dict[str, Any]:
    return {
        "entry_date": row["timestamp"],
        "entry_price": float(execution_price),
        "side": side,
        "quantity": abs(float(quantity)),
        "high_since_entry": float(row.get("high", row["close"])),
        "low_since_entry": float(row.get("low", row["close"])),
        "trailing_active": False,
        "exit_reason": "",
    }


def _update_open_trade_path(open_trade: Optional[Dict[str, Any]], row: pd.Series) -> None:
    if not open_trade:
        return
    open_trade["high_since_entry"] = max(float(open_trade["high_since_entry"]), float(row.get("high", row["close"])))
    open_trade["low_since_entry"] = min(float(open_trade["low_since_entry"]), float(row.get("low", row["close"])))


def _trailing_exit_triggered(open_trade: Optional[Dict[str, Any]], row: pd.Series, limits: Dict[str, Any]) -> bool:
    if not open_trade:
        return False
    activation = limits.get("trailing_activation_pct")
    trail = limits.get("trailing_stop_pct")
    if activation is None or trail is None:
        return False
    entry = float(open_trade["entry_price"])
    close = float(row["close"])
    if open_trade["side"] == "long":
        favorable = float(open_trade["high_since_entry"]) / max(entry, 1e-9) - 1.0
        if favorable >= activation:
            open_trade["trailing_active"] = True
        return bool(open_trade["trailing_active"] and close / max(float(open_trade["high_since_entry"]), 1e-9) - 1.0 <= -trail)
    favorable = entry / max(float(open_trade["low_since_entry"]), 1e-9) - 1.0
    if favorable >= activation:
        open_trade["trailing_active"] = True
    return bool(open_trade["trailing_active"] and float(open_trade["low_since_entry"]) / max(close, 1e-9) - 1.0 <= -trail)


def _close_open_trade(
    open_trade: Optional[Dict[str, Any]],
    row: pd.Series,
    execution_price: float,
    exit_reason: str,
) -> Optional[Dict[str, Any]]:
    if not open_trade:
        return None
    entry = float(open_trade["entry_price"])
    side = str(open_trade["side"])
    if side == "long":
        mae_pct = (float(open_trade["low_since_entry"]) / max(entry, 1e-9) - 1.0) * 100.0
        mfe_pct = (float(open_trade["high_since_entry"]) / max(entry, 1e-9) - 1.0) * 100.0
        final_pnl_pct = (float(execution_price) / max(entry, 1e-9) - 1.0) * 100.0
    else:
        mae_pct = (entry / max(float(open_trade["high_since_entry"]), 1e-9) - 1.0) * 100.0
        mfe_pct = (entry / max(float(open_trade["low_since_entry"]), 1e-9) - 1.0) * 100.0
        final_pnl_pct = (entry / max(float(execution_price), 1e-9) - 1.0) * 100.0
    entry_date = pd.to_datetime(open_trade["entry_date"], errors="coerce")
    exit_date = pd.to_datetime(row["timestamp"], errors="coerce")
    holding_days = int(max((exit_date - entry_date).days, 0)) if not pd.isna(entry_date) and not pd.isna(exit_date) else 0
    return {
        "entry_date": str(open_trade["entry_date"]),
        "exit_date": str(row["timestamp"]),
        "entry_price": round(entry, 6),
        "exit_price": round(float(execution_price), 6),
        "quantity": round(float(open_trade["quantity"]), 6),
        "side": side,
        "final_pnl_pct": round(float(final_pnl_pct), 6),
        "mae_pct": round(float(mae_pct), 6),
        "mfe_pct": round(float(mfe_pct), 6),
        "holding_days": holding_days,
        "exit_reason": exit_reason,
        "trailing_was_active": bool(open_trade.get("trailing_active")),
    }


def _build_mae_mfe_feedback(
    closed_trades: List[Dict[str, Any]],
    risk_limits_used: Dict[str, Any],
) -> Dict[str, Any]:
    if len(closed_trades) < 2:
        return {
            "status": "insufficient_trades",
            "enabled": False,
            "sample_size": len(closed_trades),
            "reason": "Need at least two closed trades before adapting stop/take-profit parameters.",
        }

    mae = np.array([_as_float(item.get("mae_pct")) for item in closed_trades], dtype=float)
    mfe = np.array([_as_float(item.get("mfe_pct")) for item in closed_trades], dtype=float)
    pnl = np.array([_as_float(item.get("final_pnl_pct")) for item in closed_trades], dtype=float)
    winners = pnl > 0
    winner_mae_abs = np.abs(mae[winners]) if winners.any() else np.abs(mae)
    winner_mfe = mfe[winners] if winners.any() else mfe
    winner_pnl = pnl[winners] if winners.any() else np.array([], dtype=float)

    current_stop = risk_limits_used.get("stop_loss_pct")
    current_take = risk_limits_used.get("take_profit_pct")
    base_stop = 0.06 if current_stop is None else abs(float(current_stop))
    base_take = 0.10 if current_take is None else abs(float(current_take))

    mae_guard = float(np.percentile(winner_mae_abs, 85)) / 100.0 + 0.005 if len(winner_mae_abs) else base_stop
    recommended_stop = float(np.clip(mae_guard, 0.015, min(max(base_stop, 0.03), 0.12)))

    mfe_anchor = float(np.percentile(np.maximum(winner_mfe, 0.0), 65)) / 100.0 if len(winner_mfe) else base_take
    recommended_take = float(np.clip(max(mfe_anchor * 0.82, recommended_stop * 1.55), 0.025, 0.25))

    mfe_utilization = 0.0
    if len(winner_pnl):
        usable_mfe = np.maximum(winner_mfe, 1e-9)
        mfe_utilization = float(np.mean(np.clip(winner_pnl / usable_mfe, 0.0, 1.5)))
    missed_profit_ratio = float(np.mean((mfe >= max(5.0, recommended_take * 100.0)) & (pnl < recommended_take * 50.0)))
    trailing_activation = float(np.clip(max(recommended_take * 0.60, recommended_stop * 1.20), 0.025, 0.15))
    trailing_stop = float(np.clip(max(recommended_stop * 0.70, 0.012), 0.012, 0.08))

    return {
        "status": "ready",
        "enabled": True,
        "sample_size": len(closed_trades),
        "method": "mae_mfe_closed_trade_quantiles",
        "diagnostics": {
            "avg_mae_pct": round(float(np.mean(mae)), 6),
            "median_mae_pct": round(float(np.median(mae)), 6),
            "worst_mae_pct": round(float(np.min(mae)), 6),
            "avg_mfe_pct": round(float(np.mean(mfe)), 6),
            "median_mfe_pct": round(float(np.median(mfe)), 6),
            "best_mfe_pct": round(float(np.max(mfe)), 6),
            "mfe_utilization_rate": round(float(mfe_utilization), 6),
            "missed_profit_ratio": round(missed_profit_ratio, 6),
            "win_rate": round(float(np.mean(winners)), 6),
        },
        "recommended_risk_limits": {
            "enabled": True,
            "sample_size": len(closed_trades),
            "stop_loss_pct": round(recommended_stop, 6),
            "take_profit_pct": round(recommended_take, 6),
            "trailing_activation_pct": round(trailing_activation, 6),
            "trailing_stop_pct": round(trailing_stop, 6),
        },
        "rule": "Profitable-trade MAE tightens hard stop; MFE utilization sets take-profit and trailing stop.",
    }


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
    risk_limits_used = _resolve_adaptive_risk_limits(spec)
    stop_loss = risk_limits_used.get("stop_loss_pct")
    take_profit = risk_limits_used.get("take_profit_pct")
    open_trade: Optional[Dict[str, Any]] = None
    closed_trades: List[Dict[str, Any]] = []

    for idx in range(1, len(enriched)):
        row = enriched.iloc[idx]
        prev = enriched.iloc[idx - 1]
        price = float(row["close"])
        signal = 0
        exit_reason = ""
        long_enter = spec.get("entry_rules", [])
        long_exit = spec.get("exit_rules", [])
        short_enter = spec.get("short_entry_rules", [])
        short_exit = spec.get("short_exit_rules", [])
        _update_open_trade_path(open_trade, row)

        if current_qty > 0:
            signal = 1
            pnl_pct = price / entry_price - 1 if entry_price else 0
            if long_exit and all(_condition(enriched, idx, c) for c in long_exit):
                signal = 0
                exit_reason = "rule_exit"
            if stop_loss is not None and pnl_pct <= -abs(float(stop_loss)):
                signal = 0
                exit_reason = "stop_loss"
            if take_profit is not None and pnl_pct >= abs(float(take_profit)):
                signal = 0
                exit_reason = "take_profit"
            if _trailing_exit_triggered(open_trade, row, risk_limits_used):
                signal = 0
                exit_reason = "trailing_stop"
        elif current_qty < 0:
            signal = -1
            pnl_pct = entry_price / price - 1 if entry_price else 0
            if short_exit and all(_condition(enriched, idx, c) for c in short_exit):
                signal = 0
                exit_reason = "rule_exit"
            if stop_loss is not None and pnl_pct <= -abs(float(stop_loss)):
                signal = 0
                exit_reason = "stop_loss"
            if take_profit is not None and pnl_pct >= abs(float(take_profit)):
                signal = 0
                exit_reason = "take_profit"
            if _trailing_exit_triggered(open_trade, row, risk_limits_used):
                signal = 0
                exit_reason = "trailing_stop"
        else:
            if long_enter and all(_condition(enriched, idx, c) for c in long_enter):
                signal = 1
                exit_reason = "enter_long"
            elif direction == "long_short" and short_enter and all(_condition(enriched, idx, c) for c in short_enter):
                signal = -1
                exit_reason = "enter_short"

        target_qty = 0.0
        if signal != 0:
            target_qty = min(max_units, (cash + abs(current_qty) * price) * fixed_fraction / max(price, 0.01))
            target_qty *= signal

        delta = target_qty - current_qty
        if abs(delta) > 1e-9:
            execution_price = price * (1 + slippage_bps / 10_000 * np.sign(delta))
            fee = abs(delta) * execution_price * fee_bps / 10_000
            if current_qty != 0 and target_qty == 0:
                closed = _close_open_trade(open_trade, row, execution_price, exit_reason or "position_flattened")
                if closed:
                    closed_trades.append(closed)
                open_trade = None
            cash -= delta * execution_price + fee
            current_qty = target_qty
            if current_qty != 0:
                entry_price = execution_price
                if open_trade is None:
                    open_trade = _new_open_trade(
                        row,
                        side="long" if current_qty > 0 else "short",
                        quantity=current_qty,
                        execution_price=execution_price,
                    )
            trades.append(
                {
                    "timestamp": row["timestamp"],
                    "side": "buy" if delta > 0 else "sell",
                    "quantity": round(abs(delta), 4),
                    "price": round(float(execution_price), 4),
                    "fee": round(float(fee), 4),
                    "signal": int(signal),
                    "reason": exit_reason or ("rebalance" if target_qty else "exit"),
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

    if open_trade is not None and not enriched.empty:
        final_row = enriched.iloc[-1]
        closed = _close_open_trade(open_trade, final_row, float(final_row["close"]), "end_of_backtest_mark")
        if closed:
            closed_trades.append(closed)

    mae_mfe_feedback = _build_mae_mfe_feedback(closed_trades, risk_limits_used)
    closed_win_rate = (
        float(sum(1 for trade in closed_trades if float(trade.get("final_pnl_pct", 0.0)) > 0) / len(closed_trades))
        if closed_trades
        else 0.0
    )
    validation_spec = spec.get("validation", {}) if isinstance(spec.get("validation", {}), dict) else {}
    effective_trials = int(_as_float(validation_spec.get("effective_trials", validation_spec.get("trial_count", 1)), 1))
    cpcv_config = validation_spec.get("cpcv", {}) if isinstance(validation_spec.get("cpcv", {}), dict) else {}
    purge_window = cpcv_config.get("purge_window", validation_spec.get("purge_window", 5))
    robustness_validation = {
        "deflated_sharpe_ratio": deflated_sharpe_ratio(
            returns,
            effective_trials=effective_trials,
            periods_per_year=int(_as_float(validation_spec.get("periods_per_year", 252), 252)),
        ),
        "cpcv": evaluate_cpcv_returns(
            returns,
            n_groups=int(_as_float(cpcv_config.get("n_groups", 6), 6)),
            test_group_count=int(_as_float(cpcv_config.get("test_group_count", 2), 2)),
            purge_window=int(_as_float(purge_window, 5)),
            embargo_pct=float(_as_float(cpcv_config.get("embargo_pct", validation_spec.get("embargo_pct", 0.01)), 0.01)),
            max_paths=int(_as_float(cpcv_config.get("max_paths", 30), 30)),
            periods_per_year=int(_as_float(validation_spec.get("periods_per_year", 252), 252)),
        ),
        "gate": "CPCV and DSR are audit gates; complexity upgrades should not promote if either fails.",
    }

    result = BacktestResult(
        strategy_id=strategy_id,
        total_return=float(equity_series.iloc[-1] / starting_cash - 1) if not equity_series.empty else 0.0,
        annual_return=_annualized_return(equity_series) if not equity_series.empty else 0.0,
        sharpe=_sharpe(returns),
        sortino=_sortino(returns),
        max_drawdown=_max_drawdown(equity_series) if not equity_series.empty else 0.0,
        win_rate=closed_win_rate if closed_trades else (float(winning_trades / max(len(trades) // 2, 1)) if trades else 0.0),
        p_value=_p_value(returns),
        trade_count=len(trades),
        equity_curve=equity_curve[-180:],
        trades=trades[-50:],
        stats={
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "starting_cash": starting_cash,
            "ending_equity": float(equity_series.iloc[-1]) if not equity_series.empty else starting_cash,
            "risk_limits_used": {
                key: (round(float(value), 6) if isinstance(value, (int, float)) else value)
                for key, value in risk_limits_used.items()
            },
            "closed_trades": closed_trades[-100:],
            "mae_mfe_feedback": mae_mfe_feedback,
            "robustness_validation": robustness_validation,
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
