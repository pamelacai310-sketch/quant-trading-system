from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pandas as pd

from .indicators import compute_indicators, snapshot_indicators
from .models import StrategyRunResult


def _resolve_value(row: pd.Series, key: Any) -> float:
    if isinstance(key, (int, float)):
        return float(key)
    if isinstance(key, str):
        if key in row.index:
            return float(row[key])
        return float(key)
    raise ValueError(f"Unsupported rule operand: {key}")


def _evaluate_condition(frame: pd.DataFrame, index: int, condition: Dict[str, Any]) -> bool:
    row = frame.iloc[index]
    left = _resolve_value(row, condition["left"])
    right = _resolve_value(row, condition["right"])
    op = condition["op"]
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    if op == "==":
        return math.isclose(left, right, rel_tol=1e-9)
    if op == "crosses_above":
        if index == 0:
            return False
        prev = frame.iloc[index - 1]
        prev_left = _resolve_value(prev, condition["left"])
        prev_right = _resolve_value(prev, condition["right"])
        return prev_left <= prev_right and left > right
    if op == "crosses_below":
        if index == 0:
            return False
        prev = frame.iloc[index - 1]
        prev_left = _resolve_value(prev, condition["left"])
        prev_right = _resolve_value(prev, condition["right"])
        return prev_left >= prev_right and left < right
    raise ValueError(f"Unsupported operation: {op}")


def _all_conditions(frame: pd.DataFrame, index: int, conditions: List[Dict[str, Any]]) -> bool:
    return all(_evaluate_condition(frame, index, item) for item in conditions)


def _position_units(position_sizing: Dict[str, Any], capital: float, last_price: float) -> float:
    mode = position_sizing.get("mode", "fixed_units")
    max_units = float(position_sizing.get("max_units", 100))
    if mode == "fixed_units":
        return max_units
    if mode == "fixed_fraction":
        risk_fraction = float(position_sizing.get("risk_fraction", 0.1))
        units = capital * risk_fraction / max(last_price, 0.01)
        return min(units, max_units)
    raise ValueError(f"Unsupported sizing mode: {mode}")


def prepare_frame(frame: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    enriched = compute_indicators(frame, spec.get("indicators", []))
    return enriched.dropna().reset_index(drop=True)


def run_strategy_once(
    frame: pd.DataFrame,
    strategy_id: str,
    strategy_name: str,
    spec: Dict[str, Any],
    capital: float,
    current_position: float,
) -> StrategyRunResult:
    enriched = prepare_frame(frame, spec)
    if len(enriched) < 2:
        raise ValueError("Not enough data after indicator warmup")

    index = len(enriched) - 1
    long_entry = spec.get("entry_rules", [])
    long_exit = spec.get("exit_rules", [])
    short_entry = spec.get("short_entry_rules", [])
    short_exit = spec.get("short_exit_rules", [])
    direction = spec.get("direction", "long_only")
    position_signal = 0
    reason = "hold"

    if current_position > 0:
        position_signal = 1
        if long_exit and _all_conditions(enriched, index, long_exit):
            position_signal = 0
            reason = "exit_long"
    elif current_position < 0:
        position_signal = -1
        if short_exit and _all_conditions(enriched, index, short_exit):
            position_signal = 0
            reason = "exit_short"
    else:
        if long_entry and _all_conditions(enriched, index, long_entry):
            position_signal = 1
            reason = "enter_long"
        elif direction == "long_short" and short_entry and _all_conditions(enriched, index, short_entry):
            position_signal = -1
            reason = "enter_short"

    latest_price = float(enriched.iloc[-1]["close"])
    units = _position_units(spec.get("position_sizing", {}), capital, latest_price)
    target_quantity = position_signal * units
    delta = target_quantity - current_position
    side: Optional[str] = None
    if delta > 1e-9:
        side = "buy"
    elif delta < -1e-9:
        side = "sell"

    indicator_names = [item["name"] for item in spec.get("indicators", [])]
    indicators = snapshot_indicators(enriched, indicator_names + ["close"])

    return StrategyRunResult(
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        signal=position_signal,
        side=side,
        quantity=abs(float(delta)),
        last_price=latest_price,
        reason=reason,
        indicators=indicators,
    )
