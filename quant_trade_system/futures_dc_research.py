from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .futures_specs import get_futures_contract_spec, normalize_futures_symbol
from .path_capture import DCEvent, capture_ratio, dc_path_summary, directional_change_events


DEFAULT_PRODUCTS = ("IF", "IC", "IM", "CU", "AU", "AG", "RB", "HC", "I", "M")
DEFAULT_THETA_BPS = (8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)
DEFAULT_STRATEGY_FAMILIES = ("dc_continuation", "dc_reversal", "dc_overshoot_continuation", "dc_overshoot_reversal")
DEFAULT_VOL_FILTERS = ("all", "mid_40_80", "high_70_plus")
DEFAULT_OPEN_INTEREST_FILTERS = ("all", "rising")
DEFAULT_TIME_FILTERS = ("all", "day", "night", "open30", "close30")
DEFAULT_EVENT_SPACING_BARS = (0, 2, 4, 6)
DEFAULT_OVERSHOOT_TRIGGER_MULTIPLES = (0.5, 1.0)
DEFAULT_HISTORY_MAX_SCANS = 60
HISTORY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FuturesCostModel:
    """Cost model in basis points.

    Commission and slippage are per side. Market impact is treated as a
    round-trip buffer because the realized footprint is harder to assign to one
    leg in this research layer.
    """

    commission_bps: float = 0.8
    slippage_bps: float = 1.5
    impact_bps: float = 1.7

    @property
    def round_trip_cost_bps(self) -> float:
        return float(2.0 * (self.commission_bps + self.slippage_bps) + self.impact_bps)


@dataclass(frozen=True)
class DCFuturesStrategySpec:
    symbol: str
    family: str
    theta_bps: float
    vol_filter: str = "all"
    open_interest_filter: str = "all"
    time_filter: str = "all"
    event_spacing_bars: int = 0
    max_hold_bars: int = 12
    stop_multiple: float = 1.0
    take_profit_multiple: float = 2.0
    overshoot_trigger_multiple: float = 0.0


def normalize_minute_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "hold"])

    rename = {column: str(column).strip().lower() for column in frame.columns}
    working = frame.rename(columns=rename).copy()
    if "datetime" in working.columns and "timestamp" not in working.columns:
        working["timestamp"] = working["datetime"]
    elif "date" in working.columns and "timestamp" not in working.columns:
        working["timestamp"] = working["date"]
    elif "timestamp" not in working.columns:
        working["timestamp"] = working.index

    if "close" not in working.columns:
        raise ValueError("minute frame must include a close column")

    for column in ("open", "high", "low", "close", "volume", "hold"):
        if column not in working.columns:
            working[column] = working["close"] if column in {"open", "high", "low"} else 0.0
        working[column] = pd.to_numeric(working[column], errors="coerce")

    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.dropna(subset=["timestamp", "close"]).copy()
    working = working[working["close"] > 0].copy()
    working["open"] = working["open"].fillna(working["close"])
    working["high"] = working["high"].fillna(working[["open", "close"]].max(axis=1))
    working["low"] = working["low"].fillna(working[["open", "close"]].min(axis=1))
    working["high"] = working[["high", "open", "close"]].max(axis=1)
    working["low"] = working[["low", "open", "close"]].min(axis=1)
    working["volume"] = working["volume"].fillna(0.0)
    working["hold"] = working["hold"].fillna(0.0)
    working = working.sort_values("timestamp").reset_index(drop=True)
    working["_abs_return"] = working["close"].pct_change().abs().fillna(0.0)
    working["_vol_score"] = working["_abs_return"].rolling(12, min_periods=3).mean().fillna(0.0)
    working["_vol_percentile"] = _rolling_percentile(working["_vol_score"].to_numpy(dtype=float), window=96)
    working["_hold_diff"] = working["hold"].diff().fillna(0.0)
    return working[[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "hold",
        "_abs_return",
        "_vol_score",
        "_vol_percentile",
        "_hold_diff",
    ]]


def simulate_dc_strategy(
    frame: pd.DataFrame,
    spec: DCFuturesStrategySpec,
    cost_model: Optional[FuturesCostModel] = None,
) -> Dict[str, Any]:
    cost_model = cost_model or FuturesCostModel()
    working = normalize_minute_frame(frame)
    if len(working) < 3:
        return _empty_simulation_result(spec, cost_model, "insufficient_bars")

    events = directional_change_events(working, spec.theta_bps)
    overshoot_family = _is_overshoot_family(spec.family)
    entry_events: Dict[int, DCEvent] = {}
    pending_events: Dict[int, DCEvent] = {}
    for event in events:
        entry_index = event.confirmation_index + 1
        if entry_index >= len(working):
            continue
        if overshoot_family:
            pending_events[entry_index] = event
        else:
            entry_events[entry_index] = event
    theta = float(spec.theta_bps) / 10_000.0
    position = 0
    entry_price = 0.0
    entry_index = -1
    entry_event: Optional[DCEvent] = None
    active_event: Optional[DCEvent] = None
    last_entry_index = -10**9
    equity = 1.0
    trades: List[Dict[str, Any]] = []

    for index in range(1, len(working)):
        row = working.iloc[index]
        if position != 0:
            exit_price, exit_reason = _exit_for_bar(working, index, position, entry_price, entry_index, spec, theta)
            if exit_price is not None:
                equity = _close_trade(
                    trades,
                    equity,
                    spec,
                    cost_model,
                    position,
                    entry_price,
                    entry_index,
                    index,
                    float(exit_price),
                    str(exit_reason),
                    entry_event,
                    working,
                )
                position = 0
                entry_price = 0.0
                entry_index = -1
                entry_event = None

        if overshoot_family:
            if index in pending_events:
                active_event = pending_events[index]
            if active_event is not None and index - active_event.confirmation_index > int(spec.max_hold_bars):
                active_event = None
            event = (
                active_event
                if active_event is not None and _overshoot_entry_ready(working, index, active_event, spec, theta)
                else None
            )
        else:
            event = entry_events.get(index)
        if event is None:
            continue

        intended_position = _position_for_event(event, spec.family)
        if intended_position == 0:
            continue

        if position != 0 and intended_position != position:
            equity = _close_trade(
                trades,
                equity,
                spec,
                cost_model,
                position,
                entry_price,
                entry_index,
                index,
                float(row["open"]),
                "opposite_dc",
                entry_event,
                working,
            )
            position = 0
            entry_price = 0.0
            entry_index = -1
            entry_event = None

        if position != 0:
            continue
        if index - last_entry_index < int(spec.event_spacing_bars):
            continue
        if not _entry_allowed(working, index, spec):
            continue

        position = intended_position
        entry_price = float(row["open"])
        entry_index = index
        entry_event = event
        last_entry_index = index
        if overshoot_family:
            active_event = None

    if position != 0:
        final_index = len(working) - 1
        equity = _close_trade(
            trades,
            equity,
            spec,
            cost_model,
            position,
            entry_price,
            entry_index,
            final_index,
            float(working.iloc[final_index]["close"]),
            "end_of_frame",
            entry_event,
            working,
        )

    strategy_return = float(equity - 1.0)
    path = dc_path_summary(
        working,
        spec.theta_bps,
        round_trip_cost_bps=cost_model.round_trip_cost_bps,
        include_open_segment=False,
        strategy_return=strategy_return,
    )
    return {
        "spec": asdict(spec),
        "bar_count": int(len(working)),
        "event_count": int(path["event_count"]),
        "segment_count": int(path["segment_count"]),
        "trade_count": len(trades),
        "strategy_return": strategy_return,
        "dc_path_return": float(path["dc_path_return"]),
        "gross_dc_path_return": float(path["gross_dc_path_return"]),
        "capture_ratio": capture_ratio(strategy_return, float(path["dc_path_return"])),
        "win_rate": float(sum(1 for trade in trades if trade["net_return"] > 0) / len(trades)) if trades else 0.0,
        "trades": trades,
        "lookahead_rule": "DC event is actionable only at the next bar open after confirmation.",
        "cost_model": asdict(cost_model) | {"round_trip_cost_bps": cost_model.round_trip_cost_bps},
    }


def walk_forward_evaluate(
    frame: pd.DataFrame,
    spec: DCFuturesStrategySpec,
    cost_model: Optional[FuturesCostModel] = None,
    folds: int = 3,
    random_trials: int = 128,
) -> Dict[str, Any]:
    cost_model = cost_model or FuturesCostModel()
    working = normalize_minute_frame(frame)
    fold_rows: List[Dict[str, Any]] = []
    test_trade_sets: List[List[Dict[str, Any]]] = []
    for fold_index, train_start, train_end, test_end in _walk_forward_bounds(len(working), folds=folds):
        train_frame = working.iloc[train_start:train_end].copy()
        test_frame = working.iloc[train_end:test_end].copy()
        train = simulate_dc_strategy(train_frame, spec, cost_model=cost_model)
        test = simulate_dc_strategy(test_frame, spec, cost_model=cost_model)
        test_trade_sets.append(list(test.get("trades", [])))
        fold_rows.append(
            {
                "fold": fold_index,
                "train_start": str(train_frame.iloc[0]["timestamp"]) if not train_frame.empty else "",
                "train_end": str(train_frame.iloc[-1]["timestamp"]) if not train_frame.empty else "",
                "test_start": str(test_frame.iloc[0]["timestamp"]) if not test_frame.empty else "",
                "test_end": str(test_frame.iloc[-1]["timestamp"]) if not test_frame.empty else "",
                "train_return": train["strategy_return"],
                "train_capture_ratio": train["capture_ratio"],
                "train_events": train["event_count"],
                "train_trades": train["trade_count"],
                "test_return": test["strategy_return"],
                "test_capture_ratio": test["capture_ratio"],
                "test_events": test["event_count"],
                "test_trades": test["trade_count"],
            }
        )
    cost_sensitivity = cost_sensitivity_audit(test_trade_sets, cost_model)
    random_control = random_direction_control(
        test_trade_sets,
        cost_model,
        trials=random_trials,
        seed=_stable_seed(asdict(spec)),
    )
    return classify_walk_forward_result(
        spec,
        cost_model,
        fold_rows,
        cost_sensitivity=cost_sensitivity,
        random_control=random_control,
    )


def classify_walk_forward_result(
    spec: DCFuturesStrategySpec,
    cost_model: FuturesCostModel,
    folds: Sequence[Mapping[str, Any]],
    cost_sensitivity: Optional[Mapping[str, Any]] = None,
    random_control: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    test_returns = [_as_float(row.get("test_return")) for row in folds]
    train_returns = [_as_float(row.get("train_return")) for row in folds]
    captures = [_as_float(row.get("test_capture_ratio")) for row in folds]
    test_events = [int(_as_float(row.get("test_events"))) for row in folds]
    test_trades = [int(_as_float(row.get("test_trades"))) for row in folds]
    positive_folds = sum(1 for value in test_returns if value > 0)
    target_capture_folds = sum(1 for value in captures if 0.05 <= value <= 0.20)
    passing_fold_events = [
        int(_as_float(row.get("test_events")))
        for row in folds
        if _as_float(row.get("test_return")) > 0 and 0.05 <= _as_float(row.get("test_capture_ratio")) <= 0.20
    ]
    avg_train_return = float(np.mean(train_returns)) if train_returns else 0.0
    avg_test_return = float(np.mean(test_returns)) if test_returns else 0.0
    avg_test_capture = float(np.mean(captures)) if captures else 0.0
    median_test_capture = float(np.median(captures)) if captures else 0.0
    min_passing_fold_events = min(passing_fold_events) if passing_fold_events else 0
    total_test_trades = int(sum(test_trades))

    flags: List[str] = []
    if float(spec.theta_bps) <= cost_model.round_trip_cost_bps:
        flags.append("THETA_BELOW_ROUND_TRIP_COST")
    if avg_test_return <= 0:
        flags.append("NON_POSITIVE_EXPECTANCY")
    if not 0.05 <= avg_test_capture <= 0.20:
        flags.append("CAPTURE_OUT_OF_TARGET")
    if target_capture_folds < 2:
        flags.append("FOLD_CAPTURE_INSTABILITY")
    if not 0.05 <= median_test_capture <= 0.20:
        flags.append("MEDIAN_CAPTURE_OUT_OF_TARGET")
    if min(test_events or [0]) < 20:
        flags.append("LOW_EVENT_COUNT")
    if total_test_trades < 30:
        flags.append("LOW_TRADE_COUNT")
    if avg_train_return <= 0 < avg_test_return:
        flags.append("TRAIN_TEST_DIVERGENCE")
    stress_1_5x = cost_sensitivity.get("cost_1_5x", {}) if isinstance(cost_sensitivity, Mapping) else {}
    if (
        isinstance(cost_sensitivity, Mapping)
        and isinstance(stress_1_5x, Mapping)
        and _as_float(stress_1_5x.get("avg_return")) <= 0
    ):
        flags.append("COST_STRESS_FAIL")
    if isinstance(random_control, Mapping) and not bool(random_control.get("beats_p95", False)):
        flags.append("RANDOM_CONTROL_NOT_BEATEN")
    if positive_folds >= 2 and (
        avg_test_return <= 0
        or not 0.05 <= avg_test_capture <= 0.20
        or target_capture_folds < 2
        or not 0.05 <= median_test_capture <= 0.20
    ):
        flags.append("UNSTABLE_FOLD_EDGE")

    strict_pass = (
        positive_folds >= 2
        and avg_test_return > 0
        and 0.05 <= avg_test_capture <= 0.20
        and target_capture_folds >= 2
        and 0.05 <= median_test_capture <= 0.20
        and min_passing_fold_events >= 20
        and total_test_trades >= 30
        and not flags
    )
    if strict_pass:
        status = "PASS"
    elif avg_test_return > 0 and positive_folds >= 2:
        status = "WATCH"
    else:
        status = "FAIL"

    return {
        "status": status,
        "failure_reasons": flags,
        "spec": asdict(spec),
        "folds": [dict(row) for row in folds],
        "summary": {
            "avg_train_return": avg_train_return,
            "avg_test_return": avg_test_return,
            "avg_test_capture_ratio": avg_test_capture,
            "median_test_capture_ratio": median_test_capture,
            "positive_test_folds": positive_folds,
            "target_capture_folds": target_capture_folds,
            "min_test_events": min(test_events or [0]),
            "min_passing_fold_events": min_passing_fold_events,
            "total_test_trades": total_test_trades,
            "cost_1_5x_avg_return": _as_float(stress_1_5x.get("avg_return")) if isinstance(stress_1_5x, Mapping) else 0.0,
            "random_control_p_value": _as_float(random_control.get("p_value")) if isinstance(random_control, Mapping) else 1.0,
        },
        "cost_sensitivity": dict(cost_sensitivity or {}),
        "random_direction_control": dict(random_control or {}),
        "cost_model": asdict(cost_model) | {"round_trip_cost_bps": cost_model.round_trip_cost_bps},
        "gate": "PASS requires positive OOS expectancy, stable 5%-20% capture, cost-stress survival, random-control edge, sufficient events/trades, and no bias flags.",
    }


def cost_sensitivity_audit(
    trade_sets: Sequence[Sequence[Mapping[str, Any]]],
    cost_model: FuturesCostModel,
    multipliers: Sequence[float] = (1.0, 1.5, 2.0),
) -> Dict[str, Any]:
    audit: Dict[str, Any] = {}
    for multiplier in multipliers:
        round_trip_cost_bps = cost_model.round_trip_cost_bps * float(multiplier)
        fold_returns = [
            _compound_trades_with_cost(trades, round_trip_cost_bps, randomize_direction=False)
            for trades in trade_sets
        ]
        key = f"cost_{str(multiplier).replace('.', '_')}x"
        audit[key] = {
            "round_trip_cost_bps": float(round_trip_cost_bps),
            "avg_return": float(np.mean(fold_returns)) if fold_returns else 0.0,
            "min_return": float(np.min(fold_returns)) if fold_returns else 0.0,
            "positive_folds": int(sum(1 for value in fold_returns if value > 0)),
            "fold_returns": [float(value) for value in fold_returns],
        }
    return audit


def random_direction_control(
    trade_sets: Sequence[Sequence[Mapping[str, Any]]],
    cost_model: FuturesCostModel,
    trials: int = 128,
    seed: int = 0,
) -> Dict[str, Any]:
    actual_fold_returns = [
        _compound_trades_with_cost(trades, cost_model.round_trip_cost_bps, randomize_direction=False)
        for trades in trade_sets
    ]
    actual_avg_return = float(np.mean(actual_fold_returns)) if actual_fold_returns else 0.0
    effective_trials = max(int(trials), 1)
    rng = np.random.default_rng(seed)
    null_returns: List[float] = []
    for _ in range(effective_trials):
        fold_returns = [
            _compound_trades_with_cost(
                trades,
                cost_model.round_trip_cost_bps,
                randomize_direction=True,
                rng=rng,
            )
            for trades in trade_sets
        ]
        null_returns.append(float(np.mean(fold_returns)) if fold_returns else 0.0)
    null_array = np.asarray(null_returns, dtype=float)
    p95 = float(np.percentile(null_array, 95)) if len(null_array) else 0.0
    return {
        "actual_avg_return": actual_avg_return,
        "null_mean_return": float(np.mean(null_array)) if len(null_array) else 0.0,
        "null_p95_return": p95,
        "p_value": float((np.sum(null_array >= actual_avg_return) + 1) / (len(null_array) + 1)) if len(null_array) else 1.0,
        "beats_p95": bool(actual_avg_return > p95),
        "trials": effective_trials,
        "method": "same_entry_exit_random_long_short_direction",
    }


def scan_futures_dc_strategies(
    frames_by_symbol: Mapping[str, pd.DataFrame],
    theta_bps_values: Sequence[float] = DEFAULT_THETA_BPS,
    strategy_families: Sequence[str] = DEFAULT_STRATEGY_FAMILIES,
    vol_filters: Sequence[str] = DEFAULT_VOL_FILTERS,
    open_interest_filters: Sequence[str] = DEFAULT_OPEN_INTEREST_FILTERS,
    time_filters: Sequence[str] = DEFAULT_TIME_FILTERS,
    event_spacing_bars_values: Sequence[int] = DEFAULT_EVENT_SPACING_BARS,
    overshoot_trigger_multiples: Sequence[float] = DEFAULT_OVERSHOOT_TRIGGER_MULTIPLES,
    cost_model: Optional[FuturesCostModel] = None,
    folds: int = 3,
    max_candidates: Optional[int] = None,
    random_trials: int = 128,
) -> List[Dict[str, Any]]:
    cost_model = cost_model or FuturesCostModel()
    results: List[Dict[str, Any]] = []
    for symbol, frame in frames_by_symbol.items():
        normalized = normalize_minute_frame(frame)
        if normalized.empty:
            continue
        for family in strategy_families:
            family_overshoot_values = overshoot_trigger_multiples if _is_overshoot_family(str(family)) else (0.0,)
            for overshoot_multiple in family_overshoot_values:
                for theta_bps in theta_bps_values:
                    for vol_filter in vol_filters:
                        for open_interest_filter in open_interest_filters:
                            for time_filter in time_filters:
                                for spacing in event_spacing_bars_values:
                                    spec = DCFuturesStrategySpec(
                                        symbol=str(symbol),
                                        family=str(family),
                                        theta_bps=float(theta_bps),
                                        vol_filter=str(vol_filter),
                                        open_interest_filter=str(open_interest_filter),
                                        time_filter=str(time_filter),
                                        event_spacing_bars=int(spacing),
                                        overshoot_trigger_multiple=float(overshoot_multiple),
                                    )
                                    results.append(
                                        walk_forward_evaluate(
                                            normalized,
                                            spec,
                                            cost_model=cost_model,
                                            folds=folds,
                                            random_trials=random_trials,
                                        )
                                    )
                                    if max_candidates is not None and len(results) >= max_candidates:
                                        return apply_multiple_testing_control(results)
    return apply_multiple_testing_control(results)


def split_research_holdout_frames(
    frames_by_symbol: Mapping[str, pd.DataFrame],
    holdout_fraction: float = 0.20,
    min_research_bars: int = 180,
    min_holdout_bars: int = 60,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    research_frames: Dict[str, pd.DataFrame] = {}
    holdout_frames: Dict[str, pd.DataFrame] = {}
    fraction = min(max(float(holdout_fraction), 0.0), 0.80)
    for symbol, frame in frames_by_symbol.items():
        normalized = normalize_minute_frame(frame)
        if normalized.empty:
            continue
        if fraction <= 0.0:
            research_frames[str(symbol)] = normalized
            continue
        holdout_bars = max(int(round(len(normalized) * fraction)), int(min_holdout_bars))
        split_index = len(normalized) - holdout_bars
        if split_index < int(min_research_bars) or holdout_bars <= 0:
            research_frames[str(symbol)] = normalized
            continue
        research_frames[str(symbol)] = normalized.iloc[:split_index].copy().reset_index(drop=True)
        holdout_frames[str(symbol)] = normalized.iloc[split_index:].copy().reset_index(drop=True)
    return research_frames, holdout_frames


def apply_holdout_validation(
    results: Sequence[Mapping[str, Any]],
    holdout_frames_by_symbol: Mapping[str, pd.DataFrame],
    cost_model: Optional[FuturesCostModel] = None,
    min_holdout_events: int = 10,
    min_holdout_trades: int = 5,
) -> List[Dict[str, Any]]:
    cost_model = cost_model or FuturesCostModel()
    rows: List[Dict[str, Any]] = []
    for row in results:
        copy = dict(row)
        spec_mapping = copy.get("spec", {}) if isinstance(copy.get("spec"), Mapping) else {}
        summary = dict(copy.get("summary", {})) if isinstance(copy.get("summary"), Mapping) else {}
        copy["summary"] = summary
        symbol = str(spec_mapping.get("symbol", ""))
        holdout_frame = holdout_frames_by_symbol.get(symbol)
        holdout_flags: List[str] = []
        if not isinstance(holdout_frame, pd.DataFrame) or holdout_frame.empty:
            summary.update(
                {
                    "holdout_return": 0.0,
                    "holdout_capture_ratio": 0.0,
                    "holdout_events": 0,
                    "holdout_trades": 0,
                    "holdout_start": "",
                    "holdout_end": "",
                }
            )
            holdout_flags.append("HOLDOUT_MISSING")
        else:
            spec = _strategy_spec_from_mapping(spec_mapping)
            holdout = simulate_dc_strategy(holdout_frame, spec, cost_model=cost_model)
            normalized = normalize_minute_frame(holdout_frame)
            summary.update(
                {
                    "holdout_return": _as_float(holdout.get("strategy_return")),
                    "holdout_capture_ratio": _as_float(holdout.get("capture_ratio")),
                    "holdout_events": int(_as_float(holdout.get("event_count"))),
                    "holdout_trades": int(_as_float(holdout.get("trade_count"))),
                    "holdout_start": str(normalized.iloc[0]["timestamp"]) if not normalized.empty else "",
                    "holdout_end": str(normalized.iloc[-1]["timestamp"]) if not normalized.empty else "",
                }
            )
            if summary["holdout_return"] <= 0:
                holdout_flags.append("HOLDOUT_NON_POSITIVE_EXPECTANCY")
            if not 0.05 <= summary["holdout_capture_ratio"] <= 0.20:
                holdout_flags.append("HOLDOUT_CAPTURE_OUT_OF_TARGET")
            if summary["holdout_events"] < int(min_holdout_events):
                holdout_flags.append("HOLDOUT_LOW_EVENT_COUNT")
            if summary["holdout_trades"] < int(min_holdout_trades):
                holdout_flags.append("HOLDOUT_LOW_TRADE_COUNT")

        if holdout_flags:
            copy["failure_reasons"] = _with_failure_reason(copy.get("failure_reasons", []), holdout_flags[0])
            for flag in holdout_flags[1:]:
                copy["failure_reasons"] = _with_failure_reason(copy.get("failure_reasons", []), flag)
            if copy.get("status") == "PASS":
                copy["status"] = (
                    "WATCH"
                    if _as_float(summary.get("avg_test_return")) > 0 and int(_as_float(summary.get("positive_test_folds"))) >= 2
                    else "FAIL"
                )
        rows.append(copy)
    return apply_multiple_testing_control(rows)


def apply_multiple_testing_control(
    results: Sequence[Mapping[str, Any]],
    alpha: float = 0.10,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p_values: List[float] = []
    for row in results:
        copy = dict(row)
        summary = dict(copy.get("summary", {})) if isinstance(copy.get("summary"), Mapping) else {}
        copy["summary"] = summary
        rows.append(copy)
        p_values.append(min(max(_as_float(summary.get("random_control_p_value"), 1.0), 0.0), 1.0))

    q_values = _benjamini_hochberg_q_values(p_values)
    candidate_count = len(rows)
    threshold = min(max(float(alpha), 0.0), 1.0)
    for row, q_value in zip(rows, q_values):
        summary = row["summary"]
        summary["random_control_q_value"] = float(q_value)
        summary["multiple_testing_alpha"] = threshold
        summary["multiple_testing_candidates"] = candidate_count
        row["multiple_testing"] = {
            "method": "benjamini_hochberg_fdr_on_random_direction_p_values",
            "alpha": threshold,
            "candidate_count": candidate_count,
            "random_control_q_value": float(q_value),
        }
        if (
            row.get("status") in {"PASS", "WATCH"}
            and q_value > threshold
        ):
            row["failure_reasons"] = _with_failure_reason(
                row.get("failure_reasons", []),
                "RANDOM_CONTROL_FDR_NOT_SIGNIFICANT",
            )
            row["status"] = (
                "WATCH"
                if _as_float(summary.get("avg_test_return")) > 0 and int(_as_float(summary.get("positive_test_folds"))) >= 2
                else "FAIL"
            )
    return rank_research_results(rows)


def rank_research_results(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    status_rank = {"PASS": 0, "WATCH": 1, "FAIL": 2}

    def key(row: Mapping[str, Any]) -> Tuple[int, float, float]:
        summary = row.get("summary", {}) if isinstance(row.get("summary"), Mapping) else {}
        return (
            status_rank.get(str(row.get("status")), 9),
            -_as_float(summary.get("avg_test_return")),
            abs(_as_float(summary.get("avg_test_capture_ratio")) - 0.12),
        )

    return [dict(row) for row in sorted(results, key=key)]


def aggregate_research_results(
    results: Sequence[Mapping[str, Any]],
    cost_model: Optional[FuturesCostModel] = None,
    min_members: int = 2,
) -> List[Dict[str, Any]]:
    cost_model = cost_model or FuturesCostModel()
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        spec = row.get("spec", {}) if isinstance(row.get("spec"), Mapping) else {}
        product = normalize_futures_symbol(str(spec.get("symbol", "")))
        if not product:
            continue
        key = (product,) + _strategy_parameter_key(spec)
        groups[key].append(row)

    aggregated: List[Dict[str, Any]] = []
    for key, members in groups.items():
        if len(members) < int(min_members):
            continue
        product, family, theta_bps, vol_filter, open_interest_filter, time_filter, spacing, overshoot_multiple = key
        aggregate_spec = DCFuturesStrategySpec(
            symbol=f"{product}_AGG",
            family=str(family),
            theta_bps=float(theta_bps),
            vol_filter=str(vol_filter),
            open_interest_filter=str(open_interest_filter),
            time_filter=str(time_filter),
            event_spacing_bars=int(spacing),
            overshoot_trigger_multiple=float(overshoot_multiple),
        )
        aggregate = _aggregate_research_group(
            members,
            aggregate_spec,
            cost_model,
            divergence_prefix="CROSS_CONTRACT",
            aggregation={
                "mode": "product_contract_rollup",
                "product": product,
            },
        )
        aggregated.append(aggregate)
    return rank_research_results(aggregated)


def aggregate_cross_product_research_results(
    results: Sequence[Mapping[str, Any]],
    cost_model: Optional[FuturesCostModel] = None,
    min_products: int = 2,
) -> List[Dict[str, Any]]:
    cost_model = cost_model or FuturesCostModel()
    representatives = _product_representative_results(results, cost_model)
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in representatives:
        spec = row.get("spec", {}) if isinstance(row.get("spec"), Mapping) else {}
        groups[_strategy_parameter_key(spec)].append(row)

    aggregated: List[Dict[str, Any]] = []
    for key, members in groups.items():
        products = sorted(
            {
                str(member.get("aggregation", {}).get("product"))
                for member in members
                if isinstance(member.get("aggregation"), Mapping) and member.get("aggregation", {}).get("product")
            }
        )
        if len(products) < int(min_products):
            continue
        family, theta_bps, vol_filter, open_interest_filter, time_filter, spacing, overshoot_multiple = key
        aggregate_spec = DCFuturesStrategySpec(
            symbol="CROSS_PRODUCT_AGG",
            family=str(family),
            theta_bps=float(theta_bps),
            vol_filter=str(vol_filter),
            open_interest_filter=str(open_interest_filter),
            time_filter=str(time_filter),
            event_spacing_bars=int(spacing),
            overshoot_trigger_multiple=float(overshoot_multiple),
        )
        single_contract_products = [
            str(member.get("aggregation", {}).get("product"))
            for member in members
            if isinstance(member.get("aggregation"), Mapping)
            and int(_as_float(member.get("aggregation", {}).get("member_count"))) == 1
        ]
        aggregate = _aggregate_research_group(
            members,
            aggregate_spec,
            cost_model,
            divergence_prefix="CROSS_PRODUCT",
            member_labels=products,
            aggregation={
                "mode": "cross_product_param_family",
                "products": products,
                "single_contract_products": sorted(single_contract_products),
            },
        )
        if single_contract_products:
            aggregate["failure_reasons"] = list(
                dict.fromkeys(list(aggregate.get("failure_reasons", [])) + ["CROSS_PRODUCT_SINGLE_CONTRACT_MEMBER"])
            )
            aggregate["status"] = "WATCH" if aggregate["summary"]["avg_test_return"] > 0 else "FAIL"
        aggregated.append(aggregate)
    return rank_research_results(aggregated)


def _product_representative_results(
    results: Sequence[Mapping[str, Any]],
    cost_model: FuturesCostModel,
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        spec = row.get("spec", {}) if isinstance(row.get("spec"), Mapping) else {}
        product = normalize_futures_symbol(str(spec.get("symbol", "")))
        if product:
            groups[(product,) + _strategy_parameter_key(spec)].append(row)

    representatives: List[Dict[str, Any]] = []
    for key, members in groups.items():
        product = str(key[0])
        if len(members) >= 2:
            family, theta_bps, vol_filter, open_interest_filter, time_filter, spacing, overshoot_multiple = key[1:]
            aggregate_spec = DCFuturesStrategySpec(
                symbol=f"{product}_AGG",
                family=str(family),
                theta_bps=float(theta_bps),
                vol_filter=str(vol_filter),
                open_interest_filter=str(open_interest_filter),
                time_filter=str(time_filter),
                event_spacing_bars=int(spacing),
                overshoot_trigger_multiple=float(overshoot_multiple),
            )
            representatives.append(
                _aggregate_research_group(
                    members,
                    aggregate_spec,
                    cost_model,
                    divergence_prefix="CROSS_CONTRACT",
                    aggregation={
                        "mode": "product_contract_rollup",
                        "product": product,
                    },
                )
            )
            continue

        representative = dict(members[0])
        spec = representative.get("spec", {}) if isinstance(representative.get("spec"), Mapping) else {}
        symbol = str(spec.get("symbol", ""))
        representative["aggregation"] = {
            "mode": "single_contract_product_representative",
            "product": product,
            "member_count": 1,
            "members": [symbol] if symbol else [],
            "member_positive_count": int(
                isinstance(representative.get("summary"), Mapping)
                and _as_float(representative["summary"].get("avg_test_return")) > 0
            ),
            "member_target_capture_count": int(
                isinstance(representative.get("summary"), Mapping)
                and 0.05 <= _as_float(representative["summary"].get("avg_test_capture_ratio")) <= 0.20
            ),
        }
        representatives.append(representative)
    return representatives


def _aggregate_research_group(
    members: Sequence[Mapping[str, Any]],
    aggregate_spec: DCFuturesStrategySpec,
    cost_model: FuturesCostModel,
    divergence_prefix: str,
    aggregation: Mapping[str, Any],
    member_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    folds: List[Dict[str, Any]] = []
    for member in members:
        spec = member.get("spec", {}) if isinstance(member.get("spec"), Mapping) else {}
        symbol = str(spec.get("symbol", ""))
        product = normalize_futures_symbol(symbol)
        for fold in member.get("folds", []):
            if not isinstance(fold, Mapping):
                continue
            fold_row = dict(fold)
            had_source_symbol = "source_symbol" in fold_row
            fold_row["source_symbol"] = fold_row.get("source_symbol", symbol)
            fold_row["source_product"] = fold_row.get("source_product", product)
            if not had_source_symbol:
                fold_row["fold"] = f"{fold_row.get('source_symbol', symbol)}:{fold.get('fold', '')}"
            folds.append(fold_row)

    cost_stress_values = [
        _as_float(member.get("summary", {}).get("cost_1_5x_avg_return"))
        for member in members
        if isinstance(member.get("summary"), Mapping)
    ]
    random_controls = [
        member.get("random_direction_control", {})
        for member in members
        if isinstance(member.get("random_direction_control"), Mapping)
    ]
    random_p_values = [_as_float(control.get("p_value"), 1.0) for control in random_controls]
    aggregate = classify_walk_forward_result(
        aggregate_spec,
        cost_model,
        folds,
        cost_sensitivity={
            "cost_1_5x": {
                "avg_return": float(np.mean(cost_stress_values)) if cost_stress_values else 0.0,
                "member_returns": [float(value) for value in cost_stress_values],
            }
        },
        random_control={
            "beats_p95": bool(random_controls and all(bool(control.get("beats_p95", False)) for control in random_controls)),
            "p_value": float(max(random_p_values)) if random_p_values else 1.0,
            "method": "all_member_random_controls_must_pass",
        },
    )
    member_positive = sum(
        1
        for member in members
        if isinstance(member.get("summary"), Mapping) and _as_float(member["summary"].get("avg_test_return")) > 0
    )
    member_target_capture = sum(
        1
        for member in members
        if isinstance(member.get("summary"), Mapping)
        and 0.05 <= _as_float(member["summary"].get("avg_test_capture_ratio")) <= 0.20
    )
    member_cost_survival = sum(1 for value in cost_stress_values if value > 0)
    member_random_survival = sum(1 for control in random_controls if bool(control.get("beats_p95", False)))
    holdout_summaries = [
        member.get("summary", {})
        for member in members
        if isinstance(member.get("summary"), Mapping) and "holdout_return" in member.get("summary", {})
    ]
    holdout_returns = [_as_float(summary.get("holdout_return")) for summary in holdout_summaries]
    holdout_captures = [_as_float(summary.get("holdout_capture_ratio")) for summary in holdout_summaries]
    holdout_trades = [int(_as_float(summary.get("holdout_trades"))) for summary in holdout_summaries]
    member_holdout_positive = sum(1 for value in holdout_returns if value > 0)
    member_holdout_target_capture = sum(
        1
        for value, capture in zip(holdout_returns, holdout_captures)
        if value > 0 and 0.05 <= capture <= 0.20
    )
    if holdout_summaries:
        aggregate["summary"]["holdout_return"] = float(np.mean(holdout_returns)) if holdout_returns else 0.0
        aggregate["summary"]["holdout_capture_ratio"] = float(np.mean(holdout_captures)) if holdout_captures else 0.0
        aggregate["summary"]["holdout_trades"] = int(sum(holdout_trades))
        aggregate["summary"]["holdout_member_count"] = len(holdout_summaries)
        aggregate["summary"]["holdout_positive_member_count"] = member_holdout_positive
        aggregate["summary"]["holdout_target_capture_member_count"] = member_holdout_target_capture
    extra_flags: List[str] = []
    if member_positive < len(members):
        extra_flags.append(f"{divergence_prefix}_MEMBER_EXPECTANCY_DIVERGENCE")
    if member_target_capture < len(members):
        extra_flags.append(f"{divergence_prefix}_CAPTURE_DIVERGENCE")
    if member_cost_survival < len(members):
        extra_flags.append(f"{divergence_prefix}_COST_STRESS_DIVERGENCE")
    if member_random_survival < len(members):
        extra_flags.append(f"{divergence_prefix}_RANDOM_CONTROL_DIVERGENCE")
    if holdout_summaries and member_holdout_positive < len(holdout_summaries):
        extra_flags.append(f"{divergence_prefix}_HOLDOUT_EXPECTANCY_DIVERGENCE")
    if holdout_summaries and member_holdout_target_capture < len(holdout_summaries):
        extra_flags.append(f"{divergence_prefix}_HOLDOUT_CAPTURE_DIVERGENCE")
    if extra_flags:
        aggregate["failure_reasons"] = list(dict.fromkeys(list(aggregate.get("failure_reasons", [])) + extra_flags))
        aggregate["status"] = "WATCH" if aggregate["summary"]["avg_test_return"] > 0 else "FAIL"

    if member_labels is None:
        member_labels = sorted(
            {
                str(member.get("spec", {}).get("symbol"))
                for member in members
                if isinstance(member.get("spec"), Mapping)
            }
        )
    aggregate["aggregation"] = dict(aggregation) | {
        "member_count": len(members),
        "members": list(member_labels),
        "member_positive_count": member_positive,
        "member_target_capture_count": member_target_capture,
        "member_cost_survival_count": member_cost_survival,
        "member_random_survival_count": member_random_survival,
        "member_holdout_positive_count": member_holdout_positive,
        "member_holdout_target_capture_count": member_holdout_target_capture,
    }
    return aggregate


def _strategy_parameter_key(spec: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        spec.get("family"),
        float(_as_float(spec.get("theta_bps"))),
        spec.get("vol_filter", "all"),
        spec.get("open_interest_filter", "all"),
        spec.get("time_filter", "all"),
        int(_as_float(spec.get("event_spacing_bars"))),
        float(_as_float(spec.get("overshoot_trigger_multiple"))),
    )


def _strategy_spec_from_mapping(spec: Mapping[str, Any]) -> DCFuturesStrategySpec:
    return DCFuturesStrategySpec(
        symbol=str(spec.get("symbol", "")),
        family=str(spec.get("family", "")),
        theta_bps=float(_as_float(spec.get("theta_bps"))),
        vol_filter=str(spec.get("vol_filter", "all")),
        open_interest_filter=str(spec.get("open_interest_filter", "all")),
        time_filter=str(spec.get("time_filter", "all")),
        event_spacing_bars=int(_as_float(spec.get("event_spacing_bars"))),
        max_hold_bars=int(_as_float(spec.get("max_hold_bars"), 12.0)),
        stop_multiple=float(_as_float(spec.get("stop_multiple"), 1.0)),
        take_profit_multiple=float(_as_float(spec.get("take_profit_multiple"), 2.0)),
        overshoot_trigger_multiple=float(_as_float(spec.get("overshoot_trigger_multiple"))),
    )


def _benjamini_hochberg_q_values(p_values: Sequence[float]) -> List[float]:
    if not p_values:
        return []
    sanitized = [min(max(_as_float(value, 1.0), 0.0), 1.0) for value in p_values]
    sorted_indices = sorted(range(len(sanitized)), key=lambda index: sanitized[index])
    q_values = [1.0] * len(sanitized)
    running_min = 1.0
    total = len(sanitized)
    for rank, index in reversed(list(enumerate(sorted_indices, start=1))):
        running_min = min(running_min, sanitized[index] * total / rank)
        q_values[index] = float(min(max(running_min, 0.0), 1.0))
    return q_values


def _with_failure_reason(reasons: Any, reason: str) -> List[str]:
    if isinstance(reasons, str):
        existing = [reasons]
    else:
        try:
            existing = [str(item) for item in reasons]
        except TypeError:
            existing = []
    return list(dict.fromkeys(existing + [reason]))


def fetch_main_contract_minute_frames(
    products: Sequence[str] = DEFAULT_PRODUCTS,
    period: str = "5",
    max_contracts: Optional[int] = None,
    ak: Any = None,
) -> Dict[str, pd.DataFrame]:
    if ak is None:
        import akshare as ak  # type: ignore  # noqa: WPS433

    wanted = [normalize_futures_symbol(product) for product in products]
    contracts = _main_contracts_for_products(wanted, ak)
    frames: Dict[str, pd.DataFrame] = {}
    for contract in contracts:
        if max_contracts is not None and len(frames) >= max_contracts:
            break
        try:
            frame = ak.futures_zh_minute_sina(symbol=contract, period=str(period))
        except Exception:
            continue
        normalized = normalize_minute_frame(frame)
        if not normalized.empty:
            frames[contract] = normalized
    return frames


def fetch_cached_main_contract_minute_frames(
    products: Sequence[str] = DEFAULT_PRODUCTS,
    period: str = "5",
    max_contracts: Optional[int] = None,
    cache_dir: str | Path = "state/futures_minute_cache",
    ak: Any = None,
) -> Dict[str, pd.DataFrame]:
    if ak is None:
        import akshare as ak  # type: ignore  # noqa: WPS433

    wanted = [normalize_futures_symbol(product) for product in products]
    contracts = _main_contracts_for_products(wanted, ak)
    frames: Dict[str, pd.DataFrame] = {}
    for contract in contracts:
        if max_contracts is not None and len(frames) >= max_contracts:
            break
        cached = load_minute_cache(contract, period=period, cache_dir=cache_dir)
        fresh = pd.DataFrame()
        try:
            fresh = ak.futures_zh_minute_sina(symbol=contract, period=str(period))
        except Exception:
            pass
        merged = update_minute_cache(
            contract,
            fresh,
            period=period,
            cache_dir=cache_dir,
            existing_frame=cached,
        )
        if not merged.empty:
            frames[contract] = merged
    return frames


def load_cached_minute_frames(
    cache_dir: str | Path = "state/futures_minute_cache",
    period: str = "5",
    symbols: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    period_dir = Path(cache_dir) / str(period)
    if not period_dir.exists():
        return {}
    wanted = {str(symbol).upper() for symbol in symbols or []}
    frames: Dict[str, pd.DataFrame] = {}
    for path in sorted(period_dir.glob("*.csv")):
        symbol = path.stem.upper()
        if wanted and symbol not in wanted:
            continue
        frame = load_minute_cache(symbol, period=period, cache_dir=cache_dir)
        if not frame.empty:
            frames[symbol] = frame
    return frames


def load_csv_minute_frames(
    paths: Sequence[str | Path],
    symbol_column: str = "symbol",
) -> Dict[str, pd.DataFrame]:
    frames_by_symbol: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    wanted_symbol_column = str(symbol_column).strip().lower()
    for pathlike in paths:
        path = Path(pathlike).expanduser()
        if not path.exists():
            raise FileNotFoundError(str(path))
        raw = pd.read_csv(path)
        if raw.empty:
            continue
        columns_by_lower = {str(column).strip().lower(): column for column in raw.columns}
        actual_symbol_column = columns_by_lower.get(wanted_symbol_column) if wanted_symbol_column else None
        if actual_symbol_column is None:
            symbol = _sanitize_csv_symbol(path.stem)
            frame = normalize_minute_frame(raw)
            if not frame.empty:
                frames_by_symbol[symbol].append(frame)
            continue

        for raw_symbol, group in raw.groupby(raw[actual_symbol_column].astype(str), dropna=True):
            symbol = _sanitize_csv_symbol(raw_symbol)
            if not symbol:
                continue
            frame = normalize_minute_frame(group.drop(columns=[actual_symbol_column], errors="ignore"))
            if not frame.empty:
                frames_by_symbol[symbol].append(frame)

    frames: Dict[str, pd.DataFrame] = {}
    for symbol, symbol_frames in frames_by_symbol.items():
        if not symbol_frames:
            continue
        merged = pd.concat(symbol_frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
        merged = normalize_minute_frame(merged)
        if not merged.empty:
            frames[symbol] = merged
    return dict(sorted(frames.items()))


def minute_cache_path(symbol: str, period: str = "5", cache_dir: str | Path = "state/futures_minute_cache") -> Path:
    safe_symbol = "".join(ch for ch in str(symbol).upper() if ch.isalnum() or ch in {"_", "-"})
    safe_period = "".join(ch for ch in str(period) if ch.isalnum() or ch in {"_", "-"})
    return Path(cache_dir) / safe_period / f"{safe_symbol}.csv"


def load_minute_cache(symbol: str, period: str = "5", cache_dir: str | Path = "state/futures_minute_cache") -> pd.DataFrame:
    path = minute_cache_path(symbol, period=period, cache_dir=cache_dir)
    if not path.exists():
        return pd.DataFrame()
    try:
        return normalize_minute_frame(pd.read_csv(path))
    except Exception:
        return pd.DataFrame()


def update_minute_cache(
    symbol: str,
    fresh_frame: pd.DataFrame,
    period: str = "5",
    cache_dir: str | Path = "state/futures_minute_cache",
    existing_frame: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    existing = existing_frame if existing_frame is not None else load_minute_cache(symbol, period=period, cache_dir=cache_dir)
    if isinstance(existing, pd.DataFrame) and not existing.empty:
        frames.append(normalize_minute_frame(existing))
    if isinstance(fresh_frame, pd.DataFrame) and not fresh_frame.empty:
        frames.append(normalize_minute_frame(fresh_frame))
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
    merged = normalize_minute_frame(merged)
    path = minute_cache_path(symbol, period=period, cache_dir=cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged


def candidate_history_key(row: Mapping[str, Any]) -> str:
    spec = row.get("spec", {}) if isinstance(row.get("spec"), Mapping) else {}
    parts = [
        str(spec.get("symbol", "")).upper(),
        str(spec.get("family", "")),
        _format_float_key(spec.get("theta_bps")),
        str(spec.get("vol_filter", "all")),
        str(spec.get("open_interest_filter", "all")),
        str(spec.get("time_filter", "all")),
        str(int(_as_float(spec.get("event_spacing_bars")))),
        str(int(_as_float(spec.get("max_hold_bars"), 12.0))),
        _format_float_key(spec.get("stop_multiple", 1.0)),
        _format_float_key(spec.get("take_profit_multiple", 2.0)),
        _format_float_key(spec.get("overshoot_trigger_multiple")),
    ]
    return "|".join(parts)


def candidate_history_path(state_dir: str | Path = "state") -> Path:
    return Path(state_dir) / "futures_dc_candidate_history.json"


def load_candidate_history(state_dir: str | Path = "state") -> Dict[str, Any]:
    path = candidate_history_path(state_dir)
    if not path.exists():
        return _empty_candidate_history()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_candidate_history()
    if not isinstance(payload, Mapping):
        return _empty_candidate_history()
    scans = payload.get("scans", [])
    if not isinstance(scans, list):
        scans = []
    return {
        "schema_version": int(_as_float(payload.get("schema_version"), HISTORY_SCHEMA_VERSION)),
        "updated_at": str(payload.get("updated_at", "")),
        "scans": [dict(scan) for scan in scans if isinstance(scan, Mapping)],
    }


def update_candidate_history(
    results: Sequence[Mapping[str, Any]],
    state_dir: str | Path = "state",
    generated_at: Optional[datetime] = None,
    max_scans: int = DEFAULT_HISTORY_MAX_SCANS,
) -> Tuple[Dict[str, Any], Path]:
    generated_at = generated_at or datetime.now()
    path = candidate_history_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = apply_multiple_testing_control(results)
    rows = [_candidate_history_row(row) for row in ranked]
    scan_signature = hashlib.sha256(
        json.dumps(_json_safe(rows), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    scan_id = scan_signature[:16]
    history = load_candidate_history(state_dir)
    scans = history.get("scans", []) if isinstance(history.get("scans"), list) else []
    existing_ids = {str(scan.get("scan_id", "")) for scan in scans if isinstance(scan, Mapping)}
    if scan_id not in existing_ids:
        scans.append(
            {
                "scan_id": scan_id,
                "signature": scan_signature,
                "generated_at": generated_at.isoformat(timespec="seconds"),
                "candidate_count": len(rows),
                "pass_count": sum(1 for row in rows if row.get("status") == "PASS"),
                "watch_count": sum(1 for row in rows if row.get("status") == "WATCH"),
                "rows": rows,
            }
        )
    scan_limit = max(int(max_scans), 1)
    if len(scans) > scan_limit:
        scans = scans[-scan_limit:]
    updated = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "updated_at": generated_at.isoformat(timespec="seconds"),
        "scans": scans,
    }
    path.write_text(json.dumps(_json_safe(updated), ensure_ascii=False, indent=2), encoding="utf-8")
    return updated, path


def summarize_candidate_history(
    history: Mapping[str, Any],
    current_results: Optional[Sequence[Mapping[str, Any]]] = None,
    limit: int = 15,
) -> List[Dict[str, Any]]:
    scans = history.get("scans", []) if isinstance(history.get("scans"), list) else []
    current_keys = (
        {candidate_history_key(row) for row in current_results}
        if current_results is not None
        else set()
    )
    rows_by_key: Dict[str, List[Tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for scan_index, scan in enumerate(scans):
        if not isinstance(scan, Mapping):
            continue
        for row in scan.get("rows", []):
            if not isinstance(row, Mapping):
                continue
            key = str(row.get("candidate_key", ""))
            if not key:
                continue
            if current_results is not None and key not in current_keys:
                continue
            rows_by_key[key].append((scan_index, row))

    summaries: List[Dict[str, Any]] = []
    for key, entries in rows_by_key.items():
        ordered = sorted(entries, key=lambda item: item[0])
        latest = ordered[-1][1]
        holdout_returns = [_as_float(row.get("holdout_return")) for _, row in ordered]
        holdout_captures = [_as_float(row.get("holdout_capture_ratio")) for _, row in ordered]
        avg_returns = [_as_float(row.get("avg_test_return")) for _, row in ordered]
        avg_captures = [_as_float(row.get("avg_test_capture_ratio")) for _, row in ordered]
        positive_holdout = sum(1 for value in holdout_returns if value > 0)
        target_holdout = sum(
            1
            for value, capture in zip(holdout_returns, holdout_captures)
            if value > 0 and 0.05 <= capture <= 0.20
        )
        positive_research = sum(1 for value in avg_returns if value > 0)
        target_research = sum(
            1
            for value, capture in zip(avg_returns, avg_captures)
            if value > 0 and 0.05 <= capture <= 0.20
        )
        full_target = sum(
            1
            for avg_return, avg_capture, holdout_return, holdout_capture in zip(
                avg_returns,
                avg_captures,
                holdout_returns,
                holdout_captures,
            )
            if (
                avg_return > 0
                and 0.05 <= avg_capture <= 0.20
                and holdout_return > 0
                and 0.05 <= holdout_capture <= 0.20
            )
        )
        summaries.append(
            {
                "candidate_key": key,
                "label": _candidate_history_label(latest),
                "latest_status": str(latest.get("status", "")),
                "seen_scans": len(ordered),
                "full_target_scans": full_target,
                "research_positive_scans": positive_research,
                "research_target_scans": target_research,
                "holdout_positive_scans": positive_holdout,
                "holdout_target_scans": target_holdout,
                "median_holdout_return": float(np.median(holdout_returns)) if holdout_returns else 0.0,
                "median_holdout_capture_ratio": float(np.median(holdout_captures)) if holdout_captures else 0.0,
                "latest_avg_test_return": _as_float(latest.get("avg_test_return")),
                "latest_avg_test_capture_ratio": _as_float(latest.get("avg_test_capture_ratio")),
                "latest_holdout_return": _as_float(latest.get("holdout_return")),
                "latest_holdout_capture_ratio": _as_float(latest.get("holdout_capture_ratio")),
                "latest_random_control_q_value": _as_float(latest.get("random_control_q_value"), 1.0),
                "latest_failure_reasons": list(latest.get("failure_reasons", []))
                if isinstance(latest.get("failure_reasons"), list)
                else [],
            }
        )

    def sort_key(row: Mapping[str, Any]) -> Tuple[int, int, int, int, int, float, float]:
        status_rank = {"PASS": 0, "WATCH": 1, "FAIL": 2}
        return (
            -int(_as_float(row.get("full_target_scans"))),
            -int(_as_float(row.get("research_target_scans"))),
            -int(_as_float(row.get("holdout_target_scans"))),
            -int(_as_float(row.get("research_positive_scans"))),
            -int(_as_float(row.get("holdout_positive_scans"))),
            -int(_as_float(row.get("seen_scans"))),
            status_rank.get(str(row.get("latest_status")), 9),
            -_as_float(row.get("latest_avg_test_return")),
            abs(_as_float(row.get("latest_avg_test_capture_ratio")) - 0.12),
        )

    return [dict(row) for row in sorted(summaries, key=sort_key)[: max(int(limit), 0)]]


def _build_persistence_watchlist_lines(
    history_summary: Sequence[Mapping[str, Any]],
    history_scan_count: int,
) -> List[str]:
    lines = [
        "",
        "## Persistence Watchlist",
        "",
        f"- Unique scans in history: {int(history_scan_count)}",
        "- Persistence is not proven until the same parameter family survives at least 3 unique scans with positive holdout and 5%-20% holdout capture.",
        "",
        "| latest | candidate | seen | full_target | research_pos/target | holdout_pos/target | median_holdout_return | median_holdout_capture | latest_avg_return | latest_capture | latest_rand_q | latest_reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    if not history_summary:
        lines.append("| - | - | 0 | 0 | 0/0 | 0/0 | 0.0000% | 0.00% | 0.0000% | 0.00% | 1.000 | no history yet |")
        return lines
    for row in history_summary[:15]:
        reasons = ",".join(str(reason) for reason in row.get("latest_failure_reasons", [])) or "-"
        lines.append(
            "| {status} | {label} | {seen} | {full_target} | {research_pos}/{research_target} | {holdout_pos}/{holdout_target} | {median_holdout_ret:.4%} | {median_holdout_capture:.2%} | {latest_ret:.4%} | {latest_capture:.2%} | {rand_q:.3f} | {reasons} |".format(
                status=row.get("latest_status", ""),
                label=row.get("label", ""),
                seen=int(_as_float(row.get("seen_scans"))),
                full_target=int(_as_float(row.get("full_target_scans"))),
                research_pos=int(_as_float(row.get("research_positive_scans"))),
                research_target=int(_as_float(row.get("research_target_scans"))),
                holdout_pos=int(_as_float(row.get("holdout_positive_scans"))),
                holdout_target=int(_as_float(row.get("holdout_target_scans"))),
                median_holdout_ret=_as_float(row.get("median_holdout_return")),
                median_holdout_capture=_as_float(row.get("median_holdout_capture_ratio")),
                latest_ret=_as_float(row.get("latest_avg_test_return")),
                latest_capture=_as_float(row.get("latest_avg_test_capture_ratio")),
                rand_q=_as_float(row.get("latest_random_control_q_value"), 1.0),
                reasons=reasons,
            )
        )
    return lines


def _candidate_history_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    spec = row.get("spec", {}) if isinstance(row.get("spec"), Mapping) else {}
    summary = row.get("summary", {}) if isinstance(row.get("summary"), Mapping) else {}
    folds = row.get("folds", []) if isinstance(row.get("folds"), list) else []
    test_starts = [str(fold.get("test_start", "")) for fold in folds if isinstance(fold, Mapping) and fold.get("test_start")]
    test_ends = [str(fold.get("test_end", "")) for fold in folds if isinstance(fold, Mapping) and fold.get("test_end")]
    symbol = str(spec.get("symbol", "")).upper()
    return {
        "candidate_key": candidate_history_key(row),
        "status": str(row.get("status", "")),
        "symbol": symbol,
        "product": normalize_futures_symbol(symbol),
        "family": str(spec.get("family", "")),
        "theta_bps": _as_float(spec.get("theta_bps")),
        "vol_filter": str(spec.get("vol_filter", "all")),
        "open_interest_filter": str(spec.get("open_interest_filter", "all")),
        "time_filter": str(spec.get("time_filter", "all")),
        "event_spacing_bars": int(_as_float(spec.get("event_spacing_bars"))),
        "max_hold_bars": int(_as_float(spec.get("max_hold_bars"), 12.0)),
        "stop_multiple": _as_float(spec.get("stop_multiple", 1.0)),
        "take_profit_multiple": _as_float(spec.get("take_profit_multiple", 2.0)),
        "overshoot_trigger_multiple": _as_float(spec.get("overshoot_trigger_multiple")),
        "avg_test_return": _as_float(summary.get("avg_test_return")),
        "avg_test_capture_ratio": _as_float(summary.get("avg_test_capture_ratio")),
        "median_test_capture_ratio": _as_float(summary.get("median_test_capture_ratio")),
        "positive_test_folds": int(_as_float(summary.get("positive_test_folds"))),
        "target_capture_folds": int(_as_float(summary.get("target_capture_folds"))),
        "total_test_trades": int(_as_float(summary.get("total_test_trades"))),
        "cost_1_5x_avg_return": _as_float(summary.get("cost_1_5x_avg_return")),
        "random_control_p_value": _as_float(summary.get("random_control_p_value"), 1.0),
        "random_control_q_value": _as_float(summary.get("random_control_q_value"), 1.0),
        "holdout_return": _as_float(summary.get("holdout_return")),
        "holdout_capture_ratio": _as_float(summary.get("holdout_capture_ratio")),
        "holdout_events": int(_as_float(summary.get("holdout_events"))),
        "holdout_trades": int(_as_float(summary.get("holdout_trades"))),
        "holdout_start": str(summary.get("holdout_start", "")),
        "holdout_end": str(summary.get("holdout_end", "")),
        "test_start": min(test_starts) if test_starts else "",
        "test_end": max(test_ends) if test_ends else "",
        "failure_reasons": [str(reason) for reason in row.get("failure_reasons", [])],
    }


def _candidate_history_label(row: Mapping[str, Any]) -> str:
    filters = ",".join(
        [
            str(row.get("vol_filter", "all")),
            str(row.get("open_interest_filter", "all")),
            str(row.get("time_filter", "all")),
            f"spacing={int(_as_float(row.get('event_spacing_bars')))}",
            f"overshoot={_as_float(row.get('overshoot_trigger_multiple')):.1f}",
        ]
    )
    return "{symbol} {family} theta={theta:.1f} {filters}".format(
        symbol=row.get("symbol", ""),
        family=row.get("family", ""),
        theta=_as_float(row.get("theta_bps")),
        filters=filters,
    )


def _empty_candidate_history() -> Dict[str, Any]:
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "updated_at": "",
        "scans": [],
    }


def _sanitize_csv_symbol(symbol: Any) -> str:
    return "".join(ch for ch in str(symbol).strip().upper() if ch.isalnum() or ch in {"_", "-"})


def _format_float_key(value: Any) -> str:
    return f"{_as_float(value):.6f}".rstrip("0").rstrip(".") or "0"


def build_research_report(
    results: Sequence[Mapping[str, Any]],
    generated_at: Optional[datetime] = None,
    history_summary: Optional[Sequence[Mapping[str, Any]]] = None,
    history_scan_count: int = 0,
) -> str:
    generated_at = generated_at or datetime.now()
    ranked = apply_multiple_testing_control(results)
    pass_count = sum(1 for row in ranked if row.get("status") == "PASS")
    watch_count = sum(1 for row in ranked if row.get("status") == "WATCH")
    holdout_positive_count = sum(
        1
        for row in ranked
        if isinstance(row.get("summary"), Mapping) and _as_float(row["summary"].get("holdout_return")) > 0
    )
    holdout_target_count = sum(
        1
        for row in ranked
        if isinstance(row.get("summary"), Mapping)
        and _as_float(row["summary"].get("holdout_return")) > 0
        and 0.05 <= _as_float(row["summary"].get("holdout_capture_ratio")) <= 0.20
    )
    lines = [
        "# Futures DC Capture Research Report",
        "",
        f"- Generated at: {generated_at.isoformat(timespec='seconds')}",
        f"- Candidates scanned: {len(ranked)}",
        f"- PASS: {pass_count}",
        f"- WATCH: {watch_count}",
        f"- Holdout positive: {holdout_positive_count}",
        f"- Holdout positive with 5%-20% capture: {holdout_target_count}",
        "- Gate: OOS net positive, average and median capture_ratio in 5%-20%, untouched holdout confirmation, at least two target-capture folds, positive 1.5x-cost stress, random-direction control beaten after FDR adjustment, event/trade sufficiency, no bias flags.",
        "- Trading rule: DC confirmation is tradable only from the next bar open.",
        "",
        "## Top Candidates",
        "",
        "| status | symbol | family | theta_bps | filters | avg_test_return | avg_capture | holdout_return | holdout_capture | holdout_trades | stress_1_5x | rand_p | rand_q | trades | reasons |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked[:25]:
        spec = row.get("spec", {}) if isinstance(row.get("spec"), Mapping) else {}
        summary = row.get("summary", {}) if isinstance(row.get("summary"), Mapping) else {}
        filters = ",".join(
            [
                str(spec.get("vol_filter", "all")),
                str(spec.get("open_interest_filter", "all")),
                str(spec.get("time_filter", "all")),
                f"spacing={spec.get('event_spacing_bars', 0)}",
                f"overshoot={_as_float(spec.get('overshoot_trigger_multiple')):.1f}",
            ]
        )
        reasons = ",".join(str(reason) for reason in row.get("failure_reasons", []))
        lines.append(
            "| {status} | {symbol} | {family} | {theta:.1f} | {filters} | {ret:.4%} | {capture:.2%} | {holdout_ret:.4%} | {holdout_capture:.2%} | {holdout_trades} | {stress:.4%} | {rand_p:.3f} | {rand_q:.3f} | {trades} | {reasons} |".format(
                status=row.get("status", ""),
                symbol=spec.get("symbol", ""),
                family=spec.get("family", ""),
                theta=_as_float(spec.get("theta_bps")),
                filters=filters,
                ret=_as_float(summary.get("avg_test_return")),
                capture=_as_float(summary.get("avg_test_capture_ratio")),
                holdout_ret=_as_float(summary.get("holdout_return")),
                holdout_capture=_as_float(summary.get("holdout_capture_ratio")),
                holdout_trades=int(_as_float(summary.get("holdout_trades"))),
                stress=_as_float(summary.get("cost_1_5x_avg_return")),
                rand_p=_as_float(summary.get("random_control_p_value"), 1.0),
                rand_q=_as_float(summary.get("random_control_q_value"), 1.0),
                trades=int(_as_float(summary.get("total_test_trades"))),
                reasons=reasons or "-",
            )
        )
    lines.extend(
        _build_persistence_watchlist_lines(
            history_summary or [],
            history_scan_count=history_scan_count,
        )
    )
    contract_aggregates = apply_multiple_testing_control(aggregate_research_results(ranked))
    lines.extend(
        [
            "",
            "## Cross-Contract Aggregates",
            "",
            "| status | product | members | family | theta_bps | filters | avg_test_return | avg_capture | holdout_return | holdout_capture | holdout_pos/target/total | rand_q | trades | reasons |",
            "|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    if not contract_aggregates:
        lines.append("| - | - | 0 | - | 0.0 | - | 0.0000% | 0.00% | 0.0000% | 0.00% | 0/0/0 | 1.000 | 0 | no multi-contract product groups |")
    for row in contract_aggregates[:15]:
        spec = row.get("spec", {}) if isinstance(row.get("spec"), Mapping) else {}
        summary = row.get("summary", {}) if isinstance(row.get("summary"), Mapping) else {}
        aggregation = row.get("aggregation", {}) if isinstance(row.get("aggregation"), Mapping) else {}
        filters = ",".join(
            [
                str(spec.get("vol_filter", "all")),
                str(spec.get("open_interest_filter", "all")),
                str(spec.get("time_filter", "all")),
                f"spacing={spec.get('event_spacing_bars', 0)}",
                f"overshoot={_as_float(spec.get('overshoot_trigger_multiple')):.1f}",
            ]
        )
        reasons = ",".join(str(reason) for reason in row.get("failure_reasons", []))
        holdout_members = "{positive}/{target}/{total}".format(
            positive=int(_as_float(aggregation.get("member_holdout_positive_count"))),
            target=int(_as_float(aggregation.get("member_holdout_target_capture_count"))),
            total=int(_as_float(summary.get("holdout_member_count"))),
        )
        lines.append(
            "| {status} | {product} | {members} | {family} | {theta:.1f} | {filters} | {ret:.4%} | {capture:.2%} | {holdout_ret:.4%} | {holdout_capture:.2%} | {holdout_members} | {rand_q:.3f} | {trades} | {reasons} |".format(
                status=row.get("status", ""),
                product=aggregation.get("product", spec.get("symbol", "")),
                members=int(_as_float(aggregation.get("member_count"))),
                family=spec.get("family", ""),
                theta=_as_float(spec.get("theta_bps")),
                filters=filters,
                ret=_as_float(summary.get("avg_test_return")),
                capture=_as_float(summary.get("avg_test_capture_ratio")),
                holdout_ret=_as_float(summary.get("holdout_return")),
                holdout_capture=_as_float(summary.get("holdout_capture_ratio")),
                holdout_members=holdout_members,
                rand_q=_as_float(summary.get("random_control_q_value"), 1.0),
                trades=int(_as_float(summary.get("total_test_trades"))),
                reasons=reasons or "-",
            )
        )
    product_aggregates = apply_multiple_testing_control(aggregate_cross_product_research_results(ranked))
    lines.extend(
        [
            "",
            "## Cross-Product Aggregates",
            "",
            "| status | products | members | family | theta_bps | filters | avg_test_return | avg_capture | holdout_return | holdout_capture | holdout_pos/target/total | stress_1_5x | rand_p | rand_q | trades | reasons |",
            "|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    if not product_aggregates:
        lines.append("| - | - | 0 | - | 0.0 | - | 0.0000% | 0.00% | 0.0000% | 0.00% | 0/0/0 | 0.0000% | 1.000 | 1.000 | 0 | no multi-product parameter groups |")
    for row in product_aggregates[:15]:
        spec = row.get("spec", {}) if isinstance(row.get("spec"), Mapping) else {}
        summary = row.get("summary", {}) if isinstance(row.get("summary"), Mapping) else {}
        aggregation = row.get("aggregation", {}) if isinstance(row.get("aggregation"), Mapping) else {}
        filters = ",".join(
            [
                str(spec.get("vol_filter", "all")),
                str(spec.get("open_interest_filter", "all")),
                str(spec.get("time_filter", "all")),
                f"spacing={spec.get('event_spacing_bars', 0)}",
                f"overshoot={_as_float(spec.get('overshoot_trigger_multiple')):.1f}",
            ]
        )
        products = ",".join(str(product) for product in aggregation.get("members", []))
        reasons = ",".join(str(reason) for reason in row.get("failure_reasons", []))
        holdout_members = "{positive}/{target}/{total}".format(
            positive=int(_as_float(aggregation.get("member_holdout_positive_count"))),
            target=int(_as_float(aggregation.get("member_holdout_target_capture_count"))),
            total=int(_as_float(summary.get("holdout_member_count"))),
        )
        lines.append(
            "| {status} | {products} | {members} | {family} | {theta:.1f} | {filters} | {ret:.4%} | {capture:.2%} | {holdout_ret:.4%} | {holdout_capture:.2%} | {holdout_members} | {stress:.4%} | {rand_p:.3f} | {rand_q:.3f} | {trades} | {reasons} |".format(
                status=row.get("status", ""),
                products=products or "-",
                members=int(_as_float(aggregation.get("member_count"))),
                family=spec.get("family", ""),
                theta=_as_float(spec.get("theta_bps")),
                filters=filters,
                ret=_as_float(summary.get("avg_test_return")),
                capture=_as_float(summary.get("avg_test_capture_ratio")),
                holdout_ret=_as_float(summary.get("holdout_return")),
                holdout_capture=_as_float(summary.get("holdout_capture_ratio")),
                holdout_members=holdout_members,
                stress=_as_float(summary.get("cost_1_5x_avg_return")),
                rand_p=_as_float(summary.get("random_control_p_value"), 1.0),
                rand_q=_as_float(summary.get("random_control_q_value"), 1.0),
                trades=int(_as_float(summary.get("total_test_trades"))),
                reasons=reasons or "-",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "PASS means the fragment survived the strict research gate, including untouched holdout. WATCH means the path fragment is interesting but still fails at least one gate, usually event count, train/test stability, target capture band, holdout, cost stress, random-direction control after FDR adjustment, cross-contract consistency, or cross-product scalability. FAIL means it is not evidence of a repeatable edge.",
            "",
        ]
    )
    return "\n".join(lines)


def write_research_outputs(
    results: Sequence[Mapping[str, Any]],
    state_dir: str | Path = "state",
    report_path: str | Path = "FUTURES_DC_CAPTURE_REPORT.md",
) -> Dict[str, str]:
    ranked = apply_multiple_testing_control(results)
    generated_at = datetime.now()
    state_path = Path(state_dir)
    state_path.mkdir(parents=True, exist_ok=True)
    json_path = state_path / "futures_dc_capture_candidates.json"
    json_path.write_text(json.dumps(_json_safe(ranked), ensure_ascii=False, indent=2), encoding="utf-8")
    history, history_path = update_candidate_history(
        ranked,
        state_dir=state_path,
        generated_at=generated_at,
    )
    history_summary = summarize_candidate_history(history, current_results=ranked)
    report = build_research_report(
        ranked,
        generated_at=generated_at,
        history_summary=history_summary,
        history_scan_count=len(history.get("scans", [])) if isinstance(history, Mapping) else 0,
    )
    markdown_path = Path(report_path)
    markdown_path.write_text(report, encoding="utf-8")
    return {"json_path": str(json_path), "history_path": str(history_path), "report_path": str(markdown_path)}


def _empty_simulation_result(
    spec: DCFuturesStrategySpec,
    cost_model: FuturesCostModel,
    reason: str,
) -> Dict[str, Any]:
    return {
        "spec": asdict(spec),
        "bar_count": 0,
        "event_count": 0,
        "segment_count": 0,
        "trade_count": 0,
        "strategy_return": 0.0,
        "dc_path_return": 0.0,
        "gross_dc_path_return": 0.0,
        "capture_ratio": 0.0,
        "win_rate": 0.0,
        "trades": [],
        "reason": reason,
        "lookahead_rule": "DC event is actionable only at the next bar open after confirmation.",
        "cost_model": asdict(cost_model) | {"round_trip_cost_bps": cost_model.round_trip_cost_bps},
    }


def _exit_for_bar(
    frame: pd.DataFrame,
    index: int,
    position: int,
    entry_price: float,
    entry_index: int,
    spec: DCFuturesStrategySpec,
    theta: float,
) -> Tuple[Optional[float], Optional[str]]:
    row = frame.iloc[index]
    if index - entry_index >= int(spec.max_hold_bars):
        return float(row["open"]), "max_hold"

    stop_distance = theta * max(float(spec.stop_multiple), 0.0)
    take_distance = theta * max(float(spec.take_profit_multiple), 0.0)
    if position > 0:
        stop_price = entry_price * (1.0 - stop_distance)
        take_price = entry_price * (1.0 + take_distance)
        if float(row["low"]) <= stop_price:
            return float(stop_price), "stop"
        if float(row["high"]) >= take_price:
            return float(take_price), "take_profit"
    else:
        stop_price = entry_price * (1.0 + stop_distance)
        take_price = entry_price * (1.0 - take_distance)
        if float(row["high"]) >= stop_price:
            return float(stop_price), "stop"
        if float(row["low"]) <= take_price:
            return float(take_price), "take_profit"
    return None, None


def _close_trade(
    trades: List[Dict[str, Any]],
    equity: float,
    spec: DCFuturesStrategySpec,
    cost_model: FuturesCostModel,
    position: int,
    entry_price: float,
    entry_index: int,
    exit_index: int,
    exit_price: float,
    exit_reason: str,
    entry_event: Optional[DCEvent],
    frame: pd.DataFrame,
) -> float:
    gross_return = (exit_price / max(entry_price, 1e-12) - 1.0) * float(position)
    net_return = gross_return - cost_model.round_trip_cost_bps / 10_000.0
    trades.append(
        {
            "symbol": spec.symbol,
            "side": "long" if position > 0 else "short",
            "entry_index": int(entry_index),
            "exit_index": int(exit_index),
            "entry_timestamp": str(frame.iloc[entry_index]["timestamp"]),
            "exit_timestamp": str(frame.iloc[exit_index]["timestamp"]),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "gross_return": float(gross_return),
            "net_return": float(net_return),
            "exit_reason": exit_reason,
            "signal_kind": entry_event.kind if entry_event else "",
            "signal_index": int(entry_event.confirmation_index) if entry_event else -1,
            "entry_lag_bars": int(entry_index - entry_event.confirmation_index) if entry_event else -1,
        }
    )
    return float(equity * (1.0 + net_return))


def _compound_trades_with_cost(
    trades: Sequence[Mapping[str, Any]],
    round_trip_cost_bps: float,
    randomize_direction: bool,
    rng: Optional[np.random.Generator] = None,
) -> float:
    equity = 1.0
    cost = max(_as_float(round_trip_cost_bps), 0.0) / 10_000.0
    for trade in trades:
        entry_price = max(_as_float(trade.get("entry_price")), 1e-12)
        exit_price = _as_float(trade.get("exit_price"))
        raw_return = exit_price / entry_price - 1.0
        if randomize_direction:
            if rng is None:
                raise ValueError("rng is required when randomize_direction=True")
            position = 1.0 if int(rng.integers(0, 2)) == 1 else -1.0
        else:
            position = 1.0 if str(trade.get("side")) == "long" else -1.0
        net_return = raw_return * position - cost
        equity *= max(1.0 + net_return, 1e-9)
    return float(equity - 1.0)


def _position_for_event(event: DCEvent, family: str) -> int:
    if family in {"dc_continuation", "dc_overshoot_continuation"}:
        return 1 if event.kind == "DC_UP" else -1
    if family in {"dc_reversal", "dc_overshoot_reversal"}:
        return -1 if event.kind == "DC_UP" else 1
    raise ValueError(f"unknown DC strategy family: {family}")


def _is_overshoot_family(family: str) -> bool:
    return str(family) in {"dc_overshoot_continuation", "dc_overshoot_reversal"}


def _overshoot_entry_ready(
    frame: pd.DataFrame,
    index: int,
    event: DCEvent,
    spec: DCFuturesStrategySpec,
    theta: float,
) -> bool:
    if index <= event.confirmation_index or index >= len(frame):
        return False
    trigger = max(float(spec.overshoot_trigger_multiple), 0.0) * max(theta, 0.0)
    prior_close = _as_float(frame.iloc[index - 1].get("close"))
    confirmation_price = max(float(event.confirmation_price), 1e-12)
    if event.kind == "DC_UP":
        return bool(prior_close >= confirmation_price * (1.0 + trigger))
    if event.kind == "DC_DOWN":
        return bool(prior_close <= confirmation_price * (1.0 - trigger))
    return False


def _entry_allowed(frame: pd.DataFrame, index: int, spec: DCFuturesStrategySpec) -> bool:
    vol_percentile = _as_float(frame.iloc[index].get("_vol_percentile"))
    if spec.vol_filter in {"mid", "mid_40_80"} and not 0.40 <= vol_percentile <= 0.80:
        return False
    if spec.vol_filter in {"high", "high_70_plus"} and vol_percentile < 0.70:
        return False
    if spec.open_interest_filter == "rising" and _as_float(frame.iloc[index].get("_hold_diff")) <= 0:
        return False
    return _time_allowed(frame.iloc[index]["timestamp"], spec.time_filter)


def _time_allowed(timestamp: Any, time_filter: str) -> bool:
    if time_filter == "all":
        return True
    ts = pd.Timestamp(timestamp)
    minutes = ts.hour * 60 + ts.minute
    day = 9 * 60 <= minutes <= 15 * 60
    night = minutes >= 21 * 60 or minutes <= 2 * 60 + 30
    if time_filter == "day":
        return bool(day)
    if time_filter == "night":
        return bool(night)
    if time_filter == "open30":
        return bool(9 * 60 <= minutes <= 9 * 60 + 30 or 21 * 60 <= minutes <= 21 * 60 + 30)
    if time_filter == "close30":
        return bool(14 * 60 + 30 <= minutes <= 15 * 60 or 2 * 60 <= minutes <= 2 * 60 + 30)
    return True


def _walk_forward_bounds(length: int, folds: int = 3) -> List[Tuple[int, int, int, int]]:
    if length < 30:
        return [(0, max(int(length * 0.6), 1), length, length)]
    effective_folds = max(int(folds), 1)
    bounds: List[Tuple[int, int, int, int]] = []
    train_start = 0
    for fold_index in range(effective_folds):
        train_end = int(length * (0.40 + 0.20 * fold_index))
        test_end = int(length * (0.60 + 0.20 * fold_index))
        if fold_index == effective_folds - 1:
            test_end = length
        train_end = min(max(train_end, 2), length - 1)
        test_end = min(max(test_end, train_end + 1), length)
        if train_end >= test_end:
            continue
        bounds.append((fold_index + 1, train_start, train_end, test_end))
    return bounds


def _main_contracts_for_products(products: Sequence[str], ak: Any) -> List[str]:
    exchange_symbols = {
        "CFFEX": "cffex",
        "SHFE": "shfe",
        "DCE": "dce",
        "CZCE": "czce",
        "GFEX": "gfex",
        "INE": "shfe",
    }
    product_order = [product for product in products if product]
    by_exchange: Dict[str, set[str]] = {}
    for product in products:
        spec = get_futures_contract_spec(product)
        if spec is None:
            continue
        exchange_symbol = exchange_symbols.get(spec.exchange)
        if exchange_symbol:
            by_exchange.setdefault(exchange_symbol, set()).add(product)

    contract_by_product: Dict[str, str] = {}
    for exchange_symbol, exchange_products in by_exchange.items():
        try:
            raw = ak.match_main_contract(symbol=exchange_symbol)
        except Exception:
            continue
        if isinstance(raw, str):
            candidates = [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]
        elif isinstance(raw, pd.DataFrame):
            candidates = [str(value).strip() for value in raw.to_numpy().ravel().tolist() if str(value).strip()]
        else:
            candidates = [str(value).strip() for value in list(raw or []) if str(value).strip()]
        for contract in candidates:
            product = normalize_futures_symbol(contract)
            if product in exchange_products and product not in contract_by_product:
                contract_by_product[product] = contract
    return [contract_by_product[product] for product in product_order if product in contract_by_product]


def _rolling_percentile(values: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    for index, value in enumerate(values):
        start = max(0, index - window + 1)
        sample = values[start : index + 1]
        sample = sample[np.isfinite(sample)]
        if len(sample) == 0:
            out[index] = 0.0
            continue
        out[index] = float(np.mean(sample <= value))
    return out


def _stable_seed(payload: Mapping[str, Any]) -> int:
    text = json.dumps(_json_safe(dict(payload)), ensure_ascii=True, sort_keys=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default
