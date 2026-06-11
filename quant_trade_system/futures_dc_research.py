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
DEFAULT_STRATEGY_FAMILIES = ("dc_continuation", "dc_reversal")
DEFAULT_VOL_FILTERS = ("all", "mid_40_80", "high_70_plus")
DEFAULT_OPEN_INTEREST_FILTERS = ("all", "rising")
DEFAULT_TIME_FILTERS = ("all", "day", "night", "open30", "close30")
DEFAULT_EVENT_SPACING_BARS = (0, 2, 4, 6)


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
    entry_events: Dict[int, DCEvent] = {
        event.confirmation_index + 1: event
        for event in events
        if event.confirmation_index + 1 < len(working)
    }
    theta = float(spec.theta_bps) / 10_000.0
    position = 0
    entry_price = 0.0
    entry_index = -1
    entry_event: Optional[DCEvent] = None
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
                                    return rank_research_results(results)
    return rank_research_results(results)


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
        key = (
            product,
            spec.get("family"),
            float(_as_float(spec.get("theta_bps"))),
            spec.get("vol_filter", "all"),
            spec.get("open_interest_filter", "all"),
            spec.get("time_filter", "all"),
            int(_as_float(spec.get("event_spacing_bars"))),
        )
        groups[key].append(row)

    aggregated: List[Dict[str, Any]] = []
    for key, members in groups.items():
        if len(members) < int(min_members):
            continue
        product, family, theta_bps, vol_filter, open_interest_filter, time_filter, spacing = key
        member_symbols = sorted(
            {
                str(member.get("spec", {}).get("symbol"))
                for member in members
                if isinstance(member.get("spec"), Mapping)
            }
        )
        folds: List[Dict[str, Any]] = []
        for member in members:
            symbol = str(member.get("spec", {}).get("symbol", ""))
            for fold in member.get("folds", []):
                if not isinstance(fold, Mapping):
                    continue
                fold_row = dict(fold)
                fold_row["source_symbol"] = symbol
                fold_row["fold"] = f"{symbol}:{fold.get('fold', '')}"
                folds.append(fold_row)

        aggregate_spec = DCFuturesStrategySpec(
            symbol=f"{product}_AGG",
            family=str(family),
            theta_bps=float(theta_bps),
            vol_filter=str(vol_filter),
            open_interest_filter=str(open_interest_filter),
            time_filter=str(time_filter),
            event_spacing_bars=int(spacing),
        )
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
        extra_flags: List[str] = []
        if member_positive < len(members):
            extra_flags.append("CROSS_CONTRACT_MEMBER_EXPECTANCY_DIVERGENCE")
        if member_target_capture < len(members):
            extra_flags.append("CROSS_CONTRACT_CAPTURE_DIVERGENCE")
        if extra_flags:
            aggregate["failure_reasons"] = list(dict.fromkeys(list(aggregate.get("failure_reasons", [])) + extra_flags))
            aggregate["status"] = "WATCH" if aggregate["summary"]["avg_test_return"] > 0 else "FAIL"
        aggregate["aggregation"] = {
            "mode": "product_contract_rollup",
            "product": product,
            "member_count": len(members),
            "members": member_symbols,
            "member_positive_count": member_positive,
            "member_target_capture_count": member_target_capture,
        }
        aggregated.append(aggregate)
    return rank_research_results(aggregated)


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


def build_research_report(results: Sequence[Mapping[str, Any]], generated_at: Optional[datetime] = None) -> str:
    generated_at = generated_at or datetime.now()
    ranked = rank_research_results(results)
    pass_count = sum(1 for row in ranked if row.get("status") == "PASS")
    watch_count = sum(1 for row in ranked if row.get("status") == "WATCH")
    lines = [
        "# Futures DC Capture Research Report",
        "",
        f"- Generated at: {generated_at.isoformat(timespec='seconds')}",
        f"- Candidates scanned: {len(ranked)}",
        f"- PASS: {pass_count}",
        f"- WATCH: {watch_count}",
        "- Gate: OOS net positive, average and median capture_ratio in 5%-20%, at least two target-capture folds, positive 1.5x-cost stress, random-direction control beaten, event/trade sufficiency, no bias flags.",
        "- Trading rule: DC confirmation is tradable only from the next bar open.",
        "",
        "## Top Candidates",
        "",
        "| status | symbol | family | theta_bps | filters | avg_test_return | avg_capture | stress_1_5x | rand_p | trades | reasons |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|",
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
            ]
        )
        reasons = ",".join(str(reason) for reason in row.get("failure_reasons", []))
        lines.append(
            "| {status} | {symbol} | {family} | {theta:.1f} | {filters} | {ret:.4%} | {capture:.2%} | {stress:.4%} | {rand_p:.3f} | {trades} | {reasons} |".format(
                status=row.get("status", ""),
                symbol=spec.get("symbol", ""),
                family=spec.get("family", ""),
                theta=_as_float(spec.get("theta_bps")),
                filters=filters,
                ret=_as_float(summary.get("avg_test_return")),
                capture=_as_float(summary.get("avg_test_capture_ratio")),
                stress=_as_float(summary.get("cost_1_5x_avg_return")),
                rand_p=_as_float(summary.get("random_control_p_value"), 1.0),
                trades=int(_as_float(summary.get("total_test_trades"))),
                reasons=reasons or "-",
            )
        )
    aggregates = aggregate_research_results(ranked)
    lines.extend(
        [
            "",
            "## Cross-Contract Aggregates",
            "",
            "| status | product | members | family | theta_bps | filters | avg_test_return | avg_capture | trades | reasons |",
            "|---|---:|---:|---|---:|---|---:|---:|---:|---|",
        ]
    )
    if not aggregates:
        lines.append("| - | - | 0 | - | 0.0 | - | 0.0000% | 0.00% | 0 | no multi-contract product groups |")
    for row in aggregates[:15]:
        spec = row.get("spec", {}) if isinstance(row.get("spec"), Mapping) else {}
        summary = row.get("summary", {}) if isinstance(row.get("summary"), Mapping) else {}
        aggregation = row.get("aggregation", {}) if isinstance(row.get("aggregation"), Mapping) else {}
        filters = ",".join(
            [
                str(spec.get("vol_filter", "all")),
                str(spec.get("open_interest_filter", "all")),
                str(spec.get("time_filter", "all")),
                f"spacing={spec.get('event_spacing_bars', 0)}",
            ]
        )
        reasons = ",".join(str(reason) for reason in row.get("failure_reasons", []))
        lines.append(
            "| {status} | {product} | {members} | {family} | {theta:.1f} | {filters} | {ret:.4%} | {capture:.2%} | {trades} | {reasons} |".format(
                status=row.get("status", ""),
                product=aggregation.get("product", spec.get("symbol", "")),
                members=int(_as_float(aggregation.get("member_count"))),
                family=spec.get("family", ""),
                theta=_as_float(spec.get("theta_bps")),
                filters=filters,
                ret=_as_float(summary.get("avg_test_return")),
                capture=_as_float(summary.get("avg_test_capture_ratio")),
                trades=int(_as_float(summary.get("total_test_trades"))),
                reasons=reasons or "-",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "PASS means the fragment survived the strict research gate. WATCH means the path fragment is interesting but still fails at least one gate, usually event count, train/test stability, target capture band, cost stress, random-direction control, or cross-contract consistency. FAIL means it is not evidence of a repeatable edge.",
            "",
        ]
    )
    return "\n".join(lines)


def write_research_outputs(
    results: Sequence[Mapping[str, Any]],
    state_dir: str | Path = "state",
    report_path: str | Path = "FUTURES_DC_CAPTURE_REPORT.md",
) -> Dict[str, str]:
    ranked = rank_research_results(results)
    state_path = Path(state_dir)
    state_path.mkdir(parents=True, exist_ok=True)
    json_path = state_path / "futures_dc_capture_candidates.json"
    json_path.write_text(json.dumps(_json_safe(ranked), ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_research_report(ranked)
    markdown_path = Path(report_path)
    markdown_path.write_text(report, encoding="utf-8")
    return {"json_path": str(json_path), "report_path": str(markdown_path)}


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
    if family == "dc_continuation":
        return 1 if event.kind == "DC_UP" else -1
    if family == "dc_reversal":
        return -1 if event.kind == "DC_UP" else 1
    raise ValueError(f"unknown DC strategy family: {family}")


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
