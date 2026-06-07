from __future__ import annotations

import json
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
) -> Dict[str, Any]:
    cost_model = cost_model or FuturesCostModel()
    working = normalize_minute_frame(frame)
    fold_rows: List[Dict[str, Any]] = []
    for fold_index, train_start, train_end, test_end in _walk_forward_bounds(len(working), folds=folds):
        train_frame = working.iloc[train_start:train_end].copy()
        test_frame = working.iloc[train_end:test_end].copy()
        train = simulate_dc_strategy(train_frame, spec, cost_model=cost_model)
        test = simulate_dc_strategy(test_frame, spec, cost_model=cost_model)
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
    return classify_walk_forward_result(spec, cost_model, fold_rows)


def classify_walk_forward_result(
    spec: DCFuturesStrategySpec,
    cost_model: FuturesCostModel,
    folds: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    test_returns = [_as_float(row.get("test_return")) for row in folds]
    train_returns = [_as_float(row.get("train_return")) for row in folds]
    captures = [_as_float(row.get("test_capture_ratio")) for row in folds]
    test_events = [int(_as_float(row.get("test_events"))) for row in folds]
    test_trades = [int(_as_float(row.get("test_trades"))) for row in folds]
    positive_folds = sum(1 for value in test_returns if value > 0)
    passing_fold_events = [
        int(_as_float(row.get("test_events")))
        for row in folds
        if _as_float(row.get("test_return")) > 0 and 0.05 <= _as_float(row.get("test_capture_ratio")) <= 0.20
    ]
    avg_train_return = float(np.mean(train_returns)) if train_returns else 0.0
    avg_test_return = float(np.mean(test_returns)) if test_returns else 0.0
    avg_test_capture = float(np.mean(captures)) if captures else 0.0
    min_passing_fold_events = min(passing_fold_events) if passing_fold_events else 0
    total_test_trades = int(sum(test_trades))

    flags: List[str] = []
    if float(spec.theta_bps) <= cost_model.round_trip_cost_bps:
        flags.append("THETA_BELOW_ROUND_TRIP_COST")
    if avg_test_return <= 0:
        flags.append("NON_POSITIVE_EXPECTANCY")
    if not 0.05 <= avg_test_capture <= 0.20:
        flags.append("CAPTURE_OUT_OF_TARGET")
    if min(test_events or [0]) < 20:
        flags.append("LOW_EVENT_COUNT")
    if total_test_trades < 30:
        flags.append("LOW_TRADE_COUNT")
    if avg_train_return <= 0 < avg_test_return:
        flags.append("TRAIN_TEST_DIVERGENCE")
    if positive_folds >= 2 and (avg_test_return <= 0 or not 0.05 <= avg_test_capture <= 0.20):
        flags.append("UNSTABLE_FOLD_EDGE")

    strict_pass = (
        positive_folds >= 2
        and avg_test_return > 0
        and 0.05 <= avg_test_capture <= 0.20
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
            "positive_test_folds": positive_folds,
            "min_test_events": min(test_events or [0]),
            "min_passing_fold_events": min_passing_fold_events,
            "total_test_trades": total_test_trades,
        },
        "cost_model": asdict(cost_model) | {"round_trip_cost_bps": cost_model.round_trip_cost_bps},
        "gate": "PASS requires positive OOS expectancy, 5%-20% capture, sufficient events/trades, and no bias flags.",
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
                                results.append(walk_forward_evaluate(normalized, spec, cost_model=cost_model, folds=folds))
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
        "- Gate: OOS net positive, capture_ratio in 5%-20%, event/trade sufficiency, no bias flags.",
        "- Trading rule: DC confirmation is tradable only from the next bar open.",
        "",
        "## Top Candidates",
        "",
        "| status | symbol | family | theta_bps | filters | avg_test_return | avg_capture | trades | reasons |",
        "|---|---:|---|---:|---|---:|---:|---:|---|",
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
            "| {status} | {symbol} | {family} | {theta:.1f} | {filters} | {ret:.4%} | {capture:.2%} | {trades} | {reasons} |".format(
                status=row.get("status", ""),
                symbol=spec.get("symbol", ""),
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
            "PASS means the fragment survived the strict research gate. WATCH means the path fragment is interesting but still fails at least one gate, usually event count, train/test stability, or target capture band. FAIL means it is not evidence of a repeatable edge.",
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
