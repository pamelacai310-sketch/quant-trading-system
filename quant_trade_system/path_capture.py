from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DCPivot:
    index: int
    timestamp: Any
    price: float
    kind: str


@dataclass(frozen=True)
class DCEvent:
    kind: str
    confirmation_index: int
    confirmation_timestamp: Any
    confirmation_price: float
    extreme_index: int
    extreme_timestamp: Any
    extreme_price: float
    threshold_bps: float


@dataclass(frozen=True)
class DCSegment:
    start_index: int
    end_index: int
    start_timestamp: Any
    end_timestamp: Any
    direction: str
    start_price: float
    end_price: float
    gross_return: float
    net_upper_return: float


def capture_ratio(strategy_return: float, dc_path_return: float) -> float:
    """Return the share of an ex-post DC path benchmark captured by a strategy."""

    benchmark = _finite_float(dc_path_return)
    if benchmark <= 0:
        return 0.0
    return _finite_float(strategy_return) / benchmark


def directional_change_events(
    frame: pd.DataFrame | pd.Series | Sequence[float],
    theta_bps: float,
    price_col: str = "close",
) -> List[DCEvent]:
    events, _ = _directional_change_path(frame, theta_bps, price_col=price_col, include_open_segment=False)
    return events


def directional_change_segments(
    frame: pd.DataFrame | pd.Series | Sequence[float],
    theta_bps: float,
    round_trip_cost_bps: float = 0.0,
    risk_buffer_bps: float = 0.0,
    price_col: str = "close",
    include_open_segment: bool = False,
) -> List[DCSegment]:
    _, pivots = _directional_change_path(
        frame,
        theta_bps,
        price_col=price_col,
        include_open_segment=include_open_segment,
    )
    cost_threshold = max((_finite_float(round_trip_cost_bps) + _finite_float(risk_buffer_bps)) / 10_000.0, 0.0)
    segments: List[DCSegment] = []
    for start, end in zip(pivots, pivots[1:]):
        if start.price <= 0 or end.price <= 0 or start.price == end.price:
            continue
        direction = "long" if end.price > start.price else "short"
        gross_return = abs(end.price / start.price - 1.0)
        segments.append(
            DCSegment(
                start_index=start.index,
                end_index=end.index,
                start_timestamp=start.timestamp,
                end_timestamp=end.timestamp,
                direction=direction,
                start_price=float(start.price),
                end_price=float(end.price),
                gross_return=float(gross_return),
                net_upper_return=float(max(gross_return - cost_threshold, 0.0)),
            )
        )
    return segments


def dc_path_summary(
    frame: pd.DataFrame | pd.Series | Sequence[float],
    theta_bps: float,
    round_trip_cost_bps: float = 0.0,
    risk_buffer_bps: float = 0.0,
    price_col: str = "close",
    include_open_segment: bool = False,
    strategy_return: Optional[float] = None,
) -> dict[str, Any]:
    events = directional_change_events(frame, theta_bps, price_col=price_col)
    segments = directional_change_segments(
        frame,
        theta_bps,
        round_trip_cost_bps=round_trip_cost_bps,
        risk_buffer_bps=risk_buffer_bps,
        price_col=price_col,
        include_open_segment=include_open_segment,
    )
    gross_path_return = float(sum(segment.gross_return for segment in segments))
    dc_path_return = float(sum(segment.net_upper_return for segment in segments))
    payload: dict[str, Any] = {
        "theta_bps": float(theta_bps),
        "round_trip_cost_bps": float(round_trip_cost_bps),
        "risk_buffer_bps": float(risk_buffer_bps),
        "event_count": len(events),
        "segment_count": len(segments),
        "gross_dc_path_return": gross_path_return,
        "dc_path_return": dc_path_return,
        "events": [asdict(event) for event in events],
        "segments": [asdict(segment) for segment in segments],
    }
    if strategy_return is not None:
        payload["strategy_return"] = float(strategy_return)
        payload["capture_ratio"] = capture_ratio(float(strategy_return), dc_path_return)
    return payload


def capture_target_status(
    ratio: float,
    lower: float = 0.05,
    upper: float = 0.20,
) -> str:
    value = _finite_float(ratio)
    if value < lower:
        return "BELOW_TARGET"
    if value > upper:
        return "ABOVE_TARGET"
    return "IN_TARGET"


def _directional_change_path(
    frame: pd.DataFrame | pd.Series | Sequence[float],
    theta_bps: float,
    price_col: str = "close",
    include_open_segment: bool = False,
) -> Tuple[List[DCEvent], List[DCPivot]]:
    theta = _finite_float(theta_bps) / 10_000.0
    if theta <= 0:
        raise ValueError("theta_bps must be positive")

    timestamps, prices = _extract_price_path(frame, price_col=price_col)
    if len(prices) < 2:
        return [], []

    events: List[DCEvent] = []
    pivots: List[DCPivot] = []
    mode: Optional[str] = None
    high_price = low_price = float(prices[0])
    high_index = low_index = 0

    for index in range(1, len(prices)):
        price = float(prices[index])
        timestamp = timestamps[index]

        if mode is None:
            if price > high_price:
                high_price = price
                high_index = index
            if price < low_price:
                low_price = price
                low_index = index
            if price >= low_price * (1.0 + theta):
                _append_pivot(pivots, DCPivot(low_index, timestamps[low_index], low_price, "low"))
                events.append(
                    DCEvent(
                        kind="DC_UP",
                        confirmation_index=index,
                        confirmation_timestamp=timestamp,
                        confirmation_price=price,
                        extreme_index=low_index,
                        extreme_timestamp=timestamps[low_index],
                        extreme_price=low_price,
                        threshold_bps=float(theta_bps),
                    )
                )
                mode = "up"
                high_price = price
                high_index = index
            elif price <= high_price * (1.0 - theta):
                _append_pivot(pivots, DCPivot(high_index, timestamps[high_index], high_price, "high"))
                events.append(
                    DCEvent(
                        kind="DC_DOWN",
                        confirmation_index=index,
                        confirmation_timestamp=timestamp,
                        confirmation_price=price,
                        extreme_index=high_index,
                        extreme_timestamp=timestamps[high_index],
                        extreme_price=high_price,
                        threshold_bps=float(theta_bps),
                    )
                )
                mode = "down"
                low_price = price
                low_index = index
            continue

        if mode == "up":
            if price > high_price:
                high_price = price
                high_index = index
            elif price <= high_price * (1.0 - theta):
                _append_pivot(pivots, DCPivot(high_index, timestamps[high_index], high_price, "high"))
                events.append(
                    DCEvent(
                        kind="DC_DOWN",
                        confirmation_index=index,
                        confirmation_timestamp=timestamp,
                        confirmation_price=price,
                        extreme_index=high_index,
                        extreme_timestamp=timestamps[high_index],
                        extreme_price=high_price,
                        threshold_bps=float(theta_bps),
                    )
                )
                mode = "down"
                low_price = price
                low_index = index
        else:
            if price < low_price:
                low_price = price
                low_index = index
            elif price >= low_price * (1.0 + theta):
                _append_pivot(pivots, DCPivot(low_index, timestamps[low_index], low_price, "low"))
                events.append(
                    DCEvent(
                        kind="DC_UP",
                        confirmation_index=index,
                        confirmation_timestamp=timestamp,
                        confirmation_price=price,
                        extreme_index=low_index,
                        extreme_timestamp=timestamps[low_index],
                        extreme_price=low_price,
                        threshold_bps=float(theta_bps),
                    )
                )
                mode = "up"
                high_price = price
                high_index = index

    if include_open_segment and mode == "up":
        _append_pivot(pivots, DCPivot(high_index, timestamps[high_index], high_price, "high"))
    elif include_open_segment and mode == "down":
        _append_pivot(pivots, DCPivot(low_index, timestamps[low_index], low_price, "low"))

    return events, pivots


def _append_pivot(pivots: List[DCPivot], pivot: DCPivot) -> None:
    if pivots and pivots[-1].index == pivot.index:
        return
    pivots.append(pivot)


def _extract_price_path(
    frame: pd.DataFrame | pd.Series | Sequence[float],
    price_col: str = "close",
) -> Tuple[List[Any], np.ndarray]:
    if isinstance(frame, pd.DataFrame):
        working = frame.copy()
        if price_col not in working.columns:
            raise ValueError(f"missing price column: {price_col}")
        if "timestamp" in working.columns:
            timestamps: Iterable[Any] = working["timestamp"].tolist()
        elif "datetime" in working.columns:
            timestamps = working["datetime"].tolist()
        else:
            timestamps = working.index.tolist()
        prices = pd.to_numeric(working[price_col], errors="coerce").to_numpy(dtype=float)
        timestamp_list = list(timestamps)
    elif isinstance(frame, pd.Series):
        prices = pd.to_numeric(frame, errors="coerce").to_numpy(dtype=float)
        timestamp_list = frame.index.tolist()
    else:
        prices = np.asarray(list(frame), dtype=float)
        timestamp_list = list(range(len(prices)))

    valid = np.isfinite(prices) & (prices > 0)
    return [timestamp for timestamp, keep in zip(timestamp_list, valid) if keep], prices[valid]


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default
