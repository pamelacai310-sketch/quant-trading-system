"""Event intensity indexing for narrative-to-signal conversion.

The classes in this module turn qualitative news and policy records into
auditable time-series features.  They intentionally do not fetch data by
themselves; callers pass already timestamped records so downstream training can
trace every value to the original event and avoid forward-looking leakage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if np.isfinite(value):
            return float(value)
        return default
    text = str(value).replace(",", "").replace("%", "").strip()
    if text in {"", "-", "--", "None", "nan"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _records_from_any(payload: Any) -> List[Mapping[str, Any]]:
    if payload is None:
        return []
    if hasattr(payload, "to_dict"):
        try:
            return payload.to_dict(orient="records")
        except TypeError:
            return list(payload.to_dict().values())
    if isinstance(payload, Mapping):
        if "records" in payload and isinstance(payload["records"], list):
            return [item for item in payload["records"] if isinstance(item, Mapping)]
        return [payload]
    if isinstance(payload, list):
        return [item if isinstance(item, Mapping) else {"title": str(item)} for item in payload]
    return [{"title": str(payload)}]


@dataclass(frozen=True)
class EventIntensitySpec:
    """Formula contract for one qualitative event domain."""

    factor_id: str
    description: str
    keyword_weights: Dict[str, float]
    asset_exposures: Dict[str, float]
    half_life_days: float
    rolling_window: int = 60
    momentum_lag: int = 5

    @property
    def decay_lambda(self) -> float:
        return float(np.log(2.0) / max(self.half_life_days, 1e-6))


@dataclass
class EventIntensitySnapshot:
    """Serializable audit output for an event intensity run."""

    status: str
    as_of: str
    event_count: int
    factor_columns: List[str]
    asset_exposure_columns: List[str]
    latest_values: Dict[str, float]
    factor_audit: Dict[str, Dict[str, Any]]
    feature_frame: pd.DataFrame = field(repr=False)

    def to_audit_dict(self, include_records: bool = False) -> Dict[str, Any]:
        payload = {
            "status": self.status,
            "as_of": self.as_of,
            "event_count": self.event_count,
            "factor_columns": list(self.factor_columns),
            "asset_exposure_columns": list(self.asset_exposure_columns),
            "latest_values": dict(self.latest_values),
            "factor_audit": self.factor_audit,
            "formula": (
                "EventIntensity=sum(relevance*sentiment*keyword_weight*asset_link*exp(-lambda*age)); "
                "EventZ uses rolling t-1 mean/std to prevent lookahead."
            ),
        }
        if include_records:
            records = self.feature_frame.reset_index().rename(columns={"index": "date"}).copy()
            if "date" in records.columns:
                records["date"] = pd.to_datetime(records["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            payload["feature_frame_records"] = records.to_dict(orient="records")
        return payload


class EventIntensityFactor:
    """Base event factor contract required by qualitative causal factors."""

    def __init__(self, spec: EventIntensitySpec) -> None:
        self.spec = spec

    def fetch_raw_inputs(self, records: Any) -> pd.DataFrame:
        """Normalize caller-supplied records into a timestamped event frame."""

        rows: List[Dict[str, Any]] = []
        for index, record in enumerate(_records_from_any(records)):
            title = self._first_text(record, ["title", "headline", "event", "事件", "内容", "name"])
            summary = self._first_text(record, ["summary", "content", "description", "摘要", "详情"]) or ""
            timestamp = self._first_text(record, ["timestamp", "datetime", "date", "time", "日期"])
            if not timestamp:
                timestamp = datetime.utcnow().isoformat()
            event_time = pd.to_datetime(timestamp, errors="coerce")
            if pd.isna(event_time):
                continue
            tags = record.get("tags") or record.get("tag") or record.get("分类") or []
            if isinstance(tags, str):
                tags_text = tags
            elif isinstance(tags, Iterable):
                tags_text = " ".join(str(item) for item in tags)
            else:
                tags_text = str(tags)
            text = " ".join([str(title or ""), str(summary), tags_text]).lower()
            rows.append(
                {
                    "event_id": str(record.get("event_id") or f"event_{index}"),
                    "date": event_time.normalize(),
                    "text": text,
                    "relevance": _clip(_to_float(record.get("relevance_score", record.get("relevance", 1.0)), 1.0), 0.0, 1.5),
                    "sentiment": _clip(_to_float(record.get("sentiment_score", record.get("sentiment", 0.0)), 0.0), -1.0, 1.0),
                }
            )
        if not rows:
            return pd.DataFrame(columns=["event_id", "date", "text", "relevance", "sentiment"])
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def compute_intensity(
        self,
        records: Any,
        calendar_index: Optional[Sequence[Any]] = None,
        as_of: Optional[Any] = None,
    ) -> pd.Series:
        """Compute decayed daily intensity for the configured event domain."""

        events = self.fetch_raw_inputs(records)
        index = self._calendar_index(events, calendar_index=calendar_index, as_of=as_of)
        intensity = pd.Series(0.0, index=index, dtype=float)
        if events.empty or intensity.empty:
            return intensity

        for current_date in intensity.index:
            eligible = events[events["date"] <= current_date]
            if eligible.empty:
                continue
            age_days = (current_date - eligible["date"]).dt.days.astype(float)
            term_scores = eligible["text"].apply(self._keyword_score)
            signed_sentiment = eligible["sentiment"].replace(0.0, 0.15)
            decayed = (
                eligible["relevance"].astype(float)
                * signed_sentiment.abs().astype(float)
                * term_scores.astype(float)
                * np.exp(-self.spec.decay_lambda * age_days)
            )
            intensity.loc[current_date] = float(decayed.sum())
        return intensity

    def rolling_normalize(self, intensity: pd.Series) -> pd.DataFrame:
        """Create leakage-safe z-score, momentum and decay-pressure features."""

        factor_id = self.spec.factor_id
        series = pd.to_numeric(intensity, errors="coerce").fillna(0.0).astype(float)
        shifted = series.shift(1)
        mean = shifted.rolling(self.spec.rolling_window, min_periods=max(5, self.spec.rolling_window // 6)).mean()
        std = shifted.rolling(self.spec.rolling_window, min_periods=max(5, self.spec.rolling_window // 6)).std()
        zscore = ((series - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        momentum = series.diff(self.spec.momentum_lag).fillna(0.0)
        rolling_max = shifted.rolling(self.spec.rolling_window, min_periods=5).max().replace(0.0, np.nan)
        decay_pressure = (series / rolling_max).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 3.0)

        return pd.DataFrame(
            {
                f"event_intensity_{factor_id}": series,
                f"event_zscore_{factor_id}": zscore,
                f"event_momentum_{factor_id}": momentum,
                f"event_decay_pressure_{factor_id}": decay_pressure,
            },
            index=series.index,
        )

    def fit_transform(
        self,
        records: Any,
        calendar_index: Optional[Sequence[Any]] = None,
        as_of: Optional[Any] = None,
    ) -> pd.DataFrame:
        intensity = self.compute_intensity(records, calendar_index=calendar_index, as_of=as_of)
        return self.rolling_normalize(intensity)

    def _keyword_score(self, text: str) -> float:
        hits = [weight for term, weight in self.spec.keyword_weights.items() if term.lower() in text]
        if not hits:
            return 0.0
        return _clip(0.70 * max(hits) + 0.30 * min(sum(hits) / 2.5, 1.0), 0.0, 1.5)

    @staticmethod
    def _first_text(record: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
        lower_map = {str(key).lower(): key for key in record.keys()}
        for key in keys:
            actual = lower_map.get(key.lower())
            if actual is None:
                continue
            value = record.get(actual)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    @staticmethod
    def _calendar_index(events: pd.DataFrame, calendar_index: Optional[Sequence[Any]], as_of: Optional[Any]) -> pd.DatetimeIndex:
        if calendar_index is not None:
            index = pd.to_datetime(pd.Index(calendar_index), errors="coerce")
            index = pd.DatetimeIndex(index.dropna()).normalize().unique().sort_values()
            return pd.DatetimeIndex(index)
        if events.empty:
            if as_of is None:
                return pd.DatetimeIndex([])
            stamp = pd.to_datetime(as_of, errors="coerce")
            return pd.DatetimeIndex([] if pd.isna(stamp) else [stamp.normalize()])
        end = pd.to_datetime(as_of, errors="coerce") if as_of is not None else events["date"].max()
        if pd.isna(end):
            end = events["date"].max()
        start = min(events["date"].min(), end)
        return pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")


class EventIntensityEngine:
    """Build event intensity matrices for the whole causal library."""

    DEFAULT_SPECS: Dict[str, EventIntensitySpec] = {
        "geopolitical_energy": EventIntensitySpec(
            factor_id="geopolitical_energy",
            description="War, sanctions, OPEC/cartel, Hormuz and shipping disruption risk.",
            half_life_days=7.0,
            keyword_weights={
                "war": 1.0,
                "iran": 0.9,
                "hormuz": 1.0,
                "sanction": 0.8,
                "opec": 0.8,
                "shipping": 0.7,
                "supply disruption": 1.0,
                "oil": 0.7,
                "geopolitical": 0.8,
            },
            asset_exposures={"SC": 1.0, "crude_oil": 1.0, "gold": 0.75, "shipping": 0.65, "copper": 0.35},
        ),
        "policy_risk": EventIntensitySpec(
            factor_id="policy_risk",
            description="Fiscal, regulatory, tariff and central-bank policy uncertainty.",
            half_life_days=30.0,
            keyword_weights={
                "policy": 0.7,
                "regulation": 0.8,
                "tariff": 0.9,
                "fiscal": 0.7,
                "subsidy": 0.6,
                "tax": 0.6,
                "rate hike": 0.8,
                "fed": 0.7,
                "central bank": 0.8,
            },
            asset_exposures={"equities": 0.85, "bonds": 0.70, "DXY": 0.55, "gold": 0.45, "industrial_metals": 0.40},
        ),
        "ai_capex": EventIntensitySpec(
            factor_id="ai_capex",
            description="AI capex, chips, data-center, power-grid and software monetization events.",
            half_life_days=18.0,
            keyword_weights={
                "ai": 1.0,
                "artificial intelligence": 1.0,
                "capex": 0.9,
                "data center": 0.9,
                "gpu": 0.8,
                "semiconductor": 0.8,
                "cloud": 0.7,
                "power grid": 0.8,
                "robot": 0.7,
            },
            asset_exposures={"Nasdaq": 0.85, "SOX": 1.0, "CSI1000": 0.75, "copper": 0.65, "power_equipment": 0.75},
        ),
        "trade_credit": EventIntensitySpec(
            factor_id="trade_credit",
            description="Trade diplomacy, tariffs, private credit, refinancing and margin pressure.",
            half_life_days=21.0,
            keyword_weights={
                "trade": 0.7,
                "tariff": 1.0,
                "export control": 0.9,
                "credit spread": 0.9,
                "private credit": 1.0,
                "default": 0.8,
                "refinancing": 0.7,
                "supply chain": 0.8,
            },
            asset_exposures={"credit": 1.0, "banks": 0.75, "copper": 0.55, "steel": 0.55, "agriculture": 0.45},
        ),
        "market_sentiment": EventIntensitySpec(
            factor_id="market_sentiment",
            description="Risk appetite, panic, positioning and broad market sentiment shocks.",
            half_life_days=10.0,
            keyword_weights={
                "risk appetite": 1.0,
                "panic": 1.0,
                "vix": 0.9,
                "selloff": 0.8,
                "short squeeze": 0.7,
                "fund inflow": 0.7,
                "risk-off": 1.0,
                "risk-on": 0.8,
            },
            asset_exposures={"VIX": 1.0, "equities": 0.75, "gold": 0.55, "credit": 0.60},
        ),
    }

    def __init__(self, specs: Optional[Mapping[str, EventIntensitySpec]] = None) -> None:
        self.specs = dict(specs or self.DEFAULT_SPECS)
        self.factors = {factor_id: EventIntensityFactor(spec) for factor_id, spec in self.specs.items()}

    def fit_transform(
        self,
        records: Any,
        calendar_index: Optional[Sequence[Any]] = None,
        as_of: Optional[Any] = None,
        include_records: bool = False,
    ) -> EventIntensitySnapshot:
        frames = []
        factor_audit: Dict[str, Dict[str, Any]] = {}
        for factor_id, factor in self.factors.items():
            frame = factor.fit_transform(records, calendar_index=calendar_index, as_of=as_of)
            frames.append(frame)
            factor_audit[factor_id] = {
                "description": factor.spec.description,
                "half_life_days": factor.spec.half_life_days,
                "rolling_window": factor.spec.rolling_window,
                "asset_exposures": dict(factor.spec.asset_exposures),
                "formula": "sum(relevance*abs(sentiment or 0.15)*keyword_weight*exp(-lambda*age_days))",
            }
        feature_frame = pd.concat(frames, axis=1) if frames else pd.DataFrame()
        asset_features = self._asset_exposure_features(feature_frame)
        if not asset_features.empty:
            feature_frame = pd.concat([feature_frame, asset_features], axis=1)
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        event_count = len(_records_from_any(records))
        latest = feature_frame.iloc[-1].to_dict() if not feature_frame.empty else {}
        snapshot = EventIntensitySnapshot(
            status="active" if event_count else "no_events",
            as_of=str(pd.to_datetime(as_of).date()) if as_of is not None and not pd.isna(pd.to_datetime(as_of, errors="coerce")) else "",
            event_count=event_count,
            factor_columns=[column for column in feature_frame.columns if column.startswith("event_") and "_asset_" not in column],
            asset_exposure_columns=[column for column in feature_frame.columns if column.startswith("event_asset_")],
            latest_values={name: round(float(value), 6) for name, value in latest.items()},
            factor_audit=factor_audit,
            feature_frame=feature_frame,
        )
        # Build the optional records once so tests and callers can request a
        # serializable matrix without mutating the snapshot.
        if include_records:
            snapshot.to_audit_dict(include_records=True)
        return snapshot

    def to_audit_dict(self, snapshot: EventIntensitySnapshot, include_records: bool = False) -> Dict[str, Any]:
        return snapshot.to_audit_dict(include_records=include_records)

    def _asset_exposure_features(self, factor_frame: pd.DataFrame) -> pd.DataFrame:
        if factor_frame.empty:
            return pd.DataFrame(index=factor_frame.index)
        asset_columns: Dict[str, pd.Series] = {}
        for factor_id, spec in self.specs.items():
            z_col = f"event_zscore_{factor_id}"
            pressure_col = f"event_decay_pressure_{factor_id}"
            if z_col not in factor_frame:
                continue
            base = factor_frame[z_col].abs() * 0.70 + factor_frame.get(pressure_col, 0.0) * 0.30
            for asset, weight in spec.asset_exposures.items():
                column = f"event_asset_{self._safe_name(asset)}_exposure"
                if column not in asset_columns:
                    asset_columns[column] = pd.Series(0.0, index=factor_frame.index)
                asset_columns[column] = asset_columns[column].add(base * float(weight), fill_value=0.0)
        return pd.DataFrame(asset_columns, index=factor_frame.index).clip(0.0, 5.0)

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")


def create_event_intensity_engine() -> EventIntensityEngine:
    return EventIntensityEngine()
