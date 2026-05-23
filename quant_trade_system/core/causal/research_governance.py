"""Research governance primitives for causal quant iteration.

The classes in this module are intentionally lightweight and dependency-poor.
They give the trading pipeline auditable hooks for:

1. causal identification status;
2. feature lineage and leakage checks;
3. experiment records;
4. model promotion / demotion records.

They are not a replacement for a production metadata store. They are the
in-repo contract that makes every nightly decision traceable before a heavier
database-backed registry is introduced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


IDENTIFIABLE = "identifiable"
WEAK_IDENTIFIABLE = "weak_identifiable"
CORRELATION_ONLY = "correlation_only"
UNAVAILABLE = "unavailable"


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:14]
    return f"{prefix}_{digest}"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if not np.isfinite(result):
            return default
        return result
    except Exception:
        return default


@dataclass
class CausalEdgeValidationSnapshot:
    """Compact validation status for a factor/edge used by production code."""

    edge_id: str
    source: str
    target: str
    identification_status: str
    validation_score: float
    p_value: float
    effect_size: float
    stability_score: float
    oos_score: float
    observation_count: int
    can_trade: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)


@dataclass
class FeatureRecord:
    feature_id: str
    name: str
    financial_meaning: str
    formula: str
    data_lineage: Dict[str, Any]
    available_timestamp: str
    leakage_check: Dict[str, Any]
    validation_status: Dict[str, Any]
    created_at: str = field(default_factory=_utc_now)


@dataclass
class ExperimentRecord:
    experiment_id: str
    name: str
    data_version: Dict[str, Any]
    training_window: Dict[str, Any]
    test_window: Dict[str, Any]
    metrics: Dict[str, Any]
    failure_reasons: List[str]
    status: str
    created_at: str = field(default_factory=_utc_now)


@dataclass
class ModelRecord:
    model_id: str
    name: str
    version: str
    training_window: Dict[str, Any]
    validation_summary: Dict[str, Any]
    promotion_status: str
    promotion_reason: str
    created_at: str = field(default_factory=_utc_now)


class CausalValidationLoop:
    """Small causal-identification gate for factors before they can trade.

    This deliberately uses transparent diagnostics instead of a black-box score:
    in-sample explanatory power, split stability, out-of-sample correlation and
    a simple placebo-style p-value proxy. It is conservative enough for the
    current project contract: unvalidated narratives may be reported, but they
    should not directly increase position size.
    """

    def __init__(self, min_observations: int = 60) -> None:
        self.min_observations = int(min_observations)

    def validate_feature(
        self,
        feature_name: str,
        factor_series: pd.Series,
        target_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        adjustment_frame: Optional[pd.DataFrame] = None,
        backdoor_adjustment: Optional[Mapping[str, Any]] = None,
        discovery_support: float = 0.0,
    ) -> CausalEdgeValidationSnapshot:
        aligned = pd.concat(
            [
                pd.to_numeric(factor_series, errors="coerce").rename("source"),
                pd.to_numeric(target_returns, errors="coerce").rename("target"),
            ],
            axis=1,
        ).dropna()
        if benchmark_returns is not None:
            benchmark = pd.Series(benchmark_returns).reset_index(drop=True)
            benchmark = pd.to_numeric(benchmark, errors="coerce")
            if len(benchmark) >= len(aligned):
                benchmark = benchmark.iloc[-len(aligned):].reset_index(drop=True)
            benchmark.index = aligned.index
            aligned["benchmark"] = benchmark
            aligned = aligned.dropna()
        if adjustment_frame is not None and not adjustment_frame.empty:
            controls = adjustment_frame.copy().reset_index(drop=True)
            controls = controls.apply(pd.to_numeric, errors="coerce")
            if len(controls) >= len(aligned):
                controls = controls.iloc[-len(aligned):].reset_index(drop=True)
            controls.index = aligned.index
            controls = controls.loc[:, controls.std(numeric_only=True) > 1e-10]
            controls = controls.add_prefix("adjust_")
            aligned = aligned.join(controls, how="left").dropna()

        edge_id = f"{feature_name}_to_forward_return"
        if len(aligned) < self.min_observations:
            return CausalEdgeValidationSnapshot(
                edge_id=edge_id,
                source=feature_name,
                target="forward_return",
                identification_status=UNAVAILABLE,
                validation_score=0.0,
                p_value=1.0,
                effect_size=0.0,
                stability_score=0.0,
                oos_score=0.0,
                observation_count=int(len(aligned)),
                can_trade=False,
                diagnostics={"reason": "insufficient_observations", "min_observations": self.min_observations},
            )

        source = aligned["source"].astype(float)
        target = aligned["target"].astype(float)
        if float(source.std()) < 1e-10 or float(target.std()) < 1e-10:
            return CausalEdgeValidationSnapshot(
                edge_id=edge_id,
                source=feature_name,
                target="forward_return",
                identification_status=UNAVAILABLE,
                validation_score=0.0,
                p_value=1.0,
                effect_size=0.0,
                stability_score=0.0,
                oos_score=0.0,
                observation_count=int(len(aligned)),
                can_trade=False,
                diagnostics={"reason": "near_constant_series"},
            )

        r2, slope, corr = self._incremental_regression(aligned)
        split = max(self.min_observations // 2, len(aligned) // 2)
        first_corr = self._safe_corr(aligned["source"].iloc[:split], aligned["target"].iloc[:split])
        second_corr = self._safe_corr(aligned["source"].iloc[split:], aligned["target"].iloc[split:])
        stability = self._stability_score(first_corr, second_corr)
        oos_score = abs(second_corr)
        p_value = self._p_value_proxy(corr, len(aligned))
        effect_size = abs(slope) * float(source.std())

        adjustment_quality = _to_float((backdoor_adjustment or {}).get("adjustment_quality"), 0.0)
        validation_score = float(
            np.clip(
                0.28 * min(r2 / 0.70, 1.0)
                + 0.20 * min(abs(corr) / 0.50, 1.0)
                + 0.22 * stability
                + 0.12 * min(oos_score / 0.30, 1.0)
                + 0.10 * min(max(float(discovery_support), 0.0), 1.0)
                + 0.08 * min(adjustment_quality, 1.0),
                0.0,
                1.0,
            )
        )
        backdoor_ok = adjustment_quality >= 0.55 or benchmark_returns is not None
        if p_value <= 0.05 and r2 >= 0.15 and stability >= 0.55 and oos_score >= 0.05 and backdoor_ok:
            status = IDENTIFIABLE
        elif p_value <= 0.10 and r2 >= 0.05 and stability >= 0.35 and (backdoor_ok or discovery_support >= 0.10):
            status = WEAK_IDENTIFIABLE
        elif abs(corr) >= 0.05:
            status = CORRELATION_ONLY
        else:
            status = UNAVAILABLE

        return CausalEdgeValidationSnapshot(
            edge_id=edge_id,
            source=feature_name,
            target="forward_return",
            identification_status=status,
            validation_score=round(validation_score, 6),
            p_value=round(float(p_value), 6),
            effect_size=round(float(effect_size), 6),
            stability_score=round(float(stability), 6),
            oos_score=round(float(oos_score), 6),
            observation_count=int(len(aligned)),
            can_trade=status in {IDENTIFIABLE, WEAK_IDENTIFIABLE},
            diagnostics={
                "r_squared": round(float(r2), 6),
                "correlation": round(float(corr), 6),
                "slope": round(float(slope), 6),
                "first_half_correlation": round(float(first_corr), 6),
                "second_half_correlation": round(float(second_corr), 6),
                "method": "backdoor_adjusted_split_stability_incremental_regression",
                "discovery_support": round(float(discovery_support), 6),
                "backdoor_adjustment": dict(backdoor_adjustment or {}),
                "adjustment_columns": [column for column in aligned.columns if column.startswith("adjust_")],
            },
        )

    @staticmethod
    def _safe_corr(left: pd.Series, right: pd.Series) -> float:
        left = pd.to_numeric(left, errors="coerce")
        right = pd.to_numeric(right, errors="coerce")
        aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
        if len(aligned) < 3 or float(aligned["left"].std()) < 1e-10 or float(aligned["right"].std()) < 1e-10:
            return 0.0
        corr = float(np.corrcoef(aligned["left"], aligned["right"])[0, 1])
        return corr if np.isfinite(corr) else 0.0

    @staticmethod
    def _incremental_regression(aligned: pd.DataFrame) -> tuple[float, float, float]:
        y = aligned["target"].to_numpy(dtype=float)
        source = aligned["source"].to_numpy(dtype=float)
        control_columns = [column for column in aligned.columns if column not in {"source", "target"}]
        if control_columns:
            controls = aligned[control_columns].to_numpy(dtype=float)
            x = np.column_stack([np.ones(len(aligned)), controls, source])
            base = np.column_stack([np.ones(len(aligned)), controls])
            base_coef, *_ = np.linalg.lstsq(base, y, rcond=None)
            base_hat = base @ base_coef
            base_ss = float(np.sum((y - base_hat) ** 2))
        else:
            x = np.column_stack([np.ones(len(aligned)), source])
            base_ss = float(np.sum((y - y.mean()) ** 2))
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        y_hat = x @ coef
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if control_columns:
            r2 = max(0.0, (base_ss - ss_res) / max(ss_tot, 1e-12))
        else:
            r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
        corr = CausalValidationLoop._safe_corr(aligned["source"], aligned["target"])
        return max(0.0, float(r2)), float(coef[-1]), corr

    @staticmethod
    def _stability_score(first_corr: float, second_corr: float) -> float:
        if first_corr == 0.0 or second_corr == 0.0:
            return 0.0
        if np.sign(first_corr) != np.sign(second_corr):
            return 0.0
        ratio = min(abs(first_corr), abs(second_corr)) / max(abs(first_corr), abs(second_corr), 1e-9)
        return float(np.clip(0.35 + 0.65 * ratio, 0.0, 1.0))

    @staticmethod
    def _p_value_proxy(corr: float, nobs: int) -> float:
        if nobs <= 3 or abs(corr) >= 1:
            return 1.0 if nobs <= 3 else 0.0
        t_stat = abs(corr) * np.sqrt((nobs - 2) / max(1.0 - corr * corr, 1e-9))
        return float(np.clip(np.exp(-0.72 * t_stat), 0.0, 1.0))


class FeatureStore:
    """In-memory feature registry with optional JSONL persistence."""

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else None
        self.records: Dict[str, FeatureRecord] = {}
        if self.base_dir:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def register_feature(
        self,
        name: str,
        financial_meaning: str,
        formula: str,
        data_lineage: Mapping[str, Any],
        leakage_check: Mapping[str, Any],
        validation_status: Mapping[str, Any],
        available_timestamp: Optional[str] = None,
    ) -> FeatureRecord:
        payload = {
            "name": name,
            "formula": formula,
            "financial_meaning": financial_meaning,
            "data_lineage": dict(data_lineage),
        }
        record = FeatureRecord(
            feature_id=_stable_id("feature", payload),
            name=name,
            financial_meaning=financial_meaning,
            formula=formula,
            data_lineage=dict(data_lineage),
            available_timestamp=available_timestamp or _utc_now(),
            leakage_check=dict(leakage_check),
            validation_status=dict(validation_status),
        )
        self.records[name] = record
        self._append_jsonl("features.jsonl", asdict(record))
        return record

    def latest_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [asdict(record) for record in list(self.records.values())[-limit:]]

    def _append_jsonl(self, filename: str, payload: Mapping[str, Any]) -> None:
        if not self.base_dir:
            return
        path = self.base_dir / filename
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class ExperimentRegistry:
    """Append-only experiment registry."""

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else None
        self.records: List[ExperimentRecord] = []
        if self.base_dir:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def record_experiment(
        self,
        name: str,
        data_version: Mapping[str, Any],
        training_window: Mapping[str, Any],
        test_window: Mapping[str, Any],
        metrics: Mapping[str, Any],
        failure_reasons: Optional[Iterable[str]] = None,
        status: str = "completed",
    ) -> ExperimentRecord:
        payload = {
            "name": name,
            "data_version": dict(data_version),
            "training_window": dict(training_window),
            "test_window": dict(test_window),
            "metrics": dict(metrics),
            "status": status,
        }
        record = ExperimentRecord(
            experiment_id=_stable_id("experiment", payload),
            name=name,
            data_version=dict(data_version),
            training_window=dict(training_window),
            test_window=dict(test_window),
            metrics=dict(metrics),
            failure_reasons=list(failure_reasons or []),
            status=status,
        )
        self.records.append(record)
        self._append_jsonl("experiments.jsonl", asdict(record))
        return record

    def latest_records(self, limit: int = 20) -> List[Dict[str, Any]]:
        return [asdict(record) for record in self.records[-limit:]]

    def _append_jsonl(self, filename: str, payload: Mapping[str, Any]) -> None:
        if not self.base_dir:
            return
        path = self.base_dir / filename
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class ModelRegistry:
    """Track candidate models and whether they may enter nightly production."""

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else None
        self.records: List[ModelRecord] = []
        if self.base_dir:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        name: str,
        training_window: Mapping[str, Any],
        validation_summary: Mapping[str, Any],
        promotion_status: str,
        promotion_reason: str,
    ) -> ModelRecord:
        version_payload = {
            "name": name,
            "training_window": dict(training_window),
            "validation_summary": dict(validation_summary),
        }
        version = hashlib.sha1(json.dumps(version_payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
        record = ModelRecord(
            model_id=f"model_{version}",
            name=name,
            version=version,
            training_window=dict(training_window),
            validation_summary=dict(validation_summary),
            promotion_status=promotion_status,
            promotion_reason=promotion_reason,
        )
        self.records.append(record)
        self._append_jsonl("models.jsonl", asdict(record))
        return record

    def latest_records(self, limit: int = 20) -> List[Dict[str, Any]]:
        return [asdict(record) for record in self.records[-limit:]]

    def _append_jsonl(self, filename: str, payload: Mapping[str, Any]) -> None:
        if not self.base_dir:
            return
        path = self.base_dir / filename
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
