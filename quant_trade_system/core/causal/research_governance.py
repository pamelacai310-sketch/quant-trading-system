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

ALLOW = "ALLOW"
REDUCE = "REDUCE"
OBSERVE_ONLY = "OBSERVE_ONLY"
NO_TRADE = "NO_TRADE"


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


@dataclass
class CausalAbstentionDecision:
    """Unified trade eligibility decision for a causal signal."""

    decision: str
    weight_multiplier: float
    risk_score: float
    reasons: List[str]
    inputs: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)


@dataclass
class InstrumentRecord:
    """Auditable instrument-variable diagnostics for one causal edge."""

    edge_id: str
    source: str
    target: str
    instruments: List[str]
    first_stage_strength: float
    exclusion_proxy: float
    weak_instrument: bool
    validity_status: str
    exclusion_restriction_assumption: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)


@dataclass
class LLMHypothesisAuditRecord:
    """Causal LLM output captured as an audit-only research hypothesis."""

    hypothesis_id: str
    treatment: str
    outcome: str
    controls: List[str]
    instrument: str
    expected_path: List[str]
    source: str
    validation_required: List[str]
    actionability: str = "audit_only"
    created_at: str = field(default_factory=_utc_now)


class CausalAbstentionGate:
    """Convert causal evidence quality into ALLOW/REDUCE/OBSERVE/NO_TRADE.

    The gate is intentionally deterministic and conservative. It does not
    discover signals; it only decides whether an existing signal may size up,
    must be haircut, or must remain an audit-only observation.
    """

    def evaluate(
        self,
        *,
        identification_status: str,
        backdoor_quality: float,
        hmm_state_entropy: float,
        data_validation_passed: bool,
        model_disagreement: float,
        counterfactual_tail_risk: float,
        instrument_status: str = "unavailable",
    ) -> CausalAbstentionDecision:
        status = str(identification_status or UNAVAILABLE)
        iv_status = str(instrument_status or "unavailable")
        backdoor = float(np.clip(_to_float(backdoor_quality), 0.0, 1.0))
        entropy = float(np.clip(_to_float(hmm_state_entropy, 1.0), 0.0, 1.0))
        disagreement = float(np.clip(_to_float(model_disagreement), 0.0, 1.0))
        tail_risk = float(np.clip(_to_float(counterfactual_tail_risk), 0.0, 1.0))
        reasons: List[str] = []

        if not data_validation_passed:
            return CausalAbstentionDecision(
                decision=NO_TRADE,
                weight_multiplier=0.0,
                risk_score=1.0,
                reasons=["data_validation_failed"],
                inputs={
                    "identification_status": status,
                    "backdoor_quality": backdoor,
                    "hmm_state_entropy": entropy,
                    "model_disagreement": disagreement,
                    "counterfactual_tail_risk": tail_risk,
                    "instrument_status": iv_status,
                },
            )

        status_risk = {
            IDENTIFIABLE: 0.05,
            WEAK_IDENTIFIABLE: 0.25,
            CORRELATION_ONLY: 0.70,
            UNAVAILABLE: 0.85,
        }.get(status, 0.85)
        iv_risk = {
            "valid": -0.08,
            "weak": 0.30,
            "invalid": 0.45,
            "unavailable": 0.0,
        }.get(iv_status, 0.0)
        risk_score = float(
            np.clip(
                0.25 * status_risk
                + 0.18 * (1.0 - backdoor)
                + 0.20 * entropy
                + 0.17 * disagreement
                + 0.20 * tail_risk
                + iv_risk,
                0.0,
                1.0,
            )
        )

        if status in {CORRELATION_ONLY, UNAVAILABLE}:
            reasons.append(f"identification_status={status}")
        if backdoor < 0.55:
            reasons.append(f"backdoor_quality={backdoor:.2f}<0.55")
        if entropy >= 0.85:
            reasons.append(f"hmm_state_entropy={entropy:.2f}>=0.85")
        if disagreement >= 0.50:
            reasons.append(f"model_disagreement={disagreement:.2f}>=0.50")
        if tail_risk >= 0.55:
            reasons.append(f"counterfactual_tail_risk={tail_risk:.2f}>=0.55")
        if iv_status in {"weak", "invalid"}:
            reasons.append(f"instrument_status={iv_status}")

        if status in {CORRELATION_ONLY, UNAVAILABLE} or iv_status in {"weak", "invalid"}:
            decision = OBSERVE_ONLY if risk_score < 0.85 else NO_TRADE
            weight_multiplier = 0.0
        elif risk_score >= 0.75:
            decision = NO_TRADE
            weight_multiplier = 0.0
        elif risk_score >= 0.45 or status == WEAK_IDENTIFIABLE:
            decision = REDUCE
            weight_multiplier = float(np.clip(1.0 - risk_score, 0.15, 0.65))
        else:
            decision = ALLOW
            weight_multiplier = float(np.clip(1.0 - 0.35 * risk_score, 0.65, 1.10))

        if not reasons:
            reasons.append("causal_evidence_passed")
        return CausalAbstentionDecision(
            decision=decision,
            weight_multiplier=round(float(weight_multiplier), 6),
            risk_score=round(float(risk_score), 6),
            reasons=reasons,
            inputs={
                "identification_status": status,
                "backdoor_quality": round(backdoor, 6),
                "hmm_state_entropy": round(entropy, 6),
                "data_validation_passed": bool(data_validation_passed),
                "model_disagreement": round(disagreement, 6),
                "counterfactual_tail_risk": round(tail_risk, 6),
                "instrument_status": iv_status,
            },
        )


class InstrumentRegistry:
    """Lightweight IV/Deep-IV readiness registry for causal edges."""

    INSTRUMENT_TOKENS = (
        "iv_",
        "instrument",
        "policy",
        "weather",
        "supply",
        "inventory",
        "opec",
        "margin",
        "exchange_rule",
        "global_peer",
        "shipping",
        "tariff",
        "dxy",
        "usd",
        "rate",
    )

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else None
        self.records: Dict[str, InstrumentRecord] = {}
        if self.base_dir:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def diagnose_edge(
        self,
        source: str,
        target: str,
        factor_matrix: pd.DataFrame,
        target_returns: pd.Series,
        candidate_instruments: Optional[Iterable[str]] = None,
    ) -> InstrumentRecord:
        edge_id = f"{source}_to_{target}"
        instruments = list(candidate_instruments or self._candidate_instruments(source, factor_matrix.columns))
        instruments = [name for name in instruments if name in factor_matrix.columns and name != source]
        if not instruments or source not in factor_matrix.columns:
            return self.register_instrument(
                source=source,
                target=target,
                instruments=[],
                first_stage_strength=0.0,
                exclusion_proxy=1.0,
                weak_instrument=False,
                validity_status="unavailable",
                diagnostics={"reason": "no_candidate_instrument"},
            )

        aligned = pd.concat(
            [
                pd.to_numeric(factor_matrix[source], errors="coerce").rename("source"),
                pd.to_numeric(target_returns, errors="coerce").rename("target"),
                factor_matrix[instruments].apply(pd.to_numeric, errors="coerce"),
            ],
            axis=1,
        ).dropna()
        if len(aligned) < 30:
            return self.register_instrument(
                source=source,
                target=target,
                instruments=instruments,
                first_stage_strength=0.0,
                exclusion_proxy=1.0,
                weak_instrument=True,
                validity_status="weak",
                diagnostics={"reason": "insufficient_iv_observations", "observation_count": int(len(aligned))},
            )

        scores = []
        for name in instruments:
            first_stage = abs(CausalValidationLoop._safe_corr(aligned[name], aligned["source"]))
            exclusion_proxy = abs(self._target_residual_corr(aligned[name], aligned["source"], aligned["target"]))
            scores.append((name, first_stage, exclusion_proxy))
        best = max(scores, key=lambda item: (item[1], -item[2]))
        weak = best[1] < 0.10
        if best[1] >= 0.15 and best[2] <= 0.35:
            status = "valid"
        elif weak:
            status = "weak"
        else:
            status = "invalid"
        return self.register_instrument(
            source=source,
            target=target,
            instruments=[best[0]],
            first_stage_strength=best[1],
            exclusion_proxy=best[2],
            weak_instrument=weak,
            validity_status=status,
            diagnostics={
                "observation_count": int(len(aligned)),
                "candidate_scores": [
                    {
                        "instrument": name,
                        "first_stage_strength": round(float(first), 6),
                        "exclusion_proxy": round(float(exclusion), 6),
                    }
                    for name, first, exclusion in scores
                ],
                "method": "lightweight_deep_iv_readiness_proxy",
            },
        )

    def register_instrument(
        self,
        *,
        source: str,
        target: str,
        instruments: Iterable[str],
        first_stage_strength: float,
        exclusion_proxy: float,
        weak_instrument: bool,
        validity_status: str,
        diagnostics: Optional[Mapping[str, Any]] = None,
        exclusion_restriction_assumption: str = (
            "instrument affects forward returns only through the treatment edge after observed controls"
        ),
    ) -> InstrumentRecord:
        record = InstrumentRecord(
            edge_id=f"{source}_to_{target}",
            source=source,
            target=target,
            instruments=list(instruments),
            first_stage_strength=round(float(first_stage_strength), 6),
            exclusion_proxy=round(float(exclusion_proxy), 6),
            weak_instrument=bool(weak_instrument),
            validity_status=str(validity_status),
            exclusion_restriction_assumption=exclusion_restriction_assumption,
            diagnostics=dict(diagnostics or {}),
        )
        self.records[record.edge_id] = record
        self._append_jsonl("instruments.jsonl", asdict(record))
        return record

    def latest_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [asdict(record) for record in list(self.records.values())[-limit:]]

    def _candidate_instruments(self, source: str, columns: Iterable[str]) -> List[str]:
        candidates = []
        source_lower = source.lower()
        for column in columns:
            lower = str(column).lower()
            if column == source:
                continue
            if any(token in lower for token in self.INSTRUMENT_TOKENS):
                candidates.append(str(column))
            elif "global_peer" in source_lower and lower.startswith("base_"):
                candidates.append(str(column))
        return candidates[:8]

    @staticmethod
    def _target_residual_corr(instrument: pd.Series, source: pd.Series, target: pd.Series) -> float:
        aligned = pd.concat(
            [instrument.rename("instrument"), source.rename("source"), target.rename("target")],
            axis=1,
        ).dropna()
        if len(aligned) < 5:
            return 1.0
        x = np.column_stack([np.ones(len(aligned)), aligned["source"].to_numpy(dtype=float)])
        y = aligned["target"].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = pd.Series(y - x @ coef, index=aligned.index)
        return CausalValidationLoop._safe_corr(aligned["instrument"], residual)

    def _append_jsonl(self, filename: str, payload: Mapping[str, Any]) -> None:
        if not self.base_dir:
            return
        path = self.base_dir / filename
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class CausalLLMAuditor:
    """Keep LLM-generated causal hypotheses audit-only until validated."""

    REQUIRED_FIELDS = ("treatment", "outcome", "expected_path")

    def audit_hypotheses(self, payload: Any) -> List[LLMHypothesisAuditRecord]:
        records = []
        for item in self._records_from_any(payload):
            item = self._normalize_hypothesis_record(item)
            if not all(item.get(field) for field in self.REQUIRED_FIELDS):
                continue
            source = str(item.get("source") or "causal_llm")
            treatment = str(item.get("treatment"))
            outcome = str(item.get("outcome"))
            expected_path = item.get("expected_path")
            if isinstance(expected_path, str):
                path = [part.strip() for part in expected_path.replace("→", ">").split(">") if part.strip()]
            else:
                path = [str(part) for part in expected_path]
            controls = item.get("controls") or []
            if isinstance(controls, str):
                controls = [part.strip() for part in controls.split(",") if part.strip()]
            record = LLMHypothesisAuditRecord(
                hypothesis_id=_stable_id(
                    "llm_hypothesis",
                    {"treatment": treatment, "outcome": outcome, "expected_path": path},
                ),
                treatment=treatment,
                outcome=outcome,
                controls=[str(part) for part in controls],
                instrument=str(item.get("instrument") or ""),
                expected_path=path,
                source=source,
                validation_required=["price_confirmation", "scm_dag", "backdoor_or_iv_validation"],
                actionability="audit_only",
            )
            records.append(record)
        return records

    @staticmethod
    def _normalize_hypothesis_record(item: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = dict(item)
        if all(normalized.get(field) for field in CausalLLMAuditor.REQUIRED_FIELDS):
            return normalized
        text = str(
            normalized.get("title")
            or normalized.get("headline")
            or normalized.get("summary")
            or normalized.get("content")
            or normalized.get("text")
            or ""
        ).strip()
        if not text:
            return normalized
        normalized.setdefault("treatment", text[:120])
        normalized.setdefault("outcome", normalized.get("target") or "market_forward_return")
        normalized.setdefault("controls", normalized.get("controls") or ["market_regime", "liquidity", "benchmark_return"])
        normalized.setdefault("instrument", normalized.get("instrument") or "")
        normalized.setdefault("expected_path", [text[:120], "expectation_change", str(normalized["outcome"])])
        normalized.setdefault("source", normalized.get("source") or "news_or_policy_record")
        return normalized

    @staticmethod
    def _records_from_any(payload: Any) -> List[Mapping[str, Any]]:
        if payload is None:
            return []
        if isinstance(payload, Mapping):
            if "hypotheses" in payload:
                return CausalLLMAuditor._records_from_any(payload.get("hypotheses"))
            return [payload]
        if isinstance(payload, pd.DataFrame):
            return [row.dropna().to_dict() for _, row in payload.iterrows()]
        if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
            return [item for item in payload if isinstance(item, Mapping)]
        return []


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
