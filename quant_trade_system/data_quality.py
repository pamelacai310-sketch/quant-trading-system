from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class DataQualityAssessment:
    market: str
    score: float
    decision: str
    passed_validation: bool
    requested_date: Optional[str]
    actual_date: Optional[str]
    sample_count: int
    expected_sample_count: int
    date_lag_days: Optional[int]
    freshness_score: float
    coverage_score: float
    consistency_score: float
    source_score: float
    fallback_penalty: float
    reasons: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def assess_market_data_quality(
    validation: Any,
    expected_sample_count: int = 1,
) -> Dict[str, Any]:
    payload = validation if isinstance(validation, dict) else asdict(validation)
    market = str(payload.get("market") or "")
    passed = bool(payload.get("passed", False))
    requested_date = _text_or_none(payload.get("requested_date"))
    actual_date = _text_or_none(payload.get("actual_date"))
    sample_count = max(int(float(payload.get("sample_count", 0) or 0)), 0)
    expected = max(int(expected_sample_count or 1), 1)
    reason = str(payload.get("reason") or "")
    price_source = str(payload.get("price_source") or "")
    fallback = str(payload.get("settlement_fallback") or "")

    date_lag = _date_lag_days(requested_date, actual_date)
    freshness = _freshness_score(passed, date_lag, requested_date, actual_date)
    coverage = min(sample_count / expected, 1.0) if passed else 0.0
    fallback_used = bool(fallback) or any(token in reason for token in ["回退", "降级", "最近有效", "fallback"])
    consistency = 1.0 if passed and not fallback_used else (0.76 if passed else 0.0)
    source = _source_score(price_source, fallback_used)
    fallback_penalty = 0.08 if fallback_used else 0.0

    score = 0.35 * freshness + 0.25 * coverage + 0.25 * consistency + 0.15 * source - fallback_penalty
    if not passed:
        score = min(score, 0.30)
    score = round(float(max(0.0, min(1.0, score))), 6)
    assessment = DataQualityAssessment(
        market=market,
        score=score,
        decision=_quality_decision(passed, score),
        passed_validation=passed,
        requested_date=requested_date,
        actual_date=actual_date,
        sample_count=sample_count,
        expected_sample_count=expected,
        date_lag_days=date_lag,
        freshness_score=round(float(freshness), 6),
        coverage_score=round(float(coverage), 6),
        consistency_score=round(float(consistency), 6),
        source_score=round(float(source), 6),
        fallback_penalty=round(float(fallback_penalty), 6),
        reasons=_quality_reasons(passed, date_lag, coverage, fallback_used, price_source, reason),
    )
    return assessment.to_dict()


def summarize_data_quality(assessments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not assessments:
        return {
            "status": "missing",
            "min_score": 0.0,
            "avg_score": 0.0,
            "decision_counts": {},
            "weak_markets": [],
            "gate": "No market data quality assessments were available.",
        }
    scores = [float(item.get("score", 0.0) or 0.0) for item in assessments.values()]
    counts: Dict[str, int] = {}
    weak_markets = []
    for market, item in assessments.items():
        decision = str(item.get("decision") or "UNKNOWN")
        counts[decision] = counts.get(decision, 0) + 1
        if decision != "ALLOW":
            weak_markets.append(market)
    return {
        "status": "ok",
        "min_score": round(min(scores), 6),
        "avg_score": round(sum(scores) / len(scores), 6),
        "decision_counts": counts,
        "weak_markets": weak_markets,
        "gate": "Data quality is scored from freshness, sample coverage, source confidence, fallback use and validation consistency.",
    }


def _text_or_none(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return text or None


def _date_lag_days(requested_date: Optional[str], actual_date: Optional[str]) -> Optional[int]:
    if not requested_date or not actual_date:
        return None
    try:
        requested = datetime.strptime(requested_date[:10], "%Y-%m-%d").date()
        actual = datetime.strptime(actual_date[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    return max((requested - actual).days, 0)


def _freshness_score(
    passed: bool,
    date_lag: Optional[int],
    requested_date: Optional[str],
    actual_date: Optional[str],
) -> float:
    if not passed or not actual_date:
        return 0.0
    if requested_date and actual_date == requested_date:
        return 1.0
    if date_lag is None:
        return 0.70
    if date_lag <= 1:
        return 0.92
    if date_lag <= 4:
        return 0.78
    return max(0.0, 0.78 - 0.08 * (date_lag - 4))


def _source_score(price_source: str, fallback_used: bool) -> float:
    if not price_source:
        return 0.78 if not fallback_used else 0.68
    source_count = len([item for item in price_source.split("+") if item.strip()])
    if source_count >= 2:
        return 1.0 if not fallback_used else 0.88
    if "unknown" in price_source.lower():
        return 0.62
    return 0.86 if not fallback_used else 0.74


def _quality_decision(passed: bool, score: float) -> str:
    if not passed:
        return "NO_TRADE"
    if score >= 0.82:
        return "ALLOW"
    if score >= 0.66:
        return "REDUCE"
    return "OBSERVE_ONLY"


def _quality_reasons(
    passed: bool,
    date_lag: Optional[int],
    coverage: float,
    fallback_used: bool,
    price_source: str,
    validation_reason: str,
) -> list[str]:
    reasons: list[str] = []
    if not passed:
        reasons.append("validation_failed")
    if date_lag is None:
        reasons.append("date_lag_unavailable")
    elif date_lag > 0:
        reasons.append(f"date_lag_days={date_lag}")
    if coverage < 0.50:
        reasons.append("low_sample_coverage")
    elif coverage < 1.0:
        reasons.append("partial_sample_coverage")
    if fallback_used:
        reasons.append("fallback_or_recent_effective_date_used")
    if not price_source:
        reasons.append("price_source_missing")
    if validation_reason:
        reasons.append(validation_reason[:160])
    return reasons
