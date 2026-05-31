"""Robustness controls for complex quant-system upgrades.

This module keeps expensive or overfit-prone research ideas behind auditable,
lightweight gates:

* CPCV-style purged/embargoed validation paths for time-series returns;
* Deflated Sharpe Ratio to penalize multiple trials and non-normal returns;
* effective breadth diagnostics so nominal signal count cannot fake diversity;
* Lowdin/democratic orthogonalization to remove collinearity while preserving
  the closest possible representation of the original factors.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _normal_ppf(probability: float) -> float:
    """Acklam inverse-normal approximation without scipy dependency."""

    p = float(np.clip(probability, 1e-12, 1.0 - 1e-12))
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return float(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return float(
            -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return float(
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def _returns_stats(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "observation_count": 0,
            "mean_return": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "skew": 0.0,
            "kurtosis": 3.0,
            "max_drawdown": 0.0,
        }
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    sharpe = float(math.sqrt(periods_per_year) * mean / std) if std > 1e-12 else 0.0
    centered = clean - mean
    skew = float((centered.pow(3).mean()) / max(float(centered.pow(2).mean()) ** 1.5, 1e-12))
    kurtosis = float((centered.pow(4).mean()) / max(float(centered.pow(2).mean()) ** 2, 1e-12))
    equity = (1.0 + clean).cumprod()
    max_drawdown = float((equity / equity.cummax() - 1.0).min()) if not equity.empty else 0.0
    return {
        "observation_count": int(len(clean)),
        "mean_return": mean,
        "volatility": std,
        "sharpe": sharpe,
        "skew": skew,
        "kurtosis": kurtosis,
        "max_drawdown": max_drawdown,
    }


@dataclass
class CPCVConfig:
    n_groups: int = 6
    test_group_count: int = 2
    purge_window: int = 5
    embargo_pct: float = 0.01
    max_paths: int = 30


class CombinatorialPurgedCV:
    """Generate purged/embargoed train-test splits for financial series."""

    def __init__(self, config: CPCVConfig | None = None) -> None:
        self.config = config or CPCVConfig()

    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = int(max(n_samples, 0))
        if n <= 0:
            return
        groups = np.array_split(np.arange(n), max(2, int(self.config.n_groups)))
        group_ids = range(len(groups))
        combos = list(itertools.combinations(group_ids, min(int(self.config.test_group_count), len(groups))))
        if self.config.max_paths > 0:
            combos = combos[: int(self.config.max_paths)]
        embargo = int(math.ceil(n * max(float(self.config.embargo_pct), 0.0)))
        purge = max(int(self.config.purge_window), 0)
        all_idx = np.arange(n)
        for combo in combos:
            test_idx = np.sort(np.concatenate([groups[i] for i in combo if len(groups[i])]))
            if test_idx.size == 0:
                continue
            blocked = np.zeros(n, dtype=bool)
            blocked[test_idx] = True
            for idx in test_idx:
                start = max(0, int(idx) - purge)
                end = min(n, int(idx) + purge + embargo + 1)
                blocked[start:end] = True
            train_idx = all_idx[~blocked]
            if train_idx.size == 0:
                continue
            yield train_idx, test_idx


def evaluate_cpcv_returns(
    returns: pd.Series,
    *,
    n_groups: int = 6,
    test_group_count: int = 2,
    purge_window: int = 5,
    embargo_pct: float = 0.01,
    max_paths: int = 30,
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(clean) < max(30, n_groups * 3):
        return {
            "status": "insufficient_observations",
            "path_count": 0,
            "observation_count": int(len(clean)),
            "gate": "CPCV requires enough time-series observations before promotion.",
        }
    splitter = CombinatorialPurgedCV(
        CPCVConfig(
            n_groups=n_groups,
            test_group_count=test_group_count,
            purge_window=purge_window,
            embargo_pct=embargo_pct,
            max_paths=max_paths,
        )
    )
    path_stats = []
    for train_idx, test_idx in splitter.split(len(clean)):
        test_returns = clean.iloc[test_idx]
        stats = _returns_stats(test_returns, periods_per_year=periods_per_year)
        path_stats.append(
            {
                "train_count": int(len(train_idx)),
                "test_count": int(len(test_idx)),
                "sharpe": round(float(stats["sharpe"]), 6),
                "max_drawdown": round(float(stats["max_drawdown"]), 6),
                "total_return": round(float((1.0 + test_returns).prod() - 1.0), 6),
            }
        )
    if not path_stats:
        return {
            "status": "no_valid_paths",
            "path_count": 0,
            "observation_count": int(len(clean)),
        }
    sharpes = np.array([item["sharpe"] for item in path_stats], dtype=float)
    drawdowns = np.array([item["max_drawdown"] for item in path_stats], dtype=float)
    return {
        "status": "ready",
        "path_count": int(len(path_stats)),
        "observation_count": int(len(clean)),
        "median_path_sharpe": round(float(np.median(sharpes)), 6),
        "p10_path_sharpe": round(float(np.percentile(sharpes, 10)), 6),
        "min_path_sharpe": round(float(np.min(sharpes)), 6),
        "positive_path_rate": round(float(np.mean(sharpes > 0.0)), 6),
        "worst_path_drawdown": round(float(np.min(drawdowns)), 6),
        "passed": bool(np.percentile(sharpes, 10) > 0.0 and np.mean(sharpes > 0.0) >= 0.60),
        "paths": path_stats[:10],
        "gate": "Promotion requires positive CPCV lower-tail Sharpe and broad path pass-rate.",
    }


def deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    effective_trials: int = 1,
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    stats = _returns_stats(clean, periods_per_year=periods_per_year)
    n = int(stats["observation_count"])
    if n < 30:
        return {
            **{key: round(float(value), 6) if isinstance(value, float) else value for key, value in stats.items()},
            "status": "insufficient_observations",
            "effective_trials": int(max(effective_trials, 1)),
            "dsr_probability": 0.0,
            "passed_dsr_95": False,
        }
    trials = max(int(effective_trials), 1)
    sharpe = float(stats["sharpe"])
    skew = float(stats["skew"])
    kurtosis = max(float(stats["kurtosis"]), 1.0)
    sr_std = math.sqrt(max(1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe, 1e-9) / max(n - 1, 1))
    gamma = 0.5772156649
    if trials <= 1:
        benchmark_sr = 0.0
    else:
        benchmark_sr = sr_std * (
            (1.0 - gamma) * _normal_ppf(1.0 - 1.0 / trials)
            + gamma * _normal_ppf(1.0 - 1.0 / (math.e * trials))
        )
    z_score = (sharpe - benchmark_sr) / max(sr_std, 1e-12)
    probability = _normal_cdf(z_score)
    return {
        **{key: round(float(value), 6) if isinstance(value, float) else value for key, value in stats.items()},
        "status": "ready",
        "effective_trials": int(trials),
        "benchmark_sharpe_after_deflation": round(float(benchmark_sr), 6),
        "dsr_z_score": round(float(z_score), 6),
        "dsr_probability": round(float(probability), 6),
        "passed_dsr_95": bool(probability >= 0.95),
        "gate": "DSR penalizes multiple trials, skew and fat tails; production promotion target >=0.95.",
    }


def effective_breadth(frame: pd.DataFrame | Sequence[Sequence[float]] | Sequence[float]) -> Dict[str, Any]:
    data = pd.DataFrame(frame).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=1, how="all").dropna(axis=0, how="all")
    n = int(data.shape[1])
    if n <= 1:
        return {
            "nominal_breadth": n,
            "average_abs_pairwise_correlation": 0.0,
            "effective_breadth": float(n),
            "breadth_ratio": 1.0 if n else 0.0,
            "status": "single_or_empty",
        }
    corr = data.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0).abs().to_numpy(dtype=float)
    mask = ~np.eye(n, dtype=bool)
    avg_corr = float(np.clip(corr[mask].mean() if mask.any() else 0.0, 0.0, 1.0))
    breadth = float(n / (1.0 + avg_corr * (n - 1)))
    return {
        "nominal_breadth": n,
        "average_abs_pairwise_correlation": round(avg_corr, 6),
        "effective_breadth": round(breadth, 6),
        "breadth_ratio": round(float(breadth / max(n, 1)), 6),
        "status": "ready",
        "formula": "N / (1 + avg_abs_pairwise_corr * (N - 1))",
    }


def democratic_orthogonalize(frame: pd.DataFrame, *, ridge: float = 1e-8) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    numeric = pd.DataFrame(frame).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all").ffill().fillna(0.0)
    if numeric.empty:
        return pd.DataFrame(index=getattr(frame, "index", None)), {
            "status": "empty",
            "method": "lowdin_symmetric_democratic_orthogonalization",
        }
    centered = numeric - numeric.mean()
    scaled = centered / numeric.std(ddof=0).replace(0.0, 1.0)
    x = scaled.to_numpy(dtype=float)
    covariance = (x.T @ x) / max(len(scaled) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, float(ridge), None)
    inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    orth = x @ inv_sqrt
    columns = [f"dem_orth_{column}" for column in numeric.columns]
    orth_frame = pd.DataFrame(orth, index=numeric.index, columns=columns)
    before = effective_breadth(scaled)
    after = effective_breadth(orth_frame)
    corr = orth_frame.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)] if corr.size else np.array([0.0])
    frobenius_distance = float(np.linalg.norm(orth - x, ord="fro") / max(math.sqrt(x.size), 1e-9))
    diagnostics = {
        "status": "ready",
        "method": "lowdin_symmetric_democratic_orthogonalization",
        "input_factor_count": int(numeric.shape[1]),
        "observation_count": int(numeric.shape[0]),
        "max_abs_offdiag_corr": round(float(np.max(np.abs(off_diag))) if off_diag.size else 0.0, 6),
        "frobenius_distance_per_element": round(frobenius_distance, 6),
        "effective_breadth_before": before,
        "effective_breadth_after": after,
        "eigenvalue_min": round(float(np.min(eigenvalues)), 6),
        "eigenvalue_max": round(float(np.max(eigenvalues)), 6),
        "gate": "Use orthogonal factors only if marginal breadth gain survives costs and validation.",
    }
    return orth_frame, diagnostics


def shapley_deployment_policy(
    *,
    model_family: str,
    feature_count: int,
    frequency: str = "weekly",
) -> Dict[str, Any]:
    """Audit whether Shapley attribution belongs in hot path or offline review."""

    family = str(model_family or "").lower()
    feature_count = int(max(feature_count, 0))
    if any(token in family for token in ["tree", "lightgbm", "xgboost", "random_forest"]):
        method = "TreeSHAP"
        hot_path_allowed = False
        reason = "TreeSHAP is efficient, but attribution still belongs in offline weekly/monthly model governance."
    elif feature_count <= 20 and frequency in {"weekly", "monthly", "offline"}:
        method = "KernelSHAP_small_sample"
        hot_path_allowed = False
        reason = "KernelSHAP is only acceptable for small offline diagnostics because correlated financial factors bias sampling."
    else:
        method = "disabled"
        hot_path_allowed = False
        reason = "Exact/Kernel Shapley is too expensive and unstable for broad nightly or intraday trading paths."
    return {
        "method": method,
        "model_family": model_family,
        "feature_count": feature_count,
        "frequency": frequency,
        "hot_path_allowed": hot_path_allowed,
        "recommended_use": "offline_factor_decay_diagnostics",
        "reason": reason,
    }


def robustness_summary_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Round dataclass/numpy-heavy payloads for JSON-friendly audit records."""

    def convert(value: Any) -> Any:
        if hasattr(value, "__dataclass_fields__"):
            return convert(asdict(value))
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, (np.floating, float)):
            return round(float(value), 6)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        return value

    return convert(payload)
