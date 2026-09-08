"""Point-in-time daily research with evidence-gated publication.

This module produces no broker orders. A retrospective simulation cannot confer
forward validation, and an abstention is not a zero-return directional forecast.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DailyPolicy:
    min_train: int = 60
    max_train: int = 120
    ridge_alpha: float = 10.0
    shrinkage: float = 0.5
    gate_days: int = 60
    block_days: int = 5
    family_candidates: int = 2
    error_alpha: float = 0.05
    calibration_days: int = 60
    min_calibration: int = 30
    interval_mass: float = 0.8


SECTORS = {
    **dict.fromkeys("CU AL ZN PB NI SN AO AD BC".split(), "base_metals"),
    **dict.fromkeys("AU AG PT PD".split(), "precious_metals"),
    **dict.fromkeys("RB HC SS I J JM SF SM".split(), "steel"),
    **dict.fromkeys("SC FU BU LU PG".split(), "energy"),
    **dict.fromkeys("L V PP EG EB BZ TA MA PF PX PR PL SH".split(), "chemicals"),
    **dict.fromkeys("B M Y P JD RR PK AP CJ".split(), "agriculture"),
    **dict.fromkeys("BR NR".split(), "rubber"),
    **dict.fromkeys("SP OP FB".split(), "paper_and_board"),
    **dict.fromkeys("FG SA".split(), "glass_chain"),
    **dict.fromkeys("SI PS".split(), "silicon_chain"),
    **dict.fromkeys("IF IH IC IM".split(), "equity"),
    **dict.fromkeys("T TF TS TL".split(), "bonds"),
}


def align_bars(rows: pd.DataFrame, calendar: Sequence[str], asof: str) -> pd.DataFrame:
    """No filling, invented sessions, or labels spanning a missing observation."""
    dates = list(calendar)
    if dates != sorted(set(dates)) or asof not in dates:
        raise ValueError("Calendar must be ordered, unique and contain asof")
    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    frame = frame.loc[frame.date <= asof].sort_values("date")
    if frame.empty or frame.date.duplicated().any():
        raise ValueError("Empty or duplicate contract observations")
    if set(frame.date) - set(dates):
        raise ValueError("Quote dates absent from the supplied exchange calendar")
    columns = ["open", "high", "low", "close", "volume"]
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="coerce")
    invalid = (~np.isfinite(frame[columns])).any(axis=1) | (frame[columns] <= 0).any(axis=1)
    invalid |= frame.high < frame[["open", "close", "low"]].max(axis=1)
    invalid |= frame.low > frame[["open", "close", "high"]].min(axis=1)
    frame.loc[invalid, columns] = np.nan
    selected = [day for day in dates if frame.date.iloc[0] <= day <= asof]
    return frame.set_index("date").reindex(selected)


def local_features(frame: pd.DataFrame) -> pd.DataFrame:
    c, v = frame.close, frame.volume
    return pd.DataFrame({
        "return_1d": c.pct_change(fill_method=None),
        "return_5d_div5": c.pct_change(5, fill_method=None) / 5,
        "return_20d_div20": c.pct_change(20, fill_method=None) / 20,
        "close_vs_ma20": c / c.rolling(20).mean() - 1,
        "volume_vs_ma20": v / v.rolling(20).mean() - 1,
        "intraday_return": c / frame.open - 1,
        "daily_range": (frame.high - frame.low) / c,
    }).replace([np.inf, -np.inf], np.nan)


def feature_panel(frames: Mapping[str, pd.DataFrame]) -> dict:
    returns = pd.DataFrame({s: f.close.pct_change(fill_method=None) for s, f in frames.items()})
    sectors = {s: SECTORS.get(re.sub(r"\d+$", "", s).upper(), "unknown") for s in frames}
    financial = {"equity", "bonds"}
    output = {}
    for symbol, frame in frames.items():
        base = local_features(frame)
        extended = base.copy()
        for level, peers, minimum in [
            ("market", [s for s in frames if s != symbol and
                        (sectors[s] in financial) == (sectors[symbol] in financial)], 10),
            ("sector", [s for s in frames if s != symbol and sectors[symbol] != "unknown"
                        and sectors[s] == sectors[symbol]], 3),
        ]:
            peer = returns.reindex(index=frame.index, columns=peers)
            count = peer.notna().sum(axis=1)
            daily = peer.mean(axis=1).where(count >= minimum)
            extended[level + "_return"] = daily
            extended[level + "_return5"] = daily.rolling(5).mean()
            extended[level + "_breadth"] = ((peer > 0).sum(axis=1) / count).where(count >= minimum)
            extended[level + "_dispersion"] = peer.std(axis=1).where(count >= minimum)
        output[symbol] = {"local": base, "context": extended.replace([np.inf, -np.inf], np.nan)}
    return output


def ridge(x, y, latest, policy: DailyPolicy):
    mean, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1)
    z = np.clip((x - mean) / sd, -5, 5)
    q = np.clip((latest - mean) / sd, -5, 5)
    center = z.mean(axis=0)
    beta = np.linalg.solve((z - center).T @ (z - center) + policy.ridge_alpha * np.eye(x.shape[1]),
                           (z - center).T @ (y - y.mean()))
    raw = float(y.mean() + (q - center) @ beta)
    return raw, float(raw * policy.shrinkage)


def evidence_gate(records: list[dict], asof: str, policy: DailyPolicy) -> dict:
    """Only already-realized forecasts enter the gate; correlated days form blocks.

    Bounds are approximate screening diagnostics, not formal live validation.
    """
    known = [r for r in records if r["target_date"] <= asof and np.isfinite(r["actual"])]
    if len({r["target_date"] for r in known}) != len(known):
        raise ValueError("Gate evidence must have one forecast per target date")
    known = sorted(known, key=lambda r: r["target_date"])[-policy.gate_days:]
    if len(known) < policy.gate_days:
        return {"passed": False, "reason": "insufficient_oos_days", "days": len(known)}
    actual = np.array([r["actual"] for r in known])
    pred = np.array([r["prediction"] for r in known])
    improvement = np.abs(actual) - np.abs(actual - pred)
    hit_advantage = ((np.sign(actual) == np.sign(pred)) & (actual != 0) & (pred != 0)).astype(float) - 0.5
    z = NormalDist().inv_cdf(1 - policy.error_alpha / policy.family_candidates)

    def bound(values):
        groups = [values[j:j + policy.block_days].mean()
                  for j in range(0, len(values) - policy.block_days + 1, policy.block_days)]
        if len(groups) < 2:
            return float("-inf")
        return float(np.mean(groups) - z * np.std(groups, ddof=1) / np.sqrt(len(groups)))

    mae_lower, hit_lower = bound(improvement), bound(hit_advantage)
    midpoint = len(improvement) // 2
    stable = bool(improvement[:midpoint].mean() > 0 and improvement[midpoint:].mean() > 0)
    passed = bool(mae_lower > 0 and hit_lower > 0 and stable)
    return {"passed": passed, "reason": "historical_gate_passed" if passed else "no_stable_oos_advantage",
            "days": len(known), "mae_improvement": float(improvement.mean()),
            "mae_improvement_lower_bound": mae_lower, "direction_advantage_lower_bound": hit_lower,
            "both_halves_improve": stable, "last_target_date": known[-1]["target_date"],
            "bound_type": "approximate_block_normal_screen_not_forward_evidence"}


def calibrated_radius(records: list[dict], asof: str, sigma: float, policy: DailyPolicy,
                      reference=False):
    known = [r for r in records if r["target_date"] <= asof and
             np.isfinite(r["actual"]) and r.get("sigma") is not None
             and np.isfinite(r["sigma"]) and r["sigma"] > 0]
    known = known[-policy.calibration_days:]
    if len(known) < policy.min_calibration or not np.isfinite(sigma) or sigma <= 0:
        return None
    scores = sorted(abs(r["actual"] - (0 if reference else r["prediction"])) / r["sigma"] for r in known)
    rank = int(np.ceil((len(scores) + 1) * policy.interval_mass))
    return float(scores[min(rank, len(scores)) - 1] * sigma)


def replay_panel(bars: Mapping[str, pd.DataFrame], calendar: Sequence[str], asof: str,
                 policy: DailyPolicy = DailyPolicy()) -> dict:
    """Historical paper predictions, never granted forward-validated status."""
    for symbol in bars:
        if not re.fullmatch(r"[A-Za-z]+\d{3,4}", symbol):
            raise ValueError("Only exact dated contracts are accepted")
    frames = {s: align_bars(f, calendar, asof) for s, f in bars.items()}
    matrices = feature_panel(frames)
    all_results = {}
    for symbol, frame in frames.items():
        targets = frame.close.shift(-1) / frame.close - 1
        volatility = frame.close.pct_change(fill_method=None).rolling(20).std(ddof=1)
        per_candidate = {}
        for name, matrix in matrices[symbol].items():
            values = matrix.to_numpy(float)
            labels = targets.to_numpy(float)
            usable = np.isfinite(values).all(axis=1)
            trainable = np.flatnonzero(usable & np.isfinite(labels))
            records = []
            latest = None
            for j in np.flatnonzero(usable):
                # Label at i ends at i+1. At decision j only i < j is known.
                train = trainable[trainable < j][-policy.max_train:]
                if len(train) < policy.min_train:
                    continue
                day = str(frame.index[j])
                target = str(frame.index[j + 1]) if j + 1 < len(frame) else None
                raw, prediction = ridge(values[train], labels[train], values[j], policy)
                gate = evidence_gate(records, day, policy)
                sigma = float(volatility.iloc[j])
                radius = calibrated_radius(records, day, sigma, policy)
                reference_radius = calibrated_radius(records, day, sigma, policy, reference=True)
                prior_errors = [r["actual"] - r["prediction"] for r in records
                                if r["target_date"] <= day][-policy.calibration_days:]
                old_interval = (np.quantile(prior_errors, [.1, .9]).tolist()
                                if len(prior_errors) >= policy.min_calibration else None)
                latest_record = {
                    "feature_date": day, "target_date": target,
                    "raw_prediction": raw, "prediction": prediction,
                    "actual": float(labels[j]) if np.isfinite(labels[j]) else None,
                    "sigma": sigma if np.isfinite(sigma) else None,
                    "training_rows": len(train),
                    "last_training_label_date": str(frame.index[train[-1] + 1]),
                    "gate": gate, "radius": radius, "reference_radius": reference_radius,
                    "legacy_residual_quantiles": old_interval,
                }
                if target is not None and np.isfinite(labels[j]):
                    records.append(latest_record)
                if day == asof:
                    latest = latest_record
            per_candidate[name] = {"oos": records, "latest": latest,
                                   "features": list(matrix.columns)}
        all_results[symbol] = per_candidate
    return {"policy": asdict(policy), "asof": asof, "mode": "retrospective",
            "forward_validated": False, "contracts": all_results}


def publication_record(symbol: str, candidates: dict, asof: str) -> dict:
    """Retrospective gates do not unlock production; explicit null is abstention."""
    diagnostics = {}
    for name, candidate in candidates.items():
        latest = candidate["latest"]
        diagnostics[name] = {
            "candidate_point": latest["prediction"] if latest else None,
            "gate": latest["gate"] if latest else {"passed": False, "reason": "missing_features_or_training"},
        }
    return {"contract": symbol, "asof": asof, "status": "abstain",
            "prediction_close_return": None, "prediction_settlement_return": None,
            "reference_close_return": 0.0, "reference_label": "price_unchanged_reference_not_prediction",
            "can_trade": False, "forward_validated": False,
            "reason": "retrospective_only_no_forward_validation", "research_diagnostics": diagnostics}


def main():
    """A reproducible paper-only entry point for exact-contract data manifests."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON with calendar and contracts {symbol: bars}")
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.input).read_text())
    result = replay_panel({s: pd.DataFrame(rows) for s, rows in manifest["contracts"].items()},
                          manifest["calendar"], args.asof)
    result["publication"] = [publication_record(s, c, args.asof) for s, c in result["contracts"].items()]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Immutable outputs make retrospective replays distinguishable from predictions.
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
