"""Run the fixed repair experiment against the frozen September forecast files."""
from pathlib import Path
import argparse
from collections import Counter
import csv
import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_trade_system.daily_forecast import replay_panel, publication_record
from quant_trade_system.cn_daily_calendar import sessions, SOURCES, CLOSURES


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def stats(rows):
    if not rows:
        return {"n": 0}
    f = pd.DataFrame(rows)
    error = f.prediction - f.actual
    direction_hits = (np.sign(f.prediction) == np.sign(f.actual)) & (f.prediction != 0) & (f.actual != 0)
    day_errors = pd.DataFrame({"date": f.target_date, "model": abs(error), "zero": abs(f.actual)})
    means = day_errors.groupby("date")[["model", "zero"]].mean()
    interval = [(abs(r["actual"] - r["prediction"]) <= r["radius"]) for r in rows if r["radius"] is not None]
    paired = [r for r in rows if r["radius"] is not None and r.get("legacy_residual_quantiles") is not None]
    interval_comparison = {
        "n": len(paired),
        "legacy_hits": sum(r["legacy_residual_quantiles"][0] <= r["actual"] - r["prediction"] <= r["legacy_residual_quantiles"][1] for r in paired),
        "new_hits": sum(abs(r["actual"] - r["prediction"]) <= r["radius"] for r in paired),
        "legacy_mean_width": float(np.mean([r["legacy_residual_quantiles"][1] - r["legacy_residual_quantiles"][0] for r in paired])) if paired else None,
        "new_mean_width": float(np.mean([2 * r["radius"] for r in paired])) if paired else None,
    }
    return {"n": len(f), "target_dates": len(means),
            "mae": float(abs(error).mean()), "baseline_mae": float(abs(f.actual).mean()),
            "date_equal_mae": float(means.model.mean()), "date_equal_baseline_mae": float(means.zero.mean()),
            "direction_hits": int(direction_hits.sum()),
            "direction_accuracy": float(direction_hits.mean()),
            "mean_prediction": float(f.prediction.mean()), "mean_actual": float(f.actual.mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "same_direction_bias_share_mse": float(error.mean() ** 2 / np.mean(error ** 2)) if np.mean(error ** 2) else 0,
            "interval_count": len(interval), "interval_hits": int(sum(interval)),
            "interval_coverage": float(np.mean(interval)) if interval else None,
            "interval_paired_comparison": interval_comparison,
            "historical_gate_pass_count": int(sum(r["gate"]["passed"] for r in rows))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.workspace, args.output
    original = json.loads((root / "analysis-20260908/analysis.json").read_text())
    actual = json.loads((root / "actual-validation-20260908/accuracy.json").read_text())
    actual_by = {r["contract"]: r for r in actual["details"]}
    old_calendar = json.loads((root / "analysis-20260908/evidence/trading_calendar.json").read_text())["dates"]
    calendar = sessions("2025-01-01", "2026-09-08")
    bars, input_hashes = {}, {}
    diagnostics = {"price_revisions": [], "invalid_ohlc": [], "zero_volume_on_target": [], "input_hashes": input_hashes}
    native_statuses = Counter()
    for source in sorted((root / "analysis-20260908/evidence").glob("native_*.json")):
        input_hashes[str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
        native = json.loads(source.read_text())
        for cycle in native["cycles"].values():
            native_statuses.update(row["status"] for row in cycle["symbols"].values())
    if not native_statuses:
        raise ValueError("Frozen native engine evidence is required")
    diagnostics["native_symbol_statuses"] = dict(native_statuses)
    diagnostics["calendar_repair"] = {
        "sources": SOURCES, "added_sessions_before_cutoff": sorted(set(calendar[:-1]) - set(old_calendar)),
        "start": calendar[0], "end": calendar[-1], "method": "weekdays_minus_exchange_notified_closures"}
    dump(out / "exchange_calendar.json", {"dates": calendar, "closure_ranges": CLOSURES, "sources": SOURCES})
    original_by = {r["contract"]: r for r in original["contracts"] if r["prediction_close"] is not None}
    for symbol, row in original_by.items():
        source = root / "analysis-20260908/evidence/clean" / (symbol + ".json")
        input_hashes[str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
        frozen = [r for r in json.loads(source.read_text()) if r["date"] >= "2025-01-01"]
        history_source = root / "actual-validation-20260908/evidence/histories" / (symbol + ".json")
        input_hashes[str(history_source.relative_to(root))] = hashlib.sha256(history_source.read_bytes()).hexdigest()
        history = json.loads(history_source.read_text())
        revised = next((r for r in history["rows"] if r["date"] == "2026-09-07"), None)
        if revised and abs(float(revised["close"]) - row["close"]) > 1e-8:
            diagnostics["price_revisions"].append(symbol)
        last = next(r for r in history["rows"] if r["date"] == "2026-09-08")
        last = dict(last, close=actual_by[symbol]["actual_close"], settle=actual_by[symbol]["actual_settle"])
        frame = pd.DataFrame(frozen + [last])
        for col in ["open", "high", "low", "close", "volume"]:
            frame[col] = pd.to_numeric(frame[col])
        invalid = (frame.high < frame[["open", "low", "close"]].max(axis=1)) | (frame.low > frame[["open", "high", "close"]].min(axis=1))
        if invalid.any():
            diagnostics["invalid_ohlc"].append({"contract": symbol, "dates": frame.loc[invalid, "date"].tolist()})
        if float(last["volume"]) <= 0:
            diagnostics["zero_volume_on_target"].append(symbol)
        bars[symbol] = frame
    # September 8 is only a realized label for the September 7 decision.
    replay = replay_panel(bars, calendar, "2026-09-08")
    flat = []
    final = []
    comparison = []
    for symbol, candidates in replay["contracts"].items():
        published_candidates = {}
        for candidate, obj in candidates.items():
            records = obj["oos"]
            for r in records:
                assert r["last_training_label_date"] <= r["feature_date"] < r["target_date"]
                if "last_target_date" in r["gate"]:
                    assert r["gate"]["last_target_date"] <= r["feature_date"]
                flat.append(dict(r, contract=symbol, candidate=candidate))
            last = next((r for r in records if r["target_date"] == "2026-09-08"), None)
            published_candidates[candidate] = {"latest": last}
            if last is not None:
                comparison.append(dict(last, contract=symbol, candidate=candidate,
                                       original_kind=original_by[symbol]["model_kind"],
                                       original_prediction=original_by[symbol]["prediction_close"]))
        final.append(publication_record(symbol, published_candidates, "2026-09-07"))
    # All old baselines remain references, not retrospectively promoted forecasts.
    periods = {
        "earlier_diagnostic_through_2026_08_07": lambda d: d <= "2026-08-07",
        "later_diagnostic_2026_08_10_to_09_07": lambda d: "2026-08-10" <= d <= "2026-09-07",
        "known_event_2026_09_08": lambda d: d == "2026-09-08",
    }
    summary = {"mode": "post_event_design_retrospective_only", "stages": {},
               "published_directional_forecasts": sum(r["prediction_close_return"] is not None for r in final),
               "abstentions": len(final), "forecast_coverage": 0,
               "native_active_signals_before_repair": native_statuses["trained"], "new_model_has_forward_evidence": False}
    for name, accept in periods.items():
        stage = [r for r in flat if accept(r["target_date"])]
        keys = {(r["contract"], r["target_date"]) for r in stage if r["candidate"] == "context"}
        common_local = [r for r in stage if r["candidate"] == "local" and (r["contract"], r["target_date"]) in keys]
        summary["stages"][name] = {
            "local_all": stats([r for r in stage if r["candidate"] == "local"]),
            "context": stats([r for r in stage if r["candidate"] == "context"]),
            "local_on_context_same_samples": stats(common_local),
        }
    matched = [r for r in comparison if r["candidate"] == "local"]
    diagnostics["reproduced_original_predictions"] = sum(abs(r["prediction"] - r["original_prediction"]) < 1e-10 for r in matched)
    diagnostics["original_prediction_mismatches"] = [r["contract"] for r in matched if abs(r["prediction"] - r["original_prediction"]) >= 1e-10]
    # Verify the old forecasts were frozen before using their known labels.
    old = [r for r in actual["details"] if r["model_kind"] != "价格不变基准"]
    errors = np.array([r["forecast_error"] for r in old])
    diagnostics["common_bias_share_mse_original"] = float(errors.mean() ** 2 / np.mean(errors ** 2))
    diagnostics["old_direction_hits"] = sum(r["direction_hit"] for r in old)
    diagnostics["old_nonbaseline_count"] = len(old)
    fitted = [r for r in original_by.values() if r.get("fit")]
    diagnostics["mean_feature_contributions_close_return"] = {
        key: float(np.mean([r["fit"]["contributions"][key] * .5 for r in fitted]))
        for key in fitted[0]["fit"]["contributions"]
    }
    diagnostics["mean_intercept_contribution"] = float(np.mean([r["fit"]["intercept"] * .5 for r in fitted]))
    diagnostics["raw_unshrunk_mae_event"] = float(np.mean([abs(2*r["predicted_close_return"] - r["actual_close_return"]) for r in old]))
    diagnostics["shrunk_mae_event"] = float(np.mean([r["absolute_error"] for r in old]))
    diagnostics["counterfactual_zero_reference_mae_event"] = float(np.mean([abs(r["actual_close_return"]) for r in old]))
    diagnostics["concentration_largest_errors"] = sorted(old, key=lambda r: r["absolute_error"], reverse=True)[:8]
    for path in [root / "analysis-20260908/analysis.json", root / "actual-validation-20260908/accuracy.json",
                 root / "analysis-20260908/evidence/trading_calendar.json"]:
        input_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(out / "summary.json", summary)
    dump(out / "root_cause_evidence.json", diagnostics)
    dump(out / "retrospective_publication.json", final)
    dump(out / "event_comparison.json", comparison)
    dump(out / "all_walk_forward_records.json", flat)
    with (out / "event_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["contract", "candidate", "feature_date", "target_date", "prediction", "actual", "raw_prediction", "radius", "original_prediction", "original_kind"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("REPRODUCTION", diagnostics["reproduced_original_predictions"], diagnostics["original_prediction_mismatches"])


if __name__ == "__main__":
    main()
