from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_trade_system.futures_dc_research import (  # noqa: E402
    DEFAULT_PRODUCTS,
    FuturesCostModel,
    apply_holdout_validation,
    fetch_cached_main_contract_minute_frames,
    fetch_main_contract_minute_frames,
    load_cached_minute_frames,
    load_csv_minute_frames,
    scan_futures_dc_strategies,
    split_research_holdout_frames,
    write_research_outputs,
)
from quant_trade_system.futures_specs import normalize_futures_symbol  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan China futures DC path-capture candidates.")
    parser.add_argument("--products", default=",".join(DEFAULT_PRODUCTS), help="Comma-separated product codes. Use --products= with CSV input to scan all symbols.")
    parser.add_argument("--period", default="5", choices=["1", "5", "15", "30", "60"], help="AkShare minute period.")
    parser.add_argument("--max-contracts", type=int, default=None, help="Limit fetched main contracts.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller parameter grid for a fast smoke run.")
    parser.add_argument("--min-bars", type=int, default=180, help="Skip contracts with fewer bars.")
    parser.add_argument("--random-trials", type=int, default=128, help="Random-direction control trials per candidate.")
    parser.add_argument("--overshoot-multiples", default="0.5,1.0", help="Comma-separated theta multiples required after DC confirmation for overshoot families.")
    parser.add_argument("--cache-dir", default="state/futures_minute_cache", help="Minute data cache directory.")
    parser.add_argument("--no-cache", action="store_true", help="Disable local minute cache updates.")
    parser.add_argument("--include-cached-contracts", action="store_true", help="Also scan cached contracts that are no longer current main contracts.")
    parser.add_argument("--csv-dir", default="", help="Directory of local minute CSV files. When set, AkShare fetching is skipped.")
    parser.add_argument("--csv-glob", default="*.csv", help="Glob used with --csv-dir.")
    parser.add_argument("--csv-files", default="", help="Comma-separated local minute CSV files. Can be combined with --csv-dir.")
    parser.add_argument("--csv-symbol-column", default="symbol", help="Optional CSV column used to split a file into symbols.")
    parser.add_argument("--stale-days", type=int, default=3, help="Warn when latest cached bar is older than this many days.")
    parser.add_argument("--holdout-fraction", type=float, default=0.20, help="Final fraction of each contract reserved from scanning.")
    parser.add_argument("--min-holdout-bars", type=int, default=60, help="Minimum bars required for an untouched holdout split.")
    parser.add_argument("--no-holdout", action="store_true", help="Disable final untouched holdout validation.")
    parser.add_argument("--state-dir", default="state", help="Directory for JSON candidate output.")
    parser.add_argument("--report-path", default="FUTURES_DC_CAPTURE_REPORT.md", help="Markdown report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    products = [item.strip().upper() for item in args.products.split(",") if item.strip()]
    overshoot_multiples = [float(item.strip()) for item in args.overshoot_multiples.split(",") if item.strip()]
    csv_paths = _csv_paths(args)
    if csv_paths:
        frames = load_csv_minute_frames(csv_paths, symbol_column=args.csv_symbol_column)
        if products:
            wanted_products = {normalize_futures_symbol(product) for product in products}
            frames = {
                symbol: frame
                for symbol, frame in frames.items()
                if normalize_futures_symbol(symbol) in wanted_products
            }
    elif args.no_cache:
        frames = fetch_main_contract_minute_frames(products=products, period=args.period, max_contracts=args.max_contracts)
    else:
        frames = fetch_cached_main_contract_minute_frames(
            products=products,
            period=args.period,
            max_contracts=args.max_contracts,
            cache_dir=args.cache_dir,
        )
        if args.include_cached_contracts:
            for symbol, frame in load_cached_minute_frames(cache_dir=args.cache_dir, period=args.period).items():
                frames.setdefault(symbol, frame)
    frames = {symbol: frame for symbol, frame in frames.items() if len(frame) >= args.min_bars}
    if not frames:
        print("No usable minute frames loaded. Check CSV paths, AkShare availability, products, period, or min-bars.")
        return 2

    cost_model = FuturesCostModel()
    if args.no_holdout:
        research_frames = frames
        holdout_frames = {}
    else:
        research_frames, holdout_frames = split_research_holdout_frames(
            frames,
            holdout_fraction=args.holdout_fraction,
            min_research_bars=args.min_bars,
            min_holdout_bars=args.min_holdout_bars,
        )
    if not research_frames:
        print("No usable research frames after holdout split. Lower --min-bars or --min-holdout-bars.")
        return 2

    if args.quick:
        results = scan_futures_dc_strategies(
            research_frames,
            theta_bps_values=(16.0, 24.0, 32.0),
            vol_filters=("all", "high_70_plus"),
            open_interest_filters=("all",),
            time_filters=("all", "day", "night"),
            event_spacing_bars_values=(0, 4),
            overshoot_trigger_multiples=overshoot_multiples[:1] or (0.5,),
            cost_model=cost_model,
            random_trials=args.random_trials,
        )
    else:
        results = scan_futures_dc_strategies(
            research_frames,
            overshoot_trigger_multiples=overshoot_multiples or (0.5,),
            cost_model=cost_model,
            random_trials=args.random_trials,
        )

    if holdout_frames:
        results = apply_holdout_validation(results, holdout_frames, cost_model=cost_model)

    paths = write_research_outputs(results, state_dir=args.state_dir, report_path=args.report_path)
    pass_count = sum(1 for row in results if row.get("status") == "PASS")
    watch_count = sum(1 for row in results if row.get("status") == "WATCH")
    print(f"Fetched contracts: {', '.join(frames)}")
    print("Bars: " + ", ".join(_frame_coverage(symbol, frame, args.stale_days) for symbol, frame in frames.items()))
    print("Research bars: " + ", ".join(_frame_coverage(symbol, frame, args.stale_days) for symbol, frame in research_frames.items()))
    if holdout_frames:
        print("Holdout bars: " + ", ".join(_frame_coverage(symbol, frame, args.stale_days) for symbol, frame in holdout_frames.items()))
    print(f"Candidates scanned: {len(results)}")
    print(f"PASS: {pass_count}, WATCH: {watch_count}")
    print(f"JSON: {paths['json_path']}")
    print(f"Report: {paths['report_path']}")
    return 0


def _csv_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.csv_dir:
        paths.extend(sorted(Path(args.csv_dir).expanduser().glob(str(args.csv_glob))))
    for item in str(args.csv_files).split(","):
        item = item.strip()
        if item:
            paths.append(Path(item).expanduser())
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path)] = path
    return list(unique.values())


def _frame_coverage(symbol, frame, stale_days: int) -> str:
    if frame.empty or "timestamp" not in frame:
        return f"{symbol}=0"
    start = frame["timestamp"].min()
    end = frame["timestamp"].max()
    stale = ""
    try:
        age_days = (datetime.now() - end.to_pydatetime()).days
        if age_days > int(stale_days):
            stale = f", stale={age_days}d"
    except Exception:
        pass
    return f"{symbol}={len(frame)}[{start} -> {end}{stale}]"


if __name__ == "__main__":
    raise SystemExit(main())
