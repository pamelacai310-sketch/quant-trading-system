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
    fetch_cached_main_contract_minute_frames,
    fetch_main_contract_minute_frames,
    load_cached_minute_frames,
    scan_futures_dc_strategies,
    write_research_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan China futures DC path-capture candidates.")
    parser.add_argument("--products", default=",".join(DEFAULT_PRODUCTS), help="Comma-separated product codes.")
    parser.add_argument("--period", default="5", choices=["1", "5", "15", "30", "60"], help="AkShare minute period.")
    parser.add_argument("--max-contracts", type=int, default=None, help="Limit fetched main contracts.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller parameter grid for a fast smoke run.")
    parser.add_argument("--min-bars", type=int, default=180, help="Skip contracts with fewer bars.")
    parser.add_argument("--random-trials", type=int, default=128, help="Random-direction control trials per candidate.")
    parser.add_argument("--cache-dir", default="state/futures_minute_cache", help="Minute data cache directory.")
    parser.add_argument("--no-cache", action="store_true", help="Disable local minute cache updates.")
    parser.add_argument("--include-cached-contracts", action="store_true", help="Also scan cached contracts that are no longer current main contracts.")
    parser.add_argument("--stale-days", type=int, default=3, help="Warn when latest cached bar is older than this many days.")
    parser.add_argument("--state-dir", default="state", help="Directory for JSON candidate output.")
    parser.add_argument("--report-path", default="FUTURES_DC_CAPTURE_REPORT.md", help="Markdown report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    products = [item.strip().upper() for item in args.products.split(",") if item.strip()]
    if args.no_cache:
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
        print("No usable minute frames fetched. Check AkShare availability, products, period, or min-bars.")
        return 2

    if args.quick:
        results = scan_futures_dc_strategies(
            frames,
            theta_bps_values=(16.0, 24.0, 32.0),
            vol_filters=("all", "high_70_plus"),
            open_interest_filters=("all",),
            time_filters=("all", "day", "night"),
            event_spacing_bars_values=(0, 4),
            cost_model=FuturesCostModel(),
            random_trials=args.random_trials,
        )
    else:
        results = scan_futures_dc_strategies(frames, cost_model=FuturesCostModel(), random_trials=args.random_trials)

    paths = write_research_outputs(results, state_dir=args.state_dir, report_path=args.report_path)
    pass_count = sum(1 for row in results if row.get("status") == "PASS")
    watch_count = sum(1 for row in results if row.get("status") == "WATCH")
    print(f"Fetched contracts: {', '.join(frames)}")
    print("Bars: " + ", ".join(_frame_coverage(symbol, frame, args.stale_days) for symbol, frame in frames.items()))
    print(f"Candidates scanned: {len(results)}")
    print(f"PASS: {pass_count}, WATCH: {watch_count}")
    print(f"JSON: {paths['json_path']}")
    print(f"Report: {paths['report_path']}")
    return 0


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
