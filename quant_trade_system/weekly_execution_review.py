from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import List, Optional

from .nightly_quant_orders import CHINA_TZ, generate_weekly_execution_review


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate weekly execution review for nightly quant orders.")
    parser.add_argument("--week-start", required=True, help="Week start in YYYY-MM-DD.")
    parser.add_argument("--week-end", required=True, help="Week end in YYYY-MM-DD.")
    parser.add_argument("--evaluation-date", help="Mark-to-market date in YYYY-MM-DD. Defaults to week end.")
    args = parser.parse_args(argv)

    week_start = datetime.strptime(args.week_start, "%Y-%m-%d").date()
    week_end = datetime.strptime(args.week_end, "%Y-%m-%d").date()
    evaluation_date = (
        datetime.strptime(args.evaluation_date, "%Y-%m-%d").date()
        if args.evaluation_date
        else datetime.now(CHINA_TZ).date()
    )
    review = generate_weekly_execution_review(week_start, week_end, evaluation_date=evaluation_date)
    print(json.dumps(review, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
