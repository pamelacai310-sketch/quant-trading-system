from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .nightly_quant_orders import (
    CHINA_TZ,
    FAILURE_ALL_MARKETS_INVALID,
    FAILURE_DATA_VALIDATION_PARTIAL,
    FAILURE_NONE,
    FAILURE_RUNTIME_EXCEPTION,
    FAILURE_SCHEDULER_NOT_RUN,
    _coerce_date,
    _log_dir,
    _repo_root,
    _state_dir,
)


def _latest_file(paths: List[Path]) -> Optional[Path]:
    return max(paths, key=lambda item: item.stem) if paths else None


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"status": "runtime_error", "failure_category": FAILURE_RUNTIME_EXCEPTION, "error": str(exc)}


def check_nightly_health(target_day: date, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root or _repo_root()
    report_dir = _state_dir(root)
    logs_dir = _log_dir(root)
    target_name = f"{target_day.isoformat()}.json"
    report_path = report_dir / target_name
    report_exists = report_path.exists()
    report = _load_json(report_path) if report_exists else {}
    latest_report = _latest_file(sorted(report_dir.glob("*.json"))) if report_dir.exists() else None
    latest_log = _latest_file(sorted(logs_dir.glob("*.log"))) if logs_dir.exists() else None

    if not report_exists:
        status = "failed"
        failure_category = FAILURE_SCHEDULER_NOT_RUN
        reason = f"未找到目标日期夜报 {target_name}，20:00 调度可能未触发或写入了其他目录。"
    else:
        report_status = str(report.get("status", "unknown"))
        failure_category = str(report.get("failure_category") or FAILURE_NONE)
        if report_status in {"ok", "partial_ok"}:
            status = "ok"
            reason = f"找到目标日期夜报，状态为 {report_status}。"
        elif report_status == "failed_validation":
            status = "failed"
            if failure_category == FAILURE_NONE:
                failure_category = FAILURE_ALL_MARKETS_INVALID
            reason = "夜报已运行，但所有市场数据校验均不可交易。"
        else:
            status = "failed"
            failure_category = FAILURE_RUNTIME_EXCEPTION
            reason = f"夜报状态异常：{report_status}。"

    return {
        "status": status,
        "failure_category": failure_category,
        "reason": reason,
        "target_date": target_day.isoformat(),
        "report_path": str(report_path),
        "report_exists": report_exists,
        "latest_report": str(latest_report) if latest_report else None,
        "latest_log": str(latest_log) if latest_log else None,
        "generated_at": report.get("generated_at"),
        "report_status": report.get("status"),
        "market_status": report.get("market_status", {}),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check Nightly Quant Orders health.")
    parser.add_argument("--date", help="Target report date in YYYY-MM-DD. Defaults to Asia/Shanghai today.")
    parser.add_argument("--repo-root", help="Repository root override for tests or non-standard deployments.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    target_day = _coerce_date(args.date) if args.date else datetime.now(CHINA_TZ).date()
    result = check_nightly_health(target_day, Path(args.repo_root) if args.repo_root else None)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"状态：{result['status']}")
        print(f"日期：{result['target_date']}")
        print(f"失败分类：{result['failure_category']}")
        print(f"原因：{result['reason']}")
        print(f"夜报：{result['report_path']}")
        print(f"最新日志：{result['latest_log']}")
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
