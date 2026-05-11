from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .causal_ai import AccountHealthMonitor, EnhancedCausalTradingAgent
from .core.causal import CausalFactorLibrary, CrossAssetCausalEngine, SelfIteratingCausalEngine
from .factors.factor_library import FactorLibrary


CHINA_TZ = ZoneInfo("Asia/Shanghai")
STATE_DIRNAME = "nightly_reports"

HK_CORE_UNIVERSE = [
    "00700.HK",
    "03690.HK",
    "09988.HK",
    "09618.HK",
    "00005.HK",
    "00941.HK",
    "00388.HK",
    "02318.HK",
    "00981.HK",
    "01211.HK",
]

SHFE_CORE_UNIVERSE = ["CU0", "AU0", "AG0", "RB0"]


@dataclass
class MarketValidation:
    market: str
    requested_date: str
    actual_date: Optional[str]
    passed: bool
    reason: str
    sample_count: int = 0


def _require_module(name: str):
    try:
        return __import__(name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"缺少依赖 {name}: {exc}") from exc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _state_dir(repo_root: Path) -> Path:
    return repo_root / "state" / STATE_DIRNAME


def _normalize_live_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).lower() for column in result.columns]
    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"]).reset_index(drop=True)


def _normalize_yfinance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        normalized_columns = [str(column[0]).lower().replace("adj close", "adj_close") for column in frame.columns]
        frame = frame.copy()
        frame.columns = normalized_columns
    else:
        frame = frame.rename(columns={column: str(column).lower().replace("adj close", "adj_close") for column in frame.columns})
    frame = frame.reset_index().rename(columns={"Date": "date", "date": "date"})
    columns = {str(column).lower(): column for column in frame.columns}
    result = pd.DataFrame()
    result["date"] = pd.to_datetime(frame[columns["date"]]).dt.strftime("%Y-%m-%d")
    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(frame[columns[column]], errors="coerce")
    return result.dropna(subset=["close"]).reset_index(drop=True)


def _pct_change_over_window(frame: pd.DataFrame, window: int) -> float:
    close = frame["close"].astype(float)
    if len(close) <= window:
        return 0.0
    return float(close.iloc[-1] / close.iloc[-window - 1] - 1.0)


def _run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _repo_status(repo_root: Path) -> Dict[str, Any]:
    branch = _run_git(repo_root, "branch", "--show-current")
    head = _run_git(repo_root, "rev-parse", "--short", "HEAD")
    try:
        _run_git(repo_root, "fetch", "origin", "--prune")
        origin_main = _run_git(repo_root, "rev-parse", "--short", "origin/main")
        diff = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--quiet", "HEAD..origin/main"],
            capture_output=True,
            check=False,
        )
        synced = diff.returncode == 0
    except Exception as exc:  # noqa: BLE001
        origin_main = None
        synced = False
        return {
            "branch": branch,
            "head": head,
            "origin_main": origin_main,
            "synced_with_origin_main": synced,
            "sync_error": str(exc),
        }
    return {
        "branch": branch,
        "head": head,
        "origin_main": origin_main,
        "synced_with_origin_main": synced,
    }


def _today_in_china() -> date:
    return datetime.now(CHINA_TZ).date()


def _coerce_date(date_arg: Optional[str]) -> date:
    if not date_arg:
        return _today_in_china()
    return datetime.strptime(date_arg, "%Y-%m-%d").date()


def _compact_date(day: date) -> str:
    return day.strftime("%Y%m%d")


def _is_weekend(day: date) -> bool:
    return day.weekday() >= 5


def _next_weekday(day: date) -> date:
    cursor = day + timedelta(days=1)
    while cursor.weekday() >= 5:
        cursor += timedelta(days=1)
    return cursor


def _fetch_hk_data(ak: Any, symbols: Iterable[str]) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        code = symbol.split(".")[0]
        data[symbol] = _normalize_live_frame(ak.stock_hk_daily(symbol=code)).tail(126).reset_index(drop=True)
    return data


def _validate_hk_close(hk_data: Dict[str, pd.DataFrame], target_date: str) -> MarketValidation:
    last_dates = []
    for frame in hk_data.values():
        if frame.empty:
            continue
        last_dates.append(str(frame["date"].iloc[-1]))
    if not last_dates:
        return MarketValidation(
            market="HK",
            requested_date=target_date,
            actual_date=None,
            passed=False,
            reason="港股样本日线为空，无法验证 T 日收盘。",
            sample_count=0,
        )
    actual_date = max(set(last_dates), key=last_dates.count)
    if actual_date == target_date:
        return MarketValidation(
            market="HK",
            requested_date=target_date,
            actual_date=actual_date,
            passed=True,
            reason="港股核心样本最新日线一致落在 T 日收盘。",
            sample_count=len(last_dates),
        )
    reason = (
        f"港股未能证明 T 日收盘；核心样本最新日期为 {actual_date}。"
        if actual_date
        else "港股未能取得有效日线日期。"
    )
    if _is_weekend(datetime.strptime(target_date, "%Y-%m-%d").date()):
        reason += " 当前日期是周末，按硬校验规则不允许沿用最近有效交易日替代港股 T 日收盘。"
    return MarketValidation(
        market="HK",
        requested_date=target_date,
        actual_date=actual_date,
        passed=False,
        reason=reason,
        sample_count=len(last_dates),
    )


def _resolve_shfe_settle(ak: Any, target_day: date, max_lookback: int = 10) -> tuple[pd.DataFrame, MarketValidation]:
    for offset in range(max_lookback + 1):
        probe_day = target_day - timedelta(days=offset)
        try:
            frame = ak.futures_settle_shfe(date=_compact_date(probe_day))
        except Exception:
            continue
        if frame is None or frame.empty:
            continue
        actual_date = str(frame["date"].iloc[0])
        passed = offset == 0
        if passed:
            reason = "SHFE 结算表直接命中 T 日。"
        else:
            reason = f"SHFE T 日无结算表，最近有效交易日回退到 {probe_day.strftime('%Y-%m-%d')}。"
        return frame, MarketValidation(
            market="SHFE",
            requested_date=target_day.strftime("%Y-%m-%d"),
            actual_date=probe_day.strftime("%Y-%m-%d"),
            passed=True,
            reason=reason,
            sample_count=int(len(frame)),
        )
    return pd.DataFrame(), MarketValidation(
        market="SHFE",
        requested_date=target_day.strftime("%Y-%m-%d"),
        actual_date=None,
        passed=False,
        reason="SHFE 在回看窗口内都没有可验证的结算表。",
        sample_count=0,
    )


def _fetch_shfe_data(ak: Any, symbols: Iterable[str]) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        data[symbol] = _normalize_live_frame(ak.futures_zh_daily_sina(symbol=symbol)).tail(126).reset_index(drop=True)
    return data


def _build_market_context(cross_asset_engine: CrossAssetCausalEngine, qqq: pd.DataFrame, gold: pd.DataFrame, copper: pd.DataFrame) -> tuple[Dict[str, Any], Dict[str, Any]]:
    qqq_mom = _pct_change_over_window(qqq, 60)
    gold_mom = _pct_change_over_window(gold, 60)
    copper_mom = _pct_change_over_window(copper, 30)

    growth = 0.02 + qqq_mom * 0.4
    inflation = 0.025 + max(gold_mom, 0.0) * 0.25
    liquidity = 0.01 - max(-qqq_mom, 0.0) * 0.35 - max(-copper_mom, 0.0) * 0.15
    regime = cross_asset_engine.detect_macro_regime(
        growth=growth,
        inflation=inflation,
        liquidity=liquidity,
    )
    market_context = {
        "crisis_probability": round(max(0.12, min(0.35, 1.0 - regime.confidence + 0.15)), 4),
        "cross_asset_regime": {
            "regime": regime.regime.value,
            "growth": regime.growth,
            "inflation": regime.inflation,
            "liquidity": regime.liquidity,
            "confidence": regime.confidence,
        },
    }
    market_data = {
        "US_Debt": {"value": 38_500_000_000_000},
        "Central_Bank_Gold_Purchase": {"value": round(78 + max(gold_mom, 0.0) * 120, 2)},
        "ON_RRP_Balance": {
            "value": max(350_000_000_000, 620_000_000_000 - max(qqq_mom, 0.0) * 400_000_000_000)
        },
        "LME_Inventory_Days": {"value": round(max(1.5, min(12.0, 5.5 - max(copper_mom, 0.0) * 14)), 2)},
        "AI_DataCenter_Capex": {"growth": round(max(0.08, 0.12 + max(qqq_mom, 0.0) * 0.9), 4)},
        "cross_asset_regime": market_context["cross_asset_regime"],
    }
    return market_context, market_data


def _summarize_symbol_report(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": report.get("status"),
        "latest_signal_score": report.get("latest_signal_score"),
        "latest_confidence": report.get("latest_confidence"),
        "selected_features": [
            {
                "factor_name": item.get("factor_name"),
                "rs_score": item.get("rs_score"),
                "r_squared": item.get("r_squared"),
            }
            for item in report.get("selected_features", [])[:5]
        ],
        "top_rejections": [
            {
                "factor_name": item.get("factor_name"),
                "rs_score": item.get("rs_score"),
                "r_squared": item.get("r_squared"),
                "rejection_reason": item.get("rejection_reason"),
            }
            for item in report.get("rejected_features", [])[:3]
        ],
    }


def _latest_price_map(data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for symbol, frame in data.items():
        result[symbol] = {
            "date": str(frame["date"].iloc[-1]),
            "close": float(frame["close"].iloc[-1]),
        }
    return result


def _find_previous_report(report_dir: Path, target_day: date) -> Optional[Path]:
    candidates = sorted(report_dir.glob("*.json"))
    previous = None
    for path in candidates:
        try:
            stamp = datetime.strptime(path.stem, "%Y-%m-%d").date()
        except ValueError:
            continue
        if stamp < target_day:
            previous = path
    return previous


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _compute_close_to_close_return(previous_close: float, current_close: float, action: str) -> float:
    if action == "LONG":
        return current_close / previous_close - 1.0
    if action == "SHORT":
        return previous_close / current_close - 1.0
    return 0.0


def _build_recap(previous_report: Optional[Dict[str, Any]], current_prices: Dict[str, Dict[str, Any]]) -> List[str]:
    if not previous_report:
        return ["上一份夜报不存在，今晚无法做可比口径复盘。"]

    previous_date = previous_report.get("report_date")
    previous_actions = previous_report.get("primary_actions", [])
    if not previous_actions:
        return [f"{previous_date} 没有主动方向单，仅有安全端/尾部保护。"]

    lines: List[str] = []
    for action in previous_actions:
        symbol = action.get("symbol")
        action_name = action.get("action")
        previous_close = action.get("reference_close")
        current_close = current_prices.get(symbol, {}).get("close")
        if action_name not in {"LONG", "SHORT"}:
            lines.append(f"{previous_date} 的 {symbol} 属于非方向性动作 `{action_name}`，不做单边盈亏复盘。")
            continue
        if previous_close in {None, 0} or current_close in {None, 0}:
            lines.append(f"{previous_date} 的 {symbol} 无法取得可比收盘价，跳过复盘。")
            continue
        pnl = _compute_close_to_close_return(float(previous_close), float(current_close), str(action_name))
        lines.append(
            f"{previous_date} 的 {symbol} {action_name} 按收盘到收盘代理口径复盘：{pnl * 100:+.2f}% "
            f"({previous_close:.4f} -> {current_close:.4f})。"
        )
    return lines or [f"{previous_date} 没有可复盘的方向性动作。"]


def _build_instruction(action: Dict[str, Any], last_close: float) -> str:
    action_name = str(action.get("action"))
    if action_name not in {"LONG", "SHORT"}:
        return f"{action_name} {action.get('symbol')} 权重 {float(action.get('target_weight', 0.0)) * 100:.1f}%"
    stop_pct = float(action.get("stop_loss_pct", 0.0))
    take_pct = float(action.get("take_profit_pct", 0.0))
    if action_name == "LONG":
        stop_price = last_close * (1.0 - stop_pct)
        take_price = last_close * (1.0 + take_pct)
    else:
        stop_price = last_close * (1.0 + stop_pct)
        take_price = last_close * (1.0 - take_pct)
    return (
        f"{action_name} {action.get('symbol')} 参考入场 {last_close:.4f}，"
        f"止损 {stop_price:.4f}，止盈 {take_price:.4f}，"
        f"目标权重 {float(action.get('target_weight', 0.0)) * 100:.1f}%，"
        f"置信度 {float(action.get('confidence', 0.0)):.2f}。"
    )


def _observation_lines(symbol_reports: Dict[str, Dict[str, Any]], latest_prices: Dict[str, Dict[str, Any]]) -> List[str]:
    scored: List[tuple[float, str, Dict[str, Any]]] = []
    for symbol, report in symbol_reports.items():
        rejections = report.get("top_rejections", [])
        if not rejections:
            continue
        top = rejections[0]
        score = float(top.get("r_squared") or 0.0)
        scored.append((score, symbol, top))
    scored.sort(key=lambda item: item[0], reverse=True)
    lines: List[str] = []
    for score, symbol, top in scored[:3]:
        close = latest_prices.get(symbol, {}).get("close")
        lines.append(
            f"{symbol} 收盘 {close:.4f}，最接近放行的因子是 `{top.get('factor_name')}`，"
            f"RS={float(top.get('rs_score') or 0.0):.2f}，R²={float(top.get('r_squared') or 0.0):.4f}，"
            f"仍被 `{top.get('rejection_reason')}` 拒绝。"
        )
    return lines


def _classify_primary_actions(actions: List[Dict[str, Any]], price_map: Dict[str, Dict[str, Any]]) -> tuple[List[str], List[Dict[str, Any]]]:
    primary = [action for action in actions if action.get("action") in {"LONG", "SHORT"}]
    if not primary:
        return [], []
    lines = [
        _build_instruction(action, float(price_map[action["symbol"]]["close"]))
        for action in primary
        if action.get("symbol") in price_map
    ]
    return lines, primary


def _format_action_bucket(actions: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for action in actions:
        lines.append(
            f"{action.get('action')} {action.get('symbol')} 权重 {float(action.get('target_weight', 0.0)) * 100:.1f}%："
            f"{action.get('reason', '系统联合优化输出')}"
        )
    return lines


def _next_market_day_text(target_day: date) -> str:
    next_day = _next_weekday(target_day)
    return f"按周历推算，下一个交易日预计为 {next_day.strftime('%Y-%m-%d')}；若遇交易所节假日调整，以官方日历为准。"


def generate_report(target_day: date) -> Dict[str, Any]:
    repo_root = _repo_root()
    report_dir = _state_dir(repo_root)
    report_dir.mkdir(parents=True, exist_ok=True)

    ak = _require_module("akshare")
    yf = _require_module("yfinance")

    repo_status = _repo_status(repo_root)
    target_date = target_day.strftime("%Y-%m-%d")

    hk_data = _fetch_hk_data(ak, HK_CORE_UNIVERSE)
    hk_validation = _validate_hk_close(hk_data, target_date)

    shfe_settle, shfe_validation = _resolve_shfe_settle(ak, target_day)
    shfe_data = _fetch_shfe_data(ak, SHFE_CORE_UNIVERSE)

    report: Dict[str, Any] = {
        "status": "failed_validation",
        "report_date": target_date,
        "generated_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "repo": repo_status,
        "hk_validation": asdict(hk_validation),
        "shfe_validation": asdict(shfe_validation),
        "hk_last": _latest_price_map(hk_data),
        "shfe_last": _latest_price_map(shfe_data),
        "recap_lines": [],
        "primary_actions": [],
        "report_text": "",
    }

    previous_report = _load_json(_find_previous_report(report_dir, target_day))
    current_price_map = {}
    current_price_map.update(report["hk_last"])
    current_price_map.update(report["shfe_last"])
    report["recap_lines"] = _build_recap(previous_report, current_price_map)

    if not hk_validation.passed or not shfe_validation.passed:
        report["report_text"] = render_report_text(report)
        _save_report(report_dir, report)
        return report

    factor_library = FactorLibrary(cache_dir=str(repo_root / "state" / "factor_cache"))
    causal_factor_library = CausalFactorLibrary()
    cross_asset_engine = CrossAssetCausalEngine(causal_factor_library)
    self_iterating_engine = SelfIteratingCausalEngine(
        factor_library=factor_library,
        causal_factor_library=causal_factor_library,
    )
    trading_agent = EnhancedCausalTradingAgent(AccountHealthMonitor(1_000_000.0), use_causal_ai_agent=False)
    qqq = _normalize_yfinance_frame(yf.download("QQQ", period="6mo", interval="1d", auto_adjust=False, progress=False))
    gold = _normalize_yfinance_frame(yf.download("GC=F", period="6mo", interval="1d", auto_adjust=False, progress=False))
    copper_peer = _normalize_yfinance_frame(yf.download("HG=F", period="6mo", interval="1d", auto_adjust=False, progress=False))

    market_context, market_data = _build_market_context(cross_asset_engine, qqq, gold, shfe_data["CU0"])
    hk_cycle = self_iterating_engine.run_learning_cycle(
        hk_data,
        benchmark_frame=None,
        market_context=market_context,
        global_peer_datasets={},
    )
    shfe_cycle = self_iterating_engine.run_learning_cycle(
        shfe_data,
        benchmark_frame=None,
        market_context=market_context,
        global_peer_datasets={
            "CU0": {"HG": copper_peer},
            "AU0": {"COMEX_Gold": gold},
            "AG0": {"COMEX_Gold": gold},
        },
    )
    legacy = trading_agent.execute_decision(current_date=target_date, market_data=market_data)

    report.update(
        {
            "status": "ok",
            "market_context": market_context,
            "market_data": market_data,
            "hk_cycle_status": hk_cycle.get("status"),
            "shfe_cycle_status": shfe_cycle.get("status"),
            "hk_actions": hk_cycle.get("trade_actions", []),
            "shfe_actions": shfe_cycle.get("trade_actions", []),
            "legacy_actions": legacy.get("actions", []),
            "hk_reports": {symbol: _summarize_symbol_report(item) for symbol, item in hk_cycle.get("symbols", {}).items()},
            "shfe_reports": {symbol: _summarize_symbol_report(item) for symbol, item in shfe_cycle.get("symbols", {}).items()},
            "calendar_note": _next_market_day_text(target_day),
        }
    )

    hk_primary_lines, hk_primary_actions = _classify_primary_actions(report["hk_actions"], report["hk_last"])
    shfe_primary_lines, shfe_primary_actions = _classify_primary_actions(report["shfe_actions"], report["shfe_last"])
    report["primary_actions"] = [
        {
            **action,
            "reference_close": float(report["hk_last"].get(action["symbol"], report["shfe_last"].get(action["symbol"], {})).get("close", 0.0)),
        }
        for action in [*hk_primary_actions, *shfe_primary_actions]
    ]
    report["hk_primary_lines"] = hk_primary_lines
    report["shfe_primary_lines"] = shfe_primary_lines
    report["hk_secondary_lines"] = _format_action_bucket([item for item in report["hk_actions"] if item.get("action") not in {"LONG", "SHORT"}])
    report["shfe_secondary_lines"] = _format_action_bucket([item for item in report["shfe_actions"] if item.get("action") not in {"LONG", "SHORT"}])
    report["hk_observations"] = _observation_lines(report["hk_reports"], report["hk_last"])
    report["shfe_observations"] = _observation_lines(report["shfe_reports"], report["shfe_last"])
    report["legacy_lines"] = [
        f"{action.get('action')} {action.get('symbol')}：{action.get('reason')}，置信度 {float(action.get('confidence', 0.0)):.2f}"
        for action in report["legacy_actions"]
    ]
    report["report_text"] = render_report_text(report)
    _save_report(report_dir, report)
    return report


def _save_report(report_dir: Path, report: Dict[str, Any]) -> None:
    path = report_dir / f"{report['report_date']}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def render_report_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"状态：{report.get('status')}")
    lines.append(f"报告日期：{report.get('report_date')}")
    lines.append(f"生成时间：{report.get('generated_at')}")
    repo = report.get("repo", {})
    lines.append(
        "代码基线："
        f"branch={repo.get('branch')} head={repo.get('head')} origin/main={repo.get('origin_main')} "
        f"synced={repo.get('synced_with_origin_main')}"
    )
    if repo.get("sync_error"):
        lines.append(f"同步备注：{repo.get('sync_error')}")

    hk_validation = report.get("hk_validation", {})
    shfe_validation = report.get("shfe_validation", {})
    lines.append("")
    lines.append("数据校验：")
    lines.append(
        f"港股：passed={hk_validation.get('passed')} requested={hk_validation.get('requested_date')} "
        f"actual={hk_validation.get('actual_date')}；{hk_validation.get('reason')}"
    )
    lines.append(
        f"SHFE：passed={shfe_validation.get('passed')} requested={shfe_validation.get('requested_date')} "
        f"actual={shfe_validation.get('actual_date')}；{shfe_validation.get('reason')}"
    )

    lines.append("")
    lines.append("复盘：")
    lines.extend([f"- {line}" for line in report.get("recap_lines", [])])

    if report.get("status") != "ok":
        lines.append("")
        lines.append("结果：")
        lines.append("- 本次任务因时间戳校验未全部通过，按硬规则直接失败退出，不生成新的港股/SHFE 交易指令。")
        return "\n".join(lines)

    regime = report.get("market_context", {}).get("cross_asset_regime", {})
    lines.append("")
    lines.append("市场状态：")
    lines.append(
        f"- 制度={regime.get('regime')} growth={regime.get('growth'):.4f} "
        f"inflation={regime.get('inflation'):.4f} liquidity={regime.get('liquidity'):.4f} "
        f"confidence={regime.get('confidence'):.4f}"
    )
    lines.append(f"- {report.get('calendar_note')}")

    lines.append("")
    lines.append("主决策：")
    hk_primary = report.get("hk_primary_lines", [])
    shfe_primary = report.get("shfe_primary_lines", [])
    if hk_primary:
        lines.append("- 港股：")
        lines.extend([f"  {line}" for line in hk_primary])
    else:
        lines.append("- 港股：没有通过 `RS>70 且 R²>0.7` 的主动仓信号。")
    if shfe_primary:
        lines.append("- SHFE：")
        lines.extend([f"  {line}" for line in shfe_primary])
    else:
        lines.append("- SHFE：没有通过 `RS>70 且 R²>0.7` 的主动仓信号。")

    lines.append("")
    lines.append("安全端 / 尾部保护：")
    for line in report.get("hk_secondary_lines", []):
        lines.append(f"- 港股组合：{line}")
    for line in report.get("shfe_secondary_lines", []):
        lines.append(f"- SHFE组合：{line}")

    lines.append("")
    lines.append("观察名单：")
    for line in report.get("hk_observations", []):
        lines.append(f"- 港股：{line}")
    for line in report.get("shfe_observations", []):
        lines.append(f"- SHFE：{line}")

    legacy_lines = report.get("legacy_lines", [])
    if legacy_lines:
        lines.append("")
        lines.append("兜底偏向：")
        for line in legacy_lines:
            lines.append(f"- {line}")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate nightly quant orders report.")
    parser.add_argument("--date", help="Target date in YYYY-MM-DD. Defaults to Asia/Shanghai today.")
    args = parser.parse_args(argv)

    try:
        target_day = _coerce_date(args.date)
        report = generate_report(target_day)
        print(report["report_text"])
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"状态：runtime_error\n原因：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
