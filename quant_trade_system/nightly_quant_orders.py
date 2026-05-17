from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .causal_ai import AccountHealthMonitor, EnhancedCausalTradingAgent
from .core.causal import CausalFactorLibrary, CrossAssetCausalEngine, SelfIteratingCausalEngine
from .factors.factor_library import FactorLibrary
from .universe_provider import MarketUniverseProvider


CHINA_TZ = ZoneInfo("Asia/Shanghai")
STATE_DIRNAME = "nightly_reports"

CN_FUTURES_EXCHANGES = ["SHFE", "INE", "DCE", "CZCE", "CFFEX", "GFEX"]
US_TAIL_HEDGE_SYMBOL = "GLD"
US_SAFE_RESERVE_SYMBOL = "USD_CASH"
HK_TAIL_HEDGE_SYMBOL = "02840.HK"
HK_SAFE_RESERVE_SYMBOL = "HKD_CASH"
CN_FUTURES_TAIL_HEDGE_SYMBOL = "AU0"
CN_FUTURES_SAFE_RESERVE_SYMBOL = "CNY_CASH"
SHFE_TAIL_HEDGE_SYMBOL = CN_FUTURES_TAIL_HEDGE_SYMBOL
SHFE_SAFE_RESERVE_SYMBOL = CN_FUTURES_SAFE_RESERVE_SYMBOL
US_EXECUTION_SUPPORT_UNIVERSE = [US_TAIL_HEDGE_SYMBOL]
HK_EXECUTION_SUPPORT_UNIVERSE = [HK_TAIL_HEDGE_SYMBOL]


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


def _market_provider(prefer_live: bool = False) -> MarketUniverseProvider:
    return MarketUniverseProvider(data_dir=_repo_root() / "data", prefer_live=prefer_live)


def _us_core_universe() -> List[str]:
    return _market_provider(prefer_live=False).get_symbols("us_core", include_contracts=False)


def _hk_universe() -> List[str]:
    return _market_provider(prefer_live=False).get_symbols("hk_hsi", include_contracts=False)


def _cn_futures_exchange_universe() -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for item in _market_provider(prefer_live=False).get_universe("cn_futures_products", include_contracts=False):
        exchange = str(item.exchange)
        grouped[exchange].append(f"{item.symbol}0")
    return {
        exchange: sorted(symbols)
        for exchange, symbols in grouped.items()
        if exchange in CN_FUTURES_EXCHANGES
    }


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


def _previous_weekday(day: date) -> date:
    cursor = day - timedelta(days=1)
    while cursor.weekday() >= 5:
        cursor -= timedelta(days=1)
    return cursor


def _cash_price_snapshot(target_date: str) -> Dict[str, Any]:
    return {"date": target_date, "close": 1.0}


def _latest_sample_date(data: Dict[str, pd.DataFrame]) -> tuple[Optional[str], int]:
    last_dates: List[str] = []
    for frame in data.values():
        if frame.empty:
            continue
        last_dates.append(str(frame["date"].iloc[-1]))
    if not last_dates:
        return None, 0
    actual_date = max(set(last_dates), key=last_dates.count)
    return actual_date, len(last_dates)


def _fetch_hk_data(ak: Any, symbols: Iterable[str], end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        code = symbol.split(".")[0]
        try:
            frame = _normalize_live_frame(ak.stock_hk_daily(symbol=code))
        except Exception:
            continue
        if end_date:
            frame = frame.loc[frame["date"] <= end_date]
        if frame.empty:
            continue
        data[symbol] = frame.tail(126).reset_index(drop=True)
    return data


def _validate_hk_close(hk_data: Dict[str, pd.DataFrame], target_date: str) -> MarketValidation:
    actual_date, sample_count = _latest_sample_date(hk_data)
    if not actual_date:
        return MarketValidation(
            market="HK",
            requested_date=target_date,
            actual_date=None,
            passed=False,
            reason="港股样本日线为空，无法验证 T 日收盘。",
            sample_count=0,
        )
    if actual_date == target_date:
        return MarketValidation(
            market="HK",
            requested_date=target_date,
            actual_date=actual_date,
            passed=True,
            reason="港股核心样本最新日线一致落在 T 日收盘。",
            sample_count=sample_count,
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
        sample_count=sample_count,
    )


def _fetch_us_data(yf: Any, symbols: Iterable[str], end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            frame = yf.download(
                symbol,
                period="6mo",
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
        except Exception:
            continue
        if frame is None or frame.empty:
            continue
        normalized = _normalize_yfinance_frame(frame)
        if end_date:
            normalized = normalized.loc[normalized["date"] <= end_date]
        if normalized.empty:
            continue
        data[symbol] = normalized.tail(126).reset_index(drop=True)
    return data


def _validate_us_close(us_data: Dict[str, pd.DataFrame], target_day: date, max_lookback: int = 4) -> MarketValidation:
    target_date = target_day.strftime("%Y-%m-%d")
    actual_date, sample_count = _latest_sample_date(us_data)
    if not actual_date:
        return MarketValidation(
            market="US",
            requested_date=target_date,
            actual_date=None,
            passed=False,
            reason="美股样本日线为空，无法验证最近完整收盘。",
            sample_count=0,
        )

    expected_date = _previous_weekday(target_day)
    if actual_date == expected_date.strftime("%Y-%m-%d"):
        return MarketValidation(
            market="US",
            requested_date=target_date,
            actual_date=actual_date,
            passed=True,
            reason="美股样本最新日线与 T 日晚间可用的最近完整美股收盘一致。",
            sample_count=sample_count,
        )

    allowed_dates: List[str] = []
    cursor = expected_date
    while len(allowed_dates) < max_lookback:
        if cursor.weekday() < 5:
            allowed_dates.append(cursor.strftime("%Y-%m-%d"))
        cursor -= timedelta(days=1)
    if actual_date in allowed_dates:
        return MarketValidation(
            market="US",
            requested_date=target_date,
            actual_date=actual_date,
            passed=True,
            reason=f"美股按最近有效交易日回退校验，实际使用数据日期为 {actual_date}。",
            sample_count=sample_count,
        )
    return MarketValidation(
        market="US",
        requested_date=target_date,
        actual_date=actual_date,
        passed=False,
        reason=(
            "美股未能验证最近完整收盘；"
            f"预期日期不晚于 {expected_date.strftime('%Y-%m-%d')}，实际样本最新日期为 {actual_date}。"
        ),
        sample_count=sample_count,
    )


def _resolve_futures_settle(ak: Any, exchange: str, target_day: date, max_lookback: int = 10) -> tuple[pd.DataFrame, MarketValidation]:
    for offset in range(max_lookback + 1):
        probe_day = target_day - timedelta(days=offset)
        try:
            frame = ak.futures_settle(date=_compact_date(probe_day), market=exchange)
        except Exception:
            continue
        if frame is None or frame.empty:
            continue
        passed = offset == 0
        if passed:
            reason = f"{exchange} 结算参数直接命中 T 日。"
        else:
            reason = f"{exchange} T 日无结算参数，最近有效交易日回退到 {probe_day.strftime('%Y-%m-%d')}。"
        return frame, MarketValidation(
            market=exchange,
            requested_date=target_day.strftime("%Y-%m-%d"),
            actual_date=probe_day.strftime("%Y-%m-%d"),
            passed=True,
            reason=reason,
            sample_count=int(len(frame)),
        )
    return pd.DataFrame(), MarketValidation(
        market=exchange,
        requested_date=target_day.strftime("%Y-%m-%d"),
        actual_date=None,
        passed=False,
        reason=f"{exchange} 在回看窗口内都没有可验证的结算参数。",
        sample_count=0,
    )


def _fetch_futures_data(ak: Any, symbols: Iterable[str], end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            frame = _normalize_live_frame(ak.futures_zh_daily_sina(symbol=symbol))
        except Exception:
            continue
        if end_date:
            frame = frame.loc[frame["date"] <= end_date]
        if frame.empty:
            continue
        data[symbol] = frame.tail(126).reset_index(drop=True)
    return data


def _validate_futures_close(
    exchange: str,
    futures_data: Dict[str, pd.DataFrame],
    settle_validation: MarketValidation,
) -> MarketValidation:
    if not settle_validation.passed or settle_validation.actual_date is None:
        return settle_validation
    actual_date, sample_count = _latest_sample_date(futures_data)
    if not actual_date:
        return MarketValidation(
            market=exchange,
            requested_date=settle_validation.requested_date,
            actual_date=None,
            passed=False,
            reason=f"{exchange} 主力连续合约样本为空，无法与结算参数交叉校验。",
            sample_count=0,
        )
    if actual_date == settle_validation.actual_date:
        return MarketValidation(
            market=exchange,
            requested_date=settle_validation.requested_date,
            actual_date=actual_date,
            passed=True,
            reason=f"{settle_validation.reason} {exchange} 主力连续合约日线与该日期一致。",
            sample_count=sample_count,
        )
    return MarketValidation(
        market=exchange,
        requested_date=settle_validation.requested_date,
        actual_date=actual_date,
        passed=False,
        reason=(
            f"{exchange} 结算参数有效日期为 {settle_validation.actual_date}，"
            f"但主力连续合约最新日线为 {actual_date}，时间戳不一致。"
        ),
        sample_count=sample_count,
    )


def _build_futures_peer_datasets(
    futures_data: Dict[str, pd.DataFrame],
    gold_peer: pd.DataFrame,
    copper_peer: pd.DataFrame,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    peer_datasets: Dict[str, Dict[str, pd.DataFrame]] = {}
    for symbol in futures_data:
        upper = symbol.upper()
        if any(token in upper for token in ["CU", "BC", "COPPER", "HG"]):
            peer_datasets[symbol] = {"HG": copper_peer}
            continue
        if any(token in upper for token in ["AU", "AG", "GOLD"]):
            peer_datasets[symbol] = {"COMEX_Gold": gold_peer}
    return peer_datasets


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


def _price_on_or_before(frame: pd.DataFrame, target_date: str) -> Optional[Dict[str, Any]]:
    if frame.empty:
        return None
    filtered = frame.loc[frame["date"] <= target_date]
    if filtered.empty:
        return None
    row = filtered.iloc[-1]
    return {"date": str(row["date"]), "close": float(row["close"])}


def _materialize_execution_actions(
    actions: List[Dict[str, Any]],
    market: str,
    price_map: Dict[str, Dict[str, Any]],
    target_date: str,
) -> List[Dict[str, Any]]:
    execution_actions: List[Dict[str, Any]] = []
    market_upper = market.upper()
    if market_upper == "HK":
        tail_hedge_symbol = HK_TAIL_HEDGE_SYMBOL
        safe_reserve_symbol = HK_SAFE_RESERVE_SYMBOL
    elif market_upper == "US":
        tail_hedge_symbol = US_TAIL_HEDGE_SYMBOL
        safe_reserve_symbol = US_SAFE_RESERVE_SYMBOL
    else:
        tail_hedge_symbol = CN_FUTURES_TAIL_HEDGE_SYMBOL
        safe_reserve_symbol = CN_FUTURES_SAFE_RESERVE_SYMBOL
    for action in actions:
        action_name = str(action.get("action"))
        target_weight = float(action.get("target_weight", 0.0))
        if action_name == "TAIL_HEDGE":
            symbol = tail_hedge_symbol
            snapshot = price_map.get(symbol)
            execution_actions.append(
                {
                    "market": market_upper,
                    "bucket_action": action_name,
                    "action": "LONG",
                    "symbol": symbol,
                    "target_weight": target_weight,
                    "stop_loss_pct": 0.03,
                    "take_profit_pct": 0.06,
                    "reference_close": float(snapshot.get("close", 0.0)) if snapshot else 0.0,
                    "reference_date": str(snapshot.get("date")) if snapshot else target_date,
                    "return_model": "close_to_close",
                    "reason": action.get("reason", ""),
                }
            )
            continue
        if action_name == "SAFE_RESERVE":
            symbol = safe_reserve_symbol
            execution_actions.append(
                {
                    "market": market_upper,
                    "bucket_action": action_name,
                    "action": "HOLD",
                    "symbol": symbol,
                    "target_weight": target_weight,
                    "reference_close": 1.0,
                    "reference_date": target_date,
                    "return_model": "cash_flat",
                    "reason": action.get("reason", ""),
                }
            )
            continue
        symbol = str(action.get("symbol"))
        snapshot = price_map.get(symbol)
        execution_actions.append(
            {
                "market": market_upper,
                "bucket_action": action_name,
                **action,
                "reference_close": float(snapshot.get("close", 0.0)) if snapshot else 0.0,
                "reference_date": str(snapshot.get("date")) if snapshot else target_date,
                "return_model": "close_to_close",
            }
        )
    return execution_actions


def _build_execution_instruction(action: Dict[str, Any]) -> str:
    action_name = str(action.get("action"))
    symbol = str(action.get("symbol"))
    if action_name == "HOLD":
        return (
            f"{action.get('bucket_action')} -> HOLD {symbol} 参考价 {float(action.get('reference_close', 1.0)):.4f}，"
            f"目标权重 {float(action.get('target_weight', 0.0)) * 100:.1f}%。"
        )
    line = _build_instruction(action, float(action.get("reference_close", 0.0)))
    return f"{action.get('bucket_action')} -> {line}"


def _execution_price_map(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    price_map: Dict[str, Dict[str, Any]] = {}
    price_map.update(report.get("us_last", {}))
    price_map.update(report.get("us_execution_last", {}))
    price_map.update(report.get("hk_last", {}))
    price_map.update(report.get("hk_execution_last", {}))
    for bucket in report.get("futures_last", {}).values():
        price_map.update(bucket)
    price_map.update(report.get("futures_execution_last", {}))
    for symbol in [US_SAFE_RESERVE_SYMBOL, HK_SAFE_RESERVE_SYMBOL, CN_FUTURES_SAFE_RESERVE_SYMBOL]:
        price_map.setdefault(symbol, _cash_price_snapshot(report.get("report_date", "")))
    return price_map


def _compute_execution_action_return(action: Dict[str, Any], current_prices: Dict[str, Dict[str, Any]]) -> Optional[float]:
    action_name = str(action.get("action"))
    if action.get("return_model") == "cash_flat" or action_name == "HOLD":
        return 0.0
    reference_close = float(action.get("reference_close", 0.0) or 0.0)
    if reference_close <= 0:
        return None
    current_close = current_prices.get(str(action.get("symbol")), {}).get("close")
    if current_close in {None, 0}:
        return None
    current_close = float(current_close)
    if action_name == "LONG":
        return current_close / reference_close - 1.0
    if action_name == "SHORT":
        return reference_close / current_close - 1.0
    return 0.0


def _evaluate_execution_actions(
    actions: List[Dict[str, Any]],
    current_prices: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    weighted_return = 0.0
    gross_weight = 0.0
    futures_weight = 0.0
    realized_returns: List[float] = []
    for action in actions:
        pnl = _compute_execution_action_return(action, current_prices)
        weight = float(action.get("target_weight", 0.0))
        symbol = str(action.get("symbol"))
        is_cash = action.get("return_model") == "cash_flat" or symbol.endswith("_CASH")
        if pnl is None:
            continue
        weighted_return += weight * pnl
        if not is_cash:
            gross_weight += weight
            if str(action.get("market")) == "SHFE":
                futures_weight += weight
            realized_returns.append(pnl)
        details.append(
            {
                "symbol": symbol,
                "action": action.get("action"),
                "bucket_action": action.get("bucket_action"),
                "target_weight": weight,
                "reference_close": float(action.get("reference_close", 0.0)),
                "current_close": float(current_prices.get(symbol, {}).get("close", action.get("reference_close", 0.0))),
                "return_pct": round(pnl, 6),
            }
        )
    wins = [item for item in realized_returns if item > 0]
    losses = [item for item in realized_returns if item < 0]
    payoff_ratio = (
        float(sum(wins) / len(wins)) / abs(float(sum(losses) / len(losses)))
        if wins and losses
        else (float("inf") if wins else 0.0)
    )
    return {
        "portfolio_return": round(weighted_return, 6),
        "gross_weight": round(gross_weight, 6),
        "futures_weight": round(futures_weight, 6),
        "risk_asset_count": len(realized_returns),
        "win_rate": round(len(wins) / len(realized_returns), 6) if realized_returns else 0.0,
        "payoff_ratio": round(payoff_ratio, 6) if payoff_ratio != float("inf") else 999.0,
        "details": details,
    }


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
    previous_actions = previous_report.get("execution_actions") or previous_report.get("primary_actions", [])
    if not previous_actions:
        return [f"{previous_date} 没有主动方向单，仅有安全端/尾部保护。"]

    lines: List[str] = []
    for action in previous_actions:
        symbol = action.get("symbol")
        action_name = action.get("action")
        if action_name == "SAFE_RESERVE":
            lines.append(f"{previous_date} 的 {symbol} {action_name} 按现金缓冲口径复盘：+0.00% (1.0000 -> 1.0000)。")
            continue
        pnl = _compute_execution_action_return(action, current_prices)
        if pnl is None:
            lines.append(f"{previous_date} 的 {symbol} 无法取得可比收盘价，跳过复盘。")
            continue
        previous_close = float(action.get("reference_close", 0.0))
        current_close = float(current_prices.get(symbol, {}).get("close", previous_close))
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


def _format_execution_bucket(actions: List[Dict[str, Any]]) -> List[str]:
    return [_build_execution_instruction(action) for action in actions]


def _next_market_day_text(target_day: date) -> str:
    next_day = _next_weekday(target_day)
    return f"按周历推算，下一个交易日预计为 {next_day.strftime('%Y-%m-%d')}；若遇交易所节假日调整，以官方日历为准。"


def _build_cycle_payload(
    market: str,
    cycle: Dict[str, Any],
    price_map: Dict[str, Dict[str, Any]],
    execution_price_map: Dict[str, Dict[str, Any]],
    target_date: str,
) -> Dict[str, Any]:
    actions = cycle.get("trade_actions", [])
    reports = {symbol: _summarize_symbol_report(item) for symbol, item in cycle.get("symbols", {}).items()}
    primary_lines, primary_actions = _classify_primary_actions(actions, price_map)
    execution_actions = _materialize_execution_actions(actions, market, execution_price_map, target_date)
    return {
        "cycle_status": cycle.get("status"),
        "actions": actions,
        "reports": reports,
        "primary_lines": primary_lines,
        "primary_actions": primary_actions,
        "secondary_lines": _format_action_bucket([item for item in actions if item.get("action") not in {"LONG", "SHORT"}]),
        "execution_actions": execution_actions,
        "execution_lines": _format_execution_bucket(execution_actions),
        "observations": _observation_lines(reports, price_map),
    }


def _append_market_summary(
    report: Dict[str, Any],
    label: str,
    prefix: str,
    section_lines: List[str],
) -> None:
    primary_key = f"{prefix}_primary_lines"
    secondary_key = f"{prefix}_secondary_lines"
    execution_key = f"{prefix}_execution_lines"
    observation_key = f"{prefix}_observations"

    if report.get(primary_key):
        section_lines.append(f"- {label}：")
        section_lines.extend([f"  {line}" for line in report.get(primary_key, [])])
    else:
        section_lines.append(f"- {label}：没有通过 `RS>70 且 R²>0.7` 的主动仓信号。")

    for line in report.get(secondary_key, []):
        section_lines.append(f"- {label}安全端：{line}")
    for line in report.get(execution_key, []):
        section_lines.append(f"- {label}执行：{line}")
    for line in report.get(observation_key, []):
        section_lines.append(f"- {label}观察：{line}")


def generate_report(target_day: date) -> Dict[str, Any]:
    repo_root = _repo_root()
    report_dir = _state_dir(repo_root)
    report_dir.mkdir(parents=True, exist_ok=True)

    ak = _require_module("akshare")
    yf = _require_module("yfinance")

    repo_status = _repo_status(repo_root)
    target_date = target_day.strftime("%Y-%m-%d")
    us_universe = _us_core_universe()
    hk_universe = _hk_universe()
    futures_universe_by_exchange = _cn_futures_exchange_universe()

    us_data = _fetch_us_data(yf, us_universe, end_date=target_date)
    us_validation = _validate_us_close(us_data, target_day)
    us_execution_data = _fetch_us_data(yf, US_EXECUTION_SUPPORT_UNIVERSE, end_date=target_date)

    hk_data = _fetch_hk_data(ak, hk_universe, end_date=target_date)
    hk_validation = _validate_hk_close(hk_data, target_date)
    hk_execution_data = _fetch_hk_data(ak, HK_EXECUTION_SUPPORT_UNIVERSE, end_date=target_date)

    futures_market_data: Dict[str, Dict[str, Any]] = {}
    futures_last: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for exchange, symbols in futures_universe_by_exchange.items():
        _, settle_validation = _resolve_futures_settle(ak, exchange, target_day)
        exchange_data = _fetch_futures_data(ak, symbols, end_date=target_date)
        validation = _validate_futures_close(exchange, exchange_data, settle_validation)
        futures_market_data[exchange] = {
            "validation": validation,
            "data": exchange_data,
        }
        futures_last[exchange] = _latest_price_map(exchange_data)
    futures_execution_data = _fetch_futures_data(ak, [CN_FUTURES_TAIL_HEDGE_SYMBOL], end_date=target_date)
    futures_execution_last = _latest_price_map(futures_execution_data)

    report: Dict[str, Any] = {
        "status": "failed_validation",
        "report_date": target_date,
        "generated_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "repo": repo_status,
        "us_validation": asdict(us_validation),
        "hk_validation": asdict(hk_validation),
        "futures_validations": {
            exchange: asdict(item["validation"])
            for exchange, item in futures_market_data.items()
        },
        "us_last": _latest_price_map(us_data),
        "hk_last": _latest_price_map(hk_data),
        "futures_last": futures_last,
        "us_execution_last": _latest_price_map(us_execution_data),
        "hk_execution_last": _latest_price_map(hk_execution_data),
        "futures_execution_last": futures_execution_last,
        "recap_lines": [],
        "primary_actions": [],
        "report_text": "",
    }

    previous_report = _load_json(_find_previous_report(report_dir, target_day))
    current_price_map = {}
    current_price_map.update(report["us_last"])
    current_price_map.update(report["us_execution_last"])
    current_price_map.update(report["hk_last"])
    current_price_map.update(report["hk_execution_last"])
    for exchange_prices in report["futures_last"].values():
        current_price_map.update(exchange_prices)
    current_price_map.update(report["futures_execution_last"])
    current_price_map[US_SAFE_RESERVE_SYMBOL] = _cash_price_snapshot(target_date)
    current_price_map[HK_SAFE_RESERVE_SYMBOL] = _cash_price_snapshot(target_date)
    current_price_map[CN_FUTURES_SAFE_RESERVE_SYMBOL] = _cash_price_snapshot(target_date)
    report["recap_lines"] = _build_recap(previous_report, current_price_map)

    if not us_validation.passed or not hk_validation.passed or not all(
        item["validation"].passed for item in futures_market_data.values()
    ):
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
    copper_proxy = next(
        (
            item["data"]["CU0"]
            for item in futures_market_data.values()
            if "CU0" in item["data"]
        ),
        copper_peer,
    )
    market_context, market_data = _build_market_context(cross_asset_engine, qqq, gold, copper_proxy)
    us_cycle = self_iterating_engine.run_learning_cycle(
        us_data,
        benchmark_frame=qqq,
        market_context=market_context,
        global_peer_datasets={},
    )
    hk_cycle = self_iterating_engine.run_learning_cycle(
        hk_data,
        benchmark_frame=None,
        market_context=market_context,
        global_peer_datasets={},
    )
    futures_cycles: Dict[str, Dict[str, Any]] = {}
    for exchange, payload in futures_market_data.items():
        exchange_data = payload["data"]
        futures_cycles[exchange] = self_iterating_engine.run_learning_cycle(
            exchange_data,
            benchmark_frame=None,
            market_context=market_context,
            global_peer_datasets=_build_futures_peer_datasets(exchange_data, gold, copper_peer),
        )
    legacy = trading_agent.execute_decision(current_date=target_date, market_data=market_data)

    report.update(
        {
            "status": "ok",
            "market_context": market_context,
            "market_data": market_data,
            "us_cycle_status": us_cycle.get("status"),
            "hk_cycle_status": hk_cycle.get("status"),
            "futures_cycle_status": {
                exchange: cycle.get("status")
                for exchange, cycle in futures_cycles.items()
            },
            "us_actions": us_cycle.get("trade_actions", []),
            "hk_actions": hk_cycle.get("trade_actions", []),
            "legacy_actions": legacy.get("actions", []),
            "us_reports": {symbol: _summarize_symbol_report(item) for symbol, item in us_cycle.get("symbols", {}).items()},
            "hk_reports": {symbol: _summarize_symbol_report(item) for symbol, item in hk_cycle.get("symbols", {}).items()},
            "futures_reports": {
                exchange: {
                    symbol: _summarize_symbol_report(item)
                    for symbol, item in cycle.get("symbols", {}).items()
                }
                for exchange, cycle in futures_cycles.items()
            },
            "calendar_note": _next_market_day_text(target_day),
        }
    )

    us_execution_price_map = {
        **report["us_last"],
        **report["us_execution_last"],
        US_SAFE_RESERVE_SYMBOL: _cash_price_snapshot(target_date),
    }
    hk_execution_price_map = {**report["hk_last"], **report["hk_execution_last"], HK_SAFE_RESERVE_SYMBOL: _cash_price_snapshot(target_date)}
    report.update(
        {
            f"us_{key}": value
            for key, value in _build_cycle_payload(
                "US",
                us_cycle,
                report["us_last"],
                us_execution_price_map,
                target_date,
            ).items()
        }
    )
    report.update(
        {
            f"hk_{key}": value
            for key, value in _build_cycle_payload(
                "HK",
                hk_cycle,
                report["hk_last"],
                hk_execution_price_map,
                target_date,
            ).items()
        }
    )

    futures_execution_actions: List[Dict[str, Any]] = []
    futures_primary_actions: List[Dict[str, Any]] = []
    for exchange, cycle in futures_cycles.items():
        exchange_price_map = report["futures_last"].get(exchange, {})
        execution_price_map = {
            **exchange_price_map,
            **report["futures_execution_last"],
            CN_FUTURES_SAFE_RESERVE_SYMBOL: _cash_price_snapshot(target_date),
        }
        payload = _build_cycle_payload(exchange, cycle, exchange_price_map, execution_price_map, target_date)
        report[f"{exchange.lower()}_actions"] = payload["actions"]
        report[f"{exchange.lower()}_reports"] = payload["reports"]
        report[f"{exchange.lower()}_primary_lines"] = payload["primary_lines"]
        report[f"{exchange.lower()}_secondary_lines"] = payload["secondary_lines"]
        report[f"{exchange.lower()}_execution_actions"] = payload["execution_actions"]
        report[f"{exchange.lower()}_execution_lines"] = payload["execution_lines"]
        report[f"{exchange.lower()}_observations"] = payload["observations"]
        futures_execution_actions.extend(payload["execution_actions"])
        futures_primary_actions.extend(
            [
                {
                    **action,
                    "market": exchange,
                    "reference_close": float(exchange_price_map.get(action["symbol"], {}).get("close", 0.0)),
                }
                for action in payload["primary_actions"]
            ]
        )

    report["execution_actions"] = [
        *report["us_execution_actions"],
        *report["hk_execution_actions"],
        *futures_execution_actions,
    ]
    report["primary_actions"] = [
        *[
            {
                **action,
                "market": "US",
                "reference_close": float(report["us_last"].get(action["symbol"], {}).get("close", 0.0)),
            }
            for action in report["us_primary_actions"]
        ],
        *[
            {
                **action,
                "market": "HK",
                "reference_close": float(report["hk_last"].get(action["symbol"], {}).get("close", 0.0)),
            }
            for action in report["hk_primary_actions"]
        ],
        *futures_primary_actions,
    ]
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

    us_validation = report.get("us_validation", {})
    hk_validation = report.get("hk_validation", {})
    futures_validations = report.get("futures_validations", {})
    lines.append("")
    lines.append("数据校验：")
    lines.append(
        f"美股：passed={us_validation.get('passed')} requested={us_validation.get('requested_date')} "
        f"actual={us_validation.get('actual_date')}；{us_validation.get('reason')}"
    )
    lines.append(
        f"港股：passed={hk_validation.get('passed')} requested={hk_validation.get('requested_date')} "
        f"actual={hk_validation.get('actual_date')}；{hk_validation.get('reason')}"
    )
    for exchange in CN_FUTURES_EXCHANGES:
        validation = futures_validations.get(exchange, {})
        lines.append(
            f"{exchange}：passed={validation.get('passed')} requested={validation.get('requested_date')} "
            f"actual={validation.get('actual_date')}；{validation.get('reason')}"
        )

    lines.append("")
    lines.append("复盘：")
    lines.extend([f"- {line}" for line in report.get("recap_lines", [])])

    if report.get("status") != "ok":
        lines.append("")
        lines.append("结果：")
        lines.append("- 本次任务因至少一个市场或期货交易所校验未全部通过，按硬规则直接失败退出，不生成新的美股/港股/中国期货交易指令。")
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
    _append_market_summary(report, "美股", "us", lines)
    _append_market_summary(report, "港股", "hk", lines)
    for exchange in CN_FUTURES_EXCHANGES:
        _append_market_summary(report, exchange, exchange.lower(), lines)

    legacy_lines = report.get("legacy_lines", [])
    if legacy_lines:
        lines.append("")
        lines.append("兜底偏向：")
        for line in legacy_lines:
            lines.append(f"- {line}")

    return "\n".join(lines)


def _load_report_files(report_dir: Path, week_start: date, week_end: date) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for path in sorted(report_dir.glob("*.json")):
        try:
            stamp = datetime.strptime(path.stem, "%Y-%m-%d").date()
        except ValueError:
            continue
        if week_start <= stamp <= week_end:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
    reports.sort(key=lambda item: str(item.get("report_date")))
    return reports


def _fetch_historical_price_maps(target_date: str, hk_symbols: Iterable[str], shfe_symbols: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    ak = _require_module("akshare")
    price_map: Dict[str, Dict[str, Any]] = {
        HK_SAFE_RESERVE_SYMBOL: _cash_price_snapshot(target_date),
        SHFE_SAFE_RESERVE_SYMBOL: _cash_price_snapshot(target_date),
    }
    for symbol in hk_symbols:
        if symbol.endswith("_CASH"):
            continue
        code = symbol.split(".")[0]
        frame = _normalize_live_frame(ak.stock_hk_daily(symbol=code))
        snapshot = _price_on_or_before(frame, target_date)
        if snapshot:
            price_map[symbol] = snapshot
    for symbol in shfe_symbols:
        if symbol.endswith("_CASH"):
            continue
        frame = _normalize_live_frame(ak.futures_zh_daily_sina(symbol=symbol))
        snapshot = _price_on_or_before(frame, target_date)
        if snapshot:
            price_map[symbol] = snapshot
    return price_map


def generate_weekly_execution_review(
    week_start: date,
    week_end: date,
    evaluation_date: Optional[date] = None,
) -> Dict[str, Any]:
    repo_root = _repo_root()
    report_dir = _state_dir(repo_root)
    reports = _load_report_files(report_dir, week_start, week_end)
    evaluation_date = evaluation_date or week_end

    if not reports:
        return {
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
            "evaluation_date": evaluation_date.isoformat(),
            "report_count": 0,
            "days": [],
            "combined_nav_end": 1.0,
            "combined_return": 0.0,
        }

    hk_nav = 1.0
    shfe_nav = 1.0
    combined_nav = 1.0
    nav_curve = [1.0]
    day_reviews: List[Dict[str, Any]] = []
    constraint_checks = {
        "max_single_weight_ok": True,
        "max_futures_weight_ok": True,
        "max_gross_weight_ok": True,
    }

    for index, report in enumerate(reports):
        next_report = reports[index + 1] if index + 1 < len(reports) else None
        mark_date = str(next_report.get("report_date")) if next_report else evaluation_date.isoformat()

        execution_actions = report.get("execution_actions")
        if not execution_actions:
            hk_price_map = {**report.get("hk_last", {}), **report.get("hk_execution_last", {}), HK_SAFE_RESERVE_SYMBOL: _cash_price_snapshot(str(report.get("report_date")))}
            shfe_price_map = {**report.get("shfe_last", {}), SHFE_SAFE_RESERVE_SYMBOL: _cash_price_snapshot(str(report.get("report_date")))}
            execution_actions = [
                *_materialize_execution_actions(report.get("hk_actions", []), "HK", hk_price_map, str(report.get("report_date"))),
                *_materialize_execution_actions(report.get("shfe_actions", []), "SHFE", shfe_price_map, str(report.get("report_date"))),
            ]

        hk_symbols = sorted({str(item.get("symbol")) for item in execution_actions if str(item.get("market")) == "HK"})
        shfe_symbols = sorted({str(item.get("symbol")) for item in execution_actions if str(item.get("market")) == "SHFE"})
        current_prices = _fetch_historical_price_maps(mark_date, hk_symbols, shfe_symbols)
        hk_actions = [item for item in execution_actions if str(item.get("market")) == "HK"]
        shfe_actions = [item for item in execution_actions if str(item.get("market")) == "SHFE"]
        hk_eval = _evaluate_execution_actions(hk_actions, current_prices)
        shfe_eval = _evaluate_execution_actions(shfe_actions, current_prices)

        for bucket in [hk_actions, shfe_actions]:
            for action in bucket:
                weight = float(action.get("target_weight", 0.0))
                is_cash = action.get("return_model") == "cash_flat" or str(action.get("symbol")).endswith("_CASH")
                if not is_cash:
                    constraint_checks["max_single_weight_ok"] &= weight <= 0.20 + 1e-9
        constraint_checks["max_futures_weight_ok"] &= shfe_eval["gross_weight"] <= 0.50 + 1e-9
        constraint_checks["max_gross_weight_ok"] &= hk_eval["gross_weight"] <= 1.00 + 1e-9 and shfe_eval["gross_weight"] <= 1.00 + 1e-9

        hk_nav *= 1.0 + float(hk_eval["portfolio_return"])
        shfe_nav *= 1.0 + float(shfe_eval["portfolio_return"])
        combined_daily_return = 0.5 * float(hk_eval["portfolio_return"]) + 0.5 * float(shfe_eval["portfolio_return"])
        combined_nav *= 1.0 + combined_daily_return
        nav_curve.append(combined_nav)

        day_reviews.append(
            {
                "report_date": report.get("report_date"),
                "mark_date": mark_date,
                "hk": hk_eval,
                "shfe": shfe_eval,
                "combined_return": round(combined_daily_return, 6),
                "combined_nav": round(combined_nav, 6),
            }
        )

    peaks: List[float] = []
    rolling_peak = 1.0
    max_drawdown = 0.0
    for nav in nav_curve:
        rolling_peak = max(rolling_peak, nav)
        peaks.append(rolling_peak)
        drawdown = nav / rolling_peak - 1.0
        max_drawdown = min(max_drawdown, drawdown)

    return {
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "evaluation_date": evaluation_date.isoformat(),
        "report_count": len(reports),
        "aggregation_assumption": "HK 与 SHFE 两个子组合按 50/50 等权聚合；SAFE_RESERVE 现金腿按 0 收益计。",
        "days": day_reviews,
        "hk_nav_end": round(hk_nav, 6),
        "shfe_nav_end": round(shfe_nav, 6),
        "combined_nav_end": round(combined_nav, 6),
        "hk_return": round(hk_nav - 1.0, 6),
        "shfe_return": round(shfe_nav - 1.0, 6),
        "combined_return": round(combined_nav - 1.0, 6),
        "max_drawdown": round(max_drawdown, 6),
        "constraint_checks": constraint_checks,
    }


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
