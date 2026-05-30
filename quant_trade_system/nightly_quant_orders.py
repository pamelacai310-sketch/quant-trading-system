from __future__ import annotations

import argparse
import json
import os
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
from .core.causal import (
    CausalFactorLibrary,
    CrossAssetCausalEngine,
    GameCausalAnalysisEngine,
    MacroEventStateEngine,
    SelfIteratingCausalEngine,
)
from .factors.factor_library import FactorLibrary
from .futures_specs import build_one_lot_margin_table
from .universe_provider import MarketUniverseProvider


CHINA_TZ = ZoneInfo("Asia/Shanghai")
STATE_DIRNAME = "nightly_reports"
LOG_DIRNAME = "nightly_logs"

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
FUTURES_SETTLE_FALLBACK_ENV = "QTS_FUTURES_SETTLE_FALLBACK"
FUTURES_SETTLE_FALLBACK_DAILY = "daily_main_contract"
CFFEX_SETTLE_FALLBACK_CONTRACT_SPECS = "daily_main_contract+contract_specs"
US_CLOSE_AVAILABLE_CHINA_HOUR = 5
US_CLOSE_AVAILABLE_CHINA_MINUTE = 30
FAILURE_NONE = "NONE"
FAILURE_SCHEDULER_NOT_RUN = "SCHEDULER_NOT_RUN"
FAILURE_DATA_VALIDATION_PARTIAL = "DATA_VALIDATION_PARTIAL"
FAILURE_ALL_MARKETS_INVALID = "ALL_MARKETS_INVALID"
FAILURE_RUNTIME_EXCEPTION = "RUNTIME_EXCEPTION"


@dataclass
class MarketValidation:
    market: str
    requested_date: str
    actual_date: Optional[str]
    passed: bool
    reason: str
    sample_count: int = 0
    settlement_fallback: Optional[str] = None
    margin_source: Optional[str] = None


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


def _log_dir(repo_root: Path) -> Path:
    return repo_root / "state" / LOG_DIRNAME


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


def _optional_yf_frame(yf: Any, symbol: str, period: str = "3mo") -> pd.DataFrame:
    try:
        raw = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    try:
        return _normalize_yfinance_frame(raw)
    except Exception:
        return pd.DataFrame()


def _latest_close(frame: pd.DataFrame) -> Optional[float]:
    if frame is None or frame.empty or "close" not in frame.columns:
        return None
    values = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _env_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _env_bool(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None:
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "y", "signed"}


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


def _us_t_close_available(target_day: date, as_of: Optional[datetime] = None) -> bool:
    """Return whether the US cash close for target_day should be available in China time."""
    current = as_of or datetime.now(CHINA_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=CHINA_TZ)
    current = current.astimezone(CHINA_TZ)
    current_day = current.date()
    if current_day > target_day + timedelta(days=1):
        return True
    if current_day == target_day + timedelta(days=1):
        marker = current.hour * 60 + current.minute
        close_marker = US_CLOSE_AVAILABLE_CHINA_HOUR * 60 + US_CLOSE_AVAILABLE_CHINA_MINUTE
        return marker >= close_marker
    return False


def _futures_settle_fallback_mode() -> str:
    return os.getenv(FUTURES_SETTLE_FALLBACK_ENV, FUTURES_SETTLE_FALLBACK_DAILY).strip().lower()


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


def _validate_us_close(
    us_data: Dict[str, pd.DataFrame],
    target_day: date,
    max_lookback: int = 4,
    as_of: Optional[datetime] = None,
) -> MarketValidation:
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

    if actual_date == target_date and _us_t_close_available(target_day, as_of=as_of):
        return MarketValidation(
            market="US",
            requested_date=target_date,
            actual_date=actual_date,
            passed=True,
            reason="美股样本最新日线一致落在 T 日，且运行时点已过美股现金市场收盘可用窗口。",
            sample_count=sample_count,
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
    actual_date, sample_count = _latest_sample_date(futures_data)
    if (
        exchange.upper() == "CFFEX"
        and actual_date
        and actual_date == settle_validation.requested_date
        and settle_validation.actual_date != actual_date
    ):
        if settle_validation.actual_date:
            settle_note = f"结算参数未命中 T 日（最近有效日期 {settle_validation.actual_date}）"
        else:
            settle_note = "结算参数不可用"
        return MarketValidation(
            market=exchange,
            requested_date=settle_validation.requested_date,
            actual_date=actual_date,
            passed=True,
            reason=(
                f"{exchange} {settle_note}，但主力连续合约日线命中 T 日；"
                f"settlement_fallback={CFFEX_SETTLE_FALLBACK_CONTRACT_SPECS}，"
                "最低保证金使用项目内置合约乘数和交易所保证金率估算。"
            ),
            sample_count=sample_count,
            settlement_fallback=CFFEX_SETTLE_FALLBACK_CONTRACT_SPECS,
            margin_source="contract_specs",
        )
    if not settle_validation.passed or settle_validation.actual_date is None:
        fallback_mode = _futures_settle_fallback_mode()
        if actual_date and fallback_mode in {FUTURES_SETTLE_FALLBACK_DAILY, "daily", "main_contract", "recent_daily"}:
            requested_date = settle_validation.requested_date
            if actual_date == requested_date:
                reason = (
                    f"{exchange} 结算参数不可用，按 {FUTURES_SETTLE_FALLBACK_ENV}={fallback_mode} "
                    "显式降级到主力连续合约 T 日日线。"
                )
            else:
                reason = (
                    f"{exchange} 结算参数不可用，按 {FUTURES_SETTLE_FALLBACK_ENV}={fallback_mode} "
                    f"显式降级到主力连续合约最近有效交易日 {actual_date}。"
                )
            return MarketValidation(
                market=exchange,
                requested_date=requested_date,
                actual_date=actual_date,
                passed=True,
                reason=reason,
                sample_count=sample_count,
                settlement_fallback=fallback_mode,
            )
        return settle_validation
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


def _market_status_from_validation(validation: MarketValidation) -> Dict[str, Any]:
    status = {
        "status": "OK" if validation.passed else "NO_TRADE_DATA_INVALID",
        "tradable": bool(validation.passed),
        "actual_date": validation.actual_date,
        "reason": validation.reason,
    }
    if validation.settlement_fallback:
        status["settlement_fallback"] = validation.settlement_fallback
    if validation.margin_source:
        status["margin_source"] = validation.margin_source
    return status


def _build_market_status(
    us_validation: MarketValidation,
    hk_validation: MarketValidation,
    futures_market_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    status = {
        "US": _market_status_from_validation(us_validation),
        "HK": _market_status_from_validation(hk_validation),
    }
    for exchange, payload in futures_market_data.items():
        status[exchange] = _market_status_from_validation(payload["validation"])
    return status


def _market_is_tradable(market_status: Dict[str, Dict[str, Any]], market: str) -> bool:
    return bool(market_status.get(market.upper(), {}).get("tradable"))


def _any_market_tradable(market_status: Dict[str, Dict[str, Any]]) -> bool:
    return any(bool(item.get("tradable")) for item in market_status.values())


def _all_markets_tradable(market_status: Dict[str, Dict[str, Any]]) -> bool:
    return all(bool(item.get("tradable")) for item in market_status.values())


def _failure_category_from_market_status(market_status: Dict[str, Dict[str, Any]]) -> str:
    if not market_status:
        return FAILURE_SCHEDULER_NOT_RUN
    tradable = [bool(item.get("tradable")) for item in market_status.values()]
    if all(tradable):
        return FAILURE_NONE
    if any(tradable):
        return FAILURE_DATA_VALIDATION_PARTIAL
    return FAILURE_ALL_MARKETS_INVALID


def _real_rates_rising(market_context: Dict[str, Any]) -> bool:
    explicit = market_context.get("real_rates_rising")
    if explicit is not None:
        return bool(explicit)
    direction = str(market_context.get("real_rates_direction", "")).lower()
    if direction in {"up", "rising", "higher"}:
        return True
    macro = market_context.get("macro_signals", {})
    if isinstance(macro, dict):
        return str(macro.get("real_rates_direction", "")).lower() in {"up", "rising", "higher"}
    return False


def _tail_hedge_effectiveness_gate(
    frame: Optional[pd.DataFrame],
    market_context: Dict[str, Any],
    short_window: int = 20,
) -> Dict[str, Any]:
    if frame is None or frame.empty or len(frame) < max(5, short_window):
        return {
            "active": True,
            "mode": "tail_hedge",
            "reason": "尾部保护价格历史不足，保持默认保护腿但保留复盘监控。",
        }
    close = frame["close"].astype(float)
    latest_close = float(close.iloc[-1])
    short_ma = float(close.rolling(short_window, min_periods=max(5, short_window // 2)).mean().iloc[-1])
    real_rates_up = _real_rates_rising(market_context)
    if latest_close < short_ma and real_rates_up:
        return {
            "active": False,
            "mode": "cash_instead_of_gold",
            "latest_close": round(latest_close, 6),
            "short_ma": round(short_ma, 6),
            "real_rates_rising": True,
            "reason": "黄金/黄金ETF低于短均线且实际利率上行，尾部保护有效性不足，切换为现金安全端。",
        }
    return {
        "active": True,
        "mode": "tail_hedge",
        "latest_close": round(latest_close, 6),
        "short_ma": round(short_ma, 6),
        "real_rates_rising": real_rates_up,
        "reason": "尾部保护价格趋势或实际利率条件未触发失效门控。",
    }


def _build_macro_event_inputs(
    yf: Any,
    futures_market_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build optional macro/event signals without blocking nightly execution."""
    asset_signals: Dict[str, Dict[str, Any]] = {}
    result: Dict[str, Any] = {"asset_signals": asset_signals, "symbol_tags": {}}

    cffex_data = futures_market_data.get("CFFEX", {}).get("data", {})
    im_frame = cffex_data.get("IM0")
    ic_frame = cffex_data.get("IC0")
    if im_frame is not None and ic_frame is not None and not im_frame.empty and not ic_frame.empty:
        im_momentum = _pct_change_over_window(im_frame, 20)
        ic_momentum = _pct_change_over_window(ic_frame, 20)
        result["csi1000_500_excess_return"] = round(im_momentum - ic_momentum, 6)
        asset_signals["CSI1000"] = _directional_signal(im_momentum)
        asset_signals["CSI500"] = _directional_signal(ic_momentum)
        asset_signals["IM0"] = asset_signals["CSI1000"]
        asset_signals["IC0"] = asset_signals["CSI500"]
        result["symbol_tags"]["IM0"] = ["csi1000", "small_cap", "ai_industrial_chain_proxy"]
        result["symbol_tags"]["IC0"] = ["csi500", "mid_cap"]

    for symbol, key in [("^TNX", "us10y_yield"), ("^TYX", "us30y_yield"), ("^MOVE", "move_index")]:
        close = _latest_close(_optional_yf_frame(yf, symbol))
        if close is not None:
            result[key] = close

    dxy_frame = _optional_yf_frame(yf, "DX-Y.NYB")
    if dxy_frame.empty:
        dxy_frame = _optional_yf_frame(yf, "DX=F")
    if not dxy_frame.empty:
        dxy_return = _pct_change_over_window(dxy_frame, 5)
        result["dxy_return_5d"] = round(dxy_return, 6)
        asset_signals["DXY"] = _directional_signal(dxy_return)

    usdcny_frame = _optional_yf_frame(yf, "USDCNY=X")
    if not usdcny_frame.empty:
        asset_signals["USDCNY"] = _directional_signal(_pct_change_over_window(usdcny_frame, 5))

    for key, env_name in {
        "sofr_5d_change": "QTS_SOFR_5D_CHANGE",
        "bond_straddle_activity": "QTS_BOND_STRADDLE_ACTIVITY",
        "hormuz_reopen_probability": "QTS_HORMUZ_REOPEN_PROBABILITY",
    }.items():
        env_value = _env_float(env_name)
        if env_value is not None:
            result[key] = env_value
    signed = _env_bool("QTS_HORMUZ_REOPEN_SIGNED")
    if signed is not None:
        result["hormuz_reopen_signed"] = signed
    if "hormuz_reopen_probability" in result or "hormuz_reopen_signed" in result:
        result["event_probabilities"] = {
            "hormuz_reopen_probability": result.get("hormuz_reopen_probability", 0.0),
            "hormuz_reopen_signed": result.get("hormuz_reopen_signed", False),
        }
    return result


def _build_market_context(
    cross_asset_engine: CrossAssetCausalEngine,
    qqq: pd.DataFrame,
    gold: pd.DataFrame,
    copper: pd.DataFrame,
    macro_event_inputs: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
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
        "asset_signals": {
            "Nasdaq": _directional_signal(qqq_mom),
            "SOX": _directional_signal(qqq_mom),
            "risk_assets": _directional_signal(qqq_mom),
            "gold": _directional_signal(gold_mom),
            "copper": _directional_signal(copper_mom),
        },
        "real_rates_direction": "up" if gold_mom < 0 and copper_mom < 0 else "flat_or_down",
    }
    macro_event_inputs = macro_event_inputs or {}
    for key, value in macro_event_inputs.items():
        if key == "asset_signals" and isinstance(value, dict):
            market_context["asset_signals"].update(value)
        elif key == "symbol_tags" and isinstance(value, dict):
            market_context.setdefault("symbol_tags", {}).update(value)
        else:
            market_context[key] = value
    macro_event_state = MacroEventStateEngine().analyze(market_context)
    market_context["macro_event_state"] = macro_event_state
    market_context["crisis_probability"] = round(
        max(float(market_context["crisis_probability"]), float(macro_event_state.get("tail_risk_score", 0.0) or 0.0)),
        4,
    )
    game_analysis = GameCausalAnalysisEngine().analyze(
        news_items=[],
        policy_records=[],
        market_context=market_context,
    )
    market_context["game_causal_analysis"] = game_analysis
    market_data = {
        "US_Debt": {"value": 38_500_000_000_000},
        "Central_Bank_Gold_Purchase": {"value": round(78 + max(gold_mom, 0.0) * 120, 2)},
        "ON_RRP_Balance": {
            "value": max(350_000_000_000, 620_000_000_000 - max(qqq_mom, 0.0) * 400_000_000_000)
        },
        "LME_Inventory_Days": {"value": round(max(1.5, min(12.0, 5.5 - max(copper_mom, 0.0) * 14)), 2)},
        "AI_DataCenter_Capex": {"growth": round(max(0.08, 0.12 + max(qqq_mom, 0.0) * 0.9), 4)},
        "cross_asset_regime": market_context["cross_asset_regime"],
        "asset_signals": market_context["asset_signals"],
        "game_causal_analysis": game_analysis,
        "dominant_game_logics": game_analysis.get("dominant_game_logics", []),
        "game_relation_reports": game_analysis.get("game_relation_reports", []),
        "macro_event_state": macro_event_state,
    }
    return market_context, market_data


def _directional_signal(momentum: float, deadband: float = 0.005) -> Dict[str, Any]:
    if momentum > deadband:
        direction = "up"
    elif momentum < -deadband:
        direction = "down"
    else:
        direction = "flat"
    return {
        "direction": direction,
        "return": round(float(momentum), 6),
        "momentum": round(float(momentum), 6),
    }


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
    tail_hedge_gate: Optional[Dict[str, Any]] = None,
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
            gate = tail_hedge_gate or {"active": True, "mode": "tail_hedge"}
            if not bool(gate.get("active", True)):
                execution_actions.append(
                    {
                        "market": market_upper,
                        "bucket_action": action_name,
                        "action": "HOLD",
                        "symbol": safe_reserve_symbol,
                        "target_weight": target_weight,
                        "reference_close": 1.0,
                        "reference_date": target_date,
                        "return_model": "cash_flat",
                        "reason": f"{action.get('reason', '')}；{gate.get('reason', '尾部保护有效性门控切换现金')}",
                        "tail_hedge_gate": gate,
                    }
                )
                continue
            symbol = tail_hedge_symbol
            snapshot = price_map.get(symbol)
            reference_close = float(snapshot.get("close", 0.0)) if snapshot else 0.0
            execution_actions.append(
                {
                    "market": market_upper,
                    "bucket_action": action_name,
                    "action": "LONG",
                    "symbol": symbol,
                    "target_weight": target_weight,
                    "stop_loss_pct": 0.03,
                    "take_profit_pct": 0.06,
                    "reference_close": reference_close,
                    "reference_date": str(snapshot.get("date")) if snapshot else target_date,
                    "return_model": "close_to_close",
                    "reason": action.get("reason", ""),
                    "tail_hedge_gate": gate,
                    **_futures_margin_fields(market_upper, symbol, reference_close, action.get("margin_rate")),
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
        reference_close = float(snapshot.get("close", 0.0)) if snapshot else 0.0
        futures_margin = _futures_margin_fields(market_upper, symbol, reference_close, action.get("margin_rate"))
        execution_actions.append(
            {
                "market": market_upper,
                "bucket_action": action_name,
                **action,
                "reference_close": reference_close,
                "reference_date": str(snapshot.get("date")) if snapshot else target_date,
                "return_model": "close_to_close",
                **futures_margin,
            }
        )
    return execution_actions


def _futures_margin_fields(
    market: str,
    symbol: str,
    reference_close: float,
    margin_rate: Optional[float] = None,
) -> Dict[str, Any]:
    if market in {"US", "HK"} or reference_close <= 0:
        return {}
    overrides = {symbol: float(margin_rate)} if margin_rate is not None else None
    row = build_one_lot_margin_table({symbol: reference_close}, overrides)[0]
    return {
        "contract_multiplier": row["contract_multiplier"],
        "margin_rate": row["margin_rate"],
        "one_lot_notional": row["one_lot_notional"],
        "one_lot_min_margin": row["one_lot_margin"],
        "margin_formula": "latest_price * contract_multiplier * margin_rate",
        "margin_source": "contract_specs",
    }


def _build_execution_instruction(action: Dict[str, Any]) -> str:
    action_name = str(action.get("action"))
    symbol = str(action.get("symbol"))
    if action_name == "HOLD":
        return (
            f"{action.get('bucket_action')} -> HOLD {symbol} 参考价 {float(action.get('reference_close', 1.0)):.4f}，"
            f"目标权重 {float(action.get('target_weight', 0.0)) * 100:.1f}%。"
        )
    line = _build_instruction(action, float(action.get("reference_close", 0.0)))
    if action.get("one_lot_min_margin") is not None:
        line = (
            f"{line}，1手最低保证金 {float(action.get('one_lot_min_margin', 0.0)):,.2f} "
            f"= {float(action.get('reference_close', 0.0)):,.2f} * "
            f"{float(action.get('contract_multiplier', 1.0)):g} * "
            f"{float(action.get('margin_rate', 0.0)):.2%}"
        )
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
    price_unavailable_count = 0
    for action in actions:
        pnl = _compute_execution_action_return(action, current_prices)
        weight = float(action.get("target_weight", 0.0))
        symbol = str(action.get("symbol"))
        is_cash = action.get("return_model") == "cash_flat" or symbol.endswith("_CASH")
        if pnl is None:
            price_unavailable_count += 1
            details.append(
                {
                    "symbol": symbol,
                    "action": action.get("action"),
                    "bucket_action": action.get("bucket_action"),
                    "target_weight": weight,
                    "reference_close": float(action.get("reference_close", 0.0)),
                    "current_close": None,
                    "return_pct": None,
                    "price_status": "price_unavailable",
                }
            )
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
                "price_status": "ok",
            }
        )
    wins = [item for item in realized_returns if item > 0]
    losses = [item for item in realized_returns if item < 0]
    payoff_ratio = (
        float(sum(wins) / len(wins)) / abs(float(sum(losses) / len(losses)))
        if wins and losses
        else (float("inf") if wins else 0.0)
    )
    avg_abs_return = float(sum(abs(item) for item in realized_returns) / len(realized_returns)) if realized_returns else 0.0
    elasticity = avg_abs_return / max(abs(weighted_return), 1e-9) if realized_returns and abs(weighted_return) > 1e-9 else 0.0
    return {
        "portfolio_return": round(weighted_return, 6),
        "gross_weight": round(gross_weight, 6),
        "futures_weight": round(futures_weight, 6),
        "risk_asset_count": len(realized_returns),
        "win_rate": round(len(wins) / len(realized_returns), 6) if realized_returns else 0.0,
        "payoff_ratio": round(payoff_ratio, 6) if payoff_ratio != float("inf") else 999.0,
        "elasticity": round(float(elasticity), 6),
        "slippage_bps": 0.0,
        "execution_quality": "close_to_close_proxy",
        "price_unavailable_count": price_unavailable_count,
        "failure_attribution": (
            "price_unavailable"
            if price_unavailable_count and not realized_returns
            else ("partial_price_unavailable" if price_unavailable_count else ("not_evaluated" if realized_returns else "no_directional_fill"))
        ),
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
    tail_hedge_gate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    actions = cycle.get("trade_actions", [])
    reports = {symbol: _summarize_symbol_report(item) for symbol, item in cycle.get("symbols", {}).items()}
    primary_lines, primary_actions = _classify_primary_actions(actions, price_map)
    execution_actions = _materialize_execution_actions(actions, market, execution_price_map, target_date, tail_hedge_gate)
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


def _build_evidence_snapshot(
    report: Dict[str, Any],
    cycles: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    market_data = report.get("market_data", {})
    game_analysis = market_data.get("game_causal_analysis", {})
    relation_reports = game_analysis.get("game_relation_reports", [])
    top_relations = sorted(
        relation_reports,
        key=lambda item: float(item.get("current_judgement", {}).get("confidence", 0.0) or 0.0),
        reverse=True,
    )[:6]
    model_records = {
        name: cycle.get("model_registry_record", {})
        for name, cycle in cycles.items()
    }
    validation_summaries = {
        name: cycle.get("causal_validation_summary", {})
        for name, cycle in cycles.items()
    }
    feature_records = {
        name: cycle.get("feature_store_records", [])[:8]
        for name, cycle in cycles.items()
    }
    decoder_records = {
        name: cycle.get("invariance_decoder", {})
        for name, cycle in cycles.items()
        if cycle.get("invariance_decoder")
    }
    scm_records = {
        name: cycle.get("scm_dag", {})
        for name, cycle in cycles.items()
        if cycle.get("scm_dag")
    }
    abstention_records = {
        name: cycle.get("abstention_gate", {})
        for name, cycle in cycles.items()
        if cycle.get("abstention_gate")
    }
    instrument_records = {
        name: cycle.get("instrument_registry", [])
        for name, cycle in cycles.items()
        if cycle.get("instrument_registry")
    }
    llm_audit_records = {
        name: cycle.get("causal_llm_audit", [])
        for name, cycle in cycles.items()
        if cycle.get("causal_llm_audit")
    }
    return {
        "report_date": report.get("report_date"),
        "generated_at": report.get("generated_at"),
        "repo": report.get("repo", {}),
        "data_timestamps": {
            "US": report.get("us_validation", {}),
            "HK": report.get("hk_validation", {}),
            "CN_FUTURES": report.get("futures_validations", {}),
        },
        "input_events": game_analysis.get("events", [])[:12],
        "event_windows": game_analysis.get("event_windows", [])[:12],
        "event_intensity": game_analysis.get("event_intensity", {}),
        "event_driven_causal_chains": game_analysis.get("event_causal_chains", [])[:10],
        "macro_event_state": market_data.get("macro_event_state", {}),
        "sensitive_asset_confirmations": [
            {
                "relation_id": item.get("relation_id"),
                "winner": item.get("current_judgement", {}).get("winner"),
                "confidence": item.get("current_judgement", {}).get("confidence"),
                "price_confirmation": item.get("price_confirmation"),
                "bilateral_probability": item.get("bilateral_probability"),
                "identification_status": item.get("identification_status"),
                "actionability": item.get("actionability"),
            }
            for item in top_relations
        ],
        "causal_validation": validation_summaries,
        "invariance_decoder": decoder_records,
        "scm_dag": scm_records,
        "causal_abstention_gate": abstention_records,
        "instrument_registry": instrument_records,
        "causal_llm_audit": llm_audit_records,
        "model_versions": model_records,
        "feature_lineage": feature_records,
        "position_constraints": {
            name: cycle.get("constraints", {})
            for name, cycle in cycles.items()
        },
        "execution_assumptions": {
            "price_basis": "latest_valid_close",
            "return_model": "close_to_close_for_review",
            "tail_hedge_mapping": {
                "US": US_TAIL_HEDGE_SYMBOL,
                "HK": HK_TAIL_HEDGE_SYMBOL,
                "CN_FUTURES": CN_FUTURES_TAIL_HEDGE_SYMBOL,
            },
            "safe_reserve_mapping": {
                "US": US_SAFE_RESERVE_SYMBOL,
                "HK": HK_SAFE_RESERVE_SYMBOL,
                "CN_FUTURES": CN_FUTURES_SAFE_RESERVE_SYMBOL,
            },
        },
        "audit_contract": "Every actionable order must trace to data timestamp, causal validation, model version, constraints and execution assumptions.",
    }


def _append_market_summary(
    report: Dict[str, Any],
    label: str,
    prefix: str,
    section_lines: List[str],
) -> None:
    market_key = prefix.upper()
    market_status = report.get("market_status", {}).get(market_key, {})
    if market_status and not bool(market_status.get("tradable")):
        section_lines.append(
            f"- {label}：NO_TRADE_DATA_INVALID；{market_status.get('reason', '数据校验未通过，本市场不出单。')}"
        )
        return

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
    market_status = _build_market_status(us_validation, hk_validation, futures_market_data)
    failure_category = _failure_category_from_market_status(market_status)

    report: Dict[str, Any] = {
        "status": "failed_validation",
        "failure_category": failure_category,
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
        "market_status": market_status,
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

    if not _any_market_tradable(market_status):
        report["failure_category"] = FAILURE_ALL_MARKETS_INVALID
        report["evidence_snapshot"] = _build_evidence_snapshot(report, {})
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
    macro_event_inputs = _build_macro_event_inputs(yf, futures_market_data)
    market_context, market_data = _build_market_context(
        cross_asset_engine,
        qqq,
        gold,
        copper_proxy,
        macro_event_inputs=macro_event_inputs,
    )
    tail_hedge_gates = {
        "US": _tail_hedge_effectiveness_gate(us_execution_data.get(US_TAIL_HEDGE_SYMBOL), market_context),
        "HK": _tail_hedge_effectiveness_gate(hk_execution_data.get(HK_TAIL_HEDGE_SYMBOL), market_context),
        "CN_FUTURES": _tail_hedge_effectiveness_gate(futures_execution_data.get(CN_FUTURES_TAIL_HEDGE_SYMBOL), market_context),
    }
    report["tail_hedge_gates"] = tail_hedge_gates
    us_cycle = (
        self_iterating_engine.run_learning_cycle(
            us_data,
            benchmark_frame=qqq,
            market_context=market_context,
            global_peer_datasets={},
        )
        if _market_is_tradable(market_status, "US")
        else {"status": "NO_TRADE_DATA_INVALID", "symbols": {}, "portfolio_plan": None, "trade_actions": []}
    )
    hk_cycle = (
        self_iterating_engine.run_learning_cycle(
            hk_data,
            benchmark_frame=None,
            market_context=market_context,
            global_peer_datasets={},
        )
        if _market_is_tradable(market_status, "HK")
        else {"status": "NO_TRADE_DATA_INVALID", "symbols": {}, "portfolio_plan": None, "trade_actions": []}
    )
    futures_cycles: Dict[str, Dict[str, Any]] = {}
    for exchange, payload in futures_market_data.items():
        if not _market_is_tradable(market_status, exchange):
            futures_cycles[exchange] = {"status": "NO_TRADE_DATA_INVALID", "symbols": {}, "portfolio_plan": None, "trade_actions": []}
            continue
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
            "status": "ok" if _all_markets_tradable(market_status) else "partial_ok",
            "failure_category": FAILURE_NONE if _all_markets_tradable(market_status) else FAILURE_DATA_VALIDATION_PARTIAL,
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
                tail_hedge_gates.get("US"),
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
                tail_hedge_gates.get("HK"),
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
        payload = _build_cycle_payload(
            exchange,
            cycle,
            exchange_price_map,
            execution_price_map,
            target_date,
            tail_hedge_gates.get("CN_FUTURES"),
        )
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
    all_cycles = {
        "US": us_cycle,
        "HK": hk_cycle,
        **{exchange: cycle for exchange, cycle in futures_cycles.items()},
    }
    report["evidence_snapshot"] = _build_evidence_snapshot(report, all_cycles)
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
    if report.get("failure_category") and report.get("failure_category") != FAILURE_NONE:
        lines.append(f"失败分类：{report.get('failure_category')}")
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
    market_status = report.get("market_status", {})
    if market_status:
        lines.append("")
        lines.append("市场出单状态：")
        for market, status in market_status.items():
            lines.append(f"{market}：{status.get('status')}；{status.get('reason')}")

    lines.append("")
    lines.append("复盘：")
    lines.extend([f"- {line}" for line in report.get("recap_lines", [])])

    if report.get("status") == "failed_validation":
        lines.append("")
        lines.append("结果：")
        lines.append("- 本次任务没有任何市场通过数据校验，按硬规则直接失败退出，不生成新的交易指令。")
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

    snapshot = report.get("evidence_snapshot", {})
    if snapshot:
        causal_validation = snapshot.get("causal_validation", {})
        validation_text = "; ".join(
            f"{market}:edges={item.get('edge_count', 0)} tradable={item.get('tradable_edge_count', 0)}"
            for market, item in causal_validation.items()
        )
        lines.append("")
        lines.append("证据快照：")
        lines.append(f"- 因果验证：{validation_text or '本次无可交易因果边'}")
        event_intensity = snapshot.get("event_intensity", {})
        if event_intensity:
            latest_values = event_intensity.get("latest_values", {})
            top_event_values = sorted(
                (
                    (name, float(value or 0.0))
                    for name, value in latest_values.items()
                    if str(name).startswith("event_zscore_")
                ),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:3]
            intensity_text = ", ".join(f"{name}={value:.2f}" for name, value in top_event_values) or "无显著事件Z分数"
            lines.append(
                "- 事件强度因子："
                f"status={event_intensity.get('status')} "
                f"events={event_intensity.get('event_count', 0)} "
                f"{intensity_text}"
            )
        decoder_snapshot = snapshot.get("invariance_decoder", {})
        if decoder_snapshot:
            decoder_text = "; ".join(
                f"{market}:active={item.get('active_count', 0)}/{item.get('decoder_count', 0)} "
                f"entropy={float(item.get('avg_state_entropy', 0.0) or 0.0):.3f} "
                f"risk_off={float(item.get('max_risk_off_probability', 0.0) or 0.0):.3f}"
                for market, item in decoder_snapshot.items()
            )
            lines.append(f"- 不变性/HMM解码：{decoder_text}")
        scm_snapshot = snapshot.get("scm_dag", {})
        if scm_snapshot:
            scm_text = "; ".join(
                f"{market}:graphs={item.get('graph_count', 0)} "
                f"edges={item.get('candidate_edge_count', 0)} "
                f"cf_tail={float(item.get('max_counterfactual_tail_risk', 0.0) or 0.0):.3f}"
                for market, item in scm_snapshot.items()
            )
            lines.append(f"- SCM/DAG因果图：{scm_text}")
        macro_event_snapshot = snapshot.get("macro_event_state", {})
        if macro_event_snapshot:
            overlays = macro_event_snapshot.get("factor_weight_overlays", {})
            alerts = macro_event_snapshot.get("alerts", [])
            lines.append(
                "- 宏观事件状态："
                f"status={macro_event_snapshot.get('status')} "
                f"tail={float(macro_event_snapshot.get('tail_risk_score', 0.0) or 0.0):.3f} "
                f"hedge_mult={float(macro_event_snapshot.get('tail_hedge_multiplier', 1.0) or 1.0):.2f} "
                f"ai_small_cap={float(overlays.get('ai_small_cap_momentum_multiplier', 1.0) or 1.0):.2f} "
                f"rate_sensitive={float(overlays.get('rate_sensitive_multiplier', 1.0) or 1.0):.2f} "
                f"vol={float(overlays.get('volatility_multiplier', 1.0) or 1.0):.2f} "
                f"alerts={len(alerts)}"
            )
        abstention_snapshot = snapshot.get("causal_abstention_gate", {})
        if abstention_snapshot:
            abstention_text = "; ".join(
                f"{market}:{item.get('decision_counts', {})} "
                f"avg_risk={float(item.get('avg_risk_score', 0.0) or 0.0):.3f}"
                for market, item in abstention_snapshot.items()
            )
            lines.append(f"- 统一弃权门：{abstention_text}")
        instrument_snapshot = snapshot.get("instrument_registry", {})
        if instrument_snapshot:
            instrument_text = "; ".join(
                f"{market}:iv_edges={len(records)} "
                f"valid={sum(1 for record in records if record.get('validity_status') == 'valid')}"
                for market, records in instrument_snapshot.items()
            )
            lines.append(f"- 工具变量诊断：{instrument_text}")
        llm_snapshot = snapshot.get("causal_llm_audit", {})
        if llm_snapshot:
            llm_text = "; ".join(
                f"{market}:audit_only={len(records)}"
                for market, records in llm_snapshot.items()
            )
            lines.append(f"- Causal LLM审计：{llm_text}（不得直接改变仓位）")
        lines.append(
            f"- 博弈确认：chains={len(snapshot.get('event_driven_causal_chains', []))} "
            f"relations={len(snapshot.get('sensitive_asset_confirmations', []))}，"
            f"审计规则={snapshot.get('audit_contract')}"
        )

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
        try:
            frame = _normalize_live_frame(ak.stock_hk_daily(symbol=code))
            snapshot = _price_on_or_before(frame, target_date)
            if snapshot:
                price_map[symbol] = snapshot
            else:
                price_map[symbol] = {"date": target_date, "close": None, "price_unavailable": True, "reason": "no_price_on_or_before_target"}
        except Exception as exc:  # noqa: BLE001
            price_map[symbol] = {"date": target_date, "close": None, "price_unavailable": True, "reason": str(exc)}
    for symbol in shfe_symbols:
        if symbol.endswith("_CASH"):
            continue
        try:
            frame = _normalize_live_frame(ak.futures_zh_daily_sina(symbol=symbol))
            snapshot = _price_on_or_before(frame, target_date)
            if snapshot:
                price_map[symbol] = snapshot
            else:
                price_map[symbol] = {"date": target_date, "close": None, "price_unavailable": True, "reason": "no_price_on_or_before_target"}
        except Exception as exc:  # noqa: BLE001
            price_map[symbol] = {"date": target_date, "close": None, "price_unavailable": True, "reason": str(exc)}
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
    quality_rows: List[Dict[str, float]] = []
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
        quality_rows.extend([hk_eval, shfe_eval])

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
        "quality_metrics": _aggregate_weekly_quality_metrics(quality_rows),
    }


def _aggregate_weekly_quality_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    price_unavailable_count = sum(int(row.get("price_unavailable_count", 0) or 0) for row in rows)
    active = [row for row in rows if int(row.get("risk_asset_count", 0) or 0) > 0]
    if not active:
        failures = ["no_directional_actions"]
        if price_unavailable_count:
            failures.append("price_unavailable")
        return {
            "win_rate": 0.0,
            "payoff_ratio": 0.0,
            "elasticity": 0.0,
            "slippage_bps": 0.0,
            "execution_quality": "no_directional_fill",
            "price_unavailable_count": price_unavailable_count,
            "failure_attribution": failures,
        }
    weights = [int(row.get("risk_asset_count", 0) or 0) for row in active]
    total_weight = sum(weights) or 1
    win_rate = sum(float(row.get("win_rate", 0.0)) * weight for row, weight in zip(active, weights)) / total_weight
    payoff = sum(float(row.get("payoff_ratio", 0.0)) * weight for row, weight in zip(active, weights)) / total_weight
    elasticity = sum(float(row.get("elasticity", 0.0)) * weight for row, weight in zip(active, weights)) / total_weight
    failures = {str(row.get("failure_attribution")) for row in active if row.get("failure_attribution")}
    if price_unavailable_count:
        failures.add("price_unavailable")
    return {
        "win_rate": round(float(win_rate), 6),
        "payoff_ratio": round(float(payoff), 6),
        "elasticity": round(float(elasticity), 6),
        "slippage_bps": 0.0,
        "execution_quality": "close_to_close_proxy_until_broker_fills_are_connected",
        "price_unavailable_count": price_unavailable_count,
        "failure_attribution": sorted(failures),
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
