from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def _ensure_home() -> None:
    home = os.environ.get("HOME")
    if home:
        Path(home).mkdir(parents=True, exist_ok=True)


def _load_akshare():
    _ensure_home()
    import akshare as ak  # noqa: WPS433

    return ak


def _compact_date(value: Optional[str]) -> str:
    text = (value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 8:
        return digits
    return datetime.utcnow().strftime("%Y%m%d")


def _recent_dates(anchor: str, lookback_days: int = 7) -> List[str]:
    try:
        base = datetime.strptime(anchor, "%Y%m%d")
    except ValueError:
        base = datetime.utcnow()
    return [
        (base - timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range(lookback_days)
    ]


def _to_native(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _serialize_frame(frame: Optional[pd.DataFrame], limit: int = 5, tail: bool = True) -> List[Dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    working = frame.tail(limit) if tail else frame.head(limit)
    if isinstance(working.index, pd.DatetimeIndex):
        working = working.reset_index()
    rows: List[Dict[str, Any]] = []
    for _, row in working.iterrows():
        rows.append({str(column): _to_native(value) for column, value in row.items()})
    return rows


def _latest_record(frame: Optional[pd.DataFrame]) -> Dict[str, Any]:
    records = _serialize_frame(frame, limit=1, tail=True)
    return records[-1] if records else {}


def _to_float(value: Any) -> Optional[float]:
    native = _to_native(value)
    if native is None or native == "":
        return None
    if isinstance(native, bool):
        return float(native)
    if isinstance(native, (int, float)):
        return float(native)
    text = str(native).replace(",", "").replace("%", "").strip()
    if text in {"", "-", "--", "None", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _value_from_record(record: Dict[str, Any], aliases: Iterable[str]) -> Optional[float]:
    alias_list = [alias.lower() for alias in aliases]
    for key, value in record.items():
        key_lower = str(key).lower()
        if any(alias in key_lower for alias in alias_list):
            numeric = _to_float(value)
            if numeric is not None:
                return numeric
    for key in ["value", "latest", "latest_value", "今值", "现值", "数值", "收盘"]:
        if key in record:
            numeric = _to_float(record[key])
            if numeric is not None:
                return numeric
    return None


def _value_from_records(records: List[Dict[str, Any]], aliases: Iterable[str]) -> Optional[float]:
    for record in reversed(records):
        numeric = _value_from_record(record, aliases)
        if numeric is not None:
            return numeric
    return None


def _series_snapshot(frame: Optional[pd.DataFrame], aliases: Iterable[str], limit: int = 3, tail: bool = True) -> Dict[str, Any]:
    latest = _latest_record(frame)
    records = _serialize_frame(frame, limit=limit, tail=tail)
    payload: Dict[str, Any] = {
        "record": latest,
        "records": records,
    }
    numeric = _value_from_record(latest, aliases)
    if numeric is None:
        numeric = _value_from_records(records, aliases)
    if numeric is not None:
        payload["value"] = numeric
    return payload


def _lookup_recent_frame(loader, anchor_date: str, lookback_days: int = 7) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    last_error: Optional[str] = None
    for date_text in _recent_dates(anchor_date, lookback_days=lookback_days):
        try:
            frame = loader(date_text)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            continue
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame, date_text, None
    return None, None, last_error


def _normalize_inventory_symbol(symbol: str) -> str:
    mapping = {
        "铜": "沪铜",
        "黄金": "沪金",
        "金": "沪金",
        "银": "沪银",
        "铝": "沪铝",
        "锌": "沪锌",
        "铅": "沪铅",
    }
    return mapping.get(symbol, symbol)


def _cmd_status() -> Dict[str, Any]:
    ak = _load_akshare()
    return {
        "version": getattr(ak, "__version__", "unknown"),
        "python": sys.version.split()[0],
    }


def _cmd_market_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    ak = _load_akshare()
    anchor_date = _compact_date(payload.get("date"))
    equity_symbol = payload.get("equity_symbol") or "600000"
    hk_symbol = payload.get("hk_symbol") or "00700"
    inventory_symbols = payload.get("inventory_symbols") or ["沪铜", "沪金"]
    futures_rank_symbols = payload.get("futures_rank_symbols") or ["CU", "AU", "RB"]

    macro = {
        "lpr_1y": _series_snapshot(ak.macro_china_lpr(), aliases=["lpr1y", "1年", "1y", "利率"]),
        "shibor_1y": _series_snapshot(ak.macro_china_shibor_all(), aliases=["1y", "1年", "shibor"]),
        "m2_yoy": _series_snapshot(ak.macro_china_m2_yearly(), aliases=["同比", "增速", "数值", "今值", "value"]),
        "gdp_yoy": _series_snapshot(ak.macro_china_gdp_yearly(), aliases=["同比", "增速", "今值", "value", "数据"]),
        "cpi_yoy": _series_snapshot(ak.macro_china_cpi_yearly(), aliases=["同比", "增速", "今值", "value", "数据"]),
        "ppi_yoy": _series_snapshot(ak.macro_china_ppi_yearly(), aliases=["同比", "增速", "今值", "value", "数据"]),
    }

    inventory: Dict[str, Any] = {}
    warehouse_frame, warehouse_date, warehouse_error = _lookup_recent_frame(
        lambda current_date: ak.futures_shfe_warehouse_receipt(date=current_date),
        anchor_date,
    )
    inventory["shfe_warehouse_receipt"] = {
        "date": warehouse_date,
        "records": _serialize_frame(warehouse_frame, limit=5, tail=False),
    }
    if warehouse_error and warehouse_date is None:
        inventory["shfe_warehouse_receipt"]["error"] = warehouse_error

    for symbol in inventory_symbols:
        normalized_symbol = _normalize_inventory_symbol(symbol)
        key = f"{normalized_symbol}_inventory"
        try:
            inventory[key] = _series_snapshot(
                ak.futures_inventory_em(symbol=normalized_symbol),
                aliases=["库存", "存量", "数量", "value", "数值"],
            )
            inventory[key]["symbol"] = normalized_symbol
        except Exception as exc:  # noqa: BLE001
            inventory[key] = {"symbol": normalized_symbol, "error": str(exc)}

    valuation: Dict[str, Any] = {}
    try:
        valuation["a_share_pe_ttm"] = _series_snapshot(
            ak.stock_zh_valuation_baidu(symbol=equity_symbol, indicator="市盈率(TTM)", period="近一年"),
            aliases=["市盈率", "close", "value"],
        )
        valuation["a_share_pe_ttm"]["symbol"] = equity_symbol
    except Exception as exc:  # noqa: BLE001
        valuation["a_share_pe_ttm"] = {"symbol": equity_symbol, "error": str(exc)}
    try:
        valuation["a_share_pb"] = _series_snapshot(
            ak.stock_zh_valuation_baidu(symbol=equity_symbol, indicator="市净率", period="近一年"),
            aliases=["市净率", "close", "value"],
        )
        valuation["a_share_pb"]["symbol"] = equity_symbol
    except Exception as exc:  # noqa: BLE001
        valuation["a_share_pb"] = {"symbol": equity_symbol, "error": str(exc)}
    try:
        valuation["hk_pe_ttm"] = _series_snapshot(
            ak.stock_hk_valuation_baidu(symbol=hk_symbol, indicator="市盈率(TTM)", period="近一年"),
            aliases=["市盈率", "close", "value"],
        )
        valuation["hk_pe_ttm"]["symbol"] = hk_symbol
    except Exception as exc:  # noqa: BLE001
        valuation["hk_pe_ttm"] = {"symbol": hk_symbol, "error": str(exc)}

    holdings: Dict[str, Any] = {}
    try:
        northbound = ak.stock_hsgt_hold_stock_em(market="北向", indicator="今日排行")
        holdings["northbound_rank"] = {
            "records": _serialize_frame(northbound, limit=5, tail=False),
            "count": int(len(northbound)),
        }
    except Exception as exc:  # noqa: BLE001
        holdings["northbound_rank"] = {"error": str(exc)}
    try:
        shfe_rank = ak.get_shfe_rank_table(date=anchor_date, vars_list=futures_rank_symbols)
        if isinstance(shfe_rank, dict):
            holdings["shfe_rank_table"] = {
                symbol: _serialize_frame(frame, limit=3, tail=False)
                for symbol, frame in shfe_rank.items()
                if isinstance(frame, pd.DataFrame)
            }
        else:
            holdings["shfe_rank_table"] = {}
    except Exception as exc:  # noqa: BLE001
        holdings["shfe_rank_table"] = {"error": str(exc)}

    policy_frame, policy_date, policy_error = _lookup_recent_frame(
        lambda current_date: ak.macro_info_ws(date=current_date),
        anchor_date,
    )
    policy: Dict[str, Any] = {
        "calendar": {
            "date": policy_date,
            "records": _serialize_frame(policy_frame, limit=5, tail=False),
        },
        "event_count": {
            "date": policy_date,
            "value": len(policy_frame) if isinstance(policy_frame, pd.DataFrame) else 0,
        },
    }
    if policy_error and policy_date is None:
        policy["calendar"]["error"] = policy_error

    return {
        "macro": macro,
        "inventory": inventory,
        "valuation": valuation,
        "holdings": holdings,
        "policy": policy,
    }


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    payload: Dict[str, Any] = {}
    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        if raw:
            payload = json.loads(raw)
    commands = {
        "status": lambda: _cmd_status(),
        "market_context": lambda: _cmd_market_context(payload),
    }
    try:
        data = commands[command]()
        print(json.dumps({"ok": True, "data": data}, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
