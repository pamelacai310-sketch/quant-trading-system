from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _ensure_home() -> None:
    home = os.environ.get("HOME")
    if home:
        Path(home).mkdir(parents=True, exist_ok=True)


def _serialize_snapshot(snapshot: Any) -> Dict[str, Any]:
    if snapshot is None:
        return {}
    if hasattr(snapshot, "model_dump"):
        data = snapshot.model_dump()
    elif hasattr(snapshot, "dict"):
        data = snapshot.dict()
    elif isinstance(snapshot, pd.DataFrame):
        if snapshot.empty:
            return {}
        data = snapshot.iloc[-1].to_dict()
    elif isinstance(snapshot, dict):
        data = dict(snapshot)
    else:
        data = {
            key: value
            for key, value in snapshot.__dict__.items()
            if not key.startswith("_")
        }
    normalized = {}
    for key, value in data.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def _serialize_frame(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    normalized = frame.copy()
    if isinstance(normalized.index, pd.DatetimeIndex):
        normalized = normalized.reset_index()
    return json.loads(normalized.to_json(orient="records", date_format="iso"))


def _load_finshare():
    _ensure_home()
    import finshare  # noqa: WPS433

    return finshare


def _cmd_status() -> Dict[str, Any]:
    finshare = _load_finshare()
    return {
        "version": getattr(finshare, "__version__", "unknown"),
        "python": sys.version.split()[0],
    }


def _cmd_history(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    finshare = _load_finshare()
    symbol = payload["symbol"]
    period = payload.get("period", "1mo")
    end = pd.Timestamp.utcnow().normalize()
    mapping = {"1mo": 31, "3mo": 92, "6mo": 184, "1y": 366, "max": 3650}
    start = end - pd.Timedelta(days=mapping.get(period, 184))
    candidates = {
        "AAPL": ["AAPL"],
        "MSFT": ["MSFT"],
        "GOOGL": ["GOOGL"],
        "NVDA": ["NVDA"],
        "QQQ": ["QQQ"],
        "600000": ["600000.SH", "600000"],
        "XAUUSD": ["AU0", "GC=F"],
        "COMEX_Gold": ["AU0", "GC=F"],
        "LME_Copper": ["CU0", "HG=F"],
    }.get(symbol, [symbol])
    for candidate in candidates:
        try:
            frame = finshare.get_historical_data(
                code=candidate,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                period="daily",
                adjust="qfq",
            )
        except Exception:
            continue
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return _serialize_frame(frame)
    return []


def _cmd_snapshot(payload: Dict[str, Any]) -> Dict[str, Any]:
    finshare = _load_finshare()
    symbol = payload["symbol"]
    candidates = {
        "600000": ["600000.SH", "600000"],
        "AAPL": ["AAPL"],
        "MSFT": ["MSFT"],
        "GOOGL": ["GOOGL"],
        "QQQ": ["QQQ"],
    }.get(symbol, [symbol])
    for candidate in candidates:
        try:
            snapshot = finshare.get_snapshot_data(candidate)
        except Exception:
            continue
        data = _serialize_snapshot(snapshot)
        if data:
            return data
    return {}


def _cmd_batch_snapshots(payload: Dict[str, Any]) -> Dict[str, Any]:
    finshare = _load_finshare()
    symbols = payload.get("symbols", [])
    code_map = {
        "600000": "600000.SH",
        "AAPL": "AAPL",
        "MSFT": "MSFT",
        "GOOGL": "GOOGL",
        "QQQ": "QQQ",
    }
    codes = [code_map.get(symbol, symbol) for symbol in symbols]
    try:
        result = finshare.get_batch_snapshots(codes)
    except Exception:
        result = {}
    if not isinstance(result, dict):
        return {}
    normalized = {}
    reverse_map = {value: key for key, value in code_map.items()}
    for code, snapshot in result.items():
        normalized[reverse_map.get(code, code)] = _serialize_snapshot(snapshot)
    return normalized


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    payload = {}
    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        if raw:
            payload = json.loads(raw)
    commands = {
        "status": lambda: _cmd_status(),
        "history": lambda: _cmd_history(payload),
        "snapshot": lambda: _cmd_snapshot(payload),
        "batch_snapshots": lambda: _cmd_batch_snapshots(payload),
    }
    try:
        data = commands[command]()
        print(json.dumps({"ok": True, "data": data}, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
