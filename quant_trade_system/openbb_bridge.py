from __future__ import annotations

import json
import sys
from typing import Any, Dict


def _load_openbb():
    from openbb import obb  # type: ignore

    return obb


def _cmd_status() -> Dict[str, Any]:
    obb = _load_openbb()
    return {"python": sys.version.split()[0], "available": bool(obb)}


def _cmd_market(payload: Dict[str, Any]) -> Dict[str, Any]:
    obb = _load_openbb()
    symbols = payload.get("symbols") or ["AAPL", "MSFT", "GOOGL"]
    data = {}
    for symbol in symbols:
        try:
            quote = obb.equity.price.quote(symbol).to_df().tail(1).to_dict(orient="records")
            data[symbol] = quote[0] if quote else {}
        except Exception:
            data[symbol] = {}
    return data


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    payload = {}
    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        if raw:
            payload = json.loads(raw)
    commands = {
        "status": _cmd_status,
        "market": lambda: _cmd_market(payload),
    }
    try:
        data = commands[command]()
        print(json.dumps({"ok": True, "data": data}, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
