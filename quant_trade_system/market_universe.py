"""
Market universe definitions.

The strategy modules used to keep short, hard-coded demo pools inside each
strategy class.  This module centralises production-grade universes so Hong
Kong equity scans can cover the Hang Seng Index and Shanghai futures scans can
cover the full SHFE/SHFE-group product set.

Notes
-----
* Hang Seng constituents are reviewed quarterly.  If AkShare is installed and
  exposes an index-constituent endpoint, ``get_hang_seng_symbols`` will prefer
  live data.  Otherwise it falls back to the static list below.
* SHFE product coverage follows the SHFE English product manual categories:
  metals, energy, chemical products and index futures.
"""

from __future__ import annotations

from typing import Any, Iterable, List


# Static fallback for the Hang Seng Index.  Kept as HK-suffixed symbols because
# yfinance/most HK data bridges use the ``00001.HK`` style.
# The dynamic loader below should be preferred when an index-constituent data
# provider is available.
HANG_SENG_INDEX_SYMBOLS: List[str] = [
    "00001.HK", "00002.HK", "00003.HK", "00005.HK", "00006.HK",
    "00011.HK", "00012.HK", "00016.HK", "00017.HK", "00027.HK",
    "00066.HK", "00101.HK", "00175.HK", "00241.HK", "00267.HK",
    "00288.HK", "00316.HK", "00386.HK", "00388.HK", "00669.HK",
    "00688.HK", "00700.HK", "00762.HK", "00823.HK", "00836.HK",
    "00857.HK", "00868.HK", "00939.HK", "00941.HK", "00960.HK",
    "00968.HK", "00981.HK", "00992.HK", "01024.HK", "01038.HK",
    "01044.HK", "01088.HK", "01093.HK", "01099.HK", "01109.HK",
    "01113.HK", "01177.HK", "01211.HK", "01299.HK", "01398.HK",
    "01810.HK", "01876.HK", "01928.HK", "01929.HK", "01972.HK",
    "01997.HK", "02015.HK", "02020.HK", "02269.HK", "02313.HK",
    "02318.HK", "02331.HK", "02333.HK", "02382.HK", "02388.HK",
    "02628.HK", "02688.HK", "02899.HK", "03690.HK", "03692.HK",
    "03750.HK", "03968.HK", "03988.HK", "03993.HK", "06030.HK",
    "06181.HK", "06618.HK", "06690.HK", "06862.HK", "09618.HK",
    "09633.HK", "09688.HK", "09863.HK", "09866.HK", "09868.HK",
    "09888.HK", "09901.HK", "09961.HK", "09988.HK", "09992.HK",
    "09999.HK", "02018.HK", "02057.HK", "02319.HK", "02423.HK",
    "02618.HK", "03328.HK", "03888.HK", "06098.HK", "06969.HK",
]


# SHFE core + SHFE-group/INE product symbols.  This intentionally excludes DCE,
# CZCE and CFFEX symbols that had leaked into the old mixed futures pools.
SHFE_FUTURES_PRODUCTS: List[dict[str, str]] = [
    {"symbol": "CU", "name": "Copper", "name_zh": "铜", "exchange": "SHFE"},
    {"symbol": "BC", "name": "Bonded Copper", "name_zh": "国际铜", "exchange": "INE"},
    {"symbol": "AL", "name": "Aluminum", "name_zh": "铝", "exchange": "SHFE"},
    {"symbol": "ZN", "name": "Zinc", "name_zh": "锌", "exchange": "SHFE"},
    {"symbol": "PB", "name": "Lead", "name_zh": "铅", "exchange": "SHFE"},
    {"symbol": "NI", "name": "Nickel", "name_zh": "镍", "exchange": "SHFE"},
    {"symbol": "SN", "name": "Tin", "name_zh": "锡", "exchange": "SHFE"},
    {"symbol": "AO", "name": "Aluminium Oxide", "name_zh": "氧化铝", "exchange": "SHFE"},
    {"symbol": "AD", "name": "Cast Aluminum Alloy", "name_zh": "铸造铝合金", "exchange": "SHFE"},
    {"symbol": "AU", "name": "Gold", "name_zh": "黄金", "exchange": "SHFE"},
    {"symbol": "AG", "name": "Silver", "name_zh": "白银", "exchange": "SHFE"},
    {"symbol": "RB", "name": "Steel Rebar", "name_zh": "螺纹钢", "exchange": "SHFE"},
    {"symbol": "WR", "name": "Steel Wire Rod", "name_zh": "线材", "exchange": "SHFE"},
    {"symbol": "HC", "name": "Hot Rolled Coils", "name_zh": "热轧卷板", "exchange": "SHFE"},
    {"symbol": "SS", "name": "Stainless Steel", "name_zh": "不锈钢", "exchange": "SHFE"},
    {"symbol": "SC", "name": "Crude Oil", "name_zh": "原油", "exchange": "INE"},
    {"symbol": "LU", "name": "LSFO", "name_zh": "低硫燃料油", "exchange": "INE"},
    {"symbol": "FU", "name": "Fuel Oil", "name_zh": "燃料油", "exchange": "SHFE"},
    {"symbol": "BU", "name": "Bitumen", "name_zh": "石油沥青", "exchange": "SHFE"},
    {"symbol": "BR", "name": "Synthetic Rubber", "name_zh": "合成橡胶", "exchange": "SHFE"},
    {"symbol": "RU", "name": "Natural Rubber", "name_zh": "天然橡胶", "exchange": "SHFE"},
    {"symbol": "NR", "name": "TSR 20", "name_zh": "20号胶", "exchange": "INE"},
    {"symbol": "SP", "name": "Woodpulp", "name_zh": "纸浆", "exchange": "SHFE"},
    {"symbol": "OP", "name": "Offset Paper", "name_zh": "胶版印刷纸", "exchange": "SHFE"},
    {"symbol": "EC", "name": "SCFIS(Europe)", "name_zh": "集运指数欧线", "exchange": "INE"},
]

SHFE_FUTURES_SYMBOLS: List[str] = [item["symbol"] for item in SHFE_FUTURES_PRODUCTS]
SHFE_FUTURES_NAME_MAP: dict[str, str] = {
    item["symbol"]: item["name_zh"] for item in SHFE_FUTURES_PRODUCTS
}


def _normalise_hk_code(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().upper().replace("HK.", "").replace(".HK", "")
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None
    return f"{digits.zfill(5)}.HK"


def _extract_hk_symbols(frame: Any) -> List[str]:
    try:
        columns: Iterable[str] = list(frame.columns)
    except Exception:
        return []

    candidate_columns = [
        col for col in columns
        if str(col).lower() in {"code", "symbol", "stock_code", "证券代码", "代码"}
        or "code" in str(col).lower()
        or "代码" in str(col)
    ]
    if not candidate_columns:
        return []

    symbols: List[str] = []
    for value in frame[candidate_columns[0]].tolist():
        symbol = _normalise_hk_code(value)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def get_hang_seng_symbols(prefer_live: bool = True) -> List[str]:
    """Return Hang Seng Index symbols with a live-provider-first fallback."""
    if prefer_live:
        try:
            import akshare as ak  # type: ignore

            # AkShare naming has changed across releases, so try common patterns
            # without making the core system depend on one exact version.
            candidate_calls = [
                ("stock_hk_index_spot_sina", {"symbol": "HSI"}),
                ("stock_hk_index_spot_sina", {"symbol": "恒生指数"}),
                ("stock_hk_index_cons", {"symbol": "HSI"}),
                ("stock_hk_index_cons", {"index": "HSI"}),
            ]
            for fn_name, kwargs in candidate_calls:
                fn = getattr(ak, fn_name, None)
                if fn is None:
                    continue
                try:
                    frame = fn(**kwargs)
                except TypeError:
                    try:
                        frame = fn(*kwargs.values())
                    except Exception:
                        continue
                except Exception:
                    continue
                symbols = _extract_hk_symbols(frame)
                if len(symbols) >= 20:
                    return symbols
        except Exception:
            pass

    return HANG_SENG_INDEX_SYMBOLS.copy()


def get_shfe_futures_symbols() -> List[str]:
    """Return all SHFE/SHFE-group futures product symbols."""
    return SHFE_FUTURES_SYMBOLS.copy()


def get_market_universe_summary() -> dict[str, Any]:
    hs_symbols = get_hang_seng_symbols()
    return {
        "hong_kong_equities": {
            "name": "Hang Seng Index constituents",
            "count": len(hs_symbols),
            "symbols": hs_symbols,
        },
        "shanghai_futures": {
            "name": "SHFE/INE futures products",
            "count": len(SHFE_FUTURES_SYMBOLS),
            "symbols": SHFE_FUTURES_SYMBOLS,
            "products": SHFE_FUTURES_PRODUCTS,
        },
    }
