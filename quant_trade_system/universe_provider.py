"""
Dynamic market universe provider and lightweight screener.

This module turns the project from demo-only symbol lists into a dynamic
universe-driven system.  It supports these logical universes:

* us_sp500: S&P 500 constituents
* us_nasdaq100: Nasdaq-100 constituents
* hk_hsi: Hang Seng Index constituents
* cn_star_leaders: STAR Market leaders, defaulting to Sci-Tech 50 proxies
* cn_futures_all: all major China futures products/contracts across SHFE/INE,
  DCE, CZCE, CFFEX and GFEX
* global_core/default_global: combined cross-market universe

The provider uses live public data sources when optional dependencies/network
are available and deterministic fallbacks when they are not.  The screener is
intentionally cheap-first: it ranks by local dataset features when available and
otherwise returns the universe metadata for downstream real-time quote bridges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .market_universe import (
    CFFEX_FUTURES_PRODUCTS,
    CN_FUTURES_PRODUCTS_BY_EXCHANGE,
    CZCE_FUTURES_PRODUCTS,
    DCE_FUTURES_PRODUCTS,
    GFEX_FUTURES_PRODUCTS,
    HANG_SENG_INDEX_SYMBOLS,
    get_hang_seng_symbols,
)


@dataclass(frozen=True)
class MarketSymbol:
    symbol: str
    name: str
    asset_type: str
    market: str
    exchange: str
    currency: str = ""
    tradable: bool = True
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = payload.get("metadata") or {}
        return payload


US_SP500_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "AVGO", "TSLA",
    "JPM", "LLY", "V", "XOM", "MA", "UNH", "COST", "WMT", "HD", "PG",
    "NFLX", "JNJ", "ABBV", "BAC", "CRM", "ORCL", "KO", "CVX", "WFC", "MRK",
    "CSCO", "ACN", "AMD", "PEP", "LIN", "MCD", "TMO", "IBM", "ABT", "GE",
]

US_NASDAQ100_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "AVGO", "GOOGL", "GOOG", "TSLA", "COST",
    "NFLX", "AMD", "PEP", "ADBE", "CSCO", "TMUS", "INTU", "QCOM", "TXN", "AMAT",
    "ISRG", "BKNG", "AMGN", "HON", "CMCSA", "VRTX", "PANW", "ADP", "MU", "ADI",
]

STAR_MARKET_LEADERS_FALLBACK = [
    "688981.SH", "688111.SH", "688012.SH", "688008.SH", "688036.SH", "688271.SH",
    "688126.SH", "688396.SH", "688599.SH", "688256.SH", "688303.SH", "688223.SH",
    "688363.SH", "688009.SH", "688187.SH", "688180.SH", "688169.SH", "688005.SH",
    "688120.SH", "688385.SH", "688032.SH", "688114.SH", "688561.SH", "688122.SH",
    "688220.SH", "688072.SH", "688777.SH", "688082.SH", "688235.SH", "688728.SH",
    "688188.SH", "688777.SH", "688521.SH", "688047.SH", "688208.SH", "688538.SH",
    "688516.SH", "688052.SH", "688065.SH", "688200.SH", "688772.SH", "688349.SH",
    "688234.SH", "688536.SH", "688686.SH", "688295.SH", "688029.SH", "688289.SH",
    "688516.SH", "688072.SH",
]

class MarketUniverseProvider:
    """Central provider for all market universes used by strategies and APIs."""

    def __init__(self, data_dir: str | Path | None = None, prefer_live: bool = True) -> None:
        self.data_dir = Path(data_dir) if data_dir else None
        self.prefer_live = prefer_live

    def list_universes(self) -> List[str]:
        return [
            "us_sp500", "us_nasdaq100", "us_core", "hk_hsi",
            "cn_star_leaders", "cn_futures_products", "cn_futures_all", "default_global", "global_core",
        ]

    def get_universe(self, name: str = "default_global", include_contracts: bool = True) -> List[MarketSymbol]:
        key = (name or "default_global").lower()
        if key == "us_sp500":
            return self._us_index("sp500")
        if key == "us_nasdaq100":
            return self._us_index("nasdaq100")
        if key == "us_core":
            return self._dedupe(self._us_index("sp500") + self._us_index("nasdaq100"))
        if key == "hk_hsi":
            return self._hk_hsi()
        if key == "cn_star_leaders":
            return self._cn_star_leaders()
        if key == "cn_futures_products":
            return self._cn_futures_products()
        if key == "cn_futures_all":
            return self._cn_futures_contracts() if include_contracts else self._cn_futures_products()
        if key in {"global_core", "default_global", "all"}:
            return self._dedupe(
                self._us_index("sp500")
                + self._us_index("nasdaq100")
                + self._hk_hsi()
                + self._cn_star_leaders()
                + (self._cn_futures_contracts() if include_contracts else self._cn_futures_products())
            )
        raise ValueError(f"Unknown universe: {name}. Available: {', '.join(self.list_universes())}")

    def get_symbols(self, name: str = "default_global", include_contracts: bool = True, limit: int | None = None) -> List[str]:
        symbols = [item.symbol for item in self.get_universe(name, include_contracts=include_contracts)]
        return symbols[:limit] if limit else symbols

    def summary(self, include_symbols: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key in self.list_universes():
            if key == "cn_futures_all":
                universe = self.get_universe(key, include_contracts=True)
            elif key == "default_global":
                continue
            else:
                universe = self.get_universe(key, include_contracts=False)
            result[key] = {
                "count": len(universe),
                "asset_types": sorted({item.asset_type for item in universe}),
                "markets": sorted({item.market for item in universe}),
                "symbols": [item.symbol for item in universe] if include_symbols else [],
            }
        global_symbols = self.get_universe("global_core", include_contracts=True)
        result["global_core"] = {
            "count": len(global_symbols),
            "asset_types": sorted({item.asset_type for item in global_symbols}),
            "markets": sorted({item.market for item in global_symbols}),
            "symbols": [item.symbol for item in global_symbols] if include_symbols else [],
        }
        return result

    def screen(
        self,
        universe: str = "default_global",
        top_n: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        filters = filters or {}
        items = self.get_universe(universe, include_contracts=True)
        scored = [self._score_symbol(item, filters) for item in items]

        min_turnover = float(filters.get("min_turnover", 0) or 0)
        min_volume = float(filters.get("min_volume", 0) or 0)
        min_score = float(filters.get("min_score", -1e9) or -1e9)

        candidates = [
            row for row in scored
            if row["score"] >= min_score
            and row.get("turnover", 0) >= min_turnover
            and row.get("volume", 0) >= min_volume
        ]
        candidates.sort(key=lambda row: row["score"], reverse=True)
        return {
            "universe": universe,
            "as_of": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "input_count": len(items),
            "candidate_count": len(candidates),
            "candidates": candidates[:top_n],
            "notes": [
                "Uses live/derived features when local datasets or optional data providers are available.",
                "Falls back to metadata ranking when quote data is unavailable.",
            ],
        }

    def _us_index(self, index: str) -> List[MarketSymbol]:
        if self.prefer_live:
            symbols = self._read_wikipedia_symbols(index)
            if symbols:
                return [MarketSymbol(s, s, "equity", "US", "NASDAQ/NYSE", "USD", metadata={"source": "wikipedia_live", "index": index}) for s in symbols]
        fallback = US_SP500_FALLBACK if index == "sp500" else US_NASDAQ100_FALLBACK
        return [MarketSymbol(s, s, "equity", "US", "NASDAQ/NYSE", "USD", metadata={"source": "fallback", "index": index}) for s in fallback]

    def _read_wikipedia_symbols(self, index: str) -> List[str]:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies" if index == "sp500" else "https://en.wikipedia.org/wiki/Nasdaq-100"
        try:
            tables = pd.read_html(url)
        except Exception:
            return []
        candidate_columns = ["Symbol", "Ticker"]
        for table in tables:
            for column in candidate_columns:
                if column in table.columns:
                    return [str(item).strip().replace(".", "-") for item in table[column].dropna().tolist()]
        return []

    def _hk_hsi(self) -> List[MarketSymbol]:
        symbols = get_hang_seng_symbols(prefer_live=self.prefer_live) or HANG_SENG_INDEX_SYMBOLS
        return [MarketSymbol(s, s, "equity", "HK", "HKEX", "HKD", metadata={"index": "HSI"}) for s in symbols]

    def _cn_star_leaders(self) -> List[MarketSymbol]:
        symbols = self._akshare_star_symbols() if self.prefer_live else []
        symbols = symbols or STAR_MARKET_LEADERS_FALLBACK
        return [MarketSymbol(s, s, "equity", "CN", "SSE STAR", "CNY", metadata={"universe": "star_leaders"}) for s in self._dedupe_strings(symbols)]

    def _akshare_star_symbols(self) -> List[str]:
        try:
            import akshare as ak  # type: ignore
        except Exception:
            return []
        calls = [
            ("stock_zh_index_spot_em", {}),
            ("index_stock_cons", {"symbol": "000688"}),
        ]
        for fn_name, kwargs in calls:
            fn = getattr(ak, fn_name, None)
            if fn is None:
                continue
            try:
                frame = fn(**kwargs)
            except Exception:
                continue
            for column in ["品种代码", "代码", "stock_code", "成分券代码"]:
                if column in frame.columns:
                    values = [str(v).strip() for v in frame[column].dropna().tolist()]
                    star = [f"{v.zfill(6)}.SH" for v in values if v.startswith("688")]
                    if star:
                        return star[:100]
        return []

    def _cn_futures_products(self) -> List[MarketSymbol]:
        products = []
        for exchange, product_list in CN_FUTURES_PRODUCTS_BY_EXCHANGE.items():
            for product in product_list:
                product_exchange = product.get("exchange", exchange)
                products.append(MarketSymbol(
                    product["symbol"],
                    product.get("name_zh", product["symbol"]),
                    "future_product",
                    "CN",
                    product_exchange,
                    "CNY",
                    metadata={"product_symbol": product["symbol"], "exchange": product_exchange},
                ))
        return self._dedupe(products)

    def _cn_futures_contracts(self, months_ahead: int = 12) -> List[MarketSymbol]:
        now = datetime.utcnow()
        contracts: List[MarketSymbol] = []
        for product in self._cn_futures_products():
            for offset in range(1, months_ahead + 1):
                month_number = now.month + offset
                year = now.year + (month_number - 1) // 12
                month = ((month_number - 1) % 12) + 1
                suffix = f"{year % 100:02d}{month:02d}"
                contracts.append(MarketSymbol(
                    f"{product.symbol}{suffix}",
                    f"{product.name}{suffix}",
                    "future_contract",
                    "CN",
                    product.exchange,
                    "CNY",
                    metadata={
                        "underlying": product.symbol,
                        "delivery_month": month,
                        "delivery_year": year,
                        "exchange": product.exchange,
                    },
                ))
        return contracts

    def _score_symbol(self, item: MarketSymbol, filters: Dict[str, Any]) -> Dict[str, Any]:
        local_features = self._local_dataset_features(item.symbol)
        score = 0.0
        reasons: List[str] = []
        if local_features:
            score += local_features.get("momentum_20", 0) * 40
            score += min(local_features.get("volume_z", 0), 5) * 5
            score += min(local_features.get("turnover", 0) / 1_000_000_000, 5)
            reasons.append("local_dataset_features")
        else:
            if item.asset_type == "future_contract":
                score += 1.0
                reasons.append("future_contract_available")
            if item.market in {"US", "HK"}:
                score += 0.5
                reasons.append("index_constituent")
            if item.exchange == "SSE STAR":
                score += 0.7
                reasons.append("star_market_leader")

        return {
            **item.to_dict(),
            "score": round(float(score), 6),
            "momentum_20": round(float(local_features.get("momentum_20", 0)), 6) if local_features else 0,
            "volume": float(local_features.get("volume", 0)) if local_features else 0,
            "turnover": float(local_features.get("turnover", 0)) if local_features else 0,
            "reasons": reasons,
        }

    def _local_dataset_features(self, symbol: str) -> Dict[str, float]:
        if self.data_dir is None or not self.data_dir.exists():
            return {}
        for path in self.data_dir.glob("*.csv"):
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if len(frame) < 25 or "close" not in frame.columns:
                continue
            # Dataset names are often not symbol names in this project, so use
            # them only when symbol is explicitly present or the file stem matches.
            if "symbol" in frame.columns and symbol not in set(frame["symbol"].astype(str)):
                continue
            if "symbol" not in frame.columns and path.stem.lower() not in symbol.lower().replace(".", "_").lower():
                continue
            subset = frame[frame["symbol"].astype(str) == symbol] if "symbol" in frame.columns else frame
            if len(subset) < 25:
                continue
            close = subset["close"].astype(float)
            volume = subset["volume"].astype(float) if "volume" in subset.columns else pd.Series([0.0] * len(subset))
            momentum_20 = close.iloc[-1] / close.iloc[-21] - 1
            volume_tail = volume.tail(20)
            volume_z = 0.0 if volume_tail.std() == 0 else (volume.iloc[-1] - volume_tail.mean()) / volume_tail.std()
            return {
                "momentum_20": float(momentum_20),
                "volume_z": float(volume_z),
                "volume": float(volume.iloc[-1]),
                "turnover": float(close.iloc[-1] * volume.iloc[-1]),
            }
        return {}

    @staticmethod
    def _dedupe(items: Iterable[MarketSymbol]) -> List[MarketSymbol]:
        seen = set()
        result = []
        for item in items:
            key = (item.symbol, item.asset_type, item.exchange)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    @staticmethod
    def _dedupe_strings(items: Iterable[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result
