"""
Live quote helpers for API routes.

These functions keep the HTTP server thin while allowing the screener to enrich
cheap universe scores with real-time / near-real-time quotes from optional data
providers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .market_data_sources import RealTimeMarketDataHub
from .universe_provider import MarketUniverseProvider


def market_data_status() -> Dict[str, Any]:
    hub = RealTimeMarketDataHub()
    return hub.status()


def quote_symbols(payload: Dict[str, Any]) -> Dict[str, Any]:
    symbols = payload.get("symbols") or []
    if isinstance(symbols, str):
        symbols = [item.strip() for item in symbols.split(",") if item.strip()]
    provider = payload.get("provider")
    hub = RealTimeMarketDataHub(providers=[provider] if provider else None)
    quotes = hub.get_quotes(symbols, provider=provider)
    return {
        "provider": provider or "auto",
        "requested": symbols,
        "returned": len(quotes),
        "quotes": {symbol: quote.to_dict() for symbol, quote in quotes.items()},
    }


def screen_live_universe(base_dir: str | Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    data_dir = Path(base_dir) / "data"
    provider_name = payload.get("provider")
    universe = payload.get("universe", "default_global")
    top_n = int(payload.get("top_n", 50))
    prefilter_n = int(payload.get("prefilter_n", max(top_n * 5, 100)))
    filters = payload.get("filters", {}) or {}

    universe_provider = MarketUniverseProvider(data_dir, prefer_live=True)
    prefilter = universe_provider.screen(universe=universe, top_n=prefilter_n, filters=filters)
    symbols = [item["symbol"] for item in prefilter.get("candidates", [])]

    hub = RealTimeMarketDataHub(providers=[provider_name] if provider_name else None)
    quotes = hub.get_quotes(symbols, provider=provider_name)

    enriched: List[Dict[str, Any]] = []
    for candidate in prefilter.get("candidates", []):
        symbol = candidate["symbol"]
        quote = quotes.get(symbol) or quotes.get(symbol.upper())
        row = dict(candidate)
        row["live_quote"] = quote.to_dict() if quote else None
        if quote:
            live_score = float(row.get("score", 0.0))
            if quote.turnover:
                live_score += min(float(quote.turnover) / 1_000_000_000, 5.0)
            if quote.volume:
                live_score += min(float(quote.volume) / 10_000_000, 5.0)
            if quote.change_pct is not None:
                live_score += max(min(float(quote.change_pct) * 20, 3.0), -3.0)
            row["score"] = round(live_score, 6)
            row.setdefault("reasons", []).append(f"live_quote:{quote.provider}")
        enriched.append(row)

    min_score = float(filters.get("min_score", -1e9) or -1e9)
    enriched = [item for item in enriched if float(item.get("score", 0.0)) >= min_score]
    enriched.sort(key=lambda item: item.get("score", 0.0), reverse=True)

    return {
        "universe": universe,
        "provider": provider_name or "auto",
        "input_count": prefilter.get("input_count", 0),
        "prefilter_count": len(symbols),
        "quote_count": len(quotes),
        "candidate_count": len(enriched),
        "candidates": enriched[:top_n],
        "notes": [
            "Live screener first runs cheap universe prefiltering, then enriches the top symbols with configured real-time quote adapters.",
            "If no live provider is configured or quote retrieval fails, candidates fall back to metadata/local-dataset scores.",
        ],
    }
