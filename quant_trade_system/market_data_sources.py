"""
Real-time market data adapters.

The adapters are optional-dependency based.  If a package, credential, or local
trading gateway is unavailable, the adapter reports that it is not configured
instead of breaking the whole system.

Supported adapter names:
* akshare  - public/free China, HK, US and futures data where available
* tushare  - China market data, requires TUSHARE_TOKEN for pro endpoints
* polygon  - US/global market data, requires POLYGON_API_KEY
* iex      - US market data, requires IEX_TOKEN or IEX_CLOUD_TOKEN
* ib       - Interactive Brokers TWS/IB Gateway via ib_insync
* rqdata   - RiceQuant RQData, requires RQDATA_USERNAME/PASSWORD
* gm       - 掘金量化 gm.api, requires GM_TOKEN
* vnpy_ctp - CTP availability probe via vn.py; live quote routing is gateway driven
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class Quote:
    symbol: str
    price: float | None = None
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None
    turnover: float | None = None
    open_interest: float | None = None
    change_pct: float | None = None
    timestamp: str | None = None
    provider: str = ""
    raw: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["raw"] = payload.get("raw") or {}
        return payload


class MarketDataAdapter:
    name = "base"

    def is_configured(self) -> bool:
        return True

    def status(self) -> Dict[str, Any]:
        return {"name": self.name, "configured": self.is_configured()}

    def get_quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        raise NotImplementedError


class AkShareAdapter(MarketDataAdapter):
    name = "akshare"

    def is_configured(self) -> bool:
        try:
            import akshare  # noqa: F401
            return True
        except Exception:
            return False

    def get_quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        import akshare as ak  # type: ignore

        requested = {str(s).upper() for s in symbols}
        result: Dict[str, Quote] = {}

        def normalise(code: Any) -> str:
            raw = str(code).strip().upper()
            digits = "".join(ch for ch in raw if ch.isdigit())
            if raw.endswith(".HK"):
                return f"{digits.zfill(5)}.HK" if digits else raw
            if raw.startswith("HK") and digits:
                return f"{digits.zfill(5)}.HK"
            if digits and len(digits) == 6:
                suffix = ".SH" if digits.startswith(("6", "9")) else ".SZ"
                return f"{digits}{suffix}"
            return raw

        frames = []
        for fn_name in ["stock_zh_a_spot_em", "stock_hk_spot_em", "stock_us_spot_em"]:
            fn = getattr(ak, fn_name, None)
            if fn is None:
                continue
            try:
                frames.append((fn_name, fn()))
            except Exception:
                continue

        # Futures spot interfaces differ by AkShare version; try common names.
        for fn_name, kwargs in [
            ("futures_zh_spot", {"market": "CF", "symbol": "所有"}),
            ("futures_zh_spot", {"market": "FF", "symbol": "所有"}),
            ("futures_display_main_sina", {}),
        ]:
            fn = getattr(ak, fn_name, None)
            if fn is None:
                continue
            try:
                frames.append((fn_name, fn(**kwargs)))
            except Exception:
                continue

        for provider_name, frame in frames:
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                continue
            columns = {str(c): c for c in frame.columns}
            code_col = next((columns[c] for c in columns if c.lower() in {"代码", "symbol", "code", "品种代码"} or "代码" in c), None)
            price_col = next((columns[c] for c in columns if c in {"最新价", "最新", "price", "last", "现价"}), None)
            volume_col = next((columns[c] for c in columns if c in {"成交量", "volume", "总手"}), None)
            turnover_col = next((columns[c] for c in columns if c in {"成交额", "turnover", "金额"}), None)
            change_col = next((columns[c] for c in columns if c in {"涨跌幅", "change_pct", "涨幅"}), None)
            oi_col = next((columns[c] for c in columns if c in {"持仓量", "open_interest"}), None)
            if code_col is None:
                continue
            for _, row in frame.iterrows():
                symbol = normalise(row.get(code_col))
                if symbol not in requested and symbol.replace(".", "") not in requested:
                    continue
                price = _to_float(row.get(price_col)) if price_col else None
                volume = _to_float(row.get(volume_col)) if volume_col else None
                turnover = _to_float(row.get(turnover_col)) if turnover_col else None
                change_pct = _to_float(row.get(change_col)) if change_col else None
                if change_pct is not None and abs(change_pct) > 1:
                    change_pct = change_pct / 100.0
                result[symbol] = Quote(
                    symbol=symbol,
                    price=price,
                    volume=volume,
                    turnover=turnover,
                    open_interest=_to_float(row.get(oi_col)) if oi_col else None,
                    change_pct=change_pct,
                    timestamp=pd.Timestamp.utcnow().isoformat(),
                    provider=self.name,
                    raw={"source": provider_name},
                )
        return result


class TushareAdapter(MarketDataAdapter):
    name = "tushare"

    def is_configured(self) -> bool:
        return bool(os.getenv("TUSHARE_TOKEN"))

    def get_quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        import tushare as ts  # type: ignore

        token = os.getenv("TUSHARE_TOKEN")
        if token:
            ts.set_token(token)
        requested = [str(s).replace(".SH", "").replace(".SZ", "") for s in symbols]
        result: Dict[str, Quote] = {}
        try:
            frame = ts.get_realtime_quotes(requested)
        except Exception:
            return result
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return result
        for _, row in frame.iterrows():
            code = str(row.get("code", "")).zfill(6)
            suffix = ".SH" if code.startswith(("6", "9")) else ".SZ"
            price = _to_float(row.get("price"))
            volume = _to_float(row.get("volume"))
            result[f"{code}{suffix}"] = Quote(
                symbol=f"{code}{suffix}",
                price=price,
                bid=_to_float(row.get("b1_p")),
                ask=_to_float(row.get("a1_p")),
                volume=volume,
                timestamp=f"{row.get('date', '')} {row.get('time', '')}".strip(),
                provider=self.name,
                raw={},
            )
        return result


class PolygonAdapter(MarketDataAdapter):
    name = "polygon"

    def is_configured(self) -> bool:
        return bool(os.getenv("POLYGON_API_KEY"))

    def get_quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        from polygon import RESTClient  # type: ignore

        client = RESTClient(os.environ["POLYGON_API_KEY"])
        result: Dict[str, Quote] = {}
        for symbol in symbols:
            ticker = str(symbol).upper().replace("-", ".")
            try:
                snap = client.get_snapshot_ticker("stocks", ticker)
                last_trade = getattr(snap, "last_trade", None)
                day = getattr(snap, "day", None)
                prev_day = getattr(snap, "prev_day", None)
                price = _to_float(getattr(last_trade, "price", None)) or _to_float(getattr(day, "close", None))
                prev_close = _to_float(getattr(prev_day, "close", None))
                change_pct = (price / prev_close - 1) if price and prev_close else None
                result[ticker] = Quote(
                    symbol=ticker,
                    price=price,
                    volume=_to_float(getattr(day, "volume", None)),
                    turnover=None,
                    change_pct=change_pct,
                    timestamp=pd.Timestamp.utcnow().isoformat(),
                    provider=self.name,
                    raw={},
                )
            except Exception:
                continue
        return result


class IEXAdapter(MarketDataAdapter):
    name = "iex"

    def is_configured(self) -> bool:
        return bool(os.getenv("IEX_TOKEN") or os.getenv("IEX_CLOUD_TOKEN"))

    def get_quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        import requests  # type: ignore

        token = os.getenv("IEX_TOKEN") or os.getenv("IEX_CLOUD_TOKEN")
        result: Dict[str, Quote] = {}
        batch = ",".join(str(s).upper().replace(".", "-") for s in symbols)
        if not batch:
            return result
        url = "https://cloud.iexapis.com/stable/stock/market/batch"
        params = {"symbols": batch, "types": "quote", "token": token}
        try:
            data = requests.get(url, params=params, timeout=8).json()
        except Exception:
            return result
        for symbol, payload in data.items():
            quote = payload.get("quote", {}) if isinstance(payload, dict) else {}
            result[symbol] = Quote(
                symbol=symbol,
                price=_to_float(quote.get("latestPrice")),
                bid=_to_float(quote.get("iexBidPrice")),
                ask=_to_float(quote.get("iexAskPrice")),
                volume=_to_float(quote.get("latestVolume")),
                turnover=None,
                change_pct=_to_float(quote.get("changePercent")),
                timestamp=str(quote.get("latestTime", "")),
                provider=self.name,
                raw={},
            )
        return result


class InteractiveBrokersAdapter(MarketDataAdapter):
    name = "ib"

    def is_configured(self) -> bool:
        return bool(os.getenv("IB_HOST", "127.0.0.1"))

    def get_quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        from ib_insync import IB, Stock  # type: ignore

        host = os.getenv("IB_HOST", "127.0.0.1")
        port = int(os.getenv("IB_PORT", "7497"))
        client_id = int(os.getenv("IB_CLIENT_ID", "8108"))
        timeout = float(os.getenv("IB_TIMEOUT", "4"))
        ib = IB()
        result: Dict[str, Quote] = {}
        try:
            ib.connect(host, port, clientId=client_id, timeout=timeout, readonly=True)
            contracts = [Stock(str(s).replace("-", "."), "SMART", "USD") for s in symbols if ".HK" not in str(s)]
            if not contracts:
                return result
            ib.qualifyContracts(*contracts)
            tickers = [ib.reqMktData(contract, "", False, False) for contract in contracts]
            ib.sleep(float(os.getenv("IB_SNAPSHOT_SLEEP", "1.5")))
            for ticker in tickers:
                symbol = ticker.contract.symbol
                price = _to_float(ticker.marketPrice()) or _to_float(ticker.last) or _to_float(ticker.close)
                result[symbol] = Quote(
                    symbol=symbol,
                    price=price,
                    bid=_to_float(ticker.bid),
                    ask=_to_float(ticker.ask),
                    volume=_to_float(ticker.volume),
                    timestamp=pd.Timestamp.utcnow().isoformat(),
                    provider=self.name,
                    raw={},
                )
                ib.cancelMktData(ticker.contract)
        finally:
            if ib.isConnected():
                ib.disconnect()
        return result


class RQDataAdapter(MarketDataAdapter):
    name = "rqdata"

    def is_configured(self) -> bool:
        return bool(os.getenv("RQDATA_USERNAME") and os.getenv("RQDATA_PASSWORD"))

    def get_quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        import rqdatac  # type: ignore

        rqdatac.init(os.environ["RQDATA_USERNAME"], os.environ["RQDATA_PASSWORD"])
        result: Dict[str, Quote] = {}
        try:
            snapshots = rqdatac.current_snapshot(list(symbols))
        except Exception:
            return result
        if isinstance(snapshots, dict):
            iterator = snapshots.items()
        else:
            iterator = [(str(s), snapshots) for s in symbols]
        for symbol, snap in iterator:
            result[str(symbol)] = Quote(
                symbol=str(symbol),
                price=_to_float(getattr(snap, "last", None) or getattr(snap, "last_price", None)),
                bid=_to_float(getattr(snap, "bid1", None)),
                ask=_to_float(getattr(snap, "ask1", None)),
                volume=_to_float(getattr(snap, "volume", None)),
                open_interest=_to_float(getattr(snap, "open_interest", None)),
                timestamp=str(getattr(snap, "datetime", "")),
                provider=self.name,
                raw={},
            )
        return result


class GmAdapter(MarketDataAdapter):
    name = "gm"

    def is_configured(self) -> bool:
        return bool(os.getenv("GM_TOKEN"))

    def get_quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        from gm.api import current, set_token  # type: ignore

        set_token(os.environ["GM_TOKEN"])
        result: Dict[str, Quote] = {}
        for symbol in symbols:
            try:
                data = current(str(symbol))
            except Exception:
                continue
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                sym = str(row.get("symbol", symbol)) if isinstance(row, dict) else str(symbol)
                result[sym] = Quote(
                    symbol=sym,
                    price=_to_float(row.get("price")) if isinstance(row, dict) else None,
                    volume=_to_float(row.get("cum_volume")) if isinstance(row, dict) else None,
                    turnover=_to_float(row.get("cum_amount")) if isinstance(row, dict) else None,
                    timestamp=str(row.get("created_at", "")) if isinstance(row, dict) else "",
                    provider=self.name,
                    raw={},
                )
        return result


class VnpyCtpAdapter(MarketDataAdapter):
    name = "vnpy_ctp"

    def is_configured(self) -> bool:
        try:
            import vnpy  # noqa: F401
            import vnpy_ctp  # noqa: F401
            return True
        except Exception:
            return False

    def get_quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        # CTP live market data requires broker front address, investor id,
        # password and the event-driven vn.py gateway loop.  The adapter reports
        # availability here; production subscriptions should be wired through
        # a long-running gateway service rather than per-request polling.
        return {}

    def status(self) -> Dict[str, Any]:
        payload = super().status()
        payload["note"] = "Use vn.py CTP gateway as a long-running quote/trading service; per-request polling is intentionally disabled."
        return payload


class RealTimeMarketDataHub:
    def __init__(self, providers: Optional[List[str]] = None, cache_ttl_seconds: float = 5.0) -> None:
        adapters: Dict[str, MarketDataAdapter] = {
            "akshare": AkShareAdapter(),
            "tushare": TushareAdapter(),
            "polygon": PolygonAdapter(),
            "iex": IEXAdapter(),
            "ib": InteractiveBrokersAdapter(),
            "rqdata": RQDataAdapter(),
            "gm": GmAdapter(),
            "vnpy_ctp": VnpyCtpAdapter(),
        }
        self.adapters = adapters
        self.providers = providers or ["akshare", "polygon", "iex", "tushare", "rqdata", "gm", "ib"]
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, tuple[float, Dict[str, Quote]]] = {}

    def status(self) -> Dict[str, Any]:
        return {name: adapter.status() for name, adapter in self.adapters.items()}

    def get_quotes(self, symbols: Iterable[str], provider: str | None = None) -> Dict[str, Quote]:
        symbols = [str(s) for s in symbols]
        providers = [provider] if provider else self.providers
        result: Dict[str, Quote] = {}
        missing = set(symbols)
        for provider_name in providers:
            adapter = self.adapters.get(provider_name)
            if adapter is None or not adapter.is_configured() or not missing:
                continue
            cache_key = f"{provider_name}:{','.join(sorted(missing))}"
            cached = self._cache.get(cache_key)
            now = time.time()
            if cached and now - cached[0] < self.cache_ttl_seconds:
                quotes = cached[1]
            else:
                try:
                    quotes = adapter.get_quotes(missing)
                except Exception:
                    quotes = {}
                self._cache[cache_key] = (now, quotes)
            result.update(quotes)
            quote_keys = {q.symbol for q in quotes.values()}
            missing = {s for s in missing if s not in quote_keys and s.upper() not in quote_keys}
        return result


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        text = str(value).replace(",", "").replace("%", "").strip()
        if text in {"", "--", "nan", "None"}:
            return None
        return float(text)
    except Exception:
        return None
