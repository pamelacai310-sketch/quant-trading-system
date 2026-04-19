from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

try:
    from ta.momentum import RSIIndicator
    from ta.trend import ADXIndicator, MACD
    from ta.volatility import BollingerBands

    HAS_TA_STACK = True
except Exception:
    HAS_TA_STACK = False


def compute_indicators(frame: pd.DataFrame, indicator_specs: list[dict]) -> pd.DataFrame:
    enriched = frame.copy()
    for spec in indicator_specs:
        name = spec["name"]
        source = spec.get("source", "close")
        series = enriched[source]
        kind = spec["type"]
        window = int(spec.get("window", 14))
        if kind == "sma":
            enriched[name] = series.rolling(window).mean()
        elif kind == "ema":
            enriched[name] = series.ewm(span=window, adjust=False).mean()
        elif kind == "zscore":
            mean = series.rolling(window).mean()
            std = series.rolling(window).std(ddof=0)
            enriched[name] = (series - mean) / std.replace(0, np.nan)
        elif kind == "volatility":
            returns = series.pct_change()
            enriched[name] = returns.rolling(window).std(ddof=0) * np.sqrt(252)
        elif kind == "momentum":
            enriched[name] = series / series.shift(window) - 1
        elif kind == "atr":
            prev_close = enriched["close"].shift(1)
            true_range = pd.concat(
                [
                    enriched["high"] - enriched["low"],
                    (enriched["high"] - prev_close).abs(),
                    (enriched["low"] - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            enriched[name] = true_range.rolling(window).mean()
        elif kind == "volume_sma":
            enriched[name] = enriched["volume"].rolling(window).mean()
        elif kind == "rsi":
            if HAS_TA_STACK:
                enriched[name] = RSIIndicator(series, window=window).rsi()
            else:
                delta = series.diff()
                gain = delta.clip(lower=0).rolling(window).mean()
                loss = (-delta.clip(upper=0)).rolling(window).mean()
                rs = gain / loss.replace(0, np.nan)
                enriched[name] = 100 - (100 / (1 + rs))
        elif kind == "macd_diff":
            if HAS_TA_STACK:
                enriched[name] = MACD(series).macd_diff()
            else:
                enriched[name] = series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()
        elif kind == "bb_width":
            if HAS_TA_STACK:
                bb = BollingerBands(series, window=window)
                enriched[name] = (bb.bollinger_hband() - bb.bollinger_lband()) / series.replace(0, np.nan)
            else:
                mean = series.rolling(window).mean()
                std = series.rolling(window).std(ddof=0)
                enriched[name] = (4 * std) / mean.replace(0, np.nan)
        elif kind == "adx":
            if HAS_TA_STACK:
                enriched[name] = ADXIndicator(enriched["high"], enriched["low"], enriched["close"], window=window).adx()
            else:
                returns = enriched["close"].pct_change().abs()
                enriched[name] = returns.rolling(window).mean() * 1000
        else:
            raise ValueError(f"Unsupported indicator type: {kind}")
    return enriched


def snapshot_indicators(frame: pd.DataFrame, names: list[str]) -> Dict[str, float]:
    if frame.empty:
        return {}
    row = frame.iloc[-1]
    snapshot = {}
    for name in names:
        value = row.get(name)
        if pd.isna(value):
            continue
        snapshot[name] = float(value)
    return snapshot
