from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def _build_dataset(
    symbol: str,
    start: str,
    periods: int,
    seed: int,
    start_price: float,
    drift: float,
    volatility: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    noise = rng.normal(drift, volatility, len(dates))
    close = start_price * np.exp(np.cumsum(noise))
    close = np.maximum(close, 1.0)
    open_price = close * (1 + rng.normal(0, volatility / 3, len(dates)))
    high = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0.002, 0.004, len(dates))))
    low = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0.002, 0.004, len(dates))))
    base_volume = 900_000 if symbol == "XAUUSD" else 1_500_000
    trend_volume = np.linspace(0, 250_000, len(dates))
    cyclical = 120_000 * np.sin(np.linspace(0, 12 * math.pi, len(dates)))
    volume = np.maximum(base_volume + trend_volume + cyclical + rng.normal(0, 80_000, len(dates)), 50_000)

    return pd.DataFrame(
        {
            "timestamp": dates.strftime("%Y-%m-%d"),
            "open": np.round(open_price, 4),
            "high": np.round(high, 4),
            "low": np.round(low, 4),
            "close": np.round(close, 4),
            "volume": np.round(volume, 0).astype(int),
        }
    )


def ensure_demo_data(base_dir: str) -> None:
    data_dir = Path(base_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "gold_daily.csv": _build_dataset("XAUUSD", "2022-01-03", 720, 7, 1800.0, 0.00035, 0.010),
        "nasdaq_daily.csv": _build_dataset("QQQ", "2022-01-03", 720, 11, 320.0, 0.00045, 0.012),
        "copper_daily.csv": _build_dataset("HG", "2022-01-03", 720, 23, 4.35, 0.0004, 0.014),
    }
    for filename, frame in datasets.items():
        path = data_dir / filename
        if not path.exists():
            frame.to_csv(path, index=False)
