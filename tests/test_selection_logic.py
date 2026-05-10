from __future__ import annotations

import unittest

import pandas as pd

from quant_trade_system.selection_logic import DailySelectionEngine


class DailySelectionEngineTests(unittest.TestCase):
    def test_rank_nasdaq_net_inflow_filters_nasdaq_and_sorts(self) -> None:
        frame = pd.DataFrame(
            [
                {"f12": "AAPL", "f13": "105", "f14": "Apple", "f2": 195.0, "f3": 1.2, "f6": 1000.0, "f62": 120.0},
                {"f12": "MSFT", "f13": "105", "f14": "Microsoft", "f2": 420.0, "f3": 0.8, "f6": 900.0, "f62": 220.0},
                {"f12": "IBM", "f13": "106", "f14": "IBM", "f2": 180.0, "f3": 0.2, "f6": 300.0, "f62": 999.0},
            ]
        )
        result = DailySelectionEngine.rank_nasdaq_net_inflow(frame, topn=5)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["records"][0]["ticker"], "MSFT")
        self.assertEqual(result["records"][1]["ticker"], "AAPL")
        self.assertTrue(all(item["exchange"] == "NASDAQ" for item in result["records"]))

    def test_rank_china_futures_open_interest_uses_top20_proxy(self) -> None:
        frame = pd.DataFrame(
            [
                {"symbol": "CU2506", "var": "CU", "long_open_interest_top20": 100, "short_open_interest_top20": 110, "vol_top20": 500},
                {"symbol": "RB2510", "var": "RB", "long_open_interest_top20": 300, "short_open_interest_top20": 250, "vol_top20": 450},
                {"symbol": "AU2508", "var": "AU", "long_open_interest_top20": 200, "short_open_interest_top20": 180, "vol_top20": 600},
            ]
        )
        result = DailySelectionEngine.rank_china_futures_open_interest(frame, topn=2)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["records"][0]["contract"], "RB2510")
        self.assertEqual(result["records"][0]["open_interest_proxy"], 550.0)
        self.assertEqual(result["records"][1]["contract"], "AU2508")

    def test_screen_us_ytd_hot_stocks_excludes_otc_and_applies_thresholds(self) -> None:
        spot = pd.DataFrame(
            [
                {"代码": "105.AAA", "名称": "AAA Corp", "最新价": 160.0},
                {"代码": "106.BBB", "名称": "BBB Inc", "最新价": 130.0},
                {"代码": "153.OTC1", "名称": "OTC Name", "最新价": 500.0},
                {"代码": "105.CCC", "名称": "CCC Ltd", "最新价": 90.0},
            ]
        )
        pink = pd.DataFrame([{"代码": "153.OTC1", "名称": "OTC Name", "最新价": 500.0}])

        history_map = {
            "105.AAA": pd.DataFrame({"日期": ["2026-01-02", "2026-05-08"], "收盘": [100.0, 160.0]}),
            "106.BBB": pd.DataFrame({"日期": ["2026-01-02", "2026-05-08"], "收盘": [120.0, 130.0]}),
            "153.OTC1": pd.DataFrame({"日期": ["2026-01-02", "2026-05-08"], "收盘": [100.0, 500.0]}),
        }

        result = DailySelectionEngine.screen_us_ytd_hot_stocks(
            spot_frame=spot,
            history_loader=lambda symbol_code, _year_start: history_map.get(symbol_code, pd.DataFrame()),
            pink_frame=pink,
            anchor_date="20260508",
            min_price=100.0,
            min_ytd_return=0.50,
            topn=10,
        )
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["records"][0]["ticker"], "AAA")
        self.assertTrue(result["criteria"]["exclude_otc"])

    def test_extract_default_watchlist_symbols_prioritizes_inflow_then_ytd(self) -> None:
        selection_logic = {
            "nasdaq_top_net_inflow": {
                "records": [{"ticker": "MSFT"}, {"ticker": "NVDA"}],
            },
            "us_ytd_hot_non_otc": {
                "records": [{"ticker": "NVDA"}, {"ticker": "AAPL"}, {"ticker": "AMD"}],
            },
        }
        symbols = DailySelectionEngine.extract_default_watchlist_symbols(selection_logic, limit=4)
        self.assertEqual(symbols, ["MSFT", "NVDA", "AAPL", "AMD"])


if __name__ == "__main__":
    unittest.main()
