from __future__ import annotations

from datetime import date
import unittest

import pandas as pd

from quant_trade_system.nightly_quant_orders import (
    _build_instruction,
    _build_recap,
    _next_weekday,
    _observation_lines,
    _validate_hk_close,
)


def _frame(last_date: str, last_close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-05-09", last_date],
            "open": [100.0, last_close],
            "high": [101.0, last_close + 1],
            "low": [99.0, last_close - 1],
            "close": [100.0, last_close],
            "volume": [1_000, 2_000],
        }
    )


class NightlyQuantOrdersTests(unittest.TestCase):
    def test_next_weekday_skips_weekend(self) -> None:
        self.assertEqual(_next_weekday(date(2026, 5, 8)).isoformat(), "2026-05-11")
        self.assertEqual(_next_weekday(date(2026, 5, 11)).isoformat(), "2026-05-12")

    def test_validate_hk_close_requires_t_day(self) -> None:
        passed = _validate_hk_close({"00700.HK": _frame("2026-05-11", 464.4)}, "2026-05-11")
        failed = _validate_hk_close({"00700.HK": _frame("2026-05-08", 464.4)}, "2026-05-11")
        self.assertTrue(passed.passed)
        self.assertFalse(failed.passed)
        self.assertEqual(failed.actual_date, "2026-05-08")

    def test_build_instruction_uses_reference_close(self) -> None:
        long_line = _build_instruction(
            {
                "action": "LONG",
                "symbol": "CU0",
                "target_weight": 0.2,
                "confidence": 0.63,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.06,
            },
            100.0,
        )
        short_line = _build_instruction(
            {
                "action": "SHORT",
                "symbol": "AU0",
                "target_weight": 0.1,
                "confidence": 0.51,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.05,
            },
            200.0,
        )
        self.assertIn("止损 97.0000", long_line)
        self.assertIn("止盈 106.0000", long_line)
        self.assertIn("止损 204.0000", short_line)
        self.assertIn("止盈 190.0000", short_line)

    def test_build_recap_handles_directional_trade(self) -> None:
        previous_report = {
            "report_date": "2026-05-10",
            "primary_actions": [
                {"symbol": "CU0", "action": "LONG", "reference_close": 100.0},
                {"symbol": "SAFE_ASSET_BUCKET", "action": "SAFE_RESERVE", "reference_close": 0.0},
            ],
        }
        current_prices = {"CU0": {"close": 105.0}}
        lines = _build_recap(previous_report, current_prices)
        self.assertTrue(any("+5.00%" in line for line in lines))
        self.assertTrue(any("SAFE_RESERVE" in line for line in lines))

    def test_observation_lines_surface_best_rejection(self) -> None:
        reports = {
            "00700.HK": {
                "top_rejections": [
                    {"factor_name": "rsi_6", "rs_score": 100.0, "r_squared": 0.22, "rejection_reason": "R2=0.2200 < 0.70"}
                ]
            },
            "00941.HK": {
                "top_rejections": [
                    {"factor_name": "ema_12", "rs_score": 95.0, "r_squared": 0.18, "rejection_reason": "R2=0.1800 < 0.70"}
                ]
            },
        }
        latest = {
            "00700.HK": {"close": 464.4},
            "00941.HK": {"close": 86.2},
        }
        lines = _observation_lines(reports, latest)
        self.assertIn("00700.HK", lines[0])
        self.assertIn("rsi_6", lines[0])


if __name__ == "__main__":
    unittest.main()
