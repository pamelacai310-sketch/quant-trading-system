from __future__ import annotations

import unittest

import pandas as pd

from quant_trade_system.backtest import backtest_strategy


class BacktestAdaptiveRiskTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        closes = [100, 101, 103, 104, 105, 102, 101, 103, 106, 108, 104, 103, 106, 109]
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=len(closes), freq="D"),
                "open": [value * 0.999 for value in closes],
                "high": [value * 1.012 for value in closes],
                "low": [value * 0.992 for value in closes],
                "close": closes,
                "volume": [1000 + i * 10 for i in range(len(closes))],
            }
        )

    def _spec(self) -> dict:
        return {
            "symbol": "TEST",
            "direction": "long_only",
            "indicators": [],
            "entry_rules": [{"left": "close", "op": ">", "right": 0}],
            "exit_rules": [],
            "position_sizing": {"mode": "fixed_fraction", "risk_fraction": 0.20, "max_units": 1000},
            "risk_limits": {"stop_loss_pct": 0.08, "take_profit_pct": 0.03},
        }

    def test_backtest_generates_mae_mfe_feedback(self) -> None:
        result = backtest_strategy("s1", "adaptive", self._frame(), self._spec())

        feedback = result.stats["mae_mfe_feedback"]

        self.assertEqual(feedback["status"], "ready")
        self.assertTrue(feedback["recommended_risk_limits"]["enabled"])
        self.assertGreaterEqual(feedback["sample_size"], 2)
        self.assertIn("closed_trades", result.stats)
        self.assertGreaterEqual(len(result.stats["closed_trades"]), 2)
        self.assertIn("robustness_validation", result.stats)
        self.assertIn("deflated_sharpe_ratio", result.stats["robustness_validation"])
        self.assertIn("cpcv", result.stats["robustness_validation"])

    def test_backtest_applies_previous_mae_mfe_feedback(self) -> None:
        first = backtest_strategy("s1", "adaptive", self._frame(), self._spec())
        spec = self._spec()
        spec["risk_limits"]["mae_mfe_feedback"] = first.stats["mae_mfe_feedback"]

        second = backtest_strategy("s1", "adaptive", self._frame(), spec)

        self.assertTrue(second.stats["risk_limits_used"]["feedback_applied"])
        self.assertEqual(
            second.stats["risk_limits_used"]["feedback_sample_size"],
            first.stats["mae_mfe_feedback"]["sample_size"],
        )


if __name__ == "__main__":
    unittest.main()
