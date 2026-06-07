from __future__ import annotations

import unittest

import pandas as pd

from quant_trade_system.futures_dc_research import (
    DCFuturesStrategySpec,
    FuturesCostModel,
    classify_walk_forward_result,
    cost_sensitivity_audit,
    fetch_main_contract_minute_frames,
    normalize_minute_frame,
    random_direction_control,
    simulate_dc_strategy,
)


def _minute_frame(closes: list[float]) -> pd.DataFrame:
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2026-06-01 09:00", periods=len(close), freq="5min"),
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1000,
            "hold": range(10_000, 10_000 + len(close)),
        }
    )


class FuturesDCResearchTests(unittest.TestCase):
    def test_default_cost_model_uses_round_trip_cost(self) -> None:
        self.assertAlmostEqual(FuturesCostModel().round_trip_cost_bps, 6.3)

    def test_fetch_main_contracts_preserves_product_order(self) -> None:
        class FakeAk:
            def match_main_contract(self, symbol: str):
                return {
                    "cffex": "IF2606,IC2606",
                    "shfe": "CU2607,AG2608",
                    "dce": "M2609,I2609",
                }.get(symbol, "")

            def futures_zh_minute_sina(self, symbol: str, period: str):
                return _minute_frame([100, 101, 102, 103])

        frames = fetch_main_contract_minute_frames(
            products=["IF", "CU", "AG"],
            period="5",
            max_contracts=2,
            ak=FakeAk(),
        )

        self.assertEqual(list(frames), ["IF2606", "CU2607"])

    def test_simulation_enters_after_dc_confirmation_next_bar(self) -> None:
        frame = _minute_frame([100, 99, 98, 102, 105, 104, 100, 96, 95, 98, 101, 103, 100, 97, 96, 99, 103])
        spec = DCFuturesStrategySpec(symbol="IF2606", family="dc_continuation", theta_bps=300, max_hold_bars=20)

        result = simulate_dc_strategy(
            frame,
            spec,
            cost_model=FuturesCostModel(commission_bps=1, slippage_bps=1, impact_bps=1),
        )

        self.assertGreaterEqual(result["event_count"], 3)
        self.assertGreater(result["trade_count"], 0)
        self.assertGreater(result["dc_path_return"], 0)
        self.assertTrue(all(trade["entry_lag_bars"] == 1 for trade in result["trades"]))
        self.assertTrue(all(trade["entry_index"] > trade["signal_index"] for trade in result["trades"]))

    def test_normalize_minute_frame_accepts_akshare_style_columns(self) -> None:
        raw = pd.DataFrame(
            {
                "datetime": ["2026-06-01 09:00:00", "2026-06-01 09:05:00"],
                "open": ["100", "101"],
                "high": ["102", "103"],
                "low": ["99", "100"],
                "close": ["101", "102"],
                "volume": ["10", "11"],
                "hold": ["1000", "1001"],
            }
        )

        normalized = normalize_minute_frame(raw)

        self.assertEqual(list(normalized.columns[:7]), ["timestamp", "open", "high", "low", "close", "volume", "hold"])
        self.assertEqual(normalized["close"].tolist(), [101.0, 102.0])
        self.assertEqual(normalized["_hold_diff"].iloc[-1], 1.0)

    def test_classification_requires_average_oos_not_local_luck(self) -> None:
        spec = DCFuturesStrategySpec(symbol="M2609", family="dc_reversal", theta_bps=24)
        cost = FuturesCostModel(commission_bps=0.8, slippage_bps=1.5, impact_bps=1.7)
        folds = [
            {"train_return": 0.02, "test_return": 0.02, "test_capture_ratio": 0.10, "test_events": 25, "test_trades": 12},
            {"train_return": 0.01, "test_return": 0.01, "test_capture_ratio": 0.08, "test_events": 24, "test_trades": 12},
            {"train_return": 0.01, "test_return": -0.06, "test_capture_ratio": -0.20, "test_events": 23, "test_trades": 12},
        ]

        result = classify_walk_forward_result(spec, cost, folds)

        self.assertEqual(result["status"], "FAIL")
        self.assertIn("NON_POSITIVE_EXPECTANCY", result["failure_reasons"])
        self.assertIn("UNSTABLE_FOLD_EDGE", result["failure_reasons"])

    def test_classification_rejects_average_capture_masking_fold_instability(self) -> None:
        spec = DCFuturesStrategySpec(symbol="AG2608", family="dc_continuation", theta_bps=24)
        cost = FuturesCostModel(commission_bps=0.8, slippage_bps=1.5, impact_bps=1.7)
        folds = [
            {"train_return": 0.02, "test_return": 0.010, "test_capture_ratio": 0.047, "test_events": 28, "test_trades": 10},
            {"train_return": 0.03, "test_return": 0.001, "test_capture_ratio": 0.006, "test_events": 22, "test_trades": 10},
            {"train_return": 0.02, "test_return": 0.027, "test_capture_ratio": 0.107, "test_events": 27, "test_trades": 12},
        ]

        result = classify_walk_forward_result(spec, cost, folds)

        self.assertEqual(result["status"], "WATCH")
        self.assertIn("FOLD_CAPTURE_INSTABILITY", result["failure_reasons"])
        self.assertIn("MEDIAN_CAPTURE_OUT_OF_TARGET", result["failure_reasons"])
        self.assertIn("UNSTABLE_FOLD_EDGE", result["failure_reasons"])

    def test_cost_sensitivity_detects_cost_fragile_edge(self) -> None:
        trades = [
            [{"side": "long", "entry_price": 100.0, "exit_price": 100.08}],
            [{"side": "short", "entry_price": 100.0, "exit_price": 99.92}],
        ]

        audit = cost_sensitivity_audit(trades, FuturesCostModel())

        self.assertGreater(audit["cost_1_0x"]["avg_return"], 0)
        self.assertLess(audit["cost_1_5x"]["avg_return"], 0)

    def test_random_direction_control_is_deterministic_and_reports_p_value(self) -> None:
        trades = [
            [{"side": "long", "entry_price": 100.0, "exit_price": 101.0} for _ in range(8)],
            [{"side": "short", "entry_price": 100.0, "exit_price": 99.0} for _ in range(8)],
        ]

        first = random_direction_control(trades, FuturesCostModel(), trials=64, seed=7)
        second = random_direction_control(trades, FuturesCostModel(), trials=64, seed=7)

        self.assertEqual(first, second)
        self.assertLess(first["p_value"], 0.10)
        self.assertTrue(first["beats_p95"])

    def test_classification_rejects_audits_that_do_not_survive_controls(self) -> None:
        spec = DCFuturesStrategySpec(symbol="IF2606", family="dc_reversal", theta_bps=32)
        cost = FuturesCostModel()
        folds = [
            {"train_return": 0.02, "test_return": 0.02, "test_capture_ratio": 0.10, "test_events": 25, "test_trades": 12},
            {"train_return": 0.02, "test_return": 0.01, "test_capture_ratio": 0.08, "test_events": 24, "test_trades": 12},
            {"train_return": 0.02, "test_return": -0.001, "test_capture_ratio": 0.06, "test_events": 23, "test_trades": 12},
        ]

        result = classify_walk_forward_result(
            spec,
            cost,
            folds,
            cost_sensitivity={"cost_1_5x": {"avg_return": -0.001}},
            random_control={"beats_p95": False, "p_value": 0.60},
        )

        self.assertEqual(result["status"], "WATCH")
        self.assertIn("COST_STRESS_FAIL", result["failure_reasons"])
        self.assertIn("RANDOM_CONTROL_NOT_BEATEN", result["failure_reasons"])

    def test_classification_passes_only_strict_repeatable_capture(self) -> None:
        spec = DCFuturesStrategySpec(symbol="AG2608", family="dc_continuation", theta_bps=24)
        cost = FuturesCostModel(commission_bps=0.8, slippage_bps=1.5, impact_bps=1.7)
        folds = [
            {"train_return": 0.02, "test_return": 0.02, "test_capture_ratio": 0.10, "test_events": 25, "test_trades": 12},
            {"train_return": 0.01, "test_return": 0.01, "test_capture_ratio": 0.08, "test_events": 24, "test_trades": 12},
            {"train_return": 0.01, "test_return": -0.001, "test_capture_ratio": 0.06, "test_events": 23, "test_trades": 12},
        ]

        result = classify_walk_forward_result(spec, cost, folds)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["failure_reasons"], [])


if __name__ == "__main__":
    unittest.main()
