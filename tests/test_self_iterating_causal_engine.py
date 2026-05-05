from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_trade_system.core.causal import (
    FeatureSelectionPolicy,
    PortfolioConstraintConfig,
    SelfIteratingCausalEngine,
)


class SelfIteratingCausalEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SelfIteratingCausalEngine(
            selection_policy=FeatureSelectionPolicy(
                min_rs_score=70,
                min_r_squared=0.7,
                min_history=20,
                max_selected_features=5,
                target_horizon=1,
                signal_threshold=0.1,
            ),
            constraints=PortfolioConstraintConfig(
                max_positions=3,
                max_single_weight=0.2,
                max_futures_weight=0.5,
            ),
        )

    def test_auto_select_features_respects_rs_and_r2_thresholds(self) -> None:
        idx = pd.RangeIndex(80)
        driver = pd.Series(np.linspace(-2.0, 2.0, len(idx)), index=idx)
        noise = pd.Series(np.sin(np.linspace(0, 7, len(idx))), index=idx)
        target = driver * 0.8 + 0.02
        factors = pd.DataFrame(
            {
                "causal_quant_growth_premium": driver,
                "base_ret_5": noise,
            },
            index=idx,
        )
        selected, rejected = self.engine.auto_select_features(factors, target)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].factor_name, "causal_quant_growth_premium")
        self.assertGreaterEqual(selected[0].rs_score, 70)
        self.assertGreaterEqual(selected[0].r_squared, 0.7)
        self.assertTrue(any(item.factor_name == "base_ret_5" for item in rejected))

    def test_portfolio_optimizer_enforces_constraints(self) -> None:
        plan = self.engine.optimize_portfolio(
            [
                {
                    "symbol": "AAPL",
                    "asset_type": "stock",
                    "direction": "long",
                    "raw_score": 0.8,
                    "confidence": 0.7,
                    "objective_score": 0.75,
                    "selected_features": ["f1", "f2"],
                },
                {
                    "symbol": "CU2606",
                    "asset_type": "futures",
                    "direction": "long",
                    "raw_score": 0.9,
                    "confidence": 0.8,
                    "objective_score": 0.8,
                    "selected_features": ["f3"],
                },
                {
                    "symbol": "AU2606",
                    "asset_type": "futures",
                    "direction": "short",
                    "raw_score": -0.6,
                    "confidence": 0.6,
                    "objective_score": 0.7,
                    "selected_features": ["f4"],
                },
                {
                    "symbol": "MSFT",
                    "asset_type": "stock",
                    "direction": "long",
                    "raw_score": 0.5,
                    "confidence": 0.5,
                    "objective_score": 0.55,
                    "selected_features": ["f5"],
                },
            ],
            market_context={"crisis_probability": 0.35, "cross_asset_regime": {"regime": "liquidity_stress"}},
        )
        self.assertLessEqual(len(plan.signal_allocations), 3)
        self.assertLessEqual(max(item.target_weight for item in plan.signal_allocations), 0.2)
        futures_weight = sum(item.target_weight for item in plan.signal_allocations if item.asset_type == "futures")
        self.assertLessEqual(futures_weight, 0.5)
        self.assertGreater(plan.tail_hedge_weight, 0.0)

    def test_learning_cycle_produces_actions_on_synthetic_trend(self) -> None:
        rows = 160
        close = pd.Series(np.linspace(100.0, 140.0, rows) + 0.5 * np.sin(np.linspace(0, 12, rows)))
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=rows, freq="D"),
                "open": close * 0.998,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.linspace(1000, 2000, rows) + 50 * np.cos(np.linspace(0, 8, rows)),
            }
        )
        benchmark = frame.copy()
        benchmark["close"] = np.linspace(100.0, 120.0, rows)
        result = self.engine.run_learning_cycle(
            {"TEST": frame},
            benchmark_frame=benchmark,
            market_context={"crisis_probability": 0.12, "cross_asset_regime": {"regime": "soft_landing"}},
        )
        self.assertIn(result["status"], {"trained", "no_actionable_signals"})
        self.assertIn("TEST", result["symbols"])
        self.assertIn("portfolio_plan", result)
        self.assertIn("trade_actions", result)

    def test_futures_candidate_matrix_includes_global_peer_linkage_features(self) -> None:
        rows = 140
        dates = pd.date_range("2025-01-01", periods=rows, freq="D")
        local_close = pd.Series(np.linspace(100.0, 132.0, rows) + 0.6 * np.sin(np.linspace(0, 10, rows)))
        peer_close_a = pd.Series(np.linspace(98.0, 128.0, rows) + 0.5 * np.sin(np.linspace(0, 9, rows)))
        peer_close_b = pd.Series(np.linspace(99.0, 130.0, rows) + 0.4 * np.cos(np.linspace(0, 8, rows)))
        local_frame = pd.DataFrame(
            {
                "date": dates,
                "open": local_close * 0.998,
                "high": local_close * 1.01,
                "low": local_close * 0.99,
                "close": local_close,
                "volume": np.linspace(1200, 2200, rows),
            }
        )
        peer_a = pd.DataFrame(
            {
                "date": dates,
                "open": peer_close_a * 0.999,
                "high": peer_close_a * 1.01,
                "low": peer_close_a * 0.99,
                "close": peer_close_a,
                "volume": np.linspace(2000, 2600, rows),
            }
        )
        peer_b = pd.DataFrame(
            {
                "date": dates,
                "open": peer_close_b * 0.999,
                "high": peer_close_b * 1.01,
                "low": peer_close_b * 0.99,
                "close": peer_close_b,
                "volume": np.linspace(1800, 2400, rows),
            }
        )

        matrix = self.engine._build_candidate_factor_matrix(
            local_frame,
            symbol="AU2608",
            peer_frames={"COMEX_Gold": peer_a, "XAUUSD": peer_b},
        )

        self.assertIn("global_peer_return_1d_lag", matrix.columns)
        self.assertIn("global_peer_relative_momentum_20", matrix.columns)
        self.assertIn("global_peer_spillover_score", matrix.columns)
        self.assertGreater(float(matrix["global_peer_spillover_score"].abs().sum()), 0.0)

    def test_learning_cycle_tracks_global_peer_count_for_futures(self) -> None:
        rows = 120
        dates = pd.date_range("2025-01-01", periods=rows, freq="D")
        close = pd.Series(np.linspace(100.0, 124.0, rows) + 0.3 * np.sin(np.linspace(0, 8, rows)))
        peer_close = pd.Series(np.linspace(99.0, 121.0, rows) + 0.2 * np.cos(np.linspace(0, 7, rows)))
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": close * 0.998,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.linspace(1000, 1800, rows),
            }
        )
        peer = pd.DataFrame(
            {
                "date": dates,
                "open": peer_close * 0.999,
                "high": peer_close * 1.01,
                "low": peer_close * 0.99,
                "close": peer_close,
                "volume": np.linspace(1600, 2100, rows),
            }
        )
        result = self.engine.run_learning_cycle(
            {"CU2608": frame},
            global_peer_datasets={"CU2608": {"LME_Copper": peer}},
        )
        self.assertIn("CU2608", result["symbols"])
        self.assertEqual(result["symbols"]["CU2608"]["global_peer_count"], 1)


if __name__ == "__main__":
    unittest.main()
