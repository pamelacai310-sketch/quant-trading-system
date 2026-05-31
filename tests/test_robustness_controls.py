from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_trade_system.core.robustness import (
    deflated_sharpe_ratio,
    democratic_orthogonalize,
    effective_breadth,
    evaluate_cpcv_returns,
    shapley_deployment_policy,
)
from quant_trade_system.core.causal.self_iterating_causal_engine import SelfIteratingCausalEngine


class RobustnessControlTests(unittest.TestCase):
    def test_effective_breadth_penalizes_highly_correlated_signals(self) -> None:
        x = np.linspace(0, 1, 120)
        frame = pd.DataFrame(
            {
                "signal_a": x,
                "signal_b": x * 1.01,
                "signal_c": -x,
            }
        )

        audit = effective_breadth(frame)

        self.assertEqual(audit["nominal_breadth"], 3)
        self.assertLess(audit["effective_breadth"], 2.0)
        self.assertLess(audit["breadth_ratio"], 0.70)

    def test_democratic_orthogonalization_reduces_offdiag_correlation(self) -> None:
        idx = pd.RangeIndex(160)
        base = np.sin(np.linspace(0, 6, len(idx)))
        frame = pd.DataFrame(
            {
                "factor_1": base,
                "factor_2": base * 0.8 + np.linspace(0, 0.1, len(idx)),
                "factor_3": np.cos(np.linspace(0, 6, len(idx))),
            },
            index=idx,
        )

        orthogonal, diagnostics = democratic_orthogonalize(frame)

        self.assertEqual(diagnostics["status"], "ready")
        self.assertEqual(orthogonal.shape, frame.shape)
        self.assertLess(diagnostics["max_abs_offdiag_corr"], 0.05)
        self.assertGreaterEqual(
            diagnostics["effective_breadth_after"]["effective_breadth"],
            diagnostics["effective_breadth_before"]["effective_breadth"],
        )

    def test_cpcv_and_dsr_return_auditable_gates(self) -> None:
        returns = pd.Series([0.001] * 180)

        cpcv = evaluate_cpcv_returns(returns, n_groups=6, test_group_count=2, purge_window=2)
        dsr = deflated_sharpe_ratio(returns, effective_trials=10)

        self.assertEqual(cpcv["status"], "ready")
        self.assertGreater(cpcv["path_count"], 0)
        self.assertEqual(dsr["status"], "ready")
        self.assertIn("dsr_probability", dsr)
        self.assertIn("benchmark_sharpe_after_deflation", dsr)

    def test_hmm_hysteresis_retains_crisis_until_exit_threshold(self) -> None:
        engine = SelfIteratingCausalEngine()
        plan = engine.optimize_portfolio(
            [
                {
                    "symbol": "AAPL",
                    "asset_type": "stock",
                    "direction": "long",
                    "raw_score": 0.9,
                    "confidence": 0.9,
                    "objective_score": 0.9,
                    "objective_metrics": {"win_rate": 0.7, "payoff_ratio": 2.0, "elasticity": 1.5},
                    "selected_features": ["base_ret_5"],
                    "decoder_state_entropy": 0.20,
                    "decoder_risk_off_probability": 0.30,
                }
            ],
            market_context={
                "previous_hmm_barbell_state": "state2_liquidity_crisis",
                "crisis_probability": 0.20,
            },
        )

        self.assertEqual(plan.hmm_barbell_state, "state2_liquidity_crisis")
        self.assertTrue(plan.hmm_barbell_audit["hysteresis"]["retained_crisis"])
        self.assertLessEqual(plan.active_weight, 0.15)

    def test_shapley_policy_keeps_attribution_off_hot_path(self) -> None:
        policy = shapley_deployment_policy(model_family="LightGBM", feature_count=200, frequency="nightly")

        self.assertEqual(policy["method"], "TreeSHAP")
        self.assertFalse(policy["hot_path_allowed"])
        self.assertEqual(policy["recommended_use"], "offline_factor_decay_diagnostics")


if __name__ == "__main__":
    unittest.main()
