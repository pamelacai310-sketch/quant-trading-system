from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_trade_system.core.causal import CausalGraphLayer, CausalValidationLoop


class CausalGraphLayerTests(unittest.TestCase):
    def test_discovery_generates_pc_fci_pcmci_candidate_edges(self) -> None:
        idx = pd.RangeIndex(120)
        driver = pd.Series(np.linspace(-2.0, 2.0, len(idx)), index=idx)
        target = driver.shift(1).fillna(0.0) * 0.8
        factors = pd.DataFrame(
            {
                "causal_driver": driver,
                "noise_factor": np.sin(np.linspace(0, 30, len(idx))),
            },
            index=idx,
        )
        layer = CausalGraphLayer(min_observations=40)

        edges = layer.discover_candidate_edges(factors, target)

        self.assertIn("causal_driver", edges)
        self.assertIn("pcmci", edges["causal_driver"].algorithms)
        self.assertGreater(edges["causal_driver"].confidence, 0.1)

    def test_backdoor_adjustment_builds_controls_and_validation_diagnostics(self) -> None:
        idx = pd.RangeIndex(120)
        benchmark = pd.Series(np.linspace(-1.0, 1.0, len(idx)), index=idx)
        idiosyncratic = pd.Series(np.sin(np.linspace(0, 8, len(idx))), index=idx)
        source = benchmark * 0.6 + idiosyncratic
        target = source * 0.5 + benchmark * 0.2
        factors = pd.DataFrame(
            {
                "alpha_source": source,
                "base_ret_20": benchmark,
                "hmm_state_entropy": np.linspace(0.2, 0.8, len(idx)),
            },
            index=idx,
        )
        layer = CausalGraphLayer(min_observations=40)
        adjustment = layer.backdoor_adjustment(
            "alpha_source",
            "forward_return",
            factors,
            benchmark_returns=benchmark,
        )
        controls = layer.adjustment_frame(adjustment, factors, benchmark_returns=benchmark)
        validation = CausalValidationLoop(min_observations=40).validate_feature(
            "alpha_source",
            factors["alpha_source"],
            target,
            benchmark_returns=benchmark,
            adjustment_frame=controls,
            backdoor_adjustment=adjustment.__dict__,
            discovery_support=0.8,
        )

        self.assertIn("benchmark_return", adjustment.adjustment_variables)
        self.assertGreaterEqual(adjustment.adjustment_quality, 0.55)
        self.assertIn("backdoor_adjustment", validation.diagnostics)
        self.assertTrue(validation.can_trade)

    def test_counterfactual_stress_outputs_tail_hedge_multiplier(self) -> None:
        idx = pd.RangeIndex(100)
        source = pd.Series(np.linspace(-1.0, 1.0, len(idx)), index=idx)
        target = source.shift(1).fillna(0.0) * -0.4
        factors = pd.DataFrame({"risk_driver": source}, index=idx)
        layer = CausalGraphLayer(min_observations=40)
        edges = layer.discover_candidate_edges(factors, target)

        stress = layer.counterfactual_stress_test(edges.values(), factors, target)

        self.assertGreaterEqual(stress.tail_risk_score, 0.0)
        self.assertGreaterEqual(stress.tail_hedge_multiplier, 1.0)
        self.assertTrue(stress.affected_paths)


if __name__ == "__main__":
    unittest.main()
