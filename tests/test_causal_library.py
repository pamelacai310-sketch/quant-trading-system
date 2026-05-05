from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_trade_system.core.causal import (
    CausalFactorLibrary,
    CrossAssetCausalEngine,
    MacroRegime,
)


class CausalLibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.library = CausalFactorLibrary()
        self.engine = CrossAssetCausalEngine(self.library)

    def test_factor_library_has_broad_cross_asset_coverage(self) -> None:
        summary = self.library.get_factor_coverage_summary()
        self.assertGreaterEqual(summary["total_factors"], 55)
        self.assertGreaterEqual(summary["quantized_factor_count"], summary["total_factors"])
        self.assertGreaterEqual(summary["cross_asset_factors"], 20)
        self.assertIn("country_premium", summary["high_reliability_factor_ids"])
        self.assertIn("growth_premium", self.library.get_factor_ids())
        self.assertIn("import_export_balance", self.library.get_factor_ids())
        self.assertIn("causal_quant_growth_premium", self.library.get_quantized_factor_ids())

    def test_search_includes_causal_mechanism_text(self) -> None:
        results = self.library.search_factors("贴现率")
        factor_ids = {factor.factor_id for factor in results}
        self.assertIn("interest_rate_premium", factor_ids)
        self.assertIn("refinancing_cost", factor_ids)

    def test_quantized_commodity_factors_flag_global_peer_linkage(self) -> None:
        gold_factor = self.library.get_quantized_factor("causal_quant_inflation_premium")
        copper_factor = self.library.get_quantized_factor("causal_quant_supply_demand_balance")
        self.assertIsNotNone(gold_factor)
        self.assertIsNotNone(copper_factor)
        self.assertTrue(gold_factor.metadata["global_peer_linkage_relevant"])
        self.assertEqual(gold_factor.metadata["global_peer_family"], "precious_metals")
        self.assertTrue(copper_factor.metadata["global_peer_linkage_relevant"])
        self.assertEqual(copper_factor.metadata["global_peer_family"], "base_metals")

    def test_unitize_exposures_produces_cross_sectional_scores(self) -> None:
        exposures = pd.DataFrame(
            {
                "asset_a": [1.2, 0.4, -0.3],
                "asset_b": [0.8, 0.9, 0.1],
                "asset_c": [1.5, -0.2, 0.7],
            },
            index=["factor_1", "factor_2", "factor_3"],
        ).T
        vol = pd.Series({"asset_a": 0.20, "asset_b": 0.25, "asset_c": 0.30})
        normalized = self.engine.unitize_exposures(exposures, volatility=vol)
        self.assertEqual(normalized.shape, exposures.shape)
        self.assertTrue(np.allclose(normalized.mean(axis=1).values, 0.0, atol=1e-8))

    def test_pca_orthogonalization_returns_nearly_uncorrelated_components(self) -> None:
        factor_returns = pd.DataFrame(
            {
                "growth": [0.02, 0.01, 0.00, 0.03, 0.015, -0.01],
                "inflation": [0.01, 0.011, 0.005, 0.018, 0.013, -0.002],
                "liquidity": [0.019, 0.012, 0.001, 0.024, 0.010, -0.008],
            }
        )
        result = self.engine.orthogonalize_pca(factor_returns, n_components=2)
        components = result["components"]
        dot_value = float(np.dot(components.iloc[:, 0], components.iloc[:, 1]))
        self.assertEqual(components.shape[1], 2)
        self.assertAlmostEqual(dot_value, 0.0, places=8)

    def test_regime_detection_and_risk_budget(self) -> None:
        regime = self.engine.detect_macro_regime(growth=0.04, inflation=0.035, liquidity=0.01)
        self.assertEqual(regime.regime, MacroRegime.HIGH_GROWTH_HIGH_INFLATION)

        covariance = pd.DataFrame(
            [
                [0.04, 0.01, 0.00],
                [0.01, 0.03, 0.002],
                [0.00, 0.002, 0.02],
            ],
            index=["growth", "inflation", "liquidity"],
            columns=["growth", "inflation", "liquidity"],
        )
        weights = pd.Series({"growth": 0.4, "inflation": 0.35, "liquidity": 0.25})
        report = self.engine.compute_factor_risk_contributions(weights, covariance)
        self.assertTrue((report["contribution_pct"] >= 0).all())
        self.assertAlmostEqual(float(report["contribution_pct"].sum()), 1.0, places=6)

    def test_stress_test_returns_asset_impacts(self) -> None:
        exposures = pd.DataFrame(
            {
                "growth": {"equity": 1.1, "copper": 0.8},
                "inflation": {"equity": -0.2, "copper": 0.9},
                "liquidity": {"equity": 0.7, "copper": -0.1},
            }
        )
        stress = self.engine.stress_test_macro_scenario(
            exposures,
            scenario_shocks={"growth": -0.03, "inflation": 0.02, "liquidity": -0.01},
        )
        self.assertIn("equity", stress["asset_impacts"])
        self.assertIn("copper", stress["asset_impacts"])
        self.assertGreater(stress["total_abs_impact"], 0.0)


if __name__ == "__main__":
    unittest.main()
