from __future__ import annotations

import unittest

from quant_trade_system.core.causal import MacroEventStateEngine


class MacroEventStateEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = MacroEventStateEngine()

    def test_csi1000_ai_small_cap_leadership_boosts_momentum_overlay(self) -> None:
        state = self.engine.analyze(
            {
                "asset_signals": {
                    "CSI1000": {"return": 0.08},
                    "CSI500": {"return": 0.03},
                    "AI_small_caps": {"return": 0.06},
                }
            }
        )
        self.assertTrue(state["regime_flags"]["csi1000_ai_small_cap_leadership"])
        self.assertGreater(state["factor_weight_overlays"]["ai_small_cap_momentum_multiplier"], 1.0)

    def test_us_yield_five_percent_and_sofr_jump_raise_tail_and_cut_rate_sensitive(self) -> None:
        state = self.engine.analyze({"us10y_yield": 5.05, "sofr_5d_change": 35})
        self.assertTrue(state["regime_flags"]["us_yield_5pct_break"])
        self.assertTrue(state["regime_flags"]["sofr_fast_change"])
        self.assertGreaterEqual(state["tail_risk_score"], 0.45)
        self.assertLess(state["factor_weight_overlays"]["rate_sensitive_multiplier"], 1.0)

    def test_move_low_with_straddle_activity_is_bond_vol_warning(self) -> None:
        state = self.engine.analyze({"move_index": 82.0, "bond_straddle_activity": 0.82})
        self.assertTrue(state["regime_flags"]["move_straddle_warning"])
        self.assertGreater(state["factor_weight_overlays"]["volatility_multiplier"], 1.0)

    def test_hawkish_fed_with_dollar_weakness_changes_cny_readthrough(self) -> None:
        state = self.engine.analyze({"sofr_5d_change": 30, "dxy_return_5d": -0.015})
        self.assertTrue(state["regime_flags"]["hawkish_fed_usd_weakness"])
        self.assertEqual(state["fx_implications"]["regime"], "hawkish_fed_usd_weakness")
        self.assertIn("CNY resilience", state["fx_implications"]["cny_readthrough"])

    def test_hormuz_reopen_probability_prefers_volatility_before_signature(self) -> None:
        state = self.engine.analyze(
            {
                "event_probabilities": {
                    "hormuz_reopen_probability": 0.65,
                    "hormuz_reopen_signed": False,
                }
            }
        )
        scenario = state["event_scenarios"]["hormuz_reopen"]
        self.assertTrue(state["regime_flags"]["hormuz_pending_reopen"])
        self.assertEqual(scenario["pre_confirmation_bias"], "long_volatility_not_direction")


if __name__ == "__main__":
    unittest.main()
