from __future__ import annotations

import unittest

import pandas as pd

from quant_trade_system.core.causal import GameCausalAnalysisEngine


class GameCausalAnalysisEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = GameCausalAnalysisEngine()

    def test_geopolitical_news_creates_energy_chain_and_crude_dominance(self) -> None:
        result = self.engine.analyze(
            news_items=[
                {
                    "timestamp": "2026-05-16T20:00:00",
                    "title": "Iran war risk threatens Hormuz Strait and creates oil supply disruption",
                    "summary": "OPEC+ stability is questioned as shipping insurance jumps.",
                    "source": "unit_test",
                    "relevance_score": 1.0,
                    "sentiment_score": -0.7,
                }
            ],
            market_context={"growth": 0.02, "inflation": 0.03, "liquidity": 0.0},
        )

        self.assertGreaterEqual(result["risk_scores"]["geopolitical_energy"]["score"], 0.55)
        self.assertIn("geo_energy_supply_shock", [chain["chain_id"] for chain in result["event_causal_chains"]])
        crude = self._dominance(result, "crude_oil")
        self.assertEqual(crude["dominant_logic"], "geopolitical_supply_shock")
        self.assertEqual(crude["expected_direction"], "bullish_crude")

    def test_gold_logic_switches_to_real_rate_pressure_when_policy_dominates(self) -> None:
        result = self.engine.analyze(
            news_items=[
                {
                    "title": "Fed Warsh rate hike path tests policy credibility as inflation and real rate pressure rise",
                    "source": "unit_test",
                    "relevance_score": 1.0,
                    "sentiment_score": -0.4,
                }
            ],
            market_context={"growth": 0.025, "inflation": 0.04, "liquidity": -0.02},
        )

        gold = self._dominance(result, "gold")
        self.assertEqual(gold["dominant_logic"], "real_rate_pressure_logic")
        self.assertEqual(gold["expected_direction"], "bearish_gold")

    def test_ai_capex_news_drives_copper_ai_demand_chain(self) -> None:
        result = self.engine.analyze(
            news_items=[
                {
                    "title": "AI capex and data center power grid energy bottleneck lift semiconductor demand",
                    "summary": "Copper demand is repriced by power infrastructure scarcity.",
                    "source": "unit_test",
                    "relevance_score": 1.0,
                    "sentiment_score": 0.5,
                }
            ],
            market_context={"ai_capex_growth": 0.35, "copper_inventory_days": 2.5, "growth": 0.035},
        )

        self.assertIn("ai_capex_power_materials_chain", [chain["chain_id"] for chain in result["event_causal_chains"]])
        copper = self._dominance(result, "copper")
        self.assertEqual(copper["dominant_logic"], "ai_power_grid_demand")
        self.assertEqual(copper["expected_direction"], "bullish_copper")

    def test_six_step_report_uses_sensitive_asset_confirmation(self) -> None:
        result = self.engine.analyze(
            news_items=[
                {
                    "title": "Iran war risk threatens Hormuz Strait oil supply disruption",
                    "source": "unit_test",
                    "relevance_score": 1.0,
                    "sentiment_score": -0.5,
                }
            ],
            market_context={
                "asset_signals": {
                    "Brent": {"direction": "up"},
                    "gold": {"direction": "up"},
                    "VIX": {"direction": "up"},
                    "credit_spreads": {"direction": "widen"},
                }
            },
        )

        report = self._relation(result, "geopolitical_risk_vs_risk_appetite")
        self.assertIn("core_logic", report)
        self.assertIn("transmission_mechanisms", report)
        self.assertIn("dominance_conditions", report)
        self.assertIn("market_pricing_forecast", report)
        self.assertEqual(report["current_judgement"]["winner"], "geopolitical_risk_premium")
        self.assertTrue(report["price_confirmation"]["A"]["data_available"])
        self.assertGreater(report["price_confirmation"]["A"]["score"], report["price_confirmation"]["B"]["score"])
        self.assertGreater(report["bilateral_probability"]["A"], report["bilateral_probability"]["B"])
        self.assertIn("event_zscore_geopolitical_energy", report["bilateral_probability"]["side_a_quant_inputs"])
        self.assertGreater(report["bilateral_probability"]["exposure_scaler"], 0.0)

    def test_price_non_confirmation_can_make_risk_appetite_win(self) -> None:
        result = self.engine.analyze(
            news_items=[
                {
                    "title": "Iran war risk threatens Hormuz Strait oil supply disruption",
                    "source": "unit_test",
                    "relevance_score": 1.0,
                    "sentiment_score": -0.5,
                }
            ],
            market_context={
                "growth": 0.035,
                "liquidity": 0.025,
                "asset_signals": {
                    "Brent": {"direction": "down"},
                    "gold": {"direction": "down"},
                    "VIX": {"direction": "down"},
                    "Nasdaq": {"direction": "up"},
                    "credit_spreads": {"direction": "narrow"},
                },
            },
        )

        report = self._relation(result, "geopolitical_risk_vs_risk_appetite")
        self.assertEqual(report["current_judgement"]["winner"], "risk_appetite_looks_through_tail_risk")
        self.assertEqual(
            report["layer_winners"]["risk_assets"]["winner"],
            "risk_appetite_looks_through_tail_risk",
        )

    def test_event_windows_are_created_from_pricing_asset_panels(self) -> None:
        panel = pd.DataFrame(
            {
                "date": pd.date_range("2026-05-01", periods=20, freq="D"),
                "close": [100 + i for i in range(20)],
                "volume": [1000 + i * 10 for i in range(20)],
            }
        )
        result = self.engine.analyze(
            news_items=[
                {
                    "timestamp": "2026-05-10",
                    "title": "OPEC supply disruption raises crude oil risk",
                    "source": "unit_test",
                }
            ],
            market_context={"pricing_asset_panels": {"Brent": panel}},
        )

        self.assertTrue(result["event_windows"])
        window = result["event_windows"][0]
        self.assertEqual(window["asset"], "Brent")
        self.assertEqual(window["observed_direction"], "up")
        self.assertTrue(window["usable_for_learning"])

    def test_price_confirmation_learning_reweights_reliable_assets(self) -> None:
        result = self.engine.analyze(
            news_items=[
                {
                    "title": "Iran war risk threatens Hormuz Strait oil supply disruption",
                    "source": "unit_test",
                }
            ],
            market_context={
                "price_confirmation_history": [
                    {
                        "relation_id": "geopolitical_risk_vs_risk_appetite",
                        "side_id": "geo_risk",
                        "asset": "Brent",
                        "expected_direction": "up",
                        "observed_direction": "up",
                        "outcome": "success",
                        "lead_lag_days": 1,
                    },
                    {
                        "relation_id": "geopolitical_risk_vs_risk_appetite",
                        "side_id": "geo_risk",
                        "asset": "Brent",
                        "expected_direction": "up",
                        "observed_direction": "up",
                        "outcome": "success",
                        "lead_lag_days": 2,
                    },
                ],
                "asset_signals": {"Brent": {"direction": "up"}},
            },
        )

        report = self._relation(result, "geopolitical_risk_vs_risk_appetite")
        brent_confirmation = next(
            item for item in report["price_confirmation"]["A"]["confirmations"] if item["asset"] == "Brent"
        )
        self.assertGreater(brent_confirmation["effective_weight"], brent_confirmation["weight"])
        self.assertGreaterEqual(brent_confirmation["learned_sample_count"], 2)
        self.assertIn(report["actionability"], {"trade_allowed", "observe_only"})
        self.assertIn("price_confirmation_quality", report["bilateral_probability"])
        self.assertGreater(report["bilateral_probability"]["position_multiplier"], 0.0)

    def test_game_probability_overlay_summarizes_position_and_tail_pressure(self) -> None:
        result = self.engine.analyze(
            news_items=[
                {
                    "title": "Iran war risk threatens Hormuz Strait oil supply disruption",
                    "source": "unit_test",
                    "relevance_score": 1.0,
                    "sentiment_score": -0.6,
                }
            ],
            market_context={
                "asset_signals": {
                    "Brent": {"direction": "up"},
                    "WTI": {"direction": "up"},
                    "gold": {"direction": "up"},
                    "VIX": {"direction": "up"},
                    "credit_spreads": {"direction": "widen"},
                }
            },
        )

        overlay = result["game_probability_overlay"]
        self.assertEqual(overlay["status"], "active")
        self.assertGreaterEqual(overlay["max_dominant_probability"], 0.5)
        self.assertGreater(overlay["max_exposure_scaler"], 0.0)
        self.assertGreaterEqual(overlay["position_multiplier"], 0.25)
        self.assertIn("geopolitical_risk_vs_risk_appetite", [
            item["relation_id"] for item in result["game_relation_reports"]
        ])

    def test_event_intensity_snapshot_is_returned_for_game_analysis(self) -> None:
        result = self.engine.analyze(
            news_items=[
                {
                    "timestamp": "2026-05-10",
                    "title": "AI capex and data center power grid bottleneck",
                    "relevance_score": 1.0,
                    "sentiment_score": 0.7,
                }
            ],
            market_context={"as_of": "2026-05-15"},
        )

        event_intensity = result["event_intensity"]
        self.assertEqual(event_intensity["status"], "active")
        self.assertIn("event_zscore_ai_capex", event_intensity["factor_columns"])
        self.assertIn("feature_frame_records", event_intensity)

    @staticmethod
    def _dominance(result: dict, asset: str) -> dict:
        for item in result["dominant_game_logics"]:
            if item["asset"] == asset:
                return item
        raise AssertionError(f"missing dominance for {asset}")

    @staticmethod
    def _relation(result: dict, relation_id: str) -> dict:
        for item in result["game_relation_reports"]:
            if item["relation_id"] == relation_id:
                return item
        raise AssertionError(f"missing relation report for {relation_id}")


if __name__ == "__main__":
    unittest.main()
