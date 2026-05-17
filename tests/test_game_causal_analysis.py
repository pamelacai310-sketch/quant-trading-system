from __future__ import annotations

import unittest

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
