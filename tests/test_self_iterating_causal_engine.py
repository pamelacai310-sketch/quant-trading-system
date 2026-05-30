from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_trade_system.core.causal import (
    FeatureSelectionPolicy,
    LearningObjectiveConfig,
    PortfolioConstraintConfig,
    SelectedFeature,
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
        self.assertIn("causal_validation_summary", result)
        self.assertIn("experiment_record", result)
        self.assertIn("model_registry_record", result)
        self.assertIn("feature_store_records", result)
        self.assertIn("scm_dag", result)
        self.assertIn("TEST", result["scm_dag"]["symbols"])

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
        self.assertIn("invariance_vol_norm_ret_1", matrix.columns)
        self.assertIn("hmm_prob_risk_on", matrix.columns)
        self.assertIn("hmm_sub_prob_liquidity_stress", matrix.columns)
        self.assertIn("kernel_analog_forward_mean", matrix.columns)
        self.assertIn("noisy_channel_long_posterior", matrix.columns)

    def test_candidate_matrix_includes_event_intensity_features(self) -> None:
        rows = 90
        dates = pd.date_range("2026-01-01", periods=rows, freq="D")
        close = pd.Series(np.linspace(100.0, 118.0, rows) + np.sin(np.linspace(0, 8, rows)))
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

        matrix = self.engine._build_candidate_factor_matrix(
            frame,
            symbol="SC2608",
            market_context={
                "news_items": [
                    {
                        "timestamp": "2026-02-10",
                        "title": "Iran war threatens Hormuz oil shipping supply disruption",
                        "relevance_score": 1.0,
                        "sentiment_score": -0.8,
                    }
                ]
            },
        )

        self.assertIn("event_intensity_geopolitical_energy", matrix.columns)
        self.assertIn("event_zscore_geopolitical_energy", matrix.columns)
        self.assertIn("event_asset_sc_exposure", matrix.columns)
        self.assertGreater(float(matrix["event_asset_sc_exposure"].sum()), 0.0)

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
        self.assertIn("invariance_decoder", result)
        self.assertEqual(result["invariance_decoder"]["decoder_count"], 1)
        self.assertIn("invariance_decoder", result["symbols"]["CU2608"])
        self.assertIn("scm_dag", result["symbols"]["CU2608"])

    def test_portfolio_penalty_uses_decoder_uncertainty_and_risk_off(self) -> None:
        low_risk_plan = self.engine.optimize_portfolio(
            [
                {
                    "symbol": "AAPL",
                    "asset_type": "stock",
                    "direction": "long",
                    "raw_score": 0.8,
                    "confidence": 0.7,
                    "objective_score": 0.75,
                    "selected_features": ["noisy_channel_long_posterior"],
                    "decoder_state_entropy": 0.10,
                    "decoder_risk_off_probability": 0.05,
                }
            ],
            market_context={"crisis_probability": 0.12},
        )
        high_risk_plan = self.engine.optimize_portfolio(
            [
                {
                    "symbol": "AAPL",
                    "asset_type": "stock",
                    "direction": "long",
                    "raw_score": 0.8,
                    "confidence": 0.7,
                    "objective_score": 0.75,
                    "selected_features": ["noisy_channel_long_posterior"],
                    "decoder_state_entropy": 0.95,
                    "decoder_risk_off_probability": 0.90,
                }
            ],
            market_context={"crisis_probability": 0.12},
        )
        self.assertGreater(low_risk_plan.projected_objective_score, high_risk_plan.projected_objective_score)

    def test_state_conditioning_changes_factor_weights(self) -> None:
        idx = pd.RangeIndex(90)
        driver = pd.Series(np.linspace(-2.0, 2.0, len(idx)), index=idx)
        factors = pd.DataFrame(
            {
                "base_ret_5": driver,
                "base_drawdown_20": driver,
            },
            index=idx,
        )
        target = driver.shift(-1).fillna(driver.iloc[-1])
        selected = [
            SelectedFeature("base_ret_5", "momentum", "x", 90.0, 0.9, 0.9, 1.0, 1, True, can_trade=True),
            SelectedFeature("base_drawdown_20", "drawdown", "x", 90.0, 0.9, 0.9, 1.0, 1, True, can_trade=True),
        ]
        decoder_audit = {
            "state_probabilities": {"risk_on": 0.90, "risk_off": 0.05, "transition_choppy": 0.05},
            "state_entropy": 0.10,
            "transition_stability": 0.80,
            "audit_metadata": {"sub_state_probabilities": {"trend": 0.85, "mean_reversion": 0.10, "liquidity_stress": 0.05}},
        }

        ensemble = self.engine._train_factor_ensemble(
            factors,
            target,
            selected,
            decoder_audit=decoder_audit,
            symbol="TEST",
        )

        self.assertGreater(ensemble["state_conditioning"]["base_ret_5"], ensemble["state_conditioning"]["base_drawdown_20"])
        self.assertGreater(ensemble["factor_weights"]["base_ret_5"], ensemble["factor_weights"]["base_drawdown_20"])

    def test_macro_event_overlay_boosts_csi1000_ai_momentum_and_haircuts_rate_sensitive(self) -> None:
        idx = pd.RangeIndex(90)
        driver = pd.Series(np.linspace(-2.0, 2.0, len(idx)), index=idx)
        factors = pd.DataFrame(
            {
                "base_ret_5": driver,
                "rate_sensitive_duration_factor": driver,
            },
            index=idx,
        )
        target = driver.shift(-1).fillna(driver.iloc[-1])
        selected = [
            SelectedFeature("base_ret_5", "momentum", "x", 90.0, 0.9, 0.9, 1.0, 1, True, can_trade=True),
            SelectedFeature("rate_sensitive_duration_factor", "rate", "x", 90.0, 0.9, 0.9, 1.0, 1, True, can_trade=True),
        ]
        macro_context = {
            "symbol_tags": {"IM0": ["csi1000", "small_cap", "ai_industrial_chain"]},
            "macro_event_state": {
                "factor_weight_overlays": {
                    "ai_small_cap_momentum_multiplier": 1.30,
                    "rate_sensitive_multiplier": 0.60,
                    "earnings_driven_multiplier": 1.0,
                    "volatility_multiplier": 1.0,
                    "fx_cny_resilience_multiplier": 1.0,
                }
            },
        }

        ensemble = self.engine._train_factor_ensemble(
            factors,
            target,
            selected,
            symbol="IM0",
            market_context=macro_context,
        )

        self.assertGreater(ensemble["macro_event_overlay"]["base_ret_5"], 1.0)
        self.assertLess(ensemble["macro_event_overlay"]["rate_sensitive_duration_factor"], 1.0)
        self.assertGreater(ensemble["factor_weights"]["base_ret_5"], ensemble["factor_weights"]["rate_sensitive_duration_factor"])

    def test_fractional_kelly_caps_position_size(self) -> None:
        high_plan = self.engine.optimize_portfolio(
            [
                {
                    "symbol": "AAPL",
                    "asset_type": "stock",
                    "direction": "long",
                    "raw_score": 0.9,
                    "confidence": 0.9,
                    "objective_score": 0.9,
                    "objective_metrics": {"win_rate": 0.72, "payoff_ratio": 2.8, "elasticity": 1.6},
                    "selected_features": ["f1"],
                    "decoder_state_entropy": 0.05,
                }
            ],
            market_context={"crisis_probability": 0.12},
        )
        low_plan = self.engine.optimize_portfolio(
            [
                {
                    "symbol": "AAPL",
                    "asset_type": "stock",
                    "direction": "long",
                    "raw_score": 0.9,
                    "confidence": 0.9,
                    "objective_score": 0.9,
                    "objective_metrics": {"win_rate": 0.51, "payoff_ratio": 1.1, "elasticity": 0.4},
                    "selected_features": ["f1"],
                    "decoder_state_entropy": 0.05,
                }
            ],
            market_context={"crisis_probability": 0.12},
        )

        self.assertGreater(high_plan.active_weight, low_plan.active_weight)
        self.assertLessEqual(high_plan.signal_allocations[0].target_weight, self.engine.constraints.max_single_weight)
        self.assertLessEqual(high_plan.signal_allocations[0].target_weight, high_plan.signal_allocations[0].kelly_fraction)

    def test_capacity_limit_caps_allocation_and_adds_penalty(self) -> None:
        plan = self.engine.optimize_portfolio(
            [
                {
                    "symbol": "ILLIQ",
                    "asset_type": "stock",
                    "direction": "long",
                    "raw_score": 0.9,
                    "confidence": 0.9,
                    "objective_score": 0.9,
                    "objective_metrics": {"win_rate": 0.75, "payoff_ratio": 3.0, "elasticity": 2.0},
                    "selected_features": ["f1"],
                    "decoder_state_entropy": 0.0,
                }
            ],
            market_context={
                "portfolio_notional": 1_000_000,
                "capacity": {"ILLIQ": {"adv_notional": 200_000, "max_participation_rate": 0.10}},
                "execution_costs": {"ILLIQ": {"commission_bps": 2.0, "slippage_bps": 8.0, "impact_bps": 30.0}},
            },
        )

        self.assertLessEqual(plan.signal_allocations[0].target_weight, 0.020001)
        self.assertGreaterEqual(plan.estimated_slippage_penalty, 0.0)
        self.assertGreater(plan.estimated_capacity_penalty, 0.0)

    def test_cross_asset_transfer_requires_validated_memory(self) -> None:
        feature = SelectedFeature("base_ret_5", "momentum", "x", 90.0, 0.9, 0.9, 1.0, 1, True, can_trade=True)
        self.assertEqual(self.engine._cross_asset_transfer_multiplier("base_ret_5", "AAPL"), 1.0)

        self.engine._update_cross_asset_factor_memory(
            feature,
            "CU2608",
            {"validation_score": 0.8, "identification_status": "identifiable"},
        )

        self.assertGreater(self.engine._cross_asset_transfer_multiplier("base_ret_5", "AAPL"), 1.0)

    def test_scm_counterfactual_stress_increases_tail_risk(self) -> None:
        low_context = {"crisis_probability": 0.05}
        high_context = {
            "crisis_probability": 0.05,
            "scm_counterfactual_stress": {"max_tail_risk_score": 0.45, "max_tail_hedge_multiplier": 1.45},
        }

        self.assertGreater(
            self.engine._extract_tail_risk_score(high_context),
            self.engine._extract_tail_risk_score(low_context),
        )

        no_signal_plan = self.engine.optimize_portfolio([], high_context)
        self.assertGreater(no_signal_plan.tail_hedge_weight, 0.10)

    def test_hmm_crisis_state_forces_dynamic_barbell_defense(self) -> None:
        plan = self.engine.optimize_portfolio(
            [
                {
                    "symbol": "AAPL",
                    "asset_type": "stock",
                    "direction": "long",
                    "raw_score": 0.9,
                    "confidence": 0.9,
                    "objective_score": 0.9,
                    "objective_metrics": {"win_rate": 0.7, "payoff_ratio": 2.0, "elasticity": 1.5},
                    "selected_features": ["noisy_channel_long_posterior"],
                    "decoder_state_entropy": 0.15,
                    "decoder_risk_off_probability": 0.88,
                }
            ],
            market_context={"crisis_probability": 0.12},
        )

        self.assertEqual(plan.hmm_barbell_state, "state2_liquidity_crisis")
        self.assertLessEqual(plan.active_weight, 0.15)
        self.assertGreaterEqual(plan.safe_weight, 0.70)
        self.assertGreaterEqual(plan.tail_hedge_weight, 0.15)

    def test_hmm_trend_state_releases_active_risk_budget(self) -> None:
        plan = self.engine.optimize_portfolio(
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
                    "decoder_state_entropy": 0.10,
                    "decoder_risk_off_probability": 0.05,
                }
            ],
            market_context={"hmm_barbell_state": "risk_on", "crisis_probability": 0.05},
        )

        self.assertEqual(plan.hmm_barbell_state, "state1_trend_or_normal")
        self.assertGreaterEqual(max(plan.hmm_barbell_audit["active_weight_grid"]), 0.85)
        self.assertGreater(plan.active_weight, 0.0)
        self.assertLessEqual(plan.tail_hedge_weight, 0.12)

    def test_validation_gate_marks_unstable_features_observation_only(self) -> None:
        idx = pd.RangeIndex(80)
        feature = pd.Series(np.r_[np.linspace(0, 1, 40), np.linspace(1, 0, 40)], index=idx)
        target = pd.Series(np.r_[np.linspace(0, 1, 40), np.linspace(0, 1, 40)], index=idx)
        validation = self.engine.causal_validation_loop.validate_feature(
            "unstable_factor",
            feature,
            target,
        )
        self.assertIn(validation.identification_status, {"correlation_only", "unavailable", "weak_identifiable"})
        if validation.identification_status in {"correlation_only", "unavailable"}:
            self.assertFalse(validation.can_trade)


if __name__ == "__main__":
    unittest.main()
