from __future__ import annotations

from datetime import date, datetime
import unittest

import pandas as pd

from quant_trade_system.nightly_quant_orders import (
    CHINA_TZ,
    MarketValidation,
    _aggregate_weekly_quality_metrics,
    _build_evidence_snapshot,
    _build_instruction,
    _build_market_status,
    _build_recap,
    _evaluate_execution_actions,
    _materialize_execution_actions,
    _next_weekday,
    _observation_lines,
    _tail_hedge_effectiveness_gate,
    _validate_futures_close,
    _validate_hk_close,
    _validate_us_close,
    render_report_text,
)


def _frame(last_date: str, last_close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-05-09", last_date],
            "open": [100.0, last_close],
            "high": [101.0, last_close + 1],
            "low": [99.0, last_close - 1],
            "close": [100.0, last_close],
            "volume": [1_000, 2_000],
        }
    )


class NightlyQuantOrdersTests(unittest.TestCase):
    def test_next_weekday_skips_weekend(self) -> None:
        self.assertEqual(_next_weekday(date(2026, 5, 8)).isoformat(), "2026-05-11")
        self.assertEqual(_next_weekday(date(2026, 5, 11)).isoformat(), "2026-05-12")

    def test_validate_hk_close_requires_t_day(self) -> None:
        passed = _validate_hk_close({"00700.HK": _frame("2026-05-11", 464.4)}, "2026-05-11")
        failed = _validate_hk_close({"00700.HK": _frame("2026-05-08", 464.4)}, "2026-05-11")
        self.assertTrue(passed.passed)
        self.assertFalse(failed.passed)
        self.assertEqual(failed.actual_date, "2026-05-08")

    def test_validate_us_close_accepts_latest_completed_session(self) -> None:
        direct = _validate_us_close({"AAPL": _frame("2026-05-11", 215.0)}, date(2026, 5, 12))
        fallback = _validate_us_close({"AAPL": _frame("2026-05-08", 215.0)}, date(2026, 5, 12))
        stale = _validate_us_close({"AAPL": _frame("2026-05-04", 215.0)}, date(2026, 5, 12))
        self.assertTrue(direct.passed)
        self.assertTrue(fallback.passed)
        self.assertFalse(stale.passed)
        self.assertEqual(fallback.actual_date, "2026-05-08")

    def test_validate_us_close_accepts_t_after_us_session_is_available(self) -> None:
        after_close = datetime(2026, 5, 16, 6, 0, tzinfo=CHINA_TZ)
        before_close = datetime(2026, 5, 16, 4, 30, tzinfo=CHINA_TZ)
        passed = _validate_us_close({"AAPL": _frame("2026-05-15", 215.0)}, date(2026, 5, 15), as_of=after_close)
        failed = _validate_us_close({"AAPL": _frame("2026-05-15", 215.0)}, date(2026, 5, 15), as_of=before_close)
        self.assertTrue(passed.passed)
        self.assertIn("T 日", passed.reason)
        self.assertFalse(failed.passed)

    def test_validate_futures_close_falls_back_to_daily_when_settle_missing(self) -> None:
        settle = MarketValidation(
            market="DCE",
            requested_date="2026-05-15",
            actual_date=None,
            passed=False,
            reason="DCE 在回看窗口内都没有可验证的结算参数。",
            sample_count=0,
        )
        validation = _validate_futures_close("DCE", {"I0": _frame("2026-05-15", 805.5)}, settle)
        self.assertTrue(validation.passed)
        self.assertEqual(validation.actual_date, "2026-05-15")
        self.assertIn("显式降级", validation.reason)

    def test_build_instruction_uses_reference_close(self) -> None:
        long_line = _build_instruction(
            {
                "action": "LONG",
                "symbol": "CU0",
                "target_weight": 0.2,
                "confidence": 0.63,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.06,
            },
            100.0,
        )
        short_line = _build_instruction(
            {
                "action": "SHORT",
                "symbol": "AU0",
                "target_weight": 0.1,
                "confidence": 0.51,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.05,
            },
            200.0,
        )
        self.assertIn("止损 97.0000", long_line)
        self.assertIn("止盈 106.0000", long_line)
        self.assertIn("止损 204.0000", short_line)
        self.assertIn("止盈 190.0000", short_line)

    def test_build_recap_handles_directional_trade(self) -> None:
        previous_report = {
            "report_date": "2026-05-10",
            "primary_actions": [
                {"symbol": "CU0", "action": "LONG", "reference_close": 100.0},
                {"symbol": "SAFE_ASSET_BUCKET", "action": "SAFE_RESERVE", "reference_close": 0.0},
            ],
        }
        current_prices = {"CU0": {"close": 105.0}}
        lines = _build_recap(previous_report, current_prices)
        self.assertTrue(any("+5.00%" in line for line in lines))
        self.assertTrue(any("SAFE_RESERVE" in line for line in lines))

    def test_observation_lines_surface_best_rejection(self) -> None:
        reports = {
            "00700.HK": {
                "top_rejections": [
                    {"factor_name": "rsi_6", "rs_score": 100.0, "r_squared": 0.22, "rejection_reason": "R2=0.2200 < 0.70"}
                ]
            },
            "00941.HK": {
                "top_rejections": [
                    {"factor_name": "ema_12", "rs_score": 95.0, "r_squared": 0.18, "rejection_reason": "R2=0.1800 < 0.70"}
                ]
            },
        }
        latest = {
            "00700.HK": {"close": 464.4},
            "00941.HK": {"close": 86.2},
        }
        lines = _observation_lines(reports, latest)
        self.assertIn("00700.HK", lines[0])
        self.assertIn("rsi_6", lines[0])

    def test_materialize_execution_actions_maps_tail_and_cash(self) -> None:
        actions = [
            {"action": "TAIL_HEDGE", "symbol": "TAIL_RISK_PROTECTION", "target_weight": 0.1, "reason": "hedge"},
            {"action": "SAFE_RESERVE", "symbol": "SAFE_ASSET_BUCKET", "target_weight": 0.9, "reason": "cash"},
        ]
        price_map = {"02840.HK": {"date": "2026-05-11", "close": 100.0}}
        execution = _materialize_execution_actions(actions, "HK", price_map, "2026-05-11")
        self.assertEqual(execution[0]["symbol"], "02840.HK")
        self.assertEqual(execution[0]["action"], "LONG")
        self.assertEqual(execution[1]["symbol"], "HKD_CASH")
        self.assertEqual(execution[1]["action"], "HOLD")

    def test_materialize_execution_actions_maps_us_tail_and_cash(self) -> None:
        actions = [
            {"action": "TAIL_HEDGE", "symbol": "TAIL_RISK_PROTECTION", "target_weight": 0.2, "reason": "hedge"},
            {"action": "SAFE_RESERVE", "symbol": "SAFE_ASSET_BUCKET", "target_weight": 0.8, "reason": "cash"},
        ]
        price_map = {"GLD": {"date": "2026-05-11", "close": 300.0}}
        execution = _materialize_execution_actions(actions, "US", price_map, "2026-05-11")
        self.assertEqual(execution[0]["symbol"], "GLD")
        self.assertEqual(execution[0]["action"], "LONG")
        self.assertEqual(execution[1]["symbol"], "USD_CASH")
        self.assertEqual(execution[1]["action"], "HOLD")

    def test_tail_hedge_gate_switches_ineffective_gold_to_cash(self) -> None:
        dates = pd.date_range("2026-04-01", periods=25, freq="D")
        frame = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "open": [120.0 - idx for idx in range(25)],
                "high": [121.0 - idx for idx in range(25)],
                "low": [119.0 - idx for idx in range(25)],
                "close": [120.0 - idx for idx in range(25)],
                "volume": [1_000] * 25,
            }
        )
        gate = _tail_hedge_effectiveness_gate(frame, {"real_rates_direction": "up"})
        actions = [{"action": "TAIL_HEDGE", "symbol": "TAIL_RISK_PROTECTION", "target_weight": 0.2, "reason": "hedge"}]
        execution = _materialize_execution_actions(actions, "HK", {"02840.HK": {"date": "2026-05-11", "close": 100.0}}, "2026-05-11", gate)
        self.assertFalse(gate["active"])
        self.assertEqual(execution[0]["action"], "HOLD")
        self.assertEqual(execution[0]["symbol"], "HKD_CASH")
        self.assertEqual(execution[0]["bucket_action"], "TAIL_HEDGE")

    def test_materialize_futures_action_includes_one_lot_margin(self) -> None:
        actions = [
            {
                "action": "LONG",
                "symbol": "CU2607",
                "target_weight": 0.2,
                "confidence": 0.7,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.06,
                "margin_rate": 0.17,
            }
        ]
        price_map = {"CU2607": {"date": "2026-05-11", "close": 74_717.64705882352}}
        execution = _materialize_execution_actions(actions, "SHFE", price_map, "2026-05-11")
        self.assertEqual(execution[0]["contract_multiplier"], 5.0)
        self.assertAlmostEqual(execution[0]["one_lot_min_margin"], 63_510.0, places=2)
        self.assertEqual(execution[0]["margin_formula"], "latest_price * contract_multiplier * margin_rate")

    def test_evaluate_execution_actions_quantifies_nav(self) -> None:
        actions = [
            {
                "market": "HK",
                "bucket_action": "TAIL_HEDGE",
                "action": "LONG",
                "symbol": "02840.HK",
                "target_weight": 0.1,
                "reference_close": 100.0,
                "return_model": "close_to_close",
            },
            {
                "market": "HK",
                "bucket_action": "SAFE_RESERVE",
                "action": "HOLD",
                "symbol": "HKD_CASH",
                "target_weight": 0.9,
                "reference_close": 1.0,
                "return_model": "cash_flat",
            },
        ]
        current_prices = {"02840.HK": {"close": 110.0}, "HKD_CASH": {"close": 1.0}}
        summary = _evaluate_execution_actions(actions, current_prices)
        self.assertAlmostEqual(summary["portfolio_return"], 0.01)
        self.assertAlmostEqual(summary["gross_weight"], 0.1)
        self.assertEqual(summary["risk_asset_count"], 1)
        self.assertIn("elasticity", summary)

    def test_evaluate_execution_actions_records_price_unavailable(self) -> None:
        actions = [
            {
                "market": "SHFE",
                "bucket_action": "TAIL_HEDGE",
                "action": "LONG",
                "symbol": "AU0",
                "target_weight": 0.1,
                "reference_close": 1000.0,
                "return_model": "close_to_close",
            }
        ]
        summary = _evaluate_execution_actions(actions, {"AU0": {"close": None, "price_unavailable": True}})
        self.assertEqual(summary["portfolio_return"], 0.0)
        self.assertEqual(summary["price_unavailable_count"], 1)
        self.assertEqual(summary["failure_attribution"], "price_unavailable")
        self.assertEqual(summary["details"][0]["price_status"], "price_unavailable")

    def test_weekly_quality_metrics_preserve_price_unavailable(self) -> None:
        metrics = _aggregate_weekly_quality_metrics(
            [
                {
                    "risk_asset_count": 1,
                    "win_rate": 1.0,
                    "payoff_ratio": 2.0,
                    "elasticity": 1.5,
                    "price_unavailable_count": 0,
                    "failure_attribution": "not_evaluated",
                },
                {
                    "risk_asset_count": 0,
                    "price_unavailable_count": 1,
                    "failure_attribution": "price_unavailable",
                },
            ]
        )
        self.assertEqual(metrics["price_unavailable_count"], 1)
        self.assertIn("price_unavailable", metrics["failure_attribution"])

    def test_render_report_text_marks_invalid_market_without_global_failure(self) -> None:
        report = {
            "status": "partial_ok",
            "report_date": "2026-05-15",
            "generated_at": "2026-05-16T06:00:00+08:00",
            "repo": {"branch": "main", "head": "abc", "origin_main": "abc", "synced_with_origin_main": True},
            "us_validation": {"passed": True, "requested_date": "2026-05-15", "actual_date": "2026-05-15", "reason": "ok"},
            "hk_validation": {"passed": True, "requested_date": "2026-05-15", "actual_date": "2026-05-15", "reason": "ok"},
            "futures_validations": {
                "SHFE": {"passed": True, "requested_date": "2026-05-15", "actual_date": "2026-05-15", "reason": "ok"},
                "DCE": {"passed": False, "requested_date": "2026-05-15", "actual_date": None, "reason": "settle unavailable"},
            },
            "market_status": {
                "US": {"status": "OK", "tradable": True, "reason": "ok"},
                "HK": {"status": "OK", "tradable": True, "reason": "ok"},
                "SHFE": {"status": "OK", "tradable": True, "reason": "ok"},
                "DCE": {"status": "NO_TRADE_DATA_INVALID", "tradable": False, "reason": "settle unavailable"},
            },
            "recap_lines": [],
            "market_context": {"cross_asset_regime": {"regime": "test", "growth": 0.0, "inflation": 0.0, "liquidity": 0.0, "confidence": 1.0}},
            "calendar_note": "next",
            "evidence_snapshot": {},
        }
        text = render_report_text(report)
        self.assertIn("DCE：NO_TRADE_DATA_INVALID", text)
        self.assertNotIn("没有任何市场通过数据校验", text)

    def test_evidence_snapshot_links_models_features_and_confirmations(self) -> None:
        report = {
            "report_date": "2026-05-11",
            "generated_at": "2026-05-11T20:00:00+08:00",
            "repo": {"head": "abc123"},
            "us_validation": {"passed": True, "actual_date": "2026-05-08"},
            "hk_validation": {"passed": True, "actual_date": "2026-05-11"},
            "futures_validations": {"SHFE": {"passed": True, "actual_date": "2026-05-11"}},
            "market_data": {
                "game_causal_analysis": {
                    "events": [{"event_id": "evt1"}],
                    "event_windows": [{"event_id": "evt1", "asset": "gold"}],
                    "event_causal_chains": [{"chain_id": "geo_energy_supply_shock"}],
                    "game_relation_reports": [
                        {
                            "relation_id": "geopolitical_risk_vs_risk_appetite",
                            "current_judgement": {"winner": "geopolitical_risk_premium", "confidence": 0.8},
                            "price_confirmation": {"A": {"score": 1.0}},
                            "identification_status": {"identification_status": "identifiable"},
                            "actionability": "trade_allowed",
                        }
                    ],
                }
            },
        }
        cycles = {
            "HK": {
                "causal_validation_summary": {"edge_count": 3, "tradable_edge_count": 1},
                "invariance_decoder": {
                    "decoder_count": 1,
                    "active_count": 1,
                    "avg_state_entropy": 0.2,
                    "max_risk_off_probability": 0.1,
                },
                "scm_dag": {
                    "graph_count": 1,
                    "candidate_edge_count": 2,
                    "max_counterfactual_tail_risk": 0.15,
                },
                "model_registry_record": {"version": "v1"},
                "feature_store_records": [{"name": "f1"}],
                "constraints": {"max_single_weight": 0.2},
            }
        }
        snapshot = _build_evidence_snapshot(report, cycles)
        self.assertEqual(snapshot["causal_validation"]["HK"]["tradable_edge_count"], 1)
        self.assertEqual(snapshot["invariance_decoder"]["HK"]["active_count"], 1)
        self.assertEqual(snapshot["scm_dag"]["HK"]["candidate_edge_count"], 2)
        self.assertEqual(snapshot["model_versions"]["HK"]["version"], "v1")
        self.assertEqual(snapshot["sensitive_asset_confirmations"][0]["actionability"], "trade_allowed")


if __name__ == "__main__":
    unittest.main()
