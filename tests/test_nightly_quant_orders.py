from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from quant_trade_system.nightly_health_check import check_nightly_health
from quant_trade_system.nightly_quant_orders import (
    CHINA_TZ,
    CFFEX_SETTLE_FALLBACK_CONTRACT_SPECS,
    FAILURE_ALL_MARKETS_INVALID,
    FAILURE_DATA_VALIDATION_PARTIAL,
    FAILURE_NONE,
    FAILURE_RUNTIME_EXCEPTION,
    FAILURE_SCHEDULER_NOT_RUN,
    MarketValidation,
    _aggregate_weekly_quality_metrics,
    _attach_data_quality_to_market_status,
    _build_data_quality_report,
    _build_evidence_snapshot,
    _build_instruction,
    _build_cycle_payload,
    _build_market_status,
    _build_weekly_path_capture_metrics,
    _build_weekly_robustness_validation,
    _build_weekly_optimization_recommendations,
    _consolidate_shared_futures_defensive_actions,
    _build_recap,
    _evaluate_execution_actions,
    _fetch_futures_data,
    generate_weekly_execution_review,
    _materialize_execution_actions,
    _next_weekday,
    _observation_lines,
    _failure_category_from_market_status,
    _fetch_us_data,
    _market_context_with_data_quality,
    _optional_yf_frame,
    _pct_change_over_window,
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


def _yfinance_batch_frame(symbols: list[str]) -> pd.DataFrame:
    index = pd.to_datetime(["2026-05-28", "2026-05-29"])
    index.name = "Date"
    columns = pd.MultiIndex.from_product([symbols, ["Open", "High", "Low", "Close", "Volume"]])
    data = {}
    for offset, symbol in enumerate(symbols):
        base = 100.0 + offset * 10
        data[(symbol, "Open")] = [base, base + 1.0]
        data[(symbol, "High")] = [base + 2.0, base + 3.0]
        data[(symbol, "Low")] = [base - 2.0, base - 1.0]
        data[(symbol, "Close")] = [base + 0.5, base + 1.5]
        data[(symbol, "Volume")] = [1000 + offset, 2000 + offset]
    return pd.DataFrame(data, index=index, columns=columns)


class FakeBatchYF:
    def __init__(self, frame: pd.DataFrame | None = None, fail: bool = False) -> None:
        self.frame = frame if frame is not None else _yfinance_batch_frame(["AAPL", "MSFT"])
        self.fail = fail
        self.calls: list[tuple[object, dict]] = []

    def download(self, symbols: object, **kwargs: object) -> pd.DataFrame:
        self.calls.append((symbols, dict(kwargs)))
        if self.fail:
            raise TimeoutError("simulated yfinance timeout")
        return self.frame


def _futures_main_frame(last_date: str = "2026-05-29") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日期": ["2026-05-28", last_date],
            "开盘价": [103000.0, 104450.0],
            "最高价": [104000.0, 105500.0],
            "最低价": [102000.0, 104100.0],
            "收盘价": [103880.0, 105000.0],
            "成交量": [1000, 2000],
        }
    )


class FakeFuturesAK:
    def futures_main_sina(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return _futures_main_frame()

    def futures_zh_daily_sina(self, symbol: str) -> pd.DataFrame:
        raise IndexError("legacy sina endpoint failed")

    def get_futures_daily(self, start_date: str, end_date: str, market: str) -> pd.DataFrame:
        raise AssertionError("exchange daily fallback should not be needed")


class NightlyQuantOrdersTests(unittest.TestCase):
    def test_fetch_us_data_uses_batch_download_with_timeout(self) -> None:
        fake_yf = FakeBatchYF(_yfinance_batch_frame(["AAPL", "MSFT"]))

        data = _fetch_us_data(fake_yf, ["AAPL", "MSFT"], end_date="2026-05-29")

        self.assertEqual(len(fake_yf.calls), 1)
        requested_symbols, kwargs = fake_yf.calls[0]
        self.assertEqual(list(requested_symbols), ["AAPL", "MSFT"])
        self.assertEqual(kwargs.get("group_by"), "ticker")
        self.assertTrue(kwargs.get("threads"))
        self.assertIn("timeout", kwargs)
        self.assertIn("AAPL", data)
        self.assertIn("MSFT", data)
        self.assertEqual(str(data["AAPL"]["date"].iloc[-1]), "2026-05-29")
        self.assertAlmostEqual(float(data["MSFT"]["close"].iloc[-1]), 111.5)

    def test_fetch_us_data_degrades_quickly_when_yfinance_fails(self) -> None:
        fake_yf = FakeBatchYF(fail=True)

        with patch.dict("os.environ", {"QTS_US_SERIAL_FALLBACK_LIMIT": "2"}):
            data = _fetch_us_data(fake_yf, ["AAPL", "MSFT", "NVDA"], end_date="2026-05-29")

        self.assertEqual(data, {})
        self.assertEqual(len(fake_yf.calls), 3)
        self.assertEqual(list(fake_yf.calls[0][0]), ["AAPL", "MSFT", "NVDA"])
        self.assertEqual(fake_yf.calls[1][0], "AAPL")
        self.assertEqual(fake_yf.calls[2][0], "MSFT")

    def test_optional_yf_frame_and_pct_change_tolerate_provider_failure(self) -> None:
        fake_yf = FakeBatchYF(fail=True)

        frame = _optional_yf_frame(fake_yf, "QQQ", period="6mo")

        self.assertTrue(frame.empty)
        self.assertEqual(_pct_change_over_window(frame, 60), 0.0)

    def test_fetch_futures_data_prefers_main_sina_when_legacy_endpoint_is_empty(self) -> None:
        data = _fetch_futures_data(FakeFuturesAK(), ["CU0"], end_date="2026-05-29", exchange="SHFE")

        self.assertIn("CU0", data)
        self.assertEqual(str(data["CU0"]["date"].iloc[-1]), "2026-05-29")
        self.assertEqual(float(data["CU0"]["close"].iloc[-1]), 105000.0)
        self.assertEqual(data["CU0"].attrs.get("price_source"), "futures_main_sina")

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

    def test_validate_futures_close_surfaces_price_source(self) -> None:
        settle = MarketValidation(
            market="SHFE",
            requested_date="2026-05-29",
            actual_date="2026-05-29",
            passed=True,
            reason="SHFE 结算参数直接命中 T 日。",
            sample_count=10,
        )
        frame = _frame("2026-05-29", 105000.0)
        frame.attrs["price_source"] = "futures_main_sina"
        validation = _validate_futures_close("SHFE", {"CU0": frame}, settle)

        self.assertTrue(validation.passed)
        self.assertEqual(validation.price_source, "futures_main_sina")
        self.assertIn("price_source=futures_main_sina", validation.reason)

    def test_validate_cffex_uses_contract_specs_fallback_when_settle_is_stale(self) -> None:
        settle = MarketValidation(
            market="CFFEX",
            requested_date="2026-05-27",
            actual_date="2026-05-18",
            passed=True,
            reason="CFFEX T 日无结算参数，最近有效交易日回退到 2026-05-18。",
            sample_count=46,
        )
        validation = _validate_futures_close("CFFEX", {"IF0": _frame("2026-05-27", 4200.0)}, settle)
        self.assertTrue(validation.passed)
        self.assertEqual(validation.actual_date, "2026-05-27")
        self.assertEqual(validation.settlement_fallback, CFFEX_SETTLE_FALLBACK_CONTRACT_SPECS)
        self.assertEqual(validation.margin_source, "contract_specs")
        self.assertIn("settlement_fallback=daily_main_contract+contract_specs", validation.reason)

    def test_non_cffex_settle_daily_mismatch_still_fails(self) -> None:
        settle = MarketValidation(
            market="SHFE",
            requested_date="2026-05-27",
            actual_date="2026-05-18",
            passed=True,
            reason="SHFE T 日无结算参数，最近有效交易日回退到 2026-05-18。",
            sample_count=46,
        )
        validation = _validate_futures_close("SHFE", {"CU0": _frame("2026-05-27", 100000.0)}, settle)
        self.assertFalse(validation.passed)
        self.assertIn("时间戳不一致", validation.reason)

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
        self.assertTrue(any("净收益" in line for line in lines))
        self.assertTrue(any("SAFE_RESERVE" in line for line in lines))

    def test_build_recap_surfaces_cost_drag_after_execution_costs(self) -> None:
        previous_report = {
            "report_date": "2026-05-10",
            "execution_actions": [
                {
                    "market": "HK",
                    "bucket_action": "CORE_SIGNAL",
                    "action": "LONG",
                    "symbol": "00700.HK",
                    "target_weight": 1.0,
                    "reference_close": 100.0,
                    "return_model": "close_to_close",
                    "execution_cost_assumption": {"commission_bps": 4.0, "slippage_bps": 3.0, "impact_bps": 2.0},
                }
            ],
        }
        lines = _build_recap(previous_report, {"00700.HK": {"close": 100.1}})

        self.assertEqual(len(lines), 1)
        self.assertIn("毛收益 +0.10%", lines[0])
        self.assertIn("净收益 -0.08%", lines[0])
        self.assertIn("归因=cost_drag", lines[0])

    def test_build_recap_consolidates_legacy_shared_futures_defensive_legs(self) -> None:
        previous_report = {
            "report_date": "2026-05-10",
            "execution_actions": [
                {
                    "market": "SHFE",
                    "bucket_action": "TAIL_HEDGE",
                    "action": "LONG",
                    "symbol": "AU0",
                    "target_weight": 0.23,
                    "reference_close": 1000.0,
                    "return_model": "close_to_close",
                },
                {
                    "market": "GFEX",
                    "bucket_action": "TAIL_HEDGE",
                    "action": "LONG",
                    "symbol": "AU0",
                    "target_weight": 0.23,
                    "reference_close": 1000.0,
                    "return_model": "close_to_close",
                },
                {
                    "market": "SHFE",
                    "bucket_action": "SAFE_RESERVE",
                    "action": "HOLD",
                    "symbol": "CNY_CASH",
                    "target_weight": 0.77,
                    "reference_close": 1.0,
                    "return_model": "cash_flat",
                },
                {
                    "market": "GFEX",
                    "bucket_action": "SAFE_RESERVE",
                    "action": "HOLD",
                    "symbol": "CNY_CASH",
                    "target_weight": 0.77,
                    "reference_close": 1.0,
                    "return_model": "cash_flat",
                },
            ],
        }
        lines = _build_recap(previous_report, {"AU0": {"close": 1000.0}, "CNY_CASH": {"close": 1.0}})

        self.assertEqual(sum("AU0" in line for line in lines), 1)
        self.assertEqual(sum("CNY_CASH" in line for line in lines), 1)

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
        self.assertEqual(execution[0]["margin_source"], "contract_specs")

    def test_materialize_execution_actions_blocks_low_net_edge_active_signal(self) -> None:
        actions = [
            {
                "action": "LONG",
                "symbol": "00700.HK",
                "target_weight": 0.15,
                "confidence": 0.50,
                "objective_score": 0.10,
                "reason": "weak edge",
            }
        ]
        price_map = {"00700.HK": {"date": "2026-05-11", "close": 480.0}}

        execution = _materialize_execution_actions(actions, "HK", price_map, "2026-05-11")

        self.assertEqual(execution[0]["action"], "HOLD")
        self.assertEqual(execution[0]["symbol"], "HKD_CASH")
        self.assertEqual(execution[0]["bucket_action"], "COST_GATE_REJECTED")
        self.assertEqual(execution[0]["rejected_symbol"], "00700.HK")
        self.assertEqual(execution[0]["net_edge_gate"]["decision"], "OBSERVE_ONLY")
        self.assertLess(
            execution[0]["net_edge_gate"]["expected_edge_bps"],
            execution[0]["net_edge_gate"]["required_edge_bps"],
        )

    def test_materialize_execution_actions_allows_high_net_edge_active_signal(self) -> None:
        actions = [
            {
                "action": "LONG",
                "symbol": "00700.HK",
                "target_weight": 0.15,
                "confidence": 0.90,
                "objective_score": 0.80,
                "reason": "strong edge",
            }
        ]
        price_map = {"00700.HK": {"date": "2026-05-11", "close": 480.0}}

        execution = _materialize_execution_actions(actions, "HK", price_map, "2026-05-11")

        self.assertEqual(execution[0]["action"], "LONG")
        self.assertEqual(execution[0]["symbol"], "00700.HK")
        self.assertEqual(execution[0]["net_edge_gate"]["decision"], "ALLOW")
        self.assertGreaterEqual(
            execution[0]["net_edge_gate"]["expected_edge_bps"],
            execution[0]["net_edge_gate"]["required_edge_bps"],
        )

    def test_materialize_execution_actions_caps_active_signal_by_kelly_and_capacity(self) -> None:
        actions = [
            {
                "action": "LONG",
                "symbol": "00700.HK",
                "target_weight": 0.30,
                "confidence": 0.90,
                "objective_score": 0.80,
                "kelly_fraction": 0.12,
                "capacity_weight_limit": 0.15,
                "reason": "strong but size capped",
            }
        ]
        price_map = {"00700.HK": {"date": "2026-05-11", "close": 480.0}}

        execution = _materialize_execution_actions(actions, "HK", price_map, "2026-05-11")

        self.assertEqual(execution[0]["action"], "LONG")
        self.assertAlmostEqual(execution[0]["target_weight"], 0.12)
        self.assertEqual(execution[0]["position_sizing_gate"]["decision"], "REDUCE")
        self.assertEqual(execution[0]["position_sizing_gate"]["binding_constraint"], "kelly_fraction")
        self.assertAlmostEqual(execution[0]["position_sizing_gate"]["weight_reduction"], 0.18)

    def test_materialize_execution_actions_caps_high_contradiction_game_signal(self) -> None:
        actions = [
            {
                "action": "LONG",
                "symbol": "00700.HK",
                "target_weight": 0.15,
                "confidence": 0.90,
                "objective_score": 0.80,
                "game_probability_audit": {
                    "contradiction_score": 0.90,
                    "exposure_scaler": 0.10,
                    "dominant_probability": 0.55,
                },
                "reason": "strong but contested game",
            }
        ]
        price_map = {"00700.HK": {"date": "2026-05-11", "close": 480.0}}

        execution = _materialize_execution_actions(actions, "HK", price_map, "2026-05-11")

        self.assertEqual(execution[0]["action"], "LONG")
        self.assertEqual(execution[0]["position_sizing_gate"]["decision"], "REDUCE")
        self.assertEqual(execution[0]["position_sizing_gate"]["binding_constraint"], "game_probability_position_multiplier")
        self.assertLess(execution[0]["target_weight"], 0.15)
        self.assertEqual(
            execution[0]["position_sizing_gate"]["game_probability_execution_overlay"]["status"],
            "active",
        )

    def test_materialize_execution_actions_uses_execution_feedback_to_raise_costs(self) -> None:
        actions = [
            {
                "action": "LONG",
                "symbol": "00700.HK",
                "target_weight": 0.15,
                "confidence": 0.80,
                "objective_score": 0.40,
                "execution_feedback": {
                    "fill_ratio": 0.60,
                    "realized_slippage_bps": 15.0,
                    "realized_impact_bps": 10.0,
                    "cost_drag_bps": 5.0,
                },
                "reason": "paper edge with poor execution",
            }
        ]
        price_map = {"00700.HK": {"date": "2026-05-11", "close": 480.0}}

        execution = _materialize_execution_actions(actions, "HK", price_map, "2026-05-11")

        self.assertEqual(execution[0]["bucket_action"], "COST_GATE_REJECTED")
        self.assertEqual(execution[0]["net_edge_gate"]["decision"], "OBSERVE_ONLY")
        self.assertGreater(execution[0]["net_edge_gate"]["cost_bps"], 100.0)
        self.assertEqual(
            execution[0]["net_edge_gate"]["game_probability_execution_overlay"]["status"],
            "missing",
        )

    def test_materialize_execution_actions_applies_mae_mfe_feedback(self) -> None:
        actions = [
            {
                "action": "LONG",
                "symbol": "00700.HK",
                "target_weight": 0.15,
                "confidence": 0.90,
                "objective_score": 0.80,
                "stop_loss_pct": 0.08,
                "take_profit_pct": 0.03,
                "mae_mfe_feedback": {
                    "enabled": True,
                    "sample_size": 5,
                    "method": "mae_mfe_closed_trade_quantiles",
                    "source_backtest_id": "bt1",
                    "recommended_risk_limits": {
                        "enabled": True,
                        "sample_size": 5,
                        "stop_loss_pct": 0.02,
                        "take_profit_pct": 0.08,
                        "trailing_activation_pct": 0.05,
                        "trailing_stop_pct": 0.015,
                    },
                    "diagnostics": {"mfe_utilization_rate": 0.42},
                },
            }
        ]
        price_map = {"00700.HK": {"date": "2026-05-11", "close": 480.0}}

        execution = _materialize_execution_actions(actions, "HK", price_map, "2026-05-11")

        self.assertEqual(execution[0]["mae_mfe_risk_overlay"]["status"], "applied")
        self.assertAlmostEqual(execution[0]["stop_loss_pct"], 0.02)
        self.assertAlmostEqual(execution[0]["take_profit_pct"], 0.08)
        self.assertAlmostEqual(execution[0]["trailing_activation_pct"], 0.05)
        self.assertAlmostEqual(execution[0]["trailing_stop_pct"], 0.015)
        line = _build_instruction(execution[0], 480.0)
        self.assertIn("跟踪止盈", line)

    def test_materialize_execution_actions_keeps_base_risk_when_mae_mfe_sample_short(self) -> None:
        actions = [
            {
                "action": "LONG",
                "symbol": "00700.HK",
                "target_weight": 0.15,
                "confidence": 0.90,
                "objective_score": 0.80,
                "stop_loss_pct": 0.08,
                "take_profit_pct": 0.03,
                "mae_mfe_feedback": {
                    "enabled": True,
                    "sample_size": 1,
                    "recommended_risk_limits": {
                        "enabled": True,
                        "sample_size": 1,
                        "stop_loss_pct": 0.02,
                        "take_profit_pct": 0.08,
                    },
                },
            }
        ]
        price_map = {"00700.HK": {"date": "2026-05-11", "close": 480.0}}

        execution = _materialize_execution_actions(actions, "HK", price_map, "2026-05-11")

        self.assertEqual(execution[0]["mae_mfe_risk_overlay"]["status"], "insufficient_samples")
        self.assertAlmostEqual(execution[0]["stop_loss_pct"], 0.08)
        self.assertAlmostEqual(execution[0]["take_profit_pct"], 0.03)

    def test_cycle_payload_primary_lines_follow_net_edge_gate(self) -> None:
        cycle = {
            "status": "ok",
            "trade_actions": [
                {
                    "action": "LONG",
                    "symbol": "00700.HK",
                    "target_weight": 0.15,
                    "confidence": 0.50,
                    "objective_score": 0.10,
                    "reason": "weak edge",
                },
                {"action": "TAIL_HEDGE", "symbol": "TAIL_RISK_PROTECTION", "target_weight": 0.1, "reason": "hedge"},
            ],
            "symbols": {},
        }
        price_map = {"00700.HK": {"date": "2026-05-11", "close": 480.0}, "02840.HK": {"date": "2026-05-11", "close": 100.0}}

        payload = _build_cycle_payload("HK", cycle, price_map, price_map, "2026-05-11")

        self.assertEqual(payload["primary_actions"], [])
        self.assertEqual(payload["primary_lines"], [])
        self.assertEqual(payload["execution_actions"][0]["bucket_action"], "COST_GATE_REJECTED")
        self.assertEqual(payload["execution_actions"][1]["bucket_action"], "TAIL_HEDGE")
        self.assertIn("拒绝标的 00700.HK", payload["execution_lines"][0])
        self.assertIn("净边际", payload["execution_lines"][0])

    def test_consolidate_shared_cn_futures_defensive_legs(self) -> None:
        actions = [
            {
                "market": "SHFE",
                "bucket_action": "TAIL_HEDGE",
                "action": "LONG",
                "symbol": "AU0",
                "target_weight": 0.23,
                "reference_close": 1082.4,
                "reference_date": "2026-05-29",
                "reason": "tail",
            },
            {
                "market": "INE",
                "bucket_action": "TAIL_HEDGE",
                "action": "LONG",
                "symbol": "AU0",
                "target_weight": 0.23,
                "reference_close": 1082.4,
                "reference_date": "2026-05-29",
                "reason": "tail",
            },
            {
                "market": "DCE",
                "bucket_action": "SAFE_RESERVE",
                "action": "HOLD",
                "symbol": "CNY_CASH",
                "target_weight": 0.75,
                "reference_close": 1.0,
                "reference_date": "2026-05-29",
                "reason": "cash",
            },
            {
                "market": "GFEX",
                "bucket_action": "SAFE_RESERVE",
                "action": "HOLD",
                "symbol": "CNY_CASH",
                "target_weight": 0.77,
                "reference_close": 1.0,
                "reference_date": "2026-05-29",
                "reason": "cash",
            },
            {
                "market": "DCE",
                "bucket_action": "LONG",
                "action": "LONG",
                "symbol": "I0",
                "target_weight": 0.08,
                "reference_close": 805.5,
            },
        ]

        consolidated = _consolidate_shared_futures_defensive_actions(actions)

        au0 = [item for item in consolidated if item.get("symbol") == "AU0"]
        cash = [item for item in consolidated if item.get("symbol") == "CNY_CASH"]
        active = [item for item in consolidated if item.get("symbol") == "I0"]
        self.assertEqual(len(au0), 1)
        self.assertEqual(len(cash), 1)
        self.assertEqual(len(active), 1)
        self.assertEqual(au0[0]["source_markets"], ["SHFE", "INE"])
        self.assertTrue(au0[0]["consolidated_shared_futures_defensive_leg"])
        self.assertEqual(cash[0]["target_weight"], 0.77)
        self.assertEqual(cash[0]["consolidation_method"], "max_target_weight")
        self.assertIn("防止 AU0/CNY_CASH 重复下单", au0[0]["reason"])

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
        self.assertAlmostEqual(summary["net_portfolio_return"], 0.00982)
        self.assertAlmostEqual(summary["execution_cost_return"], 0.00018)
        self.assertAlmostEqual(summary["slippage_bps"], 18.0)
        self.assertAlmostEqual(summary["gross_weight"], 0.1)
        self.assertEqual(summary["risk_asset_count"], 1)
        self.assertIn("elasticity", summary)

    def test_evaluate_execution_actions_counts_all_cn_futures_markets(self) -> None:
        actions = [
            {
                "market": "DCE",
                "bucket_action": "CORE_SIGNAL",
                "action": "LONG",
                "symbol": "I0",
                "target_weight": 0.1,
                "reference_close": 100.0,
                "return_model": "close_to_close",
            },
            {
                "market": "INE",
                "bucket_action": "CORE_SIGNAL",
                "action": "SHORT",
                "symbol": "SC0",
                "target_weight": 0.2,
                "reference_close": 200.0,
                "return_model": "close_to_close",
            },
        ]
        current_prices = {"I0": {"close": 110.0}, "SC0": {"close": 180.0}}
        summary = _evaluate_execution_actions(actions, current_prices)

        self.assertAlmostEqual(summary["portfolio_return"], 0.032222, places=6)
        self.assertAlmostEqual(summary["net_portfolio_return"], 0.031982, places=6)
        self.assertAlmostEqual(summary["execution_cost_return"], 0.00024, places=8)
        self.assertAlmostEqual(summary["slippage_bps"], 8.0)
        self.assertAlmostEqual(summary["gross_weight"], 0.3)
        self.assertAlmostEqual(summary["futures_weight"], 0.3)
        self.assertEqual(summary["risk_asset_count"], 2)

    def test_evaluate_execution_actions_uses_net_win_rate_after_costs(self) -> None:
        actions = [
            {
                "market": "HK",
                "bucket_action": "CORE_SIGNAL",
                "action": "LONG",
                "symbol": "00700.HK",
                "target_weight": 1.0,
                "reference_close": 100.0,
                "return_model": "close_to_close",
                "execution_cost_assumption": {"commission_bps": 4.0, "slippage_bps": 3.0, "impact_bps": 2.0},
            }
        ]
        current_prices = {"00700.HK": {"close": 100.1}}
        summary = _evaluate_execution_actions(actions, current_prices)

        self.assertGreater(summary["portfolio_return"], 0.0)
        self.assertLess(summary["net_portfolio_return"], 0.0)
        self.assertEqual(summary["gross_win_rate"], 1.0)
        self.assertEqual(summary["win_rate"], 0.0)
        self.assertEqual(summary["payoff_ratio"], 0.0)
        self.assertEqual(summary["failure_attribution"], "cost_drag")
        self.assertTrue(summary["cost_drag_flag"])
        self.assertAlmostEqual(summary["cost_drag_ratio"], 1.8)
        self.assertAlmostEqual(summary["details"][0]["return_pct"], 0.001)
        self.assertAlmostEqual(summary["details"][0]["net_return_pct"], -0.0008)

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

    def test_weekly_execution_review_aggregates_all_cn_futures_exchanges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            report_dir = repo_root / "state" / "nightly_reports"
            report_dir.mkdir(parents=True)
            report = {
                "report_date": "2026-05-25",
                "execution_actions": [
                    {
                        "market": "HK",
                        "bucket_action": "SAFE_RESERVE",
                        "action": "HOLD",
                        "symbol": "HKD_CASH",
                        "target_weight": 1.0,
                        "reference_close": 1.0,
                        "return_model": "cash_flat",
                    },
                    {
                        "market": "HK",
                        "bucket_action": "COST_GATE_REJECTED",
                        "action": "HOLD",
                        "symbol": "HKD_CASH",
                        "rejected_symbol": "00700.HK",
                        "target_weight": 0.1,
                        "reference_close": 1.0,
                        "return_model": "cash_flat",
                        "net_edge_gate": {
                            "decision": "OBSERVE_ONLY",
                            "expected_edge_bps": 5.0,
                            "cost_bps": 18.0,
                            "required_edge_bps": 21.6,
                        },
                    },
                    {
                        "market": "DCE",
                        "bucket_action": "CORE_SIGNAL",
                        "action": "LONG",
                        "symbol": "I0",
                        "target_weight": 0.1,
                        "reference_close": 100.0,
                        "return_model": "close_to_close",
                        "position_sizing_gate": {
                            "decision": "REDUCE",
                            "original_target_weight": 0.18,
                            "capped_target_weight": 0.10,
                            "weight_reduction": 0.08,
                            "binding_constraint": "kelly_fraction",
                            "caps": {
                                "max_active_single_weight": 0.20,
                                "kelly_fraction": 0.10,
                                "capacity_weight_limit": 0.20,
                            },
                        },
                        "mae_mfe_risk_overlay": {
                            "status": "applied",
                            "applied": True,
                            "sample_size": 8,
                            "method": "mae_mfe_closed_trade_quantiles",
                            "base_risk_limits": {"stop_loss_pct": 0.08, "take_profit_pct": 0.06},
                            "resolved_risk_limits": {
                                "stop_loss_pct": 0.03,
                                "take_profit_pct": 0.10,
                                "trailing_activation_pct": 0.05,
                                "trailing_stop_pct": 0.02,
                            },
                        },
                    },
                    {
                        "market": "INE",
                        "bucket_action": "CORE_SIGNAL",
                        "action": "SHORT",
                        "symbol": "SC0",
                        "target_weight": 0.2,
                        "reference_close": 200.0,
                        "return_model": "close_to_close",
                        "position_sizing_gate": {
                            "decision": "ALLOW",
                            "original_target_weight": 0.20,
                            "capped_target_weight": 0.20,
                            "weight_reduction": 0.0,
                            "binding_constraint": None,
                            "caps": {
                                "max_active_single_weight": 0.20,
                                "kelly_fraction": 0.25,
                                "capacity_weight_limit": 0.30,
                            },
                        },
                        "mae_mfe_risk_overlay": {
                            "status": "insufficient_samples",
                            "applied": False,
                            "sample_size": 1,
                            "min_samples": 2,
                        },
                    },
                    {
                        "market": "SHFE",
                        "bucket_action": "TAIL_HEDGE",
                        "action": "LONG",
                        "symbol": "AU0",
                        "target_weight": 0.23,
                        "reference_close": 1000.0,
                        "return_model": "close_to_close",
                    },
                    {
                        "market": "GFEX",
                        "bucket_action": "TAIL_HEDGE",
                        "action": "LONG",
                        "symbol": "AU0",
                        "target_weight": 0.23,
                        "reference_close": 1000.0,
                        "return_model": "close_to_close",
                    },
                    {
                        "market": "SHFE",
                        "bucket_action": "SAFE_RESERVE",
                        "action": "HOLD",
                        "symbol": "CNY_CASH",
                        "target_weight": 0.65,
                        "reference_close": 1.0,
                        "return_model": "cash_flat",
                    },
                    {
                        "market": "GFEX",
                        "bucket_action": "SAFE_RESERVE",
                        "action": "HOLD",
                        "symbol": "CNY_CASH",
                        "target_weight": 0.65,
                        "reference_close": 1.0,
                        "return_model": "cash_flat",
                    },
                ],
            }
            (report_dir / "2026-05-25.json").write_text(json.dumps(report), encoding="utf-8")

            with patch("quant_trade_system.nightly_quant_orders._repo_root", return_value=repo_root), patch(
                "quant_trade_system.nightly_quant_orders._fetch_historical_price_maps",
                return_value={
                    "HKD_CASH": {"close": 1.0},
                    "I0": {"close": 110.0},
                    "SC0": {"close": 180.0},
                    "AU0": {"close": 1000.0},
                    "CNY_CASH": {"close": 1.0},
                },
            ) as fetch_prices:
                review = generate_weekly_execution_review(
                    date(2026, 5, 25),
                    date(2026, 5, 29),
                    evaluation_date=date(2026, 5, 29),
                )

        futures_symbols_by_exchange = fetch_prices.call_args.args[2]
        self.assertEqual(futures_symbols_by_exchange["DCE"], {"I0"})
        self.assertEqual(futures_symbols_by_exchange["INE"], {"SC0"})
        self.assertIn("CN_FUTURES", review["aggregation_assumption"])
        self.assertAlmostEqual(review["days"][0]["cn_futures"]["portfolio_return"], 0.032222, places=6)
        self.assertAlmostEqual(review["days"][0]["cn_futures"]["net_portfolio_return"], 0.031798, places=6)
        self.assertAlmostEqual(review["days"][0]["cn_futures"]["execution_cost_return"], 0.000424, places=8)
        self.assertAlmostEqual(review["days"][0]["combined_gross_return"], 0.016111, places=6)
        self.assertAlmostEqual(review["days"][0]["cn_futures"]["futures_weight"], 0.53)
        self.assertEqual(len(review["days"][0]["cn_futures"]["details"]), 4)
        self.assertTrue(review["constraint_checks"]["max_single_weight_ok"])
        self.assertTrue(review["constraint_checks"]["max_tail_hedge_weight_ok"])
        self.assertFalse(review["constraint_checks"]["max_futures_weight_ok"])
        self.assertAlmostEqual(review["combined_return"], 0.015899, places=6)
        self.assertAlmostEqual(review["cn_futures_return"], 0.031798, places=6)
        self.assertEqual(review["shfe_return"], review["cn_futures_return"])
        self.assertEqual(
            review["robustness_validation"]["promotion_decision"],
            "OBSERVE_ONLY_INSUFFICIENT_SAMPLE",
        )
        self.assertEqual(review["robustness_validation"]["cpcv"]["status"], "insufficient_observations")
        self.assertEqual(review["robustness_validation"]["effective_breadth"]["status"], "insufficient_observations")
        self.assertEqual(review["days"][0]["net_edge_gate"]["rejected_count"], 1)
        self.assertAlmostEqual(review["net_edge_gate_metrics"]["prevented_execution_cost_return"], 0.00018, places=8)
        self.assertEqual(review["net_edge_gate_metrics"]["top_rejected_actions"][0]["symbol"], "00700.HK")
        self.assertEqual(review["days"][0]["position_sizing_gate"]["reduced_count"], 1)
        self.assertEqual(review["position_sizing_metrics"]["reduced_count"], 1)
        self.assertAlmostEqual(review["position_sizing_metrics"]["total_weight_reduction"], 0.08)
        self.assertEqual(review["position_sizing_metrics"]["binding_constraint_counts"]["kelly_fraction"], 1)
        self.assertEqual(review["days"][0]["mae_mfe_risk_overlay"]["applied_count"], 1)
        self.assertEqual(review["mae_mfe_overlay_metrics"]["applied_count"], 1)
        self.assertEqual(review["mae_mfe_overlay_metrics"]["insufficient_samples_count"], 1)
        self.assertAlmostEqual(review["mae_mfe_overlay_metrics"]["avg_stop_loss_delta_pct"], -0.05)
        self.assertAlmostEqual(review["mae_mfe_overlay_metrics"]["avg_take_profit_delta_pct"], 0.04)
        self.assertTrue(
            any(item["action"] == "extend_shadow_window_before_promotion" for item in review["optimization_recommendations"])
        )
        self.assertTrue(
            any(item["action"] == "keep_net_edge_execution_gate_active" for item in review["optimization_recommendations"])
        )
        self.assertTrue(
            any(item["action"] == "keep_fractional_kelly_capacity_sizing_gate" for item in review["optimization_recommendations"])
        )
        self.assertTrue(
            any(item["action"] == "keep_mae_mfe_adaptive_risk_overlay" for item in review["optimization_recommendations"])
        )
        self.assertTrue(review["optimization_recommendations"])

    def test_weekly_execution_review_flags_tail_hedge_cap_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            report_dir = repo_root / "state" / "nightly_reports"
            report_dir.mkdir(parents=True)
            report = {
                "report_date": "2026-05-25",
                "execution_actions": [
                    {
                        "market": "HK",
                        "bucket_action": "TAIL_HEDGE",
                        "action": "LONG",
                        "symbol": "02840.HK",
                        "target_weight": 0.26,
                        "reference_close": 100.0,
                        "return_model": "close_to_close",
                    },
                    {
                        "market": "HK",
                        "bucket_action": "SAFE_RESERVE",
                        "action": "HOLD",
                        "symbol": "HKD_CASH",
                        "target_weight": 0.74,
                        "reference_close": 1.0,
                        "return_model": "cash_flat",
                    },
                ],
            }
            (report_dir / "2026-05-25.json").write_text(json.dumps(report), encoding="utf-8")

            with patch("quant_trade_system.nightly_quant_orders._repo_root", return_value=repo_root), patch(
                "quant_trade_system.nightly_quant_orders._fetch_historical_price_maps",
                return_value={"02840.HK": {"close": 100.0}, "HKD_CASH": {"close": 1.0}},
            ):
                review = generate_weekly_execution_review(
                    date(2026, 5, 25),
                    date(2026, 5, 29),
                    evaluation_date=date(2026, 5, 29),
                )

        self.assertTrue(review["constraint_checks"]["max_single_weight_ok"])
        self.assertFalse(review["constraint_checks"]["max_tail_hedge_weight_ok"])
        self.assertTrue(
            any(item["action"] == "cap_tail_hedge_weight" for item in review["optimization_recommendations"])
        )

    def test_weekly_quality_metrics_preserve_price_unavailable(self) -> None:
        metrics = _aggregate_weekly_quality_metrics(
            [
                {
                    "risk_asset_count": 1,
                    "win_rate": 1.0,
                    "payoff_ratio": 2.0,
                    "elasticity": 1.5,
                    "price_unavailable_count": 0,
                    "execution_cost_return": 0.0001,
                    "net_portfolio_return": 0.01,
                    "gross_weight": 0.5,
                    "slippage_bps": 2.0,
                    "failure_attribution": "net_positive",
                },
                {
                    "risk_asset_count": 1,
                    "win_rate": 0.0,
                    "payoff_ratio": 0.0,
                    "elasticity": 0.8,
                    "price_unavailable_count": 0,
                    "execution_cost_return": 0.0002,
                    "net_portfolio_return": -0.001,
                    "gross_weight": 0.5,
                    "slippage_bps": 4.0,
                    "cost_drag_flag": True,
                    "cost_drag_ratio": 1.5,
                    "failure_attribution": "cost_drag",
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
        self.assertIn("cost_drag", metrics["failure_attribution"])
        self.assertEqual(metrics["cost_drag_count"], 1)
        self.assertAlmostEqual(metrics["execution_cost_return"], 0.0003)
        self.assertAlmostEqual(metrics["net_portfolio_return"], 0.009)

    def test_weekly_optimization_recommendations_surface_cost_drag(self) -> None:
        recommendations = _build_weekly_optimization_recommendations(
            {
                "failure_attribution": ["cost_drag"],
                "cost_drag_count": 2,
                "slippage_bps": 13.0,
                "execution_cost_return": 0.000598,
                "net_portfolio_return": -0.000598,
            },
            {
                "max_single_weight_ok": True,
                "max_tail_hedge_weight_ok": True,
                "max_futures_weight_ok": True,
                "max_gross_weight_ok": True,
            },
            [{"report_date": "2026-05-29"}],
        )

        actions = {item["action"]: item for item in recommendations}
        self.assertIn("raise_min_net_edge_bps", actions)
        self.assertIn("reduce_zero_edge_defensive_turnover", actions)
        self.assertEqual(actions["raise_min_net_edge_bps"]["severity"], "high")
        self.assertGreaterEqual(
            actions["raise_min_net_edge_bps"]["suggested_parameters"]["min_expected_net_edge_bps"],
            15.6,
        )

    def test_weekly_recommendations_surface_missing_mae_mfe_feedback(self) -> None:
        recommendations = _build_weekly_optimization_recommendations(
            {
                "failure_attribution": [],
                "cost_drag_count": 0,
                "slippage_bps": 0.0,
                "net_portfolio_return": 0.01,
            },
            {
                "max_single_weight_ok": True,
                "max_tail_hedge_weight_ok": True,
                "max_futures_weight_ok": True,
                "max_gross_weight_ok": True,
            },
            [{"report_date": "2026-05-29"}],
            {"promotion_decision": "PROMOTION_ALLOWED"},
            {},
            {"evaluated_count": 2, "applied_count": 0, "missing_count": 2},
        )

        actions = {item["action"]: item for item in recommendations}
        self.assertIn("connect_backtest_mae_mfe_feedback_to_nightly_actions", actions)
        self.assertEqual(
            actions["connect_backtest_mae_mfe_feedback_to_nightly_actions"]["suggested_parameters"]["source"],
            "backtest.stats.mae_mfe_feedback",
        )

    def test_weekly_recommendations_surface_missing_position_sizing_metadata(self) -> None:
        recommendations = _build_weekly_optimization_recommendations(
            {
                "failure_attribution": [],
                "cost_drag_count": 0,
                "slippage_bps": 0.0,
                "net_portfolio_return": 0.01,
            },
            {
                "max_single_weight_ok": True,
                "max_tail_hedge_weight_ok": True,
                "max_futures_weight_ok": True,
                "max_gross_weight_ok": True,
            },
            [{"report_date": "2026-05-29"}],
            {"promotion_decision": "PROMOTION_ALLOWED"},
            {},
            {},
            {"evaluated_count": 0, "missing_count": 2, "reduced_count": 0},
        )

        actions = {item["action"]: item for item in recommendations}
        self.assertIn("connect_fractional_kelly_and_capacity_metadata", actions)
        self.assertIn(
            "kelly_fraction",
            actions["connect_fractional_kelly_and_capacity_metadata"]["suggested_parameters"]["required_fields"],
        )

    def test_weekly_path_capture_metrics_identify_low_capture_efficiency(self) -> None:
        metrics = _build_weekly_path_capture_metrics([1.0, 0.98, 1.04, 0.96, 1.06], combined_return=0.001)

        self.assertEqual(metrics["status"], "ready")
        self.assertGreater(metrics["dc_path_return"], 0.0)
        self.assertEqual(metrics["capture_status"], "BELOW_TARGET")

    def test_weekly_recommendations_surface_low_path_capture(self) -> None:
        recommendations = _build_weekly_optimization_recommendations(
            {
                "failure_attribution": [],
                "cost_drag_count": 0,
                "slippage_bps": 0.0,
                "net_portfolio_return": 0.01,
            },
            {
                "max_single_weight_ok": True,
                "max_tail_hedge_weight_ok": True,
                "max_futures_weight_ok": True,
                "max_gross_weight_ok": True,
            },
            [{"report_date": "2026-05-29"}],
            robustness_validation={"promotion_decision": "PROMOTION_ALLOWED"},
            net_edge_gate_metrics={},
            mae_mfe_overlay_metrics={},
            position_sizing_metrics={},
            path_capture_metrics={
                "status": "ready",
                "capture_status": "BELOW_TARGET",
                "capture_ratio": 0.01,
                "target_band": "5%-20%",
                "theta_bps": 25.0,
            },
        )

        actions = {item["action"]: item for item in recommendations}
        self.assertIn("improve_path_capture_efficiency", actions)
        self.assertEqual(actions["improve_path_capture_efficiency"]["suggested_parameters"]["theta_bps"], 25.0)

    def test_weekly_robustness_validation_blocks_short_sample_promotion(self) -> None:
        validation = _build_weekly_robustness_validation(
            [
                {
                    "combined_return": 0.01,
                    "hk": {"net_portfolio_return": 0.012},
                    "cn_futures": {"net_portfolio_return": 0.008},
                },
                {
                    "combined_return": -0.002,
                    "hk": {"net_portfolio_return": -0.001},
                    "cn_futures": {"net_portfolio_return": -0.003},
                },
            ],
            {"net_portfolio_return": 0.008, "cost_drag_count": 0},
        )

        self.assertEqual(validation["observation_count"], 2)
        self.assertEqual(validation["cpcv"]["status"], "insufficient_observations")
        self.assertEqual(validation["deflated_sharpe_ratio"]["status"], "insufficient_observations")
        self.assertEqual(validation["effective_breadth"]["status"], "insufficient_observations")
        self.assertEqual(validation["promotion_decision"], "OBSERVE_ONLY_INSUFFICIENT_SAMPLE")

    def test_weekly_recommendations_surface_robustness_gate_failures(self) -> None:
        recommendations = _build_weekly_optimization_recommendations(
            {"failure_attribution": [], "net_portfolio_return": 0.02},
            {
                "max_single_weight_ok": True,
                "max_tail_hedge_weight_ok": True,
                "max_futures_weight_ok": True,
                "max_gross_weight_ok": True,
            },
            [{"report_date": "2026-05-29"}],
            {
                "promotion_decision": "NO_PROMOTION_ROBUSTNESS_GATE_FAILED",
                "decision_reason": "CPCV 或 DSR 未通过。",
            },
        )

        self.assertEqual(recommendations[0]["action"], "block_strategy_promotion_until_cpcv_dsr_pass")
        self.assertEqual(recommendations[0]["severity"], "high")

    def test_weekly_recommendations_surface_net_edge_cost_savings(self) -> None:
        recommendations = _build_weekly_optimization_recommendations(
            {"failure_attribution": [], "net_portfolio_return": 0.02},
            {
                "max_single_weight_ok": True,
                "max_tail_hedge_weight_ok": True,
                "max_futures_weight_ok": True,
                "max_gross_weight_ok": True,
            },
            [{"report_date": "2026-05-29"}],
            {"promotion_decision": "PROMOTION_ALLOWED"},
            {
                "rejected_count": 3,
                "allowed_count": 1,
                "prevented_execution_cost_return": 0.00042,
                "top_rejected_actions": [{"symbol": "00700.HK"}],
            },
        )

        actions = {item["action"]: item for item in recommendations}
        self.assertIn("keep_net_edge_execution_gate_active", actions)
        self.assertIn("improve_signal_quality_before_more_breadth", actions)
        self.assertEqual(actions["improve_signal_quality_before_more_breadth"]["severity"], "medium")

    def test_render_report_text_marks_invalid_market_without_global_failure(self) -> None:
        report = {
            "status": "partial_ok",
            "failure_category": FAILURE_DATA_VALIDATION_PARTIAL,
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
        self.assertIn("失败分类：DATA_VALIDATION_PARTIAL", text)
        self.assertIn("DCE：NO_TRADE_DATA_INVALID", text)
        self.assertNotIn("没有任何市场通过数据校验", text)

    def test_failure_category_from_market_status(self) -> None:
        self.assertEqual(_failure_category_from_market_status({}), FAILURE_SCHEDULER_NOT_RUN)
        self.assertEqual(
            _failure_category_from_market_status({"US": {"tradable": True}, "HK": {"tradable": True}}),
            FAILURE_NONE,
        )
        self.assertEqual(
            _failure_category_from_market_status({"US": {"tradable": True}, "DCE": {"tradable": False}}),
            FAILURE_DATA_VALIDATION_PARTIAL,
        )
        self.assertEqual(
            _failure_category_from_market_status({"US": {"tradable": False}, "HK": {"tradable": False}}),
            FAILURE_ALL_MARKETS_INVALID,
        )

    def test_data_quality_scores_fresh_and_fallback_markets(self) -> None:
        us_validation = MarketValidation("US", "2026-05-15", "2026-05-15", True, "ok", sample_count=3)
        hk_validation = MarketValidation("HK", "2026-05-15", "2026-05-12", True, "最近有效交易日回退。", sample_count=1)
        futures_data = {
            "SHFE": {
                "validation": MarketValidation(
                    "SHFE",
                    "2026-05-15",
                    "2026-05-15",
                    True,
                    "SHFE 结算参数直接命中 T 日。",
                    sample_count=2,
                    price_source="futures_main_sina",
                )
            }
        }

        quality = _build_data_quality_report(
            us_validation,
            hk_validation,
            futures_data,
            {"US": 3, "HK": 3, "SHFE": 2},
        )

        self.assertEqual(quality["markets"]["US"]["decision"], "ALLOW")
        self.assertLess(quality["markets"]["HK"]["score"], quality["markets"]["US"]["score"])
        self.assertIn("date_lag_days=3", quality["markets"]["HK"]["reasons"])
        self.assertIn("HK", quality["summary"]["weak_markets"])

    def test_data_quality_attaches_to_market_status_and_context(self) -> None:
        market_status = {"US": {"status": "OK", "tradable": True, "reason": "ok"}}
        quality = {
            "markets": {
                "US": {
                    "score": 0.61,
                    "decision": "OBSERVE_ONLY",
                    "reasons": ["low_sample_coverage"],
                }
            }
        }

        updated = _attach_data_quality_to_market_status(market_status, quality)
        context = _market_context_with_data_quality({"crisis_probability": 0.12}, quality, "US")

        self.assertEqual(updated["US"]["data_quality_score"], 0.61)
        self.assertEqual(updated["US"]["data_quality_decision"], "OBSERVE_ONLY")
        self.assertFalse(context["data_validation_passed"])
        self.assertEqual(context["data_quality_reasons"], ["low_sample_coverage"])

    def test_render_report_text_includes_data_quality_summary(self) -> None:
        report = {
            "status": "failed_validation",
            "failure_category": FAILURE_ALL_MARKETS_INVALID,
            "report_date": "2026-05-15",
            "generated_at": "2026-05-15T20:00:00+08:00",
            "repo": {"branch": "main", "head": "abc123", "origin_main": "abc123", "synced_with_origin_main": True},
            "us_validation": {"passed": False, "requested_date": "2026-05-15", "actual_date": None, "reason": "empty"},
            "hk_validation": {"passed": True, "requested_date": "2026-05-15", "actual_date": "2026-05-15", "reason": "ok"},
            "futures_validations": {},
            "market_status": {"US": {"status": "NO_TRADE_DATA_INVALID", "reason": "empty"}},
            "data_quality": {
                "summary": {"min_score": 0.0, "avg_score": 0.5, "decision_counts": {"NO_TRADE": 1}},
                "markets": {"US": {"score": 0.0, "decision": "NO_TRADE", "date_lag_days": None, "coverage_score": 0.0}},
            },
            "recap_lines": [],
        }

        text = render_report_text(report)

        self.assertIn("数据质量评分", text)
        self.assertIn("US：score=0.000 decision=NO_TRADE", text)

    def test_nightly_health_check_detects_missing_and_failed_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            report_dir = repo_root / "state" / "nightly_reports"
            report_dir.mkdir(parents=True)

            missing = check_nightly_health(date(2026, 5, 20), repo_root=repo_root)
            self.assertEqual(missing["failure_category"], FAILURE_SCHEDULER_NOT_RUN)
            self.assertFalse(missing["report_exists"])
            self.assertEqual(missing["scheduler_status"], {})

            status_dir = repo_root / "state" / "nightly_status"
            status_dir.mkdir(parents=True)
            scheduler_status = {
                "status": "failed",
                "reason": "nightly_runner_failed",
                "run_date": "2026-05-22",
                "run_exit": 143,
                "health_exit": 2,
                "log_file": str(repo_root / "state" / "nightly_logs" / "nightly_quant_orders_2026-05-22.log"),
            }
            (status_dir / "nightly_quant_orders_2026-05-22.json").write_text(
                json.dumps(scheduler_status),
                encoding="utf-8",
            )
            runtime_failed = check_nightly_health(date(2026, 5, 22), repo_root=repo_root)
            self.assertEqual(runtime_failed["status"], "failed")
            self.assertEqual(runtime_failed["failure_category"], FAILURE_RUNTIME_EXCEPTION)
            self.assertEqual(runtime_failed["scheduler_status"]["run_exit"], 143)
            self.assertIn("调度器已运行但未产出报告", runtime_failed["reason"])

            failed_report = {
                "status": "failed_validation",
                "failure_category": FAILURE_ALL_MARKETS_INVALID,
                "report_date": "2026-05-20",
                "generated_at": "2026-05-20T20:00:00+08:00",
                "market_status": {"US": {"tradable": False, "status": "NO_TRADE_DATA_INVALID"}},
            }
            (report_dir / "2026-05-20.json").write_text(json.dumps(failed_report), encoding="utf-8")
            failed = check_nightly_health(date(2026, 5, 20), repo_root=repo_root)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["failure_category"], FAILURE_ALL_MARKETS_INVALID)

            ok_report = {
                "status": "partial_ok",
                "failure_category": FAILURE_DATA_VALIDATION_PARTIAL,
                "report_date": "2026-05-21",
                "generated_at": "2026-05-21T20:00:00+08:00",
                "market_status": {"US": {"tradable": True}, "DCE": {"tradable": False}},
            }
            (report_dir / "2026-05-21.json").write_text(json.dumps(ok_report), encoding="utf-8")
            ok = check_nightly_health(date(2026, 5, 21), repo_root=repo_root)
            self.assertEqual(ok["status"], "ok")
            self.assertEqual(ok["failure_category"], FAILURE_DATA_VALIDATION_PARTIAL)

    def test_evidence_snapshot_links_models_features_and_confirmations(self) -> None:
        report = {
            "report_date": "2026-05-11",
            "generated_at": "2026-05-11T20:00:00+08:00",
            "repo": {"head": "abc123"},
            "us_validation": {"passed": True, "actual_date": "2026-05-08"},
            "hk_validation": {"passed": True, "actual_date": "2026-05-11"},
            "futures_validations": {"SHFE": {"passed": True, "actual_date": "2026-05-11"}},
            "data_quality": {
                "summary": {"min_score": 0.72, "avg_score": 0.86, "weak_markets": ["US"]},
                "markets": {"US": {"score": 0.72, "decision": "REDUCE"}},
            },
            "execution_actions": [
                {
                    "market": "HK",
                    "bucket_action": "COST_GATE_REJECTED",
                    "action": "HOLD",
                    "symbol": "HKD_CASH",
                    "rejected_symbol": "00700.HK",
                    "target_weight": 0.15,
                    "net_edge_gate": {
                        "decision": "OBSERVE_ONLY",
                        "expected_edge_bps": 5.0,
                        "cost_bps": 18.0,
                        "required_edge_bps": 21.6,
                        "reason": "edge too weak",
                    },
                },
                {
                    "market": "HK",
                    "bucket_action": "CORE_SIGNAL",
                    "action": "LONG",
                    "symbol": "09988.HK",
                    "target_weight": 0.10,
                    "mae_mfe_risk_overlay": {
                        "status": "applied",
                        "applied": True,
                        "sample_size": 6,
                        "method": "mae_mfe_closed_trade_quantiles",
                        "resolved_risk_limits": {
                            "stop_loss_pct": 0.02,
                            "take_profit_pct": 0.08,
                            "trailing_activation_pct": 0.05,
                            "trailing_stop_pct": 0.015,
                        },
                    },
                },
            ],
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
        self.assertEqual(snapshot["net_edge_execution_gate"]["rejected_count"], 1)
        self.assertEqual(snapshot["net_edge_execution_gate"]["rejected_actions"][0]["symbol"], "00700.HK")
        self.assertEqual(snapshot["mae_mfe_risk_overlay"]["applied_count"], 1)
        self.assertEqual(snapshot["mae_mfe_risk_overlay"]["applied_actions"][0]["stop_loss_pct"], 0.02)
        self.assertEqual(snapshot["data_quality"]["summary"]["min_score"], 0.72)


if __name__ == "__main__":
    unittest.main()
