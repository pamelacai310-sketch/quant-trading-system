from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_trade_system.futures_dc_research import (
    DCFuturesStrategySpec,
    FuturesCostModel,
    apply_holdout_validation,
    apply_multiple_testing_control,
    aggregate_cross_product_research_results,
    aggregate_research_results,
    build_research_report,
    candidate_history_path,
    classify_walk_forward_result,
    cost_sensitivity_audit,
    fetch_main_contract_minute_frames,
    fetch_cached_main_contract_minute_frames,
    load_candidate_history,
    load_cached_minute_frames,
    load_csv_minute_frames,
    load_minute_cache,
    minute_cache_path,
    normalize_minute_frame,
    random_direction_control,
    simulate_dc_strategy,
    split_research_holdout_frames,
    summarize_candidate_history,
    update_candidate_history,
    update_minute_cache,
    write_research_outputs,
)


def _minute_frame(closes: list[float]) -> pd.DataFrame:
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2026-06-01 09:00", periods=len(close), freq="5min"),
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1000,
            "hold": range(10_000, 10_000 + len(close)),
        }
    )


class FuturesDCResearchTests(unittest.TestCase):
    def test_default_cost_model_uses_round_trip_cost(self) -> None:
        self.assertAlmostEqual(FuturesCostModel().round_trip_cost_bps, 6.3)

    def test_fetch_main_contracts_preserves_product_order(self) -> None:
        class FakeAk:
            def match_main_contract(self, symbol: str):
                return {
                    "cffex": "IF2606,IC2606",
                    "shfe": "CU2607,AG2608",
                    "dce": "M2609,I2609",
                }.get(symbol, "")

            def futures_zh_minute_sina(self, symbol: str, period: str):
                return _minute_frame([100, 101, 102, 103])

        frames = fetch_main_contract_minute_frames(
            products=["IF", "CU", "AG"],
            period="5",
            max_contracts=2,
            ak=FakeAk(),
        )

        self.assertEqual(list(frames), ["IF2606", "CU2607"])

    def test_minute_cache_merges_and_deduplicates_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = _minute_frame([100, 101, 102])
            second = _minute_frame([101, 102, 103, 104]).iloc[1:].copy()

            merged = update_minute_cache("IF2606", first, period="1", cache_dir=tmpdir)
            merged = update_minute_cache("IF2606", second, period="1", cache_dir=tmpdir)
            cached = load_minute_cache("IF2606", period="1", cache_dir=tmpdir)

            self.assertTrue(minute_cache_path("IF2606", period="1", cache_dir=tmpdir).exists())
            self.assertEqual(len(merged), 4)
            self.assertEqual(len(cached), 4)
            self.assertEqual(cached["close"].tolist(), [100.0, 102.0, 103.0, 104.0])

    def test_cached_fetch_updates_local_cache(self) -> None:
        class FakeAk:
            def __init__(self) -> None:
                self.calls = 0

            def match_main_contract(self, symbol: str):
                return {"cffex": "IF2606,IC2606"}.get(symbol, "")

            def futures_zh_minute_sina(self, symbol: str, period: str):
                self.calls += 1
                return _minute_frame([100 + self.calls, 101 + self.calls, 102 + self.calls])

        with tempfile.TemporaryDirectory() as tmpdir:
            ak = FakeAk()
            first = fetch_cached_main_contract_minute_frames(
                products=["IF"],
                period="5",
                cache_dir=tmpdir,
                ak=ak,
            )
            second = fetch_cached_main_contract_minute_frames(
                products=["IF"],
                period="5",
                cache_dir=tmpdir,
                ak=ak,
            )

            self.assertEqual(list(first), ["IF2606"])
            self.assertEqual(list(second), ["IF2606"])
            self.assertEqual(len(second["IF2606"]), 3)
            self.assertEqual(ak.calls, 2)

    def test_load_cached_minute_frames_reads_period_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            update_minute_cache("IF2606", _minute_frame([100, 101, 102]), period="1", cache_dir=tmpdir)
            update_minute_cache("IC2606", _minute_frame([200, 201, 202]), period="1", cache_dir=tmpdir)
            update_minute_cache("IF2606", _minute_frame([300, 301, 302]), period="5", cache_dir=tmpdir)

            frames = load_cached_minute_frames(cache_dir=tmpdir, period="1")
            filtered = load_cached_minute_frames(cache_dir=tmpdir, period="1", symbols=["IF2606"])

            self.assertEqual(list(frames), ["IC2606", "IF2606"])
            self.assertEqual(list(filtered), ["IF2606"])
            self.assertEqual(filtered["IF2606"]["close"].tolist(), [100.0, 101.0, 102.0])

    def test_load_csv_minute_frames_uses_file_stem_and_symbol_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = Path(tmpdir) / "IF2606.csv"
            mixed_path = Path(tmpdir) / "mixed.csv"
            _minute_frame([100, 101, 102]).to_csv(first_path, index=False)
            mixed = pd.concat(
                [
                    _minute_frame([200, 201]).assign(symbol="IC2606"),
                    _minute_frame([300, 301]).assign(symbol="CU2607"),
                    _minute_frame([202]).assign(symbol="IC2606"),
                ],
                ignore_index=True,
            )
            mixed.to_csv(mixed_path, index=False)

            frames = load_csv_minute_frames([first_path, mixed_path])

            self.assertEqual(list(frames), ["CU2607", "IC2606", "IF2606"])
            self.assertEqual(frames["IF2606"]["close"].tolist(), [100.0, 101.0, 102.0])
            self.assertEqual(frames["IC2606"]["close"].tolist(), [202.0, 201.0])
            self.assertEqual(frames["CU2607"]["close"].tolist(), [300.0, 301.0])

    def test_simulation_enters_after_dc_confirmation_next_bar(self) -> None:
        frame = _minute_frame([100, 99, 98, 102, 105, 104, 100, 96, 95, 98, 101, 103, 100, 97, 96, 99, 103])
        spec = DCFuturesStrategySpec(symbol="IF2606", family="dc_continuation", theta_bps=300, max_hold_bars=20)

        result = simulate_dc_strategy(
            frame,
            spec,
            cost_model=FuturesCostModel(commission_bps=1, slippage_bps=1, impact_bps=1),
        )

        self.assertGreaterEqual(result["event_count"], 3)
        self.assertGreater(result["trade_count"], 0)
        self.assertGreater(result["dc_path_return"], 0)
        self.assertTrue(all(trade["entry_lag_bars"] == 1 for trade in result["trades"]))
        self.assertTrue(all(trade["entry_index"] > trade["signal_index"] for trade in result["trades"]))

    def test_overshoot_family_waits_for_confirmed_post_dc_move(self) -> None:
        frame = _minute_frame([100, 99, 98, 102, 103, 106, 108, 110])
        spec = DCFuturesStrategySpec(
            symbol="IF2606",
            family="dc_overshoot_continuation",
            theta_bps=300,
            overshoot_trigger_multiple=0.5,
            max_hold_bars=20,
        )

        result = simulate_dc_strategy(frame, spec, cost_model=FuturesCostModel())

        self.assertGreater(result["trade_count"], 0)
        self.assertTrue(all(trade["entry_lag_bars"] > 1 for trade in result["trades"]))
        self.assertEqual(result["trades"][0]["entry_index"], 6)

    def test_normalize_minute_frame_accepts_akshare_style_columns(self) -> None:
        raw = pd.DataFrame(
            {
                "datetime": ["2026-06-01 09:00:00", "2026-06-01 09:05:00"],
                "open": ["100", "101"],
                "high": ["102", "103"],
                "low": ["99", "100"],
                "close": ["101", "102"],
                "volume": ["10", "11"],
                "hold": ["1000", "1001"],
            }
        )

        normalized = normalize_minute_frame(raw)

        self.assertEqual(list(normalized.columns[:7]), ["timestamp", "open", "high", "low", "close", "volume", "hold"])
        self.assertEqual(normalized["close"].tolist(), [101.0, 102.0])
        self.assertEqual(normalized["_hold_diff"].iloc[-1], 1.0)

    def test_classification_requires_average_oos_not_local_luck(self) -> None:
        spec = DCFuturesStrategySpec(symbol="M2609", family="dc_reversal", theta_bps=24)
        cost = FuturesCostModel(commission_bps=0.8, slippage_bps=1.5, impact_bps=1.7)
        folds = [
            {"train_return": 0.02, "test_return": 0.02, "test_capture_ratio": 0.10, "test_events": 25, "test_trades": 12},
            {"train_return": 0.01, "test_return": 0.01, "test_capture_ratio": 0.08, "test_events": 24, "test_trades": 12},
            {"train_return": 0.01, "test_return": -0.06, "test_capture_ratio": -0.20, "test_events": 23, "test_trades": 12},
        ]

        result = classify_walk_forward_result(spec, cost, folds)

        self.assertEqual(result["status"], "FAIL")
        self.assertIn("NON_POSITIVE_EXPECTANCY", result["failure_reasons"])
        self.assertIn("UNSTABLE_FOLD_EDGE", result["failure_reasons"])

    def test_classification_rejects_average_capture_masking_fold_instability(self) -> None:
        spec = DCFuturesStrategySpec(symbol="AG2608", family="dc_continuation", theta_bps=24)
        cost = FuturesCostModel(commission_bps=0.8, slippage_bps=1.5, impact_bps=1.7)
        folds = [
            {"train_return": 0.02, "test_return": 0.010, "test_capture_ratio": 0.047, "test_events": 28, "test_trades": 10},
            {"train_return": 0.03, "test_return": 0.001, "test_capture_ratio": 0.006, "test_events": 22, "test_trades": 10},
            {"train_return": 0.02, "test_return": 0.027, "test_capture_ratio": 0.107, "test_events": 27, "test_trades": 12},
        ]

        result = classify_walk_forward_result(spec, cost, folds)

        self.assertEqual(result["status"], "WATCH")
        self.assertIn("FOLD_CAPTURE_INSTABILITY", result["failure_reasons"])
        self.assertIn("MEDIAN_CAPTURE_OUT_OF_TARGET", result["failure_reasons"])
        self.assertIn("UNSTABLE_FOLD_EDGE", result["failure_reasons"])

    def test_cost_sensitivity_detects_cost_fragile_edge(self) -> None:
        trades = [
            [{"side": "long", "entry_price": 100.0, "exit_price": 100.08}],
            [{"side": "short", "entry_price": 100.0, "exit_price": 99.92}],
        ]

        audit = cost_sensitivity_audit(trades, FuturesCostModel())

        self.assertGreater(audit["cost_1_0x"]["avg_return"], 0)
        self.assertLess(audit["cost_1_5x"]["avg_return"], 0)

    def test_random_direction_control_is_deterministic_and_reports_p_value(self) -> None:
        trades = [
            [{"side": "long", "entry_price": 100.0, "exit_price": 101.0} for _ in range(8)],
            [{"side": "short", "entry_price": 100.0, "exit_price": 99.0} for _ in range(8)],
        ]

        first = random_direction_control(trades, FuturesCostModel(), trials=64, seed=7)
        second = random_direction_control(trades, FuturesCostModel(), trials=64, seed=7)

        self.assertEqual(first, second)
        self.assertLess(first["p_value"], 0.10)
        self.assertTrue(first["beats_p95"])

    def test_classification_rejects_audits_that_do_not_survive_controls(self) -> None:
        spec = DCFuturesStrategySpec(symbol="IF2606", family="dc_reversal", theta_bps=32)
        cost = FuturesCostModel()
        folds = [
            {"train_return": 0.02, "test_return": 0.02, "test_capture_ratio": 0.10, "test_events": 25, "test_trades": 12},
            {"train_return": 0.02, "test_return": 0.01, "test_capture_ratio": 0.08, "test_events": 24, "test_trades": 12},
            {"train_return": 0.02, "test_return": -0.001, "test_capture_ratio": 0.06, "test_events": 23, "test_trades": 12},
        ]

        result = classify_walk_forward_result(
            spec,
            cost,
            folds,
            cost_sensitivity={"cost_1_5x": {"avg_return": -0.001}},
            random_control={"beats_p95": False, "p_value": 0.60},
        )

        self.assertEqual(result["status"], "WATCH")
        self.assertIn("COST_STRESS_FAIL", result["failure_reasons"])
        self.assertIn("RANDOM_CONTROL_NOT_BEATEN", result["failure_reasons"])

    def test_classification_passes_only_strict_repeatable_capture(self) -> None:
        spec = DCFuturesStrategySpec(symbol="AG2608", family="dc_continuation", theta_bps=24)
        cost = FuturesCostModel(commission_bps=0.8, slippage_bps=1.5, impact_bps=1.7)
        folds = [
            {"train_return": 0.02, "test_return": 0.02, "test_capture_ratio": 0.10, "test_events": 25, "test_trades": 12},
            {"train_return": 0.01, "test_return": 0.01, "test_capture_ratio": 0.08, "test_events": 24, "test_trades": 12},
            {"train_return": 0.01, "test_return": -0.001, "test_capture_ratio": 0.06, "test_events": 23, "test_trades": 12},
        ]

        result = classify_walk_forward_result(spec, cost, folds)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["failure_reasons"], [])

    def test_multiple_testing_control_downgrades_lucky_scan_result(self) -> None:
        cost = FuturesCostModel()
        folds = [
            {"train_return": 0.02, "test_return": 0.02, "test_capture_ratio": 0.10, "test_events": 25, "test_trades": 12},
            {"train_return": 0.02, "test_return": 0.01, "test_capture_ratio": 0.08, "test_events": 24, "test_trades": 12},
            {"train_return": 0.02, "test_return": -0.001, "test_capture_ratio": 0.06, "test_events": 23, "test_trades": 12},
        ]
        candidate = classify_walk_forward_result(
            DCFuturesStrategySpec(symbol="IF2606", family="dc_reversal", theta_bps=32),
            cost,
            folds,
            cost_sensitivity={"cost_1_5x": {"avg_return": 0.01}},
            random_control={"beats_p95": True, "p_value": 0.04},
        )
        null_rows = [
            classify_walk_forward_result(
                DCFuturesStrategySpec(symbol=f"IC260{i}", family="dc_reversal", theta_bps=32),
                cost,
                [{"train_return": 0.0, "test_return": -0.01, "test_capture_ratio": -0.10, "test_events": 25, "test_trades": 12}],
                random_control={"beats_p95": False, "p_value": 1.0},
            )
            for i in range(3)
        ]

        controlled = apply_multiple_testing_control([candidate, *null_rows], alpha=0.10)
        controlled_candidate = next(row for row in controlled if row["spec"]["symbol"] == "IF2606")

        self.assertEqual(controlled_candidate["status"], "WATCH")
        self.assertAlmostEqual(controlled_candidate["summary"]["random_control_q_value"], 0.16)
        self.assertIn("RANDOM_CONTROL_FDR_NOT_SIGNIFICANT", controlled_candidate["failure_reasons"])

    def test_multiple_testing_control_preserves_fdr_significant_result(self) -> None:
        cost = FuturesCostModel()
        folds = [
            {"train_return": 0.02, "test_return": 0.02, "test_capture_ratio": 0.10, "test_events": 25, "test_trades": 12},
            {"train_return": 0.02, "test_return": 0.01, "test_capture_ratio": 0.08, "test_events": 24, "test_trades": 12},
            {"train_return": 0.02, "test_return": -0.001, "test_capture_ratio": 0.06, "test_events": 23, "test_trades": 12},
        ]
        candidate = classify_walk_forward_result(
            DCFuturesStrategySpec(symbol="AU2608", family="dc_continuation", theta_bps=32),
            cost,
            folds,
            cost_sensitivity={"cost_1_5x": {"avg_return": 0.01}},
            random_control={"beats_p95": True, "p_value": 0.01},
        )
        weak = classify_walk_forward_result(
            DCFuturesStrategySpec(symbol="AG2608", family="dc_continuation", theta_bps=32),
            cost,
            [{"train_return": 0.0, "test_return": -0.01, "test_capture_ratio": -0.10, "test_events": 25, "test_trades": 12}],
            random_control={"beats_p95": False, "p_value": 0.90},
        )

        controlled = apply_multiple_testing_control([candidate, weak], alpha=0.10)
        controlled_candidate = next(row for row in controlled if row["spec"]["symbol"] == "AU2608")

        self.assertEqual(controlled_candidate["status"], "PASS")
        self.assertAlmostEqual(controlled_candidate["summary"]["random_control_q_value"], 0.02)
        self.assertNotIn("RANDOM_CONTROL_FDR_NOT_SIGNIFICANT", controlled_candidate["failure_reasons"])

    def test_split_research_holdout_frames_reserves_final_bars(self) -> None:
        frame = _minute_frame([100 + index for index in range(100)])

        research, holdout = split_research_holdout_frames(
            {"IF2606": frame},
            holdout_fraction=0.25,
            min_research_bars=50,
            min_holdout_bars=10,
        )

        self.assertEqual(len(research["IF2606"]), 75)
        self.assertEqual(len(holdout["IF2606"]), 25)
        self.assertLess(research["IF2606"]["timestamp"].max(), holdout["IF2606"]["timestamp"].min())

    def test_holdout_validation_downgrades_candidate_without_final_confirmation(self) -> None:
        cost = FuturesCostModel()
        candidate = classify_walk_forward_result(
            DCFuturesStrategySpec(symbol="IF2606", family="dc_reversal", theta_bps=32),
            cost,
            [
                {"train_return": 0.02, "test_return": 0.02, "test_capture_ratio": 0.10, "test_events": 25, "test_trades": 12},
                {"train_return": 0.02, "test_return": 0.01, "test_capture_ratio": 0.08, "test_events": 24, "test_trades": 12},
                {"train_return": 0.02, "test_return": -0.001, "test_capture_ratio": 0.06, "test_events": 23, "test_trades": 12},
            ],
            cost_sensitivity={"cost_1_5x": {"avg_return": 0.01}},
            random_control={"beats_p95": True, "p_value": 0.01},
        )

        validated = apply_holdout_validation(
            [candidate],
            {"IF2606": _minute_frame([100.0] * 80)},
            cost_model=cost,
            min_holdout_events=1,
            min_holdout_trades=1,
        )
        row = validated[0]

        self.assertEqual(row["status"], "WATCH")
        self.assertEqual(row["summary"]["holdout_events"], 0)
        self.assertEqual(row["summary"]["holdout_trades"], 0)
        self.assertIn("HOLDOUT_NON_POSITIVE_EXPECTANCY", row["failure_reasons"])
        self.assertIn("HOLDOUT_LOW_EVENT_COUNT", row["failure_reasons"])

    def test_aggregate_research_results_requires_multiple_contracts(self) -> None:
        spec = DCFuturesStrategySpec(symbol="IF2606", family="dc_reversal", theta_bps=32)
        row = classify_walk_forward_result(
            spec,
            FuturesCostModel(),
            [
                {"train_return": 0.02, "test_return": 0.02, "test_capture_ratio": 0.10, "test_events": 25, "test_trades": 12},
                {"train_return": 0.02, "test_return": 0.01, "test_capture_ratio": 0.08, "test_events": 24, "test_trades": 12},
                {"train_return": 0.02, "test_return": -0.001, "test_capture_ratio": 0.06, "test_events": 23, "test_trades": 12},
            ],
            cost_sensitivity={"cost_1_5x": {"avg_return": 0.01}},
            random_control={"beats_p95": True, "p_value": 0.02},
        )

        self.assertEqual(aggregate_research_results([row]), [])

    def test_aggregate_research_results_flags_contract_rollup_divergence(self) -> None:
        cost = FuturesCostModel()
        good = classify_walk_forward_result(
            DCFuturesStrategySpec(symbol="IF2606", family="dc_reversal", theta_bps=32),
            cost,
            [
                {"train_return": 0.02, "test_return": 0.02, "test_capture_ratio": 0.10, "test_events": 25, "test_trades": 12},
                {"train_return": 0.02, "test_return": 0.01, "test_capture_ratio": 0.08, "test_events": 24, "test_trades": 12},
                {"train_return": 0.02, "test_return": -0.001, "test_capture_ratio": 0.06, "test_events": 23, "test_trades": 12},
            ],
            cost_sensitivity={"cost_1_5x": {"avg_return": 0.01}},
            random_control={"beats_p95": True, "p_value": 0.02},
        )
        weak = classify_walk_forward_result(
            DCFuturesStrategySpec(symbol="IF2609", family="dc_reversal", theta_bps=32),
            cost,
            [
                {"train_return": 0.01, "test_return": -0.02, "test_capture_ratio": -0.10, "test_events": 25, "test_trades": 12},
                {"train_return": 0.01, "test_return": -0.01, "test_capture_ratio": -0.05, "test_events": 24, "test_trades": 12},
                {"train_return": 0.01, "test_return": 0.001, "test_capture_ratio": 0.01, "test_events": 23, "test_trades": 12},
            ],
            cost_sensitivity={"cost_1_5x": {"avg_return": -0.02}},
            random_control={"beats_p95": False, "p_value": 0.70},
        )
        good["summary"].update({"holdout_return": 0.01, "holdout_capture_ratio": 0.10, "holdout_trades": 12})
        weak["summary"].update({"holdout_return": -0.02, "holdout_capture_ratio": -0.10, "holdout_trades": 12})

        aggregate = aggregate_research_results([good, weak])[0]

        self.assertEqual(aggregate["spec"]["symbol"], "IF_AGG")
        self.assertEqual(aggregate["aggregation"]["members"], ["IF2606", "IF2609"])
        self.assertIn("CROSS_CONTRACT_MEMBER_EXPECTANCY_DIVERGENCE", aggregate["failure_reasons"])
        self.assertIn("CROSS_CONTRACT_CAPTURE_DIVERGENCE", aggregate["failure_reasons"])
        self.assertIn("CROSS_CONTRACT_HOLDOUT_EXPECTANCY_DIVERGENCE", aggregate["failure_reasons"])
        self.assertIn("CROSS_CONTRACT_HOLDOUT_CAPTURE_DIVERGENCE", aggregate["failure_reasons"])
        self.assertAlmostEqual(aggregate["summary"]["holdout_return"], -0.005)

    def test_cross_product_aggregation_uses_product_representatives(self) -> None:
        cost = FuturesCostModel()
        good_folds = [
            {"train_return": 0.02, "test_return": 0.02, "test_capture_ratio": 0.10, "test_events": 25, "test_trades": 12},
            {"train_return": 0.02, "test_return": 0.01, "test_capture_ratio": 0.08, "test_events": 24, "test_trades": 12},
            {"train_return": 0.02, "test_return": -0.001, "test_capture_ratio": 0.06, "test_events": 23, "test_trades": 12},
        ]
        weak_folds = [
            {"train_return": 0.01, "test_return": -0.02, "test_capture_ratio": -0.10, "test_events": 25, "test_trades": 12},
            {"train_return": 0.01, "test_return": -0.01, "test_capture_ratio": -0.05, "test_events": 24, "test_trades": 12},
            {"train_return": 0.01, "test_return": 0.001, "test_capture_ratio": 0.01, "test_events": 23, "test_trades": 12},
        ]
        if_june = classify_walk_forward_result(
            DCFuturesStrategySpec(symbol="IF2606", family="dc_reversal", theta_bps=32),
            cost,
            good_folds,
            cost_sensitivity={"cost_1_5x": {"avg_return": 0.01}},
            random_control={"beats_p95": True, "p_value": 0.02},
        )
        if_september = classify_walk_forward_result(
            DCFuturesStrategySpec(symbol="IF2609", family="dc_reversal", theta_bps=32),
            cost,
            good_folds,
            cost_sensitivity={"cost_1_5x": {"avg_return": 0.01}},
            random_control={"beats_p95": True, "p_value": 0.02},
        )
        ic = classify_walk_forward_result(
            DCFuturesStrategySpec(symbol="IC2606", family="dc_reversal", theta_bps=32),
            cost,
            weak_folds,
            cost_sensitivity={"cost_1_5x": {"avg_return": -0.02}},
            random_control={"beats_p95": False, "p_value": 0.70},
        )
        if_june["summary"].update({"holdout_return": 0.01, "holdout_capture_ratio": 0.10, "holdout_trades": 12})
        if_september["summary"].update({"holdout_return": 0.02, "holdout_capture_ratio": 0.12, "holdout_trades": 13})
        ic["summary"].update({"holdout_return": -0.03, "holdout_capture_ratio": -0.20, "holdout_trades": 12})

        aggregate = aggregate_cross_product_research_results([if_june, if_september, ic])[0]

        self.assertEqual(aggregate["spec"]["symbol"], "CROSS_PRODUCT_AGG")
        self.assertEqual(aggregate["aggregation"]["members"], ["IC", "IF"])
        self.assertEqual(aggregate["aggregation"]["member_count"], 2)
        self.assertIn("CROSS_PRODUCT_MEMBER_EXPECTANCY_DIVERGENCE", aggregate["failure_reasons"])
        self.assertIn("CROSS_PRODUCT_CAPTURE_DIVERGENCE", aggregate["failure_reasons"])
        self.assertIn("CROSS_PRODUCT_HOLDOUT_EXPECTANCY_DIVERGENCE", aggregate["failure_reasons"])
        self.assertIn("CROSS_PRODUCT_HOLDOUT_CAPTURE_DIVERGENCE", aggregate["failure_reasons"])
        self.assertIn("CROSS_PRODUCT_SINGLE_CONTRACT_MEMBER", aggregate["failure_reasons"])

    def test_report_includes_cross_product_section(self) -> None:
        cost = FuturesCostModel()
        row = classify_walk_forward_result(
            DCFuturesStrategySpec(symbol="IF2606", family="dc_reversal", theta_bps=32),
            cost,
            [
                {"train_return": 0.02, "test_return": 0.02, "test_capture_ratio": 0.10, "test_events": 25, "test_trades": 12},
                {"train_return": 0.02, "test_return": 0.01, "test_capture_ratio": 0.08, "test_events": 24, "test_trades": 12},
                {"train_return": 0.02, "test_return": -0.001, "test_capture_ratio": 0.06, "test_events": 23, "test_trades": 12},
            ],
            cost_sensitivity={"cost_1_5x": {"avg_return": 0.01}},
            random_control={"beats_p95": True, "p_value": 0.02},
        )

        report = build_research_report([row])

        self.assertIn("## Cross-Product Aggregates", report)
        self.assertIn("no multi-product parameter groups", report)

    def test_candidate_history_deduplicates_identical_scan_and_summarizes_persistence(self) -> None:
        cost = FuturesCostModel()
        row = classify_walk_forward_result(
            DCFuturesStrategySpec(symbol="IF2606", family="dc_reversal", theta_bps=32),
            cost,
            [
                {"train_return": 0.02, "test_return": 0.02, "test_capture_ratio": 0.10, "test_events": 25, "test_trades": 12},
                {"train_return": 0.02, "test_return": 0.01, "test_capture_ratio": 0.08, "test_events": 24, "test_trades": 12},
                {"train_return": 0.02, "test_return": -0.001, "test_capture_ratio": 0.06, "test_events": 23, "test_trades": 12},
            ],
            cost_sensitivity={"cost_1_5x": {"avg_return": 0.01}},
            random_control={"beats_p95": True, "p_value": 0.01},
        )
        row["summary"].update(
            {
                "holdout_return": 0.010,
                "holdout_capture_ratio": 0.10,
                "holdout_events": 12,
                "holdout_trades": 8,
                "holdout_start": "2026-06-01 14:00:00",
                "holdout_end": "2026-06-01 15:00:00",
            }
        )
        changed = dict(row)
        changed["summary"] = dict(row["summary"])
        changed["summary"]["holdout_return"] = 0.012
        changed["summary"]["holdout_capture_ratio"] = 0.11

        with tempfile.TemporaryDirectory() as tmpdir:
            history, path = update_candidate_history(
                [row],
                state_dir=tmpdir,
                generated_at=datetime(2026, 6, 1, 15, 0, 0),
            )
            history, _ = update_candidate_history(
                [row],
                state_dir=tmpdir,
                generated_at=datetime(2026, 6, 1, 16, 0, 0),
            )
            self.assertEqual(path, candidate_history_path(tmpdir))
            self.assertEqual(len(history["scans"]), 1)

            history, _ = update_candidate_history(
                [changed],
                state_dir=tmpdir,
                generated_at=datetime(2026, 6, 2, 15, 0, 0),
            )
            loaded = load_candidate_history(tmpdir)
            summary = summarize_candidate_history(loaded, current_results=[changed])

            self.assertEqual(len(history["scans"]), 2)
            self.assertEqual(summary[0]["seen_scans"], 2)
            self.assertEqual(summary[0]["holdout_positive_scans"], 2)
            self.assertEqual(summary[0]["holdout_target_scans"], 2)
            self.assertAlmostEqual(summary[0]["median_holdout_return"], 0.011)

    def test_write_research_outputs_creates_history_and_persistence_report(self) -> None:
        cost = FuturesCostModel()
        row = classify_walk_forward_result(
            DCFuturesStrategySpec(symbol="AU2608", family="dc_continuation", theta_bps=32),
            cost,
            [
                {"train_return": 0.02, "test_return": 0.02, "test_capture_ratio": 0.10, "test_events": 25, "test_trades": 12},
                {"train_return": 0.02, "test_return": 0.01, "test_capture_ratio": 0.08, "test_events": 24, "test_trades": 12},
                {"train_return": 0.02, "test_return": -0.001, "test_capture_ratio": 0.06, "test_events": 23, "test_trades": 12},
            ],
            cost_sensitivity={"cost_1_5x": {"avg_return": 0.01}},
            random_control={"beats_p95": True, "p_value": 0.01},
        )
        row["summary"].update(
            {
                "holdout_return": 0.010,
                "holdout_capture_ratio": 0.10,
                "holdout_events": 12,
                "holdout_trades": 8,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = f"{tmpdir}/report.md"
            paths = write_research_outputs([row], state_dir=tmpdir, report_path=report_path)
            report = Path(report_path).read_text(encoding="utf-8")

            self.assertTrue(paths["history_path"].endswith("futures_dc_candidate_history.json"))
            self.assertIn("## Persistence Watchlist", report)
            self.assertIn("AU2608 dc_continuation theta=32.0", report)


if __name__ == "__main__":
    unittest.main()
