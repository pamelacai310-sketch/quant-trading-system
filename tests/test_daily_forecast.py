import unittest
import numpy as np
import pandas as pd

from quant_trade_system.daily_forecast import (
    DailyPolicy, align_bars, calibrated_radius, evidence_gate,
    feature_panel, publication_record, replay_panel,
)


def sample_frame(n=190):
    rng = np.random.default_rng(18)
    close = 100 * np.cumprod(1 + rng.normal(0, .01, n))
    return pd.DataFrame({"date": pd.bdate_range("2025-01-02", periods=n).strftime("%Y-%m-%d"),
                         "open": close, "high": close * 1.01, "low": close * .99,
                         "close": close, "volume": 1000})


class DailyForecastTests(unittest.TestCase):
    def test_native_daily_entry_point_has_explicit_target_and_no_trades(self):
        from quant_trade_system.core.causal.self_iterating_causal_engine import SelfIteratingCausalEngine
        frame = sample_frame(90)
        engine = SelfIteratingCausalEngine()
        out = engine.run_daily_research({"CU2612": frame}, frame.date.tolist(), frame.date.iloc[-1])
        self.assertEqual(out["target_horizon"], 1)
        self.assertEqual(out["target_unit"], "exchange_trading_session")
        self.assertEqual(out["trade_actions"], [])
        self.assertIsNone(out["publication"][0]["prediction_close_return"])

    def test_exchange_calendar_includes_unquoted_days_excludes_makeup_weekends(self):
        from quant_trade_system.cn_daily_calendar import sessions
        dates = sessions("2025-01-01", "2026-09-08")
        self.assertIn("2025-01-09", dates)
        self.assertIn("2025-01-24", dates)
        self.assertNotIn("2026-02-23", dates)
        self.assertNotIn("2026-02-28", dates)
        self.assertIn("2026-02-24", dates)

    def test_missing_day_and_zero_volume_do_not_create_one_day_label(self):
        frame = sample_frame(30)
        dates = frame.date.tolist()
        frame = frame.drop(index=10)
        frame.loc[11, "volume"] = 0
        out = align_bars(frame, dates, dates[-1])
        target = out.close.shift(-1) / out.close - 1
        self.assertTrue(pd.isna(target.iloc[9]))
        self.assertTrue(pd.isna(target.iloc[10]))
        self.assertTrue(pd.isna(target.iloc[11]))

    def test_duplicate_or_unknown_calendar_is_rejected(self):
        frame = sample_frame(30)
        with self.assertRaises(ValueError):
            align_bars(pd.concat([frame, frame.iloc[[-1]]]), frame.date.tolist(), frame.date.iloc[-1])
        with self.assertRaises(ValueError):
            align_bars(frame, frame.date.tolist()[1:], frame.date.iloc[-1])

    def test_future_data_changes_cannot_change_past_forecast_or_gate(self):
        frame = sample_frame()
        calendar = frame.date.tolist()
        asof = calendar[170]
        original = replay_panel({"CU2612": frame}, calendar, asof)
        altered = frame.copy()
        altered.loc[171:, ["open", "high", "low", "close"]] *= 7
        changed = replay_panel({"CU2612": altered}, calendar, asof)
        self.assertEqual(original, changed)
        latest = original["contracts"]["CU2612"]["local"]["latest"]
        self.assertLessEqual(latest["last_training_label_date"], asof)

    def test_peers_exclude_self_and_do_not_fill_missing_inputs(self):
        frame = sample_frame(50)
        frames = {s: align_bars(frame, frame.date.tolist(), frame.date.iloc[-1])
                  for s in ["CU2612", "AL2612", "ZN2612", "PB2612"]}
        before = feature_panel(frames)["CU2612"]["context"]
        frames["CU2612"].iloc[-1, frames["CU2612"].columns.get_loc("close")] *= 1.1
        after = feature_panel(frames)["CU2612"]["context"]
        self.assertEqual(before.iloc[-1].sector_return, after.iloc[-1].sector_return)
        self.assertTrue(after.market_return.isna().all())

    def test_gate_ignores_unrealized_labels_and_requires_baseline_advantage(self):
        dates = sample_frame(70).date.tolist()
        rows = [{"target_date": d, "actual": .01, "prediction": -.01} for d in dates]
        gate = evidence_gate(rows, dates[59], DailyPolicy())
        changed = [dict(r, actual=100.0) if r["target_date"] > dates[59] else r for r in rows]
        self.assertEqual(gate, evidence_gate(changed, dates[59], DailyPolicy()))
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["days"], 60)
        favorable = [dict(r, actual=.01, prediction=.009) for r in rows[:60]]
        self.assertTrue(evidence_gate(favorable, dates[59], DailyPolicy())["passed"])
        flat = [dict(r, actual=0, prediction=0) for r in rows[:60]]
        self.assertEqual(evidence_gate(flat, dates[59], DailyPolicy())["direction_advantage_lower_bound"], -.5)

    def test_calibration_uses_only_previous_errors_and_current_volatility(self):
        dates = sample_frame(45).date.tolist()
        rows = [{"target_date": d, "actual": .02, "prediction": 0, "sigma": .01} for d in dates]
        radius = calibrated_radius(rows, dates[29], .02, DailyPolicy())
        self.assertAlmostEqual(radius, .04)
        for row in rows[30:]:
            row["actual"] = 100
        self.assertEqual(radius, calibrated_radius(rows, dates[29], .02, DailyPolicy()))

    def test_retrospective_gate_pass_does_not_publish_or_trade(self):
        candidate = {"local": {"latest": {"prediction": .01, "gate": {"passed": True}}}}
        row = publication_record("CU2612", candidate, "2026-09-07")
        self.assertIsNone(row["prediction_close_return"])
        self.assertFalse(row["can_trade"])
        self.assertFalse(row["forward_validated"])
        self.assertEqual(row["status"], "abstain")


if __name__ == "__main__":
    unittest.main()
