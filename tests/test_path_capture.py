from __future__ import annotations

import math
import unittest

import pandas as pd

from quant_trade_system.path_capture import (
    capture_ratio,
    capture_target_status,
    dc_path_summary,
    directional_change_events,
    directional_change_segments,
)


class PathCaptureTests(unittest.TestCase):
    def test_directional_change_segments_use_confirmed_extremes(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=9, freq="min"),
                "close": [100, 99, 98, 100, 104, 103, 96, 98, 100],
            }
        )

        events = directional_change_events(frame, theta_bps=300)
        segments = directional_change_segments(frame, theta_bps=300, round_trip_cost_bps=10)

        self.assertEqual([event.kind for event in events], ["DC_UP", "DC_DOWN", "DC_UP"])
        self.assertEqual(
            [(segment.start_price, segment.end_price, segment.direction) for segment in segments],
            [(98.0, 104.0, "long"), (104.0, 96.0, "short")],
        )

        gross = (104 / 98 - 1) + abs(96 / 104 - 1)
        self.assertTrue(math.isclose(sum(segment.gross_return for segment in segments), gross))
        self.assertTrue(math.isclose(sum(segment.net_upper_return for segment in segments), gross - 0.002))

    def test_dc_path_summary_reports_capture_ratio_target_band(self) -> None:
        frame = pd.DataFrame({"close": [100, 96, 101, 105, 99, 94, 100]})
        summary = dc_path_summary(frame, theta_bps=300, round_trip_cost_bps=5, strategy_return=0.01)

        self.assertGreaterEqual(summary["event_count"], 2)
        self.assertGreater(summary["dc_path_return"], 0)
        self.assertEqual(summary["capture_ratio"], capture_ratio(0.01, summary["dc_path_return"]))
        self.assertEqual(capture_target_status(0.10), "IN_TARGET")
        self.assertEqual(capture_target_status(0.01), "BELOW_TARGET")
        self.assertEqual(capture_target_status(0.25), "ABOVE_TARGET")


if __name__ == "__main__":
    unittest.main()
