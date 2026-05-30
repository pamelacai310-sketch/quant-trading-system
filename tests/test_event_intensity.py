from __future__ import annotations

import unittest

import pandas as pd

from quant_trade_system.core.causal import EventIntensityEngine


class EventIntensityEngineTests(unittest.TestCase):
    def test_event_intensity_outputs_zscore_momentum_and_asset_exposure(self) -> None:
        engine = EventIntensityEngine()
        records = [
            {
                "timestamp": "2026-05-01",
                "title": "Iran war risk threatens Hormuz oil shipping",
                "relevance_score": 1.0,
                "sentiment_score": -0.8,
            },
            {
                "timestamp": "2026-05-06",
                "title": "OPEC supply disruption lifts crude oil risk premium",
                "relevance_score": 0.9,
                "sentiment_score": -0.6,
            },
        ]
        calendar = pd.date_range("2026-05-01", periods=20, freq="D")

        snapshot = engine.fit_transform(records, calendar_index=calendar, as_of="2026-05-20")

        self.assertEqual(snapshot.status, "active")
        self.assertIn("event_intensity_geopolitical_energy", snapshot.feature_frame.columns)
        self.assertIn("event_zscore_geopolitical_energy", snapshot.feature_frame.columns)
        self.assertIn("event_asset_sc_exposure", snapshot.feature_frame.columns)
        self.assertGreater(snapshot.latest_values["event_asset_sc_exposure"], 0.0)

    def test_event_intensity_audit_can_include_serializable_records(self) -> None:
        engine = EventIntensityEngine()
        snapshot = engine.fit_transform(
            [{"timestamp": "2026-05-01", "title": "Fed policy risk and tariff uncertainty"}],
            calendar_index=pd.date_range("2026-05-01", periods=5, freq="D"),
            as_of="2026-05-05",
        )

        audit = snapshot.to_audit_dict(include_records=True)

        self.assertIn("feature_frame_records", audit)
        self.assertEqual(len(audit["feature_frame_records"]), 5)
        self.assertIn("formula", audit)


if __name__ == "__main__":
    unittest.main()
