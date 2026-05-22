from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_trade_system.core.causal import (
    HierarchicalHMMDecoder,
    InvarianceMarketDecoder,
    InvariantDecoderConfig,
)


def _ohlcv_from_close(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=len(close), freq="D"),
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1000, 2000, len(close)),
        }
    )


class InvarianceMarketDecoderTests(unittest.TestCase):
    def test_core_invariance_features_survive_price_scaling(self) -> None:
        rows = 180
        close = pd.Series(np.linspace(100.0, 130.0, rows) + np.sin(np.linspace(0, 14, rows)))
        frame = _ohlcv_from_close(close)
        scaled = frame.copy()
        for col in ["open", "high", "low", "close"]:
            scaled[col] = scaled[col] * 10.0
        decoder = InvarianceMarketDecoder(
            InvariantDecoderConfig(min_history=80, em_iterations=6, kernel_neighbors=10)
        )

        left = decoder.fit_transform(frame, symbol="TEST").feature_frame
        right = decoder.fit_transform(scaled, symbol="TEST").feature_frame

        for col in [
            "invariance_vol_norm_ret_1",
            "invariance_vol_norm_ret_5",
            "invariance_range_shape",
            "invariance_drawdown_shape_20",
        ]:
            self.assertTrue(np.allclose(left[col].tail(80), right[col].tail(80), atol=1e-8))

    def test_hmm_decoder_detects_latest_positive_risk_on_state(self) -> None:
        rng = np.random.default_rng(7)
        negative = rng.normal(loc=-1.0, scale=0.25, size=(90, 2))
        positive = rng.normal(loc=1.0, scale=0.25, size=(90, 2))
        observations = pd.DataFrame(np.vstack([negative, positive]), columns=["ret_state", "vol_state"])
        decoder = HierarchicalHMMDecoder(
            InvariantDecoderConfig(min_history=80, top_states=3, em_iterations=8, entropy_gate=0.95)
        )

        result = decoder.fit_decode(observations)

        self.assertIn(result["status"], {"decoded", "insufficient_or_uncertain"})
        self.assertEqual(result["viterbi_state"], "risk_on")
        self.assertGreater(result["state_probabilities"]["risk_on"], result["state_probabilities"]["risk_off"])

    def test_uncertain_decoder_outputs_neutral_posteriors(self) -> None:
        close = pd.Series(np.linspace(100.0, 105.0, 40))
        decoder = InvarianceMarketDecoder(InvariantDecoderConfig(min_history=120, em_iterations=4))

        snapshot = decoder.fit_transform(_ohlcv_from_close(close), symbol="SHORT")

        self.assertEqual(snapshot.status, "insufficient_or_uncertain")
        self.assertAlmostEqual(snapshot.feature_frame["noisy_channel_long_posterior"].iloc[-1], 1.0 / 3.0)
        self.assertAlmostEqual(snapshot.feature_frame["noisy_channel_short_posterior"].iloc[-1], 1.0 / 3.0)
        self.assertAlmostEqual(snapshot.feature_frame["hmm_sub_prob_trend"].iloc[-1], 1.0 / 3.0)
        self.assertEqual(snapshot.audit_metadata["participates_in_position_sizing"], False)

    def test_kernel_analog_feature_does_not_use_future_rows_for_past_signal(self) -> None:
        rows = 130
        close = pd.Series(np.linspace(100.0, 120.0, rows) + 0.5 * np.sin(np.linspace(0, 9, rows)))
        frame = _ohlcv_from_close(close)
        shocked_future = frame.copy()
        shocked_future.loc[80:, ["open", "high", "low", "close"]] *= 3.0
        decoder = InvarianceMarketDecoder(
            InvariantDecoderConfig(min_history=60, em_iterations=4, kernel_neighbors=8, target_horizon=5)
        )

        baseline = decoder.fit_transform(frame, symbol="TEST").feature_frame
        shocked = decoder.fit_transform(shocked_future, symbol="TEST").feature_frame

        self.assertAlmostEqual(
            float(baseline["kernel_analog_forward_mean"].iloc[60]),
            float(shocked["kernel_analog_forward_mean"].iloc[60]),
            places=10,
        )

    def test_noisy_channel_posteriors_sum_to_one(self) -> None:
        rows = 160
        close = pd.Series(np.linspace(100.0, 132.0, rows) + 0.4 * np.sin(np.linspace(0, 11, rows)))
        decoder = InvarianceMarketDecoder(
            InvariantDecoderConfig(min_history=80, em_iterations=6, kernel_neighbors=10)
        )

        snapshot = decoder.fit_transform(_ohlcv_from_close(close), symbol="TEST")

        self.assertAlmostEqual(sum(snapshot.noisy_channel_posteriors.values()), 1.0, places=5)
        self.assertIn("LONG", snapshot.noisy_channel_posteriors)
        self.assertIn("SHORT", snapshot.noisy_channel_posteriors)
        self.assertIn("HOLD", snapshot.noisy_channel_posteriors)
        self.assertAlmostEqual(sum(snapshot.audit_metadata["sub_state_probabilities"].values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
