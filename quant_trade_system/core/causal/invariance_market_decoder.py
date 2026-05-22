"""Lightweight invariance, HMM and noisy-channel market decoder.

The implementation is intentionally dependency-poor: only numpy and pandas are
used so the nightly pipeline can run in the current deployment environment.
It is an engineering approximation of the requested Simons-style ideas:
scale-invariant market geometry, hidden-state decoding and noisy-channel
posterior scoring. It is not a claim to reproduce any private fund algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


DECODER_VERSION = "invariance_decoder_v1"


@dataclass
class InvariantDecoderConfig:
    min_history: int = 120
    lookback: int = 252
    top_states: int = 3
    sub_states: int = 3
    em_iterations: int = 25
    entropy_gate: float = 0.85
    kernel_neighbors: int = 20
    kernel_window: int = 5
    target_horizon: int = 5


@dataclass
class DecoderSnapshot:
    status: str
    feature_frame: pd.DataFrame
    state_probabilities: Dict[str, float]
    viterbi_state: str
    transition_stability: float
    state_entropy: float
    noisy_channel_posteriors: Dict[str, float]
    audit_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "state_probabilities": dict(self.state_probabilities),
            "viterbi_state": self.viterbi_state,
            "transition_stability": self.transition_stability,
            "state_entropy": self.state_entropy,
            "noisy_channel_posteriors": dict(self.noisy_channel_posteriors),
            "audit_metadata": dict(self.audit_metadata),
        }


class HierarchicalHMMDecoder:
    """Small Gaussian HMM decoder with deterministic initialization."""

    STATE_LABELS = ["risk_off", "transition_choppy", "risk_on"]

    def __init__(self, config: Optional[InvariantDecoderConfig] = None) -> None:
        self.config = config or InvariantDecoderConfig()

    def fit_decode(self, observation_matrix: pd.DataFrame | np.ndarray) -> Dict[str, Any]:
        matrix = self._clean_matrix(observation_matrix)
        nobs = int(matrix.shape[0])
        n_states = max(2, int(self.config.top_states))
        if nobs < self.config.min_history or matrix.shape[1] == 0:
            return self._fallback_result(nobs, n_states, "insufficient_history")

        means, variances, transition, initial = self._initialize_parameters(matrix, n_states)
        responsibilities = np.full((nobs, n_states), 1.0 / n_states)
        for _ in range(max(1, int(self.config.em_iterations))):
            log_emission = self._log_emission(matrix, means, variances)
            gamma, xi_sum, log_likelihood = self._forward_backward(log_emission, transition, initial)
            responsibilities = gamma
            weights = gamma.sum(axis=0) + 1e-9
            means = (gamma.T @ matrix) / weights[:, None]
            for state in range(n_states):
                diff = matrix - means[state]
                variances[state] = (gamma[:, state][:, None] * diff * diff).sum(axis=0) / weights[state]
            variances = np.clip(variances, 1e-5, None)
            transition = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-9)
            transition = np.clip(transition, 1e-6, 1.0)
            transition = transition / transition.sum(axis=1, keepdims=True)
            initial = np.clip(gamma[0], 1e-6, 1.0)
            initial = initial / initial.sum()

        order = np.argsort(means[:, 0])
        label_by_state = self._label_states(order, n_states)
        path = self._viterbi(self._log_emission(matrix, means, variances), transition, initial)
        latest_probs_raw = responsibilities[-1]
        latest_probs = {
            label_by_state[state]: round(float(latest_probs_raw[state]), 6)
            for state in range(n_states)
        }
        for label in self.STATE_LABELS:
            latest_probs.setdefault(label, 0.0)
        latest_state = label_by_state[int(path[-1])]
        entropy = self._normalized_entropy(latest_probs_raw)
        return {
            "status": "decoded" if entropy <= self.config.entropy_gate else "insufficient_or_uncertain",
            "n_observations": nobs,
            "state_probabilities": latest_probs,
            "viterbi_state": latest_state,
            "viterbi_path": [label_by_state[int(item)] for item in path[-10:]],
            "transition_matrix": transition.round(6).tolist(),
            "transition_stability": round(float(np.mean(np.diag(transition))), 6),
            "state_entropy": round(float(entropy), 6),
            "log_likelihood": round(float(log_likelihood), 6),
            "em_iterations": int(self.config.em_iterations),
            "version": DECODER_VERSION,
        }

    @staticmethod
    def _clean_matrix(observation_matrix: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(observation_matrix, pd.DataFrame):
            matrix = observation_matrix.to_numpy(dtype=float)
        else:
            matrix = np.asarray(observation_matrix, dtype=float)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        std = matrix.std(axis=0)
        std[std == 0] = 1.0
        return (matrix - matrix.mean(axis=0)) / std

    @staticmethod
    def _fallback_result(nobs: int, n_states: int, reason: str) -> Dict[str, Any]:
        labels = HierarchicalHMMDecoder.STATE_LABELS[:n_states]
        while len(labels) < n_states:
            labels.append(f"state_{len(labels)}")
        probs = {label: round(1.0 / n_states, 6) for label in labels}
        for label in HierarchicalHMMDecoder.STATE_LABELS:
            probs.setdefault(label, 0.0)
        return {
            "status": "insufficient_or_uncertain",
            "reason": reason,
            "n_observations": nobs,
            "state_probabilities": probs,
            "viterbi_state": "transition_choppy",
            "viterbi_path": [],
            "transition_matrix": np.eye(n_states).round(6).tolist(),
            "transition_stability": 0.0,
            "state_entropy": 1.0,
            "log_likelihood": 0.0,
            "em_iterations": 0,
            "version": DECODER_VERSION,
        }

    @staticmethod
    def _initialize_parameters(matrix: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        score = matrix[:, 0]
        quantiles = np.linspace(0, 1, n_states + 2)[1:-1]
        cuts = np.quantile(score, quantiles)
        assignments = np.digitize(score, cuts)
        means = []
        variances = []
        for state in range(n_states):
            bucket = matrix[assignments == state]
            if len(bucket) == 0:
                bucket = matrix
            means.append(bucket.mean(axis=0))
            variances.append(np.clip(bucket.var(axis=0), 1e-4, None))
        transition = np.full((n_states, n_states), 0.04 / max(n_states - 1, 1))
        np.fill_diagonal(transition, 0.96)
        transition = transition / transition.sum(axis=1, keepdims=True)
        initial = np.full(n_states, 1.0 / n_states)
        return np.vstack(means), np.vstack(variances), transition, initial

    @staticmethod
    def _log_emission(matrix: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
        nobs = matrix.shape[0]
        n_states = means.shape[0]
        out = np.zeros((nobs, n_states), dtype=float)
        for state in range(n_states):
            var = np.clip(variances[state], 1e-6, None)
            diff = matrix - means[state]
            out[:, state] = -0.5 * (np.log(2.0 * np.pi * var).sum() + ((diff * diff) / var).sum(axis=1))
        return out

    @staticmethod
    def _logsumexp(values: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        max_value = np.max(values, axis=axis, keepdims=True)
        stable = np.exp(values - max_value)
        summed = np.sum(stable, axis=axis, keepdims=True)
        result = max_value + np.log(np.maximum(summed, 1e-300))
        if axis is None:
            return np.squeeze(result)
        return np.squeeze(result, axis=axis)

    def _forward_backward(
        self,
        log_emission: np.ndarray,
        transition: np.ndarray,
        initial: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        nobs, n_states = log_emission.shape
        log_transition = np.log(np.clip(transition, 1e-12, 1.0))
        log_initial = np.log(np.clip(initial, 1e-12, 1.0))
        alpha = np.zeros((nobs, n_states), dtype=float)
        beta = np.zeros((nobs, n_states), dtype=float)
        alpha[0] = log_initial + log_emission[0]
        for t in range(1, nobs):
            alpha[t] = log_emission[t] + self._logsumexp(alpha[t - 1][:, None] + log_transition, axis=0)
        for t in range(nobs - 2, -1, -1):
            beta[t] = self._logsumexp(log_transition + log_emission[t + 1][None, :] + beta[t + 1][None, :], axis=1)
        log_likelihood = float(self._logsumexp(alpha[-1]))
        gamma_log = alpha + beta - log_likelihood
        gamma = np.exp(gamma_log)
        gamma = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-12)
        xi_sum = np.zeros((n_states, n_states), dtype=float)
        for t in range(nobs - 1):
            xi_log = (
                alpha[t][:, None]
                + log_transition
                + log_emission[t + 1][None, :]
                + beta[t + 1][None, :]
                - log_likelihood
            )
            xi_sum += np.exp(xi_log)
        return gamma, xi_sum, log_likelihood

    @staticmethod
    def _viterbi(log_emission: np.ndarray, transition: np.ndarray, initial: np.ndarray) -> np.ndarray:
        nobs, n_states = log_emission.shape
        log_transition = np.log(np.clip(transition, 1e-12, 1.0))
        scores = np.zeros((nobs, n_states), dtype=float)
        back = np.zeros((nobs, n_states), dtype=int)
        scores[0] = np.log(np.clip(initial, 1e-12, 1.0)) + log_emission[0]
        for t in range(1, nobs):
            candidates = scores[t - 1][:, None] + log_transition
            back[t] = np.argmax(candidates, axis=0)
            scores[t] = np.max(candidates, axis=0) + log_emission[t]
        path = np.zeros(nobs, dtype=int)
        path[-1] = int(np.argmax(scores[-1]))
        for t in range(nobs - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]
        return path

    @classmethod
    def _label_states(cls, order: np.ndarray, n_states: int) -> Dict[int, str]:
        labels = {}
        names = cls.STATE_LABELS[:n_states]
        while len(names) < n_states:
            names.append(f"state_{len(names)}")
        for rank, state in enumerate(order.tolist()):
            labels[int(state)] = names[rank]
        return labels

    @staticmethod
    def _normalized_entropy(probabilities: np.ndarray) -> float:
        probs = np.clip(probabilities.astype(float), 1e-12, 1.0)
        entropy = -float(np.sum(probs * np.log(probs)))
        return float(np.clip(entropy / np.log(len(probs)), 0.0, 1.0)) if len(probs) > 1 else 0.0


class InvarianceMarketDecoder:
    """Generate invariant, HMM and noisy-channel factors from OHLCV panels."""

    def __init__(self, config: Optional[InvariantDecoderConfig] = None) -> None:
        self.config = config or InvariantDecoderConfig()
        self.hmm_decoder = HierarchicalHMMDecoder(self.config)

    def fit_transform(
        self,
        frame: pd.DataFrame,
        benchmark_frame: Optional[pd.DataFrame] = None,
        peer_frames: Optional[Dict[str, pd.DataFrame]] = None,
        symbol: str = "",
    ) -> DecoderSnapshot:
        normalized = self._normalize_frame(frame)
        if normalized.empty:
            return DecoderSnapshot(
                status="no_data",
                feature_frame=pd.DataFrame(),
                state_probabilities={},
                viterbi_state="unknown",
                transition_stability=0.0,
                state_entropy=1.0,
                noisy_channel_posteriors={"LONG": 1 / 3, "SHORT": 1 / 3, "HOLD": 1 / 3},
                audit_metadata={"version": DECODER_VERSION, "symbol": symbol, "rows": 0},
            )

        features = self._build_invariant_features(normalized, peer_frames=peer_frames, benchmark_frame=benchmark_frame)
        observation_cols = [
            "invariance_vol_norm_ret_1",
            "invariance_vol_norm_ret_5",
            "invariance_range_shape",
            "invariance_drawdown_shape_20",
            "invariance_peer_beta_residual_20",
        ]
        observations = features[[col for col in observation_cols if col in features.columns]].tail(self.config.lookback)
        hmm = self.hmm_decoder.fit_decode(observations)
        posteriors = self._noisy_channel_posteriors(features, hmm)
        sub_state_probabilities = self._sub_state_probabilities(features)
        features = self._append_decoder_features(features, hmm, posteriors, sub_state_probabilities)
        status = "active" if hmm.get("status") == "decoded" else "insufficient_or_uncertain"
        if status != "active":
            features = self._neutralize_uncertain_decoder_features(features)
        return DecoderSnapshot(
            status=status,
            feature_frame=features,
            state_probabilities=dict(hmm.get("state_probabilities", {})),
            viterbi_state=str(hmm.get("viterbi_state", "transition_choppy")),
            transition_stability=float(hmm.get("transition_stability", 0.0) or 0.0),
            state_entropy=float(hmm.get("state_entropy", 1.0) or 1.0),
            noisy_channel_posteriors=posteriors,
            audit_metadata={
                "version": DECODER_VERSION,
                "symbol": symbol,
                "rows": int(len(normalized)),
                "min_history": int(self.config.min_history),
                "lookback": int(self.config.lookback),
                "top_states": int(self.config.top_states),
                "sub_states": int(self.config.sub_states),
                "em_iterations": int(hmm.get("em_iterations", 0) or 0),
                "entropy_gate": float(self.config.entropy_gate),
                "participates_in_position_sizing": status == "active",
                "transition_matrix": hmm.get("transition_matrix", []),
                "viterbi_path_tail": hmm.get("viterbi_path", []),
                "sub_state_probabilities": sub_state_probabilities,
                "kernel_neighbors": int(self.config.kernel_neighbors),
            },
        )

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        normalized.columns = [str(col).lower() for col in normalized.columns]
        if "date" not in normalized.columns:
            if "timestamp" in normalized.columns:
                normalized["date"] = normalized["timestamp"]
            else:
                normalized["date"] = normalized.index
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in normalized.columns:
                normalized[col] = normalized["close"] if "close" in normalized.columns else 0.0
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
        normalized["open"] = normalized["open"].fillna(normalized["close"])
        normalized["high"] = normalized["high"].fillna(normalized["close"])
        normalized["low"] = normalized["low"].fillna(normalized["close"])
        normalized["volume"] = normalized["volume"].replace(0, np.nan).ffill().fillna(1.0)
        return normalized.dropna(subset=["close"]).reset_index(drop=True)

    def _build_invariant_features(
        self,
        frame: pd.DataFrame,
        peer_frames: Optional[Dict[str, pd.DataFrame]] = None,
        benchmark_frame: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        volume = frame["volume"].astype(float)
        returns_1 = close.pct_change().fillna(0.0)
        realized_vol_20 = returns_1.rolling(20, min_periods=5).std().replace(0, np.nan)
        features = pd.DataFrame(index=frame.index)
        for horizon in [1, 5, 20]:
            ret = close.pct_change(horizon).fillna(0.0)
            vol = returns_1.rolling(max(20, horizon), min_periods=5).std().replace(0, np.nan)
            features[f"invariance_vol_norm_ret_{horizon}"] = (ret / (vol * np.sqrt(max(horizon, 1)))).replace([np.inf, -np.inf], np.nan)
        features["invariance_volume_anomaly_20"] = (
            np.log1p(volume) - np.log1p(volume).rolling(20, min_periods=5).mean()
        ) / np.log1p(volume).rolling(20, min_periods=5).std().replace(0, np.nan)
        features["invariance_range_shape"] = ((high - low) / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        rolling_max = close.rolling(20, min_periods=5).max().replace(0, np.nan)
        features["invariance_drawdown_shape_20"] = (close / rolling_max - 1.0).fillna(0.0)
        peer_return = self._peer_return_mean(frame, peer_frames, benchmark_frame)
        beta = self._rolling_beta(returns_1, peer_return, 20)
        features["invariance_peer_beta_residual_20"] = (returns_1 - beta * peer_return).fillna(0.0)
        features["invariance_peer_corr_stability_20"] = returns_1.rolling(20, min_periods=5).corr(peer_return).fillna(0.0)
        features["invariance_cov_eigen_ratio_60"] = self._cov_eigen_ratio(returns_1, peer_return, 60)
        kernel = self._kernel_analog_features(features, close)
        features = pd.concat([features, kernel], axis=1)
        return features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    def _peer_return_mean(
        self,
        frame: pd.DataFrame,
        peer_frames: Optional[Dict[str, pd.DataFrame]],
        benchmark_frame: Optional[pd.DataFrame],
    ) -> pd.Series:
        candidates = []
        if peer_frames:
            for peer in peer_frames.values():
                normalized = self._normalize_frame(peer)
                if normalized.empty:
                    continue
                peer_close = normalized["close"].astype(float).reset_index(drop=True)
                peer_ret = peer_close.pct_change().reindex(frame.index).fillna(0.0)
                candidates.append(peer_ret)
        if benchmark_frame is not None and not benchmark_frame.empty:
            benchmark = self._normalize_frame(benchmark_frame)
            if not benchmark.empty:
                candidates.append(benchmark["close"].astype(float).pct_change().reindex(frame.index).fillna(0.0))
        if not candidates:
            return frame["close"].astype(float).pct_change().fillna(0.0)
        return pd.concat(candidates, axis=1).mean(axis=1).fillna(0.0)

    @staticmethod
    def _rolling_beta(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
        cov = left.rolling(window, min_periods=5).cov(right)
        var = right.rolling(window, min_periods=5).var().replace(0, np.nan)
        return (cov / var).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _cov_eigen_ratio(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
        values = []
        for idx in range(len(left)):
            start = max(0, idx - window + 1)
            pair = pd.concat([left.iloc[start : idx + 1], right.iloc[start : idx + 1]], axis=1).dropna()
            if len(pair) < 10:
                values.append(0.0)
                continue
            cov = np.cov(pair.to_numpy(dtype=float).T)
            eig = np.linalg.eigvalsh(cov)
            denom = float(np.sum(np.abs(eig)))
            values.append(float(np.max(np.abs(eig)) / denom) if denom else 0.0)
        return pd.Series(values, index=left.index)

    def _kernel_analog_features(self, features: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        cols = [
            "invariance_vol_norm_ret_1",
            "invariance_vol_norm_ret_5",
            "invariance_range_shape",
            "invariance_drawdown_shape_20",
            "invariance_peer_beta_residual_20",
        ]
        matrix = features[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        forward = close.shift(-self.config.target_horizon) / close - 1.0
        out = pd.DataFrame(index=features.index)
        means = []
        hit_rates = []
        payoffs = []
        for idx in range(len(matrix)):
            max_train = idx - self.config.target_horizon
            if max_train < self.config.kernel_neighbors:
                means.append(0.0)
                hit_rates.append(0.0)
                payoffs.append(0.0)
                continue
            history = matrix.iloc[:max_train]
            labels = forward.iloc[:max_train]
            current = matrix.iloc[idx]
            scale = history.std().replace(0, 1.0).fillna(1.0)
            distances = (((history - current) / scale) ** 2).sum(axis=1)
            neighbors = distances.nsmallest(min(self.config.kernel_neighbors, len(distances))).index
            neighbor_returns = labels.loc[neighbors].dropna()
            if neighbor_returns.empty:
                means.append(0.0)
                hit_rates.append(0.0)
                payoffs.append(0.0)
                continue
            wins = neighbor_returns[neighbor_returns > 0]
            losses = neighbor_returns[neighbor_returns < 0]
            means.append(float(neighbor_returns.mean()))
            hit_rates.append(float((neighbor_returns > 0).mean()))
            payoffs.append(float(wins.mean() / abs(losses.mean())) if not wins.empty and not losses.empty else 0.0)
        out["kernel_analog_forward_mean"] = means
        out["kernel_analog_hit_rate"] = hit_rates
        out["kernel_analog_payoff_ratio"] = payoffs
        return out

    @staticmethod
    def _noisy_channel_posteriors(features: pd.DataFrame, hmm: Dict[str, Any]) -> Dict[str, float]:
        if features.empty:
            return {"LONG": 1 / 3, "SHORT": 1 / 3, "HOLD": 1 / 3}
        latest = features.iloc[-1]
        risk_on = float(hmm.get("state_probabilities", {}).get("risk_on", 0.0) or 0.0)
        risk_off = float(hmm.get("state_probabilities", {}).get("risk_off", 0.0) or 0.0)
        analog = float(latest.get("kernel_analog_forward_mean", 0.0) or 0.0)
        hit_rate = float(latest.get("kernel_analog_hit_rate", 0.0) or 0.0)
        norm_ret = float(latest.get("invariance_vol_norm_ret_5", 0.0) or 0.0)
        long_logit = 0.35 + 1.25 * risk_on + 12.0 * max(analog, 0.0) + 0.35 * max(norm_ret, 0.0) + 0.75 * max(hit_rate - 0.5, 0.0)
        short_logit = 0.35 + 1.25 * risk_off + 12.0 * max(-analog, 0.0) + 0.35 * max(-norm_ret, 0.0) + 0.75 * max(0.5 - hit_rate, 0.0)
        hold_logit = 0.55 + 0.85 * float(hmm.get("state_entropy", 1.0) or 1.0)
        logits = np.array([long_logit, short_logit, hold_logit], dtype=float)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        return {
            "LONG": round(float(probs[0]), 6),
            "SHORT": round(float(probs[1]), 6),
            "HOLD": round(float(probs[2]), 6),
        }

    @staticmethod
    def _sub_state_probabilities(features: pd.DataFrame) -> Dict[str, float]:
        if features.empty:
            return {"trend": 1 / 3, "mean_reversion": 1 / 3, "liquidity_stress": 1 / 3}
        latest = features.iloc[-1]
        norm_ret = abs(float(latest.get("invariance_vol_norm_ret_5", 0.0) or 0.0))
        drawdown = abs(float(latest.get("invariance_drawdown_shape_20", 0.0) or 0.0))
        volume_anomaly = abs(float(latest.get("invariance_volume_anomaly_20", 0.0) or 0.0))
        range_shape = abs(float(latest.get("invariance_range_shape", 0.0) or 0.0))
        logits = np.array(
            [
                0.35 + 0.55 * norm_ret,
                0.35 + 0.65 * max(0.0, 1.0 - norm_ret),
                0.20 + 0.45 * volume_anomaly + 10.0 * range_shape + 1.5 * drawdown,
            ],
            dtype=float,
        )
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        return {
            "trend": round(float(probs[0]), 6),
            "mean_reversion": round(float(probs[1]), 6),
            "liquidity_stress": round(float(probs[2]), 6),
        }

    @staticmethod
    def _append_decoder_features(
        features: pd.DataFrame,
        hmm: Dict[str, Any],
        posteriors: Dict[str, float],
        sub_state_probabilities: Dict[str, float],
    ) -> pd.DataFrame:
        enriched = features.copy()
        probs = hmm.get("state_probabilities", {})
        enriched["hmm_prob_risk_on"] = float(probs.get("risk_on", 0.0) or 0.0)
        enriched["hmm_prob_risk_off"] = float(probs.get("risk_off", 0.0) or 0.0)
        enriched["hmm_prob_transition_choppy"] = float(probs.get("transition_choppy", 0.0) or 0.0)
        enriched["hmm_state_entropy"] = float(hmm.get("state_entropy", 1.0) or 1.0)
        enriched["hmm_transition_stability"] = float(hmm.get("transition_stability", 0.0) or 0.0)
        enriched["noisy_channel_long_posterior"] = float(posteriors.get("LONG", 1 / 3))
        enriched["noisy_channel_short_posterior"] = float(posteriors.get("SHORT", 1 / 3))
        enriched["noisy_channel_hold_posterior"] = float(posteriors.get("HOLD", 1 / 3))
        enriched["hmm_sub_prob_trend"] = float(sub_state_probabilities.get("trend", 1 / 3))
        enriched["hmm_sub_prob_mean_reversion"] = float(sub_state_probabilities.get("mean_reversion", 1 / 3))
        enriched["hmm_sub_prob_liquidity_stress"] = float(sub_state_probabilities.get("liquidity_stress", 1 / 3))
        return enriched

    @staticmethod
    def _neutralize_uncertain_decoder_features(features: pd.DataFrame) -> pd.DataFrame:
        neutral = features.copy()
        for col in [
            "hmm_prob_risk_on",
            "hmm_prob_risk_off",
            "hmm_prob_transition_choppy",
            "noisy_channel_long_posterior",
            "noisy_channel_short_posterior",
            "noisy_channel_hold_posterior",
            "hmm_sub_prob_trend",
            "hmm_sub_prob_mean_reversion",
            "hmm_sub_prob_liquidity_stress",
        ]:
            neutral[col] = 1.0 / 3.0
        neutral["hmm_state_entropy"] = 1.0
        neutral["hmm_transition_stability"] = 0.0
        return neutral
