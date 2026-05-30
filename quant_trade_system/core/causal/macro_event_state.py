"""Macro and event state overlays for causal quant sizing.

The engine converts a small set of auditable macro/event observations into
factor-weight overlays and tail-risk controls.  It deliberately does not fetch
data by itself; callers pass timestamps, market signals or event probabilities
from their data layer so every nightly decision can be traced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if np.isfinite(value):
            return float(value)
        return default
    text = str(value).replace(",", "").replace("%", "").strip()
    if text in {"", "-", "--", "None", "nan"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _normalize_rate(value: Any, default: float = 0.0) -> float:
    """Normalize common yield encodings into decimals.

    Examples: 0.05, 5.0 and 50.0 all become roughly 5%.
    """
    raw = _to_float(value, default)
    if abs(raw) > 20.0:
        return raw / 1000.0
    if abs(raw) > 1.0:
        return raw / 100.0
    return raw


def _normalize_change(value: Any, default: float = 0.0) -> float:
    """Normalize rate changes into decimals.

    Examples: 25 bps -> 0.0025, 0.25 percentage points -> 0.0025.
    """
    raw = _to_float(value, default)
    if abs(raw) > 1.0:
        return raw / 10000.0
    if abs(raw) > 0.10:
        return raw / 100.0
    return raw


@dataclass(frozen=True)
class MacroEventStateConfig:
    us_yield_tail_threshold: float = 0.05
    sofr_fast_change_threshold: float = 0.0025
    move_low_threshold: float = 90.0
    straddle_activity_threshold: float = 0.65
    csi1000_500_excess_threshold: float = 0.02
    hormuz_reopen_low_probability: float = 0.60
    hormuz_reopen_high_probability: float = 0.70


@dataclass
class MacroEventState:
    status: str
    tail_risk_score: float
    tail_hedge_multiplier: float
    factor_weight_overlays: Dict[str, float]
    regime_flags: Dict[str, bool]
    alerts: list[str] = field(default_factory=list)
    event_scenarios: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fx_implications: Dict[str, Any] = field(default_factory=dict)
    observations: Dict[str, Any] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MacroEventStateEngine:
    """Build macro/event overlays for factor weights and risk controls."""

    def __init__(self, config: Optional[MacroEventStateConfig] = None) -> None:
        self.config = config or MacroEventStateConfig()

    def analyze(self, market_context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        context = market_context or {}
        asset_signals = context.get("asset_signals", {}) if isinstance(context.get("asset_signals", {}), Mapping) else {}

        us10y = self._context_rate(context, asset_signals, ["us10y_yield", "US10Y", "TNX", "10y_yield"])
        us30y = self._context_rate(context, asset_signals, ["us30y_yield", "US30Y", "TYX", "30y_yield"])
        sofr_5d_change = self._context_change(context, asset_signals, ["sofr_5d_change", "SOFR_5D_CHANGE", "sofr_change_5d"])
        move_level = self._context_value(context, asset_signals, ["move_index", "MOVE", "move"], default=0.0)
        straddle_activity = self._context_value(
            context,
            asset_signals,
            ["bond_straddle_activity", "move_straddle_activity", "straddle_activity"],
            default=0.0,
        )
        dxy_return = self._context_value(context, asset_signals, ["dxy_return_5d", "DXY", "dxy"], value_keys=("return", "momentum", "change"), default=0.0)
        csi_excess = self._csi1000_500_excess(context, asset_signals)
        ai_small_cap_momentum = self._context_value(
            context,
            asset_signals,
            ["ai_small_cap_momentum", "AI_small_caps", "ai_industrial_chain_small_caps", "CSI1000"],
            value_keys=("return", "momentum", "change"),
            default=0.0,
        )

        event_probabilities = context.get("event_probabilities", {}) if isinstance(context.get("event_probabilities", {}), Mapping) else {}
        hormuz_probability = _to_float(
            context.get("hormuz_reopen_probability", event_probabilities.get("hormuz_reopen_probability")),
            0.0,
        )
        hormuz_signed = bool(context.get("hormuz_reopen_signed", event_probabilities.get("hormuz_reopen_signed", False)))

        yield_5pct_break = max(us10y, us30y) >= self.config.us_yield_tail_threshold
        sofr_fast_change = abs(sofr_5d_change) >= self.config.sofr_fast_change_threshold
        move_straddle_warning = (
            move_level > 0
            and move_level <= self.config.move_low_threshold
            and straddle_activity >= self.config.straddle_activity_threshold
        )
        csi1000_leadership = (
            csi_excess >= self.config.csi1000_500_excess_threshold
            and ai_small_cap_momentum >= 0
        )
        hawkish_fed_usd_weakness = (sofr_5d_change > 0 or yield_5pct_break) and dxy_return < -0.005
        hormuz_pending_reopen = (
            self.config.hormuz_reopen_low_probability
            <= hormuz_probability
            <= self.config.hormuz_reopen_high_probability
            and not hormuz_signed
        )

        alerts: list[str] = []
        tail_components = [0.10]
        if yield_5pct_break:
            alerts.append("US yield at/above 5%: monitor global risk-asset repricing and A-share independence narrative.")
            tail_components.append(0.45)
        if sofr_fast_change:
            alerts.append("SOFR changed quickly: switch macro regime weights between rate-sensitive and earnings-driven factors.")
            tail_components.append(0.28)
        if move_straddle_warning:
            alerts.append("MOVE is low but bond straddle activity is elevated: treat as a bond-volatility regime early warning.")
            tail_components.append(0.30)
        if hormuz_pending_reopen:
            alerts.append("Hormuz reopening probability is in the 60-70% negotiation band: prefer volatility exposure before signed confirmation.")
            tail_components.append(0.24)
        if hawkish_fed_usd_weakness:
            alerts.append("Fed/rate pressure is rising while USD weakens: do not apply the usual hike->USD up->CNY down rule mechanically.")
            tail_components.append(0.22)

        tail_risk_score = _clip(max(tail_components))
        if len(tail_components) > 2:
            tail_risk_score = _clip(tail_risk_score + 0.05 * (len(tail_components) - 2), 0.0, 0.80)

        rate_sensitive_multiplier = 1.0
        if yield_5pct_break:
            rate_sensitive_multiplier *= 0.70
        if sofr_fast_change:
            rate_sensitive_multiplier *= 0.85
        if move_straddle_warning:
            rate_sensitive_multiplier *= 0.92

        ai_small_cap_multiplier = 1.0
        if csi1000_leadership:
            ai_small_cap_multiplier = 1.0 + min(0.35, max(0.10, csi_excess * 4.0))

        earnings_driven_multiplier = 1.0
        if csi1000_leadership and not yield_5pct_break:
            earnings_driven_multiplier = 1.12
        elif yield_5pct_break:
            earnings_driven_multiplier = 0.92

        volatility_multiplier = 1.0
        if move_straddle_warning:
            volatility_multiplier *= 1.20
        if hormuz_pending_reopen:
            volatility_multiplier *= 1.18

        factor_weight_overlays = {
            "ai_small_cap_momentum_multiplier": round(ai_small_cap_multiplier, 6),
            "rate_sensitive_multiplier": round(float(np.clip(rate_sensitive_multiplier, 0.40, 1.20)), 6),
            "earnings_driven_multiplier": round(float(np.clip(earnings_driven_multiplier, 0.70, 1.25)), 6),
            "volatility_multiplier": round(float(np.clip(volatility_multiplier, 1.0, 1.50)), 6),
            "fx_cny_resilience_multiplier": 1.12 if hawkish_fed_usd_weakness else 1.0,
        }

        event_scenarios: Dict[str, Dict[str, Any]] = {}
        if hormuz_probability > 0:
            event_scenarios["hormuz_reopen"] = {
                "probability": round(_clip(hormuz_probability), 6),
                "signed": hormuz_signed,
                "pre_confirmation_bias": "long_volatility_not_direction" if not hormuz_signed else "inactive",
                "post_confirmation_bias": "directional_energy_risk_premium_compression" if hormuz_signed else "wait_for_signed_confirmation",
                "positioning_rule": "volatility_first_then_direction_after_signature",
            }

        fx_implications = {
            "regime": "hawkish_fed_usd_weakness" if hawkish_fed_usd_weakness else "standard_or_unconfirmed",
            "cny_readthrough": "CNY resilience / less imported depreciation pressure" if hawkish_fed_usd_weakness else "requires DXY and USDCNY confirmation",
        }

        state = MacroEventState(
            status="risk_watch" if alerts else "normal",
            tail_risk_score=round(tail_risk_score, 6),
            tail_hedge_multiplier=round(float(np.clip(1.0 + tail_risk_score, 1.0, 1.80)), 6),
            factor_weight_overlays=factor_weight_overlays,
            regime_flags={
                "us_yield_5pct_break": yield_5pct_break,
                "sofr_fast_change": sofr_fast_change,
                "move_straddle_warning": move_straddle_warning,
                "csi1000_ai_small_cap_leadership": csi1000_leadership,
                "hawkish_fed_usd_weakness": hawkish_fed_usd_weakness,
                "hormuz_pending_reopen": hormuz_pending_reopen,
                "hormuz_signed_reopen": hormuz_signed,
            },
            alerts=alerts,
            event_scenarios=event_scenarios,
            fx_implications=fx_implications,
            observations={
                "us10y_yield": round(us10y, 6),
                "us30y_yield": round(us30y, 6),
                "sofr_5d_change": round(sofr_5d_change, 6),
                "move_index": round(move_level, 6),
                "bond_straddle_activity": round(straddle_activity, 6),
                "dxy_return_5d": round(dxy_return, 6),
                "csi1000_500_excess_return": round(csi_excess, 6),
                "ai_small_cap_momentum": round(ai_small_cap_momentum, 6),
            },
            audit={
                "inputs": "market_context.asset_signals + explicit macro/event fields",
                "rate_normalization": "0.05, 5.0 and 50.0 are normalized to about 5%",
                "no_position_override": "overlays only adjust validated factor weights and tail-risk controls",
            },
        )
        return state.to_dict()

    def _context_rate(self, context: Mapping[str, Any], asset_signals: Mapping[str, Any], names: list[str]) -> float:
        return _normalize_rate(self._context_value(context, asset_signals, names, value_keys=("value", "level", "close"), default=0.0))

    def _context_change(self, context: Mapping[str, Any], asset_signals: Mapping[str, Any], names: list[str]) -> float:
        return _normalize_change(self._context_value(context, asset_signals, names, value_keys=("change", "return", "momentum", "value"), default=0.0))

    @staticmethod
    def _context_value(
        context: Mapping[str, Any],
        asset_signals: Mapping[str, Any],
        names: list[str],
        value_keys: tuple[str, ...] = ("value", "return", "momentum", "change", "close", "level"),
        default: float = 0.0,
    ) -> float:
        for name in names:
            if name in context:
                value = context[name]
                if isinstance(value, Mapping):
                    for key in value_keys:
                        if key in value:
                            return _to_float(value.get(key), default)
                return _to_float(value, default)
            signal = asset_signals.get(name)
            if isinstance(signal, Mapping):
                for key in value_keys:
                    if key in signal:
                        return _to_float(signal.get(key), default)
        return default

    def _csi1000_500_excess(self, context: Mapping[str, Any], asset_signals: Mapping[str, Any]) -> float:
        explicit = self._context_value(
            context,
            asset_signals,
            ["csi1000_500_excess_return", "CSI1000_500_EXCESS", "IM_IC_EXCESS"],
            default=np.nan,
        )
        if np.isfinite(explicit):
            return explicit
        csi1000 = self._context_value(context, asset_signals, ["CSI1000", "IM", "IM0"], value_keys=("return", "momentum", "change"), default=0.0)
        csi500 = self._context_value(context, asset_signals, ["CSI500", "IC", "IC0"], value_keys=("return", "momentum", "change"), default=0.0)
        return csi1000 - csi500
