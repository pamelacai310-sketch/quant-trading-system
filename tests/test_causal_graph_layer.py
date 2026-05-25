from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_trade_system.core.causal import (
    ALLOW,
    OBSERVE_ONLY,
    CausalAbstentionGate,
    CausalGraphLayer,
    CausalLLMAuditor,
    CausalValidationLoop,
    InstrumentRegistry,
)


class CausalGraphLayerTests(unittest.TestCase):
    def test_discovery_generates_pc_fci_pcmci_candidate_edges(self) -> None:
        idx = pd.RangeIndex(120)
        driver = pd.Series(np.linspace(-2.0, 2.0, len(idx)), index=idx)
        target = driver.shift(1).fillna(0.0) * 0.8
        factors = pd.DataFrame(
            {
                "causal_driver": driver,
                "noise_factor": np.sin(np.linspace(0, 30, len(idx))),
            },
            index=idx,
        )
        layer = CausalGraphLayer(min_observations=40)

        edges = layer.discover_candidate_edges(factors, target)

        self.assertIn("causal_driver", edges)
        self.assertIn("pcmci", edges["causal_driver"].algorithms)
        self.assertGreater(edges["causal_driver"].confidence, 0.1)

    def test_backdoor_adjustment_builds_controls_and_validation_diagnostics(self) -> None:
        idx = pd.RangeIndex(120)
        benchmark = pd.Series(np.linspace(-1.0, 1.0, len(idx)), index=idx)
        idiosyncratic = pd.Series(np.sin(np.linspace(0, 8, len(idx))), index=idx)
        source = benchmark * 0.6 + idiosyncratic
        target = source * 0.5 + benchmark * 0.2
        factors = pd.DataFrame(
            {
                "alpha_source": source,
                "base_ret_20": benchmark,
                "hmm_state_entropy": np.linspace(0.2, 0.8, len(idx)),
            },
            index=idx,
        )
        layer = CausalGraphLayer(min_observations=40)
        adjustment = layer.backdoor_adjustment(
            "alpha_source",
            "forward_return",
            factors,
            benchmark_returns=benchmark,
        )
        controls = layer.adjustment_frame(adjustment, factors, benchmark_returns=benchmark)
        validation = CausalValidationLoop(min_observations=40).validate_feature(
            "alpha_source",
            factors["alpha_source"],
            target,
            benchmark_returns=benchmark,
            adjustment_frame=controls,
            backdoor_adjustment=adjustment.__dict__,
            discovery_support=0.8,
        )

        self.assertIn("benchmark_return", adjustment.adjustment_variables)
        self.assertGreaterEqual(adjustment.adjustment_quality, 0.55)
        self.assertIn("backdoor_adjustment", validation.diagnostics)
        self.assertTrue(validation.can_trade)

    def test_counterfactual_stress_outputs_tail_hedge_multiplier(self) -> None:
        idx = pd.RangeIndex(100)
        source = pd.Series(np.linspace(-1.0, 1.0, len(idx)), index=idx)
        target = source.shift(1).fillna(0.0) * -0.4
        factors = pd.DataFrame({"risk_driver": source}, index=idx)
        layer = CausalGraphLayer(min_observations=40)
        edges = layer.discover_candidate_edges(factors, target)

        stress = layer.counterfactual_stress_test(edges.values(), factors, target)

        self.assertGreaterEqual(stress.tail_risk_score, 0.0)
        self.assertGreaterEqual(stress.tail_hedge_multiplier, 1.0)
        self.assertTrue(stress.affected_paths)

    def test_causal_abstention_gate_allows_and_abstains(self) -> None:
        gate = CausalAbstentionGate()

        allow = gate.evaluate(
            identification_status="identifiable",
            backdoor_quality=0.75,
            hmm_state_entropy=0.20,
            data_validation_passed=True,
            model_disagreement=0.10,
            counterfactual_tail_risk=0.05,
            instrument_status="valid",
        )
        observe = gate.evaluate(
            identification_status="correlation_only",
            backdoor_quality=0.45,
            hmm_state_entropy=0.40,
            data_validation_passed=True,
            model_disagreement=0.20,
            counterfactual_tail_risk=0.10,
            instrument_status="weak",
        )

        self.assertEqual(allow.decision, ALLOW)
        self.assertGreater(allow.weight_multiplier, 0.0)
        self.assertEqual(observe.decision, OBSERVE_ONLY)
        self.assertEqual(observe.weight_multiplier, 0.0)

    def test_instrument_registry_diagnoses_valid_and_weak_instruments(self) -> None:
        idx = pd.RangeIndex(120)
        iv = pd.Series(np.linspace(-1.0, 1.0, len(idx)), index=idx)
        source = iv * 0.8 + np.sin(np.linspace(0, 4, len(idx))) * 0.01
        target = source * 0.6 + np.cos(np.linspace(0, 5, len(idx))) * 0.01
        factors = pd.DataFrame(
            {
                "alpha_source": source,
                "iv_policy_surprise": iv,
                "iv_noise": np.random.default_rng(7).normal(0, 1, len(idx)),
            },
            index=idx,
        )
        registry = InstrumentRegistry()

        record = registry.diagnose_edge("alpha_source", "forward_return", factors, target)

        self.assertEqual(record.validity_status, "valid")
        self.assertEqual(record.instruments, ["iv_policy_surprise"])
        self.assertGreater(record.first_stage_strength, 0.15)

    def test_causal_llm_auditor_keeps_hypotheses_audit_only(self) -> None:
        auditor = CausalLLMAuditor()

        records = auditor.audit_hypotheses(
            [
                {
                    "treatment": "OPEC production cut",
                    "outcome": "SC0 forward return",
                    "controls": ["DXY", "global demand"],
                    "instrument": "policy_surprise",
                    "expected_path": "production cut -> inventory draw -> oil price up",
                    "source": "test_llm",
                }
            ]
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].actionability, "audit_only")
        self.assertIn("backdoor_or_iv_validation", records[0].validation_required)

        news_records = auditor.audit_hypotheses(
            [{"headline": "Fed policy shock raises real rates", "source": "policy_calendar"}]
        )
        self.assertEqual(news_records[0].outcome, "market_forward_return")
        self.assertEqual(news_records[0].actionability, "audit_only")


if __name__ == "__main__":
    unittest.main()
