"""Unified SCM/DAG layer for causal factor mining and risk control.

This module is deliberately lightweight. It provides auditable causal graph
contracts for the self-iterating trading pipeline without depending on heavy
PC/FCI/PCMCI packages. The discovery routines are conservative approximations:
they generate candidate edges and diagnostics, while trading eligibility still
depends on downstream validation and backdoor adjustment gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass
class CausalDAGEdge:
    source: str
    target: str
    algorithms: List[str]
    strength: float
    confidence: float
    orientation: str
    latent_confounding_risk: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackdoorAdjustmentSet:
    source: str
    target: str
    adjustment_variables: List[str]
    adjustment_quality: float
    identification_status: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterfactualStressResult:
    scenario_name: str
    intervention: Dict[str, float]
    expected_portfolio_impact: float
    tail_risk_score: float
    tail_hedge_multiplier: float
    affected_paths: List[Dict[str, Any]]


@dataclass
class SCMSnapshot:
    target: str
    nodes: List[str]
    edges: List[CausalDAGEdge]
    confounders: List[str]
    counterfactual_stress: CounterfactualStressResult
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "nodes": list(self.nodes),
            "edges": [asdict(edge) for edge in self.edges],
            "confounders": list(self.confounders),
            "counterfactual_stress": asdict(self.counterfactual_stress),
            "metadata": dict(self.metadata),
        }


class CausalGraphLayer:
    """SCM/DAG, backdoor adjustment and counterfactual stress helper."""

    DEFAULT_CONFOUNDER_TOKENS = (
        "benchmark",
        "market",
        "macro",
        "liquidity",
        "vol",
        "volatility",
        "risk",
        "hmm_",
        "sde_",
        "kernel_analog_tail",
        "global_peer",
        "base_ret_20",
        "base_ret_60",
    )

    def __init__(self, min_observations: int = 60, max_candidate_edges: int = 25) -> None:
        self.min_observations = int(min_observations)
        self.max_candidate_edges = int(max_candidate_edges)

    def discover_candidate_edges(
        self,
        factor_matrix: pd.DataFrame,
        target_returns: pd.Series,
        algorithms: Iterable[str] = ("pc", "fci", "pcmci"),
    ) -> Dict[str, CausalDAGEdge]:
        target = pd.to_numeric(target_returns, errors="coerce").rename("forward_return")
        edges: Dict[str, CausalDAGEdge] = {}
        for factor_name in factor_matrix.columns:
            source = pd.to_numeric(factor_matrix[factor_name], errors="coerce").rename("source")
            aligned = pd.concat([source, target], axis=1).dropna()
            if len(aligned) < self.min_observations or float(aligned["source"].std()) < 1e-10:
                continue
            support: Dict[str, float] = {}
            if "pc" in algorithms:
                support["pc"] = abs(self._safe_corr(aligned["source"], aligned["forward_return"]))
            if "fci" in algorithms:
                support["fci"] = self._fci_support(aligned["source"], aligned["forward_return"])
            if "pcmci" in algorithms:
                support["pcmci"] = self._pcmci_support(aligned["source"], aligned["forward_return"])
            if not support:
                continue
            confidence = float(np.clip(np.mean(list(support.values())), 0.0, 1.0))
            if confidence <= 0.03:
                continue
            orientation = "source_to_target" if self._pcmci_support(aligned["source"], aligned["forward_return"]) >= support.get("pc", 0.0) * 0.50 else "undirected_candidate"
            edges[factor_name] = CausalDAGEdge(
                source=factor_name,
                target="forward_return",
                algorithms=[name for name, score in support.items() if score > 0.03],
                strength=round(float(max(support.values())), 6),
                confidence=round(confidence, 6),
                orientation=orientation,
                latent_confounding_risk=round(max(0.0, support.get("pc", 0.0) - support.get("fci", 0.0)), 6),
                diagnostics={name: round(float(score), 6) for name, score in support.items()},
            )
        return dict(sorted(edges.items(), key=lambda item: item[1].confidence, reverse=True)[: self.max_candidate_edges])

    def build_scm_snapshot(
        self,
        factor_matrix: pd.DataFrame,
        target_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> SCMSnapshot:
        candidate_edges = self.discover_candidate_edges(factor_matrix, target_returns)
        confounders = self._infer_confounders(factor_matrix.columns)
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            confounders = ["benchmark_return", *confounders]
        stress = self.counterfactual_stress_test(
            candidate_edges.values(),
            factor_matrix=factor_matrix,
            target_returns=target_returns,
        )
        return SCMSnapshot(
            target="forward_return",
            nodes=sorted(set(factor_matrix.columns).union({"forward_return"})),
            edges=list(candidate_edges.values()),
            confounders=confounders,
            counterfactual_stress=stress,
            metadata={
                "candidate_edge_count": len(candidate_edges),
                "discovery_algorithms": ["pc", "fci", "pcmci"],
                "graph_type": "scm_dag_candidate_layer",
                "trading_gate": "candidate_edges_require_backdoor_adjusted_validation",
            },
        )

    def backdoor_adjustment(
        self,
        source: str,
        target: str,
        factor_matrix: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> BackdoorAdjustmentSet:
        candidates = [name for name in self._infer_confounders(factor_matrix.columns) if name != source]
        correlations = []
        source_series = pd.to_numeric(factor_matrix[source], errors="coerce") if source in factor_matrix else pd.Series(dtype=float)
        for name in candidates:
            corr = abs(self._safe_corr(source_series, pd.to_numeric(factor_matrix[name], errors="coerce")))
            if corr >= 0.05:
                correlations.append((name, corr))
        correlations.sort(key=lambda item: item[1], reverse=True)
        selected = [name for name, _ in correlations[:4]]
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            selected = ["benchmark_return", *selected]
        quality = float(np.clip(0.45 + 0.12 * len(selected), 0.0, 1.0)) if selected else 0.35
        status = "adjusted_identifiable" if quality >= 0.55 else "weak_or_unadjusted"
        return BackdoorAdjustmentSet(
            source=source,
            target=target,
            adjustment_variables=selected,
            adjustment_quality=round(quality, 6),
            identification_status=status,
            diagnostics={
                "candidate_confounders": candidates[:12],
                "selection_rule": "token_inferred_confounders_correlated_with_source_plus_benchmark",
            },
        )

    def adjustment_frame(
        self,
        adjustment: BackdoorAdjustmentSet,
        factor_matrix: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        columns: Dict[str, pd.Series] = {}
        for variable in adjustment.adjustment_variables:
            if variable == "benchmark_return" and benchmark_returns is not None:
                columns[variable] = pd.Series(benchmark_returns).reset_index(drop=True)
            elif variable in factor_matrix.columns:
                columns[variable] = pd.to_numeric(factor_matrix[variable], errors="coerce").reset_index(drop=True)
        if not columns:
            return pd.DataFrame(index=factor_matrix.index)
        frame = pd.DataFrame(columns)
        frame.index = factor_matrix.index[: len(frame)]
        return frame.reindex(factor_matrix.index).ffill().fillna(0.0)

    def counterfactual_stress_test(
        self,
        edges: Iterable[CausalDAGEdge],
        factor_matrix: pd.DataFrame,
        target_returns: pd.Series,
        scenario_name: str = "generic_causal_tail_shock",
        intervention_scale: float = 2.0,
    ) -> CounterfactualStressResult:
        affected_paths: List[Dict[str, Any]] = []
        expected_impact = 0.0
        target_vol = float(pd.to_numeric(target_returns, errors="coerce").std() or 0.0)
        for edge in edges:
            if edge.source not in factor_matrix:
                continue
            source = pd.to_numeric(factor_matrix[edge.source], errors="coerce")
            source_vol = float(source.std() or 0.0)
            if source_vol <= 0:
                continue
            shock = intervention_scale * source_vol
            impact = float(edge.strength) * shock * np.sign(self._safe_corr(source, target_returns))
            expected_impact += impact
            affected_paths.append(
                {
                    "path": [edge.source, edge.target],
                    "algorithm_support": edge.algorithms,
                    "edge_confidence": edge.confidence,
                    "counterfactual_impact": round(impact, 6),
                }
            )
        normalized_impact = expected_impact / max(target_vol, 1e-6)
        tail_risk = float(np.clip(abs(normalized_impact) * 0.18 + max(0.0, -normalized_impact) * 0.10, 0.0, 0.80))
        return CounterfactualStressResult(
            scenario_name=scenario_name,
            intervention={"shock_scale_std": intervention_scale},
            expected_portfolio_impact=round(float(expected_impact), 6),
            tail_risk_score=round(tail_risk, 6),
            tail_hedge_multiplier=round(float(np.clip(1.0 + tail_risk, 1.0, 1.80)), 6),
            affected_paths=affected_paths[:10],
        )

    def discovery_support_for(self, edge_map: Mapping[str, CausalDAGEdge], factor_name: str) -> float:
        edge = edge_map.get(factor_name)
        return float(edge.confidence) if edge else 0.0

    def _infer_confounders(self, columns: Iterable[str]) -> List[str]:
        out = []
        for column in columns:
            lower = str(column).lower()
            if any(token in lower for token in self.DEFAULT_CONFOUNDER_TOKENS):
                out.append(str(column))
        return out[:12]

    @staticmethod
    def _safe_corr(left: pd.Series, right: pd.Series) -> float:
        aligned = pd.concat(
            [
                pd.to_numeric(left, errors="coerce").rename("left"),
                pd.to_numeric(right, errors="coerce").rename("right"),
            ],
            axis=1,
        ).dropna()
        if len(aligned) < 5 or float(aligned["left"].std()) < 1e-10 or float(aligned["right"].std()) < 1e-10:
            return 0.0
        corr = float(np.corrcoef(aligned["left"], aligned["right"])[0, 1])
        return corr if np.isfinite(corr) else 0.0

    @classmethod
    def _pcmci_support(cls, source: pd.Series, target: pd.Series) -> float:
        lag_scores = []
        for lag in [1, 2, 3, 5]:
            lag_scores.append(abs(cls._safe_corr(source.shift(lag), target)))
        return float(max(lag_scores or [0.0]))

    @classmethod
    def _fci_support(cls, source: pd.Series, target: pd.Series) -> float:
        raw = abs(cls._safe_corr(source, target))
        source_ret = source.diff().fillna(0.0)
        target_ret = target.diff().fillna(0.0)
        differenced = abs(cls._safe_corr(source_ret, target_ret))
        return float(min(raw, 0.65 * raw + 0.35 * differenced))
