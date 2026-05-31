"""
自迭代因果AI训练引擎

把“规则打分”升级为：
1. 宽因子与因果量化因子联合生成
2. 具备金融含义约束的自动特征筛选
3. 以胜率/赔率/弹性为核心的目标函数训练
4. 把组合约束与塔勒布杠铃一并纳入联合优化

说明：
- 这里实现的是“Renaissance 风格”的宽因子、正交化、加权聚合思路，
  不是对任何机构私有算法的复刻。
- 全部实现都保持可解释、可追踪、可计算。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...factors.factor_library import FactorLibrary
from ..robustness import democratic_orthogonalize, effective_breadth, shapley_deployment_policy
from .causal_factor_library import AssetClass, CausalFactorLibrary
from .causal_graph_layer import CausalDAGEdge, CausalGraphLayer
from .invariance_market_decoder import InvarianceMarketDecoder, InvariantDecoderConfig
from .research_governance import (
    ALLOW,
    IDENTIFIABLE,
    NO_TRADE,
    OBSERVE_ONLY,
    REDUCE,
    WEAK_IDENTIFIABLE,
    CausalAbstentionGate,
    CausalLLMAuditor,
    CausalValidationLoop,
    ExperimentRegistry,
    FeatureStore,
    InstrumentRegistry,
    ModelRegistry,
)


@dataclass
class FeatureSelectionPolicy:
    min_rs_score: float = 70.0
    min_r_squared: float = 0.70
    min_history: int = 60
    min_non_null_ratio: float = 0.70
    max_selected_features: int = 12
    target_horizon: int = 5
    signal_threshold: float = 0.25
    min_discovery_support: float = 0.03
    discovery_support_weight: float = 0.20


@dataclass
class LearningObjectiveConfig:
    win_rate_weight: float = 0.45
    payoff_weight: float = 0.35
    elasticity_weight: float = 0.20
    state_conditioning_strength: float = 0.35
    cross_asset_transfer_weight: float = 0.15


@dataclass
class PortfolioConstraintConfig:
    max_positions: int = 5
    max_single_weight: float = 0.20
    max_futures_weight: float = 0.50
    max_gross_weight: float = 1.00
    base_stop_loss_pct: float = 0.03
    base_take_profit_pct: float = 0.06
    max_hold_days: int = 5
    no_weekend_hold: bool = True
    max_fractional_kelly: float = 0.25
    default_portfolio_notional: float = 1_000_000.0
    default_max_participation_rate: float = 0.10


@dataclass
class SelectedFeature:
    factor_name: str
    financial_meaning: str
    formula: str
    rs_score: float
    r_squared: float
    correlation: float
    slope: float
    direction: int
    selected: bool
    objective_score: float = 0.0
    identification_status: str = "unvalidated"
    validation_score: float = 0.0
    can_trade: bool = False
    rejection_reason: Optional[str] = None
    causal_discovery_support: float = 0.0
    backdoor_adjustment_quality: float = 0.0
    scm_edge_algorithms: List[str] = field(default_factory=list)
    instrument_status: str = "unavailable"
    instrument_first_stage_strength: float = 0.0
    instrument_exclusion_proxy: float = 1.0
    iv_weight_multiplier: float = 1.0


@dataclass
class ObjectiveMetrics:
    trade_count: int
    win_rate: float
    payoff_ratio: float
    elasticity: float
    avg_win: float
    avg_loss: float
    avg_trade_magnitude: float
    avg_benchmark_magnitude: float
    objective_score: float


@dataclass
class SignalAllocation:
    symbol: str
    direction: str
    asset_type: str
    target_weight: float
    raw_score: float
    confidence: float
    objective_score: float
    selected_features: List[str]
    stop_loss_pct: float
    take_profit_pct: float
    decoder_state_entropy: float = 1.0
    decoder_risk_off_probability: float = 0.0
    decoder_sde_tail_loss_probability: float = 0.0
    decoder_sde_downside_q05: float = 0.0
    kernel_tail_loss_rate: float = 0.0
    kelly_fraction: float = 0.0
    capacity_weight_limit: float = 1.0
    commission_bps: float = 8.0
    slippage_bps: float = 0.0
    impact_bps: float = 12.0
    state_conditioning_multiplier: float = 1.0
    cross_asset_transfer_multiplier: float = 1.0
    transferred_factor_count: int = 0
    macro_event_overlay_multiplier: float = 1.0
    abstention_decision: str = ALLOW
    abstention_risk_score: float = 0.0
    abstention_reasons: List[str] = field(default_factory=list)


@dataclass
class PortfolioPlan:
    active_weight: float
    safe_weight: float
    tail_hedge_weight: float
    signal_allocations: List[SignalAllocation] = field(default_factory=list)
    projected_objective_score: float = 0.0
    barbell_ratio: float = 0.0
    residual_cash_weight: float = 0.0
    estimated_cost_penalty: float = 0.0
    estimated_impact_penalty: float = 0.0
    estimated_slippage_penalty: float = 0.0
    estimated_capacity_penalty: float = 0.0
    estimated_margin_penalty: float = 0.0
    estimated_tail_risk_penalty: float = 0.0
    estimated_breadth_penalty: float = 0.0
    hmm_barbell_state: str = "unclassified"
    hmm_barbell_audit: Dict[str, Any] = field(default_factory=dict)
    robustness_audit: Dict[str, Any] = field(default_factory=dict)


class SelfIteratingCausalEngine:
    """特征筛选、训练、组合、杠铃一体化因果学习引擎。"""

    def __init__(
        self,
        factor_library: Optional[FactorLibrary] = None,
        causal_factor_library: Optional[CausalFactorLibrary] = None,
        selection_policy: Optional[FeatureSelectionPolicy] = None,
        objective_config: Optional[LearningObjectiveConfig] = None,
        constraints: Optional[PortfolioConstraintConfig] = None,
        invariant_decoder_config: Optional[InvariantDecoderConfig] = None,
    ) -> None:
        self.factor_library = factor_library or FactorLibrary()
        self.causal_factor_library = causal_factor_library or CausalFactorLibrary()
        self.selection_policy = selection_policy or FeatureSelectionPolicy()
        self.objective_config = objective_config or LearningObjectiveConfig()
        self.constraints = constraints or PortfolioConstraintConfig()
        self.causal_validation_loop = CausalValidationLoop(min_observations=self.selection_policy.min_history)
        self.feature_store = FeatureStore()
        self.experiment_registry = ExperimentRegistry()
        self.model_registry = ModelRegistry()
        self.instrument_registry = InstrumentRegistry()
        self.abstention_gate = CausalAbstentionGate()
        self.causal_llm_auditor = CausalLLMAuditor()
        self.invariance_decoder = InvarianceMarketDecoder(invariant_decoder_config)
        self.causal_graph_layer = CausalGraphLayer(min_observations=self.selection_policy.min_history)
        self.latest_decoder_snapshots: Dict[str, Dict[str, Any]] = {}
        self.latest_scm_snapshots: Dict[str, Dict[str, Any]] = {}
        self.latest_abstention_decisions: Dict[str, Dict[str, Any]] = {}
        self.latest_llm_audit_records: List[Dict[str, Any]] = []
        self.cross_asset_factor_memory: Dict[str, Dict[str, Any]] = {}
        self.latest_iteration: Dict[str, Any] = {}

    def describe_capabilities(self) -> Dict[str, Any]:
        """返回引擎能力摘要。"""
        return {
            "feature_selection": {
                "mode": "model_driven_with_semantic_constraints",
                "min_rs_score": self.selection_policy.min_rs_score,
                "min_r_squared": self.selection_policy.min_r_squared,
                "max_selected_features": self.selection_policy.max_selected_features,
            },
            "objectives": {
                "win_rate_weight": self.objective_config.win_rate_weight,
                "payoff_weight": self.objective_config.payoff_weight,
                "elasticity_weight": self.objective_config.elasticity_weight,
                "state_conditioning_strength": self.objective_config.state_conditioning_strength,
                "cross_asset_transfer_weight": self.objective_config.cross_asset_transfer_weight,
            },
            "position_sizing": {
                "fractional_kelly_enabled": True,
                "max_fractional_kelly": self.constraints.max_fractional_kelly,
                "hard_caps": ["max_single_weight", "max_futures_weight", "max_gross_weight", "capacity_weight_limit"],
            },
            "macro_event_overlays": {
                "enabled": True,
                "inputs": ["SOFR", "MOVE", "bond_straddles", "US10Y/US30Y 5pct threshold", "DXY divergence", "CSI1000-CSI500 excess", "Hormuz reopen probability"],
                "role": "condition factor weights and tail-risk controls without bypassing causal validation gates",
            },
            "execution_objective": {
                "net_score_includes": ["commission", "slippage", "impact", "capacity_penalty", "margin", "tail_risk"],
                "default_max_participation_rate": self.constraints.default_max_participation_rate,
            },
            "global_futures_linkage": {
                "enabled": True,
                "feature_count": len(self.factor_library._global_peer_feature_metadata()),
                "supported_peers": {
                    "AU/AG/GOLD": ["COMEX_Gold", "XAUUSD"],
                    "CU/COPPER": ["LME_Copper", "HG"],
                },
            },
            "portfolio_constraints": asdict(self.constraints),
            "research_governance": {
                "causal_validation_gate": "validated_or_weak_identifiable_edges_only_for_position_sizing",
                "feature_store_enabled": True,
                "experiment_registry_enabled": True,
                "model_registry_enabled": True,
            },
            "scm_dag_layer": {
                "enabled": True,
                "candidate_edge_generators": ["pc", "fci", "pcmci"],
                "backdoor_adjustment_gate": True,
                "counterfactual_stress_to_tail_hedge": True,
            },
            "causal_abstention_gate": {
                "enabled": True,
                "outputs": [ALLOW, REDUCE, OBSERVE_ONLY, NO_TRADE],
                "inputs": [
                    "identification_status",
                    "backdoor_quality",
                    "hmm_state_entropy",
                    "data_validation",
                    "model_disagreement",
                    "counterfactual_tail_risk",
                    "instrument_status",
                ],
            },
            "instrument_registry": {
                "enabled": True,
                "role": "Deep-IV readiness diagnostics; valid instruments may boost weights, weak instruments stay observe-only.",
            },
            "causal_llm_auditor": {
                "enabled": True,
                "role": "audit-only hypothesis extraction; never changes position size before price/SCM/backdoor/IV validation.",
            },
            "invariance_decoder": {
                "enabled": True,
                "version": "invariance_decoder_v1",
                "features": ["invariance", "hmm", "kernel_analog", "sde_risk", "noisy_channel_posteriors"],
                "dependency_policy": "numpy_pandas_only",
            },
            "monolithic_research_factory": self._summarize_cross_asset_factor_memory(),
            "quantized_causal_factor_count": len(self.causal_factor_library.get_quantized_factor_ids()),
            "latest_iteration_status": self.latest_iteration.get("status", "idle"),
        }

    def run_learning_cycle(
        self,
        symbol_datasets: Dict[str, pd.DataFrame],
        benchmark_frame: Optional[pd.DataFrame] = None,
        market_context: Optional[Dict[str, Any]] = None,
        global_peer_datasets: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    ) -> Dict[str, Any]:
        """运行一次完整自迭代学习。"""
        if not symbol_datasets:
            return {
                "status": "no_data",
                "symbols": {},
                "portfolio_plan": None,
                "trade_actions": [],
            }

        benchmark_returns = self._build_benchmark_returns(symbol_datasets, benchmark_frame)
        symbol_reports: Dict[str, Any] = {}
        signal_candidates: List[Dict[str, Any]] = []
        all_validation_records: List[Dict[str, Any]] = []
        self.latest_decoder_snapshots = {}
        self.latest_scm_snapshots = {}
        self.latest_abstention_decisions = {}
        self.latest_llm_audit_records = self._audit_llm_hypotheses(market_context or {})

        for symbol, frame in symbol_datasets.items():
            normalized = self._normalize_ohlcv_frame(frame)
            peer_frames = (global_peer_datasets or {}).get(symbol, {})
            if len(normalized) < self.selection_policy.min_history:
                symbol_reports[symbol] = {
                    "status": "insufficient_history",
                    "rows": int(len(normalized)),
                    "global_peer_count": int(len(peer_frames)),
                    "selected_features": [],
                    "rejected_features": [],
                    "abstention_gate": {
                        "decision": NO_TRADE,
                        "reason": "insufficient_history",
                    },
                }
                continue

            factor_matrix = self._build_candidate_factor_matrix(
                normalized,
                symbol=symbol,
                peer_frames=peer_frames,
                benchmark_frame=benchmark_frame,
                market_context=market_context or {},
            )
            decoder_audit = self.latest_decoder_snapshots.get(symbol, {})
            target_returns = normalized["close"].pct_change(self.selection_policy.target_horizon).shift(
                -self.selection_policy.target_horizon
            )
            selected, rejected = self.auto_select_features(
                factor_matrix,
                target_returns,
                benchmark_returns=benchmark_returns,
            )
            scm_snapshot = self.causal_graph_layer.build_scm_snapshot(
                factor_matrix,
                target_returns,
                benchmark_returns=benchmark_returns,
            )
            self.latest_scm_snapshots[symbol] = scm_snapshot.to_audit_dict()

            if not selected:
                symbol_reports[symbol] = {
                    "status": "no_feature_passed_threshold",
                    "rows": int(len(normalized)),
                    "global_peer_count": int(len(peer_frames)),
                    "invariance_decoder": decoder_audit,
                    "scm_dag": self.latest_scm_snapshots.get(symbol, {}),
                    "selected_features": [],
                    "rejected_features": [asdict(item) for item in rejected[:15]],
                }
                continue

            tradable_selected, validation_records = self._validate_selected_features(
                symbol=symbol,
                factor_matrix=factor_matrix,
                target_returns=target_returns,
                selected_features=selected,
                benchmark_returns=benchmark_returns,
                scm_edges={edge["source"]: edge for edge in self.latest_scm_snapshots[symbol].get("edges", [])},
            )
            all_validation_records.extend(validation_records)

            if not tradable_selected:
                symbol_reports[symbol] = {
                    "status": "no_validated_causal_edge",
                    "rows": int(len(normalized)),
                    "global_peer_count": int(len(peer_frames)),
                    "invariance_decoder": decoder_audit,
                    "scm_dag": self.latest_scm_snapshots.get(symbol, {}),
                    "selected_features": [asdict(item) for item in selected],
                    "rejected_features": [asdict(item) for item in rejected[:15]],
                    "causal_validation": validation_records,
                    "validation_gate": "未通过 identifiable/weak_identifiable，因子保留观察但不得提高仓位。",
                }
                continue

            ensemble = self._train_factor_ensemble(
                factor_matrix,
                target_returns,
                tradable_selected,
                benchmark_returns=benchmark_returns,
                decoder_audit=decoder_audit,
                symbol=symbol,
                market_context=market_context or {},
            )
            latest_score = ensemble["latest_signal_score"]
            latest_confidence = ensemble["latest_confidence"]
            abstention = self._evaluate_signal_abstention(
                symbol=symbol,
                selected_features=tradable_selected,
                decoder_audit=decoder_audit,
                latest_score=latest_score,
                market_context=market_context or {},
            )
            self.latest_abstention_decisions[symbol] = asdict(abstention)
            if abs(latest_score) >= self.selection_policy.signal_threshold and abstention.decision in {ALLOW, REDUCE}:
                kernel_sde_risk = decoder_audit.get("audit_metadata", {}).get("kernel_sde_risk", {})
                signal_candidates.append(
                    {
                        "symbol": symbol,
                        "asset_type": self._infer_asset_type(symbol),
                        "direction": "long" if latest_score >= 0 else "short",
                        "raw_score": latest_score,
                        "confidence": latest_confidence,
                        "objective_score": ensemble["objective_metrics"].objective_score,
                        "objective_metrics": asdict(ensemble["objective_metrics"]),
                        "selected_features": [item.factor_name for item in tradable_selected],
                        "decoder_state_entropy": float(decoder_audit.get("state_entropy", 1.0) or 1.0),
                        "decoder_risk_off_probability": float(
                            decoder_audit.get("state_probabilities", {}).get("risk_off", 0.0) or 0.0
                        ),
                        "decoder_long_posterior": float(
                            decoder_audit.get("noisy_channel_posteriors", {}).get("LONG", 0.0) or 0.0
                        ),
                        "decoder_short_posterior": float(
                            decoder_audit.get("noisy_channel_posteriors", {}).get("SHORT", 0.0) or 0.0
                        ),
                        "decoder_sde_tail_loss_probability": float(
                            kernel_sde_risk.get("sde_tail_loss_probability", 0.0) or 0.0
                        ),
                        "decoder_sde_downside_q05": float(kernel_sde_risk.get("sde_downside_q05", 0.0) or 0.0),
                        "kernel_tail_loss_rate": float(kernel_sde_risk.get("kernel_analog_tail_loss_rate", 0.0) or 0.0),
                        "state_conditioning": ensemble.get("state_conditioning", {}),
                        "cross_asset_transfer": ensemble.get("cross_asset_transfer", {}),
                        "macro_event_overlay": ensemble.get("macro_event_overlay", {}),
                        "robustness_audit": ensemble.get("robustness_audit", {}),
                        "abstention_decision": abstention.decision,
                        "abstention_weight_multiplier": abstention.weight_multiplier,
                        "abstention_risk_score": abstention.risk_score,
                        "abstention_reasons": list(abstention.reasons),
                    }
                )

            symbol_reports[symbol] = {
                "status": "trained",
                "rows": int(len(normalized)),
                "global_peer_count": int(len(peer_frames)),
                "invariance_decoder": decoder_audit,
                "scm_dag": self.latest_scm_snapshots.get(symbol, {}),
                "selected_features": [asdict(item) for item in selected],
                "tradable_feature_count": int(len(tradable_selected)),
                "rejected_features": [asdict(item) for item in rejected[:15]],
                "causal_validation": validation_records,
                "factor_weights": ensemble["factor_weights"],
                "state_conditioning": ensemble.get("state_conditioning", {}),
                "cross_asset_transfer": ensemble.get("cross_asset_transfer", {}),
                "macro_event_overlay": ensemble.get("macro_event_overlay", {}),
                "robustness_audit": ensemble.get("robustness_audit", {}),
                "objective_metrics": asdict(ensemble["objective_metrics"]),
                "latest_signal_score": latest_score,
                "latest_confidence": latest_confidence,
                "abstention_gate": asdict(abstention),
            }

        portfolio_context = dict(market_context or {})
        portfolio_context["scm_counterfactual_stress"] = self._aggregate_counterfactual_stress(self.latest_scm_snapshots)
        portfolio_plan = self.optimize_portfolio(signal_candidates, portfolio_context)
        experiment_record = self._record_learning_experiment(
            symbol_datasets=symbol_datasets,
            signal_candidates=signal_candidates,
            symbol_reports=symbol_reports,
            portfolio_plan=portfolio_plan,
            validation_records=all_validation_records,
        )
        model_record = self._register_cycle_model(
            symbol_reports=symbol_reports,
            portfolio_plan=portfolio_plan,
            validation_records=all_validation_records,
        )
        result = {
            "status": "trained" if signal_candidates else "no_actionable_signals",
            "symbols": symbol_reports,
            "portfolio_plan": self._serialize_portfolio_plan(portfolio_plan),
            "trade_actions": self._portfolio_to_actions(portfolio_plan),
            "selection_policy": asdict(self.selection_policy),
            "constraints": asdict(self.constraints),
            "causal_validation_summary": self._summarize_validation_records(all_validation_records),
            "invariance_decoder": self._summarize_decoder_snapshots(self.latest_decoder_snapshots),
            "scm_dag": self._summarize_scm_snapshots(self.latest_scm_snapshots),
            "abstention_gate": self._summarize_abstention_decisions(self.latest_abstention_decisions),
            "instrument_registry": self.instrument_registry.latest_records(limit=50),
            "causal_llm_audit": self.latest_llm_audit_records,
            "monolithic_research_factory": self._summarize_cross_asset_factor_memory(),
            "robustness_controls": portfolio_plan.robustness_audit,
            "experiment_record": asdict(experiment_record),
            "model_registry_record": asdict(model_record),
            "feature_store_records": self.feature_store.latest_records(limit=25),
        }
        self.latest_iteration = result
        return result

    def auto_select_features(
        self,
        factor_matrix: pd.DataFrame,
        target_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> tuple[List[SelectedFeature], List[SelectedFeature]]:
        """按 RS>70 和 R²>0.7 自动筛选具备金融含义的特征。"""
        raw_rows: List[Dict[str, Any]] = []
        benchmark = self._align_series(benchmark_returns, factor_matrix.index) if benchmark_returns is not None else None
        discovery_edges = self.causal_graph_layer.discover_candidate_edges(factor_matrix, target_returns)

        for factor_name in factor_matrix.columns:
            series = pd.to_numeric(factor_matrix[factor_name], errors="coerce")
            aligned = pd.concat([series.rename("factor"), target_returns.rename("target")], axis=1).dropna()
            if benchmark is not None:
                aligned = aligned.join(benchmark.rename("benchmark"), how="inner").dropna()
            if len(aligned) < self.selection_policy.min_history:
                continue
            if series.notna().mean() < self.selection_policy.min_non_null_ratio:
                continue
            if float(aligned["factor"].std()) < 1e-8:
                continue

            if "benchmark" in aligned.columns:
                regression = self._factor_regression_with_benchmark(
                    aligned["factor"],
                    aligned["target"],
                    aligned["benchmark"],
                )
            else:
                regression = self._univariate_factor_regression(aligned["factor"], aligned["target"])

            edge = discovery_edges.get(factor_name)
            discovery_support = self.causal_graph_layer.discovery_support_for(discovery_edges, factor_name)
            raw_rows.append(
                {
                    "factor_name": factor_name,
                    "r_squared": regression["r_squared"],
                    "correlation": regression["correlation"],
                    "slope": regression["slope"],
                    "financial_meaning": self._factor_financial_meaning(factor_name),
                    "formula": self._factor_formula(factor_name),
                    "predictive_power": abs(regression["correlation"])
                    * (1.0 + regression["r_squared"])
                    * (1.0 + self.selection_policy.discovery_support_weight * discovery_support),
                    "causal_discovery_support": discovery_support,
                    "scm_edge_algorithms": list(edge.algorithms) if edge else [],
                }
            )

        if not raw_rows:
            return [], []

        ranks = pd.Series([row["predictive_power"] for row in raw_rows]).rank(pct=True, method="average") * 100
        selected: List[SelectedFeature] = []
        rejected: List[SelectedFeature] = []

        for row, rs_score in zip(raw_rows, ranks.tolist()):
            passes = bool(rs_score >= self.selection_policy.min_rs_score and row["r_squared"] >= self.selection_policy.min_r_squared)
            item = SelectedFeature(
                factor_name=row["factor_name"],
                financial_meaning=row["financial_meaning"],
                formula=row["formula"],
                rs_score=float(rs_score),
                r_squared=float(row["r_squared"]),
                correlation=float(row["correlation"]),
                slope=float(row["slope"]),
                direction=1 if row["slope"] >= 0 else -1,
                selected=passes,
                causal_discovery_support=float(row["causal_discovery_support"]),
                scm_edge_algorithms=list(row["scm_edge_algorithms"]),
                rejection_reason=None if passes else self._rejection_reason(rs_score, row["r_squared"]),
            )
            self.factor_library.update_factor_metadata(
                row["factor_name"],
                {
                    "latest_rs_score": round(float(rs_score), 6),
                    "latest_r_squared": round(float(row["r_squared"]), 6),
                    "latest_correlation": round(float(row["correlation"]), 6),
                    "selected_in_latest_cycle": passes,
                    "latest_causal_discovery_support": round(float(row["causal_discovery_support"]), 6),
                    "latest_scm_edge_algorithms": list(row["scm_edge_algorithms"]),
                    "financial_meaning": row["financial_meaning"],
                    "formula": row["formula"],
                },
            )
            if passes:
                selected.append(item)
            else:
                rejected.append(item)

        selected.sort(key=lambda item: (item.rs_score, item.r_squared), reverse=True)
        rejected.sort(key=lambda item: (item.rs_score, item.r_squared), reverse=True)
        final_selected = selected[: self.selection_policy.max_selected_features]
        for item in final_selected:
            self.factor_library.update_factor_metadata(
                item.factor_name,
                {"latest_selected_rank": final_selected.index(item) + 1},
            )
        self.factor_library.save_factor_metadata()
        return final_selected, rejected

    def optimize_portfolio(
        self,
        signal_candidates: List[Dict[str, Any]],
        market_context: Dict[str, Any],
    ) -> PortfolioPlan:
        """把组合约束和杠铃配置联合纳入目标优化。"""
        market_context = dict(market_context or {})
        candidates = sorted(
            signal_candidates,
            key=lambda item: (item["objective_score"], abs(item["raw_score"]), item["confidence"]),
            reverse=True,
        )[: self.constraints.max_positions]
        breadth_audit = self._candidate_breadth_audit(candidates, market_context)
        market_context["effective_breadth_audit"] = breadth_audit
        tail_risk = self._extract_tail_risk_score(market_context)
        barbell_profile = self._hmm_barbell_profile(candidates, market_context, tail_risk)
        if not candidates:
            tail_weight = float(barbell_profile["default_tail_weight"])
            return PortfolioPlan(
                active_weight=0.0,
                safe_weight=round(1.0 - tail_weight, 6),
                tail_hedge_weight=round(tail_weight, 6),
                signal_allocations=[],
                projected_objective_score=0.0,
                barbell_ratio=round(tail_weight, 6),
                residual_cash_weight=0.0,
                hmm_barbell_state=str(barbell_profile["state"]),
                hmm_barbell_audit=barbell_profile,
                robustness_audit={"effective_breadth": breadth_audit},
            )

        best_plan: Optional[PortfolioPlan] = None
        best_score = -np.inf

        for active_weight in barbell_profile["active_weight_grid"]:
            barbell_budget = max(0.0, 1.0 - active_weight)
            for tail_weight in barbell_profile["tail_weight_grid"]:
                if tail_weight > barbell_budget:
                    continue
                safe_weight = barbell_budget - tail_weight
                if safe_weight < float(barbell_profile["min_safe_weight"]):
                    continue
                allocations = self._allocate_signal_weights(candidates, active_weight, market_context)
                allocations, residual_cash = self._enforce_portfolio_constraints(allocations)
                projected_score = self._score_portfolio(
                    allocations,
                    tail_weight=tail_weight,
                    safe_weight=safe_weight,
                    tail_risk=tail_risk,
                    market_context=market_context,
                )
                penalties = self._portfolio_penalties(allocations, tail_weight, market_context)
                if projected_score > best_score:
                    best_score = projected_score
                    best_plan = PortfolioPlan(
                        active_weight=round(sum(item.target_weight for item in allocations), 6),
                        safe_weight=round(safe_weight + residual_cash, 6),
                        tail_hedge_weight=round(tail_weight, 6),
                        signal_allocations=allocations,
                        projected_objective_score=round(projected_score, 6),
                        barbell_ratio=round(tail_weight / max(barbell_budget, 1e-9), 6),
                        residual_cash_weight=0.0,
                        estimated_cost_penalty=round(penalties["transaction_cost"], 6),
                        estimated_impact_penalty=round(penalties["impact_cost"], 6),
                        estimated_slippage_penalty=round(penalties["slippage_cost"], 6),
                        estimated_capacity_penalty=round(penalties["capacity_penalty"], 6),
                        estimated_margin_penalty=round(penalties["margin_penalty"], 6),
                        estimated_tail_risk_penalty=round(penalties["tail_risk_penalty"], 6),
                        estimated_breadth_penalty=round(penalties["breadth_penalty"], 6),
                        hmm_barbell_state=str(barbell_profile["state"]),
                        hmm_barbell_audit=barbell_profile,
                        robustness_audit={"effective_breadth": breadth_audit},
                    )

        return best_plan or PortfolioPlan(
            active_weight=0.0,
            safe_weight=round(1.0 - float(barbell_profile["default_tail_weight"]), 6),
            tail_hedge_weight=round(float(barbell_profile["default_tail_weight"]), 6),
            signal_allocations=[],
            projected_objective_score=0.0,
            barbell_ratio=round(float(barbell_profile["default_tail_weight"]), 6),
            residual_cash_weight=0.0,
            hmm_barbell_state=str(barbell_profile["state"]),
            hmm_barbell_audit=barbell_profile,
            robustness_audit={"effective_breadth": breadth_audit},
        )

    def _hmm_barbell_profile(
        self,
        candidates: List[Dict[str, Any]],
        market_context: Dict[str, Any],
        tail_risk: float,
    ) -> Dict[str, Any]:
        """Map decoded hidden states into dynamic Taleb barbell budgets."""

        risk_off_values = [float(item.get("decoder_risk_off_probability", 0.0) or 0.0) for item in candidates]
        entropy_values = [float(item.get("decoder_state_entropy", 1.0) or 1.0) for item in candidates]
        risk_off = max(risk_off_values or [0.0])
        avg_entropy = float(np.mean(entropy_values)) if entropy_values else 1.0
        cross_asset_regime = market_context.get("cross_asset_regime", {})
        cross_asset_regime_name = cross_asset_regime.get("regime", "") if isinstance(cross_asset_regime, dict) else str(cross_asset_regime)
        explicit_state = str(
            market_context.get("hmm_barbell_state")
            or market_context.get("hmm_state")
            or market_context.get("regime")
            or cross_asset_regime_name
        ).lower()
        previous_state = str(
            market_context.get("previous_hmm_barbell_state")
            or market_context.get("prior_hmm_barbell_state")
            or ""
        ).lower()
        crisis_entry_threshold = float(market_context.get("hmm_crisis_entry_threshold", 0.85))
        crisis_exit_threshold = float(market_context.get("hmm_crisis_exit_threshold", 0.25))
        explicit_crisis = explicit_state in {"risk_off", "liquidity_stress", "crisis", "bear"}
        explicit_normal = explicit_state in {"risk_on", "trend", "bull", "soft_landing"}
        crisis_probability = max(risk_off, tail_risk, 0.90 if explicit_crisis else 0.0)
        retained_crisis = previous_state in {"state2_liquidity_crisis", "risk_off", "crisis"} and crisis_probability >= crisis_exit_threshold
        enter_crisis = crisis_probability >= crisis_entry_threshold or explicit_crisis
        release_normal = explicit_normal or (risk_off <= 0.25 and avg_entropy <= 0.55 and crisis_probability < crisis_exit_threshold)
        sigmoid_crisis = 1.0 / (1.0 + np.exp(-10.0 * (crisis_probability - 0.55)))

        if enter_crisis or retained_crisis:
            state = "state2_liquidity_crisis"
            active_cap = float(np.clip(0.18 - 0.10 * sigmoid_crisis, 0.05, 0.15))
            active_grid = sorted({0.0, 0.05, round(active_cap * 0.70, 2), round(active_cap, 2)})
            tail_center = float(np.clip(0.15 + 0.08 * sigmoid_crisis + 0.08 * tail_risk, 0.15, 0.25))
            tail_grid = sorted({0.15, round(tail_center, 2), round(min(tail_center + 0.03, 0.25), 2)})
            min_safe = float(np.clip(1.0 - active_cap - max(tail_grid), 0.70, 0.90))
            default_tail = float(np.clip(tail_center, 0.15, 0.25))
            rule = "crisis hysteresis: enter above 0.85/explicit crisis, exit only below 0.25; soft sigmoid mapping avoids hard 85/15 jumps."
        elif release_normal:
            state = "state1_trend_or_normal"
            active_cap = float(np.clip(0.85 - 0.20 * sigmoid_crisis - 0.10 * avg_entropy, 0.55, 0.90))
            active_grid = sorted({0.55, 0.65, 0.75, round(active_cap, 2), 0.85})
            tail_grid = [0.06, 0.08, 0.10, 0.12]
            min_safe = 0.05
            default_tail = float(np.clip(0.08 + tail_risk * 0.20, 0.08, 0.16))
            rule = "normal/trend: release risk budget only after crisis probability leaves hysteresis band; retain modest tail hedge."
        else:
            state = "state0_transition_choppy"
            active_cap = float(np.clip(0.58 - 0.30 * sigmoid_crisis - 0.15 * avg_entropy, 0.20, 0.55))
            active_grid = sorted({0.20, 0.30, 0.40, round(active_cap, 2)})
            tail_center = float(np.clip(0.10 + 0.10 * sigmoid_crisis + tail_risk * 0.20, 0.10, 0.23))
            tail_grid = sorted({0.10, 0.12, round(tail_center, 2), round(min(tail_center + 0.04, 0.25), 2)})
            min_safe = 0.20
            default_tail = float(np.clip(tail_center, 0.10, 0.25))
            rule = "transition/choppy hysteresis band: reduce active risk, smooth tail hedge, avoid whipsaw near a single threshold."

        return {
            "state": state,
            "risk_off_probability": round(float(risk_off), 6),
            "avg_state_entropy": round(float(avg_entropy), 6),
            "tail_risk_score": round(float(tail_risk), 6),
            "crisis_probability": round(float(crisis_probability), 6),
            "sigmoid_crisis_mapping": round(float(sigmoid_crisis), 6),
            "previous_state": previous_state or "none",
            "hysteresis": {
                "entry_threshold": round(crisis_entry_threshold, 6),
                "exit_threshold": round(crisis_exit_threshold, 6),
                "retained_crisis": bool(retained_crisis),
                "entered_crisis": bool(enter_crisis),
            },
            "active_weight_grid": active_grid,
            "tail_weight_grid": tail_grid,
            "min_safe_weight": min_safe,
            "default_tail_weight": round(float(default_tail), 6),
            "rule": rule,
            "formula": "HMM posterior + tail risk -> hysteresis gate -> sigmoid-smoothed active/safe/tail grids",
        }

    def _validate_selected_features(
        self,
        symbol: str,
        factor_matrix: pd.DataFrame,
        target_returns: pd.Series,
        selected_features: List[SelectedFeature],
        benchmark_returns: Optional[pd.Series] = None,
        scm_edges: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> tuple[List[SelectedFeature], List[Dict[str, Any]]]:
        tradable: List[SelectedFeature] = []
        records: List[Dict[str, Any]] = []
        scm_edges = scm_edges or {}
        for feature in selected_features:
            backdoor = self.causal_graph_layer.backdoor_adjustment(
                feature.factor_name,
                "forward_return",
                factor_matrix,
                benchmark_returns=benchmark_returns,
            )
            adjustment_frame = self.causal_graph_layer.adjustment_frame(
                backdoor,
                factor_matrix,
                benchmark_returns=benchmark_returns,
            )
            discovery_support = float(scm_edges.get(feature.factor_name, {}).get("confidence", feature.causal_discovery_support) or 0.0)
            validation = self.causal_validation_loop.validate_feature(
                feature_name=feature.factor_name,
                factor_series=factor_matrix[feature.factor_name],
                target_returns=target_returns,
                benchmark_returns=benchmark_returns,
                adjustment_frame=adjustment_frame,
                backdoor_adjustment=asdict(backdoor),
                discovery_support=discovery_support,
            )
            instrument_record = self.instrument_registry.diagnose_edge(
                source=feature.factor_name,
                target="forward_return",
                factor_matrix=factor_matrix,
                target_returns=target_returns,
            )
            feature.identification_status = validation.identification_status
            feature.validation_score = validation.validation_score
            feature.can_trade = validation.can_trade
            feature.backdoor_adjustment_quality = backdoor.adjustment_quality
            feature.causal_discovery_support = discovery_support
            feature.instrument_status = instrument_record.validity_status
            feature.instrument_first_stage_strength = instrument_record.first_stage_strength
            feature.instrument_exclusion_proxy = instrument_record.exclusion_proxy
            feature.iv_weight_multiplier = (
                round(float(np.clip(1.0 + 0.50 * instrument_record.first_stage_strength, 1.0, 1.15)), 6)
                if instrument_record.validity_status == "valid"
                else 1.0
            )
            record = asdict(validation)
            record["symbol"] = symbol
            record["scm_edge"] = scm_edges.get(feature.factor_name, {})
            record["instrument_registry"] = asdict(instrument_record)
            records.append(record)

            self.feature_store.register_feature(
                name=feature.factor_name,
                financial_meaning=feature.financial_meaning,
                formula=feature.formula,
                data_lineage={
                    "symbol": symbol,
                    "source": "self_iterating_candidate_factor_matrix",
                    "rows": int(len(factor_matrix)),
                    "target_horizon": self.selection_policy.target_horizon,
                    "scm_edge": scm_edges.get(feature.factor_name, {}),
                },
                leakage_check={
                    "forward_target_shift_days": self.selection_policy.target_horizon,
                    "uses_only_information_available_at_or_before_signal_time": True,
                    "purged_cv_required_before_promotion": True,
                    "cross_asset_transfer_requires_validation": True,
                    "backdoor_adjustment_required_before_position_sizing": True,
                    "weak_instrument_forces_observe_only": True,
                    "causal_llm_output_is_audit_only": True,
                },
                validation_status=record,
            )
            if validation.can_trade and instrument_record.validity_status not in {"weak", "invalid"}:
                self._update_cross_asset_factor_memory(feature, symbol, record)
                tradable.append(feature)
        return tradable, records

    def _record_learning_experiment(
        self,
        symbol_datasets: Dict[str, pd.DataFrame],
        signal_candidates: List[Dict[str, Any]],
        symbol_reports: Dict[str, Any],
        portfolio_plan: PortfolioPlan,
        validation_records: List[Dict[str, Any]],
    ):
        row_counts = {
            symbol: int(len(frame))
            for symbol, frame in symbol_datasets.items()
        }
        validation_summary = self._summarize_validation_records(validation_records)
        decoder_summary = self._summarize_decoder_snapshots(self.latest_decoder_snapshots)
        scm_summary = self._summarize_scm_snapshots(self.latest_scm_snapshots)
        abstention_summary = self._summarize_abstention_decisions(self.latest_abstention_decisions)
        transfer_summary = self._summarize_cross_asset_factor_memory()
        failure_reasons = sorted(
            {
                str(report.get("status"))
                for report in symbol_reports.values()
                if str(report.get("status")) not in {"trained"}
            }
        )
        return self.experiment_registry.record_experiment(
            name="self_iterating_causal_learning_cycle",
            data_version={
                "symbols": sorted(symbol_datasets.keys()),
                "row_counts": row_counts,
                "feature_policy": asdict(self.selection_policy),
            },
            training_window={
                "target_horizon": self.selection_policy.target_horizon,
                "min_history": self.selection_policy.min_history,
                "walk_forward_required": True,
                "purged_cv_required": True,
                "embargo_required": True,
            },
            test_window={
                "mode": "latest_split_shadow_gate",
                "oos_proxy": "second_half_correlation",
            },
            metrics={
                "signal_candidate_count": len(signal_candidates),
                "projected_objective_score": portfolio_plan.projected_objective_score,
                "active_weight": portfolio_plan.active_weight,
                "validation_summary": validation_summary,
                "invariance_decoder": decoder_summary,
                "scm_dag": scm_summary,
                "abstention_gate": abstention_summary,
                "instrument_registry": self.instrument_registry.latest_records(limit=50),
                "causal_llm_audit": self.latest_llm_audit_records,
                "monolithic_research_factory": transfer_summary,
                "robustness_controls": portfolio_plan.robustness_audit,
                "complexity_defense_gate": (
                    "Shapley attribution is offline-only; effective breadth penalizes correlated bets; "
                    "CPCV/DSR required before production promotion."
                ),
            },
            failure_reasons=failure_reasons,
            status="completed",
        )

    def _register_cycle_model(
        self,
        symbol_reports: Dict[str, Any],
        portfolio_plan: PortfolioPlan,
        validation_records: List[Dict[str, Any]],
    ):
        validation_summary = self._summarize_validation_records(validation_records)
        decoder_summary = self._summarize_decoder_snapshots(self.latest_decoder_snapshots)
        scm_summary = self._summarize_scm_snapshots(self.latest_scm_snapshots)
        abstention_summary = self._summarize_abstention_decisions(self.latest_abstention_decisions)
        transfer_summary = self._summarize_cross_asset_factor_memory()
        passed_symbols = [
            symbol
            for symbol, report in symbol_reports.items()
            if report.get("status") == "trained"
        ]
        if passed_symbols and validation_summary["tradable_edge_count"] > 0:
            promotion_status = "nightly_candidate"
            promotion_reason = "至少一个符号具备可交易因果边并通过组合约束。"
        else:
            promotion_status = "shadow_only"
            promotion_reason = "缺少通过验证的因果边，保留为观察/影子模型。"
        return self.model_registry.register_model(
            name="self_iterating_causal_factor_ensemble",
            training_window={
                "target_horizon": self.selection_policy.target_horizon,
                "min_history": self.selection_policy.min_history,
                "symbol_count": len(symbol_reports),
            },
            validation_summary={
                **validation_summary,
                "projected_objective_score": portfolio_plan.projected_objective_score,
                "estimated_cost_penalty": portfolio_plan.estimated_cost_penalty,
                "estimated_impact_penalty": portfolio_plan.estimated_impact_penalty,
                "estimated_slippage_penalty": portfolio_plan.estimated_slippage_penalty,
                "estimated_capacity_penalty": portfolio_plan.estimated_capacity_penalty,
                "estimated_breadth_penalty": portfolio_plan.estimated_breadth_penalty,
                "invariance_decoder": decoder_summary,
                "scm_dag": scm_summary,
                "abstention_gate": abstention_summary,
                "instrument_registry": self.instrument_registry.latest_records(limit=50),
                "causal_llm_audit": self.latest_llm_audit_records,
                "monolithic_research_factory": transfer_summary,
                "robustness_controls": portfolio_plan.robustness_audit,
                "shapley_deployment_policy": shapley_deployment_policy(
                    model_family="tree_or_ensemble_factor_model",
                    feature_count=sum(
                        len(report.get("selected_features", []))
                        for report in symbol_reports.values()
                        if isinstance(report, dict)
                    ),
                    frequency="weekly",
                ),
            },
            promotion_status=promotion_status,
            promotion_reason=promotion_reason,
        )

    @staticmethod
    def _summarize_validation_records(validation_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for record in validation_records:
            status = str(record.get("identification_status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        scores = [float(record.get("validation_score", 0.0) or 0.0) for record in validation_records]
        tradable_count = sum(1 for record in validation_records if record.get("can_trade"))
        return {
            "edge_count": len(validation_records),
            "tradable_edge_count": tradable_count,
            "status_counts": counts,
            "avg_validation_score": round(float(np.mean(scores)), 6) if scores else 0.0,
            "gate": "only identifiable or weak_identifiable edges can increase position size",
        }

    def _evaluate_signal_abstention(
        self,
        symbol: str,
        selected_features: List[SelectedFeature],
        decoder_audit: Dict[str, Any],
        latest_score: float,
        market_context: Dict[str, Any],
    ):
        statuses = [feature.identification_status for feature in selected_features]
        identification_status = IDENTIFIABLE if statuses and all(status == IDENTIFIABLE for status in statuses) else WEAK_IDENTIFIABLE
        backdoor_quality = min([feature.backdoor_adjustment_quality for feature in selected_features] or [0.0])
        iv_statuses = [feature.instrument_status for feature in selected_features]
        if any(status == "valid" for status in iv_statuses):
            instrument_status = "valid"
        elif any(status in {"weak", "invalid"} for status in iv_statuses):
            instrument_status = "weak"
        else:
            instrument_status = "unavailable"
        posteriors = decoder_audit.get("noisy_channel_posteriors", {}) if isinstance(decoder_audit, dict) else {}
        long_p = float(posteriors.get("LONG", 0.0) or 0.0)
        short_p = float(posteriors.get("SHORT", 0.0) or 0.0)
        hold_p = float(posteriors.get("HOLD", 0.0) or 0.0)
        if latest_score >= 0:
            model_disagreement = short_p + 0.50 * hold_p
        else:
            model_disagreement = long_p + 0.50 * hold_p
        scm_stress = (self.latest_scm_snapshots.get(symbol, {}) or {}).get("counterfactual_stress", {})
        decision = self.abstention_gate.evaluate(
            identification_status=identification_status,
            backdoor_quality=backdoor_quality,
            hmm_state_entropy=float(decoder_audit.get("state_entropy", 1.0) or 1.0),
            data_validation_passed=bool(market_context.get("data_validation_passed", True)),
            model_disagreement=float(model_disagreement),
            counterfactual_tail_risk=float(scm_stress.get("tail_risk_score", 0.0) or 0.0),
            instrument_status=instrument_status,
        )
        return decision

    @staticmethod
    def _summarize_abstention_decisions(decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        risk_scores = []
        for decision in decisions.values():
            label = str(decision.get("decision", "unknown"))
            counts[label] = counts.get(label, 0) + 1
            risk_scores.append(float(decision.get("risk_score", 0.0) or 0.0))
        return {
            "decision_count": len(decisions),
            "decision_counts": counts,
            "avg_risk_score": round(float(np.mean(risk_scores)), 6) if risk_scores else 0.0,
            "symbols": decisions,
            "gate": "ALLOW sizes normally, REDUCE haircuts weights, OBSERVE_ONLY/NO_TRADE cannot enter trade_actions",
        }

    def _audit_llm_hypotheses(self, market_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        payload = (
            market_context.get("causal_llm_hypotheses")
            or market_context.get("llm_causal_hypotheses")
            or market_context.get("llm_hypotheses")
        )
        game_analysis = market_context.get("game_causal_analysis", {})
        if payload is None and isinstance(game_analysis, dict):
            payload = (
                game_analysis.get("causal_llm_hypotheses")
                or game_analysis.get("llm_hypotheses")
                or game_analysis.get("events")
            )
        records = self.causal_llm_auditor.audit_hypotheses(payload)
        return [asdict(record) for record in records]

    @staticmethod
    def _summarize_decoder_snapshots(snapshots: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if not snapshots:
            return {
                "decoder_count": 0,
                "active_count": 0,
                "avg_state_entropy": 0.0,
                "max_risk_off_probability": 0.0,
                "symbols": {},
            }
        entropies = [float(item.get("state_entropy", 1.0) or 1.0) for item in snapshots.values()]
        risk_off = [
            float(item.get("state_probabilities", {}).get("risk_off", 0.0) or 0.0)
            for item in snapshots.values()
        ]
        return {
            "decoder_count": len(snapshots),
            "active_count": sum(1 for item in snapshots.values() if item.get("status") == "active"),
            "avg_state_entropy": round(float(np.mean(entropies)), 6) if entropies else 0.0,
            "max_risk_off_probability": round(float(max(risk_off)), 6) if risk_off else 0.0,
            "symbols": snapshots,
        }

    @staticmethod
    def _aggregate_counterfactual_stress(snapshots: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        stresses = []
        for symbol, snapshot in snapshots.items():
            stress = snapshot.get("counterfactual_stress", {})
            if stress:
                stresses.append(
                    {
                        "symbol": symbol,
                        "tail_risk_score": float(stress.get("tail_risk_score", 0.0) or 0.0),
                        "tail_hedge_multiplier": float(stress.get("tail_hedge_multiplier", 1.0) or 1.0),
                        "expected_portfolio_impact": float(stress.get("expected_portfolio_impact", 0.0) or 0.0),
                    }
                )
        return {
            "max_tail_risk_score": round(max([item["tail_risk_score"] for item in stresses] or [0.0]), 6),
            "max_tail_hedge_multiplier": round(max([item["tail_hedge_multiplier"] for item in stresses] or [1.0]), 6),
            "min_expected_portfolio_impact": round(min([item["expected_portfolio_impact"] for item in stresses] or [0.0]), 6),
            "symbols": stresses,
        }

    def _summarize_scm_snapshots(self, snapshots: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if not snapshots:
            return {
                "graph_count": 0,
                "candidate_edge_count": 0,
                "max_counterfactual_tail_risk": 0.0,
                "symbols": {},
            }
        edge_count = sum(len(snapshot.get("edges", [])) for snapshot in snapshots.values())
        stress = self._aggregate_counterfactual_stress(snapshots)
        return {
            "graph_count": len(snapshots),
            "candidate_edge_count": edge_count,
            "max_counterfactual_tail_risk": stress["max_tail_risk_score"],
            "max_tail_hedge_multiplier": stress["max_tail_hedge_multiplier"],
            "symbols": snapshots,
            "gate": "SCM/DAG candidate edges require backdoor-adjusted validation before sizing",
        }

    def _update_cross_asset_factor_memory(
        self,
        feature: SelectedFeature,
        symbol: str,
        validation_record: Dict[str, Any],
    ) -> None:
        asset_type = self._infer_asset_type(symbol)
        record = self.cross_asset_factor_memory.setdefault(
            feature.factor_name,
            {
                "symbols": set(),
                "asset_types": set(),
                "validation_scores": [],
                "identification_statuses": set(),
            },
        )
        record["symbols"].add(symbol)
        record["asset_types"].add(asset_type)
        record["validation_scores"].append(float(validation_record.get("validation_score", 0.0) or 0.0))
        record["identification_statuses"].add(str(validation_record.get("identification_status", "unknown")))

    def _cross_asset_transfer_multiplier(self, factor_name: str, symbol: str) -> float:
        record = self.cross_asset_factor_memory.get(factor_name)
        if not record:
            return 1.0
        source_symbols = set(record.get("symbols", set())) - {symbol}
        if not source_symbols:
            return 1.0
        current_asset_type = self._infer_asset_type(symbol)
        source_asset_types = set(record.get("asset_types", set())) - {current_asset_type}
        validation_scores = [float(item) for item in record.get("validation_scores", [])]
        avg_validation = float(np.mean(validation_scores)) if validation_scores else 0.0
        cross_asset_bonus = 0.50 if source_asset_types else 0.25
        source_depth = min(len(source_symbols), 4) / 4.0
        multiplier = 1.0 + self.objective_config.cross_asset_transfer_weight * avg_validation * cross_asset_bonus * source_depth
        return round(float(np.clip(multiplier, 1.0, 1.20)), 6)

    def _summarize_cross_asset_factor_memory(self) -> Dict[str, Any]:
        factors: Dict[str, Any] = {}
        for name, record in self.cross_asset_factor_memory.items():
            validation_scores = [float(item) for item in record.get("validation_scores", [])]
            factors[name] = {
                "symbols": sorted(record.get("symbols", set())),
                "asset_types": sorted(record.get("asset_types", set())),
                "avg_validation_score": round(float(np.mean(validation_scores)), 6) if validation_scores else 0.0,
                "identification_statuses": sorted(record.get("identification_statuses", set())),
                "transfer_eligible": len(record.get("symbols", set())) >= 1,
            }
        return {
            "validated_factor_count": len(factors),
            "transfer_eligible_count": sum(1 for item in factors.values() if item["transfer_eligible"]),
            "factors": factors,
            "gate": "cross-asset migration can only boost weights after causal validation records exist",
        }

    def _normalize_ohlcv_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for column in frame.columns:
            lower = column.lower()
            if lower in {"timestamp", "datetime", "date", "time"}:
                rename_map[column] = "date"
            else:
                rename_map[column] = lower
        normalized = frame.rename(columns=rename_map).copy()
        for column in ["open", "high", "low", "close", "volume"]:
            if column not in normalized.columns:
                if column == "volume":
                    normalized[column] = 1.0
                else:
                    normalized[column] = normalized.get("close", pd.Series(index=normalized.index, dtype=float))
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized = normalized.dropna(subset=["close"]).reset_index(drop=True)
        return normalized

    def _build_candidate_factor_matrix(
        self,
        frame: pd.DataFrame,
        symbol: str = "",
        peer_frames: Optional[Dict[str, pd.DataFrame]] = None,
        benchmark_frame: Optional[pd.DataFrame] = None,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        enriched = frame
        global_peer = pd.DataFrame(index=frame.index)
        if self._infer_asset_type(symbol) == "futures":
            enriched = self.factor_library.attach_global_peer_context(frame, peer_frames)
            global_peer = self.factor_library.compute_global_futures_linkage_factors(enriched)

        base = pd.DataFrame(index=frame.index)
        base["base_ret_5"] = enriched["close"].pct_change(5)
        base["base_ret_20"] = enriched["close"].pct_change(20)
        base["base_ret_60"] = enriched["close"].pct_change(60)
        base["base_price_vs_ma20"] = enriched["close"] / enriched["close"].rolling(20).mean() - 1
        base["base_volume_surprise_20"] = enriched["volume"] / enriched["volume"].rolling(20).mean() - 1
        base["base_drawdown_20"] = enriched["close"] / enriched["close"].rolling(20).max() - 1

        technical = self.factor_library.compute_all_technical_factors(enriched)
        causal_names = self.factor_library.get_factor_list(category="causal_quant")
        causal = self.factor_library.compute_factor_batch(causal_names, enriched, use_polars=False, parallel=False)
        decoder_snapshot = self.invariance_decoder.fit_transform(
            enriched,
            benchmark_frame=benchmark_frame,
            peer_frames=peer_frames,
            symbol=symbol,
        )
        decoder_features = decoder_snapshot.feature_frame.reindex(frame.index).fillna(0.0)
        self.latest_decoder_snapshots[symbol] = decoder_snapshot.to_audit_dict()
        event_features = self._event_intensity_feature_frame(frame, market_context or {})
        matrix = pd.concat([base, global_peer, technical, causal, decoder_features, event_features], axis=1)
        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        matrix = matrix.loc[:, matrix.notna().mean() >= self.selection_policy.min_non_null_ratio]
        matrix = matrix.ffill().fillna(0.0)
        return matrix

    def _event_intensity_feature_frame(self, frame: pd.DataFrame, market_context: Dict[str, Any]) -> pd.DataFrame:
        """Attach leakage-safe event intensity factors to the training matrix."""

        records = (
            market_context.get("event_intensity_records")
            or market_context.get("news_items")
            or market_context.get("policy_records")
            or []
        )
        game_analysis = market_context.get("game_causal_analysis", {})
        if not records and isinstance(game_analysis, dict):
            event_payload = game_analysis.get("event_intensity", {})
            if isinstance(event_payload, dict) and event_payload.get("feature_frame_records"):
                return self._event_intensity_records_to_frame(frame, event_payload["feature_frame_records"])
            records = game_analysis.get("events", [])
        if not records:
            return pd.DataFrame(index=frame.index)
        calendar_index = frame["date"] if "date" in frame.columns else frame.index
        as_of = frame["date"].iloc[-1] if "date" in frame.columns and len(frame) else None
        snapshot = self.causal_factor_library.compute_event_intensity_factors(
            records,
            calendar_index=list(calendar_index),
            as_of=as_of,
            include_records=False,
        )
        features = snapshot.feature_frame.copy()
        if features.empty:
            return pd.DataFrame(index=frame.index)
        features = features.reset_index(drop=True)
        features.index = frame.index[: len(features)]
        return features.reindex(frame.index).ffill().fillna(0.0)

    @staticmethod
    def _event_intensity_records_to_frame(frame: pd.DataFrame, records: Any) -> pd.DataFrame:
        try:
            feature_frame = pd.DataFrame(records)
        except Exception:
            return pd.DataFrame(index=frame.index)
        if feature_frame.empty:
            return pd.DataFrame(index=frame.index)
        for column in list(feature_frame.columns):
            if str(column).lower() in {"date", "timestamp", "datetime", "time"}:
                feature_frame = feature_frame.drop(columns=[column])
        feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
        feature_frame = feature_frame.reset_index(drop=True)
        feature_frame = feature_frame.iloc[-len(frame) :] if len(feature_frame) > len(frame) else feature_frame
        feature_frame.index = frame.index[-len(feature_frame) :]
        return feature_frame.reindex(frame.index).ffill().fillna(0.0)

    def get_global_peer_symbols(self, symbol: str) -> List[str]:
        """返回某个期货品种应联动参考的全球同类合约。"""
        token = symbol.upper()
        if any(key in token for key in ["AU", "AG", "XAU", "GOLD"]):
            return ["COMEX_Gold", "XAUUSD"]
        if any(key in token for key in ["CU", "COPPER", "HG", "LME"]):
            return ["LME_Copper", "HG"]
        return []

    def _train_factor_ensemble(
        self,
        factor_matrix: pd.DataFrame,
        target_returns: pd.Series,
        selected_features: List[SelectedFeature],
        benchmark_returns: Optional[pd.Series] = None,
        decoder_audit: Optional[Dict[str, Any]] = None,
        symbol: str = "",
        market_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        weighted_signals: Dict[str, pd.Series] = {}
        raw_weights: Dict[str, float] = {}
        state_conditioning: Dict[str, float] = {}
        transfer_multipliers: Dict[str, float] = {}
        macro_event_multipliers: Dict[str, float] = {}
        selected_names = [feature.factor_name for feature in selected_features if feature.factor_name in factor_matrix.columns]
        _, orthogonalization_audit = democratic_orthogonalize(factor_matrix[selected_names]) if selected_names else (
            pd.DataFrame(index=factor_matrix.index),
            {"status": "empty"},
        )

        for feature in selected_features:
            series = pd.to_numeric(factor_matrix[feature.factor_name], errors="coerce")
            signal = self.factor_library._zscore(series).fillna(0.0) * feature.direction
            metrics = self._evaluate_objective(signal, target_returns, benchmark_returns)
            feature.objective_score = metrics.objective_score
            state_multiplier = self._state_conditioning_multiplier(feature.factor_name, decoder_audit or {})
            transfer_multiplier = self._cross_asset_transfer_multiplier(feature.factor_name, symbol)
            macro_event_multiplier = self._macro_event_factor_multiplier(feature.factor_name, symbol, market_context or {})
            raw_weight = (
                max(metrics.objective_score, 1e-6)
                * (feature.rs_score / 100.0)
                * feature.r_squared
                * state_multiplier
                * transfer_multiplier
                * macro_event_multiplier
                * feature.iv_weight_multiplier
            )
            weighted_signals[feature.factor_name] = signal
            raw_weights[feature.factor_name] = raw_weight
            state_conditioning[feature.factor_name] = state_multiplier
            transfer_multipliers[feature.factor_name] = transfer_multiplier
            macro_event_multipliers[feature.factor_name] = macro_event_multiplier

        total_weight = sum(raw_weights.values()) or 1.0
        normalized_weights = {
            name: weight / total_weight for name, weight in raw_weights.items()
        }

        aggregate_signal = pd.Series(0.0, index=factor_matrix.index)
        for factor_name, signal in weighted_signals.items():
            aggregate_signal = aggregate_signal.add(signal * normalized_weights[factor_name], fill_value=0.0)

        aggregate_signal = np.tanh(aggregate_signal).clip(-1.0, 1.0)
        objective_metrics = self._evaluate_objective(aggregate_signal, target_returns, benchmark_returns)

        aligned_signal = aggregate_signal.dropna()
        latest_score = float(aligned_signal.iloc[-1]) if not aligned_signal.empty else 0.0
        latest_confidence = float(
            min(
                0.99,
                abs(latest_score) * (0.5 + objective_metrics.objective_score),
            )
        )

        return {
            "factor_weights": {
                name: round(weight, 6) for name, weight in normalized_weights.items()
            },
            "state_conditioning": {
                name: round(multiplier, 6) for name, multiplier in state_conditioning.items()
            },
            "cross_asset_transfer": {
                name: round(multiplier, 6) for name, multiplier in transfer_multipliers.items()
            },
            "macro_event_overlay": {
                name: round(multiplier, 6) for name, multiplier in macro_event_multipliers.items()
            },
            "robustness_audit": {
                "democratic_orthogonalization": orthogonalization_audit,
                "effective_factor_breadth": effective_breadth(
                    pd.DataFrame(weighted_signals) if weighted_signals else pd.DataFrame(index=factor_matrix.index)
                ),
                "shapley_deployment_policy": shapley_deployment_policy(
                    model_family="tree_or_ensemble_factor_model",
                    feature_count=len(selected_names),
                    frequency="weekly",
                ),
            },
            "objective_metrics": objective_metrics,
            "latest_signal_score": round(latest_score, 6),
            "latest_confidence": round(latest_confidence, 6),
        }

    def _state_conditioning_multiplier(self, factor_name: str, decoder_audit: Dict[str, Any]) -> float:
        probs = decoder_audit.get("state_probabilities", {}) if isinstance(decoder_audit, dict) else {}
        metadata = decoder_audit.get("audit_metadata", {}) if isinstance(decoder_audit, dict) else {}
        sub_probs = metadata.get("sub_state_probabilities", {}) if isinstance(metadata, dict) else {}
        risk_on = float(probs.get("risk_on", 1.0 / 3.0) or 0.0)
        risk_off = float(probs.get("risk_off", 1.0 / 3.0) or 0.0)
        transition = float(probs.get("transition_choppy", 1.0 / 3.0) or 0.0)
        trend = float(sub_probs.get("trend", 1.0 / 3.0) or 0.0)
        mean_reversion = float(sub_probs.get("mean_reversion", 1.0 / 3.0) or 0.0)
        liquidity_stress = float(sub_probs.get("liquidity_stress", 1.0 / 3.0) or 0.0)
        entropy = float(decoder_audit.get("state_entropy", 1.0) or 1.0) if isinstance(decoder_audit, dict) else 1.0
        transition_stability = float(decoder_audit.get("transition_stability", 0.0) or 0.0) if isinstance(decoder_audit, dict) else 0.0
        strength = float(self.objective_config.state_conditioning_strength)
        lower_name = factor_name.lower()

        if any(token in lower_name for token in ["momentum", "ret_", "macd", "trend", "noisy_channel_long"]):
            regime_score = risk_on + trend - 2.0 / 3.0
        elif any(token in lower_name for token in ["drawdown", "vol", "atr", "risk_off", "noisy_channel_short", "tail"]):
            regime_score = risk_off + liquidity_stress - 2.0 / 3.0
        elif any(token in lower_name for token in ["rsi", "mean_reversion", "reversal"]):
            regime_score = transition + mean_reversion - 2.0 / 3.0
        elif lower_name.startswith("kernel_"):
            regime_score = transition_stability - 0.50
        else:
            regime_score = 0.25 * (risk_on - risk_off)

        uncertainty_haircut = 1.0 - 0.35 * float(np.clip(entropy, 0.0, 1.0))
        multiplier = (1.0 + strength * regime_score) * uncertainty_haircut
        return round(float(np.clip(multiplier, 0.25, 1.75)), 6)

    def _macro_event_factor_multiplier(self, factor_name: str, symbol: str, market_context: Dict[str, Any]) -> float:
        state = market_context.get("macro_event_state", {})
        if not isinstance(state, dict):
            return 1.0
        overlays = state.get("factor_weight_overlays", {})
        if not isinstance(overlays, dict):
            return 1.0

        lower_name = factor_name.lower()
        symbol_upper = symbol.upper()
        symbol_tags = market_context.get("symbol_tags", {})
        tags = []
        if isinstance(symbol_tags, dict):
            raw_tags = symbol_tags.get(symbol) or symbol_tags.get(symbol_upper) or []
            if isinstance(raw_tags, str):
                tags = [raw_tags.lower()]
            elif isinstance(raw_tags, list):
                tags = [str(item).lower() for item in raw_tags]

        multiplier = 1.0
        is_momentum = any(token in lower_name for token in ["momentum", "ret_", "macd", "trend", "noisy_channel_long"])
        is_rate_sensitive = any(token in lower_name for token in ["rate", "duration", "valuation", "discount", "pe_", "pb_", "real_rate"])
        is_earnings_driven = any(token in lower_name for token in ["earnings", "eps", "profit", "margin", "roic", "revenue", "growth"])
        is_volatility_or_tail = any(token in lower_name for token in ["vol", "atr", "drawdown", "risk_off", "tail", "sde_", "hmm_state_entropy"])
        is_fx_sensitive = any(token in lower_name for token in ["fx", "dxy", "usd", "cny", "usdcnh", "usdcny"])
        is_ai_small_cap_proxy = (
            symbol_upper.startswith("IM")
            or "csi1000" in tags
            or "small_cap" in tags
            or "ai_industrial_chain" in tags
        )

        if is_momentum and is_ai_small_cap_proxy:
            multiplier *= float(overlays.get("ai_small_cap_momentum_multiplier", 1.0) or 1.0)
        if is_rate_sensitive:
            multiplier *= float(overlays.get("rate_sensitive_multiplier", 1.0) or 1.0)
        if is_earnings_driven:
            multiplier *= float(overlays.get("earnings_driven_multiplier", 1.0) or 1.0)
        if is_volatility_or_tail:
            multiplier *= float(overlays.get("volatility_multiplier", 1.0) or 1.0)
        if is_fx_sensitive:
            multiplier *= float(overlays.get("fx_cny_resilience_multiplier", 1.0) or 1.0)
        return round(float(np.clip(multiplier, 0.35, 1.75)), 6)

    def _evaluate_objective(
        self,
        signal_scores: pd.Series,
        target_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> ObjectiveMetrics:
        aligned = pd.concat(
            [
                pd.to_numeric(signal_scores, errors="coerce").rename("signal"),
                pd.to_numeric(target_returns, errors="coerce").rename("target"),
            ],
            axis=1,
        ).dropna()
        if benchmark_returns is not None:
            benchmark = self._align_series(benchmark_returns, aligned.index)
            aligned = aligned.join(benchmark.rename("benchmark"), how="left")
        else:
            aligned["benchmark"] = aligned["target"].abs()

        positions = np.sign(aligned["signal"])
        positions[np.abs(aligned["signal"]) < self.selection_policy.signal_threshold] = 0.0
        trade_returns = positions * aligned["target"]
        trades = trade_returns[positions != 0]

        if trades.empty:
            return ObjectiveMetrics(
                trade_count=0,
                win_rate=0.0,
                payoff_ratio=0.0,
                elasticity=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                avg_trade_magnitude=0.0,
                avg_benchmark_magnitude=0.0,
                objective_score=0.0,
            )

        wins = trades[trades > 0]
        losses = trades[trades < 0]
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(abs(losses.mean())) if not losses.empty else 0.0
        payoff_ratio = float(avg_win / avg_loss) if avg_loss > 0 else (10.0 if avg_win > 0 else 0.0)
        benchmark_slice = aligned.loc[trades.index, "benchmark"].abs()
        avg_trade_mag = float(trades.abs().mean())
        avg_benchmark_mag = float(benchmark_slice.mean()) if float(benchmark_slice.mean()) > 0 else 1e-6
        elasticity = float(avg_trade_mag / avg_benchmark_mag)
        win_rate = float((trades > 0).mean())

        payoff_norm = 1 - np.exp(-min(payoff_ratio, 10.0))
        elasticity_norm = 1 - np.exp(-min(elasticity, 10.0))
        objective_score = (
            self.objective_config.win_rate_weight * win_rate
            + self.objective_config.payoff_weight * payoff_norm
            + self.objective_config.elasticity_weight * elasticity_norm
        )

        return ObjectiveMetrics(
            trade_count=int(len(trades)),
            win_rate=round(win_rate, 6),
            payoff_ratio=round(payoff_ratio, 6),
            elasticity=round(elasticity, 6),
            avg_win=round(avg_win, 6),
            avg_loss=round(avg_loss, 6),
            avg_trade_magnitude=round(avg_trade_mag, 6),
            avg_benchmark_magnitude=round(avg_benchmark_mag, 6),
            objective_score=round(float(objective_score), 6),
        )

    def _build_benchmark_returns(
        self,
        symbol_datasets: Dict[str, pd.DataFrame],
        benchmark_frame: Optional[pd.DataFrame],
    ) -> pd.Series:
        if benchmark_frame is not None and not benchmark_frame.empty:
            normalized = self._normalize_ohlcv_frame(benchmark_frame)
            return normalized["close"].pct_change(self.selection_policy.target_horizon).fillna(0.0)

        returns = []
        for frame in symbol_datasets.values():
            normalized = self._normalize_ohlcv_frame(frame)
            returns.append(normalized["close"].pct_change(self.selection_policy.target_horizon))
        if not returns:
            return pd.Series(dtype=float)
        combined = pd.concat(returns, axis=1)
        return combined.mean(axis=1).fillna(0.0)

    def _univariate_factor_regression(self, factor: pd.Series, target: pd.Series) -> Dict[str, float]:
        aligned = pd.concat([factor.rename("factor"), target.rename("target")], axis=1).dropna()
        if len(aligned) < 3:
            return {"slope": 0.0, "r_squared": 0.0, "correlation": 0.0}
        x = aligned["factor"].to_numpy(dtype=float)
        y = aligned["target"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        y_hat = design @ coef
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot else 0.0
        corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
        return {"slope": float(coef[-1]), "r_squared": max(0.0, float(r_squared)), "correlation": corr}

    def _factor_regression_with_benchmark(
        self,
        factor: pd.Series,
        target: pd.Series,
        benchmark: pd.Series,
    ) -> Dict[str, float]:
        aligned = pd.concat(
            [factor.rename("factor"), target.rename("target"), benchmark.rename("benchmark")],
            axis=1,
        ).dropna()
        if len(aligned) < 5:
            return {"slope": 0.0, "r_squared": 0.0, "correlation": 0.0}

        y = aligned["target"].to_numpy(dtype=float)
        base = np.column_stack([np.ones(len(aligned)), aligned["benchmark"].to_numpy(dtype=float)])
        full = np.column_stack(
            [np.ones(len(aligned)), aligned["benchmark"].to_numpy(dtype=float), aligned["factor"].to_numpy(dtype=float)]
        )

        base_coef, *_ = np.linalg.lstsq(base, y, rcond=None)
        full_coef, *_ = np.linalg.lstsq(full, y, rcond=None)
        base_hat = base @ base_coef
        full_hat = full @ full_coef

        ss_tot = float(np.sum((y - y.mean()) ** 2))
        base_r2 = 1.0 - float(np.sum((y - base_hat) ** 2)) / ss_tot if ss_tot else 0.0
        full_r2 = 1.0 - float(np.sum((y - full_hat) ** 2)) / ss_tot if ss_tot else 0.0
        incremental_r2 = max(0.0, full_r2 - base_r2)
        corr = float(np.corrcoef(aligned["factor"], aligned["target"])[0, 1])
        return {
            "slope": float(full_coef[-1]),
            "r_squared": float(incremental_r2),
            "correlation": corr if np.isfinite(corr) else 0.0,
        }

    def _factor_financial_meaning(self, factor_name: str) -> str:
        metadata = self.factor_library.get_factor_metadata(factor_name)
        if metadata.get("financial_meaning"):
            return str(metadata["financial_meaning"])
        factor_lower = factor_name.lower()
        if factor_lower.startswith("rsi"):
            return "RSI 因子，衡量短期超买超卖与均值回复强度。"
        if factor_lower.startswith("sma") or factor_lower.startswith("ema"):
            return "均线因子，衡量价格相对趋势位置与趋势延续性。"
        if "macd" in factor_lower:
            return "MACD 因子，衡量趋势加速度与动量切换。"
        if "vol" in factor_lower or "atr" in factor_lower:
            return "波动率因子，衡量不确定性、风险溢价与仓位弹性。"
        if "volume" in factor_lower:
            return "成交量因子，衡量资金参与度与价格发现效率。"
        if factor_lower.startswith("base_"):
            return "基础市场统计因子，直接从收益、均线距离、成交量异常或回撤中提炼。"
        if factor_lower.startswith("invariance_"):
            return "不变性因子，刻画尺度变换下稳定的价格形状、跨市场残差或协方差结构。"
        if factor_lower.startswith("hmm_"):
            return "轻量隐状态解码因子，刻画风险状态、状态熵和制度切换稳定性。"
        if factor_lower.startswith("kernel_"):
            return "核相似历史类比因子，用当前市场形状匹配过去相似窗口的后验收益分布。"
        if factor_lower.startswith("sde_"):
            return "SDE扩散近似风险因子，用漂移、波动和尾部损失概率刻画状态切换和尾部保护需求。"
        if factor_lower.startswith("noisy_channel_"):
            return "有损信道后验因子，从噪声观测中解码 LONG/SHORT/HOLD 的概率。"
        if factor_lower.startswith("event_intensity_"):
            return "事件强度因子，把新闻/政策/地缘叙事按相关度、情绪、关键词权重和时间衰减压缩成时序信号。"
        if factor_lower.startswith("event_zscore_"):
            return "事件强度滚动Z分数，衡量当前事件热度相对历史窗口的异常程度，并使用t-1窗口避免前视偏误。"
        if factor_lower.startswith("event_momentum_"):
            return "事件强度动量因子，衡量事件热度在短窗口内是否继续升温或降温。"
        if factor_lower.startswith("event_decay_pressure_"):
            return "事件衰减压力因子，衡量旧事件热度残留，防止过期叙事继续驱动新仓位。"
        if factor_lower.startswith("event_asset_"):
            return "事件-资产暴露因子，把事件域映射到受影响资产链条，用于跨资产风控和仓位缩放。"
        return f"可解释技术/因果因子: {factor_name}"

    def _factor_formula(self, factor_name: str) -> str:
        metadata = self.factor_library.get_factor_metadata(factor_name)
        if metadata.get("formula"):
            return str(metadata["formula"])
        factor_lower = factor_name.lower()
        if factor_lower == "base_ret_5":
            return "close_t / close_t-5 - 1"
        if factor_lower == "base_ret_20":
            return "close_t / close_t-20 - 1"
        if factor_lower == "base_price_vs_ma20":
            return "close_t / MA20_t - 1"
        if factor_lower == "base_volume_surprise_20":
            return "volume_t / mean(volume,20)_t - 1"
        if factor_lower == "base_drawdown_20":
            return "close_t / rolling_max(close,20)_t - 1"
        decoder_formulas = {
            "invariance_vol_norm_ret_1": "return_1d / realized_vol_20",
            "invariance_vol_norm_ret_5": "return_5d / (realized_vol_20 * sqrt(5))",
            "invariance_vol_norm_ret_20": "return_20d / (realized_vol_20 * sqrt(20))",
            "invariance_peer_beta_residual_20": "return_1d - rolling_beta_20 * peer_return_1d",
            "invariance_cov_eigen_ratio_60": "max_eigenvalue(cov(local, peer, 60)) / sum_abs_eigenvalues",
            "hmm_prob_risk_on": "GaussianHMM posterior P(risk_on | invariant_observations)",
            "hmm_prob_risk_off": "GaussianHMM posterior P(risk_off | invariant_observations)",
            "hmm_state_entropy": "normalized entropy of latest HMM state posterior",
            "hmm_sub_prob_trend": "sub-state posterior P(trend | invariant_observations)",
            "hmm_sub_prob_mean_reversion": "sub-state posterior P(mean_reversion | invariant_observations)",
            "hmm_sub_prob_liquidity_stress": "sub-state posterior P(liquidity_stress | invariant_observations)",
            "kernel_analog_forward_mean": "mean forward return of nearest invariant-history analogs",
            "kernel_analog_hit_rate": "positive return share of nearest invariant-history analogs",
            "kernel_analog_tail_loss_rate": "share of nearest invariant-history analogs with forward return below tail threshold",
            "sde_drift_20": "rolling_mean(return_1d,20)",
            "sde_volatility_20": "rolling_std(return_1d,20)",
            "sde_downside_q05": "GBM-like 5% downside quantile over target_horizon",
            "sde_upside_q95": "GBM-like 95% upside quantile over target_horizon",
            "sde_tail_loss_probability": "P(forward_return <= tail_loss_threshold) under local diffusion approximation",
            "sde_regime_switch_pressure": "clipped blend of volatility pressure, drawdown and SDE tail probability",
            "noisy_channel_long_posterior": "P(LONG | HMM state, kernel analog, invariant momentum)",
            "noisy_channel_short_posterior": "P(SHORT | HMM state, kernel analog, invariant momentum)",
        }
        if factor_lower in decoder_formulas:
            return decoder_formulas[factor_lower]
        if factor_lower.startswith("event_intensity_"):
            return "sum(relevance_i * abs(sentiment_i or 0.15) * keyword_weight_i * exp(-lambda * age_days_i))"
        if factor_lower.startswith("event_zscore_"):
            return "(EventIntensity_t - mean(EventIntensity_{t-L:t-1})) / std(EventIntensity_{t-L:t-1})"
        if factor_lower.startswith("event_momentum_"):
            return "EventIntensity_t - EventIntensity_{t-5}"
        if factor_lower.startswith("event_decay_pressure_"):
            return "EventIntensity_t / rolling_max(EventIntensity_{t-L:t-1})"
        if factor_lower.startswith("event_asset_"):
            return "sum(abs(event_zscore_domain) * 0.70 + event_decay_pressure_domain * 0.30) * asset_exposure_weight"
        return factor_name

    def _rejection_reason(self, rs_score: float, r_squared: float) -> str:
        reasons = []
        if rs_score < self.selection_policy.min_rs_score:
            reasons.append(f"RS={rs_score:.2f} < {self.selection_policy.min_rs_score:.0f}")
        if r_squared < self.selection_policy.min_r_squared:
            reasons.append(f"R2={r_squared:.4f} < {self.selection_policy.min_r_squared:.2f}")
        return " / ".join(reasons) if reasons else "unknown"

    def _infer_asset_type(self, symbol: str) -> str:
        token = symbol.upper()
        if any(
            key in token
            for key in ["XAU", "GOLD", "COPPER", "CU", "AU", "AG", "RB", "HC", "IF", "IH", "IC", "IM", "COMEX", "LME"]
        ):
            return "futures"
        return "stock"

    def _allocate_signal_weights(
        self,
        candidates: List[Dict[str, Any]],
        active_weight: float,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> List[SignalAllocation]:
        market_context = market_context or {}
        kelly_fractions = [self._fractional_kelly_fraction(item) for item in candidates]
        raw_scores = np.array(
            [
                max(item["objective_score"], 1e-6)
                * (0.5 + item["confidence"])
                * abs(item["raw_score"])
                * max(kelly_fraction, 0.002)
                * float(item.get("abstention_weight_multiplier", 1.0) or 1.0)
                for item, kelly_fraction in zip(candidates, kelly_fractions)
            ],
            dtype=float,
        )
        if float(raw_scores.sum()) <= 0:
            return []
        raw_scores = raw_scores / raw_scores.sum()
        allocations: List[SignalAllocation] = []
        for item, base_weight, kelly_fraction in zip(candidates, raw_scores.tolist(), kelly_fractions):
            confidence = float(item["confidence"])
            stop_loss = self.constraints.base_stop_loss_pct * (1.10 - min(confidence, 0.9) * 0.20)
            take_profit = self.constraints.base_take_profit_pct * (1.0 + confidence)
            execution_profile = self._execution_profile(item["symbol"], market_context)
            capacity_weight_limit = self._capacity_weight_limit(item["symbol"], market_context)
            abstention_multiplier = float(item.get("abstention_weight_multiplier", 1.0) or 1.0)
            target_weight = min(active_weight * base_weight * abstention_multiplier, max(kelly_fraction, 0.0))
            state_multipliers = list((item.get("state_conditioning") or {}).values())
            transfer_multipliers = item.get("cross_asset_transfer") or {}
            macro_event_multipliers = item.get("macro_event_overlay") or {}
            allocations.append(
                SignalAllocation(
                    symbol=item["symbol"],
                    direction=item["direction"],
                    asset_type=item["asset_type"],
                    target_weight=round(target_weight, 6),
                    raw_score=round(float(item["raw_score"]), 6),
                    confidence=round(confidence, 6),
                    objective_score=round(float(item["objective_score"]), 6),
                    selected_features=list(item["selected_features"]),
                    stop_loss_pct=round(stop_loss, 6),
                    take_profit_pct=round(take_profit, 6),
                    decoder_state_entropy=round(float(item.get("decoder_state_entropy", 1.0) or 1.0), 6),
                    decoder_risk_off_probability=round(float(item.get("decoder_risk_off_probability", 0.0) or 0.0), 6),
                    decoder_sde_tail_loss_probability=round(
                        float(item.get("decoder_sde_tail_loss_probability", 0.0) or 0.0), 6
                    ),
                    decoder_sde_downside_q05=round(float(item.get("decoder_sde_downside_q05", 0.0) or 0.0), 6),
                    kernel_tail_loss_rate=round(float(item.get("kernel_tail_loss_rate", 0.0) or 0.0), 6),
                    kelly_fraction=round(kelly_fraction, 6),
                    capacity_weight_limit=round(capacity_weight_limit, 6),
                    commission_bps=round(float(execution_profile["commission_bps"]), 6),
                    slippage_bps=round(float(execution_profile["slippage_bps"]), 6),
                    impact_bps=round(float(execution_profile["impact_bps"]), 6),
                    state_conditioning_multiplier=round(float(np.mean(state_multipliers)), 6) if state_multipliers else 1.0,
                    cross_asset_transfer_multiplier=round(max(transfer_multipliers.values()), 6)
                    if transfer_multipliers
                    else 1.0,
                    transferred_factor_count=sum(1 for value in transfer_multipliers.values() if float(value) > 1.0),
                    macro_event_overlay_multiplier=round(float(np.mean(list(macro_event_multipliers.values()))), 6)
                    if macro_event_multipliers
                    else 1.0,
                    abstention_decision=str(item.get("abstention_decision", ALLOW)),
                    abstention_risk_score=round(float(item.get("abstention_risk_score", 0.0) or 0.0), 6),
                    abstention_reasons=list(item.get("abstention_reasons", [])),
                )
            )
        return allocations

    def _fractional_kelly_fraction(self, candidate: Dict[str, Any]) -> float:
        metrics = candidate.get("objective_metrics") or {}
        objective_score = float(candidate.get("objective_score", 0.0) or 0.0)
        win_rate = float(metrics.get("win_rate", 0.50 + 0.25 * min(max(objective_score, 0.0), 1.0)) or 0.0)
        payoff_ratio = float(metrics.get("payoff_ratio", 1.0 + 2.0 * min(max(objective_score, 0.0), 1.0)) or 0.0)
        elasticity = float(metrics.get("elasticity", 1.0) or 0.0)
        win_rate = float(np.clip(win_rate, 0.0, 0.99))
        payoff_ratio = max(payoff_ratio, 1e-6)
        full_kelly = max(0.0, win_rate - (1.0 - win_rate) / payoff_ratio)
        elasticity_scale = float(np.clip(elasticity / 2.0, 0.10, 1.0))
        confidence = float(np.clip(candidate.get("confidence", 0.0) or 0.0, 0.0, 1.0))
        entropy = float(np.clip(candidate.get("decoder_state_entropy", 1.0) or 1.0, 0.0, 1.0))
        risk_off = float(np.clip(candidate.get("decoder_risk_off_probability", 0.0) or 0.0, 0.0, 1.0))
        direction = str(candidate.get("direction", "long")).lower()
        risk_haircut = 1.0 - (0.45 * risk_off if direction == "long" else 0.20 * risk_off)
        uncertainty_haircut = 1.0 - 0.50 * entropy
        fractional = (
            full_kelly
            * self.constraints.max_fractional_kelly
            * elasticity_scale
            * (0.50 + 0.50 * confidence)
            * risk_haircut
            * uncertainty_haircut
        )
        return round(float(np.clip(fractional, 0.0, self.constraints.max_single_weight)), 6)

    def _enforce_portfolio_constraints(
        self,
        allocations: List[SignalAllocation],
    ) -> tuple[List[SignalAllocation], float]:
        constrained: List[SignalAllocation] = []
        futures_weight = 0.0
        residual_cash = 0.0
        for allocation in allocations:
            original_weight = allocation.target_weight
            weight = min(allocation.target_weight, self.constraints.max_single_weight, allocation.capacity_weight_limit)
            if allocation.asset_type == "futures":
                available = max(0.0, self.constraints.max_futures_weight - futures_weight)
                if available <= 0:
                    residual_cash += weight
                    continue
                weight = min(weight, available)
                futures_weight += weight
            residual_cash += max(0.0, original_weight - weight)
            allocation.target_weight = round(weight, 6)
            if allocation.target_weight > 0:
                constrained.append(allocation)

        total = sum(item.target_weight for item in constrained)
        if total > self.constraints.max_gross_weight:
            scale = self.constraints.max_gross_weight / total
            for item in constrained:
                item.target_weight = round(item.target_weight * scale, 6)
            total = sum(item.target_weight for item in constrained)
        residual_cash += max(0.0, self.constraints.max_gross_weight - total - 0.0) * 0.0
        return constrained, round(residual_cash, 6)

    def _score_portfolio(
        self,
        allocations: List[SignalAllocation],
        tail_weight: float,
        safe_weight: float,
        tail_risk: float,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> float:
        market_context = market_context or {}
        penalties = self._portfolio_penalties(allocations, tail_weight, market_context)
        if not allocations:
            return safe_weight * 0.02 + tail_weight * max(tail_risk, 0.10) - penalties["tail_risk_penalty"]
        expected_signal_score = sum(item.target_weight * item.objective_score for item in allocations)
        concentration = sum(item.target_weight ** 2 for item in allocations) / max(
            sum(item.target_weight for item in allocations), 1e-6
        )
        effective_tail_risk = max(tail_risk, self._allocation_tail_risk_score(allocations))
        desired_tail_ratio = float(np.clip(0.10 + effective_tail_risk * 0.60, 0.10, 0.45))
        actual_tail_ratio = tail_weight / max(tail_weight + safe_weight, 1e-6)
        hedge_alignment = 1.0 - abs(actual_tail_ratio - desired_tail_ratio)
        return float(
            expected_signal_score
            + 0.08 * hedge_alignment
            - 0.10 * concentration
            - penalties["transaction_cost"]
            - penalties["impact_cost"]
            - penalties["slippage_cost"]
            - penalties["capacity_penalty"]
            - penalties["margin_penalty"]
            - penalties["tail_risk_penalty"]
            - penalties["breadth_penalty"]
        )

    def _portfolio_penalties(
        self,
        allocations: List[SignalAllocation],
        tail_weight: float,
        market_context: Dict[str, Any],
    ) -> Dict[str, float]:
        gross_weight = sum(abs(item.target_weight) for item in allocations)
        futures_weight = sum(abs(item.target_weight) for item in allocations if item.asset_type == "futures")
        turnover_estimate = float(market_context.get("turnover_estimate", gross_weight))
        turnover_scale = turnover_estimate / max(gross_weight, 1e-9) if gross_weight else 0.0
        liquidity_penalty = float(market_context.get("liquidity_penalty", 0.0))
        margin_penalty_rate = float(market_context.get("margin_penalty_rate", 0.018))
        tail_risk = max(self._extract_tail_risk_score(market_context), self._allocation_tail_risk_score(allocations))
        unhedged_tail = max(0.0, tail_risk - tail_weight)
        decoder_uncertainty = sum(abs(item.target_weight) * float(item.decoder_state_entropy) for item in allocations)
        decoder_risk_off = sum(abs(item.target_weight) * float(item.decoder_risk_off_probability) for item in allocations)
        breadth_audit = market_context.get("effective_breadth_audit", {})
        breadth_ratio = float(breadth_audit.get("breadth_ratio", 1.0) or 1.0) if isinstance(breadth_audit, dict) else 1.0
        breadth_penalty = gross_weight * max(0.0, 0.55 - breadth_ratio) * 0.030
        transaction_cost = 0.0
        slippage_cost = 0.0
        impact_cost = liquidity_penalty
        capacity_penalty = 0.0
        for item in allocations:
            traded_weight = abs(item.target_weight) * turnover_scale
            participation = abs(item.target_weight) / max(float(item.capacity_weight_limit), 1e-9)
            transaction_cost += traded_weight * float(item.commission_bps) / 10000.0
            slippage_cost += traded_weight * float(item.slippage_bps) / 10000.0
            impact_cost += traded_weight * float(item.impact_bps) * (1.0 + np.sqrt(max(participation, 0.0))) / 10000.0
            capacity_penalty += abs(item.target_weight) * max(0.0, participation - 0.80) * 0.020
        return {
            "transaction_cost": transaction_cost,
            "impact_cost": impact_cost,
            "slippage_cost": slippage_cost,
            "capacity_penalty": capacity_penalty,
            "margin_penalty": futures_weight * margin_penalty_rate,
            "tail_risk_penalty": unhedged_tail * 0.035 + decoder_uncertainty * 0.006 + decoder_risk_off * 0.010,
            "breadth_penalty": breadth_penalty,
        }

    def _candidate_breadth_audit(
        self,
        candidates: List[Dict[str, Any]],
        market_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Estimate true independent bets instead of trusting nominal count."""

        nominal = len(candidates)
        if nominal <= 1:
            return {
                "status": "single_or_empty",
                "nominal_breadth": nominal,
                "effective_breadth": float(nominal),
                "breadth_ratio": 1.0 if nominal else 0.0,
                "average_signal_overlap": 0.0,
                "gate": "single candidate cannot diversify portfolio breadth",
            }

        correlation_panel = market_context.get("candidate_return_matrix")
        if correlation_panel is None:
            correlation_panel = market_context.get("signal_return_matrix")
        if isinstance(correlation_panel, pd.DataFrame) and not correlation_panel.empty:
            audit = effective_breadth(correlation_panel)
            audit["source"] = "market_context_return_matrix"
        else:
            feature_sets = [set(map(str, item.get("selected_features", []))) for item in candidates]
            overlaps = []
            for i in range(nominal):
                for j in range(i + 1, nominal):
                    union = feature_sets[i] | feature_sets[j]
                    overlap = len(feature_sets[i] & feature_sets[j]) / max(len(union), 1)
                    overlaps.append(overlap)
            avg_overlap = float(np.mean(overlaps)) if overlaps else 0.0
            effective = float(nominal / (1.0 + avg_overlap * (nominal - 1)))
            audit = {
                "status": "ready",
                "source": "selected_feature_jaccard_overlap",
                "nominal_breadth": nominal,
                "average_signal_overlap": round(avg_overlap, 6),
                "effective_breadth": round(effective, 6),
                "breadth_ratio": round(float(effective / max(nominal, 1)), 6),
                "formula": "N / (1 + avg_feature_overlap * (N - 1))",
            }
        audit["breadth_gate"] = (
            "ALLOW" if float(audit.get("breadth_ratio", 0.0) or 0.0) >= 0.55 else "PENALIZE_ACTIVE_RISK"
        )
        audit["shapley_policy"] = shapley_deployment_policy(
            model_family="tree_or_ensemble_factor_model",
            feature_count=sum(len(item.get("selected_features", [])) for item in candidates),
            frequency="weekly",
        )
        return audit

    def _allocation_tail_risk_score(self, allocations: List[SignalAllocation]) -> float:
        if not allocations:
            return 0.0
        weighted_tail = 0.0
        total_weight = 0.0
        for item in allocations:
            weight = abs(item.target_weight)
            tail_pressure = max(
                float(item.decoder_sde_tail_loss_probability),
                float(item.kernel_tail_loss_rate),
                min(abs(float(item.decoder_sde_downside_q05)) * 5.0, 1.0),
            )
            weighted_tail += weight * tail_pressure
            total_weight += weight
        return float(np.clip(weighted_tail / max(total_weight, 1e-9), 0.0, 0.80))

    def _execution_profile(self, symbol: str, market_context: Dict[str, Any]) -> Dict[str, float]:
        profiles = market_context.get("execution_costs", {})
        profile = {}
        if isinstance(profiles, dict):
            profile = profiles.get(symbol) or profiles.get("default") or {}
        return {
            "commission_bps": float(profile.get("commission_bps", market_context.get("transaction_cost_bps", 8.0)) or 0.0),
            "slippage_bps": float(profile.get("slippage_bps", market_context.get("slippage_bps", 0.0)) or 0.0),
            "impact_bps": float(profile.get("impact_bps", market_context.get("impact_cost_bps", 12.0)) or 0.0),
        }

    def _capacity_weight_limit(self, symbol: str, market_context: Dict[str, Any]) -> float:
        capacity_map = market_context.get("capacity") or market_context.get("capacity_limits") or {}
        record = capacity_map.get(symbol, {}) if isinstance(capacity_map, dict) else {}
        if "max_weight" in record:
            return float(np.clip(float(record["max_weight"]), 0.0, 1.0))
        portfolio_notional = float(record.get("portfolio_notional", market_context.get("portfolio_notional", self.constraints.default_portfolio_notional)) or 1.0)
        adv_notional = float(record.get("adv_notional", 0.0) or 0.0)
        participation = float(record.get("max_participation_rate", self.constraints.default_max_participation_rate) or 0.0)
        if adv_notional <= 0 or portfolio_notional <= 0 or participation <= 0:
            return 1.0
        return float(np.clip((adv_notional * participation) / portfolio_notional, 0.0, 1.0))

    def _extract_tail_risk_score(self, market_context: Dict[str, Any]) -> float:
        crisis_probability = float(market_context.get("crisis_probability", 0.15))
        game_analysis = market_context.get("game_causal_analysis", {})
        if isinstance(game_analysis, dict):
            crisis_probability = max(
                crisis_probability,
                float(game_analysis.get("aggregate_risk_score", 0.0) or 0.0),
                float(
                    game_analysis.get("risk_scores", {})
                    .get("geopolitical_energy", {})
                    .get("score", 0.0)
                    or 0.0
                ),
            )
        scm_stress = market_context.get("scm_counterfactual_stress", {})
        if isinstance(scm_stress, dict):
            crisis_probability = max(
                crisis_probability,
                float(scm_stress.get("max_tail_risk_score", 0.0) or 0.0),
                min(0.80, max(0.0, float(scm_stress.get("max_tail_hedge_multiplier", 1.0) or 1.0) - 1.0)),
            )
        macro_event_state = market_context.get("macro_event_state", {})
        if isinstance(macro_event_state, dict):
            crisis_probability = max(
                crisis_probability,
                float(macro_event_state.get("tail_risk_score", 0.0) or 0.0),
                min(0.80, max(0.0, float(macro_event_state.get("tail_hedge_multiplier", 1.0) or 1.0) - 1.0)),
            )
        regime = market_context.get("regime") or market_context.get("cross_asset_regime", {}).get("regime", "")
        if isinstance(regime, dict):
            regime = regime.get("regime", "")
        if str(regime).lower() in {"crisis", "liquidity_stress", "bear"}:
            crisis_probability = max(crisis_probability, 0.35)
        return float(np.clip(crisis_probability, 0.05, 0.80))

    def _serialize_portfolio_plan(self, plan: PortfolioPlan) -> Dict[str, Any]:
        return {
            "active_weight": plan.active_weight,
            "safe_weight": plan.safe_weight,
            "tail_hedge_weight": plan.tail_hedge_weight,
            "projected_objective_score": plan.projected_objective_score,
            "barbell_ratio": plan.barbell_ratio,
            "residual_cash_weight": plan.residual_cash_weight,
            "estimated_cost_penalty": plan.estimated_cost_penalty,
            "estimated_impact_penalty": plan.estimated_impact_penalty,
            "estimated_slippage_penalty": plan.estimated_slippage_penalty,
            "estimated_capacity_penalty": plan.estimated_capacity_penalty,
            "estimated_margin_penalty": plan.estimated_margin_penalty,
            "estimated_tail_risk_penalty": plan.estimated_tail_risk_penalty,
            "estimated_breadth_penalty": plan.estimated_breadth_penalty,
            "hmm_barbell_state": plan.hmm_barbell_state,
            "hmm_barbell_audit": plan.hmm_barbell_audit,
            "robustness_audit": plan.robustness_audit,
            "signal_allocations": [asdict(item) for item in plan.signal_allocations],
        }

    def _portfolio_to_actions(self, plan: PortfolioPlan) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        for allocation in plan.signal_allocations:
            actions.append(
                {
                    "action": "LONG" if allocation.direction == "long" else "SHORT",
                    "symbol": allocation.symbol,
                    "target_weight": allocation.target_weight,
                    "confidence": allocation.confidence,
                    "objective_score": allocation.objective_score,
                    "stop_loss_pct": allocation.stop_loss_pct,
                    "take_profit_pct": allocation.take_profit_pct,
                    "max_hold_days": self.constraints.max_hold_days,
                    "no_weekend_hold": self.constraints.no_weekend_hold,
                    "selected_features": allocation.selected_features,
                    "decoder_state_entropy": allocation.decoder_state_entropy,
                    "decoder_risk_off_probability": allocation.decoder_risk_off_probability,
                    "decoder_sde_tail_loss_probability": allocation.decoder_sde_tail_loss_probability,
                    "decoder_sde_downside_q05": allocation.decoder_sde_downside_q05,
                    "kernel_tail_loss_rate": allocation.kernel_tail_loss_rate,
                    "kelly_fraction": allocation.kelly_fraction,
                    "capacity_weight_limit": allocation.capacity_weight_limit,
                    "execution_cost_assumption": {
                        "commission_bps": allocation.commission_bps,
                        "slippage_bps": allocation.slippage_bps,
                        "impact_bps": allocation.impact_bps,
                    },
                    "state_conditioning_multiplier": allocation.state_conditioning_multiplier,
                    "cross_asset_transfer_multiplier": allocation.cross_asset_transfer_multiplier,
                    "transferred_factor_count": allocation.transferred_factor_count,
                    "macro_event_overlay_multiplier": allocation.macro_event_overlay_multiplier,
                    "abstention_gate": {
                        "decision": allocation.abstention_decision,
                        "risk_score": allocation.abstention_risk_score,
                        "reasons": allocation.abstention_reasons,
                    },
                    "hmm_barbell_state": plan.hmm_barbell_state,
                }
            )
        if plan.tail_hedge_weight > 0:
            actions.append(
                {
                    "action": "TAIL_HEDGE",
                    "symbol": "TAIL_RISK_PROTECTION",
                    "target_weight": plan.tail_hedge_weight,
                    "reason": "塔勒布杠铃尾部保护与主信号联合优化",
                    "hmm_barbell_state": plan.hmm_barbell_state,
                }
            )
        if plan.safe_weight > 0:
            actions.append(
                {
                    "action": "SAFE_RESERVE",
                    "symbol": "SAFE_ASSET_BUCKET",
                    "target_weight": plan.safe_weight,
                    "reason": "杠铃安全端与残余现金缓冲",
                    "hmm_barbell_state": plan.hmm_barbell_state,
                }
            )
        return actions

    @staticmethod
    def _align_series(series: Optional[pd.Series], index: pd.Index) -> pd.Series:
        if series is None:
            return pd.Series(0.0, index=index, dtype=float)
        aligned = pd.Series(series).reset_index(drop=True)
        aligned = pd.to_numeric(aligned, errors="coerce")
        if len(aligned) >= len(index):
            aligned = aligned.iloc[-len(index):].reset_index(drop=True)
        else:
            aligned = aligned.reindex(range(len(index)))
        aligned.index = index
        return aligned.ffill().fillna(0.0)
