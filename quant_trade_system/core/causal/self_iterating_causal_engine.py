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
from .causal_factor_library import AssetClass, CausalFactorLibrary
from .invariance_market_decoder import InvarianceMarketDecoder, InvariantDecoderConfig
from .research_governance import (
    CausalValidationLoop,
    ExperimentRegistry,
    FeatureStore,
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


@dataclass
class LearningObjectiveConfig:
    win_rate_weight: float = 0.45
    payoff_weight: float = 0.35
    elasticity_weight: float = 0.20


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
    estimated_margin_penalty: float = 0.0
    estimated_tail_risk_penalty: float = 0.0


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
        self.invariance_decoder = InvarianceMarketDecoder(invariant_decoder_config)
        self.latest_decoder_snapshots: Dict[str, Dict[str, Any]] = {}
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
            "invariance_decoder": {
                "enabled": True,
                "version": "invariance_decoder_v1",
                "features": ["invariance", "hmm", "kernel_analog", "noisy_channel_posteriors"],
                "dependency_policy": "numpy_pandas_only",
            },
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
                }
                continue

            factor_matrix = self._build_candidate_factor_matrix(
                normalized,
                symbol=symbol,
                peer_frames=peer_frames,
                benchmark_frame=benchmark_frame,
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

            if not selected:
                symbol_reports[symbol] = {
                    "status": "no_feature_passed_threshold",
                    "rows": int(len(normalized)),
                    "global_peer_count": int(len(peer_frames)),
                    "invariance_decoder": decoder_audit,
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
            )
            all_validation_records.extend(validation_records)

            if not tradable_selected:
                symbol_reports[symbol] = {
                    "status": "no_validated_causal_edge",
                    "rows": int(len(normalized)),
                    "global_peer_count": int(len(peer_frames)),
                    "invariance_decoder": decoder_audit,
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
            )
            latest_score = ensemble["latest_signal_score"]
            latest_confidence = ensemble["latest_confidence"]
            if abs(latest_score) >= self.selection_policy.signal_threshold:
                signal_candidates.append(
                    {
                        "symbol": symbol,
                        "asset_type": self._infer_asset_type(symbol),
                        "direction": "long" if latest_score >= 0 else "short",
                        "raw_score": latest_score,
                        "confidence": latest_confidence,
                        "objective_score": ensemble["objective_metrics"].objective_score,
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
                    }
                )

            symbol_reports[symbol] = {
                "status": "trained",
                "rows": int(len(normalized)),
                "global_peer_count": int(len(peer_frames)),
                "invariance_decoder": decoder_audit,
                "selected_features": [asdict(item) for item in selected],
                "tradable_feature_count": int(len(tradable_selected)),
                "rejected_features": [asdict(item) for item in rejected[:15]],
                "causal_validation": validation_records,
                "factor_weights": ensemble["factor_weights"],
                "objective_metrics": asdict(ensemble["objective_metrics"]),
                "latest_signal_score": latest_score,
                "latest_confidence": latest_confidence,
            }

        portfolio_plan = self.optimize_portfolio(signal_candidates, market_context or {})
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

            raw_rows.append(
                {
                    "factor_name": factor_name,
                    "r_squared": regression["r_squared"],
                    "correlation": regression["correlation"],
                    "slope": regression["slope"],
                    "financial_meaning": self._factor_financial_meaning(factor_name),
                    "formula": self._factor_formula(factor_name),
                    "predictive_power": abs(regression["correlation"]) * (1.0 + regression["r_squared"]),
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
                rejection_reason=None if passes else self._rejection_reason(rs_score, row["r_squared"]),
            )
            self.factor_library.update_factor_metadata(
                row["factor_name"],
                {
                    "latest_rs_score": round(float(rs_score), 6),
                    "latest_r_squared": round(float(row["r_squared"]), 6),
                    "latest_correlation": round(float(row["correlation"]), 6),
                    "selected_in_latest_cycle": passes,
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
        candidates = sorted(
            signal_candidates,
            key=lambda item: (item["objective_score"], abs(item["raw_score"]), item["confidence"]),
            reverse=True,
        )[: self.constraints.max_positions]
        if not candidates:
            return PortfolioPlan(
                active_weight=0.0,
                safe_weight=0.90,
                tail_hedge_weight=0.10,
                signal_allocations=[],
                projected_objective_score=0.0,
                barbell_ratio=0.10,
                residual_cash_weight=0.0,
            )

        tail_risk = self._extract_tail_risk_score(market_context)
        best_plan: Optional[PortfolioPlan] = None
        best_score = -np.inf

        for active_weight in [0.35, 0.45, 0.55, 0.65, 0.75]:
            barbell_budget = max(0.0, 1.0 - active_weight)
            for hedge_ratio in [0.10, 0.20, 0.30, 0.40]:
                tail_weight = barbell_budget * hedge_ratio
                safe_weight = barbell_budget - tail_weight
                allocations = self._allocate_signal_weights(candidates, active_weight)
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
                        barbell_ratio=round(hedge_ratio, 6),
                        residual_cash_weight=0.0,
                        estimated_cost_penalty=round(penalties["transaction_cost"], 6),
                        estimated_impact_penalty=round(penalties["impact_cost"], 6),
                        estimated_margin_penalty=round(penalties["margin_penalty"], 6),
                        estimated_tail_risk_penalty=round(penalties["tail_risk_penalty"], 6),
                    )

        return best_plan or PortfolioPlan(
            active_weight=0.0,
            safe_weight=0.90,
            tail_hedge_weight=0.10,
            signal_allocations=[],
            projected_objective_score=0.0,
            barbell_ratio=0.10,
            residual_cash_weight=0.0,
        )

    def _validate_selected_features(
        self,
        symbol: str,
        factor_matrix: pd.DataFrame,
        target_returns: pd.Series,
        selected_features: List[SelectedFeature],
        benchmark_returns: Optional[pd.Series] = None,
    ) -> tuple[List[SelectedFeature], List[Dict[str, Any]]]:
        tradable: List[SelectedFeature] = []
        records: List[Dict[str, Any]] = []
        for feature in selected_features:
            validation = self.causal_validation_loop.validate_feature(
                feature_name=feature.factor_name,
                factor_series=factor_matrix[feature.factor_name],
                target_returns=target_returns,
                benchmark_returns=benchmark_returns,
            )
            feature.identification_status = validation.identification_status
            feature.validation_score = validation.validation_score
            feature.can_trade = validation.can_trade
            record = asdict(validation)
            record["symbol"] = symbol
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
                },
                leakage_check={
                    "forward_target_shift_days": self.selection_policy.target_horizon,
                    "uses_only_information_available_at_or_before_signal_time": True,
                    "purged_cv_required_before_promotion": True,
                },
                validation_status=record,
            )
            if validation.can_trade:
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
                "invariance_decoder": decoder_summary,
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
        matrix = pd.concat([base, global_peer, technical, causal, decoder_features], axis=1)
        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        matrix = matrix.loc[:, matrix.notna().mean() >= self.selection_policy.min_non_null_ratio]
        matrix = matrix.ffill().fillna(0.0)
        return matrix

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
    ) -> Dict[str, Any]:
        weighted_signals: Dict[str, pd.Series] = {}
        raw_weights: Dict[str, float] = {}

        for feature in selected_features:
            series = pd.to_numeric(factor_matrix[feature.factor_name], errors="coerce")
            signal = self.factor_library._zscore(series).fillna(0.0) * feature.direction
            metrics = self._evaluate_objective(signal, target_returns, benchmark_returns)
            feature.objective_score = metrics.objective_score
            raw_weight = max(metrics.objective_score, 1e-6) * (feature.rs_score / 100.0) * feature.r_squared
            weighted_signals[feature.factor_name] = signal
            raw_weights[feature.factor_name] = raw_weight

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
            "objective_metrics": objective_metrics,
            "latest_signal_score": round(latest_score, 6),
            "latest_confidence": round(latest_confidence, 6),
        }

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
        if factor_lower.startswith("noisy_channel_"):
            return "有损信道后验因子，从噪声观测中解码 LONG/SHORT/HOLD 的概率。"
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
            "noisy_channel_long_posterior": "P(LONG | HMM state, kernel analog, invariant momentum)",
            "noisy_channel_short_posterior": "P(SHORT | HMM state, kernel analog, invariant momentum)",
        }
        if factor_lower in decoder_formulas:
            return decoder_formulas[factor_lower]
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
    ) -> List[SignalAllocation]:
        raw_scores = np.array(
            [max(item["objective_score"], 1e-6) * (0.5 + item["confidence"]) * abs(item["raw_score"]) for item in candidates],
            dtype=float,
        )
        raw_scores = raw_scores / raw_scores.sum()
        allocations: List[SignalAllocation] = []
        for item, base_weight in zip(candidates, raw_scores.tolist()):
            confidence = float(item["confidence"])
            stop_loss = self.constraints.base_stop_loss_pct * (1.10 - min(confidence, 0.9) * 0.20)
            take_profit = self.constraints.base_take_profit_pct * (1.0 + confidence)
            allocations.append(
                SignalAllocation(
                    symbol=item["symbol"],
                    direction=item["direction"],
                    asset_type=item["asset_type"],
                    target_weight=round(active_weight * base_weight, 6),
                    raw_score=round(float(item["raw_score"]), 6),
                    confidence=round(confidence, 6),
                    objective_score=round(float(item["objective_score"]), 6),
                    selected_features=list(item["selected_features"]),
                    stop_loss_pct=round(stop_loss, 6),
                    take_profit_pct=round(take_profit, 6),
                    decoder_state_entropy=round(float(item.get("decoder_state_entropy", 1.0) or 1.0), 6),
                    decoder_risk_off_probability=round(float(item.get("decoder_risk_off_probability", 0.0) or 0.0), 6),
                )
            )
        return allocations

    def _enforce_portfolio_constraints(
        self,
        allocations: List[SignalAllocation],
    ) -> tuple[List[SignalAllocation], float]:
        constrained: List[SignalAllocation] = []
        futures_weight = 0.0
        residual_cash = 0.0
        for allocation in allocations:
            weight = min(allocation.target_weight, self.constraints.max_single_weight)
            if allocation.asset_type == "futures":
                available = max(0.0, self.constraints.max_futures_weight - futures_weight)
                if available <= 0:
                    residual_cash += weight
                    continue
                weight = min(weight, available)
                futures_weight += weight
            allocation.target_weight = round(weight, 6)
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
        desired_tail_ratio = float(np.clip(0.10 + tail_risk * 0.60, 0.10, 0.45))
        actual_tail_ratio = tail_weight / max(tail_weight + safe_weight, 1e-6)
        hedge_alignment = 1.0 - abs(actual_tail_ratio - desired_tail_ratio)
        return float(
            expected_signal_score
            + 0.08 * hedge_alignment
            - 0.10 * concentration
            - penalties["transaction_cost"]
            - penalties["impact_cost"]
            - penalties["margin_penalty"]
            - penalties["tail_risk_penalty"]
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
        transaction_bps = float(market_context.get("transaction_cost_bps", 8.0))
        impact_bps = float(market_context.get("impact_cost_bps", 12.0))
        liquidity_penalty = float(market_context.get("liquidity_penalty", 0.0))
        margin_penalty_rate = float(market_context.get("margin_penalty_rate", 0.018))
        tail_risk = self._extract_tail_risk_score(market_context)
        unhedged_tail = max(0.0, tail_risk - tail_weight)
        decoder_uncertainty = sum(abs(item.target_weight) * float(item.decoder_state_entropy) for item in allocations)
        decoder_risk_off = sum(abs(item.target_weight) * float(item.decoder_risk_off_probability) for item in allocations)
        return {
            "transaction_cost": turnover_estimate * transaction_bps / 10000.0,
            "impact_cost": gross_weight * impact_bps / 10000.0 + liquidity_penalty,
            "margin_penalty": futures_weight * margin_penalty_rate,
            "tail_risk_penalty": unhedged_tail * 0.035 + decoder_uncertainty * 0.006 + decoder_risk_off * 0.010,
        }

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
            "estimated_margin_penalty": plan.estimated_margin_penalty,
            "estimated_tail_risk_penalty": plan.estimated_tail_risk_penalty,
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
                }
            )
        if plan.tail_hedge_weight > 0:
            actions.append(
                {
                    "action": "TAIL_HEDGE",
                    "symbol": "TAIL_RISK_PROTECTION",
                    "target_weight": plan.tail_hedge_weight,
                    "reason": "塔勒布杠铃尾部保护与主信号联合优化",
                }
            )
        if plan.safe_weight > 0:
            actions.append(
                {
                    "action": "SAFE_RESERVE",
                    "symbol": "SAFE_ASSET_BUCKET",
                    "target_weight": plan.safe_weight,
                    "reason": "杠铃安全端与残余现金缓冲",
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
