"""
跨资产因果处理引擎。

把共有因子的处理流程程序化：
1. 单位化与波动率标准化
2. 因子正交化与残差化
3. 宏观制度识别与动态相关性
4. 跨资产动量与溢出
5. 因子风险预算与压力测试
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .causal_factor_library import CausalFactorLibrary


class OrthogonalizationMethod(Enum):
    """正交化方法。"""

    PCA = "pca"
    RESIDUAL = "residual"


class CorrelationMethod(Enum):
    """相关性估计方法。"""

    ROLLING = "rolling"


class MacroRegime(Enum):
    """宏观制度。"""

    HIGH_GROWTH_HIGH_INFLATION = "high_growth_high_inflation"
    LOW_GROWTH_HIGH_INFLATION = "low_growth_high_inflation"
    LOW_GROWTH_LOW_INFLATION = "low_growth_low_inflation"
    HIGH_GROWTH_LOW_INFLATION = "high_growth_low_inflation"
    LIQUIDITY_STRESS = "liquidity_stress"


@dataclass
class CrossAssetProcessingConfig:
    """跨资产处理配置。"""

    volatility_window: int = 60
    correlation_window: int = 60
    growth_threshold: float = 0.02
    inflation_threshold: float = 0.03
    liquidity_threshold: float = -0.01
    zscore_clip: float = 3.0


@dataclass
class MacroRegimeSnapshot:
    """宏观制度快照。"""

    regime: MacroRegime
    growth: float
    inflation: float
    liquidity: float
    confidence: float


class CrossAssetCausalEngine:
    """跨资产共有因子的处理引擎。"""

    def __init__(
        self,
        library: Optional[CausalFactorLibrary] = None,
        config: Optional[CrossAssetProcessingConfig] = None,
    ) -> None:
        self.library = library or CausalFactorLibrary()
        self.config = config or CrossAssetProcessingConfig()

    def describe_capabilities(self) -> Dict[str, Any]:
        """返回引擎能力摘要。"""
        return {
            "normalization": ["volatility_scaling", "cross_sectional_zscore"],
            "orthogonalization": [m.value for m in OrthogonalizationMethod],
            "correlation": [m.value for m in CorrelationMethod],
            "regimes": [regime.value for regime in MacroRegime],
            "risk_budget": ["mctr", "scenario_stress_test"],
        }

    def volatility_scale(
        self,
        exposures: pd.DataFrame,
        volatility: pd.DataFrame | pd.Series,
    ) -> pd.DataFrame:
        """对暴露进行波动率调整。"""
        vol = volatility.copy()
        if isinstance(vol, pd.Series):
            vol = pd.DataFrame(
                np.repeat(vol.values.reshape(-1, 1), exposures.shape[1], axis=1),
                index=exposures.index,
                columns=exposures.columns,
            )
        vol = vol.reindex_like(exposures).replace(0, np.nan)
        scaled = exposures.astype(float) / vol.astype(float)
        return scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def cross_sectional_zscore(self, exposures: pd.DataFrame) -> pd.DataFrame:
        """对每个时点做横截面标准化。"""
        centered = exposures.sub(exposures.mean(axis=1), axis=0)
        dispersion = exposures.std(axis=1, ddof=0).replace(0, np.nan)
        zscore = centered.div(dispersion, axis=0).fillna(0.0)
        return zscore.clip(-self.config.zscore_clip, self.config.zscore_clip)

    def unitize_exposures(
        self,
        exposures: pd.DataFrame,
        volatility: Optional[pd.DataFrame | pd.Series] = None,
    ) -> pd.DataFrame:
        """先波动率调整，再做横截面标准化。"""
        normalized = exposures.copy()
        if volatility is not None:
            normalized = self.volatility_scale(normalized, volatility)
        return self.cross_sectional_zscore(normalized)

    def orthogonalize_pca(
        self,
        factor_frame: pd.DataFrame,
        n_components: Optional[int] = None,
    ) -> Dict[str, Any]:
        """用 PCA 提取正交宏观因子。"""
        clean = factor_frame.astype(float).ffill().bfill().fillna(0.0)
        standardized = (clean - clean.mean()) / clean.std(ddof=0).replace(0, 1.0)
        matrix = standardized.to_numpy()
        u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
        component_count = n_components or clean.shape[1]
        component_count = min(component_count, clean.shape[1])

        scores = u[:, :component_count] * singular_values[:component_count]
        score_columns = [f"pc_{i + 1}" for i in range(component_count)]
        components = pd.DataFrame(scores, index=clean.index, columns=score_columns)
        loadings = pd.DataFrame(vt[:component_count].T, index=clean.columns, columns=score_columns)

        variance = singular_values ** 2
        explained_variance = variance[:component_count] / variance.sum() if variance.sum() else variance[:component_count]

        return {
            "components": components,
            "loadings": loadings,
            "explained_variance": explained_variance.tolist(),
            "method": OrthogonalizationMethod.PCA.value,
        }

    def residualize(
        self,
        target: pd.Series,
        controls: pd.DataFrame,
    ) -> pd.Series:
        """将目标因子对控制因子回归后取残差。"""
        aligned = pd.concat([target.rename("target"), controls], axis=1).dropna()
        if aligned.empty or controls.empty:
            return target.copy()

        y = aligned["target"].to_numpy()
        x = aligned.drop(columns=["target"]).to_numpy()
        x = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ beta
        residual = pd.Series(y - fitted, index=aligned.index, name=f"{target.name}_residual")
        return residual.reindex(target.index).fillna(0.0)

    def estimate_dynamic_correlation(
        self,
        returns: pd.DataFrame,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """估计最新的动态相关矩阵。"""
        lookback = window or self.config.correlation_window
        usable = returns.tail(lookback).astype(float).dropna(axis=1, how="all")
        if usable.empty:
            return pd.DataFrame()
        return usable.corr().fillna(0.0)

    def detect_macro_regime(
        self,
        growth: float,
        inflation: float,
        liquidity: float,
    ) -> MacroRegimeSnapshot:
        """根据增长、通胀、流动性识别宏观制度。"""
        if liquidity <= self.config.liquidity_threshold:
            regime = MacroRegime.LIQUIDITY_STRESS
        elif growth >= self.config.growth_threshold and inflation >= self.config.inflation_threshold:
            regime = MacroRegime.HIGH_GROWTH_HIGH_INFLATION
        elif growth < self.config.growth_threshold and inflation >= self.config.inflation_threshold:
            regime = MacroRegime.LOW_GROWTH_HIGH_INFLATION
        elif growth < self.config.growth_threshold and inflation < self.config.inflation_threshold:
            regime = MacroRegime.LOW_GROWTH_LOW_INFLATION
        else:
            regime = MacroRegime.HIGH_GROWTH_LOW_INFLATION

        distances = [
            abs(growth - self.config.growth_threshold),
            abs(inflation - self.config.inflation_threshold),
            abs(liquidity - self.config.liquidity_threshold),
        ]
        confidence = float(min(0.99, 0.55 + np.mean(distances) * 4))

        return MacroRegimeSnapshot(
            regime=regime,
            growth=float(growth),
            inflation=float(inflation),
            liquidity=float(liquidity),
            confidence=confidence,
        )

    def compute_time_series_momentum(
        self,
        returns: pd.DataFrame,
        lookback: int = 126,
    ) -> pd.Series:
        """计算时间序列动量。"""
        window = returns.tail(lookback).astype(float)
        compounded = (1.0 + window).prod(axis=0) - 1.0
        return compounded.rename("tsmom")

    def compute_cross_sectional_momentum(
        self,
        returns: pd.DataFrame,
        lookback: int = 126,
    ) -> pd.Series:
        """计算截面动量并标准化。"""
        cross_sectional = self.compute_time_series_momentum(returns, lookback=lookback)
        dispersion = cross_sectional.std(ddof=0)
        if dispersion == 0 or np.isnan(dispersion):
            return pd.Series(0.0, index=cross_sectional.index, name="csmom")
        standardized = (cross_sectional - cross_sectional.mean()) / dispersion
        return standardized.rename("csmom")

    def compute_factor_risk_contributions(
        self,
        weights: pd.Series,
        covariance: pd.DataFrame,
    ) -> pd.DataFrame:
        """计算因子风险贡献度和 MCTR。"""
        aligned_weights = weights.reindex(covariance.index).fillna(0.0).astype(float)
        cov = covariance.loc[aligned_weights.index, aligned_weights.index].astype(float).fillna(0.0)

        vector = aligned_weights.to_numpy()
        port_var = float(vector @ cov.to_numpy() @ vector)
        port_vol = float(np.sqrt(max(port_var, 0.0)))

        if port_vol == 0:
            return pd.DataFrame(
                {
                    "weight": aligned_weights,
                    "mctr": 0.0,
                    "absolute_contribution": 0.0,
                    "contribution_pct": 0.0,
                }
            )

        mctr = cov.to_numpy() @ vector / port_vol
        contribution = vector * mctr
        contribution_pct = contribution / port_vol

        return pd.DataFrame(
            {
                "weight": aligned_weights,
                "mctr": mctr,
                "absolute_contribution": contribution,
                "contribution_pct": contribution_pct,
            },
            index=aligned_weights.index,
        )

    def stress_test_macro_scenario(
        self,
        exposures: pd.DataFrame,
        scenario_shocks: Dict[str, float],
        correlation: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """进行宏观情景压力测试。"""
        available = [name for name in scenario_shocks if name in exposures.columns]
        if not available:
            return {
                "factor_shocks": scenario_shocks,
                "asset_impacts": {},
                "total_abs_impact": 0.0,
                "highest_impact_asset": None,
            }

        shock_vector = pd.Series({name: scenario_shocks[name] for name in available}, dtype=float)
        effective_shocks = shock_vector.copy()

        if correlation is not None and not correlation.empty:
            aligned_corr = correlation.reindex(index=available, columns=available).fillna(0.0)
            spillover = aligned_corr.to_numpy() @ shock_vector.to_numpy()
            effective_shocks = pd.Series(
                shock_vector.to_numpy() + 0.25 * (spillover - shock_vector.to_numpy()),
                index=available,
            )

        aligned_exposures = exposures.reindex(columns=available).fillna(0.0).astype(float)
        asset_impacts = aligned_exposures.mul(effective_shocks, axis=1).sum(axis=1)

        return {
            "factor_shocks": effective_shocks.to_dict(),
            "asset_impacts": asset_impacts.to_dict(),
            "total_abs_impact": float(asset_impacts.abs().sum()),
            "highest_impact_asset": None if asset_impacts.empty else str(asset_impacts.abs().idxmax()),
        }
