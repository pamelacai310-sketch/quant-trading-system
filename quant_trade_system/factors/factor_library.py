"""
因子库主模块 - Factor Library

管理3000+因子的计算、缓存、筛选和组合优化。
支持批量计算、增量更新和Polars加速。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import warnings

# 导入各因子类别
try:
    from .technical_factors import TechnicalFactors
except ImportError:
    TechnicalFactors = None
    warnings.warn("TechnicalFactors not available")

try:
    from .fundamental_factors import FundamentalFactors
except ImportError:
    FundamentalFactors = None
    warnings.warn("FundamentalFactors not available")

# 尝试导入Polars加速
try:
    from ..core.polars_adapter import PolarsDataFrame, should_use_polars
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    PolarsDataFrame = None
    should_use_polars = None

try:
    from ..core.causal.causal_factor_library import CausalFactorLibrary, QuantizedCausalFactor
except ImportError:
    CausalFactorLibrary = None
    QuantizedCausalFactor = None


class FactorLibrary:
    """
    因子库主类 - 管理3000+因子。

    特性:
    - 3000+预定义因子
    - Polars自动加速
    - 因子缓存机制
    - 批量计算优化
    - 增量更新支持
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化因子库。

        Args:
            cache_dir: 因子缓存目录
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("state/factor_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化各因子类别
        self.technical = TechnicalFactors() if TechnicalFactors else None
        self.fundamental = FundamentalFactors() if FundamentalFactors else None
        self.causal_library = CausalFactorLibrary() if CausalFactorLibrary else None
        self.causal_quant_catalog: Dict[str, QuantizedCausalFactor] = (
            self.causal_library.get_quantized_factor_catalog() if self.causal_library else {}
        )

        # 构建因子注册表
        self.factor_registry = self._build_registry()

        # 因子缓存
        self._factor_cache = {}

        # 因子元数据
        self._factor_metadata = self._load_factor_metadata()

    def _build_registry(self) -> Dict[str, Callable]:
        """
        构建因子注册表，包含所有3000+因子。

        Returns:
            因子名称到计算函数的映射
        """
        registry = {}

        # 1. 技术因子 (500+)
        if self.technical:
            registry.update(self.technical.get_all_factors())

        # 2. 基本面因子 (300+)
        if self.fundamental:
            registry.update(self.fundamental.get_all_factors())

        # 3. 宏观因子 (200+) - 待实现
        # registry.update(self.macro.get_all_factors())

        # 4. 其他因子类别...
        # 动量、反转、波动率、流动性、质量、情绪、季节性等

        # 5. 因果量化因子（把因果语义关系结构化）
        registry.update(self._build_causal_quant_registry())

        return registry

    def _build_causal_quant_registry(self) -> Dict[str, Callable]:
        """构建因果量化因子注册表。"""
        registry: Dict[str, Callable] = {}
        for factor_name, spec in self.causal_quant_catalog.items():
            registry[factor_name] = self._make_causal_quant_calculator(spec)
        return registry

    def _make_causal_quant_calculator(self, spec: QuantizedCausalFactor) -> Callable:
        """为量化因子定义生成可执行计算器。"""
        def calculator(df: pd.DataFrame) -> pd.Series:
            normalized = self._normalize_market_frame(df)
            return self._compute_causal_quant_factor(normalized, spec)

        return calculator

    def get_factor_list(
        self,
        category: Optional[str] = None
    ) -> List[str]:
        """
        获取因子列表。

        Args:
            category: 因子类别过滤

        Returns:
            因子名称列表
        """
        if category:
            return [
                name for name, meta in self._factor_metadata.items()
                if meta.get("category") == category
            ]
        return list(self.factor_registry.keys())

    def compute_factor(
        self,
        factor_name: str,
        df: pd.DataFrame,
        use_cache: bool = True
    ) -> pd.Series:
        """
        计算单个因子。

        Args:
            factor_name: 因子名称
            df: 输入DataFrame
            use_cache: 是否使用缓存

        Returns:
            因子值Series
        """
        # 检查缓存
        if use_cache and factor_name in self._factor_cache:
            return self._factor_cache[factor_name]

        # 检查因子是否存在
        if factor_name not in self.factor_registry:
            raise ValueError(f"Factor '{factor_name}' not found in library")

        # 计算因子
        try:
            factor_func = self.factor_registry[factor_name]
            result = factor_func(df)

            # 缓存结果
            if use_cache:
                self._factor_cache[factor_name] = result

            return result

        except Exception as e:
            warnings.warn(f"Error computing factor {factor_name}: {str(e)}")
            raise

    def compute_factor_batch(
        self,
        factor_names: List[str],
        df: pd.DataFrame,
        use_polars: bool = True,
        parallel: bool = True
    ) -> pd.DataFrame:
        """
        批量计算因子（Polars加速）。

        Args:
            factor_names: 因子名称列表
            df: 输入DataFrame
            use_polars: 是否使用Polars加速
            parallel: 是否并行计算

        Returns:
            包含所有因子的DataFrame
        """
        if not factor_names:
            return pd.DataFrame(index=df.index)

        # 使用Polars加速（如果可用且数据量大）
        if use_polars and HAS_POLARS and should_use_polars(df):
            return self._compute_with_polars(factor_names, df)

        # 否则使用pandas逐个计算
        result = pd.DataFrame(index=df.index)

        for factor_name in factor_names:
            try:
                result[factor_name] = self.compute_factor(factor_name, df)
            except Exception as e:
                warnings.warn(f"Failed to compute {factor_name}: {str(e)}")
                result[factor_name] = np.nan

        return result

    def _compute_with_polars(
        self,
        factor_names: List[str],
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """使用Polars批量计算因子"""
        polars_df = PolarsDataFrame(df)

        # 转换因子规范为Polars格式
        indicator_specs = []
        for name in factor_names:
            if name in self._factor_metadata:
                meta = self._factor_metadata[name]
                indicator_specs.append({
                    "name": name,
                    "type": meta["type"],
                    "window": meta.get("window", 20)
                })

        # 批量计算
        return polars_df.compute_indicators(indicator_specs)

    def compute_all_technical_factors(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算所有500+技术因子。

        Args:
            df: 输入DataFrame (必须包含OHLCV列)

        Returns:
            包含所有技术因子的DataFrame
        """
        if not self.technical:
            warnings.warn("TechnicalFactors not available")
            return pd.DataFrame(index=df.index)

        return self.technical.compute_all_factors(df)

    def _normalize_market_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """把输入统一成因子计算所需的小写市场列。"""
        normalized = df.copy()
        rename_map = {}
        for column in normalized.columns:
            lower = column.lower()
            if lower in {"date", "timestamp", "datetime", "time"}:
                rename_map[column] = "date"
            elif lower in {"open", "open_price"}:
                rename_map[column] = "open"
            elif lower in {"high", "high_price"}:
                rename_map[column] = "high"
            elif lower in {"low", "low_price"}:
                rename_map[column] = "low"
            elif lower in {"close", "close_price", "adj_close", "price"}:
                rename_map[column] = "close"
            elif lower in {"volume", "vol"}:
                rename_map[column] = "volume"
            else:
                rename_map[column] = lower
        normalized = normalized.rename(columns=rename_map)
        for column in ["open", "high", "low", "close", "volume"]:
            if column not in normalized.columns:
                if column == "volume":
                    normalized[column] = 1.0
                else:
                    normalized[column] = normalized.get("close", pd.Series(index=normalized.index, dtype=float))
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        return normalized

    @staticmethod
    def _rolling_r_squared(series: pd.Series, window: int) -> pd.Series:
        """滚动趋势R²。"""
        values = pd.to_numeric(series, errors="coerce")
        result = pd.Series(index=series.index, dtype=float)
        x = np.arange(window, dtype=float)
        x_centered = x - x.mean()
        ss_x = np.sum(x_centered ** 2)
        for idx in range(window - 1, len(values)):
            window_values = values.iloc[idx - window + 1: idx + 1]
            if window_values.isna().any():
                continue
            y = window_values.to_numpy(dtype=float)
            y_centered = y - y.mean()
            slope = float(np.dot(x_centered, y_centered) / ss_x) if ss_x else 0.0
            intercept = float(y.mean() - slope * x.mean())
            y_hat = intercept + slope * x
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            result.iloc[idx] = 1.0 - ss_res / ss_tot if ss_tot else 0.0
        return result.fillna(0.0)

    @staticmethod
    def _zscore(series: pd.Series, window: int = 20) -> pd.Series:
        """滚动z-score。"""
        values = pd.to_numeric(series, errors="coerce")
        mean = values.rolling(window, min_periods=max(5, window // 2)).mean()
        std = values.rolling(window, min_periods=max(5, window // 2)).std().replace(0, np.nan)
        result = (values - mean) / std
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _select_explicit_or_proxy(
        df: pd.DataFrame,
        aliases: List[str],
        proxy: pd.Series,
    ) -> pd.Series:
        """优先使用显式列，缺失时回落到市场代理。"""
        for alias in aliases:
            key = alias.lower()
            if key in df.columns:
                return pd.to_numeric(df[key], errors="coerce").ffill().fillna(0.0)
        return proxy.fillna(0.0)

    def _compute_causal_quant_factor(
        self,
        df: pd.DataFrame,
        spec: QuantizedCausalFactor,
    ) -> pd.Series:
        """执行因果关系到量化因子的映射计算。"""
        close = pd.to_numeric(df["close"], errors="coerce").replace(0, np.nan).ffill()
        high = pd.to_numeric(df["high"], errors="coerce").fillna(close)
        low = pd.to_numeric(df["low"], errors="coerce").fillna(close)
        volume = pd.to_numeric(df["volume"], errors="coerce").replace(0, np.nan).ffill().fillna(1.0)

        returns_1 = close.pct_change().fillna(0.0)
        momentum_5 = close.pct_change(5).fillna(0.0)
        momentum_20 = close.pct_change(20).fillna(0.0)
        realized_vol_20 = returns_1.rolling(20, min_periods=5).std().fillna(0.0)
        downside_vol_20 = returns_1.clip(upper=0).rolling(20, min_periods=5).std().fillna(0.0)
        volume_participation_20 = np.log1p(volume).diff().rolling(20, min_periods=5).mean().fillna(0.0)
        volume_surprise_5 = (volume / volume.rolling(20, min_periods=5).mean().replace(0, np.nan) - 1).fillna(0.0)
        spread_proxy_20 = ((high - low) / close.replace(0, np.nan)).rolling(20, min_periods=5).mean().fillna(0.0)
        relative_strength_20 = (momentum_20 - momentum_20.rolling(20, min_periods=5).mean()).fillna(0.0)
        price_vs_ma_20 = (close / close.rolling(20, min_periods=5).mean().replace(0, np.nan) - 1).fillna(0.0)
        drawdown_20 = (close / close.rolling(20, min_periods=5).max().replace(0, np.nan) - 1).fillna(0.0)
        drawdown_repair_proxy = (-drawdown_20).rolling(10, min_periods=3).mean().fillna(0.0)
        trend_r2_20 = self._rolling_r_squared(close, 20)
        basis_proxy_20 = ((close - close.rolling(5, min_periods=3).mean()) / close.rolling(5, min_periods=3).mean().replace(0, np.nan)).fillna(0.0)
        term_structure_proxy_20 = (close.rolling(20, min_periods=5).mean() / close.rolling(60, min_periods=10).mean().replace(0, np.nan) - 1).fillna(0.0)

        valuation_multiple_proxy = self._select_explicit_or_proxy(
            df,
            ["pe", "pb", "ps", "valuation_multiple", "market_cap_to_sales"],
            (1.0 / (price_vs_ma_20.abs() + 1e-3)).fillna(0.0),
        )
        cashflow_yield_proxy = self._select_explicit_or_proxy(
            df,
            ["cashflow_yield", "earnings_yield", "fcf_yield", "dividend_yield"],
            (momentum_20 / (realized_vol_20 + 1e-6)).fillna(0.0),
        )
        profitability_proxy = self._select_explicit_or_proxy(
            df,
            ["roic", "roe", "gross_margin", "eps_growth", "profitability"],
            (trend_r2_20 + price_vs_ma_20 - downside_vol_20).fillna(0.0),
        )
        stability_proxy = self._select_explicit_or_proxy(
            df,
            ["quality_score", "stability", "interest_coverage", "altman_z"],
            (1.0 - realized_vol_20).fillna(0.0),
        )
        rate_proxy = self._select_explicit_or_proxy(
            df,
            ["rate", "yield_10y", "shibor", "libor", "funding_rate"],
            (realized_vol_20 + spread_proxy_20).fillna(0.0),
        )
        inflation_proxy = self._select_explicit_or_proxy(
            df,
            ["inflation", "cpi", "ppi", "breakeven_inflation"],
            (momentum_20 + price_vs_ma_20).fillna(0.0),
        )
        real_rate_proxy = self._select_explicit_or_proxy(
            df,
            ["real_rate", "tips_yield", "real_yield"],
            (rate_proxy - inflation_proxy).fillna(0.0),
        )
        liquidity_growth_20 = self._select_explicit_or_proxy(
            df,
            ["m2_growth", "credit_impulse", "liquidity_growth", "social_financing_growth"],
            (volume_participation_20 - spread_proxy_20).fillna(0.0),
        )
        funding_stress_20 = self._select_explicit_or_proxy(
            df,
            ["funding_stress", "credit_spread", "basis_stress"],
            (spread_proxy_20 + downside_vol_20).fillna(0.0),
        )
        demand_proxy_20 = self._select_explicit_or_proxy(
            df,
            ["demand", "sales_growth", "consumption", "import_growth"],
            (momentum_20 + volume_participation_20).fillna(0.0),
        )
        supply_proxy_20 = self._select_explicit_or_proxy(
            df,
            ["supply", "production_growth", "export_growth"],
            (spread_proxy_20 + realized_vol_20).fillna(0.0),
        )
        inventory_proxy = self._select_explicit_or_proxy(
            df,
            ["inventory", "stocks", "warehouse_stock"],
            (-momentum_20 + realized_vol_20).fillna(0.0),
        )
        inventory_tightness_proxy = ((demand_proxy_20 - inventory_proxy) / (inventory_proxy.abs() + 1e-6)).fillna(0.0)
        storage_pressure_proxy = self._select_explicit_or_proxy(
            df,
            ["storage_cost", "carry_cost", "warehouse_cost"],
            (inventory_proxy + spread_proxy_20).fillna(0.0),
        )
        seasonal_return_same_period = momentum_20.rolling(12, min_periods=6).mean().fillna(0.0)
        calendar_flow_proxy = volume_surprise_5.rolling(5, min_periods=3).mean().fillna(0.0)
        duration_sensitive_drawdown = (drawdown_20 + rate_proxy).fillna(0.0)
        real_asset_momentum_20d = (momentum_20 - real_rate_proxy).fillna(0.0)

        family_map = {
            "rate_discount": -self._zscore(rate_proxy.diff(5)) + 0.35 * self._zscore(cashflow_yield_proxy) - 0.20 * self._zscore(duration_sensitive_drawdown),
            "inflation_real_asset": self._zscore(inflation_proxy.diff(20)) + 0.5 * self._zscore(real_asset_momentum_20d) - 0.25 * self._zscore(real_rate_proxy),
            "liquidity_policy": self._zscore(liquidity_growth_20 - funding_stress_20) + 0.25 * self._zscore(volume_participation_20),
            "participation_flow": self._zscore(volume_participation_20) - 0.5 * self._zscore(spread_proxy_20) - 0.25 * self._zscore(realized_vol_20),
            "sentiment_regime": self._zscore(momentum_5) + 0.5 * self._zscore(volume_surprise_5) - 0.5 * self._zscore(downside_vol_20),
            "volatility_tail": -self._zscore(realized_vol_20 + downside_vol_20) + 0.35 * self._zscore(drawdown_repair_proxy),
            "trend_strength": self._zscore(momentum_20 / (realized_vol_20 + 1e-6)) + 0.4 * self._zscore(trend_r2_20),
            "valuation_quality": -self._zscore(valuation_multiple_proxy) + 0.6 * self._zscore(cashflow_yield_proxy) + 0.2 * self._zscore(relative_strength_20),
            "fundamental_quality": self._zscore(profitability_proxy) + 0.45 * self._zscore(stability_proxy) + 0.25 * self._zscore(relative_strength_20),
            "supply_demand_balance": self._zscore((demand_proxy_20 - supply_proxy_20) / (supply_proxy_20.abs() + 1e-6)) + 0.25 * self._zscore(inventory_tightness_proxy),
            "carry_curve": self._zscore(basis_proxy_20) + 0.4 * self._zscore(term_structure_proxy_20) - 0.2 * self._zscore(storage_pressure_proxy),
            "seasonality": self._zscore(seasonal_return_same_period) + 0.3 * self._zscore(calendar_flow_proxy),
            "broad_causal": self._zscore(relative_strength_20) + 0.35 * self._zscore(volume_participation_20) - 0.25 * self._zscore(drawdown_20),
        }
        result = family_map.get(spec.formula_family, family_map["broad_causal"])
        return (result * float(spec.expected_sign)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def compute_cross_sectional_factors(
        self,
        df: pd.DataFrame,
        date: str
    ) -> pd.DataFrame:
        """
        计算截面因子（跨股票）。

        Args:
            df: 包含多个股票的DataFrame
            date: 计算日期

        Returns:
            截面因子DataFrame
        """
        # 这里实现截面因子计算
        # 市值、PE、PB等跨股票比较因子
        pass

    def filter_factors_by_ic(
        self,
        factor_names: List[str],
        returns: pd.Series,
        min_ic: float = 0.03
    ) -> List[str]:
        """
        根据IC值筛选因子。

        Args:
            factor_names: 候选因子列表
            returns: 收益率Series
            min_ic: 最小IC阈值

        Returns:
            通过筛选的因子列表
        """
        passed_factors = []

        for factor_name in factor_names:
            if factor_name in self._factor_cache:
                factor_values = self._factor_cache[factor_name]
            else:
                continue

            # 计算IC (Information Coefficient)
            ic = factor_values.corr(returns)

            if abs(ic) >= min_ic:
                passed_factors.append(factor_name)

        return passed_factors

    def filter_correlated_factors(
        self,
        factor_names: List[str],
        df: pd.DataFrame,
        threshold: float = 0.95
    ) -> List[str]:
        """
        去除高度相关的因子。

        Args:
            factor_names: 因子列表
            df: 因子数据
            threshold: 相关系数阈值

        Returns:
            去重后的因子列表
        """
        # 计算因子相关矩阵
        factor_data = df[factor_names]
        corr_matrix = factor_data.corr()

        # 找出高相关因子对
        to_remove = set()
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    # 保留IC更高的因子
                    factor_i = corr_matrix.index[i]
                    factor_j = corr_matrix.index[j]
                    ic_i = self._factor_metadata.get(factor_i, {}).get("ic", 0)
                    ic_j = self._factor_metadata.get(factor_j, {}).get("ic", 0)

                    if ic_i < ic_j:
                        to_remove.add(factor_i)
                    else:
                        to_remove.add(factor_j)

        return [f for f in factor_names if f not in to_remove]

    def get_factor_metadata(
        self,
        factor_name: str
    ) -> Dict[str, Any]:
        """
        获取因子元数据。

        Args:
            factor_name: 因子名称

        Returns:
            因子元数据字典
        """
        return self._factor_metadata.get(factor_name, {})

    def update_factor_metadata(
        self,
        factor_name: str,
        metadata: Dict[str, Any],
        persist: bool = False,
    ) -> Dict[str, Any]:
        """
        更新或创建因子元数据。
        """
        current = self._factor_metadata.setdefault(factor_name, {})
        current.update(metadata)
        if persist:
            self.save_factor_metadata()
        return current

    def _load_factor_metadata(self) -> Dict[str, Dict[str, Any]]:
        """加载因子元数据"""
        metadata_file = self.cache_dir / "factor_metadata.json"

        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        for factor_name, spec in self.causal_quant_catalog.items():
            metadata.setdefault(
                factor_name,
                {
                    "category": "causal_quant",
                    "family": spec.formula_family,
                    "formula": spec.formula,
                    "financial_meaning": spec.financial_meaning,
                    "required_inputs": spec.required_inputs,
                    "expected_sign": spec.expected_sign,
                    "source_factor_id": spec.source_factor_id,
                },
            )
        return metadata

    def save_factor_metadata(self):
        """保存因子元数据到缓存"""
        metadata_file = self.cache_dir / "factor_metadata.json"

        with open(metadata_file, 'w') as f:
            json.dump(self._factor_metadata, f, indent=2)

    def clear_cache(self):
        """清空因子缓存"""
        self._factor_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "cached_factors": len(self._factor_cache),
            "total_factors": len(self.factor_registry),
            "cache_hit_rate": 0.0  # TODO: 实现缓存命中率统计
        }

    def export_factors_to_csv(
        self,
        df: pd.DataFrame,
        output_path: str,
        factor_names: Optional[List[str]] = None
    ):
        """
        导出因子到CSV文件。

        Args:
            df: 输入DataFrame
            output_path: 输出文件路径
            factor_names: 要导出的因子列表（默认全部）
        """
        if factor_names is None:
            factor_names = list(self.factor_registry.keys())

        factor_df = self.compute_factor_batch(factor_names, df)
        factor_df.to_csv(output_path, index=True)


# 便捷函数
def load_factor_library(cache_dir: Optional[str] = None) -> FactorLibrary:
    """
    加载因子库（便捷函数）。

    Args:
        cache_dir: 缓存目录

    Returns:
        FactorLibrary实例
    """
    return FactorLibrary(cache_dir=cache_dir)
