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
import re
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

    @staticmethod
    def _peer_context_key(peer_name: str) -> str:
        """把全球同类合约名称规范成可拼接列名的 key。"""
        return re.sub(r"[^a-z0-9]+", "_", str(peer_name).strip().lower()).strip("_")

    def attach_global_peer_context(
        self,
        df: pd.DataFrame,
        peer_frames: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        把全球同类期货的行情上下文拼接到单品种数据上。

        约定输出列:
        - peer_<peer_key>_close
        - peer_<peer_key>_volume
        """
        normalized = self._normalize_market_frame(df)
        if not peer_frames:
            return normalized

        enriched = normalized.copy()
        original_index = enriched.index
        merge_key = "_peer_merge_key"
        if "date" in enriched.columns:
            enriched[merge_key] = pd.to_datetime(enriched["date"], errors="coerce")
        else:
            enriched[merge_key] = pd.RangeIndex(len(enriched))

        for peer_name, peer_frame in peer_frames.items():
            if peer_frame is None or getattr(peer_frame, "empty", True):
                continue
            peer = self._normalize_market_frame(peer_frame)
            peer_key = self._peer_context_key(peer_name)
            close_col = f"peer_{peer_key}_close"
            volume_col = f"peer_{peer_key}_volume"

            if "date" in peer.columns and "date" in enriched.columns:
                peer_dates = pd.to_datetime(peer["date"], errors="coerce")
                payload = pd.DataFrame(
                    {
                        merge_key: peer_dates,
                        close_col: pd.to_numeric(peer["close"], errors="coerce"),
                        volume_col: pd.to_numeric(peer.get("volume", 1.0), errors="coerce"),
                    }
                ).dropna(subset=[merge_key])
                payload = payload.drop_duplicates(subset=[merge_key], keep="last")
                enriched = enriched.merge(payload, on=merge_key, how="left")
            else:
                peer_reset = peer.reset_index(drop=True)
                enriched[close_col] = pd.to_numeric(
                    peer_reset["close"],
                    errors="coerce",
                ).reindex(range(len(enriched))).to_numpy()
                enriched[volume_col] = pd.to_numeric(
                    peer_reset.get("volume", pd.Series(1.0, index=peer_reset.index)),
                    errors="coerce",
                ).reindex(range(len(enriched))).to_numpy()

        peer_columns = [column for column in enriched.columns if column.startswith("peer_")]
        if peer_columns:
            enriched[peer_columns] = enriched[peer_columns].ffill()
        enriched = enriched.drop(columns=[merge_key], errors="ignore")
        enriched.index = original_index
        return enriched

    def compute_global_futures_linkage_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算期货品种对全球同类合约的联动特征。

        所有特征都保持明确金融含义：
        - 海外基准的上一交易日收益
        - 国内/海外相对动量差
        - 量价强弱差
        - 波动率和 Beta 传导
        """
        normalized = self._normalize_market_frame(df)
        peer_close_cols = sorted(
            column for column in normalized.columns if column.startswith("peer_") and column.endswith("_close")
        )
        if not peer_close_cols:
            return pd.DataFrame(index=normalized.index)

        peer_volume_cols = sorted(
            column for column in normalized.columns if column.startswith("peer_") and column.endswith("_volume")
        )
        peer_close = normalized[peer_close_cols].apply(pd.to_numeric, errors="coerce").replace(0, np.nan).ffill()
        peer_volume = (
            normalized[peer_volume_cols].apply(pd.to_numeric, errors="coerce").replace(0, np.nan).ffill()
            if peer_volume_cols
            else pd.DataFrame(index=normalized.index)
        )
        own_close = pd.to_numeric(normalized["close"], errors="coerce").replace(0, np.nan).ffill()
        own_volume = pd.to_numeric(normalized["volume"], errors="coerce").replace(0, np.nan).ffill().fillna(1.0)

        own_ret_1 = own_close.pct_change().fillna(0.0)
        own_ret_20 = own_close.pct_change(20).fillna(0.0)
        own_price_vs_ma20 = (
            own_close / own_close.rolling(20, min_periods=5).mean().replace(0, np.nan) - 1
        ).fillna(0.0)
        own_volume_surprise_20 = (
            own_volume / own_volume.rolling(20, min_periods=5).mean().replace(0, np.nan) - 1
        ).fillna(0.0)
        own_realized_vol_20 = own_ret_1.rolling(20, min_periods=5).std().fillna(0.0)

        peer_ret_1 = peer_close.pct_change().shift(1)
        peer_ret_5 = peer_close.pct_change(5).shift(1)
        peer_ret_20 = peer_close.pct_change(20).shift(1)
        peer_price_vs_ma20 = (
            peer_close / peer_close.rolling(20, min_periods=5).mean().replace(0, np.nan) - 1
        ).shift(1)
        peer_realized_vol_20 = peer_close.pct_change().rolling(20, min_periods=5).std().shift(1)

        if not peer_volume.empty:
            peer_volume_surprise_20 = (
                peer_volume / peer_volume.rolling(20, min_periods=5).mean().replace(0, np.nan) - 1
            ).shift(1)
        else:
            peer_volume_surprise_20 = pd.DataFrame(0.0, index=normalized.index, columns=peer_close.columns)

        peer_ret_1_mean = peer_ret_1.mean(axis=1).fillna(0.0)
        peer_ret_5_mean = peer_ret_5.mean(axis=1).fillna(0.0)
        peer_ret_20_mean = peer_ret_20.mean(axis=1).fillna(0.0)
        peer_price_vs_ma20_mean = peer_price_vs_ma20.mean(axis=1).fillna(0.0)
        peer_volume_surprise_mean = peer_volume_surprise_20.mean(axis=1).fillna(0.0)
        peer_realized_vol_mean = peer_realized_vol_20.mean(axis=1).fillna(0.0)
        peer_dispersion_20 = peer_ret_20.std(axis=1).fillna(0.0)

        rolling_corr_20 = own_ret_1.rolling(20, min_periods=5).corr(peer_ret_1_mean).fillna(0.0)
        beta_num = own_ret_1.rolling(20, min_periods=5).cov(peer_ret_1_mean)
        beta_den = peer_ret_1_mean.rolling(20, min_periods=5).var().replace(0, np.nan)
        rolling_beta_20 = (beta_num / beta_den).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        features = pd.DataFrame(index=normalized.index)
        features["global_peer_return_1d_lag"] = peer_ret_1_mean
        features["global_peer_return_5d_lag"] = peer_ret_5_mean
        features["global_peer_relative_momentum_20"] = own_ret_20 - peer_ret_20_mean
        features["global_peer_price_extension_gap_20"] = own_price_vs_ma20 - peer_price_vs_ma20_mean
        features["global_peer_volume_intensity_gap_20"] = own_volume_surprise_20 - peer_volume_surprise_mean
        features["global_peer_volatility_gap_20"] = own_realized_vol_20 - peer_realized_vol_mean
        features["global_peer_beta_20"] = rolling_beta_20
        features["global_peer_correlation_20"] = rolling_corr_20
        features["global_peer_dispersion_20"] = peer_dispersion_20
        features["global_peer_spillover_score"] = (
            0.45 * self._zscore(peer_ret_5_mean)
            + 0.35 * self._zscore(rolling_beta_20 * rolling_corr_20.clip(lower=0.0))
            - 0.20 * self._zscore(peer_dispersion_20)
        )
        return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _global_peer_feature_metadata() -> Dict[str, Dict[str, Any]]:
        """全球同类期货联动特征的说明与公式。"""
        return {
            "global_peer_return_1d_lag": {
                "category": "global_futures_linkage",
                "family": "peer_return_linkage",
                "formula": "mean_i(peer_close_i,t-1 / peer_close_i,t-2 - 1)",
                "financial_meaning": "全球同类期货上一交易日的平均收益，用来刻画海外价格发现对本地合约的隔夜传导。",
                "required_inputs": ["close", "peer_*_close"],
            },
            "global_peer_return_5d_lag": {
                "category": "global_futures_linkage",
                "family": "peer_trend_linkage",
                "formula": "mean_i(peer_close_i,t-1 / peer_close_i,t-6 - 1)",
                "financial_meaning": "全球同类期货过去 5 日的平均趋势强度，用来刻画海外中短期趋势对本地合约的引导。",
                "required_inputs": ["close", "peer_*_close"],
            },
            "global_peer_relative_momentum_20": {
                "category": "global_futures_linkage",
                "family": "peer_relative_strength",
                "formula": "(close_t / close_t-20 - 1) - mean_i(peer_close_i,t-1 / peer_close_i,t-21 - 1)",
                "financial_meaning": "本地合约相对全球同类合约的 20 日强弱差，用来衡量国内是否明显跑赢或跑输全球基准。",
                "required_inputs": ["close", "peer_*_close"],
            },
            "global_peer_price_extension_gap_20": {
                "category": "global_futures_linkage",
                "family": "peer_price_extension",
                "formula": "(close_t / MA20_t - 1) - mean_i(peer_close_i,t-1 / peer_MA20_i,t-1 - 1)",
                "financial_meaning": "本地价格偏离 20 日均线的程度，相对全球同类合约的偏离差，反映局部拥挤或补涨补跌空间。",
                "required_inputs": ["close", "peer_*_close"],
            },
            "global_peer_volume_intensity_gap_20": {
                "category": "global_futures_linkage",
                "family": "peer_flow_linkage",
                "formula": "(volume_t / mean(volume,20)_t - 1) - mean_i(peer_volume_i,t-1 / mean(peer_volume_i,20)_t-1 - 1)",
                "financial_meaning": "本地成交量强度相对全球同类合约的差值，用来衡量是否出现本地独立资金推动。",
                "required_inputs": ["volume", "peer_*_volume"],
            },
            "global_peer_volatility_gap_20": {
                "category": "global_futures_linkage",
                "family": "peer_risk_linkage",
                "formula": "stdev(ret,20)_t - mean_i(stdev(peer_ret_i,20)_t-1)",
                "financial_meaning": "本地波动率相对全球同类期货的风险溢价差，用来识别本地是否承受了额外风险定价。",
                "required_inputs": ["close", "peer_*_close"],
            },
            "global_peer_beta_20": {
                "category": "global_futures_linkage",
                "family": "peer_beta_linkage",
                "formula": "cov(ret_t, peer_ret_mean_t-1, 20) / var(peer_ret_mean_t-1, 20)",
                "financial_meaning": "本地合约对全球同类合约收益的滚动 Beta，衡量价格传导弹性。",
                "required_inputs": ["close", "peer_*_close"],
            },
            "global_peer_correlation_20": {
                "category": "global_futures_linkage",
                "family": "peer_correlation_linkage",
                "formula": "corr(ret_t, peer_ret_mean_t-1, 20)",
                "financial_meaning": "本地与全球同类合约的 20 日滚动相关性，衡量联动关系是否处于高同步区间。",
                "required_inputs": ["close", "peer_*_close"],
            },
            "global_peer_dispersion_20": {
                "category": "global_futures_linkage",
                "family": "peer_dispersion",
                "formula": "stdev_i(peer_close_i,t-1 / peer_close_i,t-21 - 1)",
                "financial_meaning": "全球同类合约之间的 20 日收益分歧，分歧越大说明海外信号越不一致。",
                "required_inputs": ["peer_*_close"],
            },
            "global_peer_spillover_score": {
                "category": "global_futures_linkage",
                "family": "peer_spillover_composite",
                "formula": "0.45*zscore(global_peer_return_5d_lag)+0.35*zscore(global_peer_beta_20*max(global_peer_correlation_20,0))-0.20*zscore(global_peer_dispersion_20)",
                "financial_meaning": "把海外趋势、Beta 传导和海外分歧聚合为一个全球联动强度分数，用来刻画外盘信号对本地期货的综合可传导性。",
                "required_inputs": ["close", "peer_*_close", "peer_*_volume"],
            },
        }

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
        for factor_name, spec in self._global_peer_feature_metadata().items():
            metadata.setdefault(factor_name, spec)
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
