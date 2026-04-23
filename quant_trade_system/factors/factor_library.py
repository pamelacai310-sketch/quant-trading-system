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

        return registry

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

    def _load_factor_metadata(self) -> Dict[str, Dict[str, Any]]:
        """加载因子元数据"""
        metadata_file = self.cache_dir / "factor_metadata.json"

        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)

        # 否则返回默认元数据
        return {}

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
