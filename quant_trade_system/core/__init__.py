"""
Core Package - 核心性能优化模块

提供Polars加速和性能优化工具。
"""

from .polars_adapter import (
    PolarsDataFrame,
    should_use_polars,
    compute_indicators_optimized,
    PerformanceBenchmark,
    HAS_POLARS,
)

__all__ = [
    'PolarsDataFrame',
    'should_use_polars',
    'compute_indicators_optimized',
    'PerformanceBenchmark',
    'HAS_POLARS',
]
