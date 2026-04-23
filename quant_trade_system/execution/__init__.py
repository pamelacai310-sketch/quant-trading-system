"""
Execution Package - 执行算法模块

提供TWAP、VWAP、POV等交易执行算法。
"""

from .execution_algorithms import (
    ExecutionAlgorithm,
    ExecutionResult,
    ExecutionOptimizer,
    optimize_execution,
    OrderRequest,
)

from .twap_algorithm import (
    TWAPAlgorithm,
    AdaptiveTWAPAlgorithm,
    execute_twap,
)

from .vwap_algorithm import (
    VWAPAlgorithm,
    POVAlgorithm,
    execute_vwap,
    execute_pov,
)

__all__ = [
    'ExecutionAlgorithm',
    'ExecutionResult',
    'ExecutionOptimizer',
    'optimize_execution',
    'OrderRequest',
    'TWAPAlgorithm',
    'AdaptiveTWAPAlgorithm',
    'execute_twap',
    'VWAPAlgorithm',
    'POVAlgorithm',
    'execute_vwap',
    'execute_pov',
]
