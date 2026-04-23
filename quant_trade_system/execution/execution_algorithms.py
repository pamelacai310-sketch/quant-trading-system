"""
执行算法基类 - Execution Algorithms

定义交易执行算法的统一接口，支持TWAP、VWAP、POV、IS等算法。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 导入现有订单类型
try:
    from ..models import OrderRequest
except ImportError:
    # 如果导入失败，定义简化版本
    @dataclass
    class OrderRequest:
        symbol: str
        side: str  # 'buy' or 'sell'
        quantity: float
        order_type: str = 'market'
        limit_price: Optional[float] = None
        strategy_id: Optional[str] = None
        metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionAlgorithm(ABC):
    """
    执行算法抽象基类。

    所有执行算法必须实现execute方法，接受父订单并返回子订单列表。
    """

    def __init__(self, name: str):
        """
        初始化执行算法。

        Args:
            name: 算法名称
        """
        self.name = name
        self.parameters = {}

    @abstractmethod
    def execute(
        self,
        order: 'OrderRequest',
        market_data: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> List['OrderRequest']:
        """
        执行订单算法。

        Args:
            order: 父订单
            market_data: 市场数据（OHLCV）
            current_price: 当前市场价格

        Returns:
            子订单列表
        """
        pass

    def get_parameters(self) -> Dict[str, Any]:
        """获取算法参数"""
        return self.parameters

    def set_parameter(self, key: str, value: Any):
        """设置算法参数"""
        self.parameters[key] = value


class ExecutionResult:
    """执行结果"""

    def __init__(
        self,
        algorithm_name: str,
        parent_order_id: str,
        child_orders: List['OrderRequest'],
        execution_stats: Dict[str, Any]
    ):
        self.algorithm_name = algorithm_name
        self.parent_order_id = parent_order_id
        self.child_orders = child_orders
        self.execution_stats = execution_stats

    def get_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            'algorithm': self.algorithm_name,
            'num_child_orders': len(self.child_orders),
            'total_quantity': sum(order.quantity for order in self.child_orders),
            'stats': self.execution_stats
        }


class ExecutionOptimizer:
    """
    执行优化器 - 选择最优执行算法。

    根据订单特征和市场条件，自动选择最优的执行算法。
    """

    def __init__(self):
        """初始化执行优化器"""
        self.algorithms = {}

    def register_algorithm(self, algorithm: ExecutionAlgorithm):
        """
        注册执行算法。

        Args:
            algorithm: 执行算法实例
        """
        self.algorithms[algorithm.name] = algorithm

    def select_algorithm(
        self,
        order: 'OrderRequest',
        market_data: pd.DataFrame
    ) -> ExecutionAlgorithm:
        """
        自动选择最优执行算法。

        Args:
            order: 订单
            market_data: 市场数据

        Returns:
            最优执行算法
        """
        # 计算订单特征
        order_value = order.quantity * market_data['close'].iloc[-1]
        avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
        volatility = market_data['close'].pct_change().rolling(20).std().iloc[-1]

        # 决策规则
        if order_value > 100000:
            # 大额订单: 使用TWAP或VWAP
            if 'vwap' in self.algorithms and avg_volume > order.quantity * 10:
                return self.algorithms['vwap']
            elif 'twap' in self.algorithms:
                return self.algorithms['twap']
        elif order_value > 20000:
            # 中等订单: 使用VWAP或TWAP
            if 'vwap' in self.algorithms:
                return self.algorithms['vwap']
            elif 'twap' in self.algorithms:
                return self.algorithms['twap']
        else:
            # 小额订单: 直接市价成交
            if 'market' in self.algorithms:
                return self.algorithms['market']

        # 默认返回第一个可用算法，或者创建一个简单的市价执行
        if self.algorithms:
            return next(iter(self.algorithms.values()))
        else:
            # 如果没有注册任何算法，返回None (由调用者处理)
            return None

    def execute_order(
        self,
        order: 'OrderRequest',
        market_data: pd.DataFrame,
        algorithm_name: Optional[str] = None
    ) -> ExecutionResult:
        """
        执行订单（自动选择最优算法）。

        Args:
            order: 订单
            market_data: 市场数据
            algorithm_name: 指定算法名称（可选）

        Returns:
            执行结果
        """
        # 选择算法
        if algorithm_name and algorithm_name in self.algorithms:
            algorithm = self.algorithms[algorithm_name]
        else:
            algorithm = self.select_algorithm(order, market_data)

        # 执行算法
        child_orders = algorithm.execute(order, market_data)

        # 计算执行统计
        execution_stats = self._calculate_execution_stats(
            order, child_orders, market_data
        )

        return ExecutionResult(
            algorithm_name=algorithm.name,
            parent_order_id=order.metadata.get('order_id', 'unknown'),
            child_orders=child_orders,
            execution_stats=execution_stats
        )

    def _calculate_execution_stats(
        self,
        parent_order: 'OrderRequest',
        child_orders: List['OrderRequest'],
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """计算执行统计信息"""
        total_child_qty = sum(order.quantity for order in child_orders)
        fill_ratio = total_child_qty / parent_order.quantity if parent_order.quantity > 0 else 0

        # 估算执行价格（使用最后一笔交易价格）
        avg_price = market_data['close'].iloc[-1]

        return {
            'fill_ratio': fill_ratio,
            'num_child_orders': len(child_orders),
            'avg_child_size': total_child_qty / len(child_orders) if child_orders else 0,
            'estimated_price': avg_price
        }


# 便捷函数
def optimize_execution(
    order: 'OrderRequest',
    market_data: pd.DataFrame,
    algorithm_name: Optional[str] = None
) -> ExecutionResult:
    """
    优化订单执行（便捷函数）。

    Args:
        order: 订单
        market_data: 市场数据
        algorithm_name: 算法名称（可选）

    Returns:
        执行结果
    """
    optimizer = ExecutionOptimizer()
    return optimizer.execute_order(order, market_data, algorithm_name)
