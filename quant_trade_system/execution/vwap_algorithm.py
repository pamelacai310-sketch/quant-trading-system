"""
VWAP算法 - Volume Weighted Average Price

成交量加权平均价格执行算法，根据历史成交量分布分配订单，
以接近市场VWAP的价格执行，降低执行成本。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .execution_algorithms import ExecutionAlgorithm, OrderRequest


class VWAPAlgorithm(ExecutionAlgorithm):
    """
    VWAP算法 - 成交量加权平均价格执行。

    特点:
    - 根据历史成交量分布分配订单
    - 追踪市场VWAP价格
    - 降低市场冲击
    - 适合大额订单且成交量稳定的品种
    """

    def __init__(
        self,
        lookback_window: str = '30D',
        time_bucket: str = '15min',
        max_participation_rate: float = 0.2,
        min_order_size: Optional[float] = None
    ):
        """
        初始化VWAP算法。

        Args:
            lookback_window: 历史成交量回看窗口
            time_bucket: 时间分桶大小 ('5min', '15min', '1H'等)
            max_participation_rate: 最大参与率 (单次成交量占比)
            min_order_size: 最小订单大小
        """
        super().__init__(name='vwap')
        self.parameters = {
            'lookback_window': lookback_window,
            'time_bucket': time_bucket,
            'max_participation_rate': max_participation_rate,
            'min_order_size': min_order_size
        }

    def execute(
        self,
        order: OrderRequest,
        market_data: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> List[OrderRequest]:
        """
        执行VWAP算法。

        Args:
            order: 父订单
            market_data: 市场数据
            current_price: 当前价格

        Returns:
            子订单列表
        """
        lookback_window = self.parameters['lookback_window']
        time_bucket = self.parameters['time_bucket']
        max_participation_rate = self.parameters['max_participation_rate']
        min_order_size = self.parameters['min_order_size']

        # 1. 计算历史成交量分布
        volume_profile = self._calculate_volume_profile(
            market_data,
            lookback_window,
            time_bucket
        )

        # 2. 根据成交量分布分配订单
        child_orders = []
        total_volume = volume_profile['volume'].sum()

        for i, (time_bucket_id, bucket_data) in enumerate(volume_profile.iterrows()):
            bucket_volume = bucket_data['volume']
            bucket_ratio = bucket_volume / total_volume

            # 计算该bucket的订单数量
            target_quantity = order.quantity * bucket_ratio

            # 限制最大参与率
            if target_quantity > bucket_volume * max_participation_rate:
                target_quantity = bucket_volume * max_participation_rate

            # 限制最小订单大小
            if min_order_size and target_quantity < min_order_size:
                continue

            # 计算VWAP限价
            vwap_price = self._calculate_vwap_price(market_data, time_bucket_id)

            # 创建子订单
            child_order = OrderRequest(
                symbol=order.symbol,
                side=order.side,
                quantity=target_quantity,
                order_type='limit',  # VWAP使用限价单
                limit_price=vwap_price,
                strategy_id=order.strategy_id,
                metadata={
                    'parent_order_id': order.metadata.get('order_id', 'unknown'),
                    'bucket_number': i,
                    'time_bucket': time_bucket_id,
                    'bucket_volume': bucket_volume,
                    'bucket_ratio': bucket_ratio,
                    'algorithm': 'VWAP',
                    'limit_price': vwap_price
                }
            )

            child_orders.append(child_order)

        return child_orders

    def _calculate_volume_profile(
        self,
        market_data: pd.DataFrame,
        lookback_window: str,
        time_bucket: str
    ) -> pd.DataFrame:
        """
        计算历史成交量分布。

        Args:
            market_data: 市场数据
            lookback_window: 回看窗口
            time_bucket: 时间分桶

        Returns:
            成交量分布DataFrame
        """
        # 获取最近的数据
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(lookback_window)
        recent_data = market_data[market_data.index >= cutoff_date].copy()

        # 按时间分桶统计成交量
        if len(recent_data) == 0:
            # 如果数据不足，使用简单的平均分布
            return pd.DataFrame({
                'volume': [1.0],
                'count': [1]
            }, index=[0])

        # 提取时间信息
        recent_data['time_bucket'] = recent_data.index.floor(time_bucket)

        # 按时间分桶统计
        volume_profile = recent_data.groupby('time_bucket').agg({
            'volume': 'sum',
            'count': 'count'
        }).reset_index()

        return volume_profile.set_index('time_bucket')

    def _calculate_vwap_price(
        self,
        market_data: pd.DataFrame,
        time_bucket: Any
    ) -> float:
        """
        计算VWAP价格。

        VWAP = Σ(Price * Volume) / Σ(Volume)

        Args:
            market_data: 市场数据
            time_bucket: 时间分桶

        Returns:
            VWAP价格
        """
        # 简化实现: 使用最近期的典型价格
        # 实际应用中应该使用该时间bucket的历史VWAP

        # 获取最近的数据
        recent_data = market_data.tail(20)

        # 计算VWAP
        typical_price = (recent_data['high'] + recent_data['low'] + recent_data['close']) / 3
        vwap = (typical_price * recent_data['volume']).sum() / recent_data['volume'].sum()

        return float(vwap)

    def estimate_vwap_execution_price(
        self,
        order: OrderRequest,
        market_data: pd.DataFrame,
        child_orders: List[OrderRequest]
    ) -> Dict[str, Any]:
        """
        估算VWAP执行价格。

        Args:
            order: 父订单
            market_data: 市场数据
            child_orders: 子订单列表

        Returns:
            VWAP执行价格分析
        """
        # 计算加权平均执行价格
        total_qty = 0
        weighted_price = 0

        for child_order in child_orders:
            limit_price = child_order.metadata.get('limit_price', 0)
            qty = child_order.quantity

            weighted_price += limit_price * qty
            total_qty += qty

        avg_execution_price = weighted_price / total_qty if total_qty > 0 else 0

        # 计算市场VWAP
        recent_data = market_data.tail(20)
        typical_price = (recent_data['high'] + recent_data['low'] + recent_data['close']) / 3
        market_vwap = (typical_price * recent_data['volume']).sum() / recent_data['volume'].sum()

        # 计算偏离度
        deviation_pct = ((avg_execution_price - market_vwap) / market_vwap * 100) if market_vwap > 0 else 0

        return {
            'avg_execution_price': avg_execution_price,
            'market_vwap': market_vwap,
            'deviation_pct': deviation_pct,
            'total_executed_qty': total_qty,
            'num_child_orders': len(child_orders)
        }

    def calculate_vwap_performance(
        self,
        order: OrderRequest,
        market_data: pd.DataFrame,
        execution_prices: pd.Series
    ) -> Dict[str, Any]:
        """
        计算VWAP执行绩效。

        Args:
            order: 订单
            market_data: 市场数据
            execution_prices: 实际执行价格序列

        Returns:
            绩效指标
        """
        # 计算基准VWAP
        recent_data = market_data.tail(len(execution_prices))
        typical_price = (recent_data['high'] + recent_data['low'] + recent_data['close']) / 3
        benchmark_vwap = (typical_price * recent_data['volume']).sum() / recent_data['volume'].sum()

        # 计算实际VWAP
        actual_vwap = (execution_prices * market_data['volume'].tail(len(execution_prices))).sum() / \
                      market_data['volume'].tail(len(execution_prices)).sum()

        # 计算改善程度
        improvement_bps = ((benchmark_vwap - actual_vwap) / benchmark_vwap * 10000) if benchmark_vwap > 0 else 0

        return {
            'benchmark_vwap': benchmark_vwap,
            'actual_vwap': actual_vwap,
            'improvement_bps': improvement_bps,
            'better_than_vwap': improvement_bps > 0
        }


class POVAlgorithm(VWAPAlgorithm):
    """
    POV算法 - Percentage of Volume

    固定成交量参与率算法，控制每次执行的成交量占市场成交量的比例。
    """

    def __init__(
        self,
        target_participation_rate: float = 0.1,
        lookback_window: str = '10D',
        max_order_size: Optional[float] = None
    ):
        """
        初始化POV算法。

        Args:
            target_participation_rate: 目标参与率 (如0.1表示10%)
            lookback_window: 历史回看窗口
            max_order_size: 最大订单大小
        """
        super().__init__(lookback_window=lookback_window)
        self.name = 'pov'
        self.parameters.update({
            'target_participation_rate': target_participation_rate,
            'max_order_size': max_order_size
        })

    def execute(
        self,
        order: OrderRequest,
        market_data: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> List[OrderRequest]:
        """
        执行POV算法。

        Args:
            order: 父订单
            market_data: 市场数据
            current_price: 当前价格

        Returns:
            子订单列表
        """
        target_participation_rate = self.parameters['target_participation_rate']
        max_order_size = self.parameters['max_order_size']

        # 估算平均成交量
        avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]

        # 计算每次执行的目标成交量
        target_qty_per_slice = avg_volume * target_participation_rate

        # 限制最大订单大小
        if max_order_size and target_qty_per_slice > max_order_size:
            target_qty_per_slice = max_order_size

        # 计算需要多少次执行
        num_slices = int(np.ceil(order.quantity / target_qty_per_slice))

        # 生成子订单
        child_orders = []
        remaining_qty = order.quantity

        for i in range(num_slices):
            slice_qty = min(target_qty_per_slice, remaining_qty)

            child_order = OrderRequest(
                symbol=order.symbol,
                side=order.side,
                quantity=slice_qty,
                order_type='market',
                strategy_id=order.strategy_id,
                metadata={
                    'parent_order_id': order.metadata.get('order_id', 'unknown'),
                    'slice_number': i,
                    'total_slices': num_slices,
                    'algorithm': 'POV',
                    'target_participation_rate': target_participation_rate,
                    'target_qty': target_qty_per_slice
                }
            )

            child_orders.append(child_order)
            remaining_qty -= slice_qty

            if remaining_qty <= 0:
                break

        return child_orders


# 便捷函数
def execute_vwap(
    order: OrderRequest,
    market_data: pd.DataFrame,
    lookback_window: str = '30D',
    time_bucket: str = '15min'
) -> List[OrderRequest]:
    """
    执行VWAP算法（便捷函数）。

    Args:
        order: 订单
        market_data: 市场数据
        lookback_window: 回看窗口
        time_bucket: 时间分桶

    Returns:
        子订单列表
    """
    vwap = VWAPAlgorithm(
        lookback_window=lookback_window,
        time_bucket=time_bucket
    )
    return vwap.execute(order, market_data)


def execute_pov(
    order: OrderRequest,
    market_data: pd.DataFrame,
    participation_rate: float = 0.1
) -> List[OrderRequest]:
    """
    执行POV算法（便捷函数）。

    Args:
        order: 订单
        market_data: 市场数据
        participation_rate: 参与率

    Returns:
        子订单列表
    """
    pov = POVAlgorithm(target_participation_rate=participation_rate)
    return pov.execute(order, market_data)
