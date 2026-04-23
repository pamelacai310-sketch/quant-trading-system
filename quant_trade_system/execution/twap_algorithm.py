"""
TWAP算法 - Time Weighted Average Price

时间加权平均价格执行算法，将大额订单拆分为多个小订单，
在指定时间窗口内平均执行，降低市场冲击。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .execution_algorithms import ExecutionAlgorithm, OrderRequest


class TWAPAlgorithm(ExecutionAlgorithm):
    """
    TWAP算法 - 时间加权平均价格执行。

    特点:
    - 将订单拆分为N个等量子订单
    - 在指定时间窗口内均匀执行
    - 降低市场冲击
    - 适合大额订单和流动性差的品种
    """

    def __init__(
        self,
        n_slices: int = 10,
        time_window: str = '1H',
        start_time: Optional[datetime] = None,
        randomize_timing: bool = False,
        randomization_range: float = 0.2
    ):
        """
        初始化TWAP算法。

        Args:
            n_slices: 拆分数量
            time_window: 时间窗口 (如 '1H', '30min', '1D')
            start_time: 开始时间 (None表示立即开始)
            randomize_timing: 是否随机化执行时间
            randomization_range: 随机化范围 (±20%)
        """
        super().__init__(name='twap')
        self.parameters = {
            'n_slices': n_slices,
            'time_window': time_window,
            'start_time': start_time,
            'randomize_timing': randomize_timing,
            'randomization_range': randomization_range
        }

    def execute(
        self,
        order: OrderRequest,
        market_data: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> List[OrderRequest]:
        """
        执行TWAP算法。

        Args:
            order: 父订单
            market_data: 市场数据
            current_price: 当前价格

        Returns:
            子订单列表
        """
        n_slices = self.parameters['n_slices']
        time_window_str = self.parameters['time_window']
        randomize_timing = self.parameters['randomize_timing']
        randomization_range = self.parameters['randomization_range']

        # 计算每个子订单的数量
        qty_per_slice = order.quantity / n_slices

        # 计算时间间隔
        time_delta = pd.Timedelta(time_window_str)
        time_interval = time_delta / n_slices

        # 确定开始时间
        start_time = self.parameters['start_time']
        if start_time is None:
            start_time = datetime.now()

        # 生成子订单
        child_orders = []
        for i in range(n_slices):
            # 计算执行时间
            if randomize_timing:
                # 添加随机化，使执行时间更难预测
                random_offset = np.random.uniform(
                    -randomization_range,
                    randomization_range
                )
                execution_time = start_time + (time_interval * i * (1 + random_offset))
            else:
                execution_time = start_time + (time_interval * i)

            # 创建子订单
            child_order = OrderRequest(
                symbol=order.symbol,
                side=order.side,
                quantity=qty_per_slice,
                order_type='market',  # TWAP通常使用市价单
                strategy_id=order.strategy_id,
                metadata={
                    'parent_order_id': order.metadata.get('order_id', 'unknown'),
                    'slice_number': i,
                    'total_slices': n_slices,
                    'algorithm': 'TWAP',
                    'execution_time': execution_time.isoformat(),
                    'target_time_window': time_window_str,
                    'slice_ratio': 1.0 / n_slices
                }
            )

            child_orders.append(child_order)

        return child_orders

    def estimate_execution_schedule(
        self,
        order: OrderRequest,
        start_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        估算执行时间表。

        Args:
            order: 订单
            start_time: 开始时间

        Returns:
            执行时间表
        """
        n_slices = self.parameters['n_slices']
        time_window_str = self.parameters['time_window']
        time_delta = pd.Timedelta(time_window_str)
        time_interval = time_delta / n_slices

        if start_time is None:
            start_time = datetime.now()

        schedule = []
        for i in range(n_slices):
            execution_time = start_time + (time_interval * i)
            schedule.append({
                'slice_number': i,
                'execution_time': execution_time.isoformat(),
                'target_quantity': order.quantity / n_slices,
                'cumulative_quantity': order.quantity * (i + 1) / n_slices
            })

        return schedule

    def calculate_expected_cost(
        self,
        order: OrderRequest,
        market_data: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        计算预期执行成本。

        Args:
            order: 订单
            market_data: 市场数据
            current_price: 当前价格

        Returns:
            预期成本分析
        """
        if current_price is None:
            current_price = market_data['close'].iloc[-1]

        order_value = order.quantity * current_price
        avg_daily_volume = market_data['volume'].rolling(20).mean().iloc[-1]

        # 估算市场冲击
        participation_rate = order.quantity / avg_daily_volume
        if participation_rate > 0.2:
            market_impact = 'high'
            estimated_slippage_bps = 30
        elif participation_rate > 0.1:
            market_impact = 'medium'
            estimated_slippage_bps = 15
        else:
            market_impact = 'low'
            estimated_slippage_bps = 5

        # TWAP降低市场冲击的效果
        twap_reduction_factor = 1.0 / np.sqrt(self.parameters['n_slices'])
        estimated_slippage_bps *= twap_reduction_factor

        total_cost = order_value * (estimated_slippage_bps / 10000)

        return {
            'order_value': order_value,
            'market_impact': market_impact,
            'participation_rate': participation_rate,
            'estimated_slippage_bps': estimated_slippage_bps,
            'twap_reduction_factor': twap_reduction_factor,
            'total_cost': total_cost,
            'cost_as_pct': total_cost / order_value if order_value > 0 else 0
        }


class AdaptiveTWAPAlgorithm(TWAPAlgorithm):
    """
    自适应TWAP算法 - 根据市场流动性调整执行速度。

    特点:
    - 高流动性时加快执行
    - 低流动性时减慢执行
    - 实时监控市场深度
    """

    def __init__(
        self,
        base_n_slices: int = 10,
        min_slices: int = 5,
        max_slices: int = 30,
        liquidity_threshold: float = 0.1
    ):
        """
        初始化自适应TWAP算法。

        Args:
            base_n_slices: 基础拆分数量
            min_slices: 最小拆分数量
            max_slices: 最大拆分数量
            liquidity_threshold: 流动性阈值 (成交量占平均成交量的比例)
        """
        super().__init__(n_slices=base_n_slices)
        self.name = 'adaptive_twap'
        self.parameters.update({
            'min_slices': min_slices,
            'max_slices': max_slices,
            'liquidity_threshold': liquidity_threshold
        })

    def execute(
        self,
        order: OrderRequest,
        market_data: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> List[OrderRequest]:
        """
        执行自适应TWAP算法。

        根据当前市场流动性动态调整拆分数量。
        """
        # 计算当前流动性
        current_volume = market_data['volume'].iloc[-1]
        avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
        liquidity_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # 动态调整拆分数量
        base_n_slices = self.parameters['n_slices']
        min_slices = self.parameters['min_slices']
        max_slices = self.parameters['max_slices']

        if liquidity_ratio > 1.5:
            # 高流动性: 减少拆分，加快执行
            adaptive_n_slices = max(min_slices, int(base_n_slices * 0.7))
        elif liquidity_ratio < 0.5:
            # 低流动性: 增加拆分，减慢执行
            adaptive_n_slices = min(max_slices, int(base_n_slices * 1.3))
        else:
            # 正常流动性
            adaptive_n_slices = base_n_slices

        # 更新参数
        self.parameters['adaptive_n_slices'] = adaptive_n_slices
        self.parameters['actual_n_slices'] = adaptive_n_slices
        self.parameters['liquidity_ratio'] = liquidity_ratio

        # 使用调整后的拆分数量执行
        return super().execute(
            order._replace(quantity=order.quantity),  # 使用原始订单
            market_data,
            current_price
        )


# 便捷函数
def execute_twap(
    order: OrderRequest,
    market_data: pd.DataFrame,
    n_slices: int = 10,
    time_window: str = '1H'
) -> List[OrderRequest]:
    """
    执行TWAP算法（便捷函数）。

    Args:
        order: 订单
        market_data: 市场数据
        n_slices: 拆分数量
        time_window: 时间窗口

    Returns:
        子订单列表
    """
    twap = TWAPAlgorithm(n_slices=n_slices, time_window=time_window)
    return twap.execute(order, market_data)
