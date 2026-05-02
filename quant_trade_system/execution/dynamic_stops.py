"""
动态止损与止盈机制

提供智能止损止盈策略，包括：
1. 动态跟踪止盈（解决"利润过山车"问题）
2. 硬止损机制（解决"死扛"问题）
3. 买点过滤器（解决"半山腰抄底"问题）
4. 实盘监控与熔断
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class StopType(Enum):
    """止损类型"""
    HARD_STOP = "hard"              # 硬止损（绝对止损线）
    TRAILING_STOP = "trailing"      # 跟踪止损
    PROFIT_PROTECT = "profit_protect"  # 保护止盈
    TIME_STOP = "time"              # 时间止损


class StopAction(Enum):
    """止损动作"""
    SELL_ALL = "sell_all"           # 全部平仓
    SELL_HALF = "sell_half"         # 平仓一半
    SELL_PARTIAL = "sell_partial"    # 部分平仓
    ALERT_ONLY = "alert"            # 仅提示


@dataclass
class StopLossConfig:
    """止损配置"""

    stop_type: StopType
    threshold_pct: float            # 触发阈值（百分比）
    action: StopAction = StopAction.SELL_ALL
    activation_threshold: Optional[float] = None  # 激活条件（用于跟踪止盈）
    trail_pct: Optional[float] = None            # 跟踪回撤幅度（百分比）
    max_hold_days: Optional[int] = None          # 最大持仓天数
    cooldown_days: int = 0                       # 冷却期（天）

    def __post_init__(self):
        """验证配置"""
        if self.stop_type == StopType.TRAILING_STOP and not self.trail_pct:
            raise ValueError("跟踪止损必须设置 trail_pct")

        if self.stop_type == StopType.PROFIT_PROTECT and not self.activation_threshold:
            raise ValueError("保护止盈必须设置 activation_threshold")


@dataclass
class StopTrigger:
    """触发记录"""

    trigger_time: datetime
    stop_type: StopType
    current_price: float
    trigger_price: float
    unrealized_pnl_pct: float
    action_taken: StopAction
    reason: str


class DynamicStopManager:
    """
    动态止损管理器

    功能：
    1. 管理多个止损策略
    2. 实时监控持仓
    3. 触发止损信号
    4. 记录止损历史
    """

    def __init__(self):
        self.stop_configs: List[StopLossConfig] = []
        self.stop_history: List[StopTrigger] = []
        self.highest_price_since_entry: float = 0.0
        self.lowest_price_since_entry: float = float('inf')
        self.entry_date: Optional[datetime] = None
        self.is_trailing_active: bool = False

    def add_hard_stop(
        self,
        threshold_pct: float = -6.0,
        action: StopAction = StopAction.SELL_ALL,
    ) -> 'DynamicStopManager':
        """
        添加硬止损

        Args:
            threshold_pct: 止损阈值（如-6表示亏损6%止损）
            action: 触发后的动作
        """
        config = StopLossConfig(
            stop_type=StopType.HARD_STOP,
            threshold_pct=threshold_pct,
            action=action,
        )
        self.stop_configs.append(config)
        return self

    def add_trailing_profit_stop(
        self,
        activation_threshold: float = 8.0,    # 浮盈8%激活
        trail_pct: float = 3.0,               # 回撤3%止盈
        action: StopAction = StopAction.SELL_ALL,
    ) -> 'DynamicStopManager':
        """
        添加动态跟踪止盈

        Args:
            activation_threshold: 激活阈值（浮盈达到此值后激活跟踪止盈）
            trail_pct: 跟踪回撤幅度（从最高点回撤此比例后止盈）
            action: 触发后的动作
        """
        config = StopLossConfig(
            stop_type=StopType.PROFIT_PROTECT,
            threshold_pct=trail_pct,
            action=action,
            activation_threshold=activation_threshold,
            trail_pct=trail_pct,
        )
        self.stop_configs.append(config)
        return self

    def add_time_stop(
        self,
        max_hold_days: int = 30,
        action: StopAction = StopAction.SELL_ALL,
    ) -> 'DynamicStopManager':
        """
        添加时间止损

        Args:
            max_hold_days: 最大持仓天数
            action: 触发后的动作
        """
        config = StopLossConfig(
            stop_type=StopType.TIME_STOP,
            threshold_pct=0.0,
            action=action,
            max_hold_days=max_hold_days,
        )
        self.stop_configs.append(config)
        return self

    def reset(self, entry_price: float, entry_date: datetime):
        """重置管理器（新开仓时调用）"""
        self.highest_price_since_entry = entry_price
        self.lowest_price_since_entry = entry_price
        self.entry_date = entry_date
        self.is_trailing_active = False
        self.stop_history = []

    def check_stops(
        self,
        current_price: float,
        entry_price: float,
        current_date: datetime,
        position_type: str = 'long',
    ) -> Optional[StopTrigger]:
        """
        检查是否触发止损

        Args:
            current_price: 当前价格
            entry_price: 入场价格
            current_date: 当前日期
            position_type: 持仓类型 ('long' or 'short')

        Returns:
            StopTrigger if triggered, None otherwise
        """

        # 更新最高价/最低价
        if position_type == 'long':
            self.highest_price_since_entry = max(
                self.highest_price_since_entry, current_price
            )
        else:
            self.lowest_price_since_entry = min(
                self.lowest_price_since_entry, current_price
            )

        # 计算当前盈亏
        if position_type == 'long':
            unrealized_pnl_pct = (current_price / entry_price - 1) * 100
        else:
            unrealized_pnl_pct = (entry_price / current_price - 1) * 100

        # 检查每个止损配置
        for config in self.stop_configs:
            trigger = self._check_single_stop(
                config, current_price, entry_price,
                current_date, unrealized_pnl_pct, position_type
            )
            if trigger:
                self.stop_history.append(trigger)
                return trigger

        return None

    def _check_single_stop(
        self,
        config: StopLossConfig,
        current_price: float,
        entry_price: float,
        current_date: datetime,
        unrealized_pnl_pct: float,
        position_type: str,
    ) -> Optional[StopTrigger]:

        # 硬止损
        if config.stop_type == StopType.HARD_STOP:
            if unrealized_pnl_pct <= config.threshold_pct:
                return StopTrigger(
                    trigger_time=current_date,
                    stop_type=config.stop_type,
                    current_price=current_price,
                    trigger_price=entry_price * (1 + config.threshold_pct / 100),
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    action_taken=config.action,
                    reason=f"触发硬止损（亏损{abs(unrealized_pnl_pct):.2f}%）",
                )

        # 保护止盈（跟踪止损）
        elif config.stop_type == StopType.PROFIT_PROTECT:
            # 检查是否激活
            if not self.is_trailing_active:
                if unrealized_pnl_pct >= config.activation_threshold:
                    self.is_trailing_active = True
            else:
                # 已激活，检查是否触发
                if position_type == 'long':
                    drawdown_from_peak = (
                        current_price / self.highest_price_since_entry - 1
                    ) * 100
                else:
                    drawdown_from_peak = (
                        self.lowest_price_since_entry / current_price - 1
                    ) * 100

                if drawdown_from_peak <= -config.trail_pct:
                    return StopTrigger(
                        trigger_time=current_date,
                        stop_type=config.stop_type,
                        current_price=current_price,
                        trigger_price=self.highest_price_since_entry * (1 - config.trail_pct / 100),
                        unrealized_pnl_pct=unrealized_pnl_pct,
                        action_taken=config.action,
                        reason=(
                            f"触发跟踪止盈（浮盈{unrealized_pnl_pct:.2f}%，"
                            f"从最高点回撤{abs(drawdown_from_peak):.2f}%）"
                        ),
                    )

        # 时间止损
        elif config.stop_type == StopType.TIME_STOP and self.entry_date:
            hold_days = (current_date - self.entry_date).days
            if hold_days >= config.max_hold_days:
                return StopTrigger(
                    trigger_time=current_date,
                    stop_type=config.stop_type,
                    current_price=current_price,
                    trigger_price=current_price,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    action_taken=config.action,
                    reason=f"触发时间止损（持仓{hold_days}天）",
                )

        return None


class EntryFilter:
    """
    买入过滤器

    用于优化买入时机，避免"半山腰抄底"。
    提供多种确认信号和过滤条件。
    """

    def __init__(self):
        self.filters: List[Callable] = []

    def add_ma_trend_filter(
        self,
        fast_period: int = 20,
        slow_period: int = 60,
        min_slope: float = 0.0,
    ) -> 'EntryFilter':
        """
        添加均线趋势过滤器

        要求：短期均线在长期均线之上，且斜率为正
        """
        def filter_func(data: pd.DataFrame) -> bool:
            if len(data) < slow_period:
                return False

            data_fast = data['close'].rolling(window=fast_period).mean()
            data_slow = data['close'].rolling(window=slow_period).mean()

            # 多头排列
            is_bullish = data_fast.iloc[-1] > data_slow.iloc[-1]

            # 均线斜率
            recent_fast = data_fast.iloc[-5:]
            slope = (recent_fast.iloc[-1] - recent_fast.iloc[0]) / len(recent_fast)
            has_positive_slope = slope > min_slope

            return is_bullish and has_positive_slope

        self.filters.append(filter_func)
        return self

    def add_macd_confirmation(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> 'EntryFilter':
        """
        添加MACD金叉确认
        """
        def filter_func(data: pd.DataFrame) -> bool:
            if len(data) < slow_period + signal_period:
                return False

            # 计算MACD
            exp1 = data['close'].ewm(span=fast_period, adjust=False).mean()
            exp2 = data['close'].ewm(span=slow_period, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=signal_period, adjust=False).mean()

            # 金叉
            is_golden_cross = (
                macd.iloc[-1] > signal.iloc[-1] and
                macd.iloc[-2] <= signal.iloc[-2]
            )

            # MACD柱状图为正
            is_positive = macd.iloc[-1] - signal.iloc[-1] > 0

            return is_golden_cross or is_positive

        self.filters.append(filter_func)
        return self

    def add_volume_breakout_filter(
        self,
        volume_ma_period: int = 20,
        min_volume_ratio: float = 1.5,
    ) -> 'EntryFilter':
        """
        添加放量突破确认

        要求：成交量大于均值的min_volume_ratio倍
        """
        def filter_func(data: pd.DataFrame) -> bool:
            if len(data) < volume_ma_period:
                return False

            vol_ma = data['volume'].rolling(window=volume_ma_period).mean()
            current_vol = data['volume'].iloc[-1]

            return current_vol > vol_ma.iloc[-1] * min_volume_ratio

        self.filters.append(filter_func)
        return self

    def add_price_above_filter(
        self,
        ma_period: int = 60,
        min_distance_pct: float = 2.0,
    ) -> 'EntryFilter':
        """
        添加价格高于均线过滤器

        要求：当前价格高于均线至少min_distance_pct%
        """
        def filter_func(data: pd.DataFrame) -> bool:
            if len(data) < ma_period:
                return False

            ma = data['close'].rolling(window=ma_period).mean().iloc[-1]
            current_price = data['close'].iloc[-1]

            distance_pct = (current_price / ma - 1) * 100

            return distance_pct >= min_distance_pct

        self.filters.append(filter_func)
        return self

    def add_rsi_filter(
        self,
        period: int = 14,
        min_rsi: float = 30.0,
        max_rsi: float = 70.0,
    ) -> 'EntryFilter':
        """
        添加RSI过滤器

        要求：RSI在[min_rsi, max_rsi]范围内（避免超买超卖）
        """
        def filter_func(data: pd.DataFrame) -> bool:
            if len(data) < period:
                return False

            # 计算RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            current_rsi = rsi.iloc[-1]

            return min_rsi <= current_rsi <= max_rsi

        self.filters.append(filter_func)
        return self

    def should_enter(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        判断是否应该买入

        Args:
            data: 价格数据（包含OHLCV）

        Returns:
            (should_enter, reason): 是否买入及原因
        """

        if not self.filters:
            return True, "无过滤条件"

        passed_filters = []
        failed_filters = []

        for i, filter_func in enumerate(self.filters):
            try:
                result = filter_func(data)
                if result:
                    passed_filters.append(f"过滤器{i+1}")
                else:
                    failed_filters.append(f"过滤器{i+1}")
            except Exception as e:
                failed_filters.append(f"过滤器{i+1}（错误: {e}）")

        should_enter = len(failed_filters) == 0
        reason = f"通过: {', '.join(passed_filters)}" if should_enter else f"被拒绝: {', '.join(failed_filters)}"

        return should_enter, reason


class OptimizedBacktester:
    """
    集成动态止损止盈的回测器

    对比原始策略与优化后策略的表现
    """

    def __init__(
        self,
        original_trades: List[Dict],
        price_data: pd.DataFrame,
    ):
        """
        Args:
            original_trades: 原始交易列表
            price_data: 价格数据
        """
        self.original_trades = original_trades
        self.price_data = price_data
        self.optimized_trades = []

    def run_with_stops(
        self,
        stop_configs: Optional[List[StopLossConfig]] = None,
        entry_filter: Optional[EntryFilter] = None,
    ) -> Dict:
        """
        运行优化回测

        Args:
            stop_configs: 止损配置列表
            entry_filter: 买入过滤器

        Returns:
            对比结果字典
        """

        # 如果没有提供配置，使用推荐配置
        if stop_configs is None:
            stop_configs = [
                StopLossConfig(
                    stop_type=StopType.HARD_STOP,
                    threshold_pct=-6.0,
                    action=StopAction.SELL_ALL,
                ),
                StopLossConfig(
                    stop_type=StopType.PROFIT_PROTECT,
                    threshold_pct=3.0,
                    action=StopAction.SELL_ALL,
                    activation_threshold=8.0,
                    trail_pct=3.0,
                ),
            ]

        # 创建止损管理器
        stop_manager = DynamicStopManager()
        stop_manager.stop_configs = stop_configs

        # 运行优化回测
        self.optimized_trades = []

        for trade in self.original_trades:
            entry_date = pd.to_datetime(trade['entry_date'])
            entry_price = float(trade['entry_price'])

            # 买入过滤
            if entry_filter:
                pre_entry_data = self.price_data[
                    self.price_data.index < entry_date
                ].tail(60)  # 使用前60根K线

                if len(pre_entry_data) > 0:
                    should_enter, reason = entry_filter.should_enter(pre_entry_data)
                    if not should_enter:
                        # 跳过此交易
                        continue

            # 模拟持仓过程
            stop_manager.reset(entry_price, entry_date)

            position_data = self.price_data[
                self.price_data.index >= entry_date
            ]

            original_exit_date = pd.to_datetime(trade['exit_date'])

            for idx, current_date in enumerate(position_data.index):
                current_price = position_data.loc[current_date, 'close']

                # 检查止损
                trigger = stop_manager.check_stops(
                    current_price, entry_price, current_date, 'long'
                )

                if trigger:
                    # 触发止损，提前退出
                    self.optimized_trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'exit_reason': trigger.reason,
                        'original_exit_date': original_exit_date,
                    })
                    break

                # 到达原定退出日
                if current_date >= original_exit_date:
                    self.optimized_trades.append({
                        'entry_date': entry_date,
                        'exit_date': original_exit_date,
                        'entry_price': entry_price,
                        'exit_price': float(trade['exit_price']),
                        'exit_reason': 'original_signal',
                        'original_exit_date': original_exit_date,
                    })
                    break

        # 生成对比报告
        return self._generate_comparison()

    def _generate_comparison(self) -> Dict:
        """生成优化前后的对比报告"""

        # 计算原始策略指标
        original_pnl = [
            (t['exit_price'] / t['entry_price'] - 1) * 100
            for t in self.original_trades
        ]
        original_win_rate = sum(1 for p in original_pnl if p > 0) / len(original_pnl)
        original_avg_pnl = np.mean(original_pnl)
        original_total_return = np.prod([1 + p/100 for p in original_pnl]) - 1

        # 计算优化后指标
        if self.optimized_trades:
            optimized_pnl = [
                (t['exit_price'] / t['entry_price'] - 1) * 100
                for t in self.optimized_trades
            ]
            optimized_win_rate = sum(1 for p in optimized_pnl if p > 0) / len(optimized_pnl)
            optimized_avg_pnl = np.mean(optimized_pnl)
            optimized_total_return = np.prod([1 + p/100 for p in optimized_pnl]) - 1
        else:
            optimized_pnl = []
            optimized_win_rate = 0
            optimized_avg_pnl = 0
            optimized_total_return = 0

        return {
            'original': {
                'trade_count': len(self.original_trades),
                'win_rate': original_win_rate,
                'avg_pnl_pct': original_avg_pnl,
                'total_return_pct': original_total_return,
                'pnls': original_pnl,
            },
            'optimized': {
                'trade_count': len(self.optimized_trades),
                'win_rate': optimized_win_rate,
                'avg_pnl_pct': optimized_avg_pnl,
                'total_return_pct': optimized_total_return,
                'pnls': optimized_pnl,
            },
            'improvement': {
                'win_rate_delta': optimized_win_rate - original_win_rate,
                'avg_pnl_delta': optimized_avg_pnl - original_avg_pnl,
                'total_return_delta': optimized_total_return - original_total_return,
            }
        }


def create_recommended_stop_manager() -> DynamicStopManager:
    """
    创建推荐配置的止损管理器

    推荐配置：
    1. 硬止损：-6% 或 -8%
    2. 动态跟踪止盈：浮盈8%激活，回撤3%止盈
    3. 时间止损：最长持仓30天
    """
    manager = DynamicStopManager()
    manager.add_hard_stop(threshold_pct=-6.0)
    manager.add_trailing_profit_stop(
        activation_threshold=8.0,
        trail_pct=3.0,
    )
    manager.add_time_stop(max_hold_days=30)
    return manager


def create_recommended_entry_filter() -> EntryFilter:
    """
    创建推荐配置的买入过滤器

    推荐配置：
    1. 均线多头排列（20日 > 60日）
    2. MACD金叉确认
    3. 放量突破（成交量 > 1.5倍均值）
    """
    filter_obj = EntryFilter()
    filter_obj.add_ma_trend_filter(fast_period=20, slow_period=60)
    filter_obj.add_macd_confirmation()
    filter_obj.add_volume_breakout_filter(min_volume_ratio=1.5)
    return filter_obj
