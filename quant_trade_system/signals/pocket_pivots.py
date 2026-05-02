"""
口袋支点（Pocket Pivot）信号检测器

实现Gil Morales和Chris Kacher提出的口袋支点概念：
- 在基底未正式突破时的提前买入信号
- 股价突破短期均线（如10日均线）
- 当日成交量大于过去10日最高下跌日成交量
- 表明机构在悄悄介入，即将突破

参考：Gil Morales & Chris Kacher, "Trade Like an O'Neil Disciple"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PocketPivotSignal:
    """口袋支点信号"""
    date: datetime
    price: float
    volume: float
    volume_ratio: float         # 成交量比率（相对前期下跌日）
    ma_broken: str               # 突破的均线类型（如"10日均线"）

    # 止损位
    stop_loss_price: float       # 止损价格（通常在10日均线下方）

    # 信号强度
    strength_score: float       # 信号强度得分（0-1）

    # 附加信息
    description: str = ""
    context: str = ""            # 市场环境描述


class PocketPivotDetector:
    """
    口袋支点检测器

    识别欧奈尔派提前买入信号：
    1. 股价仍处于上升趋势或基底末端
    2. 突破短期均线（10日或50日）
    3. 当日成交量大于过去10日所有下跌日成交量
    """

    def __init__(
        self,
        ma_period: int = 10,                 # 均线周期（默认10日）
        lookback_days: int = 10,             # 成交量回溯天数
        min_price_gain: float = 1.0,         # 最小涨幅（%）
        min_volume_ratio: float = 1.0,       # 最小成交量比率
        min_rsi: int = 50,                   # 最小RSI（避免超买）
        max_rsi: int = 80,                   # 最大RSI
        require_uptrend: bool = True,        # 是否要求上升趋势
        uptrend_ma_short: int = 50,          # 上升趋势短期均线
        uptrend_ma_long: int = 200,          # 上升趋势长期均线
    ):
        """
        Args:
            ma_period: 均线周期（通常10日或50日）
            lookback_days: 成交量回溯天数
            min_price_gain: 最小涨幅（%），避免微小波动
            min_volume_ratio: 最小成交量比率
            min_rsi: 最小RSI，避免在弱势时买入
            max_rsi: 最大RSI，避免在超买时买入
            require_uptrend: 是否要求处于上升趋势
            uptrend_ma_short: 上升趋势判断短期均线
            uptrend_ma_long: 上升趋势判断长期均线
        """
        self.ma_period = ma_period
        self.lookback_days = lookback_days
        self.min_price_gain = min_price_gain
        self.min_volume_ratio = min_volume_ratio
        self.min_rsi = min_rsi
        self.max_rsi = max_rsi
        self.require_uptrend = require_uptrend
        self.uptrend_ma_short = uptrend_ma_short
        self.uptrend_ma_long = uptrend_ma_long

    def detect_signals(
        self,
        data: pd.DataFrame,
        lookback_period: int = 252,
    ) -> List[PocketPivotSignal]:
        """
        检测口袋支点信号

        Args:
            data: 价格数据（OHLCV）
            lookback_period: 最大回溯周期

        Returns:
            口袋支点信号列表
        """
        signals = []

        if len(data) < self.lookback_days + 10:
            return signals

        # 计算技术指标
        data = data.copy()
        data['ma'] = data['close'].rolling(window=self.ma_period).mean()
        data['ma_short'] = data['close'].rolling(window=self.uptrend_ma_short).mean()
        data['ma_long'] = data['close'].rolling(window=self.uptrend_ma_long).mean()

        # 计算RSI
        if len(data) >= 14:
            data['rsi'] = self._calculate_rsi(data['close'], 14)
        else:
            data['rsi'] = 50.0  # 默认中性值

        # 寻找口袋支点
        for i in range(self.lookback_days, len(data)):
            current_data = data.iloc[i]
            lookback_data = data.iloc[i-self.lookback_days:i]

            # 1. 检查上升趋势要求
            if self.require_uptrend:
                if (
                    current_data['close'] < current_data['ma_short'] or
                    current_data['close'] < current_data['ma_long'] or
                    current_data['ma_short'] < current_data['ma_long']
                ):
                    continue

            # 2. 检查是否突破均线
            if current_data['close'] <= current_data['ma']:
                continue

            # 检查突破幅度
            price_gain = (current_data['close'] / current_data['ma'] - 1) * 100
            if price_gain < self.min_price_gain:
                continue

            # 3. 检查成交量条件
            volume_check, volume_ratio, highest_down_vol = self._check_volume_condition(
                current_data, lookback_data
            )

            if not volume_check:
                continue

            # 4. 检查RSI条件
            if not (self.min_rsi <= current_data['rsi'] <= self.max_rsi):
                continue

            # 5. 检查是否为有效突破（不是假突破）
            if not self._is_valid_breakout(data, i):
                continue

            # 计算止损位（10日均线下方或最近低点）
            stop_loss_price = self._calculate_stop_loss(data, i)

            # 计算信号强度
            strength_score = self._calculate_strength_score(
                price_gain, volume_ratio, current_data['rsi']
            )

            # 生成描述
            description = (
                f"口袋支点：{self.ma_period}日均线突破，"
                f"涨幅{price_gain:.2f}%，"
                f"成交量{volume_ratio:.1f}倍"
            )

            # 生成市场环境描述
            context = self._generate_context_description(data, i)

            signal = PocketPivotSignal(
                date=data.index[i],
                price=current_data['close'],
                volume=current_data['volume'],
                volume_ratio=volume_ratio,
                ma_broken=f"{self.ma_period}日均线",
                stop_loss_price=stop_loss_price,
                strength_score=strength_score,
                description=description,
                context=context,
            )

            signals.append(signal)

        return signals

    def _check_volume_condition(
        self,
        current_data: pd.Series,
        lookback_data: pd.DataFrame,
    ) -> Tuple[bool, float, float]:
        """
        检查成交量条件

        要求：当日成交量 > 过去N天所有下跌日的最大成交量
        """
        current_vol = current_data['volume']

        # 找出过去N天的下跌日
        down_days = lookback_data[lookback_data['close'] < lookback_data['open']]

        if len(down_days) == 0:
            # 没有下跌日，对比平均成交量
            avg_vol = lookback_data['volume'].mean()
            if avg_vol == 0:
                return False, 0.0, 0.0
            volume_ratio = current_vol / avg_vol
            return volume_ratio >= self.min_volume_ratio, volume_ratio, avg_vol

        # 找出下跌日的最大成交量
        highest_down_vol = down_days['volume'].max()

        if highest_down_vol == 0:
            return False, 0.0, 0.0

        volume_ratio = current_vol / highest_down_vol

        return volume_ratio >= self.min_volume_ratio, volume_ratio, highest_down_vol

    def _is_valid_breakout(
        self,
        data: pd.DataFrame,
        current_idx: int,
    ) -> bool:
        """
        检查是否为有效突破（排除假突破）

        方法：
        1. 检查后续几天是否维持突破
        2. 检查是否有放量确认
        """
        # 检查是否有后续确认
        if current_idx + 3 >= len(data):
            # 数据不足，保守处理
            return True

        future_data = data.iloc[current_idx+1:current_idx+4]

        # 至少有2天收在均线上方
        days_above_ma = (
            future_data['close'] > future_data['ma'].iloc[0]
        ).sum()

        if days_above_ma < 2:
            return False

        return True

    def _calculate_stop_loss(
        self,
        data: pd.DataFrame,
        current_idx: int,
    ) -> float:
        """
        计算止损价格

        方法：
        1. 10日均线下方（如果使用10日均线）
        2. 或最近回调低点
        """
        current_ma = data['ma'].iloc[current_idx]

        # 方法1：均线下方一定幅度（如2%）
        stop_from_ma = current_ma * 0.98

        # 方法2：最近低点（过去10天）
        recent_low = data['low'].iloc[max(0, current_idx-10):current_idx+1].min()

        # 取两者中较高的（更接近入场价）
        return max(stop_from_ma, recent_low)

    def _calculate_strength_score(
        self,
        price_gain: float,
        volume_ratio: float,
        rsi: float,
    ) -> float:
        """
        计算信号强度得分（0-1）

        考虑因素：
        1. 价格涨幅（适度最好，太大可能是追高）
        2. 成交量比率
        3. RSI水平（适中最好）
        """
        # 价格涨幅得分（1-5%最佳）
        if 1 <= price_gain <= 5:
            price_score = 1.0
        elif 5 < price_gain <= 10:
            price_score = 0.7
        elif price_gain < 1:
            price_score = 0.5
        else:
            price_score = 0.3

        # 成交量比率得分
        if volume_ratio >= 2.0:
            vol_score = 1.0
        elif volume_ratio >= 1.5:
            vol_score = 0.8
        elif volume_ratio >= 1.2:
            vol_score = 0.6
        else:
            vol_score = 0.4

        # RSI得分（50-70最佳）
        if 50 <= rsi <= 70:
            rsi_score = 1.0
        elif 40 <= rsi < 50 or 70 < rsi <= 80:
            rsi_score = 0.7
        else:
            rsi_score = 0.4

        # 加权平均
        return (price_score * 0.3 + vol_score * 0.5 + rsi_score * 0.2)

    def _calculate_rsi(
        self,
        prices: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _generate_context_description(
        self,
        data: pd.DataFrame,
        current_idx: int,
    ) -> str:
        """生成市场环境描述"""
        current = data.iloc[current_idx]

        # 检查趋势
        if current['ma_short'] > current['ma_long']:
            trend = "上升趋势"
        elif current['ma_short'] < current['ma_long']:
            trend = "下降趋势"
        else:
            trend = "震荡走势"

        # 检查相对位置
        position_52w = (
            (current['close'] / data['close'].iloc[-252:].max() - 1) * 100
            if len(data) >= 252 else 0
        )

        return f"{trend}，距52周高点{position_52w:+.1f}%"

    def detect_multiple_stocks(
        self,
        stocks_data: Dict[str, pd.DataFrame],
        min_strength: float = 0.6,
    ) -> Dict[str, List[PocketPivotSignal]]:
        """
        批量检测多只股票的口袋支点信号

        Args:
            stocks_data: {股票代码: 价格数据}
            min_strength: 最小信号强度

        Returns:
            {股票代码: [信号列表]} 字典
        """
        results = {}

        for symbol, data in stocks_data.items():
            try:
                signals = self.detect_signals(data)
                # 过滤信号强度
                strong_signals = [
                    s for s in signals
                    if s.strength_score >= min_strength
                ]

                if strong_signals:
                    results[symbol] = strong_signals

            except Exception as e:
                print(f"检测{symbol}的口袋支点时出错: {e}")
                continue

        return results


def is_pocket_pivot_today(
    data: pd.DataFrame,
    ma_period: int = 10,
    lookback_days: int = 10,
    min_volume_ratio: float = 1.5,
) -> Tuple[bool, Optional[PocketPivotSignal]]:
    """
    快速检查今日是否为口袋支点

    Args:
        data: 价格数据（OHLCV）
        ma_period: 均线周期
        lookback_days: 成交量回溯天数
        min_volume_ratio: 最小成交量比率

    Returns:
        (是否为口袋支点, 信号对象)
    """
    if len(data) < lookback_days + 1:
        return False, None

    detector = PocketPivotDetector(
        ma_period=ma_period,
        lookback_days=lookback_days,
        min_volume_ratio=min_volume_ratio,
    )

    signals = detector.detect_signals(data)

    if signals and signals[-1].date == data.index[-1]:
        return True, signals[-1]

    return False, None
