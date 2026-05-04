"""
周波段T0-T5短线策略

核心特点：
1. 持仓周期T0-T5，不过周末
2. 长期看好标的可每周反复做波段
3. 本金100万，目标每周净赚2万（2%）

策略逻辑：
- T0买入/T0-T5卖出，最晚周五平仓
- 在每周相对低位接回长期看好的标的
- 严格止损止盈，快速周转
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class DayOfWeek(Enum):
    """星期枚举"""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"


@dataclass
class SwingPosition:
    """波段持仓"""
    symbol: str
    name: str
    side: PositionSide
    entry_date: datetime
    entry_price: float
    shares: int
    stop_loss: float
    take_profit: float
    max_hold_days: int = 5
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    is_long_term_favor: bool = False  # 是否长期看好
    weekly_reentry_count: int = 0  # 每周重新进入次数

    @property
    def hold_days(self) -> int:
        """持仓天数"""
        if self.exit_date:
            return (self.exit_date - self.entry_date).days
        return (datetime.now() - self.entry_date).days

    @property
    def days_until_exit(self) -> int:
        """距离必须平仓的天数"""
        if self.exit_date:
            return 0

        # 计算到周五的天数
        current_day = self.entry_date.weekday()
        days_to_friday = 4 - current_day  # 周五=4

        # 加上最大持仓天数限制
        return min(self.max_hold_days - self.hold_days, days_to_friday)

    @property
    def current_value(self) -> float:
        """当前市值"""
        if self.exit_price:
            return self.exit_price * self.shares
        return self.entry_price * self.shares  # 简化计算

    @property
    def pnl(self) -> float:
        """盈亏"""
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares * (1 if self.side == PositionSide.LONG else -1)

    @property
    def pnl_pct(self) -> float:
        """盈亏百分比"""
        if self.exit_price is None:
            return 0.0
        return (self.exit_price / self.entry_price - 1) * (100 if self.side == PositionSide.LONG else -100)


@dataclass
class WeeklyTarget:
    """每周目标"""
    week_start: datetime
    week_end: datetime
    target_profit: float = 20_000  # 目标2万
    initial_capital: float = 1_000_000  # 本金100万
    target_return_pct: float = 0.02  # 2%

    current_capital: float = 1_000_000
    realized_profit: float = 0.0
    open_positions: List[SwingPosition] = field(default_factory=list)

    @property
    def progress_pct(self) -> float:
        """目标完成进度"""
        return self.realized_profit / self.target_profit

    @property
    def return_pct(self) -> float:
        """当前收益率"""
        return self.realized_profit / self.initial_capital

    @property
    def remaining_capital(self) -> float:
        """剩余可用资金"""
        used = sum(p.current_value for p in self.open_positions)
        return self.current_capital - used


class WeeklySwingStrategy:
    """
    周波段T0-T5短线策略

    核心规则：
    1. 持仓周期T0-T5，不过周末
    2. 长期看好标的可每周反复做波段
    3. 目标每周净赚2万（2%）
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        weekly_target: float = 20_000,
        max_positions: int = 5,
        max_hold_days: int = 5,
        base_stop_loss: float = 0.03,  # 3%止损
        base_take_profit: float = 0.06,  # 6%止盈
    ):
        """
        初始化策略

        参数:
            initial_capital: 初始资金（默认100万）
            weekly_target: 每周目标利润（默认2万）
            max_positions: 最大持仓数量
            max_hold_days: 最大持仓天数（默认5天）
            base_stop_loss: 基础止损幅度（默认3%）
            base_take_profit: 基础止盈幅度（默认6%）
        """
        self.initial_capital = initial_capital
        self.weekly_target = weekly_target
        self.target_return_pct = weekly_target / initial_capital
        self.max_positions = max_positions
        self.max_hold_days = max_hold_days
        self.base_stop_loss = base_stop_loss
        self.base_take_profit = base_take_profit

        # 长期看好的标的池（可反复做波段）
        self.long_term_favorites = [
            'AAPL',  # 苹果
            'MSFT',  # 微软
            'GOOGL', # 谷歌
            'TSLA',  # 特斯拉
            'NVDA',  # 英伟达
            'META',  # Meta
            'AMZN',  # 亚马逊
            '0700.HK',  # 腾讯
            '9988.HK',  # 阿里巴巴
            '3690.HK',  # 美团
        ]

        # 期货合约池（高杠杆、高波动）
        self.futures_contracts = [
            'ES',  # 标普500 E-mini
            'NQ',  # 纳斯达克100 E-mini
            'YM',  # 道琼斯E-mini
            'CL',  # 原油
            'GC',  # 黄金
            'SI',  # 白银
            'HG',  # 铜
            'ZC',  # 玉米
            'ZW',  # 小麦
            'RB',  # 螺纹钢（国内）
        ]

        # 持仓记录
        self.positions: List[SwingPosition] = []
        self.closed_positions: List[SwingPosition] = []

        # 每周目标追踪
        self.weekly_targets: List[WeeklyTarget] = []

    def is_trading_day(self, date: datetime) -> bool:
        """判断是否是交易日（周一到周五）"""
        return date.weekday() < 5  # 0-4表示周一到周五

    def get_days_until_friday(self, date: datetime) -> int:
        """计算到周五的天数"""
        current_day = date.weekday()
        return max(0, 4 - current_day)  # 周五=4

    def can_enter_position(self, date: datetime) -> bool:
        """判断是否可以开仓"""
        # 1. 必须是交易日
        if not self.is_trading_day(date):
            return False

        # 2. 不能在周五晚开新仓（会过周末）
        if date.weekday() == 4 and date.hour >= 15:  # 周五下午3点后
            return False

        # 3. 检查持仓数量限制
        active_positions = [p for p in self.positions if p.exit_date is None]
        if len(active_positions) >= self.max_positions:
            return False

        return True

    def should_exit_position(self, position: SwingPosition, current_date: datetime,
                            current_price: float) -> Tuple[bool, Optional[str]]:
        """
        判断是否应该平仓

        返回: (是否平仓, 原因)
        """
        # 1. 检查止损
        if position.side == PositionSide.LONG:
            loss_pct = (current_price / position.entry_price - 1)
        else:
            loss_pct = (position.entry_price / current_price - 1)

        if loss_pct <= -position.stop_loss:
            return True, f"触发止损: {loss_pct*100:.2f}%"

        # 2. 检查止盈
        if loss_pct >= position.take_profit:
            return True, f"触发止盈: {loss_pct*100:.2f}%"

        # 3. 检查持仓天数
        hold_days = (current_date - position.entry_date).days
        if hold_days >= position.max_hold_days:
            return True, f"达到最大持仓天数: {hold_days}天"

        # 4. 检查是否到周五必须平仓
        days_to_friday = self.get_days_until_friday(current_date)
        if days_to_friday == 0 and current_date.hour >= 15:  # 周五下午3点
            return True, "周五收盘，必须平仓"

        # 5. 检查是否接近周五（提前平仓）
        if days_to_friday <= 1 and position.hold_days >= 3:
            return True, f"接近周五，持仓{position.hold_days}天，提前平仓"

        return False, None

    def calculate_position_size(self, price: float, capital: float,
                               risk_per_trade: float = 0.02) -> int:
        """
        计算持仓数量

        参数:
            price: 价格
            capital: 可用资金
            risk_per_trade: 单笔风险（默认2%）
        """
        # 基于风险的仓位计算
        risk_amount = capital * risk_per_trade
        stop_loss_amount = price * self.base_stop_loss

        shares = int(risk_amount / stop_loss_amount)

        # 确保不超过可用资金
        max_shares = int(capital / price)
        shares = min(shares, max_shares)

        return max(0, shares)

    def scan_stocks(self, market_data: Dict[str, pd.DataFrame],
                   current_date: datetime) -> List[Dict[str, Any]]:
        """
        扫描股票，寻找交易机会

        参数:
            market_data: 市场数据字典 {symbol: DataFrame}
            current_date: 当前日期

        返回:
            交易机会列表
        """
        opportunities = []

        for symbol, data in market_data.items():
            if len(data) < 20:
                continue

            # 获取当前价格
            current_price = data['Close'].iloc[-1]

            # 计算技术指标
            ma5 = data['Close'].rolling(5).mean().iloc[-1]
            ma10 = data['Close'].rolling(10).mean().iloc[-1]
            ma20 = data['Close'].rolling(20).mean().iloc[-1]

            # 计算RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            # 计算波动率
            returns = data['Close'].pct_change().tail(20)
            volatility = returns.std() * np.sqrt(252)

            # 判断是否是长期看好标的
            is_favorite = symbol in self.long_term_favorites

            # 判断是否是期货
            is_future = symbol in self.futures_contracts

            # 买入信号（做多）
            long_signals = 0

            # 1. 均线多头排列
            if ma5 > ma10 > ma20:
                long_signals += 1

            # 2. 价格突破MA5
            if current_price > ma5:
                long_signals += 1

            # 3. RSI超卖回升
            if current_rsi < 40:
                long_signals += 1
            elif 30 < current_rsi < 50:
                long_signals += 0.5

            # 4. 波动率适中
            if 0.15 < volatility < 0.6:
                long_signals += 1

            # 5. 长期看好标的加分
            if is_favorite:
                long_signals += 1

            # 6. 期货高波动加分
            if is_future:
                long_signals += 0.5

            # 评估机会
            if long_signals >= 3:
                opportunity = {
                    'symbol': symbol,
                    'type': 'long',
                    'price': current_price,
                    'signals': long_signals,
                    'rsi': current_rsi,
                    'volatility': volatility,
                    'is_favorite': is_favorite,
                    'is_future': is_future,
                    'score': long_signals,
                }
                opportunities.append(opportunity)

        # 按分数排序
        opportunities.sort(key=lambda x: x['score'], reverse=True)

        return opportunities

    def enter_position(self, opportunity: Dict[str, Any],
                      current_date: datetime, capital: float) -> SwingPosition:
        """开仓"""
        symbol = opportunity['symbol']
        entry_price = opportunity['price']
        side = PositionSide.LONG if opportunity['type'] == 'long' else PositionSide.SHORT

        # 计算持仓数量
        shares = self.calculate_position_size(entry_price, capital)

        # 设置止损止盈
        stop_loss = self.base_stop_loss
        take_profit = self.base_take_profit

        # 期货合约可以用更宽的止损止盈
        if opportunity['is_future']:
            stop_loss = 0.05  # 5%止损
            take_profit = 0.10  # 10%止盈

        position = SwingPosition(
            symbol=symbol,
            name=symbol,  # 简化，实际应该查名称
            side=side,
            entry_date=current_date,
            entry_price=entry_price,
            shares=shares,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_hold_days=self.max_hold_days,
            is_long_term_favor=opportunity['is_favorite'],
        )

        self.positions.append(position)

        return position

    def exit_position(self, position: SwingPosition,
                     exit_date: datetime, exit_price: float,
                     reason: str):
        """平仓"""
        position.exit_date = exit_date
        position.exit_price = exit_price
        position.exit_reason = reason

        # 从持仓列表移到已平仓列表
        if position in self.positions:
            self.positions.remove(position)
        self.closed_positions.append(position)

    def get_weekly_performance(self, week_start: datetime,
                             week_end: datetime) -> Dict[str, Any]:
        """
        计算每周表现

        返回:
            {
                'week_start': datetime,
                'week_end': datetime,
                'initial_capital': float,
                'final_capital': float,
                'realized_profit': float,
                'target_profit': float,
                'target_achieved': bool,
                'return_pct': float,
                'num_trades': int,
                'win_rate': float,
            }
        """
        # 筛选本周的交易
        week_trades = [
            p for p in self.closed_positions
            if p.entry_date >= week_start and p.entry_date < week_end
        ]

        # 计算收益
        total_profit = sum(p.pnl for p in week_trades)

        # 计算胜率
        winning_trades = [p for p in week_trades if p.pnl > 0]
        win_rate = len(winning_trades) / len(week_trades) if week_trades else 0

        return {
            'week_start': week_start,
            'week_end': week_end,
            'initial_capital': self.initial_capital,
            'final_capital': self.initial_capital + total_profit,
            'realized_profit': total_profit,
            'target_profit': self.weekly_target,
            'target_achieved': total_profit >= self.weekly_target,
            'return_pct': total_profit / self.initial_capital,
            'num_trades': len(week_trades),
            'win_rate': win_rate,
            'trades': week_trades,
        }

    def generate_report(self) -> str:
        """生成策略报告"""
        report = []
        report.append("\n" + "="*80)
        report.append(" " * 25 + "周波段T0-T5策略报告")
        report.append("="*80)

        # 策略参数
        report.append(f"\n📊 策略参数:")
        report.append(f"  初始资金: ${self.initial_capital:,.0f}")
        report.append(f"  每周目标: ${self.weekly_target:,.0f} ({self.target_return_pct*100:.0f}%)")
        report.append(f"  最大持仓: {self.max_positions}只")
        report.append(f"  最大天数: {self.max_hold_days}天")
        report.append(f"  止损幅度: {self.base_stop_loss*100:.0f}%")
        report.append(f"  止盈幅度: {self.base_take_profit*100:.0f}%")

        # 当前持仓
        report.append(f"\n📈 当前持仓:")
        active_positions = [p for p in self.positions if p.exit_date is None]

        if not active_positions:
            report.append(f"  无持仓")
        else:
            report.append(f"\n{'代码':<12} {'名称':<20} {'方向':<6} {'入场价':<8} {'当前价':<8} "
                         f"{'盈亏%':<8} {'持仓天数':<8} {'距周五':<8}")
            report.append("-" * 80)

            for pos in active_positions:
                side_str = "做多" if pos.side == PositionSide.LONG else "做空"
                days_to_friday = self.get_days_until_friday(pos.entry_date)

                # 简化：使用入场价作为当前价
                current_price = pos.entry_price
                pnl_pct = 0.0

                report.append(f"{pos.symbol:<12} {pos.name:<20} {side_str:<6} "
                            f"${pos.entry_price:<7.2f} ${current_price:<7.2f} "
                            f"{pnl_pct:>7.2f}% {pos.hold_days:<8} {days_to_friday:<8}")

        # 已平仓持仓
        report.append(f"\n✅ 已平仓持仓:")
        closed_positions = self.closed_positions[-10:]  # 显示最近10笔

        if not closed_positions:
            report.append(f"  无已平仓")
        else:
            report.append(f"\n{'代码':<12} {'名称':<20} {'入场价':<8} {'出场价':<8} "
                         f"{'盈亏':<10} {'盈亏%':<8} {'天数':<6} {'原因'}")
            report.append("-" * 100)

            for pos in closed_positions:
                pnl_str = f"+${pos.pnl:,.2f}" if pos.pnl >= 0 else f"-${abs(pos.pnl):,.2f}"
                reason_str = pos.exit_reason[:15] if pos.exit_reason else ""

                report.append(f"{pos.symbol:<12} {pos.name:<20} ${pos.entry_price:<7.2f} "
                            f"${pos.exit_price:<7.2f} {pnl_str:<10} {pos.pnl_pct:>7.2f}% "
                            f"{pos.hold_days:<6} {reason_str}")

        # 统计数据
        total_trades = len(self.closed_positions)
        winning_trades = [p for p in self.closed_positions if p.pnl > 0]
        total_profit = sum(p.pnl for p in self.closed_positions)

        report.append(f"\n📊 统计数据:")
        report.append(f"  总交易次数: {total_trades}")
        report.append(f"  盈利次数: {len(winning_trades)}")
        report.append(f"  胜率: {len(winning_trades)/total_trades*100:.1f}%" if total_trades > 0 else "  胜率: N/A")
        report.append(f"  总盈亏: ${total_profit:+,.2f}")
        report.append(f"  平均盈亏: ${total_profit/total_trades:+,.2f}" if total_trades > 0 else "  平均盈亏: N/A")

        return "\n".join(report)


def simulate_weekly_swing_strategy(
    initial_capital: float = 1_000_000,
    weeks: int = 4,
    enable_weekly_reentry: bool = True,
) -> Dict[str, Any]:
    """
    模拟周波段策略

    参数:
        initial_capital: 初始资金
        weeks: 模拟周数
        enable_weekly_reentry: 是否启用每周重新进入

    返回:
        模拟结果
    """
    strategy = WeeklySwingStrategy(
        initial_capital=initial_capital,
        weekly_target=20_000,
        max_positions=5,
    )

    # 模拟每周交易
    current_capital = initial_capital
    weekly_results = []

    for week in range(weeks):
        week_start = datetime(2026, 5, 4) + timedelta(weeks=week)
        week_end = week_start + timedelta(days=7)

        print(f"\n{'='*80}")
        print(f"第{week+1}周: {week_start.strftime('%Y-%m-%d')} - {week_end.strftime('%Y-%m-%d')}")
        print(f"{'='*80}")

        # 模拟每日交易
        for day in range(5):  # 周一到周五
            current_date = week_start + timedelta(days=day)

            if not strategy.is_trading_day(current_date):
                continue

            print(f"\n{current_date.strftime('%Y-%m-%d %A')}:")

            # 生成模拟市场数据
            np.random.seed(week * 5 + day)
            market_data = generate_mock_market_data()

            # 扫描机会
            opportunities = strategy.scan_stocks(market_data, current_date)

            # 检查现有持仓是否需要平仓
            for position in strategy.positions[:]:
                if position.exit_date is None:
                    # 模拟当前价格
                    price_change = np.random.normal(0, 0.02)  # 2%波动
                    current_price = position.entry_price * (1 + price_change)

                    should_exit, reason = strategy.should_exit_position(
                        position, current_date, current_price
                    )

                    if should_exit:
                        strategy.exit_position(position, current_date, current_price, reason)
                        print(f"  平仓 {position.symbol}: {reason}")
                        print(f"    盈亏: ${position.pnl:+,.2f} ({position.pnl_pct:+.2f}%)")
                        current_capital += position.pnl

            # 开仓（如果有机会）
            if strategy.can_enter_position(current_date) and opportunities:
                # 选择最佳机会
                best_opportunity = opportunities[0]

                # 计算可用资金
                used_capital = sum(p.current_value for p in strategy.positions if p.exit_date is None)
                available_capital = current_capital - used_capital

                if available_capital > 50_000:  # 至少保留5万
                    position = strategy.enter_position(best_opportunity, current_date, available_capital)
                    print(f"  开仓 {position.symbol}: {position.shares}股 @ ${position.entry_price:.2f}")
                    print(f"    止损: -{position.stop_loss*100:.0f}% | 止盈: +{position.take_profit*100:.0f}%")

        # 周五收盘前强制平仓所有持仓
        friday_close = week_start + timedelta(days=4, hours=15)
        for position in strategy.positions[:]:
            if position.exit_date is None:
                # 模拟平仓价格
                price_change = np.random.normal(0, 0.01)
                exit_price = position.entry_price * (1 + price_change)
                strategy.exit_position(position, friday_close, exit_price, "周五收盘强制平仓")
                print(f"  周五平仓 {position.symbol}: ${position.pnl:+,.2f} ({position.pnl_pct:+.2f}%)")
                current_capital += position.pnl

        # 计算本周表现
        week_performance = strategy.get_weekly_performance(week_start, week_end)
        weekly_results.append(week_performance)

        print(f"\n📊 本周总结:")
        print(f"  交易次数: {week_performance['num_trades']}")
        print(f"  实现利润: ${week_performance['realized_profit']:,.2f}")
        print(f"  目标利润: ${week_performance['target_profit']:,.2f}")
        print(f"  目标达成: {'✅' if week_performance['target_achieved'] else '❌'}")
        print(f"  收益率: {week_performance['return_pct']*100:+.2f}%")
        print(f"  胜率: {week_performance['win_rate']*100:.1f}%")

        # 更新资金
        current_capital = initial_capital + sum(p.pnl for p in strategy.closed_positions)

    # 生成最终报告
    print(strategy.generate_report())

    return {
        'strategy': strategy,
        'weekly_results': weekly_results,
        'total_profit': sum(p.pnl for p in strategy.closed_positions),
        'total_return': sum(p.pnl for p in strategy.closed_positions) / initial_capital,
    }


def generate_mock_market_data() -> Dict[str, pd.DataFrame]:
    """生成模拟市场数据"""
    np.random.seed(42)

    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', '0700.HK', '9988.HK']

    market_data = {}

    for symbol in symbols:
        # 生成50天的价格数据
        dates = pd.date_range(start='2026-04-01', periods=50, freq='D')

        # 随机游走
        returns = np.random.normal(0.001, 0.02, 50)
        prices = [100.0]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        df = pd.DataFrame({
            'Close': prices,
        }, index=dates)

        market_data[symbol] = df

    return market_data


if __name__ == "__main__":
    # 运行4周模拟
    result = simulate_weekly_swing_strategy(
        initial_capital=1_000_000,
        weeks=4,
    )

    print(f"\n{'='*80}")
    print(f" " * 25 + "模拟总结")
    print(f"{'='*80}")
    print(f"\n总收益: ${result['total_profit']:,.2f}")
    print(f"总收益率: {result['total_return']*100:+.2f}%")
