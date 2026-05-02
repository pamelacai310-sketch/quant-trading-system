"""
欧奈尔策略引擎

整合CANSLIM选股、形态识别、口袋支点检测、风险管理的完整策略系统。

核心流程：
1. 市场趋势确认（M）
2. CANSLIM选股筛选
3. 形态识别（杯柄、VCP等）
4. 口袋支点/突破信号检测
5. 严格风险控制（止损/加仓）
6. 执行与监控
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..patterns import ONeillPatternDetector, PatternInfo, PatternType
from ..factors import CANSLIM_Screener, CANSLIM_Score, detect_follow_through_day
from ..signals import PocketPivotDetector, PocketPivotSignal
from ..execution import DynamicStopManager, StopAction, StopType
from ..monitoring import LiveMonitor, HealthMetric


@dataclass
class ONeillTradeSetup:
    """欧奈尔交易设置"""
    symbol: str
    entry_price: float
    pivot_price: float              # 枢轴点价格
    stop_loss_price: float           # 止损价格
    target_price: float              # 目标价格（通常+20-25%）

    # 信号类型
    signal_type: str                 # 'breakout', 'pocket_pivot', 'cheat_entry'
    pattern_type: Optional[PatternType] = None

    # CANSLIM评分
    canslim_score: Optional[float] = None
    pattern_quality: Optional[str] = None

    # 风险参数
    position_size: float = 0.0       # 仓位大小（股数）
    risk_amount: float = 0.0         # 风险金额
    risk_pct: float = 0.0            # 风险百分比（通常7-8%）

    # 附加信息
    entry_reason: str = ""
    exit_rules: List[str] = field(default_factory=list)


@dataclass
class ONeillPosition:
    """持仓信息"""
    symbol: str
    entry_date: datetime
    entry_price: float
    current_price: float
    position_size: float
    unrealized_pnl_pct: float
    unrealized_pnl_amount: float

    # 止损止盈
    stop_loss_price: float
    initial_stop_price: float        # 初始止损价
    trailing_stop_price: Optional[float] = None  # 跟踪止损价

    # 加仓记录
    pyramid_adds: List[Dict] = field(default_factory=list)

    # 状态
    is_active: bool = True
    exit_reason: Optional[str] = None


class ONeillStrategyEngine:
    """
    欧奈尔策略引擎

    完整实现欧奈尔CANSLIM体系的交易系统：
    1. 市场趋势确认（后续交易日FTD）
    2. CANSLIM选股
    3. 形态识别
    4. 信号检测
    5. 风险管理
    6. 执行与监控
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        max_positions: int = 5,
        risk_per_trade: float = 0.01,      # 每笔交易风险（1%）
        max_risk_per_trade: float = 0.08,   # 最大单笔风险（8%）
        max_portfolio_risk: float = 0.20,  # 最大组合风险（20%）
        min_canslim_score: float = 60.0,   # 最小CANSLIM分数
        min_pattern_quality: int = 2,      # 最小形态质量（2=Good）
        follow_ftd: bool = True,            # 是否遵循后续交易日
    ):
        """
        Args:
            initial_capital: 初始资金
            max_positions: 最大持仓数量
            risk_per_trade: 每笔交易风险占总资金比例
            max_risk_per_trade: 单笔交易最大风险（8%）
            max_portfolio_risk: 最大组合风险敞口
            min_canslim_score: 最小CANSLIM总分
            min_pattern_quality: 最小形态质量
            follow_ftd: 是否遵循后续交易日规则
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.min_canslim_score = min_canslim_score
        self.min_pattern_quality = min_pattern_quality
        self.follow_ftd = follow_ftd

        # 子系统
        self.canslim_screener = CANSLIM_Screener()
        self.pattern_detector = ONeillPatternDetector()
        self.pocket_pivot_detector = PocketPivotDetector()
        self.stop_manager = DynamicStopManager()
        self.live_monitor = LiveMonitor(strategy_id="oneill_strategy")

        # 状态
        self.market_trend = "Unknown"
        self.ftd_confirmed = False
        self.ftd_date: Optional[datetime] = None
        self.cash = initial_capital
        self.positions: Dict[str, ONeillPosition] = {}
        self.trade_history: List[Dict] = []

    def analyze_market(
        self,
        market_index_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        分析市场环境

        Args:
            market_index_data: 大盘指数数据

        Returns:
            市场分析结果
        """
        result = {
            'trend': 'Unknown',
            'is_uptrend': False,
            'ftd_confirmed': False,
            'ftd_date': None,
            'recommendation': '等待',
        }

        # 1. 检查趋势
        if len(market_index_data) >= 200:
            ma50 = market_index_data['close'].rolling(window=50).mean().iloc[-1]
            ma200 = market_index_data['close'].rolling(window=200).mean().iloc[-1]
            current = market_index_data['close'].iloc[-1]

            if current > ma50 > ma200:
                result['trend'] = 'Uptrend'
                result['is_uptrend'] = True
            elif current > ma200:
                result['trend'] = 'Sideways_Up'
            else:
                result['trend'] = 'Downtrend'

        # 2. 检查后续交易日（FTD）
        if self.follow_ftd:
            is_ftd, ftd_date, ftd_desc = detect_follow_through_day(market_index_data)
            result['ftd_confirmed'] = is_ftd
            result['ftd_date'] = ftd_date

            if is_ftd:
                self.ftd_confirmed = True
                self.ftd_date = ftd_date
                result['recommendation'] = '积极寻找机会'

        # 3. 综合建议
        if result['is_uptrend'] and result['ftd_confirmed']:
            result['recommendation'] = '积极做多'
        elif result['is_uptrend']:
            result['recommendation'] = '谨慎做多'
        else:
            result['recommendation'] = '等待或轻仓'

        self.market_trend = result['trend']

        return result

    def scan_stocks(
        self,
        stocks_data: Dict[str, pd.DataFrame],
        fundamentals_dict: Dict[str, Dict[str, Any]],
        market_index_data: pd.DataFrame,
    ) -> List[ONeillTradeSetup]:
        """
        扫描股票，寻找交易机会

        Args:
            stocks_data: {股票代码: 价格数据}
            fundamentals_dict: {股票代码: 基本面数据}
            market_index_data: 大盘指数数据

        Returns:
            交易设置列表
        """
        trade_setups = []

        # 1. CANSLIM筛选
        canslim_results = self.canslim_screener.screen_multiple_stocks(
            stocks_data,
            fundamentals_dict,
            market_index_data,
            min_score=self.min_canslim_score,
        )

        print(f"CANSLIM筛选通过: {len(canslim_results)} 只")

        # 2. 对每只股票进行形态识别和信号检测
        for symbol, canslim_score in canslim_results:
            if symbol not in stocks_data:
                continue

            stock_data = stocks_data[symbol]

            # 形态识别
            patterns = self.pattern_detector.detect_all_patterns(stock_data)

            # 过滤质量
            quality_map = {'excellent': 4, 'good': 3, 'acceptable': 2, 'poor': 1}
            quality_value = quality_map.get(canslim_score.pattern_quality, 0)

            if quality_value < self.min_pattern_quality:
                continue

            # 口袋支点检测
            pocket_pivots = self.pocket_pivot_detector.detect_signals(stock_data)

            # 生成交易设置
            if patterns:
                # 使用最佳形态
                best_pattern = patterns[0]

                setup = ONeillTradeSetup(
                    symbol=symbol,
                    entry_price=stock_data['close'].iloc[-1],
                    pivot_price=best_pattern.pivot_price,
                    stop_loss_price=best_pattern.stop_loss_price,
                    target_price=best_pattern.pivot_price * 1.25,  # +25%
                    signal_type='breakout',
                    pattern_type=best_pattern.pattern_type,
                    canslim_score=canslim_score.total_score,
                    pattern_quality=best_pattern.quality.value,
                    entry_reason=f"突破{best_pattern.pattern_type.value}形态",
                    exit_rules=[
                        "止损8%",
                        "获利20-25%部分了结",
                        "跌破10日均线平仓",
                    ],
                )

                trade_setups.append(setup)

            elif pocket_pivots:
                # 使用口袋支点
                best_pp = pocket_pivots[0]

                setup = ONeillTradeSetup(
                    symbol=symbol,
                    entry_price=best_pp.price,
                    pivot_price=best_pp.price,
                    stop_loss_price=best_pp.stop_loss_price,
                    target_price=best_pp.price * 1.25,
                    signal_type='pocket_pivot',
                    canslim_score=canslim_score.total_score,
                    entry_reason=best_pp.description,
                    exit_rules=[
                        "止损设在10日均线下方",
                        "获利20-25%部分了结",
                    ],
                )

                trade_setups.append(setup)

        # 按CANSLIM分数和形态质量排序
        trade_setups.sort(
            key=lambda x: (
                x.canslim_score if x.canslim_score else 0
            ),
            reverse=True
        )

        return trade_setups

    def calculate_position_size(
        self,
        setup: ONeillTradeSetup,
    ) -> float:
        """
        计算仓位大小

        使用欧奈尔和Minervini的风险管理方法：
        1. 每笔交易风险控制在账户的1-2%
        2. 根据止损距离反推仓位

        公式：仓位 = (账户资金 × 风险比例) / (入场价 - 止损价)

        Args:
            setup: 交易设置

        Returns:
            仓位大小（股数）
        """
        # 计算止损距离
        stop_distance = setup.entry_price - setup.stop_loss_price
        stop_distance_pct = stop_distance / setup.entry_price

        # 检查止损距离是否合理（不超过8%）
        if stop_distance_pct > self.max_risk_per_trade:
            # 止损过宽，调整仓位或跳过
            print(f"警告：{setup.symbol}止损距离过大（{stop_distance_pct:.2f}%），跳过")
            return 0.0

        # 计算风险金额
        risk_amount = self.cash * self.risk_per_trade

        # 计算仓位（股数）
        position_size = risk_amount / stop_distance

        # 检查是否超过现金限制
        required_capital = position_size * setup.entry_price
        if required_capital > self.cash:
            # 调整仓位
            position_size = self.cash / setup.entry_price

        return position_size

    def execute_trade(
        self,
        setup: ONeillTradeSetup,
    ) -> Optional[ONeillPosition]:
        """
        执行交易

        Args:
            setup: 交易设置

        Returns:
            持仓对象，如果交易未执行则返回None
        """
        # 1. 检查市场环境
        if self.market_trend == 'Downtrend':
            print(f"市场处于下降趋势，跳过{setup.symbol}")
            return None

        # 2. 检查持仓数量
        if len(self.positions) >= self.max_positions:
            print(f"已达到最大持仓数量{self.max_positions}，跳过{setup.symbol}")
            return None

        # 3. 计算仓位大小
        position_size = self.calculate_position_size(setup)

        if position_size <= 0:
            return None

        # 4. 创建持仓
        position = ONeillPosition(
            symbol=setup.symbol,
            entry_date=datetime.now(),
            entry_price=setup.entry_price,
            current_price=setup.entry_price,
            position_size=position_size,
            unrealized_pnl_pct=0.0,
            unrealized_pnl_amount=0.0,
            stop_loss_price=setup.stop_loss_price,
            initial_stop_price=setup.stop_loss_price,
        )

        # 5. 更新现金
        cost = position_size * setup.entry_price
        self.cash -= cost

        # 6. 记录持仓
        self.positions[setup.symbol] = position

        # 7. 设置止损管理器
        self.stop_manager.reset(setup.entry_price, datetime.now())

        print(f"✅ 买入 {setup.symbol}: {position_size:.0f}股 @ {setup.entry_price:.2f}")

        return position

    def update_positions(
        self,
        current_prices: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        更新持仓状态，检查止损止盈

        Args:
            current_prices: {股票代码: 当前价格}

        Returns:
            需要执行的操作列表
        """
        actions = []

        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            position.current_price = current_price

            # 更新盈亏
            position.unrealized_pnl_pct = (
                current_price / position.entry_price - 1
            ) * 100

            position.unrealized_pnl_amount = (
                (current_price - position.entry_price) * position.position_size
            )

            # 1. 检查止损
            if current_price <= position.stop_loss_price:
                actions.append({
                    'symbol': symbol,
                    'action': 'sell',
                    'reason': f'触发止损（{position.unrealized_pnl_pct:.2f}%）',
                    'price': current_price,
                })
                continue

            # 2. 检查跟踪止盈（如果盈利>8%）
            if position.unrealized_pnl_pct > 8:
                # 激活跟踪止盈：从最高点回撤3%止盈
                # 这里简化处理，实际应该记录最高价
                trailing_stop = position.entry_price * 1.08 * 0.97
                if current_price <= trailing_stop:
                    actions.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'reason': f'触发跟踪止盈（{position.unrealized_pnl_pct:.2f}%）',
                        'price': current_price,
                    })
                    continue

            # 3. 检查目标价格（+25%）
            if current_price >= position.entry_price * 1.25:
                # 部分了结
                actions.append({
                    'symbol': symbol,
                    'action': 'partial_sell',
                    'reason': f'达到目标价（{position.unrealized_pnl_pct:.2f}%），部分了结',
                    'price': current_price,
                })
                continue

            # 4. 金字塔加仓（Minervini方法）
            # 如果盈利>10%且市场环境好，可以考虑加仓
            if position.unrealized_pnl_pct > 10 and not position.pyramid_adds:
                # 这里简化，实际需要更复杂的加仓逻辑
                pass

        return actions

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str,
    ) -> Optional[Dict]:
        """
        平仓

        Args:
            symbol: 股票代码
            exit_price: 退出价格
            reason: 退出原因

        Returns:
            交易记录
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        # 计算盈亏
        pnl_amount = (exit_price - position.entry_price) * position.position_size
        pnl_pct = (exit_price / position.entry_price - 1) * 100

        # 更新现金
        self.cash += exit_price * position.position_size

        # 记录交易历史
        trade_record = {
            'symbol': symbol,
            'entry_date': position.entry_date,
            'exit_date': datetime.now(),
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'position_size': position.position_size,
            'pnl_amount': pnl_amount,
            'pnl_pct': pnl_pct,
            'exit_reason': reason,
        }

        self.trade_history.append(trade_record)

        # 移除持仓
        del self.positions[symbol]

        print(
            f"{'💰' if pnl_pct > 0 else '📉'} "
            f"卖出 {symbol}: {position.position_size:.0f}股 @ {exit_price:.2f} "
            f"({pnl_pct:+.2f}%, {pnl_amount:+.2f})"
        )

        return trade_record

    def generate_portfolio_report(self) -> Dict[str, Any]:
        """生成投资组合报告"""

        total_value = self.cash + sum(
            p.current_price * p.position_size
            for p in self.positions.values()
        )

        total_pnl = total_value - self.initial_capital
        total_pnl_pct = (total_value / self.initial_capital - 1) * 100

        # 计算持仓统计
        active_positions = list(self.positions.values())
        winners = sum(1 for p in active_positions if p.unrealized_pnl_pct > 0)
        losers = len(active_positions) - winners

        return {
            'cash': self.cash,
            'positions': len(active_positions),
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'winners': winners,
            'losers': losers,
            'win_rate': winners / len(active_positions) if active_positions else 0,
            'market_trend': self.market_trend,
            'ftd_confirmed': self.ftd_confirmed,
        }


def run_oneill_strategy(
    stocks_data: Dict[str, pd.DataFrame],
    fundamentals_dict: Dict[str, Dict[str, Any]],
    market_index_data: pd.DataFrame,
    initial_capital: float = 100_000.0,
    max_positions: int = 5,
) -> ONeillStrategyEngine:
    """
    运行完整的欧奈尔策略

    Args:
        stocks_data: 股票数据字典
        fundamentals_dict: 基本面数据字典
        market_index_data: 大盘指数数据
        initial_capital: 初始资金
        max_positions: 最大持仓数量

    Returns:
        策略引擎对象
    """
    # 创建引擎
    engine = ONeillStrategyEngine(
        initial_capital=initial_capital,
        max_positions=max_positions,
    )

    # 1. 分析市场环境
    print("=" * 80)
    print("市场环境分析")
    print("=" * 80)
    market_analysis = engine.analyze_market(market_index_data)
    print(f"趋势: {market_analysis['trend']}")
    print(f"FTD确认: {market_analysis['ftd_confirmed']}")
    print(f"建议: {market_analysis['recommendation']}")
    print()

    # 2. 扫描股票
    print("=" * 80)
    print("扫描股票机会")
    print("=" * 80)
    trade_setups = engine.scan_stocks(
        stocks_data,
        fundamentals_dict,
        market_index_data,
    )

    print(f"发现 {len(trade_setups)} 个交易机会:\n")

    for i, setup in enumerate(trade_setups[:10], 1):  # 只显示前10个
        print(f"{i}. {setup.symbol}")
        print(f"   信号: {setup.signal_type}")
        print(f"   入场价: {setup.entry_price:.2f}")
        print(f"   枢轴点: {setup.pivot_price:.2f}")
        print(f"   止损: {setup.stop_loss_price:.2f}")
        print(f"   目标: {setup.target_price:.2f}")
        print(f"   CANSLIM得分: {setup.canslim_score:.1f}")
        print(f"   理由: {setup.entry_reason}")
        print()

    # 3. 执行交易（模拟）
    print("=" * 80)
    print("执行交易")
    print("=" * 80)

    for setup in trade_setups[:engine.max_positions]:
        engine.execute_trade(setup)

    print(f"\n持仓数量: {len(engine.positions)}")
    print(f"剩余现金: {engine.cash:.2f}")

    # 4. 生成报告
    report = engine.generate_portfolio_report()
    print("\n" + "=" * 80)
    print("投资组合报告")
    print("=" * 80)
    print(f"总资产: {report['total_value']:.2f}")
    print(f"总盈亏: {report['total_pnl']:+.2f} ({report['total_pnl_pct']:+.2f}%)")
    print(f"持仓数: {report['positions']}")
    print(f"盈利: {report['winners']}, 亏损: {report['losers']}")
    print(f"胜率: {report['win_rate']*100:.1f}%")

    return engine
