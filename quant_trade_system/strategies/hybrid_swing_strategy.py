"""
融合策略：期货+股票周波段交易系统

融合"远期期货合约交易策略"和"周波段T0-T5短线策略"的优点：

1. **期货标的**：
   - 只做远期合约（+2个月以上）
   - 选择波动高于主力的合约
   - 资金管理：风险度<50%，预留1倍资金

2. **股票标的**：
   - 长期看好标的，反复做波段
   - 每周回调时接回

3. **统一规则**：
   - 持仓周期T0-T5，不过周末
   - 每日根据市场情绪选择方向
   - 目标每周净赚2万（本金100万，2%）

4. **情绪驱动**：
   - 上涨品种>60% → 做多
   - 前20涨幅>前20跌幅 → 做多
   - 信心度>0.65才开仓
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from .weekly_swing_strategy import SwingPosition, WeeklyTarget, DayOfWeek
from .far_month_futures_strategy import FuturesContract, MarketSentiment, PositionSide


class AssetType(Enum):
    """资产类型"""
    STOCK = "stock"
    FUTURES = "futures"


@dataclass
class UnifiedPosition:
    """统一持仓（股票或期货）"""
    asset_type: AssetType

    # 股票特有属性
    stock_symbol: Optional[str] = None
    stock_name: Optional[str] = None
    is_long_term_favor: bool = False
    weekly_reentry_count: int = 0

    # 期货特有属性
    futures_contract: Optional[FuturesContract] = None
    contracts: Optional[int] = None
    margin_used: Optional[float] = None
    reserve_capital: Optional[float] = None

    # 通用属性
    side: PositionSide = PositionSide.LONG
    entry_date: Optional[datetime] = None
    entry_price: Optional[float] = None
    shares: Optional[int] = None  # 股票股数
    stop_loss: float = 0.03
    take_profit: float = 0.06
    max_hold_days: int = 5

    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    @property
    def symbol(self) -> str:
        """标的代码"""
        if self.asset_type == AssetType.STOCK:
            return self.stock_symbol
        else:
            return self.futures_contract.symbol if self.futures_contract else ""

    @property
    def name(self) -> str:
        """标的名称"""
        if self.asset_type == AssetType.STOCK:
            return self.stock_name
        else:
            return self.futures_contract.name if self.futures_contract else ""

    @property
    def hold_days(self) -> int:
        """持仓天数"""
        if self.exit_date and self.entry_date:
            return (self.exit_date - self.entry_date).days
        return 0

    @property
    def days_until_exit(self) -> int:
        """距离必须平仓的天数"""
        if self.exit_date:
            return 0

        if not self.entry_date:
            return 0

        # 计算到周五的天数
        current_day = self.entry_date.weekday()
        days_to_friday = 4 - current_day
        return min(self.max_hold_days - self.hold_days, days_to_friday)

    @property
    def current_value(self) -> float:
        """当前市值"""
        if self.asset_type == AssetType.STOCK:
            if self.shares and self.entry_price:
                return self.entry_price * self.shares
            return 0.0
        else:
            if self.margin_used:
                return self.margin_used
            return 0.0

    @property
    def pnl(self) -> float:
        """盈亏"""
        if self.exit_price is None or not self.entry_price:
            return 0.0

        if self.asset_type == AssetType.STOCK:
            price_diff = self.exit_price - self.entry_price
            return price_diff * self.shares
        else:
            # 期货
            price_diff = self.exit_price - self.entry_price
            if self.side == PositionSide.SHORT:
                price_diff = -price_diff
            return price_diff * self.contracts if self.contracts else 0.0

    @property
    def pnl_pct(self) -> float:
        """盈亏百分比"""
        if self.asset_type == AssetType.STOCK:
            if self.entry_price and self.shares:
                investment = self.entry_price * self.shares
                return self.pnl / investment if investment > 0 else 0.0
            return 0.0
        else:
            if self.margin_used and self.margin_used > 0:
                return self.pnl / self.margin_used * 100
            return 0.0

    @property
    def risk_level(self) -> float:
        """风险度"""
        if self.asset_type == AssetType.FUTURES:
            if self.margin_used and self.reserve_capital:
                return self.margin_used / (self.margin_used + self.reserve_capital)
        return 0.0


class HybridSwingStrategy:
    """
    融合策略：期货+股票周波段交易系统

    融合两个策略的优点：
    1. 期货：远期合约、高波动、严格风控
    2. 股票：长期看好标的、反复波段
    3. 统一：T0-T5周期、不过周末、情绪驱动
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        weekly_target: float = 20_000,
        max_positions: int = 5,
        max_hold_days: int = 5,
        base_stop_loss: float = 0.03,
        base_take_profit: float = 0.06,
        futures_risk_level: float = 0.50,  # 期货风险度50%
        min_far_months: int = 2,  # 期货最少远期月数
    ):
        """
        初始化融合策略

        参数:
            initial_capital: 初始资金
            weekly_target: 每周目标利润
            max_positions: 最大持仓数量
            max_hold_days: 最大持仓天数
            base_stop_loss: 基础止损
            base_take_profit: 基础止盈
            futures_risk_level: 期货风险度
            min_far_months: 期货最少远期月数
        """
        self.initial_capital = initial_capital
        self.weekly_target = weekly_target
        self.target_return_pct = weekly_target / initial_capital
        self.max_positions = max_positions
        self.max_hold_days = max_hold_days
        self.base_stop_loss = base_stop_loss
        self.base_take_profit = base_take_profit
        self.futures_risk_level = futures_risk_level
        self.min_far_months = min_far_months

        # 长期看好的股票池
        self.stock_universe = [
            'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN',
            '0700.HK', '9988.HK', '3690.HK',  # 港股科技
        ]

        # 期货标的池
        self.futures_universe = [
            'RB', 'CU', 'AL', 'ZN', 'AU', 'AG', 'CL', 'MA', 'PP', 'L',
            'M', 'Y', 'P', 'A', 'C', 'JD', 'CF', 'SR', 'OI', 'RM',
            'IF', 'IH', 'IC', 'IM',
        ]

        # 持仓
        self.positions: List[UnifiedPosition] = []
        self.closed_positions: List[UnifiedPosition] = []

        # 每周目标追踪
        self.weekly_targets: List[WeeklyTarget] = []

    def is_trading_day(self, date: datetime) -> bool:
        """判断是否是交易日"""
        return date.weekday() < 5

    def get_days_until_friday(self, date: datetime) -> int:
        """计算到周五的天数"""
        return max(0, 4 - date.weekday())

    def can_enter_position(self, date: datetime) -> bool:
        """判断是否可以开仓"""
        # 1. 必须是交易日
        if not self.is_trading_day(date):
            return False

        # 2. 周五下午3点后不能开新仓
        if date.weekday() == 4 and date.hour >= 15:
            return False

        # 3. 检查持仓数量
        active_positions = [p for p in self.positions if p.exit_date is None]
        if len(active_positions) >= self.max_positions:
            return False

        return True

    def scan_stock_opportunities(self, market_data: Dict[str, pd.DataFrame],
                                current_date: datetime) -> List[Dict[str, Any]]:
        """扫描股票机会（周波段策略）"""
        opportunities = []

        for symbol in self.stock_universe:
            if symbol not in market_data:
                continue

            data = market_data[symbol]
            if len(data) < 20:
                continue

            # 计算技术指标
            current_price = data['Close'].iloc[-1]
            ma5 = data['Close'].rolling(5).mean().iloc[-1]
            ma10 = data['Close'].rolling(10).mean().iloc[-1]
            ma20 = data['Close'].rolling(20).mean().iloc[-1]

            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            # 波动率
            returns = data['Close'].pct_change().tail(20)
            volatility = returns.std() * np.sqrt(252)

            # 信号评估
            signals = 0

            # 均线多头排列
            if ma5 > ma10 > ma20:
                signals += 1

            # 价格突破MA5
            if current_price > ma5:
                signals += 1

            # RSI超卖回升
            if 30 < current_rsi < 50:
                signals += 0.5
            elif current_rsi < 40:
                signals += 1

            # 波动率适中
            if 0.15 < volatility < 0.6:
                signals += 1

            # 长期看好加分
            is_favorite = symbol in self.stock_universe
            if is_favorite:
                signals += 1

            if signals >= 3:
                opportunities.append({
                    'asset_type': AssetType.STOCK,
                    'symbol': symbol,
                    'price': current_price,
                    'signals': signals,
                    'rsi': current_rsi,
                    'volatility': volatility,
                    'is_favorite': is_favorite,
                    'score': signals,
                })

        # 按分数排序
        opportunities.sort(key=lambda x: x['score'], reverse=True)

        return opportunities

    def scan_futures_contracts(self, current_date: datetime) -> List[FuturesContract]:
        """扫描期货机会（远期合约策略）"""
        suitable_contracts = []

        for underlying in self.futures_universe:
            # 选择远期合约
            contract = self._select_far_month_contract(underlying, current_date)

            if contract is None:
                continue

            # 检查流动性
            if contract.volume < 1000:
                continue

            suitable_contracts.append(contract)

        # 按波动率排序
        suitable_contracts.sort(key=lambda c: c.volatility, reverse=True)

        return suitable_contracts

    def _select_far_month_contract(self, underlying: str, current_date: datetime) -> Optional[FuturesContract]:
        """选择远期合约"""
        # 获取所有合约
        all_contracts = self._get_available_contracts(underlying, current_date)

        # 筛选远期合约（至少+2个月）
        far_contracts = [c for c in all_contracts if c.months_to_delivery >= self.min_far_months]

        if not far_contracts:
            return None

        # 获取主力合约
        main_contract = next((c for c in all_contracts if c.is_main), None)

        if not main_contract:
            return max(far_contracts, key=lambda c: c.volatility)

        # 选择波动高于主力的远期合约
        higher_vol_contracts = [
            c for c in far_contracts
            if c.volatility > main_contract.volatility
        ]

        if not higher_vol_contracts:
            return max(far_contracts, key=lambda c: c.volatility)

        return max(higher_vol_contracts, key=lambda c: c.volatility)

    def _get_available_contracts(self, underlying: str, current_date: datetime) -> List[FuturesContract]:
        """获取某标的的所有可用合约"""
        contracts = []

        current_month = current_date.month
        current_year = current_date.year

        # 生成未来12个月的合约
        for i in range(1, 13):
            delivery_month = current_month + i
            delivery_year = current_year

            if delivery_month > 12:
                delivery_month -= 12
                delivery_year += 1

            # 模拟数据
            np.random.seed(hash(underlying + str(delivery_year) + str(delivery_month)) % 10000)

            is_main = (i == 1)
            base_volatility = 0.20 if is_main else 0.25

            contract = FuturesContract(
                symbol=f"{underlying}{delivery_year % 100:02d}{delivery_month:02d}",
                name=self._get_contract_name(underlying),
                underlying=underlying,
                delivery_month=delivery_month,
                delivery_year=delivery_year,
                is_main=is_main,
                volatility=base_volatility + np.random.uniform(-0.03, 0.05),
                volume=np.random.uniform(10000, 100000) if is_main else np.random.uniform(1000, 50000),
                current_price=np.random.uniform(2000, 8000),
                margin_rate=0.15,
            )

            contracts.append(contract)

        return contracts

    def _get_contract_name(self, underlying: str) -> str:
        """获取合约名称"""
        names = {
            'RB': '螺纹钢', 'CU': '铜', 'AL': '铝',
            'ZN': '锌', 'NI': '镍', 'SN': '锡',
            'AU': '黄金', 'AG': '白银',
            'CL': '原油', 'MA': '甲醇', 'PP': 'PP',
            'L': '塑料', 'M': '豆粕', 'Y': '豆油',
            'IF': '沪深300', 'IH': '上证50', 'IC': '中证500',
        }
        return names.get(underlying, underlying)

    def analyze_market_sentiment(self, market_data: Dict[str, pd.DataFrame],
                               current_date: datetime) -> MarketSentiment:
        """分析市场多空情绪"""
        # 计算每个品种的涨跌
        daily_changes = []

        for symbol, data in market_data.items():
            if len(data) < 2:
                continue

            change_pct = (data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1)
            daily_changes.append({
                'symbol': symbol,
                'change_pct': change_pct,
            })

        # 统计上涨和下跌品种
        up_symbols = [c for c in daily_changes if c['change_pct'] > 0]
        down_symbols = [c for c in daily_changes if c['change_pct'] < 0]

        total_symbols = len(daily_changes)
        up_count = len(up_symbols)
        down_count = len(down_symbols)
        up_ratio = up_count / total_symbols if total_symbols > 0 else 0.5

        # 计算前20大涨幅/跌幅
        top20_gains = sorted([c for c in daily_changes if c['change_pct'] > 0],
                           key=lambda x: x['change_pct'], reverse=True)[:20]
        top20_losses = sorted([c for c in daily_changes if c['change_pct'] < 0],
                            key=lambda x: x['change_pct'])[:20]

        top20_avg_gain = np.mean([c['change_pct'] for c in top20_gains]) if top20_gains else 0
        top20_avg_loss = np.mean([abs(c['change_pct']) for c in top20_losses]) if top20_losses else 0

        # 判断情绪倾向
        bias = 'neutral'
        confidence = 0.5

        if up_ratio > 0.60:
            bias = 'long'
            confidence = min(0.9, 0.5 + (up_ratio - 0.60) * 2)
        elif up_ratio < 0.40:
            bias = 'short'
            confidence = min(0.9, 0.5 + (0.40 - up_ratio) * 2)

        if top20_avg_gain > top20_avg_loss:
            if bias == 'neutral':
                bias = 'long'
                confidence = min(0.8, 0.5 + (top20_avg_gain - top20_avg_loss) * 5)
            elif bias == 'long':
                confidence = min(0.95, confidence + 0.2)
        elif top20_avg_loss > top20_avg_gain:
            if bias == 'neutral':
                bias = 'short'
                confidence = min(0.8, 0.5 + (top20_avg_loss - top20_avg_gain) * 5)
            elif bias == 'short':
                confidence = min(0.95, confidence + 0.2)

        sentiment = MarketSentiment(
            date=current_date,
            total_symbols=total_symbols,
            up_symbols=up_count,
            down_symbols=down_count,
            up_ratio=up_ratio,
            top20_avg_gain=top20_avg_gain,
            top20_avg_loss=top20_avg_loss,
            sentiment_bias=bias,
            confidence=confidence,
        )

        return sentiment

    def select_position_side(self, sentiment: MarketSentiment) -> PositionSide:
        """根据市场情绪选择方向"""
        if sentiment.sentiment_bias == 'long':
            return PositionSide.LONG
        elif sentiment.sentiment_bias == 'short':
            return PositionSide.SHORT
        else:
            # 中性，默认做多
            return PositionSide.LONG

    def enter_stock_position(self, opportunity: Dict[str, Any],
                            current_date: datetime, capital: float) -> UnifiedPosition:
        """开仓股票"""
        symbol = opportunity['symbol']
        entry_price = opportunity['price']
        side = PositionSide.LONG

        # 计算持仓数量
        shares = self._calculate_stock_shares(entry_price, capital)

        position = UnifiedPosition(
            asset_type=AssetType.STOCK,
            stock_symbol=symbol,
            stock_name=symbol,
            side=side,
            entry_date=current_date,
            entry_price=entry_price,
            shares=shares,
            stop_loss=self.base_stop_loss,
            take_profit=self.base_take_profit,
            max_hold_days=self.max_hold_days,
            is_long_term_favor=opportunity.get('is_favorite', False),
        )

        self.positions.append(position)
        return position

    def enter_futures_position(self, contract: FuturesContract, side: PositionSide,
                              current_date: datetime, capital: float) -> UnifiedPosition:
        """开仓期货"""
        # 计算合约数量
        contracts = self._calculate_futures_contracts(contract, capital)

        # 计算保证金和预留资金
        margin_used, reserve_capital = self._calculate_futures_margin(contract, contracts)

        position = UnifiedPosition(
            asset_type=AssetType.FUTURES,
            futures_contract=contract,
            side=side,
            entry_date=current_date,
            entry_price=contract.current_price,
            contracts=contracts,
            margin_used=margin_used,
            reserve_capital=reserve_capital,
            stop_loss=self.base_stop_loss,
            take_profit=self.base_take_profit,
            max_hold_days=self.max_hold_days,
        )

        self.positions.append(position)
        return position

    def _calculate_stock_shares(self, price: float, capital: float) -> int:
        """计算股票持仓数量"""
        # 单笔风险2%
        risk_amount = capital * 0.02
        stop_loss_amount = price * self.base_stop_loss
        shares = int(risk_amount / stop_loss_amount)

        # 确保不超过可用资金
        max_shares = int(capital / price)
        shares = min(shares, max_shares)

        return max(0, shares)

    def _calculate_futures_contracts(self, contract: FuturesContract, capital: float) -> int:
        """计算期货合约数量"""
        # 可用保证金（50%风险度）
        available_margin = capital * 0.50

        # 单个合约保证金
        single_contract_margin = contract.current_price * contract.margin_rate

        # 计算合约数量
        contracts = int(available_margin / single_contract_margin)

        return max(1, contracts)

    def _calculate_futures_margin(self, contract: FuturesContract,
                                contracts_count: int) -> Tuple[float, float]:
        """计算期货保证金和预留资金"""
        # 计算合约价值
        contract_value = contract.current_price * contracts_count

        # 计算保证金
        margin_required = contract_value * contract.margin_rate

        # 预留资金 = 保证金（1倍）
        reserve_capital = margin_required

        # 检查风险度
        risk_level = margin_required / (margin_required + reserve_capital)

        # 如果风险度超过50%，调整合约数量
        if risk_level > self.futures_risk_level:
            # 调整到正好50%风险度
            target_total = margin_required / self.futures_risk_level
            adjusted_contracts = int(target_total * contract.margin_rate / contract.current_price)
            margin_required = contract.current_price * adjusted_contracts * contract.margin_rate
            reserve_capital = margin_required
            contracts_count = adjusted_contracts

        return margin_required, reserve_capital

    def should_exit_position(self, position: UnifiedPosition,
                            current_price: float) -> Tuple[bool, Optional[str]]:
        """判断是否应该平仓"""
        # 计算盈亏
        if position.side == PositionSide.LONG:
            pnl_pct = (current_price / position.entry_price - 1)
        else:
            pnl_pct = (position.entry_price / current_price - 1)

        # 检查止损
        if pnl_pct <= -position.stop_loss:
            return True, f"触发止损: {pnl_pct*100:.2f}%"

        # 检查止盈
        if pnl_pct >= position.take_profit:
            return True, f"触发止盈: {pnl_pct*100:.2f}%"

        # 检查持仓天数
        hold_days = position.hold_days
        if hold_days >= position.max_hold_days:
            return True, f"达到最大持仓天数: {hold_days}天"

        # 检查是否到周五
        if position.entry_date:
            days_to_friday = self.get_days_until_friday(position.entry_date)
            current_day = datetime.now().weekday()

            # 周五下午3点必须平仓
            if days_to_friday == 0 and datetime.now().hour >= 15:
                return True, "周五收盘，必须平仓"

        return False, None

    def exit_position(self, position: UnifiedPosition, exit_date: datetime,
                     exit_price: float, reason: str):
        """平仓"""
        position.exit_date = exit_date
        position.exit_price = exit_price
        position.exit_reason = reason

        if position in self.positions:
            self.positions.remove(position)
        self.closed_positions.append(position)

    def generate_report(self) -> str:
        """生成策略报告"""
        report = []
        report.append("\n" + "="*80)
        report.append(" " * 20 + "融合策略：期货+股票周波段系统")
        report.append("="*80)

        # 策略参数
        report.append(f"\n📊 策略参数:")
        report.append(f"  初始资金: ${self.initial_capital:,.0f}")
        report.append(f"  每周目标: ${self.weekly_target:,.0f} ({self.target_return_pct*100:.0f}%)")
        report.append(f"  最大持仓: {self.max_positions}只")
        report.append(f"  最大天数: {self.max_hold_days}天")
        report.append(f"  止损: {self.base_stop_loss*100:.0f}%")
        report.append(f"  止盈: {self.base_take_profit*100:.0f}%")
        report.append(f"  期货风险度: {self.futures_risk_level*100:.0f}%")

        # 当前持仓
        report.append(f"\n📈 当前持仓:")
        active_positions = [p for p in self.positions if p.exit_date is None]

        if not active_positions:
            report.append(f"  无持仓")
        else:
            report.append(f"\n{'类型':<8} {'代码':<12} {'名称':<20} {'方向':<6} {'入场价':<10} "
                         f"{'保证金/市值':<12} {'风险度':<8} {'盈亏%':<8}")
            report.append("-" * 80)

            for pos in active_positions:
                asset_str = "期货" if pos.asset_type == AssetType.FUTURES else "股票"
                side_str = "做多" if pos.side == PositionSide.LONG else "做空"

                if pos.asset_type == AssetType.FUTURES:
                    value_str = f"${pos.margin_used:>10,.0f}"
                    risk_str = f"{pos.risk_level*100:.1f}%"
                else:
                    value_str = f"${pos.current_value:>10,.0f}"
                    risk_str = "N/A"

                pnl_str = f"{pos.pnl_pct:>7.2f}%"

                report.append(f"{asset_str:<8} {pos.symbol:<12} {pos.name:<20} {side_str:<6} "
                            f"${pos.entry_price:<9.2f} {value_str:<12} {risk_str:<8} {pnl_str:<8}")

        # 已平仓
        report.append(f"\n✅ 已平仓持仓:")
        if not self.closed_positions:
            report.append(f"  无已平仓")
        else:
            report.append(f"\n{'类型':<8} {'代码':<12} {'名称':<20} {'入场价':<10} "
                         f"{'出场价':<10} {'盈亏':<12} {'盈亏%':<10} {'天数':<6} {'原因'}")
            report.append("-" * 100)

            for pos in self.closed_positions[-10:]:
                asset_str = "期货" if pos.asset_type == AssetType.FUTURES else "股票"
                pnl_str = f"+${pos.pnl:,.2f}" if pos.pnl >= 0 else f"-${abs(pos.pnl):,.2f}"
                reason_str = pos.exit_reason[:12] if pos.exit_reason else ""

                report.append(f"{asset_str:<8} {pos.symbol:<12} {pos.name:<20} "
                            f"${pos.entry_price:<9.2f} ${pos.exit_price:<9.2f} "
                            f"{pnl_str:<12} {pos.pnl_pct:>9.2f}% {pos.hold_days:<6} {reason_str}")

        # 统计
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


def simulate_hybrid_strategy(
    initial_capital: float = 1_000_000,
    weeks: int = 4,
) -> Dict[str, Any]:
    """模拟融合策略"""
    strategy = HybridSwingStrategy(
        initial_capital=initial_capital,
        weekly_target=20_000,
        max_positions=5,
    )

    current_date = datetime(2026, 5, 4)

    print("\n" + "="*80)
    print(" " * 20 + "融合策略模拟：期货+股票周波段系统")
    print("="*80)

    for week in range(weeks):
        week_start = current_date
        week_end = week_start + timedelta(days=7)

        print(f"\n{'='*80}")
        print(f"第{week+1}周: {week_start.strftime('%Y-%m-%d')} - {week_end.strftime('%Y-%m-%d')}")
        print(f"{'='*80}")

        # 模拟每日交易
        for day in range(5):
            trade_date = week_start + timedelta(days=day)

            if trade_date.weekday() >= 5:
                continue

            print(f"\n{trade_date.strftime('%Y-%m-%d %A')}:")

            # 1. 分析市场情绪
            market_data = generate_hybrid_market_data()
            sentiment = strategy.analyze_market_sentiment(market_data, trade_date)

            bias_str = "多头" if sentiment.sentiment_bias == 'long' else \
                      "空头" if sentiment.sentiment_bias == 'short' else "中性"
            print(f"  市场情绪: {bias_str} (信心度: {sentiment.confidence:.2f})")

            # 2. 选择方向
            side = strategy.select_position_side(sentiment)
            side_str = "做多" if side == PositionSide.LONG else "做空"
            print(f"  选择方向: {side_str}")

            # 3. 扫描机会
            stock_ops = strategy.scan_stock_opportunities(market_data, trade_date)
            futures_contracts = strategy.scan_futures_contracts(trade_date)

            print(f"  股票机会: {len(stock_ops)}个")
            print(f"  期货机会: {len(futures_contracts)}个")

            # 4. 开仓（如果信心度>0.65）
            if sentiment.confidence > 0.65 and strategy.can_enter_position(trade_date):
                # 选择最佳机会（股票或期货）
                best_stock = stock_ops[0] if stock_ops else None
                best_futures = futures_contracts[0] if futures_contracts else None

                # 优先选择波动更大的
                if best_futures and best_futures.volatility > 0.25:
                    print(f"\n  开仓: 期货 {best_futures.symbol}")
                    position = strategy.enter_futures_position(
                        best_futures, side, trade_date, initial_capital
                    )
                    print(f"    合约数: {position.contracts}")
                    print(f"    保证金: ${position.margin_used:,.0f}")
                    print(f"    预留: ${position.reserve_capital:,.0f}")
                    print(f"    风险度: {position.risk_level*100:.1f}%")

                elif best_stock and best_stock['score'] >= 4:
                    print(f"\n  开仓: 股票 {best_stock['symbol']}")
                    position = strategy.enter_stock_position(
                        best_stock, trade_date, initial_capital
                    )
                    print(f"    持仓: {position.shares}股")
                    print(f"    市值: ${position.current_value:,.0f}")

                # 检查现有持仓
                if strategy.positions:
                    for position in strategy.positions[:]:
                        np.random.seed(hash(trade_date.strftime('%Y-%m-%d') + position.symbol) % 10000)
                        price_change = np.random.normal(0, 0.02)
                        current_price = position.entry_price * (1 + price_change)

                        should_exit, reason = strategy.should_exit_position(position, current_price)

                        if should_exit:
                            strategy.exit_position(position, trade_date, current_price, reason)
                            print(f"  平仓 {position.symbol}: {reason}")
                            print(f"    盈亏: ${position.pnl:+,.2f} ({position.pnl_pct:+.2f}%)")

        # 周五强制平仓
        friday = week_start + timedelta(days=4)
        for position in strategy.positions[:]:
            np.random.seed(hash(friday.strftime('%Y-%m-%d') + position.symbol) % 10000)
            price_change = np.random.normal(0, 0.01)
            exit_price = position.entry_price * (1 + price_change)
            strategy.exit_position(position, friday, exit_price, "周五收盘强制平仓")
            print(f"  周五平仓 {position.symbol}: ${position.pnl:+,.2f} ({position.pnl_pct:+.2f}%)")

        # 计算本周表现
        week_trades = [p for p in strategy.closed_positions
                       if p.entry_date >= week_start and p.entry_date < week_end]
        week_profit = sum(p.pnl for p in week_trades)

        print(f"\n📊 本周总结:")
        print(f"  交易次数: {len(week_trades)}")
        print(f"  实现利润: ${week_profit:+,.2f}")
        print(f"  目标利润: ${strategy.weekly_target:,.2f}")
        print(f"  目标达成: {'✅' if week_profit >= strategy.weekly_target else '❌'}")
        print(f"  收益率: {week_profit/strategy.initial_capital*100:+.2f}%")

        current_date = week_end

    print(strategy.generate_report())

    return {
        'strategy': strategy,
        'total_profit': sum(p.pnl for p in strategy.closed_positions),
    }


def generate_hybrid_market_data() -> Dict[str, pd.DataFrame]:
    """生成融合市场数据"""
    np.random.seed(42)

    symbols = [
        # 股票
        'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA',
        '0700.HK', '9988.HK', '3690.HK',
        # 期货
        'RB', 'CU', 'AL', 'ZN', 'AU', 'AG', 'CL', 'MA', 'PP', 'L',
        'M', 'Y', 'P', 'A', 'C', 'JD', 'CF', 'SR', 'OI', 'RM',
        'IF', 'IH', 'IC', 'IM',
    ]

    market_data = {}

    for symbol in symbols:
        dates = pd.date_range(start='2026-05-01', periods=50, freq='D')

        returns = np.random.normal(0.0005, 0.02, 50)
        prices = [100.0 if symbol.startswith('0') or len(symbol) <= 4 else 3000.0]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        df = pd.DataFrame({
            'Close': prices,
        }, index=dates)

        market_data[symbol] = df

    return market_data


if __name__ == "__main__":
    # 运行4周模拟
    result = simulate_hybrid_strategy(
        initial_capital=1_000_000,
        weeks=4,
    )

    print(f"\n{'='*80}")
    print(f" " * 30 + "模拟总结")
    print(f"{'='*80}")
    print(f"\n总收益: ${result['total_profit']:,.0f}")
