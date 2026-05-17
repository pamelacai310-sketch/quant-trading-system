"""
远期期货合约交易策略

核心特点：
1. 只做远期合约，选择波动幅度高于主力合约的
2. 期货仓位风险度维持在50%以下（保证金占比）
3. 每天根据市场多空情绪选择开仓方向

策略逻辑：
- 选择远期合约（至少+2个月）
- 对比波动幅度，选高波动的
- 严格资金管理（保证金占用<50%）
- 每日判断市场情绪，动态调整多空方向
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

from ..futures_specs import calculate_futures_margin, calculate_futures_notional, futures_contract_multiplier


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"


@dataclass
class FuturesContract:
    """期货合约"""
    symbol: str
    name: str
    underlying: str  # 标的物
    delivery_month: int  # 交割月份（1-12）
    delivery_year: int  # 交割年份
    is_main: bool  # 是否主力合约
    volatility: float = 0.0  # 波动率
    volume: float = 0.0  # 成交量
    open_interest: float = 0.0  # 持仓量
    current_price: float = 0.0
    margin_rate: float = 0.15  # 保证金比例（默认15%）
    contract_multiplier: float = 0.0  # 交易乘数，例如 CU=5吨/手

    def __post_init__(self) -> None:
        if not self.contract_multiplier or self.contract_multiplier <= 0:
            self.contract_multiplier = futures_contract_multiplier(self.symbol or self.underlying)

    def notional_value(self, contracts_count: int = 1, price: Optional[float] = None) -> float:
        """合约名义价值 = 最新价 * 交易乘数 * 手数。"""
        effective_price = self.current_price if price is None else price
        return calculate_futures_notional(
            self.symbol or self.underlying,
            effective_price,
            lots=contracts_count,
            multiplier=self.contract_multiplier,
        )

    def margin_requirement(self, contracts_count: int = 1, price: Optional[float] = None) -> float:
        """保证金 = 最新价 * 交易乘数 * 手数 * 保证金率。"""
        effective_price = self.current_price if price is None else price
        return calculate_futures_margin(
            self.symbol or self.underlying,
            effective_price,
            lots=contracts_count,
            margin_rate=self.margin_rate,
            multiplier=self.contract_multiplier,
        )

    @property
    def delivery_date(self) -> str:
        """交割日期字符串"""
        return f"{self.delivery_year}-{self.delivery_month:02d}"

    @property
    def months_to_delivery(self) -> int:
        """距离交割的月数"""
        current_date = datetime.now()
        delivery_date = datetime(self.delivery_year, self.delivery_month, 1)
        months = (delivery_date.year - current_date.year) * 12 + \
                 (delivery_date.month - current_date.month)
        return max(0, months)

    @property
    def is_far_contract(self) -> bool:
        """是否是远期合约（至少+2个月）"""
        return self.months_to_delivery >= 2


@dataclass
class FuturesPosition:
    """期货持仓"""
    contract: FuturesContract
    side: PositionSide
    entry_date: datetime
    entry_price: float
    contracts: int  # 合约数量
    margin_used: float  # 占用保证金
    reserve_capital: float  # 预留资金（1倍保证金）
    stop_loss: float
    take_profit: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    @property
    def current_value(self) -> float:
        """当前市值"""
        if self.exit_price:
            return self.contract.notional_value(self.contracts, price=self.exit_price)
        return self.contract.notional_value(self.contracts, price=self.entry_price)

    @property
    def pnl(self) -> float:
        """盈亏"""
        if self.exit_price is None:
            return 0.0

        price_diff = self.exit_price - self.entry_price
        if self.side == PositionSide.SHORT:
            price_diff = -price_diff

        return price_diff * self.contracts * self.contract.contract_multiplier

    @property
    def pnl_pct(self) -> float:
        """盈亏百分比（基于保证金）"""
        if self.margin_used == 0:
            return 0.0
        return self.pnl / self.margin_used * 100

    @property
    def risk_level(self) -> float:
        """风险度（保证金占用/（保证金+预留资金））"""
        total_capital = self.margin_used + self.reserve_capital
        if total_capital == 0:
            return 0.0
        return self.margin_used / total_capital


@dataclass
class MarketSentiment:
    """市场多空情绪"""
    date: datetime
    total_symbols: int
    up_symbols: int  # 上涨品种数
    down_symbols: int  # 下跌品种数
    up_ratio: float  # 上涨品种占比
    top20_avg_gain: float  # 前20大涨幅品种平均涨幅
    top20_avg_loss: float  # 前20大跌幅品种平均跌幅
    sentiment_bias: str  # 情绪倾向：long/short/neutral
    confidence: float  # 信心度（0-1）


class FarMonthFuturesStrategy:
    """
    远期期货合约交易策略

    核心规则：
    1. 只做远期合约（+2个月以上），选择高波动的
    2. 期货仓位风险度<50%（保证金占用<50%）
    3. 每日根据多空情绪选择方向
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        max_risk_level: float = 0.50,  # 最大风险度50%
        base_stop_loss: float = 0.03,  # 3%止损
        base_take_profit: float = 0.06,  # 6%止盈
        min_far_months: int = 2,  # 至少+2个月
    ):
        """
        初始化策略

        参数:
            initial_capital: 初始资金
            max_risk_level: 最大风险度（保证金占用比例）
            base_stop_loss: 基础止损幅度
            base_take_profit: 基础止盈幅度
            min_far_months: 最少远期月数
        """
        self.initial_capital = initial_capital
        self.max_risk_level = max_risk_level
        self.base_stop_loss = base_stop_loss
        self.base_take_profit = base_take_profit
        self.min_far_months = min_far_months

        # 可交易期货合约池
        self.futures_universe = [
            # 商品期货
            'RB',   # 螺纹钢
            'HC',   # 热轧卷板
            'CU',   # 铜
            'AL',   # 铝
            'ZN',   # 锌
            'NI',   # 镍
            'SN',   # 锡
            'AU',   # 黄金
            'AG',   # 白银
            'CL',   # 原油
            'SC',   # 原油（上海）
            'FU',   # 燃料油
            'MA',   # 甲醇
            'PP',   # PP
            'L',    # 塑料
            'V',    # PVC
            'TA',   # PTA
            'EG',   # 乙二醇
            'RB',   # 螺纹钢
            'M',    # 豆粕
            'Y',    # 豆油
            'P',    # 棕榈油
            'A',    # 豆一
            'C',    # 玉米
            'CS',   # 玉米淀粉
            'JD',   # 鸡蛋
            'HC',   # 猪期货
            'CF',   # 棉花
            'SR',   # 白糖
            'OI',   # 菜油
            'RM',   # 菜粕
            'FG',   # 玻璃
            'SA',   # 纯碱
            'UR',   # 尿素
            # 股指期货
            'IF',   # 沪深300
            'IH',   # 上证50
            'IC',   # 中证500
            'IM',   # 中证1000
            # 国债期货
            'T',    # 10年期国债
            'TF',   # 5年期国债
            'TS',   # 2年期国债
        ]

        # 持仓记录
        self.positions: List[FuturesPosition] = []
        self.closed_positions: List[FuturesPosition] = []

        # 每日市场情绪历史
        self.sentiment_history: List[MarketSentiment] = []

    def get_available_contracts(self, underlying: str,
                                current_date: datetime) -> List[FuturesContract]:
        """
        获取某标的物的所有可用合约

        参数:
            underlying: 标的物代码
            current_date: 当前日期

        返回:
            可用合约列表
        """
        contracts = []

        # 获取当前月份和年份
        current_month = current_date.month
        current_year = current_date.year

        # 生成未来12个月的合约
        for i in range(1, 13):
            delivery_month = current_month + i
            delivery_year = current_year

            if delivery_month > 12:
                delivery_month -= 12
                delivery_year += 1

            # 模拟合约数据
            # 实际应该从交易所获取
            np.random.seed(hash(underlying + str(delivery_year) + str(delivery_month)) % 10000)

            is_main = (i == 1)  # 最近月份是主力

            # 远期合约通常波动更大
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
                margin_rate=0.15,  # 商品期货15%保证金
            )

            contracts.append(contract)

        return contracts

    def _get_contract_name(self, underlying: str) -> str:
        """获取合约名称"""
        names = {
            'RB': '螺纹钢', 'HC': '热轧卷板', 'CU': '铜', 'AL': '铝',
            'ZN': '锌', 'NI': '镍', 'SN': '锡', 'AU': '黄金', 'AG': '白银',
            'CL': '原油', 'SC': '原油(上海)', 'FU': '燃料油',
            'MA': '甲醇', 'PP': 'PP', 'L': '塑料', 'V': 'PVC',
            'TA': 'PTA', 'EG': '乙二醇', 'M': '豆粕', 'Y': '豆油',
            'P': '棕榈油', 'A': '豆一', 'C': '玉米', 'CS': '玉米淀粉',
            'JD': '鸡蛋', 'HC': '猪期货', 'CF': '棉花',
            'SR': '白糖', 'OI': '菜油', 'RM': '菜粕',
            'FG': '玻璃', 'SA': '纯碱', 'UR': '尿素',
            'IF': '沪深300', 'IH': '上证50', 'IC': '中证500',
            'IM': '中证1000', 'T': '10年期国债',
            'TF': '5年期国债', 'TS': '2年期国债',
        }
        return names.get(underlying, underlying)

    def select_far_month_contract(self, underlying: str,
                                  current_date: datetime) -> Optional[FuturesContract]:
        """
        选择远期合约

        规则：
        1. 至少+2个月
        2. 波动幅度高于主力合约
        3. 如果多个远期合约，选波动最大的

        参数:
            underlying: 标的物代码
            current_date: 当前日期

        返回:
            选中的远期合约，如果没有合适的返回None
        """
        # 获取所有合约
        all_contracts = self.get_available_contracts(underlying, current_date)

        # 筛选远期合约（至少+2个月）
        far_contracts = [c for c in all_contracts if c.is_far_contract]

        if not far_contracts:
            return None

        # 获取主力合约
        main_contract = next((c for c in all_contracts if c.is_main), None)

        if not main_contract:
            # 如果没有主力合约，选波动最大的远期合约
            return max(far_contracts, key=lambda c: c.volatility)

        # 选择波动高于主力的远期合约
        higher_vol_contracts = [
            c for c in far_contracts
            if c.volatility > main_contract.volatility
        ]

        if not higher_vol_contracts:
            # 如果没有波动更高的，返回波动最大的远期合约
            return max(far_contracts, key=lambda c: c.volatility)

        # 返回波动最大的远期合约
        return max(higher_vol_contracts, key=lambda c: c.volatility)

    def scan_contracts(self, current_date: datetime) -> List[FuturesContract]:
        """
        扫描所有标的物，选择符合要求的远期合约

        返回:
            符合条件的远期合约列表
        """
        suitable_contracts = []

        for underlying in self.futures_universe:
            # 为每个标的物选择远期合约
            contract = self.select_far_month_contract(underlying, current_date)

            if contract is None:
                continue

            # 检查流动性（成交量>1000）
            if contract.volume < 1000:
                continue

            suitable_contracts.append(contract)

        # 按波动率排序
        suitable_contracts.sort(key=lambda c: c.volatility, reverse=True)

        return suitable_contracts

    def calculate_margin_requirement(self, contract: FuturesContract,
                                    contracts_count: int) -> Tuple[float, float]:
        """
        计算保证金需求和预留资金

        规则：
        - 保证金占用 < 50%
        - 预留资金 >= 保证金（1倍）

        参数:
            contract: 期货合约
            contracts_count: 合约数量

        返回:
            (保证金需求, 预留资金)
        """
        # 计算保证金：最新价 * 交易乘数 * 手数 * 保证金率
        margin_required = contract.margin_requirement(contracts_count)

        # 预留资金 = 保证金（1倍）
        reserve_capital = margin_required

        # 检查风险度
        risk_level = margin_required / (margin_required + reserve_capital)

        # 如果风险度超过50%，调整合约数量
        if risk_level > self.max_risk_level:
            # 调整到正好50%风险度
            target_reserve = margin_required
            target_total = margin_required / self.max_risk_level
            adjusted_contracts = int((target_total * contract.margin_rate) / (contract.current_price * contract.contract_multiplier))
            margin_required = contract.margin_requirement(adjusted_contracts)
            reserve_capital = margin_required
            contracts_count = adjusted_contracts

        return margin_required, reserve_capital

    def analyze_market_sentiment(self, market_data: Dict[str, pd.DataFrame],
                               current_date: datetime) -> MarketSentiment:
        """
        分析市场多空情绪

        规则：
        - 如果当天上涨品种数 > 60%，情绪偏多
        - 如果前20大涨幅品种平均涨幅 > 前20大跌幅品种平均跌幅，情绪偏多

        参数:
            market_data: 市场数据 {symbol: DataFrame}
            current_date: 当前日期

        返回:
            市场情绪
        """
        # 计算每个品种的涨跌
        daily_changes = []

        for symbol, data in market_data.items():
            if len(data) < 2:
                continue

            # 计算涨跌幅
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

        # 计算前20大涨幅品种平均涨幅
        top20_gains = sorted([c for c in daily_changes if c['change_pct'] > 0],
                           key=lambda x: x['change_pct'], reverse=True)[:20]
        top20_avg_gain = np.mean([c['change_pct'] for c in top20_gains]) if top20_gains else 0

        # 计算前20大跌幅品种平均跌幅（取绝对值）
        top20_losses = sorted([c for c in daily_changes if c['change_pct'] < 0],
                            key=lambda x: x['change_pct'])[:20]
        top20_avg_loss = np.mean([abs(c['change_pct']) for c in top20_losses]) if top20_losses else 0

        # 判断情绪倾向
        bias = 'neutral'
        confidence = 0.5

        # 条件1: 上涨品种数 > 60%
        if up_ratio > 0.60:
            bias = 'long'
            confidence = min(0.9, 0.5 + (up_ratio - 0.60) * 2)
        elif up_ratio < 0.40:
            bias = 'short'
            confidence = min(0.9, 0.5 + (0.40 - up_ratio) * 2)

        # 条件2: 前20大涨幅品种平均涨幅 > 前20大跌幅品种平均跌幅
        if top20_avg_gain > top20_avg_loss:
            if bias == 'neutral':
                bias = 'long'
                confidence = min(0.8, 0.5 + (top20_avg_gain - top20_avg_loss) * 5)
            elif bias == 'long':
                confidence = min(0.95, confidence + 0.2)
            else:  # bias == 'short'
                bias = 'neutral'
                confidence = 0.5
        elif top20_avg_loss > top20_avg_gain:
            if bias == 'neutral':
                bias = 'short'
                confidence = min(0.8, 0.5 + (top20_avg_loss - top20_avg_gain) * 5)
            elif bias == 'short':
                confidence = min(0.95, confidence + 0.2)
            else:  # bias == 'long'
                bias = 'neutral'
                confidence = 0.5

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
        """
        根据市场情绪选择开仓方向

        参数:
            sentiment: 市场情绪

        返回:
            持仓方向
        """
        if sentiment.sentiment_bias == 'long':
            return PositionSide.LONG
        elif sentiment.sentiment_bias == 'short':
            return PositionSide.SHORT
        else:
            # 中性情绪，根据其他因素决定
            # 这里默认选择多头，实际可以更复杂
            return PositionSide.LONG

    def calculate_position_size(self, contract: FuturesContract,
                               available_capital: float) -> int:
        """
        计算合约数量

        参数:
            contract: 期货合约
            available_capital: 可用资金

        返回:
            合约数量
        """
        # 计算单个合约保证金
        single_contract_margin = contract.margin_requirement(1)

        # 可用保证金（50%风险度）
        available_margin = available_capital * 0.50

        # 计算合约数量
        contracts = int(available_margin / single_contract_margin)

        return max(0, contracts)

    def enter_position(self, contract: FuturesContract, side: PositionSide,
                      entry_date: datetime, capital: float) -> FuturesPosition:
        """
        开仓

        参数:
            contract: 期货合约
            side: 持仓方向
            entry_date: 入场日期
            capital: 可用资金

        返回:
            期货持仓
        """
        # 计算合约数量
        contracts = self.calculate_position_size(contract, capital)
        if contracts <= 0:
            raise ValueError(
                f"资金不足，无法开 1 手 {contract.symbol}: "
                f"需要保证金 {contract.margin_requirement(1):,.2f}，可用资金 {capital:,.2f}"
            )

        # 计算保证金和预留资金
        margin_used, reserve_capital = self.calculate_margin_requirement(
            contract, contracts
        )

        # 创建持仓
        position = FuturesPosition(
            contract=contract,
            side=side,
            entry_date=entry_date,
            entry_price=contract.current_price,
            contracts=contracts,
            margin_used=margin_used,
            reserve_capital=reserve_capital,
            stop_loss=self.base_stop_loss,
            take_profit=self.base_take_profit,
        )

        self.positions.append(position)

        return position

    def should_exit_position(self, position: FuturesPosition,
                            current_price: float) -> Tuple[bool, Optional[str]]:
        """
        判断是否应该平仓

        参数:
            position: 期货持仓
            current_price: 当前价格

        返回:
            (是否平仓, 原因)
        """
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

        return False, None

    def exit_position(self, position: FuturesPosition, exit_date: datetime,
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
        report.append(" " * 25 + "远期期货合约策略报告")
        report.append("="*80)

        # 策略参数
        report.append(f"\n📊 策略参数:")
        report.append(f"  初始资金: ${self.initial_capital:,.0f}")
        report.append(f"  最大风险度: {self.max_risk_level*100:.0f}%")
        report.append(f"  最少远期月数: {self.min_far_months}")
        report.append(f"  止损幅度: {self.base_stop_loss*100:.0f}%")
        report.append(f"  止盈幅度: {self.base_take_profit*100:.0f}%")

        # 当前持仓
        report.append(f"\n📈 当前持仓:")
        if not self.positions:
            report.append(f"  无持仓")
        else:
            report.append(f"\n{'合约':<20} {'方向':<6} {'合约数':<8} {'保证金':<12} {'预留资金':<12} "
                         f"{'风险度':<8} {'盈亏%':<8}")
            report.append("-" * 80)

            for pos in self.positions:
                side_str = "做多" if pos.side == PositionSide.LONG else "做空"
                risk_str = f"{pos.risk_level*100:.1f}%"
                pnl_str = f"{pos.pnl_pct:+.2f}%"

                report.append(f"{pos.contract.symbol:<20} {side_str:<6} {pos.contracts:<8} "
                            f"${pos.margin_used:>10,.0f} ${pos.reserve_capital:>10,.0f} "
                            f"{risk_str:<8} {pnl_str:<8}")

        # 已平仓持仓
        report.append(f"\n✅ 已平仓持仓:")
        if not self.closed_positions:
            report.append(f"  无已平仓")
        else:
            report.append(f"\n{'合约':<20} {'方向':<6} {'入场价':<10} {'出场价':<10} "
                         f"{'盈亏':<12} {'盈亏%':<10} {'原因'}")
            report.append("-" * 100)

            for pos in self.closed_positions[-10:]:  # 显示最近10笔
                side_str = "做多" if pos.side == PositionSide.LONG else "做空"
                pnl_str = f"+${pos.pnl:,.2f}" if pos.pnl >= 0 else f"-${abs(pos.pnl):,.2f}"
                reason_str = pos.exit_reason[:15] if pos.exit_reason else ""

                report.append(f"{pos.contract.symbol:<20} {side_str:<6} ${pos.entry_price:<9.2f} "
                            f"${pos.exit_price:<9.2f} {pnl_str:<12} {pos.pnl_pct:>9.2f}% "
                            f"{reason_str}")

        # 统计数据
        total_trades = len(self.closed_positions)
        winning_trades = [p for p in self.closed_positions if p.pnl > 0]
        total_profit = sum(p.pnl for p in self.closed_positions)

        report.append(f"\n📊 统计数据:")
        report.append(f"  总交易次数: {total_trades}")
        report.append(f"  盈利次数: {len(winning_trades)}")
        report.append(f"  胜率: {len(winning_trades)/total_trades*100:.1f}%" if total_trades > 0 else "  胜率: N/A")
        report.append(f"  总盈亏: ${total_profit:+,.2f}")

        # 市场情绪历史
        if self.sentiment_history:
            report.append(f"\n🌡️  最近市场情绪:")
            for sentiment in self.sentiment_history[-5:]:
                bias_str = "多头" if sentiment.sentiment_bias == 'long' else \
                          "空头" if sentiment.sentiment_bias == 'short' else "中性"
                report.append(f"  {sentiment.date.strftime('%Y-%m-%d')}: "
                            f"{bias_str} (信心度: {sentiment.confidence:.2f})")

        return "\n".join(report)


def simulate_far_month_futures_strategy(
    initial_capital: float = 1_000_000,
    days: int = 20,
) -> Dict[str, Any]:
    """
    模拟远期期货策略

    参数:
        initial_capital: 初始资金
        days: 模拟天数

    返回:
        模拟结果
    """
    strategy = FarMonthFuturesStrategy(
        initial_capital=initial_capital,
        max_risk_level=0.50,
        min_far_months=2,
    )

    current_date = datetime(2026, 5, 4)

    print("\n" + "="*80)
    print(" " * 20 + "远期期货合约策略模拟")
    print("="*80)

    for day in range(days):
        if current_date.weekday() >= 5:  # 跳过周末
            current_date += timedelta(days=1)
            continue

        print(f"\n{'='*80}")
        print(f"{current_date.strftime('%Y-%m-%d %A')}")
        print(f"{'='*80}")

        # 1. 扫描远期合约
        print(f"\n步骤1: 扫描远期合约")
        suitable_contracts = strategy.scan_contracts(current_date)

        print(f"  找到 {len(suitable_contracts)} 个合适的远期合约")
        for contract in suitable_contracts[:5]:
            months_str = f"+{contract.months_to_delivery}个月"
            print(f"    {contract.symbol}: {contract.name}, "
                  f"交割: {contract.delivery_date}, "
                  f"波动率: {contract.volatility*100:.1f}%, "
                  f"{months_str}")

        # 2. 分析市场情绪
        print(f"\n步骤2: 分析市场情绪")

        # 生成模拟市场数据
        market_data = generate_mock_futures_market_data()
        sentiment = strategy.analyze_market_sentiment(market_data, current_date)
        strategy.sentiment_history.append(sentiment)

        bias_str = "多头" if sentiment.sentiment_bias == 'long' else \
                  "空头" if sentiment.sentiment_bias == 'short' else "中性"

        print(f"  上涨品种: {sentiment.up_symbols}/{sentiment.total_symbols} "
              f"({sentiment.up_ratio*100:.1f}%)")
        print(f"  前20涨幅平均: {sentiment.top20_avg_gain*100:+.2f}%")
        print(f"  前20跌幅平均: {sentiment.top20_avg_loss*100:+.2f}%")
        print(f"  市场情绪: {bias_str} (信心度: {sentiment.confidence:.2f})")

        # 3. 选择持仓方向
        side = strategy.select_position_side(sentiment)
        side_str = "做多" if side == PositionSide.LONG else "做空"
        print(f"\n步骤3: 选择持仓方向: {side_str}")

        # 4. 开仓（如果情绪足够强烈）
        if sentiment.confidence > 0.65 and suitable_contracts:
            # 选择波动最大的合约
            best_contract = suitable_contracts[0]

            # 检查当前持仓数量
            if len(strategy.positions) < 3:  # 最多3个持仓
                position = strategy.enter_position(
                    best_contract, side, current_date, initial_capital
                )

                print(f"\n步骤4: 开仓")
                print(f"  合约: {position.contract.symbol}")
                print(f"  方向: {side_str}")
                print(f"  合约数: {position.contracts}")
                print(f"  保证金: ${position.margin_used:,.0f}")
                print(f"  预留资金: ${position.reserve_capital:,.0f}")
                print(f"  风险度: {position.risk_level*100:.1f}%")
                print(f"  止损: {position.stop_loss*100:.0f}%")
                print(f"  止盈: {position.take_profit*100:.0f}%")

        # 5. 检查现有持仓是否需要平仓
        if strategy.positions:
            print(f"\n步骤5: 检查持仓")

            for position in strategy.positions[:]:
                # 模拟当前价格
                np.random.seed(hash(current_date.strftime('%Y-%m-%d') + position.contract.symbol) % 10000)
                price_change = np.random.normal(0, 0.02)
                current_price = position.entry_price * (1 + price_change)

                should_exit, reason = strategy.should_exit_position(position, current_price)

                if should_exit:
                    strategy.exit_position(position, current_date, current_price, reason)
                    print(f"  平仓 {position.contract.symbol}: {reason}")
                    print(f"    盈亏: ${position.pnl:+,.2f} ({position.pnl_pct:+.2f}%)")

        current_date += timedelta(days=1)

    # 生成最终报告
    print(strategy.generate_report())

    return {
        'strategy': strategy,
        'total_profit': sum(p.pnl for p in strategy.closed_positions),
        'total_trades': len(strategy.closed_positions),
    }


def generate_mock_futures_market_data() -> Dict[str, pd.DataFrame]:
    """生成模拟期货市场数据"""
    np.random.seed(42)

    symbols = [
        'RB', 'CU', 'AL', 'ZN', 'AU', 'AG', 'CL', 'MA', 'PP', 'L',
        'M', 'Y', 'P', 'A', 'C', 'JD', 'CF', 'SR', 'OI', 'RM',
        'IF', 'IH', 'IC', 'IM', 'T', 'TF', 'TS',
    ]

    market_data = {}

    for symbol in symbols:
        # 生成50天的价格数据
        dates = pd.date_range(start='2026-04-01', periods=50, freq='D')

        # 随机游走
        returns = np.random.normal(0.0005, 0.025, 50)
        prices = [3000.0]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        df = pd.DataFrame({
            'Close': prices,
        }, index=dates)

        market_data[symbol] = df

    return market_data


if __name__ == "__main__":
    # 运行模拟
    result = simulate_far_month_futures_strategy(
        initial_capital=1_000_000,
        days=20,
    )

    print(f"\n{'='*80}")
    print(f" " * 30 + "模拟总结")
    print(f"{'='*80}")
    print(f"\n总收益: ${result['total_profit']:,.2f}")
    print(f"总交易次数: {result['total_trades']}")
