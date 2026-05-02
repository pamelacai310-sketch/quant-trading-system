"""
塔勒布杠铃式尾部全天候量化模型

核心策略：
- 模块A（90-95%）：极度安全资产，产生4-6%收益
- 模块B（5-10%）：深度虚值看跌期权，黑天鹅保护

资金管理：
- 月度保险预算：0.3-0.5%
- 再平衡机制
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .tail_option_engine import (
    TailOptionEngine,
    OptionContract,
    OptionType,
    RollInstruction,
    CrisisTriggerType,
)


class MarketRegime(Enum):
    """市场状态"""
    BULL = "bull"  # 牛市
    BEAR = "bear"  # 熊市
    VOLATILE = "volatile"  # 震荡
    CRISIS = "crisis"  # 危机


@dataclass
class SafeAssetPosition:
    """安全资产持仓（模块A）"""
    asset_type: str  # "t_bill", "money_market", "neutral_arbitrage"
    amount: float
    current_yield: float  # 当前年化收益率
    monthly_return: float
    last_rebalance: datetime


@dataclass
class TailRiskPosition:
    """尾部风险持仓（模块B）"""
    options: List[OptionContract] = field(default_factory=list)
    total_premium_paid: float = 0.0
    monthly_budget: float = 0.0
    annual_budget_pct: float = 0.0  # 占账户比例


@dataclass
class TalebBarbellPortfolio:
    """塔勒布杠铃组合"""
    account_value: float
    safe_module: SafeAssetPosition
    tail_module: TailRiskPosition
    allocation: Dict[str, float]  # {"safe": 0.90, "tail": 0.10}

    @property
    def safe_allocation_pct(self) -> float:
        return self.allocation.get("safe", 0.90)

    @property
    def tail_allocation_pct(self) -> float:
        return self.allocation.get("tail", 0.10)


@dataclass
class PerformanceMetrics:
    """绩效指标"""
    total_return: float
    safe_module_return: float
    tail_module_return: float
    theta_bleed: float
    gamma_profit: float
    vega_profit: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float  # 胜率（尾部策略月度盈利月份占比）


class TalebBarbellStrategy:
    """
    塔勒布杠铃式尾部全天候量化策略

    核心思想：
    1. 90-95%资金在绝对安全资产（国债逆回购、货币市场基金）
    2. 5-10%资金在深度虚值看跌期权（黑天鹅保险）
    3. 动态再平衡，确保策略永续运行
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        safe_allocation: float = 0.90,
        tail_allocation: float = 0.10,
        monthly_budget_pct: float = 0.004,  # 0.4%月度预算
        target_safe_yield: float = 0.05,  # 5%目标收益
        option_engine: Optional[TailOptionEngine] = None,
    ):
        """
        初始化塔勒布杠铃策略

        参数:
            initial_capital: 初始资金
            safe_allocation: 安全资产比例（默认90%）
            tail_allocation: 尾部资产比例（默认10%）
            monthly_budget_pct: 月度保险预算比例（默认0.4%）
            target_safe_yield: 目标安全资产收益率（5%）
            option_engine: 期权执行引擎
        """
        self.initial_capital = initial_capital
        self.safe_allocation = safe_allocation
        self.tail_allocation = tail_allocation
        self.monthly_budget_pct = monthly_budget_pct
        self.target_safe_yield = target_safe_yield

        # 创建期权引擎
        self.option_engine = option_engine or TailOptionEngine()

        # 初始化组合
        self.portfolio = None
        self.performance_history: List[Dict] = []
        self.crisis_events: List[Dict] = []

        # 运行状态
        self.current_date = datetime.now()
        self.is_crisis_mode = False

    def initialize_portfolio(self) -> TalebBarbellPortfolio:
        """初始化组合"""

        # 模块A：安全资产
        safe_amount = self.initial_capital * self.safe_allocation
        safe_position = SafeAssetPosition(
            asset_type="t_bill",
            amount=safe_amount,
            current_yield=self.target_safe_yield,
            monthly_return=safe_amount * self.target_safe_yield / 12,
            last_rebalance=self.current_date,
        )

        # 模块B：尾部期权
        tail_amount = self.initial_capital * self.tail_allocation
        monthly_budget = self.initial_capital * self.monthly_budget_pct

        tail_position = TailRiskPosition(
            options=[],
            total_premium_paid=0.0,
            monthly_budget=monthly_budget,
            annual_budget_pct=self.monthly_budget_pct * 12,
        )

        # 创建组合
        self.portfolio = TalebBarbellPortfolio(
            account_value=self.initial_capital,
            safe_module=safe_position,
            tail_module=tail_position,
            allocation={"safe": self.safe_allocation, "tail": self.tail_allocation},
        )

        return self.portfolio

    def rebalance(
        self,
        current_date: datetime,
        underlying_price: float,
        vix: Optional[float] = None,
        option_chain: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        再平衡操作

        1. 安全资产产生利息，转入尾部模块
        2. 尾部模块买入期权（如果预算允许）
        3. 检查展期条件
        """
        self.current_date = current_date
        actions_taken = []

        # 1. 安全资产生息
        monthly_interest = (
            self.portfolio.safe_module.amount *
            self.portfolio.safe_module.current_yield / 12
        )
        self.portfolio.safe_module.amount += monthly_interest
        self.portfolio.safe_module.monthly_return = monthly_interest

        actions_taken.append({
            'action': 'interest_earned',
            'amount': monthly_interest,
            'description': f'安全资产产生利息: ${monthly_interest:,.2f}',
        })

        # 2. 将利息转入尾部模块（作为保险费）
        if monthly_interest > 0:
            self.portfolio.tail_module.monthly_budget += monthly_interest

            actions_taken.append({
                'action': 'budget_transfer',
                'amount': monthly_interest,
                'description': f'利息转入尾部预算: ${monthly_interest:,.2f}',
            })

        # 3. 检查是否有预算购买新期权
        budget_available = self.portfolio.tail_module.monthly_budget

        if budget_available > 0 and option_chain is not None:
            # 购买新期权
            new_contracts = self._purchase_tail_options(
                budget=budget_available,
                underlying_price=underlying_price,
                option_chain=option_chain,
                current_date=current_date,
            )

            if new_contracts:
                self.portfolio.tail_module.options.extend(new_contracts)
                premium_spent = sum(
                    c.entry_price * abs(c.position_size) * 100
                    for c in new_contracts
                )

                self.portfolio.tail_module.total_premium_paid += premium_spent
                self.portfolio.tail_module.monthly_budget -= premium_spent

                actions_taken.append({
                    'action': 'options_purchased',
                    'num_contracts': len(new_contracts),
                    'premium': premium_spent,
                    'description': f'购买{len(new_contracts)}份看跌期权, 花费${premium_spent:,.2f}',
                })

        # 4. 检查现有期权的展期条件
        roll_instructions = []
        for option in self.portfolio.tail_module.options:
            roll_instr = self.option_engine.check_roll_conditions(
                position=option,
                current_underlying_price=underlying_price,
                current_vix=vix,
            )

            if roll_instr.action != "hold":
                roll_instructions.append(roll_instr)

        # 执行展期操作
        for roll_instr in roll_instructions:
            if roll_instr.action == "monetize":
                # 危机止盈
                self._execute_crisis_monetization(roll_instr, underlying_price, option_chain)

            elif roll_instr.action in ["time_roll", "strike_roll"]:
                # 展期
                self._execute_roll(roll_instr, underlying_price, option_chain)

            actions_taken.append({
                'action': f'roll_{roll_instr.action}',
                'option_symbol': roll_instr.current_position.symbol,
                'reason': roll_instr.reason,
            })

        # 5. 更新账户价值
        self._update_account_value(underlying_price, vix)

        return {
            'date': current_date.strftime('%Y-%m-%d'),
            'actions': actions_taken,
            'portfolio_value': self.portfolio.account_value,
            'safe_module_value': self.portfolio.safe_module.amount,
            'tail_module_premium': self.portfolio.tail_module.total_premium_paid,
        }

    def _purchase_tail_options(
        self,
        budget: float,
        underlying_price: float,
        option_chain: pd.DataFrame,
        current_date: datetime,
    ) -> List[OptionContract]:
        """购买尾部看跌期权"""

        # 选择最优合约
        contract_spec = self.option_engine.select_optimal_contract(
            symbol="SPY",  # 假设标普ETF
            underlying_price=underlying_price,
            option_chain=option_chain,
            current_date=current_date,
        )

        if not contract_spec:
            return []

        # 计算可购买数量
        premium_per_contract = contract_spec['price'] * 100  # 每张合约100股
        num_contracts = int(budget / premium_per_contract)

        if num_contracts == 0:
            return []

        # 创建期权合约对象
        option = OptionContract(
            symbol="SPY",
            option_type=OptionType.PUT,
            strike=contract_spec['strike'],
            expiration=contract_spec['expiration'],
            delta=contract_spec['delta'],
            gamma=0.01,  # 简化
            vega=0.10,  # 简化
            theta=-contract_spec['price'] / contract_spec['dte'],  # 简化
            iv=contract_spec['iv'],
            underlying_price=underlying_price,
            position_size=num_contracts,
            entry_price=contract_spec['price'],
            entry_date=current_date,
        )

        return [option]

    def _execute_crisis_monetization(
        self,
        roll_instr: RollInstruction,
        underlying_price: float,
        option_chain: Optional[pd.DataFrame],
    ):
        """执行危机止盈"""

        option = roll_instr.current_position
        pct_to_close = roll_instr.percentage_to_close or 0.50

        # 计算当前期权价值（假设Delta激增后，期权价值大幅增长）
        # 简化：假设当前价格 = entry_price * (1 + profit_factor)
        profit_factor = abs(option.delta) / abs(self.option_engine.target_delta_range[0])
        current_option_price = option.entry_price * profit_factor * 5  # 假设5倍收益

        # 卖出50%仓位
        contracts_to_close = int(option.position_size * pct_to_close)
        profit = (current_option_price - option.entry_price) * contracts_to_close * 100

        # 更新持仓
        option.position_size -= contracts_to_close

        # 将利润转入安全资产
        self.portfolio.safe_module.amount += profit

        # 下移重置剩余50%
        remaining_contracts = int(option.position_size)
        if remaining_contracts > 0 and option_chain is not None:
            # 购买新的深度虚值期权
            new_options = self._purchase_tail_options(
                budget=current_option_price * remaining_contracts * 100 * 0.5,  # 用一半利润再投资
                underlying_price=underlying_price * 0.80,  # 假设市场已跌20%
                option_chain=option_chain,
                current_date=self.current_date,
            )

            # 移除旧期权，添加新期权
            self.portfolio.tail_module.options.remove(option)
            self.portfolio.tail_module.options.extend(new_options)

        # 记录危机事件
        self.crisis_events.append({
            'date': self.current_date,
            'type': 'crisis_monetization',
            'profit': profit,
            'description': f'危机止盈: Delta {option.delta:.2f}, 获利${profit:,.0f}',
        })

        # 进入危机模式
        self.is_crisis_mode = True

    def _execute_roll(
        self,
        roll_instr: RollInstruction,
        underlying_price: float,
        option_chain: Optional[pd.DataFrame],
    ):
        """执行展期操作"""

        option = roll_instr.current_position

        if option_chain is None:
            return

        # 平仓旧期权
        # 简化：假设剩余价值为50%的入场价格（因为可能是深度虚值）
        close_value = option.entry_price * option.position_size * 100 * 0.5
        self.portfolio.safe_module.amount += close_value

        # 移除旧期权
        self.portfolio.tail_module.options.remove(option)

        # 购买新期权
        new_options = self._purchase_tail_options(
            budget=close_value,
            underlying_price=underlying_price,
            option_chain=option_chain,
            current_date=self.current_date,
        )

        self.portfolio.tail_module.options.extend(new_options)

    def _update_account_value(
        self,
        underlying_price: float,
        vix: Optional[float],
    ):
        """更新账户价值"""

        # 安全资产价值
        safe_value = self.portfolio.safe_module.amount

        # 尾部期权价值（简化：使用Delta估算）
        tail_value = 0.0
        for option in self.portfolio.tail_module.options:
            # 更新标的价格
            option.underlying_price = underlying_price

            # 简化的期权价值计算
            if option.delta < -0.3:  # 深度实值，价值大增
                current_value = option.entry_price * 5
            elif option.delta < -0.1:  # 轻度实值
                current_value = option.entry_price * 2
            else:  # 虚值，价值衰减
                days_passed = (self.current_date - option.entry_date).days
                decay_factor = max(0.1, 1 - days_passed / option.days_to_expiration)
                current_value = option.entry_price * decay_factor

            tail_value += current_value * abs(option.position_size) * 100

        # 总账户价值
        self.portfolio.account_value = safe_value + tail_value

    def calculate_performance(self) -> PerformanceMetrics:
        """计算绩效指标"""

        if not self.performance_history:
            return PerformanceMetrics(
                total_return=0.0,
                safe_module_return=0.0,
                tail_module_return=0.0,
                theta_bleed=0.0,
                gamma_profit=0.0,
                vega_profit=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
            )

        # 计算总收益
        total_return = (
            self.portfolio.account_value - self.initial_capital
        ) / self.initial_capital

        # 计算模块收益
        safe_return = (
            self.portfolio.safe_module.amount -
            self.initial_capital * self.safe_allocation
        ) / (self.initial_capital * self.safe_allocation)

        tail_return = (
            self.portfolio.tail_module.total_premium_paid * -1  # 保险费是成本
        ) / (self.initial_capital * self.tail_allocation)

        # 计算Theta失血
        theta_bleed = self.option_engine.calculate_theta_bleed(
            self.portfolio.tail_module.options,
            days_passed=30,  # 月度
        )

        # 计算Gamma/Vega收益（从危机事件中统计）
        gamma_profit = sum(
            e.get('profit', 0) for e in self.crisis_events
            if e['type'] == 'crisis_monetization'
        )

        vega_profit = gamma_profit * 0.3  # 简化：假设30%来自Vega

        # 胜率：有危机事件的月份占比
        win_rate = len(self.crisis_events) / max(1, len(self.performance_history))

        return PerformanceMetrics(
            total_return=total_return,
            safe_module_return=safe_return,
            tail_module_return=tail_return,
            theta_bleed=theta_bleed,
            gamma_profit=gamma_profit,
            vega_profit=vega_profit,
            sharpe_ratio=self._calculate_sharpe(),
            max_drawdown=self._calculate_max_drawdown(),
            win_rate=win_rate,
        )

    def _calculate_sharpe(self) -> float:
        """计算夏普比率（简化版）"""
        if not self.performance_history:
            return 0.0

        returns = [
            p.get('daily_return', 0)
            for p in self.performance_history
        ]

        if not returns:
            return 0.0

        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # 假设无风险利率为2%
        risk_free_rate = 0.02 / 252  # 日化
        sharpe = (avg_return - risk_free_rate) / std_return

        return sharpe * np.sqrt(252)  # 年化

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.performance_history:
            return 0.0

        account_values = [
            p.get('account_value', self.initial_capital)
            for p in self.performance_history
        ]

        if not account_values:
            return 0.0

        peak = account_values[0]
        max_dd = 0.0

        for value in account_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    def generate_report(self) -> str:
        """生成策略报告"""

        performance = self.calculate_performance()

        report = f"""
{'='*80}
塔勒布杠铃式尾部全天候量化策略 - 运行报告
{'='*80}

📊 组合状态
----------------------------------------
账户总值: ${self.portfolio.account_value:,.2f}
初始资金: ${self.initial_capital:,.2f}
总收益: ${self.portfolio.account_value - self.initial_capital:,.2f}
收益率: {performance.total_return*100:.2f}%

🛡️ 模块A - 安全资产（{self.safe_allocation*100:.0f}%）
----------------------------------------
资产类型: {self.portfolio.safe_module.asset_type}
当前金额: ${self.portfolio.safe_module.amount:,.2f}
当前收益率: {self.portfolio.safe_module.current_yield*100:.2f}%
月度收益: ${self.portfolio.safe_module.monthly_return:,.2f}

⚡ 模块B - 尾部期权（{self.tail_allocation*100:.0f}%）
----------------------------------------
持仓合约数: {len(self.portfolio.tail_module.options)}
总支付保费: ${self.portfolio.tail_module.total_premium_paid:,.2f}
月度预算: ${self.portfolio.tail_module.monthly_budget:,.2f}
年度预算占比: {self.portfolio.tail_module.annual_budget_pct*100:.2f}%

📈 绩效指标
----------------------------------------
安全模块收益: {performance.safe_module_return*100:.2f}%
尾部模块收益: {performance.tail_module_return*100:.2f}%
Theta失血: ${performance.theta_bleed:,.2f}/月
Gamma收益: ${performance.gamma_profit:,.2f}
Vega收益: ${performance.vega_profit:,.2f}
夏普比率: {performance.sharpe_ratio:.2f}
最大回撤: {performance.max_drawdown*100:.2f}%
胜率: {performance.win_rate*100:.1f}%

🔥 危机事件（{len(self.crisis_events)}次）
----------------------------------------
"""

        for event in self.crisis_events[-5:]:  # 最近5次
            report += f"- {event['date'].strftime('%Y-%m-%d')}: {event['description']}\n"

        report += f"""
💡 策略状态
----------------------------------------
危机模式: {'是 ⚠️' if self.is_crisis_mode else '否 ✅'}
当前日期: {self.current_date.strftime('%Y-%m-%d')}

🎯 希腊字母暴露
----------------------------------------
"""

        exposure = self.option_engine.calculate_portfolio_exposure(
            self.portfolio.tail_module.options,
            self.portfolio.account_value,
        )

        report += f"""
总Delta: {exposure['total_delta']:.2f}
总Gamma: {exposure['total_gamma']:.4f}
总Vega: {exposure['total_vega']:.2f}
总Theta: {exposure['total_theta']:.2f}
加权平均DTE: {exposure['weighted_avg_dte']:.0f}天
保费占比: {exposure['premium_as_pct_of_account']*100:.2f}%

{'='*80}
"""

        return report


def simulate_taleb_barbell(
    initial_capital: float = 1_000_000,
    days: int = 252 * 3,  # 3年
    safe_yield: float = 0.05,
    crisis_scenario: Optional[str] = None,
) -> TalebBarbellStrategy:
    """
    模拟塔勒布杠铃策略

    参数:
        initial_capital: 初始资金
        days: 模拟天数
        safe_yield: 安全资产收益率
        crisis_scenario: 危机场景 ("covid_2020", "gfc_2008", "custom")
    """

    strategy = TalebBarbellStrategy(
        initial_capital=initial_capital,
        safe_allocation=0.90,
        tail_allocation=0.10,
        monthly_budget_pct=0.004,
        target_safe_yield=safe_yield,
    )

    # 初始化
    strategy.initialize_portfolio()

    # 模拟市场路径（简化版）
    np.random.seed(42)
    dates = pd.date_range(
        start=datetime.now(),
        periods=days,
        freq='D'
    )

    # 生成价格路径（带偶尔的暴跌）
    price_path = []
    current_price = 400.0  # SPY起始价格

    for day in range(days):
        # 99%的日子：正常波动
        if np.random.random() > 0.01:
            daily_return = np.random.normal(0.0003, 0.012)
        else:
            # 1%的日子：暴跌（黑天鹅）
            daily_return = np.random.normal(-0.05, 0.02)

        current_price *= (1 + daily_return)
        price_path.append(current_price)

    # 模拟VIX路径
    vix_path = []
    for price in price_path:
        # VIX与价格变动负相关
        base_vix = 15.0
        if strategy.is_crisis_mode:
            vix = np.random.uniform(40, 60)
        else:
            vix = max(12, min(35, base_vix + (400 - price) * 0.1))
        vix_path.append(vix)

    # 每月再平衡
    monthly_dates = dates[::30]  # 每30天

    for i, date in enumerate(monthly_dates):
        price_idx = min(i * 30, len(price_path) - 1)
        current_price = price_path[price_idx]
        current_vix = vix_path[price_idx]

        # 生成模拟期权链
        option_chain = pd.DataFrame({
            'type': ['put'] * 10,
            'strike': [current_price * (1 - x/100) for x in range(5, 55, 5)],
            'expiration': [date + timedelta(days=d) for d in range(90, 181, 10)],
            'last': [current_price * 0.02] * 10,
            'volume': [1000] * 10,
            'iv': [0.20] * 10,
            'delta': [-0.05 - x*0.01 for x in range(10)],
        })

        # 再平衡
        rebalance_result = strategy.rebalance(
            current_date=date,
            underlying_price=current_price,
            vix=current_vix,
            option_chain=option_chain,
        )

        # 记录历史
        strategy.performance_history.append({
            'date': date,
            'account_value': strategy.portfolio.account_value,
            'safe_value': strategy.portfolio.safe_module.amount,
            'tail_value': strategy.portfolio.account_value - strategy.portfolio.safe_module.amount,
            'underlying_price': current_price,
            'vix': current_vix,
        })

    return strategy
