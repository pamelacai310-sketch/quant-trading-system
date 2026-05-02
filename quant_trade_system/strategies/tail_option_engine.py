"""
塔勒布尾部期权量化执行引擎

核心功能：
1. 建仓机制：Delta -0.05至-0.10，DTE 90-180天
2. 动态展期：时间展期 + 现价展期
3. 危机止盈：Gamma/Vega挤压时自动兑现
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum


class OptionType(Enum):
    """期权类型"""
    PUT = "put"
    CALL = "call"


class CrisisTriggerType(Enum):
    """危机触发类型"""
    DELTA_SURGE = "delta_surge"  # Delta激增
    VIX_SPIKE = "vix_spike"  # VIX飙升
    IV_PERCENTILE = "iv_percentile"  # IV历史分位


@dataclass
class OptionContract:
    """期权合约"""
    symbol: str
    option_type: OptionType
    strike: float
    expiration: datetime
    delta: float
    gamma: float
    vega: float
    theta: float
    iv: float
    underlying_price: float
    position_size: int  # 正数表示多头，负数表示空头
    entry_price: float
    entry_date: datetime

    @property
    def days_to_expiration(self) -> int:
        """距离到期日天数"""
        return (self.expiration - datetime.now()).days

    @property
    def moneyness(self) -> float:
        """货币性：行权价/标的价格"""
        return self.strike / self.underlying_price

    @property
    def is_deep_otm(self) -> bool:
        """是否深度虚值"""
        if self.option_type == OptionType.PUT:
            return self.strike < self.underlying_price * 0.85
        return self.strike > self.underlying_price * 1.15


@dataclass
class CrisisEvent:
    """危机事件"""
    trigger_type: CrisisTriggerType
    trigger_time: datetime
    trigger_value: float
    description: str
    recommended_action: str


@dataclass
class RollInstruction:
    """展期指令"""
    action: str  # "time_roll", "strike_roll", "monetize", "hold"
    reason: str
    current_position: OptionContract
    new_contract_spec: Optional[Dict] = None
    percentage_to_close: Optional[float] = None


class TailOptionEngine:
    """
    尾部期权量化执行引擎

    核心算法：
    1. 建仓：Delta -0.05至-0.10，DTE 90-180天
    2. 展期：时间<45天 或 Delta<0.02
    3. 止盈：Delta>-0.50 或 VIX>40
    """

    def __init__(
        self,
        target_delta_range: Tuple[float, float] = (-0.10, -0.05),
        min_dte: int = 90,
        max_dte: int = 180,
        roll_trigger_dte: int = 45,
        roll_trigger_delta: float = 0.02,
        crisis_delta_threshold: float = -0.50,
        crisis_vix_threshold: float = 40.0,
        crisis_iv_percentile: float = 0.99,
    ):
        """
        初始化尾部期权引擎

        参数:
            target_delta_range: 目标Delta范围 (最小, 最大)
            min_dte: 最小到期日天数
            max_dte: 最大到期日天数
            roll_trigger_dte: 时间展期触发点（天）
            roll_trigger_delta: 现价展期触发点（Delta绝对值）
            crisis_delta_threshold: 危机Delta阈值
            crisis_vix_threshold: 危机VIX阈值
            crisis_iv_percentile: 危机IV历史分位阈值
        """
        self.target_delta_range = target_delta_range
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.roll_trigger_dte = roll_trigger_dte
        self.roll_trigger_delta = roll_trigger_delta
        self.crisis_delta_threshold = crisis_delta_threshold
        self.crisis_vix_threshold = crisis_vix_threshold
        self.crisis_iv_percentile = crisis_iv_percentile

        self.positions: List[OptionContract] = []
        self.crisis_history: List[CrisisEvent] = []

    def calculate_option_greeks(
        self,
        underlying_price: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        iv: float,
        option_type: OptionType = OptionType.PUT,
    ) -> Dict[str, float]:
        """
        计算期权希腊字母（使用Black-Scholes模型）

        简化版实现，生产环境建议使用量化库如py_vollib
        """
        from scipy.stats import norm

        S = underlying_price
        K = strike
        T = time_to_expiry / 365.0  # 转换为年
        r = risk_free_rate
        sigma = iv

        if T <= 0:
            return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0}

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == OptionType.PUT:
            delta = -norm.cdf(-d1)
        else:
            delta = norm.cdf(d1)

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # 每1%波动率变化
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                 r * K * np.exp(-r * T) * norm.cdf(d2)) / 365  # 每日时间衰减

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
        }

    def select_optimal_contract(
        self,
        symbol: str,
        underlying_price: float,
        option_chain: pd.DataFrame,
        current_date: datetime,
    ) -> Dict:
        """
        选择最优期权合约

        选择标准：
        1. Delta在目标范围内（-0.05至-0.10）
        2. DTE在90-180天
        3. 流动性最好（成交量最大）
        """
        # 过滤看跌期权
        puts = option_chain[option_chain['type'] == 'put'].copy()

        # 计算DTE
        puts['dte'] = (puts['expiration'] - current_date).dt.days

        # 过滤期限
        puts = puts[
            (puts['dte'] >= self.min_dte) &
            (puts['dte'] <= self.max_dte)
        ]

        if puts.empty:
            return None

        # 计算或使用已有Delta
        if 'delta' not in puts.columns:
            # 估算Delta（简化版）
            puts['delta'] = -0.3 * np.exp(-3 * (puts['strike'] / underlying_price - 0.85) ** 2)

        # 过滤Delta范围
        puts = puts[
            (puts['delta'] >= self.target_delta_range[0]) &
            (puts['delta'] <= self.target_delta_range[1])
        ]

        if puts.empty:
            # 如果没有完美匹配，选择最接近的
            puts = option_chain[option_chain['type'] == 'put'].copy()
            puts['dte'] = (puts['expiration'] - current_date).dt.days
            puts = puts[
                (puts['dte'] >= self.min_dte) &
                (puts['dte'] <= self.max_dte)
            ]
            puts['delta_distance'] = np.abs(puts['delta'] - np.mean(self.target_delta_range))
            puts = puts.sort_values('delta_distance')

        # 按成交量排序（流动性优先）
        if 'volume' in puts.columns:
            puts = puts.sort_values('volume', ascending=False)

        best_contract = puts.iloc[0]

        return {
            'symbol': symbol,
            'strike': best_contract['strike'],
            'expiration': best_contract['expiration'],
            'delta': best_contract.get('delta', -0.075),
            'dte': best_contract['dte'],
            'iv': best_contract.get('iv', 0.20),
            'price': best_contract.get('last', best_contract.get('mid', 0)),
        }

    def check_roll_conditions(
        self,
        position: OptionContract,
        current_underlying_price: float,
        current_vix: Optional[float] = None,
        iv_history: Optional[pd.Series] = None,
    ) -> RollInstruction:
        """
        检查展期条件

        返回展期指令
        """
        # 更新标的价格
        position.underlying_price = current_underlying_price

        # 1. 检查危机触发条件（优先级最高）
        crisis_event = self._check_crisis_conditions(
            position, current_vix, iv_history
        )
        if crisis_event:
            self.crisis_history.append(crisis_event)

            # 危机止盈逻辑
            if crisis_event.trigger_type in [
                CrisisTriggerType.DELTA_SURGE,
                CrisisTriggerType.VIX_SPIKE,
            ]:
                return RollInstruction(
                    action="monetize",
                    reason=f"危机触发: {crisis_event.description}",
                    current_position=position,
                    percentage_to_close=0.50,  # 卖出50%
                )

        # 2. 检查时间展期条件
        if position.days_to_expiration < self.roll_trigger_dte:
            return RollInstruction(
                action="time_roll",
                reason=f"时间展期: DTE={position.days_to_expiration}天<{self.roll_trigger_dte}天",
                current_position=position,
                new_contract_spec={
                    'target_delta': np.mean(self.target_delta_range),
                    'target_dte': np.mean([self.min_dte, self.max_dte]),
                }
            )

        # 3. 检查现价展期条件（Delta绝对值太小）
        if abs(position.delta) < self.roll_trigger_delta:
            return RollInstruction(
                action="strike_roll",
                reason=f"现价展期: Delta={position.delta:.3f}, 绝对值<{self.roll_trigger_delta}",
                current_position=position,
                new_contract_spec={
                    'target_delta': np.mean(self.target_delta_range),
                    'target_dte': np.mean([self.min_dte, self.max_dte]),
                }
            )

        # 4. 无需操作
        return RollInstruction(
            action="hold",
            reason="持有中，无需展期",
            current_position=position,
        )

    def _check_crisis_conditions(
        self,
        position: OptionContract,
        current_vix: Optional[float] = None,
        iv_history: Optional[pd.Series] = None,
    ) -> Optional[CrisisEvent]:
        """检查危机触发条件"""

        # 条件1：Delta激增（从-0.05变成-0.50或更高）
        if position.delta > self.crisis_delta_threshold:
            return CrisisEvent(
                trigger_type=CrisisTriggerType.DELTA_SURGE,
                trigger_time=datetime.now(),
                trigger_value=position.delta,
                description=f"Delta从{self.target_delta_range}激增至{position.delta:.2f}",
                recommended_action="分批兑现50%，剩余下移重置",
            )

        # 条件2：VIX飙升
        if current_vix and current_vix > self.crisis_vix_threshold:
            # 需要连续两日确认（简化版，这里单日触发）
            return CrisisEvent(
                trigger_type=CrisisTriggerType.VIX_SPIKE,
                trigger_time=datetime.now(),
                trigger_value=current_vix,
                description=f"VIX飙升至{current_vix:.1f}（阈值{self.crisis_vix_threshold}）",
                recommended_action="分批兑现50%，剩余下移重置",
            )

        # 条件3：IV历史分位
        if iv_history is not None and len(iv_history) > 0:
            current_iv_percentile = (iv_history < position.iv).mean()
            if current_iv_percentile >= self.crisis_iv_percentile:
                return CrisisEvent(
                    trigger_type=CrisisTriggerType.IV_PERCENTILE,
                    trigger_time=datetime.now(),
                    trigger_value=current_iv_percentile,
                    description=f"IV位于历史{current_iv_percentile*100:.1f}%分位",
                    recommended_action="分批兑现50%，剩余下移重置",
                )

        return None

    def calculate_theta_bleed(
        self,
        positions: List[OptionContract],
        days_passed: int = 1,
    ) -> float:
        """
        计算Theta失血（总时间价值损耗）

        参数:
            positions: 持仓列表
            days_passed: 经过天数

        返回:
            总Theta失血金额
        """
        total_theta_bleed = 0.0

        for position in positions:
            # Theta通常是负数（多头买方每天损失时间价值）
            daily_theta_cost = position.theta * position.position_size * 100  # 假设每张合约100股
            total_theta_bleed += abs(daily_theta_cost) * days_passed

        return total_theta_bleed

    def calculate_portfolio_exposure(
        self,
        positions: List[OptionContract],
        account_value: float,
    ) -> Dict:
        """
        计算组合暴露

        返回：
            总Greek暴露
            风险指标
        """
        total_delta = sum(p.delta * p.position_size for p in positions)
        total_gamma = sum(p.gamma * p.position_size for p in positions)
        total_vega = sum(p.vega * p.position_size for p in positions)
        total_theta = sum(p.theta * p.position_size for p in positions)

        total_premium = sum(
            p.entry_price * abs(p.position_size) * 100
            for p in positions
        )

        return {
            'total_delta': total_delta,
            'total_gamma': total_gamma,
            'total_vega': total_vega,
            'total_theta': total_theta,
            'total_premium': total_premium,
            'premium_as_pct_of_account': total_premium / account_value if account_value > 0 else 0,
            'num_contracts': len(positions),
            'weighted_avg_dte': np.mean([p.days_to_expiration for p in positions]) if positions else 0,
        }

    def generate_roll_report(self) -> pd.DataFrame:
        """生成展期报告"""
        if not self.positions:
            return pd.DataFrame()

        data = []
        for position in self.positions:
            roll_instruction = self.check_roll_conditions(
                position=position,
                current_underlying_price=position.underlying_price,
            )

            data.append({
                'symbol': position.symbol,
                'strike': position.strike,
                'expiration': position.expiration.strftime('%Y-%m-%d'),
                'dte': position.days_to_expiration,
                'delta': position.delta,
                'gamma': position.gamma,
                'theta': position.theta,
                'position_size': position.position_size,
                'action': roll_instruction.action,
                'reason': roll_instruction.reason,
            })

        return pd.DataFrame(data)


def calculate_annual_theta_cost(
    option_price: float,
    days_to_expiration: int,
    position_size: int,
) -> Dict:
    """
    计算年度Theta成本

    塔勒布策略核心：了解保险的真实成本
    """
    daily_theta_bleed = option_price / days_to_expiration  # 简化线性估算
    annual_theta_bleed = daily_theta_bleed * 365

    return {
        'daily_cost_per_contract': daily_theta_bleed,
        'annual_cost_per_contract': annual_theta_bleed,
        'total_annual_cost': annual_theta_bleed * position_size,
        'annual_cost_pct_of_premium': (annual_theta_bleed / option_price) * 100,
    }
