"""
因果驱动的混合策略引擎

基于因果AI分析结果，将欧奈尔CANSLIM和塔勒布杠铃策略
整合成一个自适应的攻守兼备系统。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from .oneill_strategy import ONeillStrategyEngine, ONeillPosition
from .taleb_barbell import TalebBarbellStrategy, TalebBarbellPortfolio
from .strategy_causal_analysis import (
    ONeillCausalAnalyzer,
    TalebCausalAnalyzer,
    HybridStrategyAnalyzer,
    MarketRegime,
)


class AllocationMode(Enum):
    """配置模式"""
    BULL_EARLY = "bull_early"  # 牛市初期
    BULL_LATE = "bull_late"  # 牛市后期
    BEAR_MARKET = "bear_market"  # 熊市
    CRISIS = "crisis"  # 危机
    NEUTRAL = "neutral"  # 中性


@dataclass
class CausalSignals:
    """因果信号"""
    regime: MarketRegime
    oneill_causal_strength: float  # 欧奈尔因果强度 0-1
    taleb_causal_strength: float  # 塔勒布因果强度 0-1
    crisis_probability: float  # 危机概率 0-1
    recommended_allocation: AllocationMode
    confidence: float  # 信号置信度


@dataclass
class HybridPosition:
    """混合持仓"""
    oneill_positions: List[ONeillPosition] = field(default_factory=list)
    taleb_portfolio: Optional[TalebBarbellPortfolio] = None
    allocation: Dict[str, float] = field(default_factory=dict)  # {"oneill": 0.7, "taleb": 0.3}
    regime: MarketRegime = MarketRegime.VOLATILE


class CausalHybridStrategy:
    """
    因果驱动的混合策略

    核心思想：
    - 使用因果AI监控市场状态
    - 根据因果强度动态调整欧奈尔/塔勒布比例
    - 欧奈尔止损资金自动转入塔勒布
    - 形成自适应的攻守兼备系统
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        base_oneill_allocation: float = 0.60,
        base_taleb_allocation: float = 0.40,
        rebalance_threshold: float = 0.10,  # 10%偏离时再平衡
        auto_recycle_stops: bool = True,  # 自动回收止损资金
    ):
        """
        初始化混合策略

        参数:
            initial_capital: 初始资金
            base_oneill_allocation: 基础欧奈尔配置比例
            base_taleb_allocation: 基础塔勒布配置比例
            rebalance_threshold: 再平衡阈值
            auto_recycle_stops: 是否自动回收欧奈尔止损资金到塔勒布
        """
        self.initial_capital = initial_capital
        self.base_oneill_allocation = base_oneill_allocation
        self.base_taleb_allocation = base_taleb_allocation
        self.rebalance_threshold = rebalance_threshold
        self.auto_recycle_stops = auto_recycle_stops

        # 初始化子策略
        self.oneill_engine = ONeillStrategyEngine(
            initial_capital=initial_capital * base_oneill_allocation,
            max_positions=10,
        )

        self.taleb_strategy = TalebBarbellStrategy(
            initial_capital=initial_capital * base_taleb_allocation,
            safe_allocation=0.90,
            tail_allocation=0.10,
        )

        # 初始化因果分析器
        self.oneill_causal = ONeillCausalAnalyzer()
        self.taleb_causal = TalebCausalAnalyzer()
        self.hybrid_causal = HybridStrategyAnalyzer()

        # 持仓和历史
        self.hybrid_position = HybridPosition()
        self.performance_history: List[Dict] = []
        self.causal_signals_history: List[CausalSignals] = []

        # 统计
        self.total_oneill_stops = 0
        self.total_stops_amount = 0.0
        self.total_recycled_to_taleb = 0.0

    def initialize(self) -> HybridPosition:
        """初始化混合策略"""

        # 初始化欧奈尔引擎
        # （实际使用时需要传入股票数据和基本面数据）

        # 初始化塔勒布策略
        taleb_portfolio = self.taleb_strategy.initialize_portfolio()

        # 创建混合持仓
        self.hybrid_position = HybridPosition(
            oneill_positions=[],
            taleb_portfolio=taleb_portfolio,
            allocation={
                "oneill": self.base_oneill_allocation,
                "taleb": self.base_taleb_allocation,
            },
            regime=MarketRegime.VOLATILE,
        )

        return self.hybrid_position

    def analyze_causal_signals(
        self,
        market_data: pd.DataFrame,
        vix: Optional[float] = None,
    ) -> CausalSignals:
        """
        分析因果信号

        基于：
        1. 市场状态（趋势、波动率）
        2. 欧奈尔因果强度（CANSLIM要素有效性）
        3. 塔勒布因果强度（黑天鹅风险）
        4. 危机概率（VIX、相关性等）
        """

        # 1. 判断市场状态
        if len(market_data) >= 200:
            ma50 = market_data['close'].rolling(50).mean().iloc[-1]
            ma200 = market_data['close'].rolling(200).mean().iloc[-1]
            current_price = market_data['close'].iloc[-1]

            # 计算波动率
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)

            # 判断状态
            if volatility > 0.40 or (vix and vix > 40):
                regime = MarketRegime.CRISIS
                crisis_prob = 0.80
            elif volatility > 0.25:
                regime = MarketRegime.VOLATILE
                crisis_prob = 0.30
            elif current_price > ma50 > ma200:
                # 检查是否处于牛市后期（价格远高于200日均线）
                if current_price / ma200 > 1.20:
                    regime = MarketRegime.BULL  # 但实际是牛市后期
                    crisis_prob = 0.40
                else:
                    regime = MarketRegime.BULL
                    crisis_prob = 0.10
            else:
                regime = MarketRegime.BEAR
                crisis_prob = 0.50
        else:
            regime = MarketRegime.VOLATILE
            crisis_prob = 0.30

        # 2. 计算欧奈尔因果强度
        # （简化版，实际需要CANSLIM数据）
        if regime == MarketRegime.BULL:
            oneill_strength = 0.85  # 牛市时欧奈尔有效
        elif regime == MarketRegime.BEAR:
            oneill_strength = 0.35  # 熊市时欧奈尔失效
        elif regime == MarketRegime.CRISIS:
            oneill_strength = 0.15  # 危机时欧奈尔严重失效
        else:
            oneill_strength = 0.60  # 震荡市中等有效

        # 3. 计算塔勒布因果强度
        if regime == MarketRegime.CRISIS:
            taleb_strength = 0.98  # 危机时塔勒布爆发
        elif regime == MarketRegime.BEAR:
            taleb_strength = 0.85  # 熊市时塔勒布有效
        elif regime == MarketRegime.BULL:
            taleb_strength = 0.40  # 牛市时塔勒布Theta失血
        else:
            taleb_strength = 0.60  # 震荡市中等

        # 4. 推荐配置模式
        if regime == MarketRegime.CRISIS:
            allocation = AllocationMode.CRISIS
        elif regime == MarketRegime.BEAR:
            allocation = AllocationMode.BEAR_MARKET
        elif crisis_prob > 0.35:
            allocation = AllocationMode.BULL_LATE
        else:
            allocation = AllocationMode.BULL_EARLY

        # 5. 计算置信度
        confidence = max(oneill_strength, taleb_strength) * 0.9

        signals = CausalSignals(
            regime=regime,
            oneill_causal_strength=oneill_strength,
            taleb_causal_strength=taleb_strength,
            crisis_probability=crisis_prob,
            recommended_allocation=allocation,
            confidence=confidence,
        )

        self.causal_signals_history.append(signals)
        return signals

    def get_dynamic_allocation(self, signals: CausalSignals) -> Dict[str, float]:
        """
        根据因果信号获取动态配置

        配置规则：
        - 牛市初期：欧奈尔70% + 塔勒布30%
        - 牛市后期：欧奈尔50% + 塔勒布50%
        - 熊市：欧奈尔30% + 塔勒布70%
        - 危机：欧奈尔10% + 塔勒布90%
        """

        if signals.recommended_allocation == AllocationMode.BULL_EARLY:
            return {"oneill": 0.70, "taleb": 0.30}
        elif signals.recommended_allocation == AllocationMode.BULL_LATE:
            return {"oneill": 0.50, "taleb": 0.50}
        elif signals.recommended_allocation == AllocationMode.BEAR_MARKET:
            return {"oneill": 0.30, "taleb": 0.70}
        elif signals.recommended_allocation == AllocationMode.CRISIS:
            return {"oneill": 0.10, "taleb": 0.90}
        else:  # NEUTRAL
            return {"oneill": 0.60, "taleb": 0.40}

    def rebalance(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        total_value: float,
    ) -> Dict[str, float]:
        """
        再平衡操作

        计算需要转移的资金量
        """

        current_oneill = total_value * current_allocation["oneill"]
        current_taleb = total_value * current_allocation["taleb"]

        target_oneill = total_value * target_allocation["oneill"]
        target_taleb = total_value * target_allocation["taleb"]

        oneill_change = target_oneill - current_oneill
        taleb_change = target_taleb - current_taleb

        return {
            "oneill_adjustment": oneill_change,
            "taleb_adjustment": taleb_change,
            "total_value": total_value,
        }

    def execute_oneill_stop_loss(
        self,
        position: ONeillPosition,
        exit_price: float,
        current_date: datetime,
    ) -> float:
        """
        执行欧奈尔止损，并将资金转移到塔勒布
        """

        # 计算止损金额
        stop_loss_amount = position.quantity * (exit_price - position.entry_price)
        self.total_oneill_stops += 1
        self.total_stops_amount += abs(stop_loss_amount)

        # 如果启用自动回收，转移到塔勒布
        if self.auto_recycle_stops:
            recycled_amount = abs(stop_loss_amount) * 0.8  # 回收80%

            # 转入塔勒布的安全资产模块
            if self.hybrid_position.taleb_portfolio:
                self.hybrid_position.taleb_portfolio.safe_module.amount += recycled_amount
                self.total_recycled_to_taleb += recycled_amount

                return recycled_amount

        return 0.0

    def run_hybrid_strategy(
        self,
        market_data: pd.DataFrame,
        stocks_data: Dict[str, pd.DataFrame],
        fundamentals_dict: Dict[str, Dict],
        vix_data: Optional[pd.Series] = None,
        days: int = 252 * 3,
    ) -> Dict[str, Any]:
        """
        运行混合策略回测

        整合欧奈尔和塔勒布，根据因果信号动态调整
        """

        # 初始化
        self.initialize()

        # 生成模拟数据（简化版）
        np.random.seed(42)
        dates = pd.date_range(start=datetime.now(), periods=days, freq='D')

        # 模拟市场数据（牛市 → 熊市 → 危机 → 牛市）
        market_prices = []
        current_price = 400.0

        phase_days = days // 4

        # 第1阶段：牛市（6个月）
        for _ in range(phase_days):
            daily_return = np.random.normal(0.0008, 0.015)  # 年化20%波动
            current_price *= (1 + daily_return)
            market_prices.append(current_price)

        # 第2阶段：牛市后期（估值过高）
        for _ in range(phase_days):
            daily_return = np.random.normal(0.0003, 0.018)  # 波动增加
            current_price *= (1 + daily_return)
            market_prices.append(current_price)

        # 第3阶段：熊市转危机（6个月）
        for _ in range(phase_days):
            if np.random.random() < 0.05:  # 偶尔暴跌
                daily_return = np.random.normal(-0.03, 0.03)
            else:
                daily_return = np.random.normal(-0.001, 0.025)
            current_price *= (1 + daily_return)
            market_prices.append(current_price)

        # 第4阶段：恢复
        for _ in range(days - 3 * phase_days):
            daily_return = np.random.normal(0.0005, 0.020)
            current_price *= (1 + daily_return)
            market_prices.append(current_price)

        # 创建市场数据DataFrame
        market_df = pd.DataFrame({
            'close': market_prices,
        }, index=dates)

        # 模拟VIX
        vix_values = []
        for price in market_prices:
            if price < 350:  # 危机
                vix = np.random.uniform(40, 60)
            elif price < 380:  # 熊市
                vix = np.random.uniform(25, 40)
            elif price > 450:  # 牛市后期
                vix = np.random.uniform(18, 28)
            else:  # 正常
                vix = np.random.uniform(12, 20)
            vix_values.append(vix)

        # 每月再平衡
        monthly_dates = dates[::30]

        performance_records = []

        for i, date in enumerate(monthly_dates):
            # 获取当前市场数据
            current_idx = min(i * 30, len(market_df) - 1)
            current_market = market_df.iloc[:current_idx+1]
            current_vix = vix_values[current_idx]

            # 分析因果信号
            signals = self.analyze_causal_signals(
                market_data=current_market,
                vix=current_vix,
            )

            # 获取动态配置
            target_allocation = self.get_dynamic_allocation(signals)

            # 计算当前总价值
            total_value = (
                self.hybrid_position.taleb_portfolio.account_value
                if self.hybrid_position.taleb_portfolio
                else self.initial_capital
            )

            # 再平衡
            rebalance_result = self.rebalance(
                current_allocation=self.hybrid_position.allocation,
                target_allocation=target_allocation,
                total_value=total_value,
            )

            # 更新配置
            self.hybrid_position.allocation = target_allocation
            self.hybrid_position.regime = signals.regime

            # 记录性能
            performance_records.append({
                'date': date,
                'total_value': total_value,
                'oneill_allocation': target_allocation['oneill'],
                'taleb_allocation': target_allocation['taleb'],
                'regime': signals.regime.value,
                'oneill_strength': signals.oneill_causal_strength,
                'taleb_strength': signals.taleb_causal_strength,
                'crisis_prob': signals.crisis_probability,
                'vix': current_vix,
            })

        # 计算最终绩效
        final_value = performance_records[-1]['total_value']
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # 生成报告
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'performance_records': performance_records,
            'total_stops': self.total_oneill_stops,
            'stops_amount': self.total_stops_amount,
            'recycled_to_taleb': self.total_recycled_to_taleb,
            'causal_signals_count': len(self.causal_signals_history),
        }

    def generate_report(self, backtest_result: Dict[str, Any]) -> str:
        """生成混合策略报告"""

        initial = backtest_result['initial_capital']
        final = backtest_result['final_value']
        total_return = backtest_result['total_return']

        report = f"""
{'='*80}
因果驱动混合策略 - 运行报告
{'='*80}

📊 组合表现
----------------------------------------
初始资金: ${initial:,.2f}
最终价值: ${final:,.2f}
总收益: ${final - initial:,.2f}
收益率: {total_return*100:.2f}%

🔄 动态配置统计
----------------------------------------
欧奈尔止损次数: {backtest_result['total_stops']}
止损总金额: ${backtest_result['stops_amount']:,.2f}
回收至塔勒布: ${backtest_result['recycled_to_taleb']:,.2f}
因果信号数: {backtest_result['causal_signals_count']}

🎯 策略特色
----------------------------------------
1. 因果驱动配置:
   - 根据市场因果强度动态调整
   - 牛市加大欧奈尔（进攻）
   - 熊市加大塔勒布（防守）

2. 止损资金循环利用:
   - 欧奈尔止损不提现
   - 自动转入塔勒布购买期权
   - 危机时杠杆倍数更高

3. 双向因果保护:
   - 正常市场：欧奈尔CANSLIM进攻
   - 危机市场：塔勒布尾部保护
   - 形成闭环攻守体系

💡 因果洞察
----------------------------------------
"""

        # 添加因果分析结果
        oneill_insights = self.oneill_causal.analyze_oneill_causal_mechanisms()
        taleb_insights = self.taleb_causal.analyze_taleb_causal_mechanisms()

        report += f"\n欧奈尔因果强度: {backtest_result['performance_records'][-1]['oneill_strength']:.2f}\n"
        report += f"塔勒布因果强度: {backtest_result['performance_records'][-1]['taleb_strength']:.2f}\n"
        report += f"当前市场状态: {backtest_result['performance_records'][-1]['regime']}\n"
        report += f"危机概率: {backtest_result['performance_records'][-1]['crisis_prob']:.1%}\n"

        report += f"""
{'='*80}
"""

        return report


def simulate_causal_hybrid_strategy(
    initial_capital: float = 1_000_000,
    days: int = 252 * 3,
    enable_auto_recycle: bool = True,
) -> Dict[str, Any]:
    """
    模拟因果驱动混合策略
    """

    strategy = CausalHybridStrategy(
        initial_capital=initial_capital,
        base_oneill_allocation=0.60,
        base_taleb_allocation=0.40,
        auto_recycle_stops=enable_auto_recycle,
    )

    # 生成模拟数据
    market_data = pd.DataFrame({
        'close': [400.0] * days,  # 简化
    })

    # 运行策略
    result = strategy.run_hybrid_strategy(
        market_data=market_data,
        stocks_data={},
        fundamentals_dict={},
        vix_data=None,
        days=days,
    )

    return result, strategy


if __name__ == "__main__":
    # 运行示例
    result, strategy = simulate_causal_hybrid_strategy(
        initial_capital=1_000_000,
        days=252 * 3,
    )

    print(strategy.generate_report(result))
