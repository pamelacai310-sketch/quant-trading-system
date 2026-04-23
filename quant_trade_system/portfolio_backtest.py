"""
投资组合回测模块

用于对比多个基金组合方案的历史收益表现
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FundAllocation:
    """基金配置"""
    code: str  # 基金代码
    name: str  # 基金名称
    amount: float  # 投资金额（港币）
    weight: float  # 权重


@dataclass
class Portfolio:
    """投资组合"""
    name: str  # 组合名称
    allocations: List[FundAllocation]  # 基金配置
    total_amount: float  # 总金额


@dataclass
class BacktestResult:
    """回测结果"""
    portfolio_name: str
    initial_amount: float
    final_amount: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    monthly_returns: List[float]
    equity_curve: List[Dict[str, Any]]


class FundDataSimulator:
    """基金数据模拟器"""

    # 基金基础年化收益率（基于历史数据的合理估计）
    FUND_BASE_RETURNS = {
        "968064": {  # 惠理高息股
            "name": "惠理高息股",
            "annual_return": 0.085,  # 8.5% 年化
            "volatility": 0.15,  # 15% 波动率
            "dividend_yield": 0.05,  # 5% 股息率
        },
        "968072": {  # 摩根亚洲增长
            "name": "摩根亚洲增长",
            "annual_return": 0.105,  # 10.5% 年化
            "volatility": 0.18,  # 18% 波动率
            "dividend_yield": 0.02,  # 2% 股息率
        },
        "968157": {  # 东亚联丰环球股票
            "name": "东亚联丰环球股票",
            "annual_return": 0.095,  # 9.5% 年化
            "volatility": 0.16,  # 16% 波动率
            "dividend_yield": 0.025,  # 2.5% 股息率
        },
        "968078": {  # 东方汇理创新动力基金
            "name": "东方汇理创新动力基金",
            "annual_return": 0.125,  # 12.5% 年化
            "volatility": 0.22,  # 22% 波动率
            "dividend_yield": 0.01,  # 1% 股息率
        },
        "968049": {  # 摩根亚洲股息
            "name": "摩根亚洲股息",
            "annual_return": 0.08,  # 8% 年化
            "volatility": 0.12,  # 12% 波动率
            "dividend_yield": 0.055,  # 5.5% 股息率
        },
        "018229": {  # 易方达优质企业
            "name": "易方达优质企业",
            "annual_return": 0.11,  # 11% 年化
            "volatility": 0.20,  # 20% 波动率
            "dividend_yield": 0.015,  # 1.5% 股息率
        },
    }

    def __init__(self, seed: int = 42):
        """初始化模拟器"""
        self.seed = seed
        np.random.seed(seed)

    def generate_fund_returns(
        self,
        fund_code: str,
        months: int = 24,
    ) -> pd.Series:
        """
        生成基金月度收益率

        Args:
            fund_code: 基金代码
            months: 月数

        Returns:
            月度收益率序列
        """
        if fund_code not in self.FUND_BASE_RETURNS:
            raise ValueError(f"Unknown fund code: {fund_code}")

        fund_info = self.FUND_BASE_RETURNS[fund_code]
        annual_return = fund_info["annual_return"]
        volatility = fund_info["volatility"]
        dividend_yield = fund_info["dividend_yield"]

        # 月度参数
        monthly_return = (1 + annual_return) ** (1/12) - 1
        monthly_volatility = volatility / np.sqrt(12)
        monthly_dividend = dividend_yield / 12

        # 生成收益率（带正态分布的随机波动）
        monthly_returns = []
        for i in range(months):
            # 市场环境因子（模拟牛熊市）
            market_cycle = np.sin(2 * np.pi * i / 12) * 0.02  # 年度周期

            # 随机冲击
            shock = np.random.normal(0, monthly_volatility)

            # 月度收益率
            monthly_r = monthly_return + market_cycle + shock

            # 加入股息收益
            monthly_r += monthly_dividend

            monthly_returns.append(monthly_r)

        return pd.Series(monthly_returns)

    def get_fund_info(self, fund_code: str) -> Dict[str, Any]:
        """获取基金信息"""
        return self.FUND_BASE_RETURNS.get(fund_code, {})


class PortfolioBacktester:
    """投资组合回测器"""

    def __init__(self, simulator: FundDataSimulator):
        """
        初始化回测器

        Args:
            simulator: 基金数据模拟器
        """
        self.simulator = simulator

    def backtest_portfolio(
        self,
        portfolio: Portfolio,
        months: int = 24,
    ) -> BacktestResult:
        """
        回测投资组合

        Args:
            portfolio: 投资组合
            months: 回测月数

        Returns:
            回测结果
        """
        # 计算组合权重
        total_amount = portfolio.total_amount
        weights = {
            alloc.code: alloc.amount / total_amount
            for alloc in portfolio.allocations
        }

        # 生成各基金收益率
        fund_returns = {}
        for alloc in portfolio.allocations:
            fund_returns[alloc.code] = self.simulator.generate_fund_returns(
                alloc.code, months
            )

        # 计算组合月度收益率
        portfolio_returns = []
        for month in range(months):
            monthly_return = sum(
                weights[code] * fund_returns[code].iloc[month]
                for code in weights
            )
            portfolio_returns.append(monthly_return)

        # 计算权益曲线
        equity_curve = []
        cumulative_value = total_amount
        for i, monthly_return in enumerate(portfolio_returns):
            cumulative_value *= (1 + monthly_return)
            equity_curve.append({
                "month": i + 1,
                "value": cumulative_value,
                "return": monthly_return,
                "cumulative_return": (cumulative_value / total_amount) - 1,
            })

        # 计算指标
        final_amount = cumulative_value
        total_return = (final_amount / total_amount) - 1
        annualized_return = (1 + total_return) ** (12 / months) - 1

        # 最大回撤
        cumulative_values = [e["value"] for e in equity_curve]
        peak = cumulative_values[0]
        max_drawdown = 0
        for value in cumulative_values:
            if value > peak:
                peak = value
            drawdown = (value - peak) / peak
            if drawdown < max_drawdown:
                max_drawdown = drawdown

        # 夏普比率（假设无风险利率为2%）
        returns_series = pd.Series(portfolio_returns)
        excess_returns = returns_series - (0.02 / 12)
        sharpe_ratio = np.sqrt(12) * excess_returns.mean() / returns_series.std() if returns_series.std() > 0 else 0

        # 波动率（年化）
        volatility = returns_series.std() * np.sqrt(12)

        return BacktestResult(
            portfolio_name=portfolio.name,
            initial_amount=total_amount,
            final_amount=final_amount,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            monthly_returns=portfolio_returns,
            equity_curve=equity_curve,
        )

    def compare_portfolios(
        self,
        portfolios: List[Portfolio],
        months: int = 24,
    ) -> Dict[str, Any]:
        """
        对比多个投资组合

        Args:
            portfolios: 投资组合列表
            months: 回测月数

        Returns:
            对比结果
        """
        results = []
        for portfolio in portfolios:
            result = self.backtest_portfolio(portfolio, months)
            results.append(result)

        # 排序（按年化收益率）
        results.sort(key=lambda x: x.annualized_return, reverse=True)

        return {
            "results": results,
            "months": months,
            "best_portfolio": results[0].portfolio_name if results else None,
            "best_return": results[0].annualized_return if results else 0,
            "summary": self._generate_summary(results),
        }

    def _generate_summary(
        self,
        results: List[BacktestResult],
    ) -> Dict[str, Any]:
        """生成总结报告"""
        if not results:
            return {}

        return {
            "total_portfolios": len(results),
            "avg_annual_return": np.mean([r.annualized_return for r in results]),
            "best_portfolio": {
                "name": results[0].portfolio_name,
                "annual_return": results[0].annualized_return,
                "total_return": results[0].total_return,
                "sharpe_ratio": results[0].sharpe_ratio,
                "max_drawdown": results[0].max_drawdown,
            },
            "worst_portfolio": {
                "name": results[-1].portfolio_name,
                "annual_return": results[-1].annualized_return,
                "total_return": results[-1].total_return,
                "sharpe_ratio": results[-1].sharpe_ratio,
                "max_drawdown": results[-1].max_drawdown,
            },
            "lowest_volatility": min(results, key=lambda x: x.volatility).portfolio_name,
            "highest_sharpe": max(results, key=lambda x: x.sharpe_ratio).portfolio_name,
        }


def format_result(result: BacktestResult) -> str:
    """格式化回测结果"""
    return f"""
{result.portfolio_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
初始金额: ¥{result.initial_amount:,.0f} 万
最终金额: ¥{result.final_amount:,.0f} 万
总收益率: {result.total_return:.2%}
年化收益率: {result.annualized_return:.2%}
最大回撤: {result.max_drawdown:.2%}
夏普比率: {result.sharpe_ratio:.2f}
波动率: {result.volatility:.2%}
"""


def format_comparison(results: List[BacktestResult]) -> str:
    """格式化对比结果"""
    output = "\n"
    output += "=" * 80 + "\n"
    output += "投资组合回测对比结果\n"
    output += "=" * 80 + "\n\n"

    for i, result in enumerate(results, 1):
        output += f"【方案{i}】{result.portfolio_name}\n"
        output += f"  年化收益率: {result.annualized_return:>6.2%}  "
        output += f"总收益: {result.total_return:>6.2%}  "
        output += f"夏普比率: {result.sharpe_ratio:>5.2f}\n"
        output += f"  最大回撤: {result.max_drawdown:>6.2%}  "
        output += f"波动率: {result.volatility:>6.2%}\n"
        output += "\n"

    output += "=" * 80 + "\n"
    return output
