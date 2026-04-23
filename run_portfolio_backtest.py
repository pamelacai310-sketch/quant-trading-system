#!/usr/bin/env python3
"""
投资组合回测脚本

回测4个投资组合方案，对比收益表现
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from quant_trade_system.portfolio_backtest import (
    FundAllocation,
    FundDataSimulator,
    Portfolio,
    PortfolioBacktester,
    format_comparison,
)


def create_portfolios() -> list:
    """创建4个投资组合方案"""
    portfolios = []

    # 方案一：均衡分散
    portfolios.append(Portfolio(
        name="方案一（均衡分散）",
        allocations=[
            FundAllocation("968064", "惠理高息股", 200, 200/600),
            FundAllocation("968072", "摩根亚洲增长", 200, 200/600),
            FundAllocation("968157", "东亚联丰环球股票", 200, 200/600),
        ],
        total_amount=600,
    ))

    # 方案二：增长导向
    portfolios.append(Portfolio(
        name="方案二（增长导向）",
        allocations=[
            FundAllocation("968157", "东亚联丰环球股票", 300, 300/600),
            FundAllocation("968078", "东方汇理创新动力基金", 300, 300/600),
        ],
        total_amount=600,
    ))

    # 方案三：创新+平衡
    portfolios.append(Portfolio(
        name="方案三（创新+平衡）",
        allocations=[
            FundAllocation("968078", "东方汇理创新动力基金", 200, 200/600),
            FundAllocation("968072", "摩根亚洲增长", 200, 200/600),
            FundAllocation("968064", "惠理高息股", 200, 200/600),
        ],
        total_amount=600,
    ))

    # 方案四：多元分散
    portfolios.append(Portfolio(
        name="方案四（多元分散）",
        allocations=[
            FundAllocation("018229", "易方达优质企业", 130, 130/600),
            FundAllocation("968049", "摩根亚洲股息", 50, 50/600),
            FundAllocation("968072", "摩根亚洲增长", 100, 100/600),
            FundAllocation("968064", "惠理高息股", 100, 100/600),
            FundAllocation("968078", "东方汇理创新动力基金", 220, 220/600),
        ],
        total_amount=600,
    ))

    return portfolios


def main():
    """主函数"""
    print("=" * 80)
    print("投资组合回测分析（2年期）")
    print("=" * 80)
    print()

    # 创建模拟器
    simulator = FundDataSimulator(seed=42)

    # 显示基金信息
    print("基金基础信息：")
    print("-" * 80)
    fund_codes = ["968064", "968072", "968157", "968078", "968049", "018229"]
    for code in fund_codes:
        info = simulator.get_fund_info(code)
        if info:
            print(f"{code} {info['name']:20s} "
                  f"年化收益: {info['annual_return']:>5.1%}  "
                  f"波动率: {info['volatility']:>5.1%}  "
                  f"股息率: {info['dividend_yield']:>5.1%}")
    print()

    # 创建投资组合
    portfolios = create_portfolios()

    # 显示投资组合配置
    print("投资组合配置：")
    print("-" * 80)
    for portfolio in portfolios:
        print(f"\n{portfolio.name}（总计：{portfolio.total_amount}万）")
        for alloc in portfolio.allocations:
            fund_info = simulator.get_fund_info(alloc.code)
            print(f"  {alloc.code} {fund_info['name']:25s} "
                  f"¥{alloc.amount:>4.0f}万 ({alloc.weight:>5.1%})")
    print("\n" + "-" * 80)
    print()

    # 创建回测器
    backtester = PortfolioBacktester(simulator)

    # 运行回测
    print("运行回测...")
    print()

    comparison = backtester.compare_portfolios(portfolios, months=24)

    # 显示对比结果
    print(format_comparison(comparison["results"]))

    # 显示详细排名
    print("📊 详细排名：")
    print("-" * 80)

    for i, result in enumerate(comparison["results"], 1):
        print(f"\n🏆  第{i}名：{result.portfolio_name}")
        print(f"   年化收益率：{result.annualized_return:.2%}")
        print(f"   总收益率：{result.total_return:.2%}")
        print(f"   最终金额：¥{result.final_amount:.2f}万")
        print(f"   夏普比率：{result.sharpe_ratio:.2f}")
        print(f"   最大回撤：{result.max_drawdown:.2%}")
        print(f"   波动率：{result.volatility:.2%}")

    # 显示总结
    print("\n" + "=" * 80)
    print("📈 总结分析")
    print("=" * 80)

    summary = comparison["summary"]

    print(f"\n最佳方案：{summary['best_portfolio']['name']}")
    print(f"  年化收益率：{summary['best_portfolio']['annual_return']:.2%}")
    print(f"  夏普比率：{summary['best_portfolio']['sharpe_ratio']:.2f}")
    print(f"  最大回撤：{summary['best_portfolio']['max_drawdown']:.2%}")

    print(f"\n最低波动：{backtester.backtest_portfolio([p for p in portfolios if p.name == summary['lowest_volatility']][0], 24).volatility:.2%}")
    print(f"最高夏普：{summary['highest_sharpe']}")

    print(f"\n平均年化收益：{summary['avg_annual_return']:.2%}")

    # 投资建议
    print("\n" + "=" * 80)
    print("💡 投资建议")
    print("=" * 80)

    best = comparison["results"][0]
    worst = comparison["results"][-1]

    print(f"\n✅ 推荐方案：{best.portfolio_name}")
    print(f"   - 年化收益率最高（{best.annualized_return:.2%}）")
    print(f"   - 风险调整后收益最优（夏普比率：{best.sharpe_ratio:.2f}）")

    print(f"\n⚠️  风险提示：")
    print(f"   - {worst.portfolio_name}的回撤风险较高（{worst.max_drawdown:.2%}）")
    print(f"   - 所有方案均基于历史数据模拟，实际收益可能存在差异")
    print(f"   - 建议根据个人风险承受能力和投资目标选择")

    # 保存结果到JSON
    output_file = project_root / "state" / "backtest_results.json"
    output_file.parent.mkdir(exist_ok=True)

    results_data = {
        "backtest_date": str(Path(__file__).stat().st_mtime),
        "months": 24,
        "initial_amount": 600,
        "portfolios": [
            {
                "name": r.portfolio_name,
                "final_amount": r.final_amount,
                "total_return": r.total_return,
                "annualized_return": r.annualized_return,
                "max_drawdown": r.max_drawdown,
                "sharpe_ratio": r.sharpe_ratio,
                "volatility": r.volatility,
            }
            for r in comparison["results"]
        ],
        "best_portfolio": summary["best_portfolio"]["name"],
        "best_return": summary["best_portfolio"]["annual_return"],
    }

    import json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 回测结果已保存到：{output_file}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
