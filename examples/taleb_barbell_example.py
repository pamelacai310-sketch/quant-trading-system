"""
塔勒布杠铃式尾部全天候量化模型 - 使用示例

演示如何使用塔勒布黑天鹅保护策略
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quant_trade_system.strategies.taleb_barbell import (
    TalebBarbellStrategy,
    simulate_taleb_barbell,
)
from quant_trade_system.strategies.tail_option_engine import (
    TailOptionEngine,
    OptionContract,
    OptionType,
)


def example_1_basic_strategy():
    """示例1: 基础策略初始化"""

    print("\n" + "="*80)
    print("示例1: 塔勒布杠铃策略初始化")
    print("="*80 + "\n")

    # 创建策略
    strategy = TalebBarbellStrategy(
        initial_capital=1_000_000,
        safe_allocation=0.90,  # 90%安全资产
        tail_allocation=0.10,  # 10%尾部期权
        monthly_budget_pct=0.004,  # 月度预算0.4%
        target_safe_yield=0.05,  # 5%目标收益
    )

    # 初始化组合
    portfolio = strategy.initialize_portfolio()

    print(f"✅ 组合初始化成功！\n")
    print(f"账户总值: ${portfolio.account_value:,.2f}")
    print(f"安全资产: ${portfolio.safe_module.amount:,.2f} ({portfolio.safe_allocation_pct*100:.0f}%)")
    print(f"尾部预算: ${portfolio.tail_module.monthly_budget:,.2f}/月")
    print(f"年度预算: ${portfolio.tail_module.monthly_budget*12:,.2f} ({portfolio.tail_module.annual_budget_pct*100:.2f}%)\n")


def example_2_purchase_tail_options():
    """示例2: 购买尾部看跌期权"""

    print("\n" + "="*80)
    print("示例2: 购买深度虚值看跌期权")
    print("="*80 + "\n")

    # 创建期权引擎
    engine = TailOptionEngine(
        target_delta_range=(-0.10, -0.05),
        min_dte=90,
        max_dte=180,
    )

    # 模拟期权链
    underlying_price = 400.0  # SPY价格
    strikes = [underlying_price * (1 - x/100) for x in range(5, 55, 5)]

    option_chain = pd.DataFrame({
        'type': ['put'] * len(strikes),
        'strike': strikes,
        'expiration': [datetime.now() + timedelta(days=d) for d in range(90, 181, 10)],
        'last': [underlying_price * 0.02] * len(strikes),
        'volume': [1000] * len(strikes),
        'iv': [0.20] * len(strikes),
        'delta': [-0.05 - x*0.01 for x in range(len(strikes))],
    })

    # 选择最优合约
    best_contract = engine.select_optimal_contract(
        symbol="SPY",
        underlying_price=underlying_price,
        option_chain=option_chain,
        current_date=datetime.now(),
    )

    print(f"🎯 最优期权合约:\n")
    print(f"  行权价: ${best_contract['strike']:.2f}")
    print(f"  Delta: {best_contract['delta']:.3f}")
    print(f"  DTE: {best_contract['dte']}天")
    print(f"  IV: {best_contract['iv']*100:.1f}%")
    print(f"  价格: ${best_contract['price']:.2f}\n")

    # 计算年化Theta成本
    from quant_trade_system.strategies.tail_option_engine import calculate_annual_theta_cost

    theta_cost = calculate_annual_theta_cost(
        option_price=best_contract['price'],
        days_to_expiration=best_contract['dte'],
        position_size=10,  # 10张合约
    )

    print(f"📉 Theta成本分析:\n")
    print(f"  每日每张: ${theta_cost['daily_cost_per_contract']:.2f}")
    print(f"  年度每张: ${theta_cost['annual_cost_per_contract']:.2f}")
    print(f"  年度总计: ${theta_cost['total_annual_cost']:,.2f}")
    print(f"  占保费比例: {theta_cost['annual_cost_pct_of_premium']:.1f}%\n")


def example_3_roll_conditions():
    """示例3: 展期条件检查"""

    print("\n" + "="*80)
    print("示例3: 期权展期机制")
    print("="*80 + "\n")

    engine = TailOptionEngine()

    # 场景1: 时间展期（DTE < 45天）
    option_near_expiry = OptionContract(
        symbol="SPY",
        option_type=OptionType.PUT,
        strike=360.0,
        expiration=datetime.now() + timedelta(days=30),
        delta=-0.06,
        gamma=0.01,
        vega=0.10,
        theta=-0.02,
        iv=0.20,
        underlying_price=400.0,
        position_size=10,
        entry_price=2.50,
        entry_date=datetime.now() - timedelta(days=60),
    )

    roll_instr = engine.check_roll_conditions(
        position=option_near_expiry,
        current_underlying_price=400.0,
    )

    print(f"场景1: 时间展期")
    print(f"  操作: {roll_instr.action}")
    print(f"  原因: {roll_instr.reason}\n")

    # 场景2: 现价展期（Delta绝对值太小）
    option_far_strike = OptionContract(
        symbol="SPY",
        option_type=OptionType.PUT,
        strike=300.0,
        expiration=datetime.now() + timedelta(days=120),
        delta=-0.01,  # Delta太小
        gamma=0.001,
        vega=0.05,
        theta=-0.01,
        iv=0.18,
        underlying_price=400.0,
        position_size=10,
        entry_price=0.50,
        entry_date=datetime.now() - timedelta(days=30),
    )

    roll_instr = engine.check_roll_conditions(
        position=option_far_strike,
        current_underlying_price=400.0,
    )

    print(f"场景2: 现价展期")
    print(f"  操作: {roll_instr.action}")
    print(f"  原因: {roll_instr.reason}\n")

    # 场景3: 危机止盈（Delta激增）
    option_in_the_money = OptionContract(
        symbol="SPY",
        option_type=OptionType.PUT,
        strike=360.0,
        expiration=datetime.now() + timedelta(days=90),
        delta=-0.60,  # Delta激增！
        gamma=0.03,
        vega=0.40,
        theta=-0.05,
        iv=0.45,
        underlying_price=350.0,  # 市场已下跌
        position_size=10,
        entry_price=2.50,
        entry_date=datetime.now() - timedelta(days=30),
    )

    roll_instr = engine.check_roll_conditions(
        position=option_in_the_money,
        current_underlying_price=350.0,
        current_vix=45.0,  # VIX飙升
    )

    print(f"场景3: 危机止盈 ⚠️")
    print(f"  操作: {roll_instr.action}")
    print(f"  原因: {roll_instr.reason}")
    print(f"  建议平仓: {roll_instr.percentage_to_close*100:.0f}%\n")


def example_4_full_simulation():
    """示例4: 完整3年模拟"""

    print("\n" + "="*80)
    print("示例4: 完整策略3年模拟")
    print("="*80 + "\n")

    print("正在模拟3年市场走势...")
    print("（包含随机黑天鹅事件）\n")

    # 运行模拟
    strategy = simulate_taleb_barbell(
        initial_capital=1_000_000,
        days=252 * 3,  # 3年
        safe_yield=0.05,
    )

    # 生成报告
    print(strategy.generate_report())

    # 分析收益分布
    print("📊 月度收益分布:\n")

    if len(strategy.performance_history) >= 60:  # 至少需要2个月数据
        monthly_returns = []
        for i in range(0, len(strategy.performance_history), 30):
            if i + 30 < len(strategy.performance_history):
                start_value = strategy.performance_history[i]['account_value']
                end_value = strategy.performance_history[i + 30]['account_value']
                monthly_return = (end_value - start_value) / start_value
                monthly_returns.append(monthly_return)

        if monthly_returns:
            monthly_returns = np.array(monthly_returns)

            print(f"  总月数: {len(monthly_returns)}")
            print(f"  盈利月数: {np.sum(monthly_returns > 0)}")
            print(f"  亏损月数: {np.sum(monthly_returns < 0)}")
            print(f"  平均月收益: {np.mean(monthly_returns)*100:.3f}%")
            print(f"  最佳月份: {np.max(monthly_returns)*100:.2f}%")
            print(f"  最差月份: {np.min(monthly_returns)*100:.2f}%")
            print(f"  收益标准差: {np.std(monthly_returns)*100:.3f}%\n")

            # 右偏分析
            if np.std(monthly_returns) > 0:
                skewness = ((monthly_returns - np.mean(monthly_returns))**3).mean() / (np.std(monthly_returns)**3)
                print(f"  偏度 (Skewness): {skewness:.2f}")
                print(f"  {'✅ 右偏分布（符合塔勒布理论）' if skewness > 0 else '⚠️ 左偏分布'}\n")
    else:
        print("  模拟期不足，无法计算月度收益分布\n")


def example_5_crisis_scenario():
    """示例5: 黑天鹅危机场景"""

    print("\n" + "="*80)
    print("示例5: 黑天鹅危机场景模拟")
    print("="*80 + "\n")

    # 创建策略
    strategy = TalebBarbellStrategy(
        initial_capital=1_000_000,
        safe_allocation=0.90,
        tail_allocation=0.10,
    )

    strategy.initialize_portfolio()

    print("初始状态:")
    print(f"  账户总值: ${strategy.portfolio.account_value:,.2f}")
    print(f"  安全资产: ${strategy.portfolio.safe_module.amount:,.2f}")
    print(f"  尾部预算: ${strategy.portfolio.tail_module.monthly_budget:,.2f}\n")

    # 模拟第1-3个月：平静期
    print("📅 第1-3个月: 市场平稳上涨\n")

    for month in range(1, 4):
        price = 400.0 * (1 + 0.01 * month)  # 每月上涨1%
        vix = 15.0

        option_chain = pd.DataFrame({
            'type': ['put'] * 10,
            'strike': [price * (1 - x/100) for x in range(5, 55, 5)],
            'expiration': [datetime.now() + timedelta(days=d) for d in range(90, 181, 10)],
            'last': [price * 0.02] * 10,
            'volume': [1000] * 10,
            'iv': [0.20] * 10,
            'delta': [-0.05 - x*0.01 for x in range(10)],
        })

        strategy.rebalance(
            current_date=datetime.now() + timedelta(days=30*month),
            underlying_price=price,
            vix=vix,
            option_chain=option_chain,
        )

        print(f"  第{month}月: 价格${price:.2f}, VIX{vix:.1f}, "
              f"账户${strategy.portfolio.account_value:,.2f} "
              f"({(strategy.portfolio.account_value/1_000_000 - 1)*100:+.2f}%)")

    print(f"\n  3个月Theta失血: ${strategy.option_engine.calculate_theta_bleed(strategy.portfolio.tail_module.options, days_passed=90):,.2f}\n")

    # 模拟第4个月：黑天鹅事件！
    print("🔥 第4个月: 黑天鹅事件！市场暴跌25%\n")

    crisis_price = 400.0 * 0.75  # 暴跌25%
    crisis_vix = 55.0  # VIX飙升

    option_chain = pd.DataFrame({
        'type': ['put'] * 10,
        'strike': [crisis_price * (1 - x/100) for x in range(5, 55, 5)],
        'expiration': [datetime.now() + timedelta(days=d) for d in range(90, 181, 10)],
        'last': [crisis_price * 0.10] * 10,  # 期权价格暴涨
        'volume': [5000] * 10,
        'iv': [0.55] * 10,
        'delta': [-0.05 - x*0.01 for x in range(10)],
    })

    strategy.rebalance(
        current_date=datetime.now() + timedelta(days=120),
        underlying_price=crisis_price,
        vix=crisis_vix,
        option_chain=option_chain,
    )

    print(f"  第4月: 价格${crisis_price:.2f} (-25%), VIX{crisis_vix:.1f}")
    print(f"  账户价值: ${strategy.portfolio.account_value:,.2f}")
    print(f"  总收益: ${(strategy.portfolio.account_value/1_000_000 - 1)*100:+.2f}%")
    print(f"  危机事件: {len(strategy.crisis_events)}次\n")

    if strategy.crisis_events:
        for event in strategy.crisis_events:
            print(f"  🎉 {event['description']}\n")

    # 最终总结
    final_return = (strategy.portfolio.account_value / 1_000_000 - 1) * 100

    print("💡 策略总结:")
    print(f"  4个月总收益: {final_return:+.2f}%")
    print(f"  安全模块贡献: {(strategy.portfolio.safe_module.amount/1_000_000 - 0.9)*100:.2f}%")
    print(f"  尾部模块贡献: {final_return - (strategy.portfolio.safe_module.amount/1_000_000 - 0.9)*100:.2f}%")

    if final_return > 0:
        print(f"  ✅ 策略成功: 在股灾中盈利{final_return:.2f}%")
    else:
        print(f"  ⚠️  策略亏损: {final_return:.2f}%（但优于大盘-25%）")


def example_6_comparison_with_traditional():
    """示例6: 与传统策略对比"""

    print("\n" + "="*80)
    print("示例6: 塔勒布策略 vs 传统60/40组合")
    print("="*80 + "\n")

    # 塔勒布策略
    taleb_strategy = simulate_taleb_barbell(
        initial_capital=1_000_000,
        days=252 * 5,  # 5年
        safe_yield=0.05,
    )

    taleb_final = taleb_strategy.portfolio.account_value
    taleb_return = (taleb_final / 1_000_000 - 1) * 100

    # 传统60/40组合（模拟）
    np.random.seed(123)
    traditional_value = 1_000_000
    traditional_returns = []

    for day in range(252 * 5):
        # 60%股票 + 40%债券
        daily_return = (
            0.6 * np.random.normal(0.0003, 0.012) +  # 股票
            0.4 * np.random.normal(0.0002, 0.003)    # 债券
        )
        traditional_value *= (1 + daily_return)
        traditional_returns.append(daily_return)

    traditional_final = traditional_value
    traditional_return = (traditional_final / 1_000_000 - 1) * 100

    # 最大回撤对比
    def calculate_max_drawdown(returns):
        values = [1_000_000]
        for r in returns:
            values.append(values[-1] * (1 + r))

        peak = values[0]
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    taleb_dd = taleb_strategy.calculate_performance().max_drawdown * 100
    traditional_dd = calculate_max_drawdown(traditional_returns) * 100

    # 夏普对比
    taleb_sharpe = taleb_strategy.calculate_performance().sharpe_ratio

    traditional_returns_arr = np.array(traditional_returns)
    traditional_sharpe = (
        (np.mean(traditional_returns_arr) - 0.02/252) /
        np.std(traditional_returns_arr)
    ) * np.sqrt(252)

    print("📊 5年表现对比:\n")
    print(f"{'指标':<20} {'塔勒布杠铃':>15} {'传统60/40':>15} {'优势':>10}")
    print("-" * 65)
    print(f"{'总收益率':<20} {taleb_return:>14.2f}% {traditional_return:>14.2f}% "
          f"{'✅' if taleb_return > traditional_return else '❌':>10}")
    print(f"{'最大回撤':<20} {taleb_dd:>14.2f}% {traditional_dd:>14.2f}% "
          f"{'✅' if taleb_dd < traditional_dd else '❌':>10}")
    print(f"{'夏普比率':<20} {taleb_sharpe:>14.2f} {traditional_sharpe:>14.2f} "
          f"{'✅' if taleb_sharpe > traditional_sharpe else '❌':>10}")
    print(f"{'危机保护':<20} {'内置':>15} {'无':>15} {'✅':>10}")
    print(f"{'胜率（月度）':<20} {taleb_strategy.calculate_performance().win_rate*100:>14.1f}% {'N/A':>15} {'-':>10}\n")

    print("💡 结论:")
    print(f"  塔勒布策略在{'高收益' if taleb_return > traditional_return else '低回撤'}方面{' ' if taleb_return > traditional_return else '不'}占优")
    print(f"  最大回撤降低: {traditional_dd - tale_dd:.2f}个百分点")
    print(f"  危机时刻表现: 卓越（尾部期权爆发）\n")


def main():
    """主函数"""
    print("\n" + "="*80)
    print(" " * 15 + "塔勒布杠铃式尾部全天候量化模型 - 使用示例")
    print("="*80)

    # 运行所有示例
    example_1_basic_strategy()
    example_2_purchase_tail_options()
    example_3_roll_conditions()
    example_4_full_simulation()
    example_5_crisis_scenario()
    example_6_comparison_with_traditional()

    print("\n" + "="*80)
    print(" " * 25 + "所有示例运行完成！")
    print("="*80 + "\n")

    print("📚 更多信息:")
    print("  - 塔勒布《黑天鹅》")
    print("  - 塔勒布《反脆弱》")
    print("  - Universa Investments (Mark Spitznagel)")
    print("  - Empirica Capital")
    print()


if __name__ == "__main__":
    main()
