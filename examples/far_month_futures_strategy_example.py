"""
远期期货合约交易策略示例

演示如何使用远期期货合约策略：
1. 只做远期合约，选择波动幅度高于主力合约的
2. 期货仓位风险度<50%（保证金占用<50%）
3. 每日根据市场多空情绪选择方向
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from quant_trade_system.strategies import (
    FarMonthFuturesStrategy,
    FuturesContract,
    MarketSentiment,
    PositionSide,
)


def example_1_select_far_month_contract():
    """示例1: 选择远期合约"""
    print("\n" + "="*80)
    print("示例1: 选择远期合约（波动高于主力，至少+2个月）")
    print("="*80 + "\n")

    strategy = FarMonthFuturesStrategy()
    current_date = datetime(2026, 5, 4)

    # 以螺纹钢为例
    underlying = 'RB'  # 螺纹钢

    print(f"标的物: {underlying}（螺纹钢）")
    print(f"\n获取所有可用合约:")

    # 获取所有合约
    all_contracts = strategy.get_available_contracts(underlying, current_date)

    print(f"\n{'合约':<15} {'交割月':<10} {'是否主力':<10} {'波动率':<10} {'距交割':<10}")
    print("-" * 70)

    for contract in all_contracts[:6]:
        main_str = "是" if contract.is_main else "否"
        months_str = f"+{contract.months_to_delivery}个月"
        print(f"{contract.symbol:<15} {contract.delivery_date:<10} {main_str:<10} "
              f"{contract.volatility*100:.1f}%       {months_str:<10}")

    # 选择远期合约
    print(f"\n选择远期合约:")
    far_contract = strategy.select_far_month_contract(underlying, current_date)

    if far_contract:
        print(f"  选中合约: {far_contract.symbol}")
        print(f"  交割日期: {far_contract.delivery_date}")
        print(f"  距交割: +{far_contract.months_to_delivery}个月")
        print(f"  波动率: {far_contract.volatility*100:.1f}%")
        print(f"  当前价: ${far_contract.current_price:.2f}")
        print(f"  保证金比例: {far_contract.margin_rate*100:.0f}%")


def example_2_margin_and_reserve():
    """示例2: 保证金和预留资金"""
    print("\n" + "="*80)
    print("示例2: 期货仓位风险管理（保证金<50%，预留1倍资金）")
    print("="*80 + "\n")

    strategy = FarMonthFuturesStrategy(initial_capital=1_000_000)

    # 创建一个模拟合约
    contract = FuturesContract(
        symbol='RB2610',
        name='螺纹钢',
        underlying='RB',
        delivery_month=10,
        delivery_year=2026,
        is_main=False,
        volatility=0.25,
        current_price=4000,
        margin_rate=0.15,
    )

    print(f"合约信息:")
    print(f"  合约: {contract.symbol}")
    print(f"  当前价: ${contract.current_price}")
    print(f"  保证金比例: {contract.margin_rate*100:.0f}%")

    # 计算不同合约数量的资金需求
    print(f"\n不同合约数量的资金需求:")
    print(f"\n{'合约数':<10} {'合约价值':<15} {'保证金':<15} {'预留资金':<15} {'风险度':<10}")
    print("-" * 80)

    for contracts_count in [1, 5, 10, 20, 50]:
        contract_value = contract.current_price * contracts_count
        margin, reserve = strategy.calculate_margin_requirement(contract, contracts_count)
        risk_level = margin / (margin + reserve) if (margin + reserve) > 0 else 0

        print(f"{contracts_count:<10} ${contract_value:>13,.0f} ${margin:>13,.0f} ${reserve:>13,.0f} {risk_level*100:>8.1f}%")

    print(f"\n关键规则:")
    print(f"  1. 保证金占用 < 50%")
    print(f"  2. 预留资金 >= 保证金（1倍）")
    print(f"  3. 风险度 = 保证金 / (保证金 + 预留资金)")


def example_3_market_sentiment():
    """示例3: 市场多空情绪分析"""
    print("\n" + "="*80)
    print("示例3: 每日市场多空情绪分析")
    print("="*80 + "\n")

    strategy = FarMonthFuturesStrategy()
    current_date = datetime(2026, 5, 4)

    # 生成模拟市场数据
    market_data = generate_mock_futures_data()

    # 分析市场情绪
    sentiment = strategy.analyze_market_sentiment(market_data, current_date)

    print(f"市场情绪分析 ({current_date.strftime('%Y-%m-%d')}):")
    print(f"\n  基础数据:")
    print(f"    总品种数: {sentiment.total_symbols}")
    print(f"    上涨品种: {sentiment.up_symbols}")
    print(f"    下跌品种: {sentiment.down_symbols}")
    print(f"    上涨占比: {sentiment.up_ratio*100:.1f}%")

    print(f"\n  极端数据:")
    print(f"    前20涨幅平均: {sentiment.top20_avg_gain*100:+.2f}%")
    print(f"    前20跌幅平均: {sentiment.top20_avg_loss*100:+.2f}%")

    print(f"\n  情绪判断:")
    bias_str = "多头" if sentiment.sentiment_bias == 'long' else \
              "空头" if sentiment.sentiment_bias == 'short' else "中性"
    print(f"    市场情绪: {bias_str}")
    print(f"    信心度: {sentiment.confidence:.2f}")

    print(f"\n  决策规则:")
    print(f"    规则1: 上涨品种 > 60% → 做多")
    print(f"    规则2: 前20涨幅平均 > 前20跌幅平均 → 做多")
    print(f"    当前状态: {'✅ 触发做多' if sentiment.sentiment_bias == 'long' else '✅ 触发做空' if sentiment.sentiment_bias == 'short' else '⚠️ 中性观望'}")

    # 选择持仓方向
    side = strategy.select_position_side(sentiment)
    side_str = "做多" if side == PositionSide.LONG else "做空"
    print(f"\n  开仓方向: {side_str}")


def example_4_complete_workflow():
    """示例4: 完整工作流程"""
    print("\n" + "="*80)
    print("示例4: 完整交易工作流程")
    print("="*80 + "\n")

    strategy = FarMonthFuturesStrategy(initial_capital=1_000_000)
    current_date = datetime(2026, 5, 4)

    print(f"交易日期: {current_date.strftime('%Y-%m-%d %A')}")

    # 步骤1: 扫描远期合约
    print(f"\n步骤1: 扫描远期合约")
    suitable_contracts = strategy.scan_contracts(current_date)

    print(f"  找到 {len(suitable_contracts)} 个合适的远期合约")
    for contract in suitable_contracts[:3]:
        print(f"    {contract.symbol}: {contract.name}, "
              f"交割{contract.delivery_date}, 波动{contract.volatility*100:.1f}%")

    # 步骤2: 分析市场情绪
    print(f"\n步骤2: 分析市场情绪")
    market_data = generate_mock_futures_data()
    sentiment = strategy.analyze_market_sentiment(market_data, current_date)

    bias_str = "多头" if sentiment.sentiment_bias == 'long' else \
              "空头" if sentiment.sentiment_bias == 'short' else "中性"
    print(f"  市场情绪: {bias_str} (信心度: {sentiment.confidence:.2f})")

    # 步骤3: 选择方向和开仓
    if sentiment.confidence > 0.65 and suitable_contracts:
        side = strategy.select_position_side(sentiment)
        best_contract = suitable_contracts[0]

        position = strategy.enter_position(
            best_contract, side, current_date, strategy.initial_capital
        )

        print(f"\n步骤3: 开仓")
        print(f"  合约: {position.contract.symbol}")
        print(f"  方向: {'做多' if position.side == PositionSide.LONG else '做空'}")
        print(f"  合约数: {position.contracts}")
        print(f"  保证金: ${position.margin_used:,.0f}")
        print(f"  预留资金: ${position.reserve_capital:,.0f}")
        print(f"  风险度: {position.risk_level*100:.1f}%")
        print(f"  止损: -{position.stop_loss*100:.0f}%")
        print(f"  止盈: +{position.take_profit*100:.0f}%")

        # 检查是否符合规则
        print(f"\n规则检查:")
        print(f"  ✅ 远期合约: +{position.contract.months_to_delivery}个月 (≥2个月)")
        print(f"  ✅ 风险度: {position.risk_level*100:.1f}% (<50%)")
        print(f"  ✅ 预留资金: ${position.reserve_capital:,.0f} (≥保证金 ${position.margin_used:,.0f})")
        print(f"  ✅ 市场情绪: {bias_str} (信心度: {sentiment.confidence:.2f})")

        # 步骤4: 模拟平仓
        print(f"\n步骤4: 模拟持仓3天后平仓")

        for day in range(1, 4):
            future_date = current_date + timedelta(days=day)
            np.random.seed(hash(future_date.strftime('%Y-%m-%d')))

            # 模拟价格变化
            price_change = np.random.normal(0.001, 0.02)
            current_price = position.entry_price * (1 + price_change)

            should_exit, reason = strategy.should_exit_position(position, current_price)

            if should_exit:
                strategy.exit_position(position, future_date, current_price, reason)
                print(f"  第{day}天: {reason}")
                print(f"    平仓价: ${current_price:.2f}")
                print(f"    盈亏: ${position.pnl:+,.2f} ({position.pnl_pct:+.2f}%)")
                break
        else:
            # 3天后强制平仓
            future_date = current_date + timedelta(days=3)
            current_price = position.entry_price * 1.04  # 假设涨4%
            strategy.exit_position(position, future_date, current_price, "达到最大持仓天数")
            print(f"  第3天: 达到最大持仓天数，平仓")
            print(f"    平仓价: ${current_price:.2f}")
            print(f"    盈亏: ${position.pnl:+,.2f} ({position.pnl_pct:+.2f}%)")


def example_5_contracts_comparison():
    """示例5: 主力vs远期合约对比"""
    print("\n" + "="*80)
    print("示例5: 主力合约 vs 远期合约对比")
    print("="*80 + "\n")

    strategy = FarMonthFuturesStrategy()
    current_date = datetime(2026, 5, 4)

    underlying = 'RB'  # 螺纹钢

    # 获取所有合约
    all_contracts = strategy.get_available_contracts(underlying, current_date)

    # 分为主力和远期
    main_contract = next((c for c in all_contracts if c.is_main), None)
    far_contracts = [c for c in all_contracts if not c.is_main and c.months_to_delivery >= 2]

    print(f"螺纹钢合约对比:")
    print(f"\n{'合约':<15} {'类型':<15} {'交割':<12} {'波动率':<10} {'成交量':<12}")
    print("-" * 80)

    if main_contract:
        print(f"{main_contract.symbol:<15} {'主力合约':<15} {main_contract.delivery_date:<12} "
              f"{main_contract.volatility*100:.1f}%       {main_contract.volume:>10,.0f}")

    for contract in far_contracts[:5]:
        contract_type = f"+{contract.months_to_delivery}个月"
        vol_diff = contract.volatility - main_contract.volatility if main_contract else 0
        vol_str = f"{contract.volatility*100:.1f}% ({vol_diff*100:+.1f}%)"

        print(f"{contract.symbol:<15} {contract_type:<15} {contract.delivery_date:<12} "
              f"{vol_str:<10} {contract.volume:>10,.0f}")

    print(f"\n远期合约选择规则:")
    print(f"  1. 必须是远期合约（≥+2个月）")
    print(f"  2. 波动率 > 主力合约波动率")
    print(f"  3. 在符合条件中，选择波动率最大的")

    # 选择最佳远期合约
    best_far = strategy.select_far_month_contract(underlying, current_date)

    if best_far:
        vol_vs_main = best_far.volatility - main_contract.volatility if main_contract else 0
        print(f"\n最佳远期合约:")
        print(f"  合约: {best_far.symbol}")
        print(f"  交割: {best_far.delivery_date} (+{best_far.months_to_delivery}个月)")
        print(f"  波动率: {best_far.volatility*100:.1f}% (比主力{vol_vs_main*100:+.1f}%)")


def example_6_sentiment_driven_direction():
    """示例6: 不同市场情绪下的方向选择"""
    print("\n" + "="*80)
    print("示例6: 不同市场情绪下的方向选择")
    print("="*80 + "\n")

    print(f"市场情绪与开仓方向对应关系:")
    print()

    scenarios = [
        {
            'name': '强烈多头',
            'up_ratio': 0.75,
            'top20_gain': 0.05,
            'top20_loss': 0.02,
            'expected': '做多',
            'confidence': 0.85,
        },
        {
            'name': '温和多头',
            'up_ratio': 0.65,
            'top20_gain': 0.03,
            'top20_loss': 0.02,
            'expected': '做多',
            'confidence': 0.70,
        },
        {
            'name': '中性偏多',
            'up_ratio': 0.55,
            'top20_gain': 0.04,
            'top20_loss': 0.03,
            'expected': '做多',
            'confidence': 0.60,
        },
        {
            'name': '完全中性',
            'up_ratio': 0.50,
            'top20_gain': 0.02,
            'top20_loss': 0.02,
            'expected': '中性（观望或按其他因素）',
            'confidence': 0.50,
        },
        {
            'name': '中性偏空',
            'up_ratio': 0.45,
            'top20_gain': 0.02,
            'top20_loss': 0.04,
            'expected': '做空',
            'confidence': 0.60,
        },
        {
            'name': '温和空头',
            'up_ratio': 0.35,
            'top20_gain': 0.02,
            'top20_loss': 0.03,
            'expected': '做空',
            'confidence': 0.70,
        },
        {
            'name': '强烈空头',
            'up_ratio': 0.25,
            'top20_gain': 0.01,
            'top20_loss': 0.05,
            'expected': '做空',
            'confidence': 0.85,
        },
    ]

    print(f"{'情景':<15} {'上涨比':<10} {'前20涨幅':<12} {'前20跌幅':<12} {'预期方向':<20} {'信心度'}")
    print("-" * 100)

    for scenario in scenarios:
        print(f"{scenario['name']:<15} {scenario['up_ratio']*100:>6.0f}%   "
              f"{scenario['top20_gain']*100:>10.1f}%     "
              f"{scenario['top20_loss']*100:>10.1f}%     "
              f"{scenario['expected']:<20} {scenario['confidence']:.2f}")

    print(f"\n决策规则:")
    print(f"  1. 上涨品种 > 60% → 做多")
    print(f"  2. 上涨品种 < 40% → 做空")
    print(f"  3. 前20涨幅平均 > 前20跌幅平均 → 强化方向")
    print(f"  4. 前20跌幅平均 > 前20涨幅平均 → 反向方向")
    print(f"  5. 信心度 > 0.65 才开仓")


def example_7_risk_management():
    """示例7: 风险管理详解"""
    print("\n" + "="*80)
    print("示例7: 期货仓位风险管理详解")
    print("="*80 + "\n")

    print(f"核心规则: 期货仓位风险度 < 50%")
    print()

    # 场景演示
    scenarios = [
        {
            'name': '场景1: 正常配置',
            'contracts': 10,
            'price': 4000,
            'margin_rate': 0.15,
        },
        {
            'name': '场景2: 激进配置',
            'contracts': 20,
            'price': 4000,
            'margin_rate': 0.15,
        },
        {
            'name': '场景3: 超额配置',
            'contracts': 30,
            'price': 4000,
            'margin_rate': 0.15,
        },
    ]

    for scenario in scenarios:
        print(f"{scenario['name']}:")
        print(f"  合约数: {scenario['contracts']}")
        print(f"  合约价格: ${scenario['price']}")

        contract_value = scenario['price'] * scenario['contracts']
        margin = contract_value * scenario['margin_rate']
        reserve = margin  # 1倍预留
        total = margin + reserve
        risk_level = margin / total

        print(f"  合约价值: ${contract_value:,.0f}")
        print(f"  保证金需求: ${margin:,.0f}")
        print(f"  预留资金: ${reserve:,.0f} (1倍保证金)")
        print(f"  总资金需求: ${total:,.0f}")
        print(f"  风险度: {risk_level*100:.1f}%")

        if risk_level > 0.50:
            print(f"  ⚠️  风险度超过50%，需要调整")
            # 计算调整后的合约数
            target_contracts = int((total * 0.50) / (scenario['price'] * scenario['margin_rate']))
            print(f"  调整为: {target_contracts}份合约")
        else:
            print(f"  ✅ 风险度符合要求")

        print()


def generate_mock_futures_data():
    """生成模拟期货数据"""
    np.random.seed(42)

    symbols = [
        'RB', 'CU', 'AL', 'ZN', 'AU', 'AG', 'CL', 'MA', 'PP', 'L',
        'M', 'Y', 'P', 'A', 'C', 'JD', 'CF', 'SR', 'OI', 'RM',
        'IF', 'IH', 'IC', 'IM', 'T', 'TF', 'TS',
    ]

    market_data = {}

    for symbol in symbols:
        dates = pd.date_range(start='2026-05-01', periods=10, freq='D')

        # 随机游走，但让有些涨有些跌
        trend = np.random.choice([-0.001, 0.001])
        returns = np.random.normal(trend, 0.02, 10)
        prices = [3000.0]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        df = pd.DataFrame({
            'Close': prices,
        }, index=dates)

        market_data[symbol] = df

    return market_data


def main():
    """主函数"""
    print("\n" + "="*80)
    print(" " * 15 + "远期期货合约交易策略 - 完整示例")
    print("="*80)

    # 运行所有示例
    example_1_select_far_month_contract()
    example_2_margin_and_reserve()
    example_3_market_sentiment()
    example_4_complete_workflow()
    example_5_contracts_comparison()
    example_6_sentiment_driven_direction()
    example_7_risk_management()

    print("\n" + "="*80)
    print(" " * 25 + "所有示例运行完成！")
    print("="*80)

    print("\n📊 策略核心要点总结:")
    print("-" * 80)
    print("\n1. 远期合约选择:")
    print("   - 至少+2个月")
    print("   - 波动率高于主力合约")
    print("   - 优先选择波动最大的")

    print("\n2. 资金管理:")
    print("   - 保证金占用 < 50%")
    print("   - 预留资金 >= 保证金（1倍）")
    print("   - 风险度 = 保证金/(保证金+预留)")

    print("\n3. 方向选择:")
    print("   - 每日分析市场多空情绪")
    print("   - 上涨品种 > 60% → 做多")
    print("   - 前20涨幅 > 前20跌幅 → 做多")
    print("   - 信心度 > 0.65 才开仓")

    print("\n💡 成功关键:")
    print("   - 严格遵守远期合约规则")
    print("   - 控制风险度在50%以下")
    print("   - 根据市场情绪灵活调整方向")
    print("   - 每日监控，及时止盈止损")


if __name__ == "__main__":
    main()
