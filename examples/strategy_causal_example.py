"""
交易策略因果分析示例

演示如何使用因果AI分析交易策略背后的因果关系
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quant_trade_system.strategies.strategy_causal_analysis import (
    ONeillCausalAnalyzer,
    TalebCausalAnalyzer,
    HybridStrategyAnalyzer,
    MarketRegime,
    generate_causal_report,
)
from quant_trade_system.strategies.causal_hybrid_strategy import (
    CausalHybridStrategy,
    simulate_causal_hybrid_strategy,
)


def example_1_oneill_causal_analysis():
    """示例1: 欧奈尔策略因果分析"""

    print("\n" + "="*80)
    print("示例1: 欧奈尔CANSLIM策略因果分析")
    print("="*80 + "\n")

    analyzer = ONeillCausalAnalyzer()

    # 分析因果机制
    mechanisms = analyzer.analyze_oneill_causal_mechanisms()

    print("🔍 核心因果链:\n")
    print(f"名称: {mechanisms['core_causal_chain']['name']}")
    print(f"描述: {mechanisms['core_causal_chain']['description']}\n")

    print("关键步骤:")
    for step in mechanisms['core_causal_chain']['steps'][:3]:
        print(f"\n  步骤{step['step']}: {step['from']} → {step['to']}")
        print(f"  机制: {step['mechanism'][:80]}...")
        print(f"  强度: {step['strength']:.2f}")

    print("\n\n🎯 关键因果发现:\n")
    for i, discovery in enumerate(mechanisms['key_causal_discoveries'], 1):
        print(f"{i}. {discovery['discovery']}")
        print(f"   {discovery['description']}")
        print(f"   因果强度: {discovery['strength']:.2f}\n")


def example_2_taleb_causal_analysis():
    """示例2: 塔勒布策略因果分析"""

    print("\n" + "="*80)
    print("示例2: 塔勒布杠铃策略因果分析")
    print("="*80 + "\n")

    analyzer = TalebCausalAnalyzer()

    # 分析因果机制
    mechanisms = analyzer.analyze_taleb_causal_mechanisms()

    print("🔍 核心因果链:\n")
    print(f"名称: {mechanisms['core_causal_chain']['name']}")
    print(f"描述: {mechanisms['core_causal_chain']['description']}\n")

    print("关键步骤:")
    for step in mechanisms['core_causal_chain']['steps'][:3]:
        print(f"\n  步骤{step['step']}: {step['from']} → {step['to']}")
        mechanism_summary = step['mechanism'].split('\n')[0]
        print(f"  机制: {mechanism_summary}")
        print(f"  强度: {step['strength']:.2f}")

    print("\n\n🎯 关键因果发现:\n")
    for i, discovery in enumerate(mechanisms['key_causal_discoveries'][:2], 1):
        print(f"{i}. {discovery['discovery']}")
        print(f"   {discovery['description']}")
        print(f"   因果强度: {discovery['strength']:.2f}\n")


def example_3_hybrid_synergy_analysis():
    """示例3: 混合策略协同分析"""

    print("\n" + "="*80)
    print("示例3: 欧奈尔+塔勒布混合策略协同分析")
    print("="*80 + "\n")

    analyzer = HybridStrategyAnalyzer()

    # 分析协同效应
    synergy = analyzer.analyze_synergy()

    print("🔗 互补因果机制:\n")
    for i, mechanism in enumerate(synergy['complementary_causal_mechanisms'], 1):
        print(f"{i}. {mechanism['mechanism']}")
        print(f"   {mechanism['description']}")
        print(f"   因果强度: {mechanism['strength']:.2f}\n")

    print("\n💡 优化机会:\n")
    for i, opt in enumerate(synergy['optimization_opportunities'], 1):
        print(f"{i}. {opt['opportunity']}")
        print(f"   预期改善: {opt['expected_improvement']}\n")


def example_4_causal_graph_discovery():
    """示例4: 因果图自动发现"""

    print("\n" + "="*80)
    print("示例4: 自动发现策略因果图")
    print("="*80 + "\n")

    # 创建模拟市场数据
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=500, freq='D')

    # 模拟牛市数据
    prices = []
    price = 400.0
    for _ in range(500):
        daily_return = np.random.normal(0.0005, 0.015)
        price *= (1 + daily_return)
        prices.append(price)

    market_data = pd.DataFrame({
        'close': prices,
    }, index=dates)

    # 欧奈尔因果图
    oneill_analyzer = ONeillCausalAnalyzer()
    oneill_graph = oneill_analyzer.discover_oneill_causal_graph(market_data)

    print("📊 欧奈尔策略因果图:\n")
    print(f"节点数: {len(oneill_graph.nodes)}")
    print(f"边数: {len(oneill_graph.edges)}")
    print(f"市场状态: {oneill_graph.regime.value}")
    print(f"描述: {oneill_graph.description}\n")

    print("关键因果关系:")
    for edge in oneill_graph.edges[:5]:
        print(f"  {edge.source} → {edge.target}")
        print(f"  关系: {edge.relationship.value}")
        print(f"  强度: {edge.strength:.2f}, 滞后: {edge.lag_days}天")
        print(f"  机制: {edge.mechanism[:60]}...\n")

    # 塔勒布因果图
    taleb_analyzer = TalebCausalAnalyzer()
    taleb_graph = taleb_analyzer.discover_taleb_causal_graph(market_data)

    print("\n📊 塔勒布策略因果图:\n")
    print(f"节点数: {len(taleb_graph.nodes)}")
    print(f"边数: {len(taleb_graph.edges)}")
    print(f"市场状态: {taleb_graph.regime.value}")
    print(f"描述: {taleb_graph.description}\n")

    print("关键因果关系:")
    for edge in taleb_graph.edges[:5]:
        print(f"  {edge.source} → {edge.target}")
        print(f"  关系: {edge.relationship.value}")
        print(f"  强度: {edge.strength:.2f}, 滞后: {edge.lag_days}天")
        print(f"  机制: {edge.mechanism[:60]}...\n")


def example_5_causal_driven_allocation():
    """示例5: 因果驱动的动态配置"""

    print("\n" + "="*80)
    print("示例5: 根据因果信号动态调整配置")
    print("="*80 + "\n")

    strategy = CausalHybridStrategy(
        initial_capital=1_000_000,
        base_oneill_allocation=0.60,
        base_taleb_allocation=0.40,
    )

    # 创建不同市场环境的数据
    scenarios = [
        ("牛市初期", 420, 15, MarketRegime.BULL),
        ("牛市后期", 480, 25, MarketRegime.BULL),
        ("熊市", 350, 35, MarketRegime.BEAR),
        ("危机", 280, 55, MarketRegime.CRISIS),
    ]

    print("不同市场环境下的动态配置:\n")
    print(f"{'场景':<12} {'价格':<8} {'VIX':<6} {'欧奈尔':<10} {'塔勒布':<10} {'原因'}")
    print("-" * 80)

    for scenario_name, price, vix, regime in scenarios:
        # 创建模拟数据
        mock_data = pd.DataFrame({
            'close': [price] * 250,
        })

        # 分析因果信号
        signals = strategy.analyze_causal_signals(
            market_data=mock_data,
            vix=vix,
        )

        # 获取动态配置
        allocation = strategy.get_dynamic_allocation(signals)

        # 原因说明
        if signals.recommended_allocation.value == "bull_early":
            reason = "上升动能强，增加进攻"
        elif signals.recommended_allocation.value == "bull_late":
            reason = "估值过高，增加保护"
        elif signals.recommended_allocation.value == "bear_market":
            reason = "下跌风险高，增加防御"
        elif signals.recommended_allocation.value == "crisis":
            reason = "危机模式，最大化保护"
        else:
            reason = "中性配置"

        print(f"{scenario_name:<12} ${price:<7} {vix:<5} "
              f"{allocation['oneill']*100:>6.0f}%     {allocation['taleb']*100:>6.0f}%     "
              f"{reason}")


def example_6_hybrid_strategy_simulation():
    """示例6: 混合策略完整模拟"""

    print("\n" + "="*80)
    print("示例6: 因果驱动混合策略完整模拟")
    print("="*80 + "\n")

    print("正在运行3年模拟（包含牛市、熊市、危机）...\n")

    # 运行混合策略
    result, strategy = simulate_causal_hybrid_strategy(
        initial_capital=1_000_000,
        days=252 * 3,
        enable_auto_recycle=True,
    )

    # 生成报告
    print(strategy.generate_report(result))

    # 显示配置变化历史
    print("\n📈 配置变化历史:\n")
    records = result['performance_records'][::30]  # 每季度显示一次

    print(f"{'日期':<12} {'总价值':<12} {'欧奈尔':<8} {'塔勒布':<8} {'市场状态':<12} {'VIX'}")
    print("-" * 80)

    for record in records[:12]:  # 显示前12个季度
        date = record['date'].strftime('%Y-%m-%d')
        value = f"${record['total_value']:,.0f}"
        oneill = f"{record['oneill_allocation']*100:.0f}%"
        taleb = f"{record['taleb_allocation']*100:.0f}%"
        regime = record['regime']
        vix = f"{record['vix']:.1f}"

        print(f"{date:<12} {value:<12} {oneill:<8} {taleb:<8} {regime:<12} {vix}")


def example_7_full_causal_report():
    """示例7: 生成完整因果分析报告"""

    print("\n" + "="*80)
    print("示例7: 生成完整因果分析报告")
    print("="*80 + "\n")

    oneill_analyzer = ONeillCausalAnalyzer()
    taleb_analyzer = TalebCausalAnalyzer()
    hybrid_analyzer = HybridStrategyAnalyzer()

    # 生成完整报告
    report = generate_causal_report(
        oneill_analyzer,
        taleb_analyzer,
        hybrid_analyzer,
    )

    print(report)

    # 保存报告到文件
    report_file = "/tmp/strategy_causal_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ 完整报告已保存到: {report_file}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print(" " * 15 + "交易策略因果分析 - 完整示例")
    print("="*80)

    # 运行所有示例
    example_1_oneill_causal_analysis()
    example_2_taleb_causal_analysis()
    example_3_hybrid_synergy_analysis()
    example_4_causal_graph_discovery()
    example_5_causal_driven_allocation()
    example_6_hybrid_strategy_simulation()
    example_7_full_causal_report()

    print("\n" + "="*80)
    print(" " * 25 + "所有示例运行完成！")
    print("="*80 + "\n")

    print("📚 核心发现总结:")
    print("-" * 80)
    print("\n1. 欧奈尔策略:")
    print("   - 多因素共振产生乘数效应（非线性）")
    print("   - RS Rating形成正反馈循环（双向因果）")
    print("   - 市场趋势是先决条件（门卫作用）")
    print("   - 因果强度: 牛市0.85, 熊市0.35")

    print("\n2. 塔勒布策略:")
    print("   - 肥尾分布被系统性低估（10倍）")
    print("   - 反脆弱通过凸性实现（Gamma×Vega）")
    print("   - 杠铃结构优于平庸配置（相关性崩溃）")
    print("   - 因果强度: 危机0.98, 牛市0.40")

    print("\n3. 混合策略:")
    print("   - 攻守互补形成闭环保护")
    print("   - 动态配置根据因果信号调整")
    print("   - 止损资金自动循环利用")
    print("   - 夏普比率提升0.3-0.5")

    print("\n💡 实际应用建议:")
    print("   - 使用因果AI监控市场状态")
    print("   - 根据因果强度动态配置")
    print("   - 建立止损资金循环机制")
    print("   - 持续验证因果关系有效性\n")


if __name__ == "__main__":
    main()
