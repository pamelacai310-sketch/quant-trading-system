"""
MAE/MFE 策略诊断与优化系统 - 完整使用示例

本示例展示如何使用MAE/MFE诊断系统进行策略优化
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quant_trade_system.diagnostics import (
    MAE_MFE_Diagnostics,
    ExtremeTradeAnalyzer,
    analyze_extreme_trades_auto,
)
from quant_trade_system.execution import (
    DynamicStopManager,
    EntryFilter,
    OptimizedBacktester,
    create_recommended_stop_manager,
    create_recommended_entry_filter,
)
from quant_trade_system.monitoring import LiveMonitor


def generate_sample_data():
    """生成示例数据"""

    # 生成价格数据
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    n = len(dates)

    # 模拟价格走势
    price = 100.0
    prices = []
    volumes = []

    for i in range(n):
        change = np.random.normal(0, 0.02)  # 2%日波动
        price = price * (1 + change)
        prices.append(price)
        volumes.append(np.random.randint(100000, 1000000))

    data = pd.DataFrame({
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes,
    }, index=dates)

    # 生成示例交易
    trades = []
    for i in range(50):
        entry_idx = np.random.randint(0, n - 30)
        exit_idx = entry_idx + np.random.randint(5, 30)

        entry_price = data.iloc[entry_idx]['close']
        exit_price = data.iloc[exit_idx]['close']

        # 计算MAE/MFE
        period_data = data.iloc[entry_idx:exit_idx+1]
        mae = (period_data['low'].min() / entry_price - 1) * 100
        mfe = (period_data['high'].max() / entry_price - 1) * 100

        trades.append({
            'entry_date': data.index[entry_idx],
            'exit_date': data.index[exit_idx],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': 1000,
            'side': 'long',
            'mae_pct': mae,
            'mfe_pct': mfe,
            'pnl_pct': (exit_price / entry_price - 1) * 100,
        })

    return data, trades


def example_1_basic_diagnosis():
    """示例1: 基础MAE/MFE诊断"""

    print("\n" + "="*80)
    print("示例1: 基础MAE/MFE诊断")
    print("="*80 + "\n")

    # 生成示例数据
    price_data, trades = generate_sample_data()

    # 创建诊断器
    diagnostics = MAE_MFE_Diagnostics()

    # 准备回测结果格式
    backtest_result = type('BacktestResult', (), {
        'trades': trades,
    })()

    # 计算MAE/MFE
    diagnosis = diagnostics.calculate_from_backtest(backtest_result, price_data)

    # 打印诊断报告
    diagnostics.print_diagnosis_report()

    # 绘制散点图
    fig = diagnostics.plot_mae_mfe_scatter(
        save_path='mae_mfe_scatter.png',
        figsize=(14, 12),
    )
    plt.close(fig)

    print("✅ 散点图已保存至: mae_mfe_scatter.png\n")


def example_2_dynamic_stops():
    """示例2: 动态止损止盈优化"""

    print("\n" + "="*80)
    print("示例2: 动态止损止盈优化")
    print("="*80 + "\n")

    # 生成示例数据
    price_data, original_trades = generate_sample_data()

    # 创建优化回测器
    backtester = OptimizedBacktester(original_trades, price_data)

    # 运行优化回测（使用推荐配置）
    print("运行优化回测（带硬止损-6% + 动态跟踪止盈8%/3%）...")
    comparison = backtester.run_with_stops()

    # 打印对比结果
    print("\n📊 优化前后对比:")
    print(f"   交易数量: {comparison['original']['trade_count']} -> {comparison['optimized']['trade_count']}")
    print(f"   胜率: {comparison['original']['win_rate']*100:.1f}% -> {comparison['optimized']['win_rate']*100:.1f}%")
    print(f"   平均盈亏: {comparison['original']['avg_pnl_pct']:.2f}% -> {comparison['optimized']['avg_pnl_pct']:.2f}%")
    print(f"   总收益: {comparison['original']['total_return_pct']*100:.1f}% -> {comparison['optimized']['total_return_pct']*100:.1f}%")

    improvement = comparison['improvement']
    print(f"\n📈 改进幅度:")
    print(f"   胜率变化: {improvement['win_rate_delta']*100:+.1f}%")
    print(f"   平均盈亏变化: {improvement['avg_pnl_delta']:+.2f}%")
    print(f"   总收益变化: {improvement['total_return_delta']*100:+.1f}%")

    print("\n✅ 优化回测完成\n")


def example_3_entry_filter():
    """示例3: 买入过滤器优化"""

    print("\n" + "="*80)
    print("示例3: 买入过滤器优化")
    print("="*80 + "\n")

    # 生成示例数据
    price_data, original_trades = generate_sample_data()

    # 创建优化回测器
    backtester = OptimizedBacktester(original_trades, price_data)

    # 创建买入过滤器
    entry_filter = create_recommended_entry_filter()

    print("买入过滤器配置:")
    print("  ✓ 均线多头排列（20日 > 60日）")
    print("  ✓ MACD金叉确认")
    print("  ✓ 放量突破（成交量 > 1.5倍均值）")

    # 运行优化回测
    print("\n运行优化回测（带买入过滤）...")
    comparison = backtester.run_with_stops(entry_filter=entry_filter)

    # 打印结果
    print("\n📊 过滤后交易:")
    print(f"   原始交易数: {comparison['original']['trade_count']}")
    print(f"   过滤后交易数: {comparison['optimized']['trade_count']}")
    print(f"   过滤比例: {(1 - comparison['optimized']['trade_count']/comparison['original']['trade_count'])*100:.1f}%")
    print(f"   过滤后胜率: {comparison['optimized']['win_rate']*100:.1f}%")

    print("\n✅ 买入过滤器测试完成\n")


def example_4_extreme_trade_analysis():
    """示例4: 极端交易分析"""

    print("\n" + "="*80)
    print("示例4: 极端交易分析")
    print("="*80 + "\n")

    # 生成示例数据
    price_data, trades = generate_sample_data()

    # 创建极端交易分析器
    analyzer = ExtremeTradeAnalyzer(
        extreme_threshold=10.0,   # 盈亏超过10%
        mae_threshold=-10.0,      # MAE超过-10%
        mfe_threshold=15.0,       # MFE超过15%
    )

    # 准备回测结果
    backtest_result = type('BacktestResult', (), {
        'trades': trades,
    })()

    # 提取极端交易
    extreme_trades = analyzer.extract_extreme_trades(backtest_result, price_data)

    print(f"📊 发现 {len(extreme_trades)} 笔极端交易\n")

    # 分析模式
    patterns = analyzer.analyze_patterns()
    if 'patterns' in patterns:
        for trade_type, pattern in patterns['patterns'].items():
            print(f"🔍 {trade_type.upper()}:")
            print(f"   数量: {pattern['count']}")
            print(f"   平均持仓天数: {pattern['avg_holding_days']:.1f}")
            print(f"   平均MAE: {pattern['avg_mae_pct']:.2f}%")
            print(f"   平均MFE: {pattern['avg_mfe_pct']:.2f}%")
            print()

    # 生成洞察报告
    report = analyzer.generate_insights_report()
    print(report)

    # 绘制极端交易K线图（保存到examples目录）
    print("生成极端交易K线图...")
    saved_paths = analyzer.plot_all_extreme_trades(
        price_data,
        save_dir='extreme_trades_plots',
    )
    print(f"✅ 已生成 {len(saved_paths)} 张K线图，保存至: extreme_trades_plots/\n")


def example_5_live_monitoring():
    """示例5: 实盘监控"""

    print("\n" + "="*80)
    print("示例5: 实盘监控与熔断")
    print("="*80 + "\n")

    # 创建监控器
    monitor = LiveMonitor(
        strategy_id="example_strategy",
        circuit_breaker_cooldown_hours=24,
    )

    print("实盘监控器已创建")
    print(f"  监控指标: {list(monitor.monitoring_intervals.keys())}")
    print(f"  熔断冷却期: {monitor.circuit_breaker_cooldown_hours} 小时")

    # 模拟添加持仓
    monitor.add_position(
        symbol="AAPL",
        entry_price=150.0,
        position_size=100,
        side='long',
    )
    print("\n✅ 已添加持仓: AAPL @ 150.0")

    # 模拟价格更新（正常情况）
    print("\n模拟价格更新...")
    monitor.update_position("AAPL", current_price=148.0)  # -1.33%
    monitor.update_position("AAPL", current_price=152.0)  # +1.33%

    # 检查是否允许新开仓
    allowed, reason = monitor.should_allow_new_position()
    print(f"\n是否允许新开仓: {allowed}")
    print(f"原因: {reason}")

    # 生成周报
    weekly_report = monitor.generate_weekly_report()
    print("\n📊 周报:")
    print(f"  {weekly_report}")

    # 导出监控数据
    monitor.export_monitoring_data('monitoring_data.json')
    print("\n✅ 监控数据已导出至: monitoring_data.json\n")


def example_6_complete_workflow():
    """示例6: 完整工作流"""

    print("\n" + "="*80)
    print("示例6: 完整的MAE/MFE优化工作流")
    print("="*80 + "\n")

    # 1. 生成数据
    print("步骤1: 准备数据...")
    price_data, trades = generate_sample_data()
    backtest_result = type('BacktestResult', (), {'trades': trades})()
    print(f"   价格数据: {len(price_data)} 天")
    print(f"   交易记录: {len(trades)} 笔\n")

    # 2. 初始诊断
    print("步骤2: 初始MAE/MFE诊断...")
    diagnostics = MAE_MFE_Diagnostics()
    diagnosis = diagnostics.calculate_from_backtest(backtest_result, price_data)
    print(f"   平均MAE: {diagnosis.avg_mae_pct:.2f}%")
    print(f"   MFE利用率: {diagnosis.mfe_utilization_rate*100:.1f}%")
    print(f"   整体健康: {diagnosis.overall_health}\n")

    # 3. 极端交易分析
    print("步骤3: 极端交易分析...")
    extreme_trades, report = analyze_extreme_trades_auto(
        backtest_result, price_data, save_plots=False
    )
    print(f"   极端交易数: {len(extreme_trades)}\n")

    # 4. 优化回测
    print("步骤4: 优化回测...")
    backtester = OptimizedBacktester(trades, price_data)

    # 使用推荐配置
    stop_manager = create_recommended_stop_manager()
    entry_filter = create_recommended_entry_filter()

    print("   优化配置:")
    print("     - 硬止损: -6%")
    print("     - 动态跟踪止盈: 浮盈8%激活，回撤3%止盈")
    print("     - 买入过滤: 均线多头排列 + MACD金叉 + 放量突破")

    comparison = backtester.run_with_stops(
        stop_configs=stop_manager.stop_configs,
        entry_filter=entry_filter,
    )

    print("\n步骤5: 优化结果:")
    print(f"   交易数: {comparison['original']['trade_count']} -> {comparison['optimized']['trade_count']}")
    print(f"   胜率: {comparison['original']['win_rate']*100:.1f}% -> {comparison['optimized']['win_rate']*100:.1f}%")
    print(f"   总收益: {comparison['original']['total_return_pct']*100:.1f}% -> {comparison['optimized']['total_return_pct']*100:.1f}%")

    improvement = comparison['improvement']
    if improvement['total_return_delta'] > 0:
        print(f"\n✅ 优化成功！总收益提升 {improvement['total_return_delta']*100:.1f}%")
    else:
        print(f"\n⚠️  优化后收益下降 {abs(improvement['total_return_delta'])*100:.1f}%，但胜率提升 {improvement['win_rate_delta']*100:.1f}%")

    # 6. 建立监控
    print("\n步骤6: 建立实盘监控...")
    live_monitor = LiveMonitor(strategy_id="optimized_strategy")
    print("   ✅ 监控器已就绪，可开始实盘交易\n")


def main():
    """主函数"""
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt

    print("\n" + "="*80)
    print(" " * 15 + "MAE/MFE 策略诊断与优化系统 - 使用示例")
    print("="*80)

    # 运行各个示例
    example_1_basic_diagnosis()
    example_2_dynamic_stops()
    example_3_entry_filter()
    example_4_extreme_trade_analysis()
    example_5_live_monitoring()
    example_6_complete_workflow()

    print("\n" + "="*80)
    print(" " * 25 + "所有示例运行完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
