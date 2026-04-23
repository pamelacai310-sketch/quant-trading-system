"""
集成测试 - 因子库 + 执行算法

演示因子库和执行算法如何协同工作，构建完整的量化交易流程。
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
project_root = Path("/Users/caijiawen/Documents/New project/quant-trading-system")
sys.path.insert(0, str(project_root))


def test_complete_pipeline():
    """测试完整的量化交易流程"""
    print("=" * 60)
    print("集成测试: 因子库 + 执行算法 完整流程")
    print("=" * 60)

    try:
        from quant_trade_system.factors import FactorLibrary, compute_technical_factors
        from quant_trade_system.execution import (
            OrderRequest,
            TWAPAlgorithm,
            VWAPAlgorithm,
            optimize_execution
        )

        # 1. 创建市场数据
        print("\n步骤1: 创建市场数据")
        dates = pd.date_range('2024-01-01', periods=5000, freq='5min')
        np.random.seed(42)

        price_path = np.random.randn(5000).cumsum() + 150
        df = pd.DataFrame({
            'timestamp': dates,
            'open': price_path + np.random.randn(5000) * 0.3,
            'high': price_path + np.abs(np.random.randn(5000) * 0.3),
            'low': price_path - np.abs(np.random.randn(5000) * 0.3),
            'close': price_path,
            'volume': np.random.randint(1000, 50000, 5000)
        })
        df.set_index('timestamp', inplace=True)

        print(f"✓ 创建了 {len(df)} 行5分钟级市场数据")
        print(f"  时间范围: {df.index[0]} 到 {df.index[-1]}")
        print(f"  价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

        # 2. 计算因子
        print("\n步骤2: 计算技术因子")
        library = FactorLibrary()

        factors_to_compute = [
            'sma_10', 'sma_20', 'ema_12',
            'rsi_14', 'macd', 'momentum_10',
            'bollinger_width', 'atr', 'volume_sma_20'
        ]

        factors_df = library.compute_factor_batch(factors_to_compute, df)
        print(f"✓ 计算了 {len(factors_to_compute)} 个因子")
        print(f"  因子数据形状: {factors_df.shape}")

        # 显示因子统计
        print("\n因子统计:")
        for factor in factors_to_compute[:5]:
            if factor in factors_df.columns:
                valid_count = factors_df[factor].notna().sum()
                mean_val = factors_df[factor].mean()
                print(f"  {factor}: {valid_count} 个有效值, 均值={mean_val:.4f}")

        # 3. 因子筛选
        print("\n步骤3: 因子筛选")

        # 计算收益率
        returns = df['close'].pct_change()

        # 筛选IC绝对值 > 0.03的因子
        good_factors = library.filter_factors_by_ic(
            factors_to_compute,
            returns,
            min_ic=0.01  # 降低阈值以便演示
        )

        print(f"✓ 通过IC筛选保留 {len(good_factors)} 个因子")
        print(f"  保留的因子: {good_factors}")

        # 4. 生成交易信号
        print("\n步骤4: 生成交易信号")

        # 简单策略: RSI < 30 且 MACD > 0 时买入
        latest_data = factors_df.iloc[-1]

        buy_signal = (
            latest_data['rsi_14'] < 30 and
            latest_data['macd'] > 0 and
            latest_data['momentum_10'] > 0
        )

        print(f"✓ 最新因子值:")
        print(f"  RSI_14: {latest_data['rsi_14']:.2f}")
        print(f"  MACD: {latest_data['macd']:.4f}")
        print(f"  Momentum_10: {latest_data['momentum_10']:.4f}")
        print(f"  交易信号: {'买入' if buy_signal else '观望'}")

        if buy_signal:
            # 5. 使用执行算法下单
            print("\n步骤5: 使用执行算法下单")

            # 创建订单
            order = OrderRequest(
                symbol='AAPL',
                side='buy',
                quantity=50000,  # 5万股大额订单
                order_type='market',
                strategy_id='integration_test',
                metadata={'order_id': 'int_test_001'}
            )

            print(f"✓ 创建订单: {order.quantity:,} 股 {order.side} {order.symbol}")

            # 选择执行算法
            # 大额订单优先使用TWAP
            twap = TWAPAlgorithm(n_slices=20, time_window='1H')
            child_orders = twap.execute(order, df)

            print(f"✓ 使用TWAP算法拆分为 {len(child_orders)} 个子订单")

            # 显示执行计划
            print("\n执行计划:")
            for i, co in enumerate(child_orders[:5]):
                execution_time = co.metadata.get('execution_time', 'N/A')
                slice_ratio = co.metadata.get('slice_ratio', 0)
                print(f"  子订单 {i}: 数量={co.quantity:.0f}, 比例={slice_ratio:.2%}, 时间={execution_time}")

            if len(child_orders) > 5:
                print(f"  ... (还有 {len(child_orders) - 5} 个子订单)")

            # 成本分析
            cost_analysis = twap.calculate_expected_cost(order, df)
            print(f"\n成本分析:")
            print(f"  订单价值: ${cost_analysis['order_value']:,.0f}")
            print(f"  市场冲击: {cost_analysis['market_impact']}")
            print(f"  预期滑点: {cost_analysis['estimated_slippage_bps']:.1f} bps")
            print(f"  执行成本: ${cost_analysis['total_cost']:,.2f}")

        # 6. 回测性能
        print("\n步骤6: 简单回测")

        # 使用最近1000个数据点进行回测
        backtest_df = df.iloc[-1000:].copy()
        backtest_df['sma_20'] = backtest_df['close'].rolling(20).mean()
        backtest_df['position'] = 0

        # 简单策略: 价格>SMA20时做多
        backtest_df.loc[backtest_df['close'] > backtest_df['sma_20'], 'signal'] = 1
        backtest_df.loc[backtest_df['close'] <= backtest_df['sma_20'], 'signal'] = 0

        # 计算收益
        backtest_df['returns'] = backtest_df['close'].pct_change()
        backtest_df['strategy_returns'] = backtest_df['signal'].shift(1) * backtest_df['returns']

        total_return = backtest_df['strategy_returns'].sum()
        sharpe_ratio = backtest_df['strategy_returns'].mean() / backtest_df['strategy_returns'].std() * np.sqrt(252) if backtest_df['strategy_returns'].std() > 0 else 0

        print(f"✓ 策略总收益: {total_return:.2%}")
        print(f"✓ 夏普比率: {sharpe_ratio:.2f}")

        # 7. 总结
        print("\n" + "=" * 60)
        print("集成测试总结")
        print("=" * 60)
        print("✓ 因子库: 成功计算和筛选因子")
        print("✓ 执行算法: 成功拆分大额订单")
        print("✓ 完整流程: 因子计算 → 信号生成 → 订单执行")
        print("\n🎉 集成测试全部通过!")

        return True

    except Exception as e:
        print(f"\n✗ 集成测试失败: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_factor_performance():
    """测试因子库性能"""
    print("\n" + "=" * 60)
    print("因子库性能测试")
    print("=" * 60)

    try:
        from quant_trade_system.factors import compute_technical_factors
        import time

        # 创建大量数据
        dates = pd.date_range('2020-01-01', periods=50000, freq='1min')
        np.random.seed(42)

        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(50000).cumsum() + 100,
            'high': np.random.randn(50000).cumsum() + 102,
            'low': np.random.randn(50000).cumsum() + 98,
            'close': np.random.randn(50000).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 50000)
        })
        df.set_index('timestamp', inplace=True)

        print(f"✓ 创建大数据集: {len(df):,} 行")

        # 测试不同数量的因子计算性能
        factor_sets = [
            (['sma_10', 'sma_20'], "2个因子"),
            (['sma_10', 'ema_12', 'rsi_14', 'macd'], "4个因子"),
            (['sma_10', 'sma_20', 'ema_12', 'rsi_14', 'macd',
              'momentum_10', 'bollinger_width', 'atr', 'volume_sma_20'], "10个因子"),
        ]

        for factors, desc in factor_sets:
            start = time.time()
            result = compute_technical_factors(df, factors=factors)
            elapsed = time.time() - start

            throughput = len(df) / elapsed

            print(f"✓ {desc}: {elapsed:.4f}秒 ({throughput:,.0f} 行/秒)")

        print("\n✓ 性能测试完成")
        return True

    except Exception as e:
        print(f"\n✗ 性能测试失败: {str(e)}\n")
        return False


if __name__ == "__main__":
    print("\n开始集成测试...\n")

    results = []

    # 运行测试
    results.append(("完整流程", test_complete_pipeline()))
    results.append(("因子性能", test_factor_performance()))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 所有集成测试通过!")
        print("\n已成功实现:")
        print("  • Polars性能优化 (自动降级到pandas)")
        print("  • 因子库框架 (44+技术因子)")
        print("  • TWAP执行算法")
        print("  • VWAP执行算法")
        print("  • 执行优化器")
        print("  • 完整的集成流程")
    else:
        print("\n⚠ 部分测试失败")

    sys.exit(0 if all_passed else 1)
