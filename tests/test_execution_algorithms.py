"""
测试执行算法模块

测试TWAP和VWAP算法的功能和性能。
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
project_root = Path("/Users/caijiawen/Documents/New project/quant-trading-system")
sys.path.insert(0, str(project_root))


def create_test_order_and_data():
    """创建测试订单和市场数据"""
    # 创建测试订单
    from quant_trade_system.execution import OrderRequest

    order = OrderRequest(
        symbol='AAPL',
        side='buy',
        quantity=10000,
        order_type='market',
        strategy_id='test_strategy',
        metadata={'order_id': 'test_001'}
    )

    # 创建市场数据 (1天的分钟数据)
    dates = pd.date_range('2024-01-01 09:30', periods=390, freq='1min')
    np.random.seed(42)

    # 模拟价格走势
    price_path = np.random.randn(390).cumsum() + 150
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price_path + np.random.randn(390) * 0.5,
        'high': price_path + np.abs(np.random.randn(390) * 0.5),
        'low': price_path - np.abs(np.random.randn(390) * 0.5),
        'close': price_path,
        'volume': np.random.randint(1000, 50000, 390)
    })

    df.set_index('timestamp', inplace=True)

    return order, df


def test_twap_algorithm():
    """测试TWAP算法"""
    print("=" * 50)
    print("测试 TWAP 算法")
    print("=" * 50)

    try:
        from quant_trade_system.execution import TWAPAlgorithm, execute_twap
        order, market_data = create_test_order_and_data()

        print(f"✓ 创建测试订单: {order.quantity} 股 {order.symbol}")

        # 测试基础TWAP
        twap = TWAPAlgorithm(n_slices=10, time_window='30min')
        child_orders = twap.execute(order, market_data)

        print(f"✓ TWAP拆分为 {len(child_orders)} 个子订单")
        print(f"  每个子订单数量: {order.quantity / len(child_orders):.0f} 股")

        # 验证子订单
        total_qty = sum(co.quantity for co in child_orders)
        assert abs(total_qty - order.quantity) < 1, "总数量不匹配"
        print(f"✓ 子订单总数量匹配: {total_qty:.0f}")

        # 检查元数据
        for i, co in enumerate(child_orders[:3]):
            print(f"  子订单 {i}: 数量={co.quantity:.0f}, 算法={co.metadata.get('algorithm')}")

        # 测试执行时间表
        schedule = twap.estimate_execution_schedule(order)
        print(f"✓ 执行时间表生成成功，共 {len(schedule)} 个时间片")

        # 测试成本估算
        cost_analysis = twap.calculate_expected_cost(order, market_data)
        print(f"✓ 成本估算:")
        print(f"  订单价值: ${cost_analysis['order_value']:,.0f}")
        print(f"  市场影响: {cost_analysis['market_impact']}")
        print(f"  预期滑点: {cost_analysis['estimated_slippage_bps']:.1f} bps")
        print(f"  TWAP降低系数: {cost_analysis['twap_reduction_factor']:.2f}")

        print("\n✓ TWAP算法测试通过\n")
        return True

    except Exception as e:
        print(f"\n✗ TWAP算法测试失败: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_vwap_algorithm():
    """测试VWAP算法"""
    print("=" * 50)
    print("测试 VWAP 算法")
    print("=" * 50)

    try:
        from quant_trade_system.execution import VWAPAlgorithm, execute_vwap
        order, market_data = create_test_order_and_data()

        print(f"✓ 创建测试订单: {order.quantity} 股 {order.symbol}")

        # 测试VWAP
        vwap = VWAPAlgorithm(
            lookback_window='1D',
            time_bucket='30min',
            max_participation_rate=0.2
        )
        child_orders = vwap.execute(order, market_data)

        print(f"✓ VWAP拆分为 {len(child_orders)} 个子订单")

        # 验证子订单
        total_qty = sum(co.quantity for co in child_orders)
        print(f"✓ 子订单总数量: {total_qty:.0f}")

        # 检查订单类型
        assert all(co.order_type == 'limit' for co in child_orders), "VWAP应使用限价单"
        print(f"✓ 所有子订单都是限价单")

        # 检查成交量分布
        volume_distribution = []
        for co in child_orders:
            bucket_ratio = co.metadata.get('bucket_ratio', 0)
            volume_distribution.append(bucket_ratio)

        print(f"✓ 成交量分布范围: {min(volume_distribution):.3f} - {max(volume_distribution):.3f}")

        # 测试便捷函数
        child_orders2 = execute_vwap(order, market_data, lookback_window='1D')
        assert len(child_orders2) == len(child_orders), "便捷函数结果不一致"
        print(f"✓ 便捷函数正常工作")

        print("\n✓ VWAP算法测试通过\n")
        return True

    except Exception as e:
        print(f"\n✗ VWAP算法测试失败: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_execution_optimizer():
    """测试执行优化器"""
    print("=" * 50)
    print("测试执行优化器")
    print("=" * 50)

    try:
        from quant_trade_system.execution import (
            ExecutionOptimizer,
            optimize_execution,
            TWAPAlgorithm,
            VWAPAlgorithm
        )

        order, market_data = create_test_order_and_data()

        # 创建优化器
        optimizer = ExecutionOptimizer()

        # 注册算法
        optimizer.register_algorithm(TWAPAlgorithm(n_slices=10))
        optimizer.register_algorithm(VWAPAlgorithm(lookback_window='1D'))

        print(f"✓ 注册了 2 个执行算法")

        # 测试自动选择
        selected = optimizer.select_algorithm(order, market_data)
        print(f"✓ 自动选择的算法: {selected.name}")

        # 测试执行
        result = optimizer.execute_order(order, market_data)

        summary = result.get_summary()
        print(f"✓ 执行完成:")
        print(f"  使用算法: {summary['algorithm']}")
        print(f"  子订单数: {summary['num_child_orders']}")
        print(f"  填充率: {summary['stats']['fill_ratio']:.2%}")

        # 测试便捷函数
        result2 = optimize_execution(order, market_data, algorithm_name='twap')
        assert len(result2.child_orders) == len(result.child_orders), "便捷函数结果不一致"
        print(f"✓ 便捷函数正常工作")

        print("\n✓ 执行优化器测试通过\n")
        return True

    except Exception as e:
        print(f"\n✗ 执行优化器测试失败: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """性能测试"""
    print("=" * 50)
    print("执行算法性能测试")
    print("=" * 50)

    try:
        import time
        from quant_trade_system.execution import execute_twap, execute_vwap

        # 创建大订单
        from quant_trade_system.execution import OrderRequest
        order = OrderRequest(
            symbol='AAPL',
            side='buy',
            quantity=100000,
            order_type='market',
            strategy_id='test_large_order'
        )

        # 创建大量市场数据
        dates = pd.date_range('2024-01-01', periods=10000, freq='1min')
        df = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(10000).cumsum() + 150,
            'high': np.random.randn(10000).cumsum() + 152,
            'low': np.random.randn(10000).cumsum() + 148,
            'volume': np.random.randint(1000, 50000, 10000)
        })
        df.set_index('timestamp', inplace=True)

        print(f"✓ 创建大额订单: {order.quantity:,} 股")
        print(f"✓ 市场数据: {len(df):,} 行")

        # 测试TWAP性能
        start = time.time()
        twap_orders = execute_twap(order, df, n_slices=20)
        twap_time = time.time() - start

        print(f"✓ TWAP执行: {len(twap_orders)} 个子订单, 耗时 {twap_time:.4f}秒")

        # 测试VWAP性能
        start = time.time()
        vwap_orders = execute_vwap(order, df, lookback_window='1D')
        vwap_time = time.time() - start

        print(f"✓ VWAP执行: {len(vwap_orders)} 个子订单, 耗时 {vwap_time:.4f}秒")

        # 性能评估
        if twap_time < 0.1 and vwap_time < 0.1:
            print("✓ 性能优秀 (<0.1秒)")
        elif twap_time < 0.5 and vwap_time < 0.5:
            print("✓ 性能良好 (<0.5秒)")
        else:
            print("⚠ 性能可优化 (>=0.5秒)")

        print("\n✓ 性能测试完成\n")
        return True

    except Exception as e:
        print(f"\n✗ 性能测试失败: {str(e)}\n")
        return False


if __name__ == "__main__":
    print("\n开始测试执行算法模块...\n")

    results = []

    # 运行所有测试
    results.append(("TWAP算法", test_twap_algorithm()))
    results.append(("VWAP算法", test_vwap_algorithm()))
    results.append(("执行优化器", test_execution_optimizer()))
    results.append(("性能测试", test_performance()))

    # 总结
    print("=" * 50)
    print("测试总结")
    print("=" * 50)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠ 部分测试失败，请检查错误信息")

    sys.exit(0 if all_passed else 1)
