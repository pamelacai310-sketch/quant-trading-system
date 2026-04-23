"""
测试新集成的模块

测试Polars适配器和因子库的功能。
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path("/Users/caijiawen/Documents/New project/quant-trading-system")
sys.path.insert(0, str(project_root))

def test_polars_adapter():
    """测试Polars适配器"""
    print("=" * 50)
    print("测试 Polars 适配器")
    print("=" * 50)

    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=1000, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })

    print(f"✓ 创建测试数据: {len(df)} 行")

    try:
        from quant_trade_system.core.polars_adapter import (
            PolarsDataFrame,
            should_use_polars,
            compute_indicators_optimized
        )

        # 测试Polars可用性
        if should_use_polars(df):
            print("✓ Polars可用且数据量足够大")
        else:
            print("✗ Polars不可用或数据量太小")

        # 测试指标计算
        indicator_specs = [
            {"name": "sma_20", "type": "sma", "window": 20},
            {"name": "rsi_14", "type": "rsi", "window": 14},
            {"name": "momentum_10", "type": "momentum", "window": 10},
        ]

        result = compute_indicators_optimized(df, indicator_specs)
        print(f"✓ 计算了 {len(indicator_specs)} 个指标")
        print(f"  结果形状: {result.shape}")

        # 验证结果
        assert 'sma_20' in result.columns
        assert 'rsi_14' in result.columns
        assert 'momentum_10' in result.columns
        print("✓ 所有指标计算正确")

        # 测试性能
        from quant_trade_system.core.polars_adapter import PerformanceBenchmark
        benchmark = PerformanceBenchmark()
        perf = benchmark.benchmark_indicator_computation(df, indicator_specs)
        print(f"✓ 性能测试:")
        print(f"  Pandas时间: {perf['pandas_time']:.4f}秒")
        print(f"  Polars时间: {perf['polars_time']:.4f}秒")
        print(f"  加速比: {perf['speedup']:.2f}x")

        print("\n✓ Polars适配器测试通过\n")
        return True

    except Exception as e:
        print(f"\n✗ Polars适配器测试失败: {str(e)}\n")
        return False


def test_factor_library():
    """测试因子库"""
    print("=" * 50)
    print("测试因子库")
    print("=" * 50)

    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=500, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 102,
        'low': np.random.randn(500).cumsum() + 98,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 500)
    })

    print(f"✓ 创建测试数据: {len(df)} 行")

    try:
        from quant_trade_system.factors import (
            FactorLibrary,
            compute_technical_factors
        )

        # 测试因子库初始化
        library = FactorLibrary()
        print(f"✓ 因子库初始化成功")
        print(f"  注册因子数: {len(library.factor_registry)}")

        # 测试获取因子列表
        tech_factors = library.get_factor_list(category="technical")
        print(f"✓ 技术因子数量: {len(tech_factors)}")

        # 测试单个因子计算
        if 'sma_20' in library.factor_registry:
            sma_20 = library.compute_factor('sma_20', df)
            print(f"✓ 计算单个因子 sma_20")
            print(f"  非空值数量: {sma_20.notna().sum()}")

        # 测试批量因子计算
        factor_names = ['sma_20', 'rsi_14', 'momentum_10', 'bollinger_width']
        factors_df = library.compute_factor_batch(factor_names, df)
        print(f"✓ 批量计算了 {len(factor_names)} 个因子")
        print(f"  结果形状: {factors_df.shape}")

        # 测试技术因子类
        tech_result = compute_technical_factors(df, factors=['sma_10', 'ema_12', 'macd'])
        print(f"✓ 技术因子计算: {tech_result.shape}")

        # 显示部分结果
        print("\n前5行数据:")
        print(tech_result.head())

        print("\n✓ 因子库测试通过\n")
        return True

    except Exception as e:
        print(f"\n✗ 因子库测试失败: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """性能测试"""
    print("=" * 50)
    print("性能测试")
    print("=" * 50)

    # 创建大数据集
    dates = pd.date_range('2020-01-01', periods=10000, freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(10000).cumsum() + 100,
        'high': np.random.randn(10000).cumsum() + 102,
        'low': np.random.randn(10000).cumsum() + 98,
        'close': np.random.randn(10000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 10000)
    })

    print(f"✓ 创建大数据集: {len(df)} 行 ({len(df) * 6:,} 数据点)")

    try:
        import time
        from quant_trade_system.factors import compute_technical_factors

        # 测试10个因子的计算时间
        factors_to_test = [
            'sma_10', 'sma_20', 'ema_12', 'rsi_14',
            'macd', 'momentum_10', 'bollinger_width',
            'atr', 'volume_sma_20', 'roc_10'
        ]

        start = time.time()
        result = compute_technical_factors(df, factors=factors_to_test)
        elapsed = time.time() - start

        print(f"✓ 计算了 {len(factors_to_test)} 个因子")
        print(f"  计算时间: {elapsed:.4f}秒")
        print(f"  吞吐量: {len(df) / elapsed:.0f} 行/秒")

        if elapsed < 1.0:
            print("✓ 性能优秀 (<1秒)")
        elif elapsed < 5.0:
            print("✓ 性能良好 (<5秒)")
        else:
            print("⚠ 性能可优化 (>=5秒)")

        print("\n✓ 性能测试完成\n")
        return True

    except Exception as e:
        print(f"\n✗ 性能测试失败: {str(e)}\n")
        return False


if __name__ == "__main__":
    print("\n开始测试新集成的模块...\n")

    results = []

    # 运行所有测试
    results.append(("Polars适配器", test_polars_adapter()))
    results.append(("因子库", test_factor_library()))
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
