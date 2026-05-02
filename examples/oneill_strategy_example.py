"""
欧奈尔CANSLIM交易体系 - 使用示例

演示如何使用欧奈尔交易系统的各个模块。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quant_trade_system.patterns import ONeillPatternDetector
from quant_trade_system.factors import CANSLIM_Screener, calculate_relative_strength
from quant_trade_system.signals import PocketPivotDetector
from quant_trade_system.strategies import run_oneill_strategy


def generate_sample_data(symbol="AAPL", days=500):
    """生成示例价格数据"""

    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')

    # 模拟价格走势（包含上升趋势和调整）
    price = 100.0
    prices = []
    volumes = []

    for i in range(days):
        # 趋势 + 随机游走
        trend = 0.0005  # 上升趋势
        noise = np.random.normal(0, 0.02)

        # 偶尔加入调整（形成基底）
        if i % 100 == 0 and i > 50 and i < days - 100:
            # 模拟杯柄形态
            for j in range(min(30, days - i)):
                adjustment = -0.001 * (30 - j) / 30
                price = price * (1 + adjustment + np.random.normal(0, 0.015))
                prices.append(price)
                volumes.append(np.random.randint(1000000, 5000000))
                i += 1

        price = price * (1 + trend + noise)
        prices.append(price)
        volumes.append(np.random.randint(1000000, 5000000))

    # 生成OHLCV数据
    data = pd.DataFrame({
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes,
    }, index=dates)

    return data


def example_1_canslim_screening():
    """示例1: CANSLIM 选股"""

    print("\n" + "="*80)
    print("示例1: CANSLIM 选股")
    print("="*80 + "\n")

    # 生成数据
    stock_data = generate_sample_data("AAPL", 500)
    market_data = generate_sample_data("SPY", 500)

    # 模拟基本面数据
    fundamental_data = {
        'eps_growth_current': 28.0,         # 当季EPS增长28%
        'eps_growth_annual': 32.0,          # 年化EPS增长32%
        'historical_annual_eps': [2.0, 2.6, 3.4, 4.5],  # 连续增长
        'rs_rating': 85,                      # 相对强度85
        'institutional_ownership': 48.0,     # 机构持股48%
        'institutional_count': 135,          # 135家机构
        'shares_outstanding': 50_000_000,    # 5000万股
        'all_time_high': True,
        'new_products': True,
        'week_high_52w': 2,                  # 距52周高点2周
    }

    # 创建选股器
    screener = CANSLIM_Screener()

    # 运行筛选
    score = screener.screen_stock(
        stock_data=stock_data,
        fundamental_data=fundamental_data,
        market_index_data=market_data,
        symbol="AAPL",
    )

    # 显示结果
    print(f"股票: AAPL")
    print(f"CANSLIM总分: {score.total_score:.1f}/100")
    print(f"\n各要素得分:")
    print(f"  C (当季收益): {score.c_score*100:.0f}/100")
    print(f"  A (年化收益): {score.a_score*100:.0f}/100")
    print(f"  N (新产品/新高): {score.n_score*100:.0f}/100")
    print(f"  S (供需关系): {score.s_score*100:.0f}/100")
    print(f"  L (领涨股): {score.l_score*100:.0f}/100")
    print(f"  I (机构资金): {score.i_score*100:.0f}/100")
    print(f"  M (大盘趋势): {score.m_score*100:.0f}/100")

    print(f"\n通过项: {', '.join(score.passed_criteria)}")
    print(f"未通过: {', '.join(score.failed_criteria)}")

    if score.warnings:
        print(f"\n警告:")
        for warning in score.warnings:
            print(f"  ⚠️  {warning}")

    print(f"\n投资建议: {score.recommendation}")
    print(f"大盘趋势: {score.market_trend}\n")


def example_2_pattern_recognition():
    """示例2: 形态识别"""

    print("\n" + "="*80)
    print("示例2: 欧奈尔形态识别")
    print("="*80 + "\n")

    # 生成示例数据（包含杯柄形态）
    data = generate_sample_data("CUP", 300)

    # 创建形态检测器
    detector = ONeillPatternDetector()

    # 检测所有形态
    patterns = detector.detect_all_patterns(data)

    if not patterns:
        print("未检测到欧奈尔形态")
        print("提示：在真实数据中效果更好")
        return

    print(f"检测到 {len(patterns)} 个形态:\n")

    for i, pattern in enumerate(patterns[:5], 1):  # 只显示前5个
        print(f"{i}. {pattern.pattern_type.value.upper()}")
        print(f"   开始日期: {pattern.start_date.strftime('%Y-%m-%d')}")
        print(f"   结束日期: {pattern.end_date.strftime('%Y-%m-%d')}")
        print(f"   枢轴点: ${pattern.pivot_price:.2f}")
        print(f"   止损价: ${pattern.stop_loss_price:.2f}")
        print(f"   深度: {pattern.depth_pct:.1f}%")
        print(f"   宽度: {pattern.width_days}天")
        print(f"   质量: {pattern.quality.value}")
        print(f"   对称性: {pattern.symmetry_score:.2f}")
        print(f"   描述: {pattern.description}")

        if pattern.warnings:
            print(f"   警告: {', '.join(pattern.warnings)}")
        print()


def example_3_pocket_pivot():
    """示例3: 口袋支点检测"""

    print("\n" + "="*80)
    print("示例3: 口袋支点检测")
    print("="*80 + "\n")

    # 生成上升趋势的示例数据
    data = generate_sample_data("PP", 200)

    # 创建口袋支点检测器
    detector = PocketPivotDetector()

    # 检测信号
    signals = detector.detect_signals(data)

    if not signals:
        print("未检测到口袋支点信号")
        return

    print(f"检测到 {len(signals)} 个口袋支点信号:\n")

    for i, signal in enumerate(signals[:5], 1):
        print(f"{i}. 日期: {signal.date.strftime('%Y-%m-%d')}")
        print(f"   价格: ${signal.price:.2f}")
        print(f"   成交量: {signal.volume_ratio:.1f}倍前期下跌日峰值")
        print(f"   突破: {signal.ma_broken}")
        print(f"   止损: ${signal.stop_loss_price:.2f}")
        print(f"   信号强度: {signal.strength_score:.2f}/1.0")
        print(f"   描述: {signal.description}")
        print(f"   环境: {signal.context}")
        print()


def example_4_relative_strength():
    """示例4: 相对强度计算"""

    print("\n" + "="*80)
    print("示例4: 相对强度评级（RS Rating）")
    print("="*80 + "\n")

    # 生成股票和指数数据
    stock_data = generate_sample_data("STOCK", 252)
    index_data = generate_sample_data("INDEX", 252)

    # 计算相对强度
    rs_rating = calculate_relative_strength(stock_data, index_data, period_days=252)

    print(f"股票相对强度评级: {rs_rating:.1f}/100")

    if rs_rating >= 80:
        print("评级: 优秀 - 明显领先市场")
    elif rs_rating >= 70:
        print("评级: 良好 - 领先市场")
    elif rs_rating >= 50:
        print("评级: 一般 - 与市场持平")
    else:
        print("评级: 较差 - 落后市场")
    print()


def example_5_follow_through_day():
    """示例5: 后续交易日检测"""

    print("\n" + "="*80)
    print("示例5: 后续交易日（FTD）检测")
    print("="*80 + "\n")

    # 模拟大盘下跌后的数据
    data = generate_sample_data("MARKET", 100)

    # 让前几天下跌
    for i in range(20, 30):
        data.loc[data.index[i], 'close'] *= 0.99

    # 让某一天大幅上涨（模拟FTD）
    ftd_day = 35
    data.loc[data.index[ftd_day], 'close'] *= 1.02
    data.loc[data.index[ftd_day], 'volume'] *= 2.0

    # 检测FTD
    is_ftd, ftd_date, ftd_desc = detect_follow_through_day(data)

    print(f"后续交易日确认: {'是' if is_ftd else '否'}")

    if is_ftd:
        print(f"FTD日期: {ftd_date.strftime('%Y-%m-%d')}")
        print(f"说明: {ftd_desc}")
        print("\n结论: 可以开始积极寻找买入机会")
    else:
        print("说明: 尚未出现有效的后续交易日信号")
    print()


def example_6_complete_strategy():
    """示例6: 完整策略运行"""

    print("\n" + "="*80)
    print("示例6: 完整欧奈尔策略运行")
    print("="*80 + "\n")

    # 准备多只股票的数据
    stocks_data = {
        'AAPL': generate_sample_data("AAPL", 300),
        'TSLA': generate_sample_data("TSLA", 300),
        'NVDA': generate_sample_data("NVDA", 300),
        'MSFT': generate_sample_data("MSFT", 300),
    }

    # 模拟基本面数据
    fundamentals_dict = {}
    for symbol in stocks_data.keys():
        fundamentals_dict[symbol] = {
            'eps_growth_current': np.random.uniform(20, 40),
            'eps_growth_annual': np.random.uniform(25, 45),
            'historical_annual_eps': [2.0 + i*0.5 for i in range(4)],
            'rs_rating': np.random.uniform(70, 95),
            'institutional_ownership': np.random.uniform(30, 70),
            'institutional_count': np.random.randint(50, 200),
            'shares_outstanding': np.random.randint(10_000_000, 200_000_000),
            'all_time_high': np.random.choice([True, False]),
            'new_products': np.random.choice([True, False]),
            'week_high_52w': np.random.randint(1, 10),
        }

    # 大盘数据
    market_data = generate_sample_data("SPY", 500)

    # 运行完整策略
    print("正在运行欧奈尔策略...\n")

    engine = run_oneill_strategy(
        stocks_data=stocks_data,
        fundamentals_dict=fundamentals_dict,
        market_index_data=market_data,
        initial_capital=100_000,
        max_positions=3,
    )

    print("\n✅ 策略运行完成！")
    print("\n提示：这是演示数据，实际使用时请提供真实的价格和基本面数据")


def main():
    """主函数"""
    print("\n" + "="*80)
    print(" " * 20 + "欧奈尔CANSLIM交易体系 - 使用示例")
    print("="*80)

    # 运行各个示例
    example_1_canslim_screening()
    example_2_pattern_recognition()
    example_3_pocket_pivot()
    example_4_relative_strength()
    example_5_follow_through_day()
    example_6_complete_strategy()

    print("\n" + "="*80)
    print(" " * 30 + "所有示例运行完成！")
    print("="*80 + "\n")

    print("📚 更多信息请查看:")
    print("   - docs/欧奈尔交易体系指南.md")
    print("   - quant_trade_system/patterns/")
    print("   - quant_trade_system/factors/")
    print("   - quant_trade_system/signals/")
    print("   - quant_trade_system/strategies/\n")


if __name__ == "__main__":
    main()
