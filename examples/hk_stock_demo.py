"""
欧奈尔策略 - 港股模拟扫描（2026年5月4日）

注意：这是模拟示例，展示如何使用欧奈尔系统。
实盘交易需要：实时数据+券商API+合规许可
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quant_trade_system.factors import CANSLIM_Screener
from quant_trade_system.patterns import ONeillPatternDetector
from quant_trade_system.signals import PocketPivotDetector
from quant_trade_system.strategies import ONeillStrategyEngine
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_hk_stock_sample_data(symbol, days=252):
    """生成港股模拟数据（用于演示）"""
    np.random.seed(42)
    dates = pd.date_range(end='2026-05-02', periods=days, freq='D')

    # 模拟港股走势
    price = 100.0
    prices = []
    volumes = []

    for i in range(days):
        # 趋势 + 随机
        trend = 0.0003
        noise = np.random.normal(0, 0.025)
        price = price * (1 + trend + noise)
        prices.append(price)
        volumes.append(np.random.randint(100000, 2000000))

    data = pd.DataFrame({
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.012))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.012))) for p in prices],
        'close': prices,
        'volume': volumes,
    }, index=dates)

    return data


def demo_hk_stock_analysis():
    """演示港股分析流程"""

    print("\n" + "="*80)
    print("欧奈尔策略 - 港股模拟扫描演示")
    print("目标日期：2026年5月4日（下周一）")
    print("="*80 + "\n")

    # 模拟港股列表
    hk_stocks = {
        '0700.HK': '腾讯控股',
        '09988.HK': '阿里巴巴-SW',
        '09941.HK': '中国移动',
        '1299.HK': '友邦保险',
        '2318.HK': '中国平安',
        '3988.HK': '中国银行',
        '0267.HK': '中信股份',
        '0883.HK': '中国海洋石油',
        '1093.HK': '石药集团',
        '0669.HK': '科技ETF',
    }

    # 模拟港股指数数据（恒生指数）
    hsi_data = generate_hk_stock_sample_data('HSI', days=500)

    print(f"📊 市场环境分析（恒生指数）")
    print("-" * 80)

    # 分析市场趋势
    current_price = hsi_data['close'].iloc[-1]
    ma50 = hsi_data['close'].rolling(window=50).mean().iloc[-1]
    ma200 = hsi_data['close'].rolling(window=200).mean().iloc[-1]

    print(f"当前恒指: {current_price:.2f}")
    print(f"50日均线: {ma50:.2f}")
    print(f"200日均线: {ma200:.2f}")

    if current_price > ma50 > ma200:
        market_trend = "Uptrend"
        recommendation = "✅ 可以积极寻找机会"
    elif current_price > ma200:
        market_trend = "Sideways_Up"
        recommendation = "⚠️ 谨慎做多"
    else:
        market_trend = "Downtrend"
        recommendation = "❌ 等待或观望"

    print(f"趋势: {market_trend}")
    print(f"建议: {recommendation}")

    print(f"\n📈 CANSLIM选股模拟（港股）")
    print("-" * 80)

    screener = CANSLIM_Screener(
        min_eps_growth_current=20.0,
        min_eps_growth_annual=25.0,
        min_rs_rating=70.0,
    )

    # 模拟筛选结果（前5名）
    simulated_results = [
        {
            'symbol': '0700.HK',
            'name': '腾讯控股',
            'eps_growth_current': 28.5,
            'eps_growth_annual': 32.0,
            'rs_rating': 88,
            'institutional_ownership': 52.0,
            'canslim_score': 82.3,
            'recommendation': '强烈买入',
        },
        {
            'symbol': '09988.HK',
            'name': '阿里巴巴-SW',
            'eps_growth_current': 35.2,
            'eps_growth_annual': 40.1,
            'rs_rating': 92,
            'institutional_ownership': 65.0,
            'canslim_score': 88.7,
            'recommendation': '强烈买入',
        },
        {
            'symbol': '1299.HK',
            'name': '友邦保险',
            'eps_growth_current': 22.8,
            'eps_growth_annual': 18.5,
            'rs_rating': 75,
            'institutional_ownership': 42.0,
            'canslim_score': 71.2,
            'recommendation': '买入',
        },
        {
            'symbol': '2318.HK',
            'name': '中国平安',
            'eps_growth_current': 18.5,
            'eps_growth_annual': 22.0,
            'rs_rating': 72,
            'institutional_ownership': 48.0,
            'canslim_score': 68.5,
            'recommendation': '谨慎买入',
        },
        {
            'symbol': '3988.HK',
            'name': '中国银行',
            'eps_growth_current': 15.2,
            'eps_growth_annual': 12.8,
            'rs_rating': 65,
            'institutional_ownership': 38.0,
            'canslim_score': 58.3,
            'recommendation': '观望',
        },
    ]

    print("\n排名 | 代码      | 名称       | CANSLIM | 建议")
    print("-" * 80)
    for i, stock in enumerate(simulated_results, 1):
        print(f"{i:2d}   | {stock['symbol']:10s} | {stock['name']:10s} | {stock['canslim_score']:5.1f}   | {stock['recommendation']}")

    print(f"\n📐 形态识别模拟")
    print("-" * 80)

    detector = ONeillPatternDetector()

    # 模拟形态检测结果
    simulated_patterns = [
        {
            'symbol': '0700.HK',
            'name': '腾讯控股',
            'pattern_type': 'Cup with Handle',
            'pivot_price': 385.0,
            'stop_loss': 354.0,
            'target': 481.0,
            'quality': 'Excellent',
            'description': '杯柄形态，深度25%，柄部深度8%',
        },
        {
            'symbol': '09988.HK',
            'name': '阿里巴巴-SW',
            'pattern_type': 'VCP',
            'pivot_price': 82.5,
            'stop_loss': 75.9,
            'target': 103.0,
            'quality': 'Excellent',
            'description': 'VCP形态，3次收缩：22%→15%→7%',
        },
        {
            'symbol': '0669.HK',
            'name': '科技ETF',
            'pattern_type': 'Pocket Pivot',
            'pivot_price': 25.8,
            'stop_loss': 23.7,
            'target': 32.0,
            'quality': 'Good',
            'description': '口袋支点，成交量1.8倍',
        },
    ]

    print("\n代码      | 名称       | 形态类型        | 枢轴价 | 止损  | 目标价 | 质量")
    print("-" * 80)
    for pattern in simulated_patterns:
        print(f"{pattern['symbol']:10s} | {pattern['name']:10s} | {pattern['pattern_type']:14s} | "
              f"${pattern['pivot_price']:6.2f} | ${pattern['stop_loss']:6.2f} | "
              f"${pattern['target']:6.2f} | {pattern['quality']}")

    print(f"\n💰 模拟交易信号（2026年5月4日）")
    print("=" * 80)

    print("\n【强烈买入信号 - CANSLIM得分>80 + 形态优秀】\n")

    for pattern in simulated_patterns:
        if pattern['quality'] == 'Excellent':
            print(f"🔥 {pattern['symbol']} - {pattern['name']}")
            print(f"   信号: {pattern['pattern_type']}")
            print(f"   入场: ${pattern['pivot_price']:.2f}")
            print(f"   止损: ${pattern['stop_loss']:.2f} (-8%)")
            print(f"   目标: ${pattern['target']:.2f} (+25%)")
            print(f"   说明: {pattern['description']}")
            print()

    print("【风险提示】")
    print("⚠️  以上为模拟示例，非实盘交易指令")
    print("⚠️  实盘交易需要：实时数据 + 券商API + 合规许可")
    print("⚠️  港股交易需要：港股通/本地券商 + 资金跨境汇兑")
    print()


def main():
    demo_hk_stock_analysis()


if __name__ == "__main__":
    main()
