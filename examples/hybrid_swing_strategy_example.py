"""
融合策略示例：期货+股票周波段交易系统

本示例展示如何使用HybridSwingStrategy进行融合策略交易：

1. 创建融合策略
2. 扫描股票机会（长期看好标的，周波段）
3. 扫描期货机会（远期高波动合约）
4. 分析市场情绪
5. 统一开仓/平仓
6. 完整4周模拟
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from quant_trade_system.strategies import (
    HybridSwingStrategy,
    UnifiedPosition,
    AssetType,
    PositionSide,
    simulate_hybrid_strategy,
)


def generate_sample_market_data() -> dict:
    """生成示例市场数据"""
    np.random.seed(42)

    symbols = [
        # 股票
        'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN',
        '0700.HK', '9988.HK', '3690.HK',
        # 期货
        'RB', 'CU', 'AL', 'ZN', 'AU', 'AG', 'CL', 'MA', 'PP', 'L',
        'M', 'Y', 'P', 'A', 'C', 'JD', 'CF', 'SR', 'OI', 'RM',
        'IF', 'IH', 'IC', 'IM',
    ]

    market_data = {}

    for symbol in symbols:
        dates = pd.date_range(start='2026-05-01', periods=50, freq='D')

        # 期货价格范围更大
        base_price = 3000.0 if symbol in ['RB', 'CU', 'AL', 'ZN', 'IF', 'IH', 'IC'] else 100.0

        returns = np.random.normal(0.0005, 0.02, 50)
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        df = pd.DataFrame({
            'Close': prices,
        }, index=dates)

        market_data[symbol] = df

    return market_data


# ============================================================================
# 示例1：创建融合策略
# ============================================================================

def example1_create_strategy():
    """示例1：创建融合策略"""
    print("\n" + "="*80)
    print("示例1：创建融合策略")
    print("="*80)

    strategy = HybridSwingStrategy(
        initial_capital=1_000_000,
        weekly_target=20_000,      # 每周目标2万（2%）
        max_positions=5,            # 最大5个持仓
        max_hold_days=5,            # 最大持仓5天
        base_stop_loss=0.03,        # 止损3%
        base_take_profit=0.06,      # 止盈6%
        futures_risk_level=0.50,    # 期货风险度50%
        min_far_months=2,           # 期货最少+2个月
    )

    print(f"\n✅ 融合策略创建成功")
    print(f"  初始资金: ${strategy.initial_capital:,.0f}")
    print(f"  每周目标: ${strategy.weekly_target:,.0f} ({strategy.target_return_pct*100:.1f}%)")
    print(f"  最大持仓: {strategy.max_positions}个")
    print(f"  最大天数: {strategy.max_hold_days}天")
    print(f"  止损/止盈: {strategy.base_stop_loss*100:.0f}% / {strategy.base_take_profit*100:.0f}%")
    print(f"  股票池: {len(strategy.stock_universe)}个")
    print(f"  期货池: {len(strategy.futures_universe)}个")


# ============================================================================
# 示例2：扫描股票机会
# ============================================================================

def example2_scan_stocks():
    """示例2：扫描股票机会（长期看好标的，周波段）"""
    print("\n" + "="*80)
    print("示例2：扫描股票机会（周波段策略）")
    print("="*80)

    strategy = HybridSwingStrategy(initial_capital=1_000_000)

    market_data = generate_sample_market_data()
    current_date = datetime(2026, 5, 4)

    # 扫描股票机会
    stock_opportunities = strategy.scan_stock_opportunities(market_data, current_date)

    print(f"\n✅ 扫描到 {len(stock_opportunities)} 个股票机会")
    print(f"\n{'代码':<12} {'信号数':<8} {'RSI':<8} {'波动率':<10} {'长期看好':<10}")
    print("-" * 80)

    for opp in stock_opportunities[:5]:
        favor_str = "是" if opp['is_favorite'] else "否"
        print(f"{opp['symbol']:<12} {opp['signals']:<8.1f} {opp['rsi']:<8.1f} "
              f"{opp['volatility']*100:<10.1f}% {favor_str:<10}")


# ============================================================================
# 示例3：扫描期货机会（远期合约）
# ============================================================================

def example3_scan_futures():
    """示例3：扫描期货机会（远期高波动合约）"""
    print("\n" + "="*80)
    print("示例3：扫描期货机会（远期合约策略）")
    print("="*80)

    strategy = HybridSwingStrategy(initial_capital=1_000_000)
    current_date = datetime(2026, 5, 4)

    # 扫描期货合约
    futures_contracts = strategy.scan_futures_contracts(current_date)

    print(f"\n✅ 扫描到 {len(futures_contracts)} 个期货机会")
    print(f"\n{'合约':<12} {'名称':<20} {'交割月':<10} {'波动率':<10} {'成交量':<10}")
    print("-" * 80)

    for contract in futures_contracts[:5]:
        delivery_str = f"{contract.delivery_year}-{contract.delivery_month:02d}"
        print(f"{contract.symbol:<12} {contract.name:<20} {delivery_str:<10} "
              f"{contract.volatility*100:<10.1f}% {contract.volume:<10,.0f}")


# ============================================================================
# 示例4：分析市场情绪
# ============================================================================

def example4_market_sentiment():
    """示例4：分析市场情绪（统一情绪分析）"""
    print("\n" + "="*80)
    print("示例4：分析市场情绪（统一情绪分析）")
    print("="*80)

    strategy = HybridSwingStrategy(initial_capital=1_000_000)

    market_data = generate_sample_market_data()
    current_date = datetime(2026, 5, 4)

    # 分析市场情绪
    sentiment = strategy.analyze_market_sentiment(market_data, current_date)

    print(f"\n✅ 市场情绪分析完成 ({current_date.strftime('%Y-%m-%d')})")
    print(f"\n  总品种数: {sentiment.total_symbols}")
    print(f"  上涨品种: {sentiment.up_symbols} ({sentiment.up_ratio*100:.1f}%)")
    print(f"  下跌品种: {sentiment.down_symbols}")
    print(f"  前20涨幅平均: {sentiment.top20_avg_gain*100:+.2f}%")
    print(f"  前20跌幅平均: {sentiment.top20_avg_loss*100:+.2f}%")

    bias_str = "多头" if sentiment.sentiment_bias == 'long' else \
              "空头" if sentiment.sentiment_bias == 'short' else "中性"
    print(f"\n  市场倾向: {bias_str}")
    print(f"  信心度: {sentiment.confidence:.2f}")

    # 选择方向
    side = strategy.select_position_side(sentiment)
    side_str = "做多" if side == PositionSide.LONG else "做空"
    print(f"\n  推荐方向: {side_str}")


# ============================================================================
# 示例5：开仓股票
# ============================================================================

def example5_enter_stock():
    """示例5：开仓股票（周波段）"""
    print("\n" + "="*80)
    print("示例5：开仓股票（周波段）")
    print("="*80)

    strategy = HybridSwingStrategy(initial_capital=1_000_000)

    market_data = generate_sample_market_data()
    current_date = datetime(2026, 5, 4)

    # 扫描股票机会
    stock_opportunities = strategy.scan_stock_opportunities(market_data, current_date)

    if stock_opportunities:
        # 选择最佳机会
        best_stock = stock_opportunities[0]

        # 开仓
        position = strategy.enter_stock_position(best_stock, current_date, 100_000)

        print(f"\n✅ 股票开仓成功")
        print(f"  代码: {position.symbol}")
        print(f"  方向: {'做多' if position.side == PositionSide.LONG else '做空'}")
        print(f"  入场价: ${position.entry_price:.2f}")
        print(f"  持仓: {position.shares}股")
        print(f"  市值: ${position.current_value:,.0f}")
        print(f"  止损: {position.stop_loss*100:.0f}%")
        print(f"  止盈: {position.take_profit*100:.0f}%")
        print(f"  最大天数: {position.max_hold_days}天")


# ============================================================================
# 示例6：开仓期货（远期合约）
# ============================================================================

def example6_enter_futures():
    """示例6：开仓期货（远期合约，严格风控）"""
    print("\n" + "="*80)
    print("示例6：开仓期货（远期合约，严格风控）")
    print("="*80)

    strategy = HybridSwingStrategy(initial_capital=1_000_000)
    current_date = datetime(2026, 5, 4)

    # 扫描期货合约
    futures_contracts = strategy.scan_futures_contracts(current_date)

    if futures_contracts:
        # 选择最佳合约
        best_contract = futures_contracts[0]

        # 选择方向（默认做多）
        side = PositionSide.LONG

        # 开仓
        position = strategy.enter_futures_position(best_contract, side, current_date, 100_000)

        print(f"\n✅ 期货开仓成功")
        print(f"  合约: {position.symbol} ({position.name})")
        print(f"  交割: {position.futures_contract.delivery_year}-{position.futures_contract.delivery_month:02d}")
        print(f"  方向: {'做多' if position.side == PositionSide.LONG else '做空'}")
        print(f"  入场价: ${position.entry_price:.2f}")
        print(f"  合约数: {position.contracts}手")
        print(f"  保证金: ${position.margin_used:,.0f}")
        print(f"  预留资金: ${position.reserve_capital:,.0f}")
        print(f"  风险度: {position.risk_level*100:.1f}%")
        print(f"  波动率: {position.futures_contract.volatility*100:.1f}%")


# ============================================================================
# 示例7：完整交易流程（T0-T5周期）
# ============================================================================

def example7_complete_trade_cycle():
    """示例7：完整交易流程（T0-T5周期，不过周末）"""
    print("\n" + "="*80)
    print("示例7：完整交易流程（T0-T5周期，不过周末）")
    print("="*80)

    strategy = HybridSwingStrategy(initial_capital=1_000_000)

    market_data = generate_sample_market_data()

    # T0：周一开仓
    print("\n【T0 - 周一】开仓")
    monday = datetime(2026, 5, 4, 9, 30)

    # 分析市场情绪
    sentiment = strategy.analyze_market_sentiment(market_data, monday)
    side = strategy.select_position_side(sentiment)

    bias_str = "多头" if sentiment.sentiment_bias == 'long' else "空头"
    print(f"  市场情绪: {bias_str} (信心度: {sentiment.confidence:.2f})")

    # 扫描机会
    stock_ops = strategy.scan_stock_opportunities(market_data, monday)
    futures_contracts = strategy.scan_futures_contracts(monday)

    # 选择期货开仓（波动更大）
    if futures_contracts:
        contract = futures_contracts[0]
        position = strategy.enter_futures_position(contract, side, monday, 100_000)
        print(f"\n  开仓期货: {position.symbol}")
        print(f"    保证金: ${position.margin_used:,.0f}")
        print(f"    风险度: {position.risk_level*100:.1f}%")

    # T1-T3：持仓
    for day in range(1, 4):
        trade_date = monday + timedelta(days=day)
        print(f"\n【T{day} - {trade_date.strftime('%A')}】持仓监控")

        if strategy.positions:
            for pos in strategy.positions:
                np.random.seed(hash(trade_date.strftime('%Y-%m-%d') + pos.symbol) % 10000)
                price_change = np.random.normal(0, 0.02)
                current_price = pos.entry_price * (1 + price_change)

                should_exit, reason = strategy.should_exit_position(pos, current_price)

                if should_exit:
                    strategy.exit_position(pos, trade_date, current_price, reason)
                    print(f"  平仓 {pos.symbol}: {reason}")
                    print(f"    盈亏: ${pos.pnl:+,.2f} ({pos.pnl_pct:+.2f}%)")
                else:
                    pnl_pct = (current_price / pos.entry_price - 1) * 100
                    print(f"  持仓 {pos.symbol}: ${current_price:.2f} ({pnl_pct:+.2f}%)")

    # T4：周五强制平仓
    friday = monday + timedelta(days=4)
    print(f"\n【T4 - 周五】强制平仓")

    for position in strategy.positions[:]:
        np.random.seed(hash(friday.strftime('%Y-%m-%d') + position.symbol) % 10000)
        price_change = np.random.normal(0, 0.01)
        exit_price = position.entry_price * (1 + price_change)
        strategy.exit_position(position, friday, exit_price, "周五收盘强制平仓")

        print(f"  平仓 {position.symbol}: ${position.pnl:+,.2f} ({position.pnl_pct:+.2f}%)")


# ============================================================================
# 示例8：完整4周模拟
# ============================================================================

def example8_four_week_simulation():
    """示例8：完整4周模拟（股票+期货）"""
    print("\n" + "="*80)
    print("示例8：完整4周模拟（股票+期货）")
    print("="*80)

    result = simulate_hybrid_strategy(
        initial_capital=1_000_000,
        weeks=4,
    )

    print(f"\n{'='*80}")
    print(f" " * 30 + "模拟总结")
    print(f"{'='*80}")
    print(f"\n总收益: ${result['total_profit']:,.0f}")
    print(f"收益率: {result['total_profit']/1_000_000*100:+.2f}%")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print(" " * 15 + "融合策略示例：期货+股票周波段交易系统")
    print("="*80)

    example1_create_strategy()
    example2_scan_stocks()
    example3_scan_futures()
    example4_market_sentiment()
    example5_enter_stock()
    example6_enter_futures()
    example7_complete_trade_cycle()
    example8_four_week_simulation()

    print("\n" + "="*80)
    print("✅ 所有示例运行完成！")
    print("="*80)
    print("\n💡 融合策略核心优势：")
    print("  1. 股票：长期看好标的，反复做波段")
    print("  2. 期货：远期高波动合约，严格风控")
    print("  3. 统一：T0-T5周期，不过周末")
    print("  4. 情绪：每日根据市场情绪选择方向")
    print("  5. 目标：每周净赚2万（2%）")
    print("\n")


if __name__ == "__main__":
    main()
