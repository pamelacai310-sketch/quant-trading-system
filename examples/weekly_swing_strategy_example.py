"""
周波段T0-T5短线策略示例

演示如何使用周波段策略实现：
1. T0-T5持仓周期，不过周末
2. 长期看好标的反复做波段
3. 每周净赚2万目标（本金100万，2%收益率）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from quant_trade_system.strategies import (
    WeeklySwingStrategy,
    SwingPosition,
    PositionSide,
)


def example_1_basic_usage():
    """示例1: 基本使用"""
    print("\n" + "="*80)
    print("示例1: 周波段策略基本使用")
    print("="*80 + "\n")

    # 创建策略实例
    strategy = WeeklySwingStrategy(
        initial_capital=1_000_000,  # 本金100万
        weekly_target=20_000,        # 每周目标2万
        max_positions=5,             # 最大5只持仓
        max_hold_days=5,             # 最长持仓5天
        base_stop_loss=0.03,         # 3%止损
        base_take_profit=0.06,       # 6%止盈
    )

    print("策略参数:")
    print(f"  初始资金: ${strategy.initial_capital:,.0f}")
    print(f"  每周目标: ${strategy.weekly_target:,.0f} ({strategy.target_return_pct*100:.0f}%)")
    print(f"  最大持仓: {strategy.max_positions}只")
    print(f"  最大天数: {strategy.max_hold_days}天")
    print(f"  止损: {strategy.base_stop_loss*100:.0f}%")
    print(f"  止盈: {strategy.base_take_profit*100:.0f}%")

    print(f"\n长期看好标的（可反复做波段）:")
    for symbol in strategy.long_term_favorites[:5]:
        print(f"  - {symbol}")

    print(f"\n期货合约池:")
    for symbol in strategy.futures_contracts[:5]:
        print(f"  - {symbol}")


def example_2_position_lifecycle():
    """示例2: 持仓生命周期"""
    print("\n" + "="*80)
    print("示例2: 持仓生命周期管理")
    print("="*80 + "\n")

    strategy = WeeklySwingStrategy()

    # 模拟T0开仓
    entry_date = datetime(2026, 5, 5, 9, 30)  # 周一早上

    # 检查是否可以开仓
    can_enter = strategy.can_enter_position(entry_date)
    print(f"周一早上可以开仓: {can_enter}")

    # 创建持仓
    position = SwingPosition(
        symbol='AAPL',
        name='苹果',
        side=PositionSide.LONG,
        entry_date=entry_date,
        entry_price=150.0,
        shares=100,
        stop_loss=0.03,
        take_profit=0.06,
        max_hold_days=5,
        is_long_term_favor=True,
    )

    print(f"\n开仓信息:")
    print(f"  标的: {position.symbol}")
    print(f"  方向: {'做多' if position.side == PositionSide.LONG else '做空'}")
    print(f"  入场价: ${position.entry_price}")
    print(f"  持仓数量: {position.shares}股")
    print(f"  止损价: ${position.entry_price * (1 - position.stop_loss):.2f}")
    print(f"  止盈价: ${position.entry_price * (1 + position.take_profit):.2f}")
    print(f"  长期看好: {position.is_long_term_favor}")

    # 模拟不同日期的持仓状态
    print(f"\n持仓周期状态:")
    for day in range(6):
        current_date = entry_date + timedelta(days=day)
        days_to_friday = strategy.get_days_until_friday(current_date)

        print(f"  T{day} ({current_date.strftime('%Y-%m-%d %A')}): "
              f"持仓{day}天, 距周五{days_to_friday}天")

    # 检查平仓条件
    print(f"\n平仓条件检查:")

    # 场景1: 触发止损
    current_date = entry_date + timedelta(days=1)
    current_price = 145.0  # 跌破3%止损
    should_exit, reason = strategy.should_exit_position(position, current_date, current_price)
    print(f"  场景1 - 价格${current_price} (T1): {should_exit}, {reason}")

    # 场景2: 触发止盈
    current_price = 159.0  # 涨破6%止盈
    should_exit, reason = strategy.should_exit_position(position, current_date, current_price)
    print(f"  场景2 - 价格${current_price} (T1): {should_exit}, {reason}")

    # 场景3: 达到最大持仓天数
    current_date = entry_date + timedelta(days=5)
    current_price = 152.0
    should_exit, reason = strategy.should_exit_position(position, current_date, current_price)
    print(f"  场景3 - T5周五: {should_exit}, {reason}")


def example_3_weekly_reentry():
    """示例3: 每周重新进入长期看好标的"""
    print("\n" + "="*80)
    print("示例3: 长期看好标的的每周波段操作")
    print("="*80 + "\n")

    strategy = WeeklySwingStrategy()

    # 长期看好的标的
    favorite_symbol = 'AAPL'

    print(f"长期看好标的: {favorite_symbol}")
    print(f"\n每周波段操作流程:")

    # 模拟4周操作
    for week in range(1, 5):
        week_start = datetime(2026, 5, 1) + timedelta(weeks=week-1)
        week_end = week_start + timedelta(days=4)

        print(f"\n第{week}周 ({week_start.strftime('%m/%d')} - {week_end.strftime('%m/%d')}):")

        # 周一：在相对低位开仓
        entry_date = week_start
        entry_price = 150.0 + np.random.uniform(-5, 0)  # 模拟低位价格

        print(f"  T0 周一: 开仓 @ ${entry_price:.2f} (低位)")

        # 持仓3-5天后平仓
        hold_days = np.random.randint(3, 6)
        exit_date = entry_date + timedelta(days=hold_days)
        exit_price = entry_price * (1 + np.random.uniform(-0.02, 0.08))

        # 检查是否应该在周五前平仓
        if exit_date.weekday() > 4:  # 周末
            exit_date = week_start + timedelta(days=4)  # 周五
            exit_price = entry_price * (1 + np.random.uniform(-0.01, 0.03))

        profit = (exit_price - entry_price) * 100
        profit_pct = (exit_price / entry_price - 1) * 100

        print(f"  T{hold_days} {exit_date.strftime('%A')}: 平仓 @ ${exit_price:.2f}")
        print(f"    盈亏: ${profit:+,.2f} ({profit_pct:+.2f}%)")

        # 周五平仓，等待下周一重新进入
        if exit_date.weekday() == 4:  # 周五
            print(f"    ✅ 周五已平仓，不过周末")
            print(f"    📅 等待下周{favorite_symbol}回调后再接回")


def example_4_risk_management():
    """示例4: 风险管理与目标达成"""
    print("\n" + "="*80)
    print("示例4: 风险管理与每周2万目标")
    print("="*80 + "\n")

    # 本金100万，每周目标2万
    initial_capital = 1_000_000
    weekly_target = 20_000
    target_return_pct = 0.02  # 2%

    print(f"资金管理:")
    print(f"  初始本金: ${initial_capital:,.0f}")
    print(f"  每周目标: ${weekly_target:,.0f}")
    print(f"  目标收益率: {target_return_pct*100:.0f}%")

    # 计算单笔交易风险
    risk_per_trade = 0.02  # 单笔风险2%
    max_positions = 5
    stop_loss_pct = 0.03  # 3%止损

    print(f"\n风险控制:")
    print(f"  单笔风险: {risk_per_trade*100:.0f}%")
    print(f"  最大持仓: {max_positions}只")
    print(f"  止损幅度: {stop_loss_pct*100:.0f}%")

    # 计算需要多少笔盈利交易达成目标
    avg_win_pct = 0.05  # 平均盈利5%
    trades_needed = weekly_target / (initial_capital * avg_win_pct)

    print(f"\n目标达成分析:")
    print(f"  假设平均盈利: {avg_win_pct*100:.0f}%")
    print(f"  单笔盈利: ${initial_capital * avg_win_pct:,.0f}")
    print(f"  需要盈利笔数: {trades_needed:.1f}笔")

    # 模拟一周交易
    print(f"\n模拟一周交易场景:")

    week_trades = [
        {'symbol': 'AAPL', 'entry': 150, 'exit': 157.5, 'result': '+5%'},
        {'symbol': 'TSLA', 'entry': 200, 'exit': 210, 'result': '+5%'},
        {'symbol': 'NVDA', 'entry': 450, 'exit': 427.5, 'result': '-5% (止损)'},
        {'symbol': '0700.HK', 'entry': 380, 'exit': 399, 'result': '+5%'},
        {'symbol': 'MSFT', 'entry': 300, 'exit': 315, 'result': '+5%'},
    ]

    total_profit = 0
    capital_per_trade = initial_capital / 5  # 每笔20万

    for i, trade in enumerate(week_trades, 1):
        if '止损' in trade['result']:
            profit = capital_per_trade * -0.03
        else:
            profit = capital_per_trade * 0.05

        total_profit += profit

        print(f"  交易{i}: {trade['symbol']} {trade['entry']:.0f}→{trade['exit']:.0f} "
              f"({trade['result']}) = ${profit:+,.0f}")

    print(f"\n本周汇总:")
    print(f"  总盈亏: ${total_profit:+,.0f}")
    print(f"  目标达成: {'✅' if total_profit >= weekly_target else '❌'}")
    print(f"  达成率: {total_profit/weekly_target*100:.1f}%")


def example_5_complete_simulation():
    """示例5: 完整4周模拟"""
    print("\n" + "="*80)
    print("示例5: 完整4周交易模拟")
    print("="*80 + "\n")

    from quant_trade_system.strategies import simulate_weekly_swing_strategy

    # 运行模拟
    result = simulate_weekly_swing_strategy(
        initial_capital=1_000_000,
        weeks=4,
    )

    # 显示最终结果
    print(f"\n{'='*80}")
    print(f"4周模拟总结")
    print(f"{'='*80}")
    print(f"\n总收益: ${result['total_profit']:,.0f}")
    print(f"总收益率: {result['total_return']*100:+.2f}%")
    print(f"平均周收益: ${result['total_profit']/4:,.0f}")
    print(f"目标达成率: {result['total_profit']/80_000*100:.1f}% (4周目标8万)")

    # 分析每周表现
    print(f"\n每周表现:")
    for i, week_perf in enumerate(result['weekly_results'], 1):
        achieved = '✅' if week_perf['target_achieved'] else '❌'
        print(f"  第{i}周: ${week_perf['realized_profit']:,.0f} "
              f"({week_perf['return_pct']*100:+.2f}%) {achieved}")


def example_6_futures_trading():
    """示例6: 期货合约交易"""
    print("\n" + "="*80)
    print("示例6: 期货合约T0-T5交易")
    print("="*80 + "\n")

    strategy = WeeklySwingStrategy()

    print("期货合约特点:")
    print(f"  高杠杆: 10-20倍")
    print(f"  高波动: 日内波动可达3-5%")
    print(f"  T+0: 可以当日平仓")
    print(f"  双向: 可做多可做空")

    print(f"\n期货合约池:")
    for contract in strategy.futures_contracts:
        print(f"  - {contract}")

    print(f"\n期货交易示例:")
    print(f"\n做多标普500 E-mini (ES):")
    print(f"  合口规模: $50 × 指数点位")
    print(f"  入场: 4200点")
    print(f"  止损: 3990点 (-5%)")
    print(f"  止盈: 4410点 (+5%)")
    print(f"  单手盈亏: ${(4410-4200)*50:+,.0f}")

    print(f"\n做空原油 (CL):")
    print(f"  合口规模: 1000桶")
    print(f"  入场: $80/桶")
    print(f"  止损: $84/桶 (+5%)")
    print(f"  止盈: $76/桶 (-5%)")
    print(f"  单手盈亏: ${(76-80)*1000:+,.0f}")


def example_7_practical_tips():
    """示例7: 实战技巧"""
    print("\n" + "="*80)
    print("示例7: 周波段实战技巧")
    print("="*80 + "\n")

    tips = [
        {
            'category': '选股',
            'tips': [
                '选择波动率高（日均波动>3%）的标的',
                '优先选择流动性好的大盘股',
                '长期看好标的可在每周回调时接回',
                '避免选择即将发布财报的股票（不确定性大）',
            ]
        },
        {
            'category': '择时',
            'tips': [
                '周一开盘急跌是建仓良机',
                '周中（周二-周四）持仓时间最灵活',
                '周五下午2点后必须考虑平仓',
                'T0-T3天最佳，T4-T5天谨慎',
            ]
        },
        {
            'category': '风控',
            'tips': [
                '严格执行止损，单笔亏损不超过2%',
                '止盈可以分批：一半在+5%，一半在+8%',
                '单只股票仓位不超过总资金的20%',
                '每周最多5只持仓，分散风险',
            ]
        },
        {
            'category': '心态',
            'tips': [
                '接受每周会有1-2笔止损',
                '目标达成后可以提前休息，不贪心',
                '未达成目标时不要加大仓位追',
                '记录每笔交易，总结经验',
            ]
        },
    ]

    for item in tips:
        print(f"\n{item['category']}:")
        for i, tip in enumerate(item['tips'], 1):
            print(f"  {i}. {tip}")

    print(f"\n典型一周交易计划:")
    print(f"\n  周一:")
    print(f"    - 9:30 观察开盘，寻找急跌机会")
    print(f"    - 10:30 如果有低位，开仓1-2只")
    print(f"    - 14:00 检查持仓，设置止损")

    print(f"\n  周二:")
    print(f"    - 9:30 检查持仓，考虑加仓")
    print(f"    - 10:00 扫描新的机会")
    print(f"    - 15:00 评估是否部分止盈")

    print(f"\n  周三-周四:")
    print(f"    - 每日检查止损止盈")
    print(f"    - 达到止盈目标部分平仓")
    print(f"    - 保留优质仓位继续持有")

    print(f"\n  周五:")
    print(f"    - 上午: 最后的交易机会")
    print(f"    - 14:00: 开始准备平仓")
    print(f"    - 15:00: 所有持仓必须平仓")
    print(f"    - 15:30: 总结本周交易")
    print(f"    - 周末: 休息，准备下周一")


def main():
    """主函数"""
    print("\n" + "="*80)
    print(" " * 20 + "周波段T0-T5短线策略 - 完整示例")
    print("="*80)

    # 运行所有示例
    example_1_basic_usage()
    example_2_position_lifecycle()
    example_3_weekly_reentry()
    example_4_risk_management()
    example_5_complete_simulation()
    example_6_futures_trading()
    example_7_practical_tips()

    print("\n" + "="*80)
    print(" " * 25 + "所有示例运行完成！")
    print("="*80)

    print("\n📊 策略核心要点总结:")
    print("-" * 80)
    print("\n1. 持仓周期: T0-T5，最晚周五平仓")
    print("2. 长期看好标的: 每周回调时接回，反复做波段")
    print("3. 目标收益: 每周2万（本金100万，2%收益率）")
    print("4. 风险控制: 单笔止损3%，止盈6%")
    print("5. 最大持仓: 5只")

    print("\n💡 成功关键:")
    print("   - 严格纪律，不持过周末")
    print("   - 优选波动率高、流动性好的标的")
    print("   - 长期看好标的耐心等待回调")
    print("   - 达成目标后不贪心，提前休息")
    print("   - 接受止损，不报复性交易")


if __name__ == "__main__":
    main()
