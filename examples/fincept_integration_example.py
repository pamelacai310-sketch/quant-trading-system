"""
FinceptTerminal 集成示例

演示如何使用FinceptTerminal集成桥接模块
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from datetime import datetime, timedelta
from quant_trade_system.integrations import (
    FinceptConfig,
    FinceptDataBridge,
    FinceptStrategyExporter,
    FinceptSignalPusher,
    FinceptIntegrator,
    create_fincept_integration,
)


def example_1_check_integration_status():
    """示例1: 检查FinceptTerminal集成状态"""

    print("\n" + "="*80)
    print("示例1: 检查FinceptTerminal集成状态")
    print("="*80 + "\n")

    # 创建集成实例
    integrator = create_fincept_integration()

    # 检查集成状态
    status = integrator.check_integration_status()

    print("FinceptTerminal集成状态:\n")
    print(f"  已安装: {status['fincept_installed']}")
    print(f"  安装路径: {status['fincept_path'] or '未配置'}")
    print(f"  Python可用: {status['python_available']}")
    print(f"  可用连接器: {', '.join(status['available_connectors'])}")
    print(f"  API启用: {status['api_enabled']}")
    print(f"  支持的导出格式: {', '.join(status['supported_formats'])}")


def example_2_fetch_market_data():
    """示例2: 获取市场数据"""

    print("\n" + "="*80)
    print("示例2: 从FinceptTerminal获取市场数据")
    print("="*80 + "\n")

    integrator = create_fincept_integration()

    # 获取股票数据
    print("获取AAPL历史数据:")
    data = integrator.data_bridge.fetch_market_data(
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    print(f"\n数据行数: {len(data)}")
    print(f"日期范围: {data.index[0]} 到 {data.index[-1]}")
    print(f"列名: {list(data.columns)}")
    print("\n前5行数据:")
    print(data.head())

    # 获取经济数据
    print("\n获取GDP数据:")
    try:
        gdp_data = integrator.data_bridge.fetch_economic_data(
            indicator="GDP",
            start_date="2020-01-01",
            end_date="2024-12-31",
        )
        print(f"GDP数据点数: {len(gdp_data)}")
    except Exception as e:
        print(f"获取GDP数据失败: {e}")


def example_3_export_strategy_to_json():
    """示例3: 导出策略到Fincept JSON格式"""

    print("\n" + "="*80)
    print("示例3: 导出策略到Fincept JSON格式")
    print("="*80 + "\n")

    integrator = create_fincept_integration()

    # 定义策略
    strategy = {
        "name": "Dual Moving Average Crossover",
        "description": "双均线交叉策略",
        "dataset": "yahoo_finance",
        "symbol": "AAPL",
        "indicators": [
            {"name": "fast_ma", "type": "sma", "window": 12},
            {"name": "slow_ma", "type": "sma", "window": 26},
        ],
        "entry_rules": [
            {"left": "fast_ma", "op": "crosses_above", "right": "slow_ma"},
        ],
        "exit_rules": [
            {"left": "fast_ma", "op": "crosses_below", "right": "slow_ma"},
        ],
        "position_sizing": {
            "mode": "fixed_fraction",
            "risk_fraction": 0.1,
        },
        "risk_limits": {
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.12,
        },
    }

    # 导出为JSON格式
    exporter = FinceptStrategyExporter()
    json_output = exporter.export_strategy(strategy, "fincept_json")

    print("Fincept JSON格式:")
    print(json_output)


def example_4_export_strategy_to_python():
    """示例4: 导出策略到Fincept Python脚本"""

    print("\n" + "="*80)
    print("示例4: 导出策略到Fincept Python脚本")
    print("="*80 + "\n")

    integrator = create_fincept_integration()

    # 定义RSI策略
    strategy = {
        "name": "RSI Mean Reversion",
        "description": "RSI均值回归策略",
        "dataset": "yahoo_finance",
        "symbol": "SPY",
        "indicators": [
            {"name": "rsi", "type": "rsi", "window": 14},
        ],
        "entry_rules": [
            {"left": "rsi", "op": "<", "right": "30"},
        ],
        "exit_rules": [
            {"left": "rsi", "op": ">", "right": "70"},
        ],
        "position_sizing": {
            "mode": "fixed_units",
            "units": 100,
        },
        "risk_limits": {
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.08,
        },
    }

    # 导出为Python脚本
    exporter = FinceptStrategyExporter()
    python_output = exporter.export_strategy(strategy, "fincept_python")

    print("Fincept Python脚本:")
    print(python_output)


def example_5_export_strategy_to_workflow():
    """示例5: 导出策略到Fincept工作流"""

    print("\n" + "="*80)
    print("示例5: 导出策略到Fincept可视化工作流")
    print("="*80 + "\n")

    integrator = create_fincept_integration()

    # 定义策略
    strategy = {
        "name": "MACD Strategy",
        "description": "MACD趋势跟踪策略",
        "dataset": "yahoo_finance",
        "symbol": "TSLA",
        "indicators": [
            {"name": "macd", "type": "ema", "window": 12},
            {"name": "signal", "type": "ema", "window": 26},
        ],
        "entry_rules": [
            {"left": "macd", "op": "crosses_above", "right": "signal"},
        ],
        "exit_rules": [
            {"left": "macd", "op": "crosses_below", "right": "signal"},
        ],
        "position_sizing": {
            "mode": "volatility_target",
            "target_volatility": 0.15,
        },
        "risk_limits": {
            "stop_loss_pct": 0.06,
            "take_profit_pct": 0.18,
        },
    }

    # 导出为工作流
    exporter = FinceptStrategyExporter()
    workflow_output = exporter.export_strategy(strategy, "fincept_workflow")

    print("Fincept工作流定义:")
    print(workflow_output)


def example_6_export_and_save():
    """示例6: 导出策略并保存到文件"""

    print("\n" + "="*80)
    print("示例6: 导出策略并保存到文件")
    print("="*80 + "\n")

    integrator = create_fincept_integration()

    # 定义欧奈尔CANSLIM策略
    strategy = {
        "name": "O'Neil CANSLIM",
        "description": "欧奈尔CANSLIM选股策略",
        "dataset": "yahoo_finance",
        "symbol": "NFLX",
        "indicators": [
            {"name": "eps_growth", "type": "custom", "window": 1},
            {"name": "rs_rating", "type": "custom", "window": 1},
        ],
        "entry_rules": [
            {"left": "eps_growth", "op": ">=", "right": "25"},
            {"left": "rs_rating", "op": ">=", "right": "80"},
        ],
        "exit_rules": [
            {"left": "rs_rating", "op": "<", "right": "70"},
        ],
        "position_sizing": {
            "mode": "fixed_fraction",
            "risk_fraction": 0.02,
        },
        "risk_limits": {
            "stop_loss_pct": 0.08,
            "take_profit_pct": 0.20,
        },
    }

    # 导出并保存
    result = integrator.export_strategy_to_fincept(
        strategy,
        output_file="fincept_exports/oneill_canslim",
    )

    print(result)


def example_7_push_trading_signals():
    """示例7: 推送交易信号"""

    print("\n" + "="*80)
    print("示例7: 推送交易信号到FinceptTerminal")
    print("="*80 + "\n")

    # 创建信号推送器（API模式关闭，使用文件模式）
    config = FinceptConfig(api_enabled=False)
    pusher = FinceptSignalPusher(config)

    # 推送买入信号
    print("推送买入信号:")
    success = pusher.push_signal(
        symbol="AAPL",
        action="buy",
        quantity=100,
        price=150.25,
        reason="双均线金叉",
    )
    print(f"  推送结果: {'成功' if success else '失败'}")

    # 推送卖出信号
    print("\n推送卖出信号:")
    success = pusher.push_signal(
        symbol="MSFT",
        action="sell",
        quantity=50,
        price=380.50,
        reason="止盈 - 达到目标价",
    )
    print(f"  推送结果: {'成功' if success else '失败'}")

    print(f"\n信号队列中共有 {len(pusher.signal_queue)} 条信号")


def example_8_sync_multiple_strategies():
    """示例8: 批量同步多个策略"""

    print("\n" + "="*80)
    print("示例8: 批量同步多个策略到FinceptTerminal")
    print("="*80 + "\n")

    integrator = create_fincept_integration()

    # 定义多个策略
    strategies = [
        {
            "name": "Strategy 1: SMA Crossover",
            "description": "简单均线交叉",
            "dataset": "yahoo_finance",
            "symbol": "AAPL",
            "indicators": [
                {"name": "fast_ma", "type": "sma", "window": 10},
                {"name": "slow_ma", "type": "sma", "window": 30},
            ],
            "entry_rules": [
                {"left": "fast_ma", "op": "crosses_above", "right": "slow_ma"},
            ],
            "exit_rules": [
                {"left": "fast_ma", "op": "crosses_below", "right": "slow_ma"},
            ],
        },
        {
            "name": "Strategy 2: RSI Reversal",
            "description": "RSI反转策略",
            "dataset": "yahoo_finance",
            "symbol": "SPY",
            "indicators": [
                {"name": "rsi", "type": "rsi", "window": 14},
            ],
            "entry_rules": [
                {"left": "rsi", "op": "<", "right": "30"},
            ],
            "exit_rules": [
                {"left": "rsi", "op": ">", "right": "70"},
            ],
        },
        {
            "name": "Strategy 3: Volatility Breakout",
            "description": "波动率突破",
            "dataset": "yahoo_finance",
            "symbol": "TSLA",
            "indicators": [
                {"name": "bb_upper", "type": "bollinger", "window": 20},
                {"name": "bb_lower", "type": "bollinger", "window": 20},
            ],
            "entry_rules": [
                {"left": "close", "op": "crosses_above", "right": "bb_upper"},
            ],
            "exit_rules": [
                {"left": "close", "op": "crosses_below", "right": "bb_lower"},
            ],
        },
    ]

    # 批量同步
    results = integrator.sync_strategies(strategies)

    print("批量同步结果:\n")
    for strategy_id, result in results.items():
        print(f"{strategy_id}:")
        print(f"  {result}\n")


def example_9_custom_configuration():
    """示例9: 自定义配置"""

    print("\n" + "="*80)
    print("示例9: 使用自定义FinceptTerminal配置")
    print("="*80 + "\n")

    # 自定义配置
    config = FinceptConfig(
        fincept_path="/path/to/fincept",  # 替换为实际路径
        python_executable="/usr/bin/python3.11",
        api_enabled=True,
        api_port=8080,
        data_connectors=[
            "yahoo_finance",
            "polygon",
            "fred",
            "dbnomics",
        ],
    )

    # 创建自定义集成实例
    integrator = FinceptIntegrator(config)

    # 检查状态
    status = integrator.check_integration_status()

    print("自定义配置状态:\n")
    print(f"  Fincept路径: {status['fincept_path']}")
    print(f"  Python路径: {config.python_executable}")
    print(f"  API端口: {status['api_port']}")
    print(f"  数据连接器: {', '.join(config.data_connectors)}")


def example_10_complete_workflow():
    """示例10: 完整工作流演示"""

    print("\n" + "="*80)
    print("示例10: 完整的FinceptTerminal集成工作流")
    print("="*80 + "\n")

    print("步骤1: 创建集成实例并检查状态")
    integrator = create_fincept_integration()
    status = integrator.check_integration_status()
    print(f"  可用连接器: {', '.join(status['available_connectors'])}\n")

    print("步骤2: 获取市场数据")
    data = integrator.data_bridge.fetch_market_data(
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2024-03-31",
    )
    print(f"  获取到 {len(data)} 条数据\n")

    print("步骤3: 定义并导出策略")
    strategy = {
        "name": "AAPL Momentum",
        "description": "苹果动量策略",
        "dataset": "yahoo_finance",
        "symbol": "AAPL",
        "indicators": [
            {"name": "momentum", "type": "custom", "window": 20},
        ],
        "entry_rules": [
            {"left": "momentum", "op": ">", "right": "0"},
        ],
        "exit_rules": [
            {"left": "momentum", "op": "<", "right": "0"},
        ],
    }

    export_result = integrator.export_strategy_to_fincept(strategy)
    print(f"  导出结果: {export_result}\n")

    print("步骤4: 生成交易信号")
    signal_success = integrator.signal_pusher.push_signal(
        symbol="AAPL",
        action="buy",
        quantity=50,
        price=data['Close'].iloc[-1],
        reason="动量突破",
    )
    print(f"  信号推送: {'成功' if signal_success else '失败'}\n")

    print("✅ 完整工作流演示完成！")


def main():
    """主函数"""
    print("\n" + "="*80)
    print(" " * 20 + "FinceptTerminal 集成完整示例")
    print("="*80)

    # 运行所有示例
    example_1_check_integration_status()
    example_2_fetch_market_data()
    example_3_export_strategy_to_json()
    example_4_export_strategy_to_python()
    example_5_export_strategy_to_workflow()
    example_6_export_and_save()
    example_7_push_trading_signals()
    example_8_sync_multiple_strategies()
    example_9_custom_configuration()
    example_10_complete_workflow()

    print("\n" + "="*80)
    print(" " * 25 + "所有示例运行完成！")
    print("="*80 + "\n")

    print("📚 FinceptTerminal 集成功能总结:")
    print("-" * 80)
    print("\n1. 数据桥接:")
    print("   - 支持100+数据源（Yahoo Finance, FRED, Polygon等）")
    print("   - 自动回退机制（Fincept不可用时使用yfinance等）")
    print("   - 支持股票数据和宏观经济数据")

    print("\n2. 策略导出:")
    print("   - JSON格式：Fincept配置文件")
    print("   - Python格式：可执行的Python脚本")
    print("   - 工作流格式：可视化流程图定义")

    print("\n3. 信号推送:")
    print("   - API推送：直接推送到Fincept API")
    print("   - 文件推送：保存为JSON文件供导入")
    print("   - 支持批量信号推送")

    print("\n4. 集成优势:")
    print("   - 可选依赖：未安装Fincept时自动降级")
    print("   - 无缝集成：桥接层透明处理")
    print("   - 扩展性强：支持自定义配置")

    print("\n💡 实际应用:")
    print("   - 使用Fincept丰富数据源增强回测")
    print("   - 导出策略到Fincept进行可视化编辑")
    print("   - 推送信号到Fincept执行实盘交易")
    print("   - 结合两套系统优势，实现最佳效果\n")


if __name__ == "__main__":
    main()
