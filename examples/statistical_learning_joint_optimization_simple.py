"""
统计学习模型层和联合优化层简化示例

演示API和架构，不依赖可选库
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from quant_trade_system.core.statistical_learning_layer import (
    ModelType,
    OptimizationMetric,
    LearningObjectiveConfig,
)
from quant_trade_system.core.joint_optimization_layer import (
    OptimizationStrategy,
    PortfolioConstraint,
    TalebBarbellConfig,
    JointOptimizationLayer,
    create_joint_optimization_layer,
)


def example_1_learning_objective_config():
    """示例1: 学习目标配置"""
    print("\n" + "="*80)
    print("示例1: 学习目标配置")
    print("="*80)

    # 创建学习目标配置
    config = LearningObjectiveConfig(
        primary_metric=OptimizationMetric.WIN_RATE,
        secondary_metrics=[
            OptimizationMetric.ODDS_RATIO,
            OptimizationMetric.ELASTICITY,
        ],
        metric_weights={
            OptimizationMetric.WIN_RATE: 0.4,
            OptimizationMetric.ODDS_RATIO: 0.3,
            OptimizationMetric.ELASTICITY: 0.3,
        },
        min_win_rate=0.55,
        min_odds_ratio=1.5,
        min_elasticity=1.2,
        target_sharpe=1.5,
        max_drawdown_threshold=0.15,
    )

    print("✅ 学习目标配置创建成功")
    print(f"  主要指标: {config.primary_metric.value}")
    print(f"  次要指标: {[m.value for m in config.secondary_metrics]}")
    print(f"  指标权重:")
    for metric, weight in config.metric_weights.items():
        print(f"    {metric.value}: {weight}")

    print(f"\n阈值设置:")
    print(f"  最小胜率: {config.min_win_rate:.2%}")
    print(f"  最小赔率: {config.min_odds_ratio:.2f}")
    print(f"  最小弹性: {config.min_elasticity:.2f}")
    print(f"  目标夏普: {config.target_sharpe:.2f}")
    print(f"  最大回撤: {config.max_drawdown_threshold:.2%}")


def example_2_optimization_metrics():
    """示例2: 优化指标"""
    print("\n" + "="*80)
    print("示例2: 优化指标")
    print("="*80)

    print("支持的优化指标:")
    for metric in OptimizationMetric:
        print(f"  - {metric.value}")

    print("\n指标说明:")
    print(f"  WIN_RATE: 胜率 - 相对盈利次数占总交易次数的比例")
    print(f"  ODDS_RATIO: 赔率 - 平均盈利金额与平均亏损金额的比值")
    print(f"  ELASTICITY: 弹性 - 收益变化幅度与基准变化幅度的比值")
    print(f"  SHARPE_RATIO: 夏普比率 - 风险调整后收益")
    print(f"  MAX_DRAWDOWN: 最大回撤 - 历史最大亏损幅度")
    print(f"  CVAR: 条件风险价值 - 尾部风险度量")


def example_3_portfolio_constraints():
    """示例3: 组合约束配置"""
    print("\n" + "="*80)
    print("示例3: 组合约束配置")
    print("="*80)

    # 创建组合约束
    constraints = PortfolioConstraint(
        max_position_size=0.3,
        max_total_exposure=1.0,
        min_cash_ratio=0.05,
        max_sector_exposure=0.5,
        turnover_limit=0.5,
        beta_range=(0.8, 1.2),
    )

    print("✅ 组合约束配置创建成功")
    print(f"  单个资产最大仓位: {constraints.max_position_size:.2%}")
    print(f"  最大总敞口: {constraints.max_total_exposure:.2%}")
    print(f"  最小现金比例: {constraints.min_cash_ratio:.2%}")
    print(f"  单个行业最大敞口: {constraints.max_sector_exposure:.2%}")
    print(f"  换手率限制: {constraints.turnover_limit:.2%}")
    print(f"  Beta范围: {constraints.beta_range[0]:.2f} - {constraints.beta_range[1]:.2f}")


def example_4_taleb_barbell_config():
    """示例4: 塔勒布杠铃配置"""
    print("\n" + "="*80)
    print("示例4: 塔勒布杠铃配置")
    print("="*80)

    # 创建塔勒布杠铃配置
    config = TalebBarbellConfig(
        safe_asset_ratio=0.85,
        risky_asset_ratio=0.15,
        safe_asset_types=[
            'treasury_bonds',
            'high_grade_corporate_bonds',
            'cash_equivalents',
        ],
        risky_asset_types=[
            'growth_stocks',
            'emerging_markets',
            'commodities',
        ],
        min_momentum=0.5,
        max_correlation_with_safe=0.3,
        min_asymmetric_return=1.5,
    )

    print("✅ 塔勒布杠铃配置创建成功")
    print(f"  安全资产比例: {config.safe_asset_ratio:.2%}")
    print(f"  风险资产比例: {config.risky_asset_ratio:.2%}")

    print(f"\n  安全资产类型:")
    for asset_type in config.safe_asset_types:
        print(f"    - {asset_type}")

    print(f"\n  风险资产类型:")
    for asset_type in config.risky_asset_types:
        print(f"    - {asset_type}")

    print(f"\n  风险资产选择标准:")
    print(f"    最小动量得分: {config.min_momentum}")
    print(f"    与安全资产最大相关性: {config.max_correlation_with_safe}")
    print(f"    最小非对称收益: {config.min_asymmetric_return}")


def example_5_joint_optimization_strategies():
    """示例5: 联合优化策略"""
    print("\n" + "="*80)
    print("示例5: 联合优化策略")
    print("="*80)

    print("支持的优化策略:")
    for strategy in OptimizationStrategy:
        print(f"  - {strategy.value}")

    print("\n策略说明:")
    print(f"  MEAN_VARIANCE: 均值-方差优化（马科维茨）")
    print(f"  RISK_PARITY: 风险平价（等风险贡献）")
    print(f"  TALEB_BARBELL: 塔勒布杠铃（85%安全 + 15%风险）")
    print(f"  EQUAL_WEIGHT: 等权重（1/N）")
    print(f"  MAX_DIVERSIFICATION: 最大化分散化")
    print(f"  MAX_SHARPE: 最大化夏普比率")
    print(f"  MIN_CVAR: 最小化条件风险价值")


def example_6_joint_optimization_layer():
    """示例6: 联合优化层"""
    print("\n" + "="*80)
    print("示例6: 联合优化层")
    print("="*80)

    # 创建组合约束
    constraints = PortfolioConstraint(
        max_position_size=0.3,
        max_total_exposure=1.0,
    )

    # 创建塔勒布杠铃配置
    taleb_config = TalebBarbellConfig(
        safe_asset_ratio=0.85,
        risky_asset_ratio=0.15,
    )

    # 创建联合优化层（塔勒布杠铃策略）
    layer = create_joint_optimization_layer(
        optimization_strategy=OptimizationStrategy.TALEB_BARBELL,
        portfolio_constraints=constraints,
        taleb_config=taleb_config,
    )

    print("✅ 联合优化层创建成功")
    print(f"  优化策略: {layer.optimization_strategy.value}")
    print(f"  组合约束: 已配置")
    print(f"  塔勒布杠铃配置: 已配置")

    # 模拟数据
    np.random.seed(42)
    n_assets = 10

    # 预期收益
    expected_returns = np.random.randn(n_assets) * 0.01

    # 协方差矩阵
    cov_matrix = np.random.randn(n_assets, n_assets)
    cov_matrix = cov_matrix @ cov_matrix.T * 0.0001

    # 资产类型
    asset_types = (
        ['treasury_bonds', 'high_grade_corporate_bonds', 'cash_equivalents'] +
        ['growth_stocks'] * 5 +
        ['emerging_markets'] * 2
    )[:n_assets]

    print(f"\n模拟数据:")
    print(f"  资产数量: {n_assets}")
    print(f"  预期收益范围: {expected_returns.min():.4f} - {expected_returns.max():.4f}")
    print(f"  协方差矩阵形状: {cov_matrix.shape}")

    # 优化组合
    print(f"\n开始优化...")
    result = layer.optimize_portfolio(
        expected_returns=expected_returns,
        covariance_matrix=cov_matrix,
        asset_types=asset_types,
    )

    print(f"\n✅ 优化完成")
    print(f"  预期收益: {result.expected_return:.4f}")
    print(f"  预期风险: {result.expected_risk:.4f}")
    print(f"  夏普比率: {result.sharpe_ratio:.4f}")
    print(f"  收敛状态: {result.convergence_status}")

    print(f"\n安全资产权重:")
    for asset, weight in result.safe_asset_weights.items():
        if weight > 0.001:
            print(f"  {asset}: {weight:.2%}")

    print(f"\n风险资产权重:")
    for asset, weight in result.risky_asset_weights.items():
        if weight > 0.001:
            print(f"  {asset}: {weight:.2%}")

    print(f"\n完整权重分布:")
    for i, weight in enumerate(result.optimal_weights):
        print(f"  资产{i} ({asset_types[i]}): {weight:.2%}")


def example_7_comparison_strategies():
    """示例7: 不同策略对比"""
    print("\n" + "="*80)
    print("示例7: 不同优化策略对比")
    print("="*80)

    # 模拟数据
    np.random.seed(42)
    n_assets = 10

    expected_returns = np.random.randn(n_assets) * 0.01
    cov_matrix = np.random.randn(n_assets, n_assets)
    cov_matrix = cov_matrix @ cov_matrix.T * 0.0001

    asset_types = (
        ['treasury_bonds', 'high_grade_corporate_bonds', 'cash_equivalents'] +
        ['growth_stocks'] * 4 +
        ['emerging_markets'] * 2
    )[:n_assets]

    # 测试不同策略
    strategies = [
        OptimizationStrategy.TALEB_BARBELL,
        OptimizationStrategy.EQUAL_WEIGHT,
    ]

    results = {}

    for strategy in strategies:
        layer = create_joint_optimization_layer(
            optimization_strategy=strategy,
        )

        result = layer.optimize_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            asset_types=asset_types,
        )

        results[strategy.value] = result

        print(f"\n{strategy.value}:")
        print(f"  预期收益: {result.expected_return:.4f}")
        print(f"  预期风险: {result.expected_risk:.4f}")
        print(f"  夏普比率: {result.sharpe_ratio:.4f}")

    print(f"\n✅ 策略对比完成")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("统计学习模型层和联合优化层简化示例")
    print("="*80)

    # 运行所有示例
    example_1_learning_objective_config()
    example_2_optimization_metrics()
    example_3_portfolio_constraints()
    example_4_taleb_barbell_config()
    example_5_joint_optimization_strategies()
    example_6_joint_optimization_layer()
    example_7_comparison_strategies()

    print("\n" + "="*80)
    print("所有示例运行完成!")
    print("="*80)
    print("\n说明:")
    print("  本示例演示了统计学习层和联合优化层的API和架构")
    print("  要使用完整功能，请安装可选依赖:")
    print("    - LightGBM: pip install lightgbm")
    print("    - PyTorch: pip install torch")
    print("    - CVXPY: pip install cvxpy")
    print("="*80)


if __name__ == "__main__":
    main()
