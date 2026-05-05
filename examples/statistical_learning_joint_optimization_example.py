"""
统计学习模型层和联合优化层综合示例

演示：
1. 使用Transformer/LightGBM从特征矩阵中学习
2. 优化胜率、赔率、弹性三个关键指标
3. 联合优化：嵌入Sharpe/CVaR到训练目标
4. 塔勒布杠铃策略端到端优化
5. 组合优化和信号生成为统一目标函数
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

from quant_trade_system.core.feature_engineering_layer import (
    FeatureEngineeringLayer,
    FeatureMatrix,
    FeatureGranularity,
    create_feature_engineering_layer,
)
from quant_trade_system.core.statistical_learning_layer import (
    StatisticalLearningLayer,
    ModelType,
    OptimizationMetric,
    LearningObjectiveConfig,
    TrainingResult,
    PredictionResult,
    create_statistical_learning_layer,
)
from quant_trade_system.core.joint_optimization_layer import (
    JointOptimizationLayer,
    OptimizationStrategy,
    PortfolioConstraint,
    TalebBarbellConfig,
    OptimizationResult,
    JointTrainingResult,
    create_joint_optimization_layer,
)


def generate_sample_features_and_labels(
    n_samples: int = 1000,
    n_features: int = 50,
) -> tuple:
    """
    生成示例特征和标签

    参数:
        n_samples: 样本数量
        n_features: 特征数量

    返回:
        feature_matrix, labels, returns
    """
    np.random.seed(42)

    # 生成特征
    feature_names = [f"feature_{i}" for i in range(n_features)]
    timestamps = pd.date_range(
        start='2020-01-01',
        periods=n_samples,
        freq='D',
    )
    symbols = ['ASSET_1']

    # 特征数据（随机游走 + 趋势）
    features_data = np.random.randn(n_samples, n_features)
    for i in range(n_features):
        # 添加一些趋势和自相关
        trend = np.linspace(0, 0.1, n_samples)
        ar_component = np.zeros(n_samples)
        for t in range(1, n_samples):
            ar_component[t] = 0.5 * ar_component[t-1] + np.random.randn() * 0.1

        features_data[:, i] += trend + ar_component

    # 创建特征矩阵
    feature_df = pd.DataFrame(
        features_data,
        index=timestamps,
        columns=feature_names,
    )

    feature_metadata = {}
    for name in feature_names:
        feature_metadata[name] = {
            'name': name,
            'causal_factor_id': f'factor_{name}',
            'formula': name,
            'financial_meaning': f'{name}描述金融因果关系',
            'expected_sign': 1,
            'category': 'momentum',
            'interpretability': 0.7 + np.random.rand() * 0.3,
            'independent_power': 0.6 + np.random.rand() * 0.4,
        }

    feature_matrix = FeatureMatrix(
        data=feature_df,
        feature_metadata=feature_metadata,
        timestamps=list(timestamps),
        symbols=symbols,
        granularity=FeatureGranularity.DAILY,
        sampling_start=timestamps[0],
        sampling_end=timestamps[-1],
    )

    # 生成标签（根据特征的加权组合 + 噪声）
    true_weights = np.random.randn(n_features)
    true_weights = true_weights / np.linalg.norm(true_weights)

    signal = features_data @ true_weights
    prob = 1 / (1 + np.exp(-signal))  # sigmoid

    labels = (np.random.rand(n_samples) < prob).astype(int)

    # 生成收益
    returns = np.random.randn(n_samples) * 0.02
    returns[labels == 1] += 0.01  # 上涨时收益更高
    returns[labels == 0] -= 0.005  # 下跌时收益更低

    return feature_matrix, labels, returns


def generate_multiasset_sample(
    n_samples: int = 1000,
    n_assets: int = 10,
) -> tuple:
    """
    生成多资产示例数据

    参数:
        n_samples: 样本数量
        n_assets: 资产数量

    返回:
        feature_matrix, returns, asset_types
    """
    np.random.seed(42)

    # 生成特征
    n_features = 50
    feature_names = [f"feature_{i}" for i in range(n_features)]
    timestamps = pd.date_range(
        start='2020-01-01',
        periods=n_samples,
        freq='D',
    )
    symbols = [f'ASSET_{i}' for i in range(n_assets)]

    # 特征数据
    features_data = np.random.randn(n_samples, n_features)

    feature_df = pd.DataFrame(
        features_data,
        index=timestamps,
        columns=feature_names,
    )

    feature_metadata = {}
    for name in feature_names:
        feature_metadata[name] = {
            'name': name,
            'causal_factor_id': f'factor_{name}',
            'formula': name,
            'financial_meaning': f'{name}描述金融因果关系',
            'expected_sign': 1,
            'category': 'momentum',
            'interpretability': 0.7,
            'independent_power': 0.6,
        }

    feature_matrix = FeatureMatrix(
        data=feature_df,
        feature_metadata=feature_metadata,
        timestamps=list(timestamps),
        symbols=symbols,
        granularity=FeatureGranularity.DAILY,
        sampling_start=timestamps[0],
        sampling_end=timestamps[-1],
    )

    # 生成多资产收益
    # 前3个：安全资产（低收益低风险）
    # 后7个：风险资产（高收益高风险）
    returns = np.zeros((n_samples, n_assets))

    for i in range(n_assets):
        if i < 3:  # 安全资产
            returns[:, i] = np.random.randn(n_samples) * 0.005 + 0.0002
        else:  # 风险资产
            returns[:, i] = np.random.randn(n_samples) * 0.02 + 0.0005

    # 资产类型
    asset_types = []
    for i in range(n_assets):
        if i < 3:
            asset_types.extend([
                'treasury_bonds',
                'high_grade_corporate_bonds',
                'cash_equivalents',
            ])
        else:
            asset_types.extend([
                'growth_stocks',
                'emerging_markets',
                'commodities',
            ])

    asset_types = asset_types[:n_assets]

    return feature_matrix, returns, asset_types


def example_1_statistical_learning_lightgbm():
    """示例1: 使用LightGBM进行统计学习"""
    print("\n" + "="*80)
    print("示例1: LightGBM统计学习")
    print("="*80)

    # 生成数据
    feature_matrix, labels, returns = generate_sample_features_and_labels(
        n_samples=1000,
        n_features=50,
    )

    print(f"✅ 数据生成成功")
    print(f"  样本数: {len(labels)}")
    print(f"  特征数: {len(feature_matrix.data.columns)}")
    print(f"  胜率（标签）: {labels.mean():.2%}")

    # 创建统计学习层
    layer = create_statistical_learning_layer(
        model_type=ModelType.LIGHTGBM,
        feature_selection=True,
    )

    print(f"\n✅ 统计学习层创建成功")
    print(f"  模型类型: {layer.model_type}")

    # 配置学习目标
    objective_config = LearningObjectiveConfig(
        primary_metric=OptimizationMetric.WIN_RATE,
        secondary_metrics=[
            OptimizationMetric.ODDS_RATIO,
            OptimizationMetric.ELASTICITY,
        ],
        min_win_rate=0.55,
        min_odds_ratio=1.5,
        min_elasticity=1.2,
    )

    layer.objective_config = objective_config

    # 训练模型
    print(f"\n开始训练...")
    training_result = layer.train(
        feature_matrix=feature_matrix,
        labels=labels,
        returns=returns,
        validation_split=0.2,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
    )

    print(f"\n✅ 模型训练成功")
    print(f"  训练时间: {training_result.training_time:.2f}秒")
    print(f"  选中特征数: {len(training_result.selected_features)}")

    print(f"\n训练指标:")
    for metric, value in training_result.training_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print(f"\n验证指标:")
    for metric, value in training_result.validation_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # 预测
    predictions = layer.predict(feature_matrix)

    print(f"\n✅ 预测完成")
    print(f"  预期胜率: {predictions.expected_win_rate:.2%}")
    print(f"  预期赔率: {predictions.expected_odds_ratio:.2f}")
    print(f"  预期弹性: {predictions.expected_elasticity:.2f}")


def example_2_statistical_learning_transformer():
    """示例2: 使用Transformer进行统计学习"""
    print("\n" + "="*80)
    print("示例2: Transformer统计学习")
    print("="*80)

    # 生成数据
    feature_matrix, labels, returns = generate_sample_features_and_labels(
        n_samples=500,
        n_features=30,
    )

    print(f"✅ 数据生成成功")
    print(f"  样本数: {len(labels)}")
    print(f"  特征数: {len(feature_matrix.data.columns)}")

    # 创建统计学习层
    layer = create_statistical_learning_layer(
        model_type=ModelType.TRANSFORMER,
        feature_selection=True,
    )

    print(f"\n✅ 统计学习层创建成功")
    print(f"  模型类型: {layer.model_type}")

    # 训练模型（如果PyTorch不可用，跳过）
    try:
        print(f"\n开始训练...")
        training_result = layer.train(
            feature_matrix=feature_matrix,
            labels=labels,
            returns=returns,
            validation_split=0.2,
            d_model=128,
            nhead=4,
            num_encoder_layers=4,
            seq_len=20,
            batch_size=32,
            epochs=50,
            lr=0.001,
        )

        print(f"\n✅ 模型训练成功")
        print(f"  训练时间: {training_result.training_time:.2f}秒")

        print(f"\n验证指标:")
        for metric, value in training_result.validation_metrics.items():
            print(f"  {metric}: {value:.4f}")

    except ImportError as e:
        print(f"\n❌ {e}")
        print(f"  提示: 请安装PyTorch - pip install torch")


def example_3_joint_optimization_taleb_barbell():
    """示例3: 联合优化 - 塔勒布杠铃策略"""
    print("\n" + "="*80)
    print("示例3: 联合优化 - 塔勒布杠铃策略")
    print("="*80)

    # 生成多资产数据
    feature_matrix, returns, asset_types = generate_multiasset_sample(
        n_samples=1000,
        n_assets=10,
    )

    print(f"✅ 数据生成成功")
    print(f"  样本数: {returns.shape[0]}")
    print(f"  资产数: {returns.shape[1]}")
    print(f"  资产类型: {asset_types}")

    # 计算预期收益和协方差矩阵
    expected_returns = returns.mean(axis=0)
    covariance_matrix = np.cov(returns.T)

    print(f"\n预期收益:")
    for i, (ret, asset_type) in enumerate(zip(expected_returns, asset_types)):
        print(f"  资产{i} ({asset_type}): {ret:.4f}")

    # 创建联合优化层
    taleb_config = TalebBarbellConfig(
        safe_asset_ratio=0.85,
        risky_asset_ratio=0.15,
    )

    layer = create_joint_optimization_layer(
        optimization_strategy=OptimizationStrategy.TALEB_BARBELL,
        taleb_config=taleb_config,
    )

    print(f"\n✅ 联合优化层创建成功")
    print(f"  优化策略: {layer.optimization_strategy}")
    print(f"  安全资产比例: {layer.taleb_config.safe_asset_ratio}")
    print(f"  风险资产比例: {layer.taleb_config.risky_asset_ratio}")

    # 优化组合
    print(f"\n开始优化...")
    result = layer.optimize_portfolio(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
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

    print(f"\n最优权重:")
    for i, weight in enumerate(result.optimal_weights):
        print(f"  资产{i}: {weight:.2%}")


def example_4_joint_optimization_max_sharpe():
    """示例4: 联合优化 - 最大化夏普比率"""
    print("\n" + "="*80)
    print("示例4: 联合优化 - 最大化夏普比率")
    print("="*80)

    # 生成多资产数据
    feature_matrix, returns, asset_types = generate_multiasset_sample(
        n_samples=1000,
        n_assets=10,
    )

    print(f"✅ 数据生成成功")
    print(f"  样本数: {returns.shape[0]}")
    print(f"  资产数: {returns.shape[1]}")

    # 计算预期收益和协方差矩阵
    expected_returns = returns.mean(axis=0)
    covariance_matrix = np.cov(returns.T)

    # 创建联合优化层
    layer = create_joint_optimization_layer(
        optimization_strategy=OptimizationStrategy.MAX_SHARPE,
    )

    print(f"\n✅ 联合优化层创建成功")
    print(f"  优化策略: {layer.optimization_strategy}")

    # 优化组合
    print(f"\n开始优化...")
    result = layer.optimize_portfolio(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        asset_types=asset_types,
    )

    print(f"\n✅ 优化完成")
    print(f"  预期收益: {result.expected_return:.4f}")
    print(f"  预期风险: {result.expected_risk:.4f}")
    print(f"  夏普比率: {result.sharpe_ratio:.4f}")

    print(f"\n最优权重 (Top 5):")
    top_indices = np.argsort(result.optimal_weights)[-5:][::-1]
    for i in top_indices:
        print(f"  资产{i}: {result.optimal_weights[i]:.2%}")


def example_5_end_to_end_training():
    """示例5: 端到端联合训练（信号生成 + 组合优化）"""
    print("\n" + "="*80)
    print("示例5: 端到端联合训练")
    print("="*80)

    # 生成多资产数据
    feature_matrix, returns, asset_types = generate_multiasset_sample(
        n_samples=500,
        n_assets=5,
    )

    print(f"✅ 数据生成成功")
    print(f"  样本数: {returns.shape[0]}")
    print(f"  资产数: {returns.shape[1]}")
    print(f"  特征数: {len(feature_matrix.data.columns)}")

    # 创建联合优化层
    layer = create_joint_optimization_layer(
        optimization_strategy=OptimizationStrategy.TALEB_BARBELL,
    )

    print(f"\n✅ 联合优化层创建成功")

    # 端到端训练（如果PyTorch可用）
    try:
        print(f"\n开始端到端训练...")
        print(f"  （这需要较长时间，请耐心等待）")

        joint_result = layer.train_joint_model(
            feature_matrix=feature_matrix,
            returns=returns,
            asset_types=asset_types,
            epochs=50,
            learning_rate=0.001,
            validation_split=0.2,
        )

        print(f"\n✅ 端到端训练完成")
        print(f"  训练轮数: {joint_result.training_epochs}")
        print(f"  最佳夏普比率: {joint_result.best_sharpe:.4f}")
        print(f"  最佳CVaR: {joint_result.best_cvar:.4f}")

        print(f"\n最佳权重:")
        for i, weight in enumerate(joint_result.best_weights):
            print(f"  资产{i} ({asset_types[i]}): {weight:.2%}")

    except ImportError as e:
        print(f"\n❌ {e}")
        print(f"  提示: 请安装PyTorch - pip install torch")


def example_6_complete_workflow():
    """示例6: 完整工作流（特征工程 -> 统计学习 -> 联合优化）"""
    print("\n" + "="*80)
    print("示例6: 完整工作流")
    print("="*80)

    # 第1步：特征工程
    print(f"\n第1步：特征工程")
    feature_engineering_layer = create_feature_engineering_layer(
        target_features=100,
    )

    # 生成市场数据
    symbols = ['ASSET_1', 'ASSET_2']
    market_data = {}
    for symbol in symbols:
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'close': 100 + np.random.randn(500).cumsum() * 0.5,
            'volume': np.random.randint(1000000, 10000000, 500),
            'interest_rate': 0.03 + np.random.randn(500) * 0.001,
        })
        market_data[symbol] = df.set_index('timestamp')

    # 生成特征矩阵
    feature_matrix = feature_engineering_layer.generate_feature_matrix(
        market_data=market_data,
        granularity=FeatureGranularity.DAILY,
        feature_limit=50,
    )

    print(f"  ✅ 特征矩阵生成成功")
    print(f"  特征数量: {len(feature_matrix.data.columns)}")

    # 第2步：统计学习
    print(f"\n第2步：统计学习")
    statistical_layer = create_statistical_learning_layer(
        model_type=ModelType.LIGHTGBM,
    )

    # 生成标签和收益
    labels = (np.random.rand(len(feature_matrix.data)) > 0.5).astype(int)
    returns = np.random.randn(len(feature_matrix.data)) * 0.02

    # 训练
    training_result = statistical_layer.train(
        feature_matrix=feature_matrix,
        labels=labels,
        returns=returns,
        validation_split=0.2,
    )

    print(f"  ✅ 模型训练完成")
    print(f"  验证胜率: {training_result.validation_metrics['win_rate']:.2%}")

    # 第3步：联合优化
    print(f"\n第3步：联合优化")
    joint_layer = create_joint_optimization_layer(
        optimization_strategy=OptimizationStrategy.TALEB_BARBELL,
    )

    # 生成多资产数据
    multi_returns = np.random.randn(500, 5) * 0.02
    asset_types = ['treasury_bonds', 'cash_equivalents', 'growth_stocks',
                   'emerging_markets', 'commodities']

    expected_returns = multi_returns.mean(axis=0)
    covariance_matrix = np.cov(multi_returns.T)

    # 优化
    result = joint_layer.optimize_portfolio(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        asset_types=asset_types,
    )

    print(f"  ✅ 组合优化完成")
    print(f"  夏普比率: {result.sharpe_ratio:.4f}")

    print(f"\n✅ 完整工作流执行成功！")
    print(f"  特征工程 -> 统计学习 -> 联合优化")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("统计学习模型层和联合优化层综合示例")
    print("="*80)

    # 运行所有示例
    example_1_statistical_learning_lightgbm()
    example_2_statistical_learning_transformer()
    example_3_joint_optimization_taleb_barbell()
    example_4_joint_optimization_max_sharpe()
    example_5_end_to_end_training()
    example_6_complete_workflow()

    print("\n" + "="*80)
    print("所有示例运行完成!")
    print("="*80)


if __name__ == "__main__":
    main()
