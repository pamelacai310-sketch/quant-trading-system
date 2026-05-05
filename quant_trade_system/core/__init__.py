"""
Core Package - 核心模块

提供：
1. 性能优化工具（Polars加速）
2. 特征工程层（Feature Engineering Layer）
3. 统计学习模型层（Statistical Learning Layer）
4. 联合优化层（Joint Optimization Layer）
5. 因果AI模块（Causal AI）
"""

from .polars_adapter import (
    PolarsDataFrame,
    should_use_polars,
    compute_indicators_optimized,
    PerformanceBenchmark,
    HAS_POLARS,
)

__all__ = [
    'PolarsDataFrame',
    'should_use_polars',
    'compute_indicators_optimized',
    'PerformanceBenchmark',
    'HAS_POLARS',
]

# 导出特征工程层
try:
    from .feature_engineering_layer import (
        FeatureEngineeringLayer,
        FeatureGranularity,
        DataSource,
        FeatureDomain,
        QuantizedCausalFeature,
        FeatureMatrix,
        FeatureSelectionResult,
        create_feature_engineering_layer,
    )
    __all__.extend([
        'FeatureEngineeringLayer',
        'FeatureGranularity',
        'DataSource',
        'FeatureDomain',
        'QuantizedCausalFeature',
        'FeatureMatrix',
        'FeatureSelectionResult',
        'create_feature_engineering_layer',
    ])
except ImportError:
    pass

# 导出统计学习层
try:
    from .statistical_learning_layer import (
        StatisticalLearningLayer,
        ModelType,
        OptimizationMetric,
        LossFunction,
        LearningObjectiveConfig,
        TrainingResult,
        PredictionResult,
        ModelEvaluation,
        WinRateLoss,
        OddsRatioLoss,
        ElasticityLoss,
        CombinedLoss,
        create_statistical_learning_layer,
    )
    __all__.extend([
        'StatisticalLearningLayer',
        'ModelType',
        'OptimizationMetric',
        'LossFunction',
        'LearningObjectiveConfig',
        'TrainingResult',
        'PredictionResult',
        'ModelEvaluation',
        'WinRateLoss',
        'OddsRatioLoss',
        'ElasticityLoss',
        'CombinedLoss',
        'create_statistical_learning_layer',
    ])
except ImportError:
    pass

# 导出联合优化层
try:
    from .joint_optimization_layer import (
        JointOptimizationLayer,
        OptimizationStrategy,
        RiskMeasure,
        PortfolioConstraint,
        TalebBarbellConfig,
        OptimizationResult,
        JointTrainingResult,
        create_joint_optimization_layer,
    )
    __all__.extend([
        'JointOptimizationLayer',
        'OptimizationStrategy',
        'RiskMeasure',
        'PortfolioConstraint',
        'TalebBarbellConfig',
        'OptimizationResult',
        'JointTrainingResult',
        'create_joint_optimization_layer',
    ])
except ImportError:
    pass
