"""
因果AI量化系统 - 自迭代因果推理引擎

核心组件：
1. 因果因素库 (Causal Factor Library)
2. 因果知识图谱 (Causal Knowledge Graph)
3. 因果发现引擎 (Causal Discovery Engine)
4. 因果验证引擎 (Causal Validation Engine)
5. 自迭代系统 (Self-Iterating System)
"""

from .causal_factor_library import (
    CausalFactorLibrary,
    CausalFactor,
    CausalEvidence,
    CausalEdge,
    QuantizedCausalFactor,
    FactorCategory,
    AssetClass,
    CausalType,
    EvidenceType,
    create_causal_factor_library,
)
from .cross_asset_causal_engine import (
    CrossAssetCausalEngine,
    CrossAssetProcessingConfig,
    CorrelationMethod,
    MacroRegime,
    MacroRegimeSnapshot,
    OrthogonalizationMethod,
)
from .self_iterating_causal_engine import (
    FeatureSelectionPolicy,
    LearningObjectiveConfig,
    ObjectiveMetrics,
    PortfolioConstraintConfig,
    PortfolioPlan,
    SelectedFeature,
    SelfIteratingCausalEngine,
    SignalAllocation,
)

__all__ = [
    "CausalFactorLibrary",
    "CausalFactor",
    "CausalEvidence",
    "CausalEdge",
    "QuantizedCausalFactor",
    "FactorCategory",
    "AssetClass",
    "CausalType",
    "EvidenceType",
    "create_causal_factor_library",
    "CrossAssetCausalEngine",
    "CrossAssetProcessingConfig",
    "CorrelationMethod",
    "MacroRegime",
    "MacroRegimeSnapshot",
    "OrthogonalizationMethod",
    "FeatureSelectionPolicy",
    "LearningObjectiveConfig",
    "ObjectiveMetrics",
    "PortfolioConstraintConfig",
    "PortfolioPlan",
    "SelectedFeature",
    "SelfIteratingCausalEngine",
    "SignalAllocation",
]
