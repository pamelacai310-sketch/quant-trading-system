"""
因果AI量化系统 - 自迭代因果推理引擎

核心组件：
1. 因果因素库 (Causal Factor Library)
2. 因果知识图谱 (Causal Knowledge Graph)
3. 因果发现引擎 (Causal Discovery Engine)
4. 因果验证引擎 (Causal Validation Engine)
5. 自迭代系统 (Self-Iterating System)
6. Renaissance Technologies风格统计套利 (Renaissance Stat Arb)
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
from .event_intensity import (
    EventIntensityEngine,
    EventIntensityFactor,
    EventIntensitySnapshot,
    EventIntensitySpec,
    create_event_intensity_engine,
)
from .cross_asset_causal_engine import (
    CrossAssetCausalEngine,
    CrossAssetProcessingConfig,
    CorrelationMethod,
    MacroRegime,
    MacroRegimeSnapshot,
    OrthogonalizationMethod,
)
from .invariance_market_decoder import (
    DecoderSnapshot,
    HierarchicalHMMDecoder,
    InvarianceMarketDecoder,
    InvariantDecoderConfig,
)
from .macro_event_state import (
    MacroEventState,
    MacroEventStateConfig,
    MacroEventStateEngine,
)
from .causal_graph_layer import (
    BackdoorAdjustmentSet,
    CausalDAGEdge,
    CausalGraphLayer,
    CounterfactualStressResult,
    SCMSnapshot,
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
from .game_causal_analysis import (
    ContextRule,
    EventCausalChain,
    EventWindowSnapshot,
    GameCausalAnalysisEngine,
    GameDominance,
    GameForce,
    GameRelationReport,
    GameRelationSpec,
    GameSideSpec,
    NewsEvent,
    PricingAssetRule,
    PriceConfirmationMemoryRecord,
    create_game_causal_analysis_engine,
)
from .research_governance import (
    ALLOW,
    CORRELATION_ONLY,
    IDENTIFIABLE,
    NO_TRADE,
    OBSERVE_ONLY,
    REDUCE,
    UNAVAILABLE,
    WEAK_IDENTIFIABLE,
    CausalAbstentionDecision,
    CausalAbstentionGate,
    CausalLLMAuditor,
    CausalEdgeValidationSnapshot,
    CausalValidationLoop,
    ExperimentRecord,
    ExperimentRegistry,
    FeatureRecord,
    FeatureStore,
    InstrumentRecord,
    InstrumentRegistry,
    LLMHypothesisAuditRecord,
    ModelRecord,
    ModelRegistry,
)

# 新增：因果发现引擎
try:
    from .causal_discovery_engine import (
        CausalDiscoveryEngine,
        CausalGraph,
        CausalPath,
        DiscoveryResult,
        DiscoveryAlgorithm,
        IndependenceTest,
        create_causal_discovery_engine,
    )
    CAUSAL_DISCOVERY_AVAILABLE = True
except ImportError:
    CAUSAL_DISCOVERY_AVAILABLE = False

# 新增：因果验证引擎
try:
    from .causal_validation_engine import (
        CausalValidationEngine,
        CausalValidationResult,
        ValidationResult,
        ValidationMethod,
        RobustnessTest,
        RobustnessReport,
        create_causal_validation_engine,
    )
    CAUSAL_VALIDATION_AVAILABLE = True
except ImportError:
    CAUSAL_VALIDATION_AVAILABLE = False

# 新增：Renaissance Technologies风格统计套利
try:
    from .renaissance_stat_arb import (
        RenaissanceStatArbEngine,
        PairsTradingSignal,
        FactorNeutralPosition,
        OrthogonalFactor,
        StatArbPosition,
        StatArbStrategy,
        OrthogonalizationMethod,
        create_renaissance_stat_arb_engine,
    )
    RENAISSANCE_STAT_ARB_AVAILABLE = True
except ImportError:
    RENAISSANCE_STAT_ARB_AVAILABLE = False

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
    "EventIntensityEngine",
    "EventIntensityFactor",
    "EventIntensitySnapshot",
    "EventIntensitySpec",
    "create_event_intensity_engine",
    "CrossAssetCausalEngine",
    "CrossAssetProcessingConfig",
    "CorrelationMethod",
    "MacroRegime",
    "MacroRegimeSnapshot",
    "OrthogonalizationMethod",
    "DecoderSnapshot",
    "HierarchicalHMMDecoder",
    "InvarianceMarketDecoder",
    "InvariantDecoderConfig",
    "MacroEventState",
    "MacroEventStateConfig",
    "MacroEventStateEngine",
    "BackdoorAdjustmentSet",
    "CausalDAGEdge",
    "CausalGraphLayer",
    "CounterfactualStressResult",
    "SCMSnapshot",
    "FeatureSelectionPolicy",
    "LearningObjectiveConfig",
    "ObjectiveMetrics",
    "PortfolioConstraintConfig",
    "PortfolioPlan",
    "SelectedFeature",
    "SelfIteratingCausalEngine",
    "SignalAllocation",
    "NewsEvent",
    "EventCausalChain",
    "EventWindowSnapshot",
    "GameForce",
    "GameDominance",
    "PricingAssetRule",
    "PriceConfirmationMemoryRecord",
    "ContextRule",
    "GameSideSpec",
    "GameRelationSpec",
    "GameRelationReport",
    "GameCausalAnalysisEngine",
    "create_game_causal_analysis_engine",
    "IDENTIFIABLE",
    "WEAK_IDENTIFIABLE",
    "CORRELATION_ONLY",
    "UNAVAILABLE",
    "ALLOW",
    "REDUCE",
    "OBSERVE_ONLY",
    "NO_TRADE",
    "CausalAbstentionDecision",
    "CausalAbstentionGate",
    "InstrumentRecord",
    "InstrumentRegistry",
    "LLMHypothesisAuditRecord",
    "CausalLLMAuditor",
    "CausalEdgeValidationSnapshot",
    "CausalValidationLoop",
    "FeatureRecord",
    "FeatureStore",
    "ExperimentRecord",
    "ExperimentRegistry",
    "ModelRecord",
    "ModelRegistry",
]

# 新增导出
if CAUSAL_DISCOVERY_AVAILABLE:
    __all__.extend([
        "CausalDiscoveryEngine",
        "CausalGraph",
        "CausalPath",
        "DiscoveryResult",
        "DiscoveryAlgorithm",
        "IndependenceTest",
        "create_causal_discovery_engine",
    ])

if CAUSAL_VALIDATION_AVAILABLE:
    __all__.extend([
        "CausalValidationEngine",
        "CausalValidationResult",
        "ValidationResult",
        "ValidationMethod",
        "RobustnessTest",
        "RobustnessReport",
        "create_causal_validation_engine",
    ])

if RENAISSANCE_STAT_ARB_AVAILABLE:
    __all__.extend([
        "RenaissanceStatArbEngine",
        "PairsTradingSignal",
        "FactorNeutralPosition",
        "OrthogonalFactor",
        "StatArbPosition",
        "StatArbStrategy",
        "OrthogonalizationMethod",
        "create_renaissance_stat_arb_engine",
    ])
