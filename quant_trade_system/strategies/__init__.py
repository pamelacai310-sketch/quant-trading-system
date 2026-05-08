"""
Strategies module for complete trading strategies.

Provides complete trading strategy implementations including
O'Neil CANSLIM strategy, Taleb Barbell strategy, causal-driven hybrid strategy,
and other systematic trading approaches.
"""

from .oneill_strategy import (
    ONeillPosition,
    ONeillStrategyEngine,
    ONeillTradeSetup,
    run_oneill_strategy,
)
from .taleb_barbell import (
    TalebBarbellStrategy,
    TalebBarbellPortfolio,
    simulate_taleb_barbell,
)
from .tail_option_engine import (
    TailOptionEngine,
    OptionContract,
    OptionType,
    CrisisTriggerType,
)
from .strategy_causal_analysis import (
    ONeillCausalAnalyzer,
    TalebCausalAnalyzer,
    HybridStrategyAnalyzer,
    CausalRelationship,
    MarketRegime,
    generate_causal_report,
)
from .causal_hybrid_strategy import (
    CausalHybridStrategy,
    AllocationMode,
    CausalSignals,
    HybridPosition,
    simulate_causal_hybrid_strategy,
)
from .weekly_swing_strategy import (
    WeeklySwingStrategy,
    SwingPosition,
    WeeklyTarget,
    DayOfWeek,
    PositionSide,
    simulate_weekly_swing_strategy,
)
from .far_month_futures_strategy import (
    FarMonthFuturesStrategy,
    FuturesContract,
    FuturesPosition,
    MarketSentiment,
    simulate_far_month_futures_strategy,
)
from .hybrid_swing_strategy import (
    HybridSwingStrategy,
    UnifiedPosition,
    AssetType,
    simulate_hybrid_strategy,
)
from .enhanced_hybrid_swing_strategy import (
    EnhancedHybridSwingStrategy,
    MarketState,
    EnhancedStockOpportunity,
    EnhancedFuturesOpportunity,
    simulate_enhanced_hybrid_strategy,
)
from .universe_patch import apply_expanded_universe_patch

# Expand demo strategy universes at import time.  This preserves the existing
# constructor API while making new strategy instances scan the configured
# cross-market universe rather than the old sample-only pools.
apply_expanded_universe_patch(
    weekly_cls=WeeklySwingStrategy,
    far_month_cls=FarMonthFuturesStrategy,
    hybrid_cls=HybridSwingStrategy,
)
apply_expanded_universe_patch(hybrid_cls=EnhancedHybridSwingStrategy)

__all__ = [
    # O'Neil CANSLIM
    "ONeillPosition",
    "ONeillStrategyEngine",
    "ONeillTradeSetup",
    "run_oneill_strategy",
    # Taleb Barbell
    "TalebBarbellStrategy",
    "TalebBarbellPortfolio",
    "simulate_taleb_barbell",
    # Tail Option Engine
    "TailOptionEngine",
    "OptionContract",
    "OptionType",
    "CrisisTriggerType",
    # Causal Analysis
    "ONeillCausalAnalyzer",
    "TalebCausalAnalyzer",
    "HybridStrategyAnalyzer",
    "CausalRelationship",
    "MarketRegime",
    "generate_causal_report",
    # Causal Hybrid Strategy
    "CausalHybridStrategy",
    "AllocationMode",
    "CausalSignals",
    "HybridPosition",
    "simulate_causal_hybrid_strategy",
    # Weekly Swing Strategy
    "WeeklySwingStrategy",
    "SwingPosition",
    "WeeklyTarget",
    "DayOfWeek",
    "PositionSide",
    "simulate_weekly_swing_strategy",
    # Far Month Futures Strategy
    "FarMonthFuturesStrategy",
    "FuturesContract",
    "FuturesPosition",
    "MarketSentiment",
    "simulate_far_month_futures_strategy",
    # Hybrid Swing Strategy
    "HybridSwingStrategy",
    "UnifiedPosition",
    "AssetType",
    "simulate_hybrid_strategy",
    # Enhanced Hybrid Swing Strategy
    "EnhancedHybridSwingStrategy",
    "MarketState",
    "EnhancedStockOpportunity",
    "EnhancedFuturesOpportunity",
    "simulate_enhanced_hybrid_strategy",
    # Universe patching
    "apply_expanded_universe_patch",
]
