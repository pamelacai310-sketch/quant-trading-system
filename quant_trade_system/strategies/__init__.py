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
]
