"""
Strategies module for complete trading strategies.

Provides complete trading strategy implementations including
O'Neil CANSLIM strategy, Taleb Barbell strategy, and other systematic trading approaches.
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
]
