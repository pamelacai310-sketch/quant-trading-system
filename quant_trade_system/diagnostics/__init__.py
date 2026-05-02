"""
Diagnostics module for quantitative trading strategy analysis.

This module provides tools for diagnosing and optimizing trading strategies
using MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion) analysis.
"""

from .mae_mfe_diagnostics import (
    MAE_MFE_Diagnostics,
    MAE_MFE_Diagnosis,
    TradeAnalytics,
)

from .extreme_trade_analyzer import (
    ExtremeTrade,
    ExtremeTradeAnalyzer,
    analyze_extreme_trades_auto,
)

__all__ = [
    "MAE_MFE_Diagnostics",
    "MAE_MFE_Diagnosis",
    "TradeAnalytics",
    "ExtremeTrade",
    "ExtremeTradeAnalyzer",
    "analyze_extreme_trades_auto",
]
