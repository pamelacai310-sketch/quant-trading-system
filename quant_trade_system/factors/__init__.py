"""
Factors Package - 因子库模块

提供3000+金融因子的计算、管理和优化。
"""

from .factor_library import FactorLibrary, load_factor_library
from .technical_factors import TechnicalFactors, compute_technical_factors
from .canslim_screener import (
    CANSLIM_Screener,
    CANSLIM_Score,
    calculate_relative_strength,
    detect_follow_through_day,
)

__all__ = [
    'FactorLibrary',
    'load_factor_library',
    'TechnicalFactors',
    'compute_technical_factors',
    'CANSLIM_Screener',
    'CANSLIM_Score',
    'calculate_relative_strength',
    'detect_follow_through_day',
]
