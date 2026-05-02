"""
Patterns module for technical analysis.

Provides pattern recognition for various technical analysis patterns,
including O'Neil patterns (Cup with Handle, VCP, etc.) and other classic patterns.
"""

from .oneill_patterns import (
    ONeillPatternDetector,
    PatternInfo,
    PatternQuality,
    PatternType,
)

__all__ = [
    "ONeillPatternDetector",
    "PatternInfo",
    "PatternQuality",
    "PatternType",
]
