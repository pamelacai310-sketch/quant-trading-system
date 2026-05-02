"""
Signals module for trading signals detection.

Provides various signal detection algorithms including pocket pivots,
breakout signals, and other O'Neil-style entry signals.
"""

from .pocket_pivots import (
    PocketPivotDetector,
    PocketPivotSignal,
    is_pocket_pivot_today,
)

__all__ = [
    "PocketPivotDetector",
    "PocketPivotSignal",
    "is_pocket_pivot_today",
]
