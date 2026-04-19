from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OrderRequest:
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    limit_price: Optional[float] = None
    strategy_id: Optional[str] = None
    broker_mode: str = "paper"
    reason: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskCheckResult:
    passed: bool
    violations: List[str]
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyRunResult:
    strategy_id: str
    strategy_name: str
    signal: int
    side: Optional[str]
    quantity: float
    last_price: float
    reason: str
    indicators: Dict[str, float]


@dataclass
class BacktestResult:
    strategy_id: str
    total_return: float
    annual_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    p_value: float
    trade_count: int
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    stats: Dict[str, Any]
