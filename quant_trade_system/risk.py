from __future__ import annotations

from typing import Any, Dict, List

from .models import OrderRequest, RiskCheckResult


class RiskManager:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {
            "max_gross_exposure": 400_000.0,
            "max_order_notional": 100_000.0,
            "max_position_per_symbol": 1_000.0,
            "max_daily_drawdown": -0.05,
        }

    def check_order(
        self,
        order: OrderRequest,
        portfolio_state: Dict[str, Any],
        positions: List[Dict[str, Any]],
        last_price: float,
        strategy_limits: Dict[str, Any] | None = None,
        latest_drawdown: float = 0.0,
    ) -> RiskCheckResult:
        limits = dict(self.config)
        if strategy_limits:
            limits.update({k: v for k, v in strategy_limits.items() if v is not None})
        violations: List[str] = []

        order_notional = abs(order.quantity * last_price)
        if order_notional > float(limits["max_order_notional"]):
            violations.append(f"订单名义金额 {order_notional:.2f} 超过上限 {float(limits['max_order_notional']):.2f}")

        symbol_position = next((item for item in positions if item["symbol"] == order.symbol), None)
        current_qty = float(symbol_position["quantity"]) if symbol_position else 0.0
        signed_qty = order.quantity if order.side == "buy" else -order.quantity
        target_qty = current_qty + signed_qty
        max_position = float(limits.get("max_position_per_symbol", 1_000.0))
        if abs(target_qty) > max_position:
            violations.append(f"目标仓位 {target_qty:.2f} 超过单品种仓位上限 {max_position:.2f}")

        gross_exposure = sum(abs(float(item["quantity"]) * float(item["avg_price"])) for item in positions)
        projected_exposure = gross_exposure - abs(current_qty * last_price) + abs(target_qty * last_price)
        if projected_exposure > float(limits["max_gross_exposure"]):
            violations.append(
                f"目标总敞口 {projected_exposure:.2f} 超过总敞口上限 {float(limits['max_gross_exposure']):.2f}"
            )

        if portfolio_state["cash"] - order_notional < 0 and order.side == "buy":
            violations.append("现金不足，无法执行买入订单")

        max_drawdown_limit = float(limits.get("max_daily_drawdown", -0.05))
        if latest_drawdown <= max_drawdown_limit:
            violations.append(f"当前回撤 {latest_drawdown:.2%} 已触发日内回撤熔断 {max_drawdown_limit:.2%}")

        return RiskCheckResult(
            passed=not violations,
            violations=violations,
            context={
                "order_notional": round(order_notional, 2),
                "projected_exposure": round(projected_exposure, 2),
                "last_price": round(last_price, 4),
            },
        )
