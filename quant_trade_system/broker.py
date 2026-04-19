from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Tuple
from urllib import request

from .models import OrderRequest
from .storage import Storage, utc_now


class PaperBroker:
    def __init__(self, storage: Storage, slippage_bps: float = 3.0) -> None:
        self.storage = storage
        self.slippage_bps = slippage_bps

    def execute(self, order: OrderRequest, last_price: float) -> Dict[str, Any]:
        signed_qty = order.quantity if order.side == "buy" else -order.quantity
        fill_price = last_price * (1 + self.slippage_bps / 10_000 * (1 if signed_qty > 0 else -1))
        positions = self.storage.get_positions()
        portfolio = self.storage.get_portfolio_state()
        current = next((item for item in positions if item["symbol"] == order.symbol), None)
        current_qty = float(current["quantity"]) if current else 0.0
        current_avg = float(current["avg_price"]) if current else 0.0
        current_realized = float(current["realized_pnl"]) if current else 0.0
        target_qty, avg_price, realized_pnl = self._apply_fill(
            current_qty, current_avg, current_realized, signed_qty, fill_price
        )
        cash = float(portfolio["cash"]) - signed_qty * fill_price
        self.storage.update_cash(cash)
        self.storage.upsert_position(order.symbol, target_qty, avg_price, realized_pnl)
        order_record = self.storage.add_order(
            {
                "strategy_id": order.strategy_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "order_type": order.order_type,
                "limit_price": order.limit_price,
                "status": "filled",
                "broker_mode": order.broker_mode,
                "requested_at": utc_now(),
                "filled_at": utc_now(),
                "fill_price": round(fill_price, 4),
                "reason": order.reason,
                "metadata": order.metadata,
            }
        )
        return order_record

    def _apply_fill(
        self,
        current_qty: float,
        current_avg: float,
        current_realized: float,
        signed_qty: float,
        fill_price: float,
    ) -> Tuple[float, float, float]:
        target_qty = current_qty + signed_qty
        realized = current_realized
        avg_price = current_avg

        if current_qty == 0 or current_qty * signed_qty > 0:
            gross = abs(current_qty) * current_avg + abs(signed_qty) * fill_price
            avg_price = gross / max(abs(target_qty), 1e-9)
            return target_qty, avg_price, realized

        closing_qty = min(abs(current_qty), abs(signed_qty))
        if current_qty > 0:
            realized += closing_qty * (fill_price - current_avg)
        else:
            realized += closing_qty * (current_avg - fill_price)

        if abs(target_qty) < 1e-9:
            return 0.0, 0.0, realized
        if current_qty * target_qty < 0:
            avg_price = fill_price
        return target_qty, avg_price, realized


class WebhookBroker:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def execute(self, order: OrderRequest, last_price: float) -> Dict[str, Any]:
        endpoint = os.environ.get("QUANT_BROKER_WEBHOOK")
        payload = asdict(order)
        payload["last_price"] = last_price
        status = "accepted"
        metadata: Dict[str, Any] = {"transport": "webhook"}
        if endpoint:
            req = request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with request.urlopen(req, timeout=10) as response:
                metadata["response_code"] = response.status
        else:
            status = "pending_live_config"
            metadata["warning"] = "未配置 QUANT_BROKER_WEBHOOK，订单仅被登记，未真正发送到外部券商。"

        return self.storage.add_order(
            {
                "strategy_id": order.strategy_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "order_type": order.order_type,
                "limit_price": order.limit_price,
                "status": status,
                "broker_mode": order.broker_mode,
                "requested_at": utc_now(),
                "filled_at": None,
                "fill_price": None,
                "reason": order.reason,
                "metadata": {**order.metadata, **metadata},
            }
        )
