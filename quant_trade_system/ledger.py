"""Deterministic accounting shared by research and bar-based shadow execution."""
from dataclasses import dataclass
import numpy as np


@dataclass
class PositionLedger:
    cash: float
    multiplier: float = 1.0
    futures: bool = False
    quantity: float = 0.0
    average_price: float = 0.0
    entry_fees: float = 0.0

    def equity(self, price):
        if self.futures:
            return self.cash + self.quantity * (price - self.average_price) * self.multiplier
        return self.cash + self.quantity * price * self.multiplier

    def fill(self, delta, price, fee):
        if not all(np.isfinite(x) for x in (delta, price, fee)) or price <= 0 or fee < 0:
            raise ValueError('Invalid fill')
        old = self.quantity
        if old * (old + delta) < -1e-10:
            raise ValueError('Split reversals into a close and an open')
        realized = None
        if old * delta < 0:
            closed = min(abs(old), abs(delta))
            allocated = self.entry_fees * closed / abs(old)
            gross = closed * np.sign(old) * (price - self.average_price) * self.multiplier
            realized = dict(quantity=closed, gross_pnl=gross, net_pnl=gross-allocated-fee,
                            entry_fee=allocated, exit_fee=fee, entry_price=self.average_price)
            self.entry_fees -= allocated
            if self.futures:
                self.cash += gross
        elif delta:
            self.average_price = (abs(old)*self.average_price + abs(delta)*price) / abs(old+delta)
            self.entry_fees += fee
        self.cash -= fee
        if not self.futures:
            self.cash -= delta * price * self.multiplier
        self.quantity += delta
        if abs(self.quantity) < 1e-10:
            self.quantity = self.average_price = self.entry_fees = 0.0
        return realized


def position_returns(positions, forward_returns, cost_bps=0.0):
    """w[t] earns explicitly aligned executable forward return[t]; flatten at end.

    This is a period-return diagnostic, not a closed-trade win-rate estimator.
    """
    w, r = np.asarray(positions, float), np.asarray(forward_returns, float)
    if w.ndim != 1 or w.shape != r.shape or not np.isfinite(w).all() or not np.isfinite(r).all():
        raise ValueError('Positions and forward returns must be finite aligned vectors')
    if cost_bps < 0:
        raise ValueError('Negative costs')
    cost = np.abs(np.diff(np.r_[0.0, w])) * cost_bps / 10000
    if len(cost):
        cost[-1] += abs(w[-1]) * cost_bps / 10000
    return w * r - cost
