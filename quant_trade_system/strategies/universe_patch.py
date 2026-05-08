"""
Runtime universe patching for strategy classes.

Several strategy classes were originally written as demonstrations and kept
small hard-coded pools inside ``__init__``.  Importing this module patches those
classes so every new instance uses the central market universe:

* Stocks: S&P 500 + Nasdaq-100 + Hang Seng Index + STAR leaders
* Futures: all China futures products across SHFE/INE, DCE, CZCE, CFFEX, GFEX

This keeps backward compatibility with the existing public class constructors
while removing the demo-only coverage limitation.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from ..market_universe import SHFE_FUTURES_NAME_MAP
from ..universe_provider import MarketUniverseProvider


_PATCHED_ATTR = "_expanded_market_universe_patched"
_PROVIDER = MarketUniverseProvider(prefer_live=True)


def _patch_init(cls: type, updater: Callable[[Any], None]) -> None:
    if getattr(cls, _PATCHED_ATTR, False):
        return

    original_init = cls.__init__

    @wraps(original_init)
    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        updater(self)

    cls.__init__ = patched_init  # type: ignore[method-assign]
    setattr(cls, _PATCHED_ATTR, True)


def _patch_contract_name_method(cls: type) -> None:
    original = getattr(cls, "_get_contract_name", None)

    def _get_contract_name(self: Any, underlying: str) -> str:
        symbol = str(underlying).upper()
        if symbol in SHFE_FUTURES_NAME_MAP:
            return SHFE_FUTURES_NAME_MAP[symbol]
        try:
            product = next(
                item for item in _PROVIDER.get_universe("cn_futures_products", include_contracts=False)
                if item.symbol == symbol
            )
            return product.name
        except Exception:
            pass
        if callable(original):
            return original(self, underlying)
        return symbol

    cls._get_contract_name = _get_contract_name  # type: ignore[attr-defined]


def _stock_symbols() -> list[str]:
    return _PROVIDER.get_symbols("us_core", include_contracts=False) + \
        _PROVIDER.get_symbols("hk_hsi", include_contracts=False) + \
        _PROVIDER.get_symbols("cn_star_leaders", include_contracts=False)


def _futures_products() -> list[str]:
    return _PROVIDER.get_symbols("cn_futures_products", include_contracts=False)


def apply_expanded_universe_patch(
    weekly_cls: type | None = None,
    far_month_cls: type | None = None,
    hybrid_cls: type | None = None,
) -> None:
    """Patch strategy classes to use central market universes."""

    if weekly_cls is not None:
        def update_weekly(instance: Any) -> None:
            stock_symbols = _stock_symbols()
            futures_symbols = _futures_products()
            instance.long_term_favorites = stock_symbols
            instance.futures_contracts = futures_symbols
            instance.market_universe_summary = {
                "stocks": len(stock_symbols),
                "china_futures_products": len(futures_symbols),
            }

        _patch_init(weekly_cls, update_weekly)

    if far_month_cls is not None:
        def update_far_month(instance: Any) -> None:
            futures_symbols = _futures_products()
            instance.futures_universe = futures_symbols
            instance.market_universe_summary = {
                "china_futures_products": len(futures_symbols),
            }

        _patch_init(far_month_cls, update_far_month)
        _patch_contract_name_method(far_month_cls)

    if hybrid_cls is not None:
        def update_hybrid(instance: Any) -> None:
            stock_symbols = _stock_symbols()
            futures_symbols = _futures_products()
            instance.stock_universe = stock_symbols
            instance.futures_universe = futures_symbols
            instance.market_universe_summary = {
                "stocks": len(stock_symbols),
                "china_futures_products": len(futures_symbols),
            }

        _patch_init(hybrid_cls, update_hybrid)
        _patch_contract_name_method(hybrid_cls)
