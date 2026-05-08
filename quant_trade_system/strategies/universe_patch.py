"""
Runtime universe patching for strategy classes.

Several strategy classes were originally written as demonstrations and kept
small hard-coded pools inside ``__init__``.  Importing this module patches those
classes so every new instance uses the central market universe:

* Hong Kong equities: Hang Seng Index constituents
* Futures: all SHFE/INE futures products

This keeps backward compatibility with the existing public class constructors
while removing the demo-only coverage limitation.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from ..market_universe import (
    SHFE_FUTURES_NAME_MAP,
    get_hang_seng_symbols,
    get_shfe_futures_symbols,
)


_PATCHED_ATTR = "_expanded_market_universe_patched"


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
        if callable(original):
            return original(self, underlying)
        return symbol

    cls._get_contract_name = _get_contract_name  # type: ignore[attr-defined]


def apply_expanded_universe_patch(
    weekly_cls: type | None = None,
    far_month_cls: type | None = None,
    hybrid_cls: type | None = None,
) -> None:
    """Patch strategy classes to use central market universes."""

    if weekly_cls is not None:
        def update_weekly(instance: Any) -> None:
            hs_symbols = get_hang_seng_symbols()
            shfe_symbols = get_shfe_futures_symbols()
            instance.long_term_favorites = hs_symbols
            instance.futures_contracts = shfe_symbols
            instance.market_universe_summary = {
                "hong_kong_equities": len(hs_symbols),
                "shanghai_futures": len(shfe_symbols),
            }

        _patch_init(weekly_cls, update_weekly)

    if far_month_cls is not None:
        def update_far_month(instance: Any) -> None:
            shfe_symbols = get_shfe_futures_symbols()
            instance.futures_universe = shfe_symbols
            instance.market_universe_summary = {
                "shanghai_futures": len(shfe_symbols),
            }

        _patch_init(far_month_cls, update_far_month)
        _patch_contract_name_method(far_month_cls)

    if hybrid_cls is not None:
        def update_hybrid(instance: Any) -> None:
            hs_symbols = get_hang_seng_symbols()
            shfe_symbols = get_shfe_futures_symbols()
            instance.stock_universe = hs_symbols
            instance.futures_universe = shfe_symbols
            instance.market_universe_summary = {
                "hong_kong_equities": len(hs_symbols),
                "shanghai_futures": len(shfe_symbols),
            }

        _patch_init(hybrid_cls, update_hybrid)
        _patch_contract_name_method(hybrid_cls)
