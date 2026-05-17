from __future__ import annotations

import unittest
from datetime import datetime

from quant_trade_system.futures_specs import (
    build_one_lot_margin_table,
    calculate_futures_margin,
    futures_contract_multiplier,
    get_futures_contract_spec,
)
from quant_trade_system.universe_provider import (
    CFFEX_FUTURES_PRODUCTS,
    CZCE_FUTURES_PRODUCTS,
    DCE_FUTURES_PRODUCTS,
    GFEX_FUTURES_PRODUCTS,
)
from quant_trade_system.market_universe import SHFE_FUTURES_PRODUCTS
from quant_trade_system.strategies.far_month_futures_strategy import FuturesContract, FarMonthFuturesStrategy, PositionSide
from quant_trade_system.strategies.hybrid_swing_strategy import HybridSwingStrategy


class FuturesMarginSpecTests(unittest.TestCase):
    def test_cu_one_lot_margin_uses_multiplier_and_platform_margin_rate(self) -> None:
        latest_price = 74_717.64705882352
        margin = calculate_futures_margin("CU2607", latest_price, lots=1, margin_rate=0.17)
        self.assertEqual(futures_contract_multiplier("CU2607"), 5.0)
        self.assertAlmostEqual(margin, 63_510.0, places=2)

    def test_one_lot_margin_table_is_auditable(self) -> None:
        rows = build_one_lot_margin_table({"CU2607": 74_717.64705882352}, {"CU": 0.17})
        self.assertEqual(rows[0]["product"], "CU")
        self.assertEqual(rows[0]["contract_multiplier"], 5.0)
        self.assertAlmostEqual(rows[0]["one_lot_margin"], 63_510.0, places=2)
        self.assertAlmostEqual(rows[0]["one_lot_notional"], 373_588.2352941176, places=2)

    def test_far_month_strategy_margin_uses_contract_multiplier(self) -> None:
        strategy = FarMonthFuturesStrategy(initial_capital=1_000_000)
        contract = FuturesContract(
            symbol="CU2607",
            name="铜",
            underlying="CU",
            delivery_month=7,
            delivery_year=2026,
            is_main=False,
            current_price=74_717.64705882352,
            margin_rate=0.17,
        )
        margin, reserve = strategy.calculate_margin_requirement(contract, 1)
        self.assertEqual(contract.contract_multiplier, 5.0)
        self.assertAlmostEqual(margin, 63_510.0, places=2)
        self.assertAlmostEqual(reserve, 63_510.0, places=2)

    def test_position_sizing_does_not_force_one_lot_when_margin_insufficient(self) -> None:
        strategy = FarMonthFuturesStrategy(initial_capital=100_000)
        contract = FuturesContract(
            symbol="CU2607",
            name="铜",
            underlying="CU",
            delivery_month=7,
            delivery_year=2026,
            is_main=False,
            current_price=74_717.64705882352,
            margin_rate=0.17,
        )
        self.assertEqual(strategy.calculate_position_size(contract, available_capital=100_000), 0)
        with self.assertRaises(ValueError):
            strategy.enter_position(contract, PositionSide.LONG, datetime.now(), 100_000)

    def test_hybrid_strategy_margin_uses_contract_multiplier(self) -> None:
        strategy = HybridSwingStrategy(initial_capital=1_000_000)
        contract = FuturesContract(
            symbol="RB2607",
            name="螺纹钢",
            underlying="RB",
            delivery_month=7,
            delivery_year=2026,
            is_main=False,
            current_price=3_500.0,
            margin_rate=0.10,
        )
        margin, reserve = strategy._calculate_futures_margin(contract, 2)
        self.assertEqual(contract.contract_multiplier, 10.0)
        self.assertAlmostEqual(margin, 7_000.0)
        self.assertAlmostEqual(reserve, 7_000.0)

    def test_all_project_china_futures_products_have_multiplier_specs(self) -> None:
        all_products = (
            SHFE_FUTURES_PRODUCTS
            + DCE_FUTURES_PRODUCTS
            + CZCE_FUTURES_PRODUCTS
            + CFFEX_FUTURES_PRODUCTS
            + GFEX_FUTURES_PRODUCTS
        )
        missing = [item["symbol"] for item in all_products if get_futures_contract_spec(item["symbol"]) is None]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
