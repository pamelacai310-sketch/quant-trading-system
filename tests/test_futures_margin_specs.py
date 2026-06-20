from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

from quant_trade_system.futures_specs import (
    build_one_lot_margin_table,
    calculate_futures_margin,
    futures_contract_multiplier,
    get_futures_contract_spec,
)
from quant_trade_system.market_universe import (
    ALL_CN_FUTURES_PRODUCTS,
    CN_FUTURES_ALIAS_MAP,
    CN_FUTURES_PRODUCTS_BY_EXCHANGE,
    SHFE_FUTURES_PRODUCTS,
    get_market_universe_summary,
)
from quant_trade_system.nightly_quant_orders import _cn_futures_exchange_universe
from quant_trade_system.universe_provider import (
    CFFEX_FUTURES_PRODUCTS,
    CZCE_FUTURES_PRODUCTS,
    DCE_FUTURES_PRODUCTS,
    GFEX_FUTURES_PRODUCTS,
    MarketUniverseProvider,
)
from quant_trade_system.strategies.far_month_futures_strategy import FuturesContract, FarMonthFuturesStrategy, PositionSide
from quant_trade_system.strategies.hybrid_swing_strategy import HybridSwingStrategy


class FuturesMarginSpecTests(unittest.TestCase):
    IMAGE_FUTURES_PRODUCTS = {
        "IF", "IH", "IC", "TS", "TF", "T", "PB", "HC", "ZC", "SC", "L", "V",
        "FG", "NR", "SR", "PM", "P", "CS", "IM", "CU", "SN", "SS", "SM", "FU",
        "PP", "TA", "SP", "CF", "OI", "JR", "A", "RR", "SI", "AL", "AU", "I",
        "SF", "LU", "EG", "SA", "BB", "CY", "RM", "RI", "B", "JD", "ZN", "AG",
        "J", "WR", "BU", "PG", "UR", "FB", "AP", "RS", "LR", "M", "LH", "NI",
        "RB", "JM", "BC", "MA", "EB", "PF", "RU", "CJ", "WH", "Y", "C", "PK",
    }

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

    def test_image_futures_products_have_specs(self) -> None:
        missing = [symbol for symbol in sorted(self.IMAGE_FUTURES_PRODUCTS) if get_futures_contract_spec(symbol) is None]
        self.assertEqual(missing, [])

    def test_new_czce_legacy_agricultural_specs(self) -> None:
        expected = {
            "PM": ("普麦", 50.0, 0.05),
            "WH": ("强麦", 20.0, 0.05),
            "JR": ("粳稻", 20.0, 0.05),
            "RI": ("早籼稻", 20.0, 0.05),
            "LR": ("晚籼稻", 20.0, 0.05),
            "RS": ("菜籽", 10.0, 0.05),
        }
        for symbol, (name, multiplier, margin_rate) in expected.items():
            with self.subTest(symbol=symbol):
                spec = get_futures_contract_spec(symbol)
                self.assertIsNotNone(spec)
                assert spec is not None
                self.assertEqual(spec.exchange, "CZCE")
                self.assertEqual(spec.name, name)
                self.assertEqual(spec.multiplier, multiplier)
                self.assertEqual(spec.exchange_min_margin_rate, margin_rate)

    def test_one_lot_margin_table_handles_new_czce_products(self) -> None:
        prices = {"PM0": 2600.0, "WH0": 2800.0, "JR0": 3000.0, "RI0": 2700.0, "LR0": 2750.0, "RS0": 6200.0}
        rows = build_one_lot_margin_table(prices)
        by_product = {row["product"]: row for row in rows}
        self.assertEqual(sorted(by_product), ["JR", "LR", "PM", "RI", "RS", "WH"])
        self.assertAlmostEqual(by_product["PM"]["one_lot_margin"], 2600.0 * 50.0 * 0.05)
        self.assertAlmostEqual(by_product["RS"]["one_lot_margin"], 6200.0 * 10.0 * 0.05)

    def test_cn_futures_provider_includes_image_products(self) -> None:
        provider = MarketUniverseProvider(prefer_live=False)
        symbols = {item.symbol for item in provider.get_universe("cn_futures_products", include_contracts=False)}
        self.assertTrue(self.IMAGE_FUTURES_PRODUCTS.issubset(symbols))

    def test_nightly_exchange_universe_groups_new_products_under_czce(self) -> None:
        exchange_universe = _cn_futures_exchange_universe()
        czce_symbols = set(exchange_universe["CZCE"])
        for symbol in ["PM0", "WH0", "JR0", "RI0", "LR0", "RS0"]:
            self.assertIn(symbol, czce_symbols)

    def test_market_universe_summary_has_china_futures_by_exchange(self) -> None:
        with patch("quant_trade_system.market_universe.get_hang_seng_symbols", return_value=["00700.HK"]):
            summary = get_market_universe_summary()["china_futures"]
        self.assertEqual(set(summary["by_exchange"]), {"SHFE", "INE", "DCE", "CZCE", "CFFEX", "GFEX"})
        self.assertEqual(len(summary["symbols"]), len(set(summary["symbols"])))
        self.assertEqual(summary["count"], len(ALL_CN_FUTURES_PRODUCTS))
        self.assertEqual(summary["by_exchange"]["CZCE"]["count"], len(CN_FUTURES_PRODUCTS_BY_EXCHANGE["CZCE"]))

    def test_explicit_image_aliases_are_mapped(self) -> None:
        expected_aliases = {
            "热轧卷": "HC",
            "塑料": "L",
            "胶板": "BB",
            "橡胶": "RU",
            "中证1000": "IM",
            "IF股指": "IF",
            "IH股指": "IH",
            "IC股指": "IC",
            "2年期国债": "TS",
            "5年期国债": "TF",
            "10年期国债": "T",
        }
        for alias, symbol in expected_aliases.items():
            self.assertEqual(CN_FUTURES_ALIAS_MAP[alias], symbol)


if __name__ == "__main__":
    unittest.main()
