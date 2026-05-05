from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quant_trade_system.service import QuantTradingService


class QuantSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmpdir.name)
        (self.base_dir / "static").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "static" / "index.html").write_text("<html></html>", encoding="utf-8")
        self.service = QuantTradingService(str(self.base_dir))

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_default_strategies_seeded(self) -> None:
        strategies = self.service.list_strategies()
        self.assertGreaterEqual(len(strategies), 2)

    def test_backtest_runs(self) -> None:
        strategy_id = self.service.list_strategies()[0]["id"]
        result = self.service.backtest_strategy(strategy_id)
        self.assertIn("sharpe", result)
        self.assertIn("equity_curve", result)
        self.assertGreater(len(result["equity_curve"]), 0)

    def test_risk_blocks_oversized_order(self) -> None:
        strategy = self.service.list_strategies()[0]
        result = self.service.submit_order(
            {
                "strategy_id": strategy["id"],
                "dataset": strategy["dataset"],
                "symbol": strategy["spec"]["symbol"],
                "side": "buy",
                "quantity": 999999,
                "broker_mode": "paper",
            }
        )
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(result["risk"]["violations"])

    def test_causal_status_available(self) -> None:
        status = self.service.causal_status()
        self.assertEqual(status["system_name"], "因果AI量化交易系统")
        self.assertIn("github_projects", status)
        self.assertIn("self_iterating_learning", status)

    def test_causal_pipeline_runs(self) -> None:
        result = self.service.run_causal_pipeline(["AAPL", "MSFT", "GOOGL"])
        self.assertIn("causal_graph", result)
        self.assertIn("decision", result)
        self.assertIn("market_data", result)
        self.assertIn("self_iterating_cycle", result)
        self.assertIn("legacy_decision", result)

    def test_market_data_includes_external_columns(self) -> None:
        causal_system = self.service.causal_system
        causal_system.ecosystem.fetch_akshare_market_context = lambda payload: {
            "macro": {
                "lpr_1y": {"value": 3.1},
                "shibor_1y": {"value": 1.8},
                "m2_yoy": {"value": 8.4},
                "gdp_yoy": {"value": 5.2},
                "cpi_yoy": {"value": 0.6},
                "ppi_yoy": {"value": -1.1},
            },
            "inventory": {
                "shfe_warehouse_receipt": {"records": [{"品种": "铜", "仓单数量": 120}]},
                "沪铜_inventory": {"value": 96000},
                "沪金_inventory": {"value": 1850},
            },
            "valuation": {
                "a_share_pe_ttm": {"value": 11.2},
                "a_share_pb": {"value": 0.74},
                "hk_pe_ttm": {"value": 18.6},
            },
            "holdings": {
                "northbound_rank": {
                    "count": 2,
                    "records": [
                        {"今日持股-市值": "150000000", "今日增持-估计市值": "12000000"},
                        {"今日持股-市值": "90000000", "今日增持-估计市值": "-5000000"},
                    ],
                },
            },
            "policy": {
                "event_count": {"value": 7},
                "calendar": {"date": "20260505", "records": [{"事件": "LPR 公布"}]},
            },
        }
        causal_system.data_adapter.get_batch_snapshots = lambda symbols: {}
        causal_system.ecosystem.fetch_openbb_market_context = lambda symbols: {}
        market_data = causal_system.build_market_data({})

        self.assertEqual(market_data["macro_columns"]["China_M2_YoY"]["value"], 8.4)
        self.assertEqual(market_data["inventory_columns"]["Copper_Inventory"]["value"], 96000)
        self.assertEqual(market_data["valuation_columns"]["A_Share_PE_TTM"]["value"], 11.2)
        self.assertEqual(market_data["holding_columns"]["Northbound_Top5_Holding_Value"]["value"], 240000000.0)
        self.assertEqual(market_data["policy_columns"]["Policy_Event_Count"]["value"], 7.0)
        self.assertIn("akshare_market_context", market_data)

    def test_data_adapter_returns_source_column(self) -> None:
        frame = self.service.causal_system.data_adapter.get_symbol_data("AAPL", period="1mo")
        self.assertIn("Source", frame.columns)
        self.assertGreater(len(frame), 0)

    def test_ecosystem_exports_and_series(self) -> None:
        strategy_id = self.service.list_strategies()[0]["id"]
        export_result = self.service.export_strategy(strategy_id, "lean")
        self.assertEqual(export_result["target"], "lean")
        self.assertTrue(Path(export_result["path"]).exists())
        series = self.service.dataset_series("gold_daily")
        self.assertIn("series", series)
        self.assertGreater(len(series["series"]), 0)


if __name__ == "__main__":
    unittest.main()
