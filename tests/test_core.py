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

    def test_causal_pipeline_runs(self) -> None:
        result = self.service.run_causal_pipeline(["AAPL", "MSFT", "GOOGL"])
        self.assertIn("causal_graph", result)
        self.assertIn("decision", result)
        self.assertIn("market_data", result)

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
