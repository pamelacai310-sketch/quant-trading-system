"""
V2升级模块测试

测试新增的桥接模块：
- Qlib Bridge
- FinRL-X Bridge
- hftbacktest Bridge
- Bloomberg Bridge
- ecosystem_v2
"""

import unittest
import sys
from pathlib import Path

import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from quant_trade_system.qlib_bridge import QlibBridge, create_qlib_bridge
    from quant_trade_system.finrl_bridge import FinRLBridge, create_finrl_bridge
    from quant_trade_system.hftbacktest_bridge import HFTBacktestBridge, create_hftbacktest_bridge
    from quant_trade_system.bloomberg_bridge import BloombergBridge, create_bloomberg_bridge
    from quant_trade_system.ecosystem_v2 import EcosystemIntegrationManagerV2, create_ecosystem_v2
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录运行测试")


class TestQlibBridge(unittest.TestCase):
    """测试Qlib桥接器"""

    def setUp(self):
        """设置测试"""
        self.bridge = create_qlib_bridge(str(project_root))

    def test_bridge_creation(self):
        """测试桥接器创建"""
        self.assertIsNotNone(self.bridge)
        self.assertIsInstance(self.bridge, QlibBridge)

    def test_get_status(self):
        """测试获取状态"""
        status = self.bridge.get_status()
        self.assertIsInstance(status, dict)
        self.assertIn("installed", status)
        self.assertIn("version", status)

    def test_python_detection(self):
        """测试Python检测"""
        # 如果Python 3.11可用，应该被检测到
        if self.bridge.python:
            self.assertTrue(Path(self.bridge.python).exists())

    def test_workflow_config_generation_is_deterministic(self):
        """Qlib-style workflow config should be stable for identical inputs."""
        spec_a = self.bridge.build_workflow_spec(
            universe=["BBB", "AAA"],
            start_date="2026-01-01",
            end_date="2026-03-31",
            benchmark="QQQ",
        )
        spec_b = self.bridge.build_workflow_spec(
            universe=["AAA", "BBB"],
            start_date="2026-01-01",
            end_date="2026-03-31",
            benchmark="QQQ",
        )
        self.assertEqual(spec_a["workflow_id"], spec_b["workflow_id"])
        self.assertEqual(self.bridge.build_workflow_config(spec_a), self.bridge.build_workflow_config(spec_b))
        self.assertIn("feature_registry", spec_a)
        self.assertIn("leakage_policy", spec_a)

    def test_workflow_runs_without_pyqlib_as_shadow_config(self):
        """pyqlib is optional; the bridge must still create an auditable workflow."""
        spec = self.bridge.build_workflow_spec(
            universe=["AAA"],
            start_date="2026-01-01",
            end_date="2026-02-28",
        )
        original_available = self.bridge.available
        self.bridge.available = False
        try:
            result = self.bridge.run_workflow_config(spec, persist=False)
        finally:
            self.bridge.available = original_available
        self.assertTrue(result["success"])
        self.assertFalse(result["available"])
        self.assertEqual(result["mode"], "shadow_config_only")
        self.assertEqual(result["metrics"]["status"], "no_market_data")
        self.assertTrue(result["leakage_report"]["passed"])

    def test_shadow_workflow_with_synthetic_data_outputs_predictions_and_backtest(self):
        """Synthetic OHLCV data should produce predictions, metrics and a shadow backtest."""
        dates = pd.date_range("2026-01-01", periods=80, freq="D")
        base = pd.Series(range(80), dtype=float)
        frame_a = pd.DataFrame(
            {
                "date": dates,
                "open": 100 + base * 0.2,
                "high": 101 + base * 0.2,
                "low": 99 + base * 0.2,
                "close": 100 + base * 0.25,
                "volume": 1_000_000 + base * 1000,
            }
        )
        frame_b = pd.DataFrame(
            {
                "date": dates,
                "open": 80 + base * 0.05,
                "high": 81 + base * 0.05,
                "low": 79 + base * 0.05,
                "close": 80 + base * 0.03,
                "volume": 800_000 + base * 500,
            }
        )
        spec = self.bridge.build_workflow_spec(
            universe=["AAA", "BBB"],
            start_date="2026-01-01",
            end_date="2026-03-21",
        )
        result = self.bridge.run_workflow_config(
            spec,
            market_data={"AAA": frame_a, "BBB": frame_b},
            persist=False,
        )
        self.assertTrue(result["success"])
        self.assertGreater(len(result["predictions"]), 0)
        self.assertEqual(result["metrics"]["status"], "evaluated")
        self.assertEqual(result["backtest"]["status"], "evaluated")
        self.assertTrue(result["leakage_report"]["label_excluded_from_inference"])
        self.assertEqual(result["leakage_report"]["future_feature_formulas"], [])

    def test_export_predictions_and_recorder_listing(self):
        """Recorder artifacts should be discoverable after a persisted shadow run."""
        spec = self.bridge.build_workflow_spec(
            universe=["AAA"],
            start_date="2026-01-01",
            end_date="2026-01-15",
        )
        result = self.bridge.run_workflow_config(spec, persist=True)
        exported = self.bridge.export_predictions(result)
        artifacts = self.bridge.get_recorder_artifacts(spec["workflow_id"])
        self.assertTrue(Path(exported["path"]).exists())
        self.assertTrue(artifacts["exists"])
        self.assertTrue(any(path.endswith("workflow_spec.json") for path in artifacts["files"]))


class TestFinRLBridge(unittest.TestCase):
    """测试FinRL-X桥接器"""

    def setUp(self):
        """设置测试"""
        self.bridge = create_finrl_bridge(str(project_root))

    def test_bridge_creation(self):
        """测试桥接器创建"""
        self.assertIsNotNone(self.bridge)
        self.assertIsInstance(self.bridge, FinRLBridge)

    def test_get_status(self):
        """测试获取状态"""
        status = self.bridge.get_status()
        self.assertIsInstance(status, dict)
        self.assertIn("installed", status)
        self.assertIn("available_algorithms", status)

    def test_algorithms_list(self):
        """测试算法列表"""
        algorithms = self.bridge.get_available_algorithms()
        self.assertIsInstance(algorithms, list)
        self.assertIn("ppo", algorithms)
        self.assertIn("dqn", algorithms)


class TestHFTBacktestBridge(unittest.TestCase):
    """测试hftbacktest桥接器"""

    def setUp(self):
        """设置测试"""
        self.bridge = create_hftbacktest_bridge(str(project_root))

    def test_bridge_creation(self):
        """测试桥接器创建"""
        self.assertIsNotNone(self.bridge)
        self.assertIsInstance(self.bridge, HFTBacktestBridge)

    def test_get_status(self):
        """测试获取状态"""
        status = self.bridge.get_status()
        self.assertIsInstance(status, dict)
        self.assertIn("installed", status)
        self.assertIn("supported_exchanges", status)

    def test_exchanges_list(self):
        """测试支持的交易所列表"""
        exchanges = self.bridge.get_supported_exchanges()
        self.assertIsInstance(exchanges, list)
        self.assertIn("binance", exchanges)

    def test_strategy_templates(self):
        """测试策略模板"""
        templates = [
            "market_making",
            "mean_reversion",
            "momentum",
            "arbitrage",
        ]
        for template_type in templates:
            template = self.bridge.get_strategy_template(template_type)
            self.assertIsNotNone(template)
            self.assertIn("template", template)


class TestBloombergBridge(unittest.TestCase):
    """测试Bloomberg桥接器"""

    def setUp(self):
        """设置测试"""
        self.bridge = create_bloomberg_bridge(str(project_root))

    def test_bridge_creation(self):
        """测试桥接器创建"""
        self.assertIsNotNone(self.bridge)
        self.assertIsInstance(self.bridge, BloombergBridge)

    def test_get_status(self):
        """测试获取状态"""
        status = self.bridge.get_status()
        self.assertIsInstance(status, dict)
        self.assertIn("installed", status)
        self.assertIn("subscription_required", status)
        self.assertTrue(status["subscription_required"])

    def test_subscription_info(self):
        """测试订阅信息"""
        status = self.bridge.get_status()
        self.assertIn("subscription_url", status)
        self.assertIn("api_documentation", status)


class TestEcosystemV2(unittest.TestCase):
    """测试V2生态系统管理器"""

    def setUp(self):
        """设置测试"""
        # 创建mock GitHub管理器
        class MockGitHubManager:
            def __init__(self):
                self.statuses = {}

            def mark_status_name(self, name, status):
                self.statuses[name] = status

        self.github_manager = MockGitHubManager()
        self.ecosystem = create_ecosystem_v2(str(project_root), self.github_manager)

    def test_ecosystem_creation(self):
        """测试生态系统创建"""
        self.assertIsNotNone(self.ecosystem)
        self.assertIsInstance(self.ecosystem, EcosystemIntegrationManagerV2)

    def test_v2_capabilities(self):
        """测试V2能力"""
        capabilities = self.ecosystem.get_v2_capabilities()
        self.assertIsInstance(capabilities, dict)

        v2_systems = ["qlib", "finrl_x", "hftbacktest", "bloomberg"]
        for system in v2_systems:
            self.assertIn(system, capabilities)
            self.assertIn("name", capabilities[system])
            self.assertIn("description", capabilities[system])
            self.assertIn("available", capabilities[system])

    def test_all_integrations(self):
        """测试所有集成"""
        integrations = self.ecosystem.get_all_integrations()
        self.assertIsInstance(integrations, dict)
        self.assertIn("v1_integrations", integrations)
        self.assertIn("v2_integrations", integrations)
        self.assertIn("export_targets", integrations)

    def test_bridge_instances(self):
        """测试桥接实例"""
        # V2桥接应该被实例化
        self.assertIsNotNone(self.ecosystem.qlib_bridge)
        self.assertIsNotNone(self.ecosystem.finrl_bridge)
        self.assertIsNotNone(self.ecosystem.hftbacktest_bridge)
        self.assertIsNotNone(self.ecosystem.bloomberg_bridge)

    def test_run_qlib_analysis_returns_workflow_result(self):
        """Ecosystem Qlib entrypoint should expose workflow artifacts and audit metrics."""
        dates = pd.date_range("2026-01-01", periods=45, freq="D")
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": [100 + idx * 0.1 for idx in range(45)],
                "high": [101 + idx * 0.1 for idx in range(45)],
                "low": [99 + idx * 0.1 for idx in range(45)],
                "close": [100 + idx * 0.15 for idx in range(45)],
                "volume": [1_000_000 + idx * 1000 for idx in range(45)],
            }
        )
        result = self.ecosystem.run_qlib_analysis(
            instrument="AAA",
            start_date="2026-01-01",
            end_date="2026-02-14",
            market_data={"AAA": frame},
            benchmark="QQQ",
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["framework"], "qlib")
        self.assertIn("workflow", result)
        self.assertIn("workflow_config", result["workflow"])
        self.assertIn("promotion_status", result)
        self.assertTrue(result["leakage_report"]["label_excluded_from_inference"])


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        """设置测试"""
        class MockGitHubManager:
            def __init__(self):
                self.statuses = {}

            def mark_status_name(self, name, status):
                self.statuses[name] = status

        self.github_manager = MockGitHubManager()
        self.ecosystem = create_ecosystem_v2(str(project_root), self.github_manager)

    def test_module_availability(self):
        """测试模块可用性"""
        v2_caps = self.ecosystem.get_v2_capabilities()

        print("\n=== V2模块可用性 ===")
        for system_name, info in v2_caps.items():
            status = "✓" if info["available"] else "✗"
            print(f"{status} {info['name']}: {info['description']}")
            if info["available"]:
                print(f"  版本: {info.get('version', 'N/A')}")
                print(f"  能力: {', '.join(info['capabilities'])}")

    def test_file_structure(self):
        """测试文件结构"""
        bridge_files = [
            "quant_trade_system/qlib_bridge.py",
            "quant_trade_system/finrl_bridge.py",
            "quant_trade_system/hftbacktest_bridge.py",
            "quant_trade_system/bloomberg_bridge.py",
            "quant_trade_system/ecosystem_v2.py",
            "install_upgrades.sh",
            "tests/test_upgrades.py",
        ]

        print("\n=== 文件结构检查 ===")
        for file_path in bridge_files:
            full_path = project_root / file_path
            exists = "✓" if full_path.exists() else "✗"
            print(f"{exists} {file_path}")
            self.assertTrue(full_path.exists(), f"文件不存在: {file_path}")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("量化交易系统 V2 升级测试")
    print("=" * 60)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestQlibBridge))
    suite.addTests(loader.loadTestsFromTestCase(TestFinRLBridge))
    suite.addTests(loader.loadTestsFromTestCase(TestHFTBacktestBridge))
    suite.addTests(loader.loadTestsFromTestCase(TestBloombergBridge))
    suite.addTests(loader.loadTestsFromTestCase(TestEcosystemV2))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
