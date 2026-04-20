"""
Ecosystem V2 - 升级版生态集成管理器

扩展原有生态系统，新增集成：
- Qlib (微软量化投资平台)
- FinRL-X (强化学习交易框架)
- hftbacktest (高频交易回测)
- Bloomberg (彭博数据)

同时保持向后兼容原有功能。
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .strategy_engine import _all_conditions, prepare_frame


class EcosystemIntegrationManagerV2:
    """升级版生态集成管理器"""

    def __init__(self, base_dir: str, github_manager: Any) -> None:
        """
        初始化V2生态管理器

        Args:
            base_dir: 项目基础目录
            github_manager: GitHub项目管理器
        """
        self.base_dir = Path(base_dir)
        self.github_manager = github_manager
        self.state_dir = self.base_dir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # 导出目录
        self.export_dir = self.state_dir / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # 静态资源
        self.static_vendor_dir = self.base_dir / "static" / "vendor"

        # Python 3.11解释器
        self.py311 = self._find_python311()

        # 原有集成（向后兼容）
        self.ccxt = self._optional_import("ccxt", "ccxt")
        self.ta = self._optional_import("ta", "ta-stack")
        self.backtrader = self._optional_import("backtrader", "backtrader")
        self.lightweight_chart_asset = self.static_vendor_dir / "lightweight-charts.standalone.production.js"

        # 新增集成：V2桥接模块
        self.qlib_bridge = self._import_bridge("qlib_bridge", "Qlib")
        self.finrl_bridge = self._import_bridge("finrl_bridge", "FinRL-X")
        self.hftbacktest_bridge = self._import_bridge("hftbacktest_bridge", "hftbacktest")
        self.bloomberg_bridge = self._import_bridge("bloomberg_bridge", "Bloomberg")

        # 原有桥接（向后兼容）
        self.openbb_bridge = self._detect_bridge("openbb", self.base_dir / "quant_trade_system" / "openbb_bridge.py")
        self.quantlib_bridge = self._detect_bridge("QuantLib", self.base_dir / "quant_trade_system" / "quantlib_bridge")
        self.finshare_bridge = self._detect_bridge("finshare", self.base_dir / "quant_trade_system" / "finshare_bridge.py")

        # 标记集成状态
        self._mark_export_integrations()
        if self.lightweight_chart_asset.exists():
            self.github_manager.mark_status_name("lightweight-charts", "tested")
        elif self.lightweight_chart_asset.parent.exists():
            self.github_manager.mark_status_name("lightweight-charts", "integrated")

        # 标记V2新增集成
        self._mark_v2_integrations()

    def _find_python311(self) -> Optional[str]:
        """查找Python 3.11解释器"""
        env_python = os.environ.get("PROJECT_BRIDGE_PYTHON")
        candidates = [
            env_python,
            "/opt/homebrew/bin/python3.11",
            shutil.which("python3.11"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        return None

    def _optional_import(self, module_name: str, project_name: str) -> Optional[Any]:
        """可选导入"""
        try:
            module = importlib.import_module(module_name)
            self.github_manager.mark_status_name(project_name, "tested")
            return module
        except Exception:
            return None

    def _import_bridge(self, module_name: str, project_name: str) -> Optional[Any]:
        """导入桥接模块"""
        try:
            module = importlib.import_module(f".{module_name}", package="quant_trade_system")
            bridge_class = getattr(module, f"{project_name.replace('-', '').replace('Bridge', '')}Bridge", None)
            if bridge_class is None:
                # 尝试其他命名模式
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if attr_name.endswith("Bridge") and isinstance(attr, type):
                        bridge_class = attr
                        break

            if bridge_class:
                instance = bridge_class(str(self.base_dir))
                self.github_manager.mark_status_name(project_name, "tested")
                return instance
        except Exception as e:
            pass
        return None

    def _detect_bridge(self, project_name: str, script_path: Path) -> Optional[Path]:
        """检测桥接脚本"""
        if not self.py311 or not script_path.exists():
            return None
        self.github_manager.mark_status_name(project_name, "integrated")
        try:
            result = subprocess.run(
                [self.py311, str(script_path), "status"],
                capture_output=True,
                text=True,
                timeout=20,
                env=self._bridge_env(project_name.lower()),
                check=False,
            )
        except Exception:
            return None
        if result.returncode == 0:
            self.github_manager.mark_status_name(project_name, "tested")
            return script_path
        return None

    def _bridge_env(self, name: str) -> Dict[str, str]:
        """创建桥接环境"""
        env = os.environ.copy()
        home = self.state_dir / f"{name}_home"
        home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        return env

    def _mark_export_integrations(self) -> None:
        """标记导出集成"""
        export_targets = ["Lean", "Freqtrade", "Hummingbot", "TradingAgents", "XuanYuan"]
        for target in export_targets:
            self.github_manager.mark_status_name(target, "integrated")

    def _mark_v2_integrations(self) -> None:
        """标记V2新增集成"""
        v2_integrations = {
            "qlib_bridge": "Qlib",
            "finrl_bridge": "FinRL-X",
            "hftbacktest_bridge": "hftbacktest",
            "bloomberg_bridge": "Bloomberg",
        }
        for module_name, project_name in v2_integrations.items():
            bridge = getattr(self, f"{module_name}", None)
            if bridge and hasattr(bridge, "available") and bridge.available:
                self.github_manager.mark_status_name(project_name, "tested")
            else:
                self.github_manager.mark_status_name(project_name, "integrated")

    # ========== V2新增功能 ==========

    def get_v2_capabilities(self) -> Dict[str, Any]:
        """获取V2新增能力"""
        return {
            "qlib": {
                "name": "Microsoft Qlib",
                "description": "AI驱动的量化投资平台",
                "available": self.qlib_bridge.available if self.qlib_bridge else False,
                "capabilities": ["ml_models", "backtesting", "portfolio_management", "factor_analysis"],
                "version": self.qlib_bridge.get_version() if self.qlib_bridge and self.qlib_bridge.available else None,
            },
            "finrl_x": {
                "name": "FinRL-X",
                "description": "强化学习交易框架",
                "available": self.finrl_bridge.available if self.finrl_bridge else False,
                "capabilities": ["rl_algorithms", "multi_asset", "custom_rewards", "policy_optimization"],
                "version": self.finrl_bridge.get_version() if self.finrl_bridge and self.finrl_bridge.available else None,
            },
            "hftbacktest": {
                "name": "hftbacktest",
                "description": "高频交易回测库",
                "available": self.hftbacktest_bridge.available if self.hftbacktest_bridge else False,
                "capabilities": ["tick_data", "order_book", "market_microstructure", "low_latency"],
                "version": self.hftbacktest_bridge.get_version() if self.hftbacktest_bridge and self.hftbacktest_bridge.available else None,
            },
            "bloomberg": {
                "name": "Bloomberg",
                "description": "彭博金融数据终端",
                "available": self.bloomberg_bridge.available if self.bloomberg_bridge else False,
                "capabilities": ["real_time_data", "historical_data", "fundamental_data", "news"],
                "version": self.bloomberg_bridge.get_version() if self.bloomberg_bridge and self.bloomberg_bridge.available else None,
                "subscription_required": True,
            },
        }

    def run_qlib_analysis(
        self,
        instrument: str,
        start_date: str,
        end_date: str,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """使用Qlib进行分析"""
        if not self.qlib_bridge or not self.qlib_bridge.available:
            return {
                "success": False,
                "error": "Qlib is not available",
            }

        try:
            # 获取数据
            data = self.qlib_bridge.get_data(
                instrument=instrument,
                start_date=start_date,
                end_date=end_date,
                fields=features or ["$open", "$high", "$low", "$close", "$volume"],
            )

            # 创建模型（简化版）
            model = self.qlib_bridge.create_model(
                model_type="mlp",
                features=features or ["$open", "$high", "$low", "$close", "$volume"],
            )

            return {
                "success": True,
                "data": data,
                "model": model,
                "framework": "qlib",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def run_finrl_training(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        algorithm: str = "ppo",
    ) -> Dict[str, Any]:
        """使用FinRL-X训练智能体"""
        if not self.finrl_bridge or not self.finrl_bridge.available:
            return {
                "success": False,
                "error": "FinRL-X is not available",
            }

        try:
            # 创建交易环境
            env = self.finrl_bridge.create_trading_env(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                initial_amount=100000,
            )

            # 训练智能体
            training_result = self.finrl_bridge.train_agent(
                algorithm=algorithm,
                env_config=env.get("config") if env else {},
                training_config={
                    "total_timesteps": 10000,
                    "learning_rate": 0.0003,
                },
            )

            return {
                "success": True,
                "environment": env,
                "training": training_result,
                "framework": "finrl-x",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def run_hft_backtest(
        self,
        strategy_config: Dict[str, Any],
        data_path: str,
        exchange: str = "binance",
    ) -> Dict[str, Any]:
        """运行高频回测"""
        if not self.hftbacktest_bridge or not self.hftbacktest_bridge.available:
            return {
                "success": False,
                "error": "hftbacktest is not available",
            }

        try:
            # 加载数据
            data = self.hftbacktest_bridge.load_tick_data(
                data_path=data_path,
                exchange=exchange,
            )

            # 运行回测
            backtest_result = self.hftbacktest_bridge.run_backtest(
                strategy_config=strategy_config,
                data_path=data_path,
                initial_capital=100000,
                commission=0.0002,
            )

            # 市场微观结构分析
            microstructure = self.hftbacktest_bridge.analyze_market_microstructure(
                data_path=data_path,
            )

            return {
                "success": True,
                "data": data,
                "backtest": backtest_result,
                "microstructure": microstructure,
                "framework": "hftbacktest",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def get_bloomberg_data(
        self,
        ticker: str,
        fields: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """获取Bloomberg数据"""
        if not self.bloomberg_bridge or not self.bloomberg_bridge.available:
            return {
                "success": False,
                "error": "Bloomberg API is not available. Requires Bloomberg Terminal subscription.",
            }

        try:
            data = self.bloomberg_bridge.get_market_data(
                ticker=ticker,
                fields=fields,
                start_date=start_date,
                end_date=end_date,
            )

            return {
                "success": True,
                "data": data,
                "framework": "bloomberg",
                "subscription_required": True,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    # ========== 原有功能（向后兼容）==========

    def compute_technical_factors(
        self,
        frame: pd.DataFrame,
        shortlist: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """计算技术因子（原有功能）"""
        if frame is None or frame.empty:
            return {"factors": pd.DataFrame()}

        indicators_config = [
            {"name": "rsi_14", "type": "rsi", "window": 14},
            {"name": "macd_diff", "type": "macd_diff"},
            {"name": "adx_14", "type": "adx", "window": 14},
            {"name": "atr_14", "type": "atr", "window": 14},
            {"name": "bb_width", "type": "bb_width", "window": 20},
            {"name": "momentum_20", "type": "momentum", "window": 20},
            {"name": "volume_ratio", "type": "volume_sma", "window": 20},
        ]

        enriched = prepare_frame(frame, {"indicators": indicators_config})

        factor_names = [ind["name"] for ind in indicators_config]
        if not factor_names:
            return {"factors": pd.DataFrame()}

        factors = enriched[factor_names].copy()
        latest_factors = factors.iloc[-1].to_dict() if len(factors) > 0 else {}

        return {
            "factors": factors,
            "latest": latest_factors,
            "count": len(factor_names),
        }

    def export_strategy(
        self,
        strategy_spec: Dict[str, Any],
        target_format: str,
    ) -> Dict[str, Any]:
        """导出策略（原有功能）"""
        export_path = self.export_dir / f"strategy_{target_format.lower()}.json"

        export_config = {
            "format": target_format,
            "strategy": strategy_spec,
            "exported_at": pd.Timestamp.now().isoformat(),
        }

        with open(export_path, "w") as f:
            json.dump(export_config, f, indent=2)

        return {
            "success": True,
            "format": target_format,
            "path": str(export_path),
        }

    def get_all_integrations(self) -> Dict[str, Any]:
        """获取所有集成状态（V1 + V2）"""
        return {
            "v1_integrations": {
                "ccxt": self.ccxt is not None,
                "ta_stack": self.ta is not None,
                "backtrader": self.backtrader is not None,
                "lightweight_charts": self.lightweight_chart_asset.exists(),
                "openbb": self.openbb_bridge is not None,
                "quantlib": self.quantlib_bridge is not None,
                "finshare": self.finshare_bridge is not None,
            },
            "v2_integrations": self.get_v2_capabilities(),
            "export_targets": ["Lean", "Freqtrade", "Hummingbot", "TradingAgents", "XuanYuan"],
        }


# 便捷函数
def create_ecosystem_v2(base_dir: str, github_manager: Any) -> EcosystemIntegrationManagerV2:
    """创建V2生态管理器实例"""
    return EcosystemIntegrationManagerV2(base_dir, github_manager)


if __name__ == "__main__":
    # 测试V2生态管理器
    from unittest.mock import MagicMock

    mock_github_manager = MagicMock()

    ecosystem = create_ecosystem_v2(
        "/Users/caijiawen/Documents/New project/quant-trading-system",
        mock_github_manager
    )

    print("=== Ecosystem V2 Status ===")
    integrations = ecosystem.get_all_integrations()

    print("\nV1 Integrations:")
    for name, status in integrations["v1_integrations"].items():
        print(f"  {name}: {'✓' if status else '✗'}")

    print("\nV2 Integrations:")
    for name, info in integrations["v2_integrations"].items():
        print(f"  {info['name']}: {'✓' if info['available'] else '✗'}")
        if info['available']:
            print(f"    Version: {info.get('version', 'N/A')}")
            print(f"    Capabilities: {', '.join(info['capabilities'])}")
