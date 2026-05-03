"""
FinceptTerminal 集成桥接模块

将量化交易系统与FinceptTerminal集成的桥接层。

功能：
1. 数据同步：从FinceptTerminal获取市场数据
2. 策略导出：将系统策略导出为FinceptTerminal格式
3. Python API：通过Fincept的嵌入Python环境运行策略
4. 信号推送：将交易信号推送到FinceptTerminal
"""

import sys
import os
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass
class FinceptConfig:
    """FinceptTerminal配置"""
    fincept_path: Optional[str] = None
    python_executable: Optional[str] = None
    api_enabled: bool = False
    api_port: int = 8080
    data_connectors: List[str] = field(default_factory=lambda: [
        "yahoo_finance",
        "polygon",
        "fred",
    ])


class FinceptDataBridge:
    """
    FinceptTerminal数据桥接器

    从FinceptTerminal获取市场数据和经济数据
    """

    def __init__(self, config: FinceptConfig):
        self.config = config
        self.available_connectors = self._detect_available_connectors()

    def _detect_available_connectors(self) -> List[str]:
        """检测可用的Fincept数据连接器"""
        # FinceptTerminal支持100+数据源
        # 这里我们主要使用：
        # - Yahoo Finance（股票、ETF、指数）
        # - FRED（经济数据）
        # - Polygon（实时行情）
        # - DBnomics（国际数据）

        connectors = []

        # 检查Fincept是否安装
        if self.config.fincept_path and os.path.exists(self.config.fincept_path):
            connectors.extend(["yahoo_finance", "fred", "polygon"])
        else:
            # Fincept未安装，使用回退方案
            connectors.extend(["yahoo_finance_fallback", "local_cache"])

        return connectors

    def fetch_market_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        connector: str = "yahoo_finance",
    ) -> pd.DataFrame:
        """
        从FinceptTerminal获取市场数据

        参数:
            symbol: 标的代码
            start_date: 开始日期
            end_date: 结束日期
            connector: 数据连接器
        """

        # 如果FinceptTerminal可用，使用其Python API
        if self.config.fincept_path and os.path.exists(self.config.fincept_path):
            return self._fetch_via_fincept_python(symbol, start_date, end_date)

        # 否则使用回退方案（直接使用yfinance等库）
        return self._fetch_via_fallback(symbol, start_date, end_date, connector)

    def _fetch_via_fincept_python(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """通过Fincept的嵌入Python环境获取数据"""

        # FinceptTerminal嵌入Python 3.11
        # 可以直接调用其数据API

        python_code = f"""
import sys
sys.path.insert(0, '{self.config.fincept_path}/python')

# Fincept的数据API（模拟，实际需要根据Fincept的API文档）
try:
    from fincept_data import MarketDataClient

    client = MarketDataClient()
    data = client.get_historical_data(
        symbol='{symbol}',
        start='{start_date}',
        end='{end_date}',
    )

    import pandas as pd
    df = pd.DataFrame(data)
    print(df.to_json())
except ImportError:
    # 如果Fincept API不可用，使用yfinance
    import yfinance as yf
    ticker = yf.Ticker('{symbol}')
    df = ticker.history(start='{start_date}', end='{end_date}')
    print(df.to_json())
"""

        result = subprocess.run(
            [self.config.python_executable or "python3", "-c", python_code],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            try:
                data_json = result.stdout.strip()
                return pd.read_json(data_json)
            except Exception as e:
                print(f"解析Fincept数据失败: {e}")

        # 回退到本地方案
        return self._fetch_via_fallback(symbol, start_date, end_date, "yahoo_finance")

    def _fetch_via_fallback(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        connector: str,
    ) -> pd.DataFrame:
        """回退方案：直接使用数据源"""

        if connector in ["yahoo_finance", "yahoo_finance_fallback"]:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                return df
            except ImportError:
                print("yfinance未安装，使用模拟数据")
                return self._generate_mock_data(symbol, start_date, end_date)

        return self._generate_mock_data(symbol, start_date, end_date)

    def _generate_mock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """生成模拟数据（当所有数据源都不可用时）"""

        import numpy as np
        from datetime import timedelta

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='D')

        np.random.seed(42)
        price = 100.0
        prices = []
        volumes = []

        for _ in range(len(dates)):
            daily_return = np.random.normal(0.0005, 0.02)
            price *= (1 + daily_return)
            prices.append(price)
            volumes.append(np.random.randint(100000, 5000000))

        df = pd.DataFrame({
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': volumes,
        }, index=dates)

        return df

    def fetch_economic_data(
        self,
        indicator: str,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """
        获取经济数据（通过FRED等）

        参数:
            indicator: 指标代码（如 'GDP', 'UNRATE', 'CPIAUCSL'）
            start_date: 开始日期
            end_date: 结束日期
        """

        # 如果Fincept可用，使用其FRED连接器
        if self.config.fincept_path and os.path.exists(self.config.fincept_path):
            python_code = f"""
import pandas as pd
try:
    from fincept_data import FredClient
    client = FredClient()
    data = client.get_series(
        series_id='{indicator}',
        start='{start_date}',
        end='{end_date}',
    )
    print(data.to_json())
except:
    # 回退到pandas-datareader
    import pandas_datareader.datareader as web
    from datetime import datetime
    data = web.DataReader('{indicator}', 'fred', '{start_date}', '{end_date}')
    print(data.to_json())
"""
            result = subprocess.run(
                [self.config.python_executable or "python3", "-c", python_code],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                try:
                    return pd.read_json(result.stdout.strip())
                except:
                    pass

        # 回退方案
        try:
            import pandas_datareader.datareader as web
            return web.DataReader(indicator, 'fred', start_date, end_date)
        except:
            # 生成模拟数据
            dates = pd.date_range(start=start_date, end=end_date, freq='M')
            return pd.Series(
                np.random.normal(100, 10, len(dates)),
                index=dates,
                name=indicator,
            )


class FinceptStrategyExporter:
    """
    FinceptTerminal策略导出器

    将系统的策略导出为FinceptTerminal可识别的格式
    """

    def __init__(self):
        self.supported_formats = [
            "fincept_json",
            "fincept_python",
            "fincept_workflow",
        ]

    def export_strategy(
        self,
        strategy_config: Dict[str, Any],
        export_format: str = "fincept_json",
    ) -> str:
        """
        导出策略到FinceptTerminal格式

        参数:
            strategy_config: 策略配置（JSON格式）
            export_format: 导出格式
        """

        if export_format == "fincept_json":
            return self._export_as_fincept_json(strategy_config)
        elif export_format == "fincept_python":
            return self._export_as_fincept_python(strategy_config)
        elif export_format == "fincept_workflow":
            return self._export_as_fincept_workflow(strategy_config)
        else:
            raise ValueError(f"不支持的格式: {export_format}")

    def _export_as_fincept_json(self, strategy_config: Dict[str, Any]) -> str:
        """导出为Fincept JSON格式"""

        fincept_config = {
            "version": "4.0",
            "type": "trading_strategy",
            "name": strategy_config.get("name", "Unnamed Strategy"),
            "description": strategy_config.get("description", ""),
            "created_by": "quant-trading-system",
            "created_at": datetime.now().isoformat(),

            "data_sources": strategy_config.get("dataset", "yahoo_finance"),

            "indicators": [
                {
                    "name": ind.get("name"),
                    "type": ind.get("type", "sma"),
                    "params": {
                        "window": ind.get("window", 20),
                    },
                }
                for ind in strategy_config.get("indicators", [])
            ],

            "rules": {
                "entry": [
                    {
                        "condition": f"{rule.get('left')} {rule.get('op')} {rule.get('right')}",
                        "action": "buy",
                    }
                    for rule in strategy_config.get("entry_rules", [])
                ],
                "exit": [
                    {
                        "condition": f"{rule.get('left')} {rule.get('op')} {rule.get('right')}",
                        "action": "sell",
                    }
                    for rule in strategy_config.get("exit_rules", [])
                ],
            },

            "risk_management": {
                "position_sizing": strategy_config.get("position_sizing", {}),
                "stop_loss": strategy_config.get("risk_limits", {}).get("stop_loss_pct", 0.05),
                "take_profit": strategy_config.get("risk_limits", {}).get("take_profit_pct", 0.12),
            },
        }

        return json.dumps(fincept_config, indent=2)

    def _export_as_fincept_python(self, strategy_config: Dict[str, Any]) -> str:
        """导出为Fincept Python脚本"""

        strategy_name_safe = strategy_config.get("name", "Strategy").replace(" ", "_")
        strategy_name = strategy_config.get("name", "Unnamed Strategy")
        strategy_desc = strategy_config.get("description", "")

        script = f'''
# FinceptTerminal Python Strategy
# Generated by quant-trading-system

import pandas as pd
import numpy as np

class {strategy_name_safe}_Strategy:
    \"\"\"
    {strategy_desc}
    \"\"\"

    def __init__(self):
        self.name = "{strategy_name}"
        self.indicators = {{}}

    def initialize(self, data):
        \"\"\"初始化指标\"\"\"
        # 添加指标
'''

        # 添加指标初始化代码
        for ind in strategy_config.get("indicators", []):
            ind_name = ind.get("name")
            ind_type = ind.get("type", "sma")
            window = ind.get("window", 20)

            script += f"""
        if "{ind_type}" == "sma":
            self.indicators["{ind_name}"] = data["Close"].rolling({window}).mean()
        elif "{ind_type}" == "ema":
            self.indicators["{ind_name}"] = data["Close"].ewm(span={window}).mean()
        elif "{ind_type}" == "rsi":
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window={window}).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window={window}).mean()
            rs = gain / loss
            self.indicators["{ind_name}"] = 100 - (100 / (1 + rs))
"""

        # 添加入场规则
        script += """

    def generate_signals(self, data):
        \"\"\"生成交易信号\"\"\"
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0  # 0=hold, 1=buy, -1=sell

"""

        for rule in strategy_config.get("entry_rules", []):
            left = rule.get("left")
            op = rule.get("op")
            right = rule.get("right")

            python_op = {
                "crosses_above": ">",
                "crosses_below": "<",
                ">": ">",
                "<": "<",
                ">=": ">=",
                "<=": "<=",
            }.get(op, "==")

            script += f"""
        # 入场: {left} {op} {right}
        entry_condition = (self.indicators.get("{left}", data["{left}"]) {python_op} self.indicators.get("{right}", data["{right}"]))
        signals.loc[entry_condition, "signal"] = 1
"""

        script += """

        return signals

    def run(self, data):
        \"\"\"运行策略\"\"\"
        self.initialize(data)
        return self.generate_signals(data)

# 使用示例
if __name__ == "__main__":
    strategy = """ + strategy_name_safe + """_Strategy()

    # 加载数据
    data = pd.read_csv("data.csv", index_col="Date", parse_dates=True)

    # 运行策略
    signals = strategy.run(data)
    print(signals)
"""

        return script

    def _export_as_fincept_workflow(self, strategy_config: Dict[str, Any]) -> str:
        """导出为Fincept可视化工作流"""

        workflow = {
            "nodes": [
                {
                    "id": "data_source",
                    "type": "DataSource",
                    "config": {
                        "connector": strategy_config.get("dataset", "yahoo_finance"),
                        "symbol": strategy_config.get("symbol", "AAPL"),
                    },
                },
            ],
            "edges": [],
        }

        # 添加指标节点
        for i, ind in enumerate(strategy_config.get("indicators", [])):
            workflow["nodes"].append({
                "id": f"indicator_{i}",
                "type": "Indicator",
                "config": {
                    "indicator_type": ind.get("type"),
                    "params": {"window": ind.get("window")},
                },
            })
            workflow["edges"].append({
                "from": "data_source",
                "to": f"indicator_{i}",
            })

        # 添加规则节点
        workflow["nodes"].append({
            "id": "entry_rules",
            "type": "RuleEngine",
            "config": {
                "rules": strategy_config.get("entry_rules", []),
            },
        })

        return json.dumps(workflow, indent=2)


class FinceptSignalPusher:
    """
    FinceptTerminal信号推送器

    将交易信号推送到FinceptTerminal
    """

    def __init__(self, config: FinceptConfig):
        self.config = config
        self.signal_queue: List[Dict] = []

    def push_signal(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        reason: str,
    ) -> bool:
        """
        推送交易信号到FinceptTerminal

        参数:
            symbol: 标的代码
            action: 动作（buy/sell）
            quantity: 数量
            price: 价格
            reason: 原因
        """

        signal = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "reason": reason,
            "source": "quant-trading-system",
        }

        self.signal_queue.append(signal)

        # 如果Fincept API可用，直接推送
        if self.config.api_enabled:
            return self._push_to_fincept_api(signal)

        # 否则保存到文件，等待导入
        return self._save_to_file(signal)

    def _push_to_fincept_api(self, signal: Dict) -> bool:
        """通过Fincept API推送信号"""

        # FinceptTerminal的REST API（如果启用）
        try:
            import requests

            url = f"http://localhost:{self.config.api_port}/api/signals"
            response = requests.post(url, json=signal, timeout=5)

            return response.status_code == 200
        except:
            return False

    def _save_to_file(self, signal: Dict) -> bool:
        """保存信号到文件"""

        try:
            signals_dir = Path("fincept_signals")
            signals_dir.mkdir(exist_ok=True)

            signal_file = signals_dir / f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            with open(signal_file, 'w') as f:
                json.dump(signal, f, indent=2)

            return True
        except Exception as e:
            print(f"保存信号文件失败: {e}")
            return False


class FinceptIntegrator:
    """
    FinceptTerminal集成主控制器

    整合数据桥接、策略导出和信号推送功能
    """

    def __init__(self, config: Optional[FinceptConfig] = None):
        self.config = config or FinceptConfig()

        self.data_bridge = FinceptDataBridge(self.config)
        self.strategy_exporter = FinceptStrategyExporter()
        self.signal_pusher = FinceptSignalPusher(self.config)

    def check_integration_status(self) -> Dict[str, Any]:
        """检查FinceptTerminal集成状态"""

        status = {
            "fincept_installed": False,
            "fincept_path": self.config.fincept_path,
            "python_available": False,
            "available_connectors": self.data_bridge.available_connectors,
            "api_enabled": self.config.api_enabled,
            "api_port": self.config.api_port,
            "supported_formats": self.strategy_exporter.supported_formats,
        }

        # 检查Fincept是否安装
        if self.config.fincept_path:
            status["fincept_installed"] = os.path.exists(self.config.fincept_path)

        # 检查Python是否可用
        if self.config.python_executable:
            status["python_available"] = os.path.exists(self.config.python_executable)

        return status

    def export_strategy_to_fincept(
        self,
        strategy_config: Dict[str, Any],
        output_file: Optional[str] = None,
    ) -> str:
        """
        导出策略到FinceptTerminal格式

        参数:
            strategy_config: 策略配置
            output_file: 输出文件路径（可选）
        """

        # 导出为JSON格式
        exported_json = self.strategy_exporter.export_strategy(
            strategy_config,
            "fincept_json",
        )

        # 导出为Python脚本
        exported_python = self.strategy_exporter.export_strategy(
            strategy_config,
            "fincept_python",
        )

        # 保存到文件
        if output_file is None:
            output_dir = Path("fincept_exports")
            output_dir.mkdir(exist_ok=True)

            strategy_name = strategy_config.get("name", "strategy").replace(" ", "_")
            output_file = str(output_dir / f"{strategy_name}")

        # 保存JSON
        json_file = f"{output_file}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            f.write(exported_json)

        # 保存Python脚本
        py_file = f"{output_file}.py"
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(exported_python)

        return f"策略已导出到:\n  - {json_file}\n  - {py_file}"

    def sync_strategies(
        self,
        strategies: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        同步多个策略到FinceptTerminal

        参数:
            strategies: 策略列表

        返回:
            导出结果字典
        """

        results = {}

        for i, strategy in enumerate(strategies):
            try:
                result = self.export_strategy_to_fincept(strategy)
                results[f"strategy_{i}"] = result
            except Exception as e:
                results[f"strategy_{i}"] = f"导出失败: {e}"

        return results


def create_fincept_integration(
    fincept_path: Optional[str] = None,
    api_enabled: bool = False,
) -> FinceptIntegrator:
    """
    创建FinceptTerminal集成实例

    参数:
        fincept_path: FinceptTerminal安装路径
        api_enabled: 是否启用API推送
    """

    config = FinceptConfig(
        fincept_path=fincept_path,
        api_enabled=api_enabled,
    )

    return FinceptIntegrator(config)


# 快速开始示例
if __name__ == "__main__":
    # 1. 检查集成状态
    integrator = create_fincept_integration()
    status = integrator.check_integration_status()

    print("FinceptTerminal集成状态:")
    print(f"  已安装: {status['fincept_installed']}")
    print(f"  可用连接器: {', '.join(status['available_connectors'])}")
    print()

    # 2. 获取市场数据
    print("获取市场数据示例:")
    data = integrator.data_bridge.fetch_market_data(
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    print(f"  AAPL数据行数: {len(data)}")
    print(f"  日期范围: {data.index[0]} 到 {data.index[-1]}")
    print()

    # 3. 导出策略
    print("导出示例策略:")
    example_strategy = {
        "name": "Dual Moving Average",
        "description": "简单双均线策略",
        "dataset": "yahoo_finance",
        "symbol": "AAPL",
        "indicators": [
            {"name": "fast_ma", "type": "sma", "window": 12},
            {"name": "slow_ma", "type": "sma", "window": 26},
        ],
        "entry_rules": [
            {"left": "fast_ma", "op": "crosses_above", "right": "slow_ma"},
        ],
        "exit_rules": [
            {"left": "fast_ma", "op": "crosses_below", "right": "slow_ma"},
        ],
        "position_sizing": {
            "mode": "fixed_fraction",
            "risk_fraction": 0.1,
        },
        "risk_limits": {
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.12,
        },
    }

    export_result = integrator.export_strategy_to_fincept(example_strategy)
    print(export_result)
