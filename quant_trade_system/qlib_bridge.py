"""
Qlib Bridge - 微软Qlib量化投资平台桥接模块

Qlib是微软开源的AI驱动的量化投资平台，提供：
- 完整的量化投资框架
- 机器学习模型支持
- 高性能数据存储和查询
- 内置的训练和推理引擎

GitHub: https://github.com/microsoft/qlib
文档: https://qlib.readthedocs.io/
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd


CAUSAL_ALPHA_FEATURES: List[Dict[str, str]] = [
    {
        "name": "causal_alpha_return_1d",
        "formula": "$close / Ref($close, 1) - 1",
        "group": "price_volume",
        "financial_meaning": "1日价格动量，刻画短期资金对最新信息的确认。",
    },
    {
        "name": "causal_alpha_return_5d",
        "formula": "$close / Ref($close, 5) - 1",
        "group": "price_volume",
        "financial_meaning": "5日价格动量，刻画周度趋势和博弈叙事的延续性。",
    },
    {
        "name": "causal_alpha_volume_z20",
        "formula": "($volume - Mean($volume, 20)) / Std($volume, 20)",
        "group": "price_volume",
        "financial_meaning": "20日成交量异常度，刻画资金流和事件确认强度。",
    },
    {
        "name": "causal_alpha_volatility_20",
        "formula": "Std($close / Ref($close, 1) - 1, 20)",
        "group": "risk_state",
        "financial_meaning": "20日实现波动率，刻画风险状态与仓位惩罚。",
    },
    {
        "name": "causal_alpha_ma_gap20",
        "formula": "$close / Mean($close, 20) - 1",
        "group": "price_volume",
        "financial_meaning": "价格相对20日均线偏离，刻画趋势拥挤与均值回复压力。",
    },
    {
        "name": "causal_alpha_range_pct",
        "formula": "($high - $low) / $close",
        "group": "risk_state",
        "financial_meaning": "日内振幅占比，刻画多空分歧和冲击成本风险。",
    },
    {
        "name": "causal_alpha_momentum_quality",
        "formula": "($close / Ref($close, 5) - 1) / (Std($close / Ref($close, 1) - 1, 20) + 1e-9)",
        "group": "causal_quant",
        "financial_meaning": "风险调整后的周度动量，要求价格确认不能只来自高噪声波动。",
    },
    {
        "name": "causal_alpha_liquidity_pressure",
        "formula": "Abs($close / Ref($close, 1) - 1) / (Log($volume + 1) + 1e-9)",
        "group": "execution_risk",
        "financial_meaning": "单位流动性承载的价格冲击，刻画执行滑点和低流动性风险。",
    },
]


@dataclass(frozen=True)
class QlibWorkflowSpec:
    """Serializable Qlib-inspired workflow contract used by the bridge."""

    workflow_id: str
    universe: List[str]
    start_date: str
    end_date: str
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str
    features: List[str]
    feature_registry: List[Dict[str, Any]]
    label: str = "Ref($close, -1) / $close - 1"
    model_type: str = "lightgbm"
    strategy: Dict[str, Any] = field(default_factory=lambda: {"class": "TopkDropoutStrategy", "topk": 5, "n_drop": 1})
    cost: Dict[str, float] = field(default_factory=lambda: {"open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5.0})
    benchmark: Optional[str] = None
    provider_uri: str = "~/.qlib/qlib_data/cn_data"
    region: str = "cn"
    market: str = "cn_equity"
    leakage_policy: Dict[str, Any] = field(
        default_factory=lambda: {
            "feature_cutoff": "signal_date_close",
            "label_lag_days": 1,
            "label_excluded_from_inference": True,
            "purged_cv_required": True,
            "embargo_required": True,
        }
    )
    promotion_policy: Dict[str, Any] = field(
        default_factory=lambda: {
            "mode": "shadow_until_oos_and_causal_validation_pass",
            "requires_causal_validation": True,
            "requires_shadow_replay": True,
        }
    )


class QlibBridge:
    """Qlib桥接器 - 通过Python 3.11子进程运行Qlib"""

    def __init__(self, base_dir: str, python_path: Optional[str] = None):
        """
        初始化Qlib桥接器

        Args:
            base_dir: 项目基础目录
            python_path: Python 3.11解释器路径（自动检测）
        """
        self.base_dir = Path(base_dir)
        self.python = self._find_python(python_path)
        self.state_dir = self.base_dir / "state" / "qlib_home"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.recorder_dir = self.state_dir / "recorders"
        self.recorder_dir.mkdir(parents=True, exist_ok=True)
        self.available = self._check_availability()

    def _find_python(self, provided_path: Optional[str]) -> Optional[str]:
        """查找Python 3.11解释器"""
        if provided_path and Path(provided_path).exists():
            return provided_path

        env_python = os.environ.get("PROJECT_BRIDGE_PYTHON")
        if env_python and Path(env_python).exists():
            return env_python

        candidates = [
            "/opt/homebrew/bin/python3.11",
            "/usr/local/bin/python3.11",
            shutil.which("python3.11"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        return None

    def _build_cmd(self, code: str) -> List[str]:
        """安全地构建subprocess命令"""
        if not self.python:
            raise RuntimeError("Python interpreter not available")
        return [self.python, "-c", code]

    def _check_availability(self) -> bool:
        """检查Qlib是否可用"""
        if not self.python:
            return False

        try:
            cmd = self._build_cmd("import qlib; print(qlib.__version__)")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=20,
                env=self._bridge_env(),
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _bridge_env(self) -> Dict[str, str]:
        """创建桥接环境变量"""
        env = os.environ.copy()
        env["HOME"] = str(self.state_dir)
        env["QLIB_LOG_LEVEL"] = "WARNING"
        return env

    def get_version(self) -> Optional[str]:
        """获取Qlib版本"""
        if not self.available or not self.python:
            return None

        try:
            cmd = self._build_cmd("import qlib; print(qlib.__version__)")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=20,
                env=self._bridge_env(),
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def build_workflow_spec(
        self,
        universe: List[str],
        start_date: str,
        end_date: str,
        features: Optional[List[str]] = None,
        label: str = "Ref($close, -1) / $close - 1",
        model_type: str = "lightgbm",
        benchmark: Optional[str] = None,
        provider_uri: str = "~/.qlib/qlib_data/cn_data",
        region: str = "cn",
        market: str = "cn_equity",
        strategy: Optional[Dict[str, Any]] = None,
        cost: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Build a deterministic Qlib-style workflow contract.

        The returned spec is intentionally serializable and usable even when
        pyqlib is not installed. It is the audit boundary between data,
        features, labels, model candidates, strategy and recorder artifacts.
        """
        clean_universe = sorted({str(item) for item in universe if str(item).strip()})
        if not clean_universe:
            raise ValueError("universe must contain at least one instrument")
        feature_names = features or [item["name"] for item in CAUSAL_ALPHA_FEATURES]
        segments = self._build_time_segments(start_date, end_date)
        feature_registry = self._feature_registry_for(feature_names)
        payload = {
            "universe": clean_universe,
            "start_date": start_date,
            "end_date": end_date,
            "segments": segments,
            "features": feature_names,
            "label": label,
            "model_type": model_type,
            "benchmark": benchmark,
            "provider_uri": provider_uri,
            "region": region,
            "market": market,
            "strategy": strategy or {"class": "TopkDropoutStrategy", "topk": 5, "n_drop": 1},
            "cost": cost or {"open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5.0},
        }
        workflow_id = self._stable_id("qlib_workflow", payload)
        spec = QlibWorkflowSpec(
            workflow_id=workflow_id,
            universe=clean_universe,
            start_date=start_date,
            end_date=end_date,
            train_start=segments["train"][0],
            train_end=segments["train"][1],
            valid_start=segments["valid"][0],
            valid_end=segments["valid"][1],
            test_start=segments["test"][0],
            test_end=segments["test"][1],
            features=feature_names,
            feature_registry=feature_registry,
            label=label,
            model_type=model_type,
            strategy=payload["strategy"],
            cost=payload["cost"],
            benchmark=benchmark,
            provider_uri=provider_uri,
            region=region,
            market=market,
        )
        return asdict(spec)

    def build_dataset_config(self, workflow_spec: Mapping[str, Any]) -> Dict[str, Any]:
        """Create a Qlib DatasetH/DataHandlerLP-inspired config."""
        spec = self._coerce_spec(workflow_spec)
        return {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "DataHandlerLP",
                    "module_path": "qlib.data.dataset.handler",
                    "kwargs": {
                        "instruments": spec["universe"],
                        "start_time": spec["start_date"],
                        "end_time": spec["end_date"],
                        "fit_start_time": spec["train_start"],
                        "fit_end_time": spec["train_end"],
                        "features": [
                            {"name": item["name"], "formula": item["formula"], "group": item["group"]}
                            for item in spec["feature_registry"]
                        ],
                        "label": spec["label"],
                        "learn_processors": ["DropnaLabel"],
                        "infer_processors": ["Fillna", "ZScoreNorm"],
                        "leakage_policy": spec["leakage_policy"],
                    },
                },
                "segments": {
                    "train": [spec["train_start"], spec["train_end"]],
                    "valid": [spec["valid_start"], spec["valid_end"]],
                    "test": [spec["test_start"], spec["test_end"]],
                },
            },
        }

    def build_workflow_config(self, workflow_spec: Mapping[str, Any]) -> Dict[str, Any]:
        """Create a stable qrun-like experiment config."""
        spec = self._coerce_spec(workflow_spec)
        return {
            "workflow_id": spec["workflow_id"],
            "qlib_init": {
                "provider_uri": spec["provider_uri"],
                "region": spec["region"],
            },
            "task": {
                "model": {
                    "class": self._model_class_name(spec["model_type"]),
                    "module_path": self._model_module_path(spec["model_type"]),
                    "kwargs": self._model_kwargs(spec["model_type"]),
                },
                "dataset": self.build_dataset_config(spec),
                "record": [
                    {"class": "SignalRecord", "module_path": "qlib.workflow.record_temp"},
                    {"class": "PortAnaRecord", "module_path": "qlib.workflow.record_temp"},
                ],
            },
            "portfolio_analysis": {
                "strategy": spec["strategy"],
                "backtest": {
                    "start_time": spec["test_start"],
                    "end_time": spec["test_end"],
                    "benchmark": spec.get("benchmark"),
                    "account": 1_000_000,
                    "exchange_kwargs": spec["cost"],
                },
            },
            "promotion_policy": spec["promotion_policy"],
            "audit": {
                "generated_at": "runtime",
                "bridge_available": "runtime",
                "contract": "Qlib handles ML workflow; causal engines gate financial meaning, validation and production promotion.",
            },
        }

    def run_workflow_config(
        self,
        workflow_spec: Mapping[str, Any],
        market_data: Optional[Mapping[str, pd.DataFrame]] = None,
        execute_qlib: bool = False,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Run the Qlib-inspired workflow.

        When pyqlib is unavailable, this still generates spec/config,
        recorder artifacts and a deterministic shadow signal/backtest if
        market_data is supplied.
        """
        spec = self._coerce_spec(workflow_spec)
        config = self.build_workflow_config(spec)
        shadow = self._run_shadow_workflow(spec, market_data or {})
        recorder_path = self.recorder_dir / spec["workflow_id"]
        artifacts: Dict[str, Any] = {"recorder_dir": str(recorder_path)}
        if persist:
            recorder_path.mkdir(parents=True, exist_ok=True)
            artifacts.update(
                {
                    "workflow_spec": self._write_json(recorder_path / "workflow_spec.json", spec),
                    "workflow_config": self._write_json(recorder_path / "workflow_config.json", config),
                    "predictions": self._write_json(recorder_path / "predictions.json", shadow["predictions"]),
                    "metrics": self._write_json(recorder_path / "metrics.json", shadow["metrics"]),
                    "backtest": self._write_json(recorder_path / "backtest.json", shadow["backtest"]),
                    "leakage_report": self._write_json(recorder_path / "leakage_report.json", shadow["leakage_report"]),
                }
            )
        qlib_execution = {
            "attempted": False,
            "reason": "pyqlib execution is opt-in; shadow workflow is used for deterministic CI and nightly audit.",
        }
        if execute_qlib:
            qlib_execution = self._execute_qlib_config(config, recorder_path)
        return {
            "success": True,
            "framework": "qlib",
            "available": self.available,
            "mode": "qlib_shadow_adapter" if self.available else "shadow_config_only",
            "workflow_spec": spec,
            "workflow_config": config,
            "recorder_artifacts": artifacts,
            "predictions": shadow["predictions"],
            "metrics": shadow["metrics"],
            "backtest": shadow["backtest"],
            "leakage_report": shadow["leakage_report"],
            "qlib_execution": qlib_execution,
            "promotion_status": self._promotion_status(shadow),
        }

    def export_predictions(self, workflow_result: Mapping[str, Any], output_path: Optional[str | Path] = None) -> Dict[str, Any]:
        """Export workflow predictions to JSON for downstream signal review."""
        predictions = list(workflow_result.get("predictions", []))
        path = Path(output_path) if output_path else self.recorder_dir / str(workflow_result["workflow_spec"]["workflow_id"]) / "predictions_export.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return {"path": str(path), "count": len(predictions)}

    def get_recorder_artifacts(self, workflow_id: str) -> Dict[str, Any]:
        """List persisted recorder files for a workflow."""
        path = self.recorder_dir / workflow_id
        if not path.exists():
            return {"workflow_id": workflow_id, "exists": False, "files": []}
        files = sorted(str(item) for item in path.iterdir() if item.is_file())
        return {"workflow_id": workflow_id, "exists": True, "recorder_dir": str(path), "files": files}

    def initialize(self, provider_uri: str = "~/.qlib/qlib_data/cn_data", region: str = "cn") -> Dict[str, Any]:
        """
        初始化Qlib

        Args:
            provider_uri: 数据提供者URI
            region: 区域 (cn/us)

        Returns:
            初始化结果
        """
        if not self.available:
            return {
                "success": False,
                "error": "Qlib not available",
                "installed": False,
            }

        try:
            code = f"""
import qlib
from pathlib import Path

# 初始化Qlib
qlib.init(provider_uri="{provider_uri}", region="{region}")

# 获取数据信息
from qlib.data import D
instruments = D.instruments(market="all")
print(f"Available instruments: {{len(instruments)}}")

# 获取日期范围
from qlib.data import D
cal = D.calendar()
print(f"Date range: {{cal[0]}} to {{cal[-1]}}")
"""

            cmd = self._build_cmd(code)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env=self._bridge_env(),
                check=False,
            )

            output = result.stdout + result.stderr
            return {
                "success": result.returncode == 0,
                "output": output,
                "installed": True,
                "version": self.get_version(),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "installed": True,
            }

    def get_data(
        self,
        instrument: str,
        start_date: str,
        end_date: str,
        fields: List[str] = ["open", "high", "low", "close", "volume"],
    ) -> Optional[Dict[str, Any]]:
        """
        获取Qlib数据

        Args:
            instrument: 交易标的（如：000001.SZ）
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段列表

        Returns:
            数据字典或None
        """
        if not self.available:
            return None

        try:
            fields_str = json.dumps(fields)
            code = f"""
import qlib
import pandas as pd
from qlib.data import D

# 初始化
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

# 获取数据
df = D.features(
    ["{instrument}"],
    {fields_str},
    start_time="{start_date}",
    end_time="{end_date}"
)

# 转换为字典
result = {{
    "index": df.index.tolist(),
    "columns": df.columns.tolist(),
    "data": df.values.tolist(),
    "shape": df.shape
}}
print(json.dumps(result, default=str))
"""

            cmd = self._build_cmd(code)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env=self._bridge_env(),
                check=False,
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception:
            pass
        return None

    def create_model(
        self,
        model_type: str = "mlp",
        features: List[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        创建Qlib模型

        Args:
            model_type: 模型类型 (mlp/lstm/gru/lightgbm)
            features: 特征列表

        Returns:
            模型信息或None
        """
        if not self.available:
            return None

        features = features or ["$open", "$high", "$low", "$close", "$volume"]
        features_str = json.dumps(features)

        try:
            code = f"""
import qlib
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.dataset import DatasetH
from qlib.constant import REG_CN

# 初始化
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

# 创建模型
model = LGBModel()

# 输出模型信息
print(f"Model type: {model_type}")
print(f"Features: {features_str}")
print("Model created successfully")
"""

            cmd = self._build_cmd(code)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env=self._bridge_env(),
                check=False,
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "model_type": model_type,
                    "features": features,
                    "output": result.stdout,
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
        return None

    def backtest(
        self,
        strategy_config: Dict[str, Any],
        start_date: str,
        end_date: str,
    ) -> Optional[Dict[str, Any]]:
        """
        使用Qlib进行回测

        Args:
            strategy_config: 策略配置
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            回测结果或None
        """
        try:
            spec = strategy_config.get("workflow_spec")
            if not spec:
                universe = strategy_config.get("universe") or [strategy_config.get("instrument", "000001.SZ")]
                spec = self.build_workflow_spec(
                    universe=list(universe),
                    start_date=start_date,
                    end_date=end_date,
                    features=strategy_config.get("features"),
                    label=strategy_config.get("label", "Ref($close, -1) / $close - 1"),
                    model_type=strategy_config.get("model_type", "lightgbm"),
                    benchmark=strategy_config.get("benchmark"),
                    provider_uri=strategy_config.get("provider_uri", "~/.qlib/qlib_data/cn_data"),
                    region=strategy_config.get("region", "cn"),
                    market=strategy_config.get("market", "cn_equity"),
                    strategy=strategy_config.get("strategy"),
                    cost=strategy_config.get("cost"),
                )
            return self.run_workflow_config(
                spec,
                market_data=strategy_config.get("market_data"),
                execute_qlib=bool(strategy_config.get("execute_qlib", False)),
            )
        except Exception as e:
            return {"success": False, "error": str(e), "framework": "qlib"}

    def _run_shadow_workflow(self, spec: Mapping[str, Any], market_data: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
        prediction_rows: List[Dict[str, Any]] = []
        feature_cols = [item["name"] for item in spec["feature_registry"]]
        matrices: Dict[str, pd.DataFrame] = {}
        for symbol in spec["universe"]:
            frame = self._normalize_shadow_frame(market_data.get(symbol), spec["start_date"], spec["end_date"])
            if frame.empty:
                continue
            matrix = self._build_shadow_factor_matrix(frame)
            matrix["label"] = frame["close"].shift(-1) / frame["close"] - 1.0
            matrix["symbol"] = symbol
            matrix["date"] = frame["date"].astype(str).values
            matrices[symbol] = matrix

        if not matrices:
            leakage_report = self._build_leakage_report(spec, [], passed=True)
            return {
                "predictions": [],
                "metrics": {"status": "no_market_data", "prediction_count": 0},
                "backtest": {"status": "no_market_data", "net_return": 0.0, "max_drawdown": 0.0},
                "leakage_report": leakage_report,
            }

        combined = pd.concat(matrices.values(), ignore_index=True)
        for col in feature_cols:
            if col not in combined.columns:
                combined[col] = 0.0
        train_mask = self._date_mask(combined["date"], spec["train_start"], spec["train_end"]) & combined["label"].notna()
        train = combined.loc[train_mask, feature_cols + ["label"]].replace([float("inf"), float("-inf")], pd.NA).dropna()
        means = train[feature_cols].mean() if not train.empty else combined[feature_cols].mean()
        stds = train[feature_cols].std().replace(0, 1.0) if not train.empty else combined[feature_cols].std().replace(0, 1.0)
        stds = stds.fillna(1.0)
        weights = self._fit_shadow_weights(train, feature_cols)
        z_features = (combined[feature_cols].fillna(0.0) - means.fillna(0.0)) / stds
        combined["score"] = z_features.fillna(0.0).mul(pd.Series(weights)).sum(axis=1)
        combined["signal"] = combined["score"].apply(lambda value: "LONG" if value > 0 else ("SHORT" if value < 0 else "HOLD"))
        for row in combined.sort_values(["date", "symbol"]).itertuples(index=False):
            label_value = getattr(row, "label")
            prediction_rows.append(
                {
                    "date": str(getattr(row, "date")),
                    "symbol": str(getattr(row, "symbol")),
                    "score": round(float(getattr(row, "score")), 8),
                    "signal": str(getattr(row, "signal")),
                    "label": None if pd.isna(label_value) else round(float(label_value), 8),
                }
            )
        metrics = self._shadow_metrics(combined, spec)
        backtest = self._shadow_backtest(combined, spec)
        leakage_report = self._build_leakage_report(spec, list(matrices.values()), passed=True)
        return {
            "predictions": prediction_rows,
            "metrics": metrics,
            "backtest": backtest,
            "leakage_report": leakage_report,
        }

    @staticmethod
    def _normalize_shadow_frame(frame: Optional[pd.DataFrame], start_date: str, end_date: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        normalized = frame.copy()
        normalized.columns = [str(col).lower() for col in normalized.columns]
        if "date" not in normalized.columns:
            if "timestamp" in normalized.columns:
                normalized["date"] = normalized["timestamp"]
            else:
                normalized["date"] = normalized.index
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        normalized = normalized.dropna(subset=["date"]).sort_values("date")
        for col in ["close", "open", "high", "low", "volume"]:
            if col not in normalized.columns:
                normalized[col] = normalized["close"] if "close" in normalized.columns else 0.0
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
        normalized["open"] = normalized["open"].fillna(normalized["close"])
        normalized["high"] = normalized["high"].fillna(normalized["close"])
        normalized["low"] = normalized["low"].fillna(normalized["close"])
        normalized["volume"] = normalized["volume"].fillna(0.0)
        normalized = normalized.loc[(normalized["date"] >= start_date) & (normalized["date"] <= end_date)]
        return normalized[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"]).reset_index(drop=True)

    @staticmethod
    def _build_shadow_factor_matrix(frame: pd.DataFrame) -> pd.DataFrame:
        returns_1d = frame["close"].pct_change()
        returns_5d = frame["close"].pct_change(5)
        volume_mean = frame["volume"].rolling(20, min_periods=3).mean()
        volume_std = frame["volume"].rolling(20, min_periods=3).std().replace(0, 1.0)
        volatility = returns_1d.rolling(20, min_periods=5).std()
        ma20 = frame["close"].rolling(20, min_periods=5).mean()
        matrix = pd.DataFrame(index=frame.index)
        matrix["causal_alpha_return_1d"] = returns_1d
        matrix["causal_alpha_return_5d"] = returns_5d
        matrix["causal_alpha_volume_z20"] = (frame["volume"] - volume_mean) / volume_std
        matrix["causal_alpha_volatility_20"] = volatility
        matrix["causal_alpha_ma_gap20"] = frame["close"] / ma20 - 1.0
        matrix["causal_alpha_range_pct"] = (frame["high"] - frame["low"]) / frame["close"].replace(0, pd.NA)
        matrix["causal_alpha_momentum_quality"] = returns_5d / (volatility + 1e-9)
        log_volume = frame["volume"].clip(lower=0.0).add(1.0).apply(math.log)
        matrix["causal_alpha_liquidity_pressure"] = returns_1d.abs() / (log_volume + 1e-9)
        return matrix.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)

    @staticmethod
    def _date_mask(values: pd.Series, start: str, end: str) -> pd.Series:
        return (values.astype(str) >= start) & (values.astype(str) <= end)

    @staticmethod
    def _fit_shadow_weights(train: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
        if train.empty or train["label"].std() == 0:
            equal = 1.0 / max(len(feature_cols), 1)
            return {col: equal for col in feature_cols}
        raw: Dict[str, float] = {}
        for col in feature_cols:
            if train[col].std() == 0:
                raw[col] = 0.0
                continue
            corr = train[col].corr(train["label"])
            raw[col] = 0.0 if pd.isna(corr) else float(corr)
        norm = sum(abs(value) for value in raw.values())
        if norm <= 1e-12:
            equal = 1.0 / max(len(feature_cols), 1)
            return {col: equal for col in feature_cols}
        return {col: value / norm for col, value in raw.items()}

    def _shadow_metrics(self, combined: pd.DataFrame, spec: Mapping[str, Any]) -> Dict[str, Any]:
        eval_mask = self._date_mask(combined["date"], spec["valid_start"], spec["test_end"]) & combined["label"].notna()
        evaluated = combined.loc[eval_mask].copy()
        if evaluated.empty:
            return {"status": "no_oos_labels", "prediction_count": int(len(combined))}
        signed = evaluated["score"] * evaluated["label"]
        gains = signed[signed > 0]
        losses = signed[signed < 0].abs()
        benchmark_move = evaluated["label"].abs().mean()
        ic = evaluated["score"].corr(evaluated["label"])
        if pd.isna(ic):
            ic = 0.0
        return {
            "status": "evaluated",
            "prediction_count": int(len(combined)),
            "oos_count": int(len(evaluated)),
            "ic": round(float(ic), 6),
            "win_rate": round(float((signed > 0).mean()), 6),
            "payoff_ratio": round(float(gains.mean() / max(losses.mean(), 1e-12)) if not gains.empty else 0.0, 6),
            "elasticity": round(float(signed.mean() / max(benchmark_move, 1e-12)), 6),
        }

    def _shadow_backtest(self, combined: pd.DataFrame, spec: Mapping[str, Any]) -> Dict[str, Any]:
        test = combined.loc[self._date_mask(combined["date"], spec["test_start"], spec["test_end"]) & combined["label"].notna()].copy()
        if test.empty:
            return {"status": "no_test_labels", "net_return": 0.0, "max_drawdown": 0.0, "turnover": 0.0}
        topk = int(spec.get("strategy", {}).get("topk", 5))
        close_cost = float(spec.get("cost", {}).get("close_cost", 0.0015))
        nav = 1.0
        peak = 1.0
        max_drawdown = 0.0
        total_turnover = 0.0
        previous_symbols: set[str] = set()
        daily_rows: List[Dict[str, Any]] = []
        for current_date, group in test.groupby("date", sort=True):
            longs = group.sort_values("score", ascending=False).head(topk)
            longs = longs.loc[longs["score"] > 0]
            current_symbols = set(longs["symbol"].astype(str))
            turnover = len(previous_symbols.symmetric_difference(current_symbols)) / max(topk, 1)
            gross_return = float(longs["label"].mean()) if not longs.empty else 0.0
            cost = turnover * close_cost
            net_return = gross_return - cost
            nav *= 1.0 + net_return
            peak = max(peak, nav)
            max_drawdown = min(max_drawdown, nav / peak - 1.0)
            total_turnover += turnover
            previous_symbols = current_symbols
            daily_rows.append(
                {
                    "date": str(current_date),
                    "long_symbols": sorted(current_symbols),
                    "gross_return": round(gross_return, 8),
                    "cost": round(cost, 8),
                    "net_return": round(net_return, 8),
                    "nav": round(nav, 8),
                }
            )
        return {
            "status": "evaluated",
            "topk": topk,
            "net_return": round(nav - 1.0, 8),
            "max_drawdown": round(max_drawdown, 8),
            "turnover": round(total_turnover / max(len(daily_rows), 1), 8),
            "daily": daily_rows,
        }

    @staticmethod
    def _build_leakage_report(spec: Mapping[str, Any], matrices: List[pd.DataFrame], passed: bool) -> Dict[str, Any]:
        feature_formulas = {item["name"]: item["formula"] for item in spec["feature_registry"]}
        future_feature_formulas = [
            name for name, formula in feature_formulas.items() if "Ref($close, -1)" in formula or "shift(-" in formula.lower()
        ]
        latest_feature_date = None
        if matrices:
            dates = []
            for matrix in matrices:
                if "date" in matrix.columns and not matrix.empty:
                    dates.append(str(matrix["date"].max()))
            latest_feature_date = max(dates) if dates else None
        return {
            "passed": bool(passed and not future_feature_formulas),
            "feature_cutoff": spec["leakage_policy"]["feature_cutoff"],
            "label_lag_days": spec["leakage_policy"]["label_lag_days"],
            "label_excluded_from_inference": spec["leakage_policy"]["label_excluded_from_inference"],
            "latest_feature_date": latest_feature_date,
            "workflow_end_date": spec["end_date"],
            "future_feature_formulas": future_feature_formulas,
        }

    def _execute_qlib_config(self, config: Mapping[str, Any], recorder_path: Path) -> Dict[str, Any]:
        if not self.available:
            return {"attempted": False, "reason": "pyqlib is not available"}
        config_path = recorder_path / "workflow_config.json"
        try:
            code = f"""
import json
from pathlib import Path
import qlib

config = json.loads(Path({str(config_path)!r}).read_text(encoding='utf-8'))
qlib.init(**config['qlib_init'])
print(json.dumps({{'loaded_workflow_id': config['workflow_id'], 'qlib_initialized': True}}))
"""
            result = subprocess.run(
                self._build_cmd(code),
                capture_output=True,
                text=True,
                timeout=120,
                env=self._bridge_env(),
                check=False,
            )
            return {
                "attempted": True,
                "success": result.returncode == 0,
                "output": result.stdout + result.stderr,
            }
        except Exception as exc:
            return {"attempted": True, "success": False, "error": str(exc)}

    def _promotion_status(self, shadow: Mapping[str, Any]) -> Dict[str, Any]:
        leakage_ok = bool(shadow.get("leakage_report", {}).get("passed"))
        metrics = shadow.get("metrics", {})
        if leakage_ok and metrics.get("status") == "evaluated" and float(metrics.get("ic", 0.0)) > 0:
            return {
                "status": "shadow_candidate",
                "reason": "Qlib-style workflow produced positive OOS IC but still requires causal validation and shadow replay.",
            }
        return {
            "status": "shadow_only",
            "reason": "Workflow is recorded for research; it may not raise production position size without causal validation.",
        }

    @staticmethod
    def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()[:12]
        return f"{prefix}_{digest}"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _write_json(path: Path, payload: Any) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return str(path)

    @staticmethod
    def _build_time_segments(start_date: str, end_date: str) -> Dict[str, List[str]]:
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        if len(dates) < 3:
            return {"train": [start_date, end_date], "valid": [end_date, end_date], "test": [end_date, end_date]}
        train_end_idx = max(0, int(len(dates) * 0.6) - 1)
        valid_end_idx = max(train_end_idx + 1, int(len(dates) * 0.8) - 1)
        valid_end_idx = min(valid_end_idx, len(dates) - 2)
        return {
            "train": [dates[0].strftime("%Y-%m-%d"), dates[train_end_idx].strftime("%Y-%m-%d")],
            "valid": [dates[train_end_idx + 1].strftime("%Y-%m-%d"), dates[valid_end_idx].strftime("%Y-%m-%d")],
            "test": [dates[valid_end_idx + 1].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")],
        }

    @staticmethod
    def _feature_registry_for(features: List[str]) -> List[Dict[str, Any]]:
        known = {item["name"]: item for item in CAUSAL_ALPHA_FEATURES}
        registry = []
        for name in features:
            item = known.get(name)
            if item:
                registry.append({**item, "available_timestamp": "signal_date_close", "source": "causal_alpha_template"})
            else:
                registry.append(
                    {
                        "name": name,
                        "formula": name,
                        "group": "custom",
                        "financial_meaning": "用户提供的自定义Qlib字段，需由上游FeatureStore补充金融含义。",
                        "available_timestamp": "signal_date_close",
                        "source": "user_supplied",
                    }
                )
        return registry

    @staticmethod
    def _coerce_spec(workflow_spec: Mapping[str, Any]) -> Dict[str, Any]:
        spec = dict(workflow_spec)
        if "workflow_spec" in spec:
            spec = dict(spec["workflow_spec"])
        required = ["workflow_id", "universe", "start_date", "end_date", "features", "feature_registry"]
        missing = [key for key in required if key not in spec]
        if missing:
            raise ValueError(f"workflow_spec missing required keys: {', '.join(missing)}")
        return spec

    @staticmethod
    def _model_class_name(model_type: str) -> str:
        mapping = {
            "lightgbm": "LGBModel",
            "lgbm": "LGBModel",
            "mlp": "MLPModel",
            "lstm": "LSTMModel",
            "gru": "GRUModel",
            "transformer": "TransformerModel",
        }
        return mapping.get(str(model_type).lower(), "LGBModel")

    @staticmethod
    def _model_module_path(model_type: str) -> str:
        mapping = {
            "lightgbm": "qlib.contrib.model.gbdt",
            "lgbm": "qlib.contrib.model.gbdt",
            "mlp": "qlib.contrib.model.pytorch_nn",
            "lstm": "qlib.contrib.model.pytorch_lstm",
            "gru": "qlib.contrib.model.pytorch_gru",
            "transformer": "qlib.contrib.model.pytorch_transformer",
        }
        return mapping.get(str(model_type).lower(), "qlib.contrib.model.gbdt")

    @staticmethod
    def _model_kwargs(model_type: str) -> Dict[str, Any]:
        if str(model_type).lower() in {"lightgbm", "lgbm"}:
            return {"loss": "mse", "num_leaves": 31, "learning_rate": 0.05, "n_estimators": 200}
        return {"loss": "mse", "early_stop": 20, "max_steps": 2000}

    def get_status(self) -> Dict[str, Any]:
        """获取Qlib状态"""
        return {
            "installed": self.available,
            "version": self.get_version() if self.available else None,
            "python_path": self.python,
            "state_dir": str(self.state_dir),
            "recorder_dir": str(self.recorder_dir),
            "capabilities": {
                "data_provider": self.available,
                "model_training": self.available,
                "backtesting": self.available,
                "portfolio_management": self.available,
                "workflow_config": True,
                "recorder_artifacts": True,
                "shadow_backtest": True,
            },
        }


# 便捷函数
def create_qlib_bridge(base_dir: str) -> QlibBridge:
    """创建Qlib桥接器实例"""
    return QlibBridge(base_dir)


if __name__ == "__main__":
    import shutil

    # 测试桥接器
    bridge = create_qlib_bridge("/Users/caijiawen/Documents/New project/quant-trading-system")
    status = bridge.get_status()

    print("Qlib Bridge Status:")
    print(json.dumps(status, indent=2, ensure_ascii=False))

    if status["installed"]:
        print("\n✓ Qlib is available")
    else:
        print("\n✗ Qlib is not installed")
        print("  Install: pip install pyqlib")
