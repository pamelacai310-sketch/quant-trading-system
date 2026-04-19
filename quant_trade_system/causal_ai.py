from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .ecosystem import EcosystemIntegrationManager


class DataSource(Enum):
    AUTO = "auto"
    LOCAL_CSV = "local_csv"
    SYNTHETIC = "synthetic"
    EXTERNAL = "external"


class GitHubProjectStatus(Enum):
    NOT_INSTALLED = "not_installed"
    INSTALLED = "installed"
    INTEGRATED = "integrated"
    TESTED = "tested"
    FAILED = "failed"


@dataclass
class GitHubProject:
    name: str
    url: str
    stars: int
    description: str
    category: str
    import_name: str
    tier: str = "core"
    integration_mode: str = "native"
    status: GitHubProjectStatus = GitHubProjectStatus.NOT_INSTALLED
    enabled: bool = True


GITHUB_PROJECTS: Dict[str, GitHubProject] = {
    "finshare": GitHubProject(
        name="finshare",
        url="https://github.com/finvfamily/finshare",
        stars=355,
        description="专业的金融数据获取工具库",
        category="数据源",
        import_name="finshare",
        tier="tier0",
        integration_mode="bridge",
    ),
    "novaaware": GitHubProject(
        name="novaaware",
        url="https://github.com/gaoxianglong/novaaware",
        stars=17,
        description="因果AI数字意识引擎",
        category="因果推理",
        import_name="novaaware",
        tier="tier0",
    ),
    "Causal-AI-Agent": GitHubProject(
        name="Causal-AI-Agent",
        url="https://github.com/jdubbert/Causal-AI-Agent",
        stars=16,
        description="因果AI智能体",
        category="智能体",
        import_name="causal_ai_agent",
        tier="tier0",
    ),
    "openbb": GitHubProject(
        name="OpenBB",
        url="https://github.com/OpenBB-finance/OpenBB",
        stars=0,
        description="金融数据分析平台",
        category="数据源",
        import_name="openbb",
        tier="tier1",
        integration_mode="bridge",
    ),
    "ccxt": GitHubProject(
        name="CCXT",
        url="https://github.com/ccxt/ccxt",
        stars=0,
        description="统一交易所 API 接口",
        category="交易接口",
        import_name="ccxt",
        tier="tier1",
    ),
    "lightweight-charts": GitHubProject(
        name="Lightweight Charts",
        url="https://github.com/tradingview/lightweight-charts",
        stars=0,
        description="轻量级金融图表库",
        category="前端图表",
        import_name="",
        tier="tier1",
        integration_mode="frontend",
    ),
    "ta-stack": GitHubProject(
        name="TA Stack",
        url="https://github.com/TA-Lib/ta-lib",
        stars=0,
        description="技术指标生态（TA-Lib / pandas-ta / ta）",
        category="技术指标",
        import_name="ta",
        tier="tier1",
    ),
    "backtrader": GitHubProject(
        name="Backtrader",
        url="https://github.com/backtrader/backtrader",
        stars=0,
        description="Python 事件驱动回测引擎",
        category="回测",
        import_name="backtrader",
        tier="tier2",
    ),
    "QuantConnect-Lean": GitHubProject(
        name="Lean",
        url="https://github.com/QuantConnect/Lean",
        stars=0,
        description="机构级量化研究与实盘框架",
        category="导出适配",
        import_name="",
        tier="tier2",
        integration_mode="export",
    ),
    "freqtrade": GitHubProject(
        name="Freqtrade",
        url="https://github.com/freqtrade/freqtrade",
        stars=0,
        description="开源加密量化机器人",
        category="导出适配",
        import_name="",
        tier="tier2",
        integration_mode="export",
    ),
    "hummingbot": GitHubProject(
        name="Hummingbot",
        url="https://github.com/hummingbot/hummingbot",
        stars=0,
        description="做市与套利机器人",
        category="导出适配",
        import_name="",
        tier="tier2",
        integration_mode="export",
    ),
    "QuantLib": GitHubProject(
        name="QuantLib",
        url="https://github.com/quantlib/quantlib",
        stars=0,
        description="衍生品定价与风险建模",
        category="定价风控",
        import_name="QuantLib",
        tier="tier2",
        integration_mode="bridge",
    ),
    "FinEval": GitHubProject(
        name="FinEval",
        url="https://github.com/SUFE-AIFLM-Lab/FinEval",
        stars=0,
        description="金融 LLM 评测基准",
        category="评测",
        import_name="",
        tier="tier3",
        integration_mode="template",
    ),
    "FinLongEval": GitHubProject(
        name="FinLongEval",
        url="https://github.com/valuesimplex/FinLongEval",
        stars=0,
        description="长上下文金融评测",
        category="评测",
        import_name="",
        tier="tier3",
        integration_mode="template",
    ),
    "XuanYuan": GitHubProject(
        name="XuanYuan",
        url="https://github.com/Duxiaoman-DI/XuanYuan",
        stars=0,
        description="中文金融对话大模型",
        category="模型连接器",
        import_name="",
        tier="tier3",
        integration_mode="connector",
    ),
    "TradingAgents-AShare": GitHubProject(
        name="TradingAgents-AShare",
        url="https://github.com/TradingAgents/TradingAgents-AShare",
        stars=0,
        description="多智能体金融研究范式",
        category="多智能体",
        import_name="",
        tier="tier3",
        integration_mode="template",
    ),
}


class GitHubProjectManager:
    def __init__(self) -> None:
        self.projects = {
            key: GitHubProject(**project.__dict__)
            for key, project in GITHUB_PROJECTS.items()
        }
        self._check_installed_status()

    def _check_installed_status(self) -> None:
        for project in self.projects.values():
            if not project.import_name:
                project.status = GitHubProjectStatus.NOT_INSTALLED
                continue
            if project.import_name == "finshare" and sys.version_info < (3, 10):
                project.status = GitHubProjectStatus.NOT_INSTALLED
                continue
            try:
                importlib.import_module(project.import_name)
                project.status = GitHubProjectStatus.INSTALLED
            except ImportError:
                project.status = GitHubProjectStatus.NOT_INSTALLED
            except Exception:
                project.status = GitHubProjectStatus.NOT_INSTALLED

    def get_project_status(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "name": project.name,
                "stars": project.stars,
                "description": project.description,
                "category": project.category,
                "tier": project.tier,
                "integration_mode": project.integration_mode,
                "status": project.status.value,
                "enabled": project.enabled,
                "url": project.url,
            }
            for name, project in self.projects.items()
        }

    def mark_status(self, project_name: str, status: GitHubProjectStatus) -> None:
        project = self.projects.get(project_name)
        if project is None:
            return
        if status.value == project.status.value:
            return
        order = {
            GitHubProjectStatus.NOT_INSTALLED: 0,
            GitHubProjectStatus.INSTALLED: 1,
            GitHubProjectStatus.INTEGRATED: 2,
            GitHubProjectStatus.TESTED: 3,
            GitHubProjectStatus.FAILED: -1,
        }
        if order[status] >= order.get(project.status, 0) or status == GitHubProjectStatus.FAILED:
            project.status = status

    def mark_status_name(self, project_name: str, status_name: str) -> None:
        try:
            status = GitHubProjectStatus(status_name)
        except ValueError:
            return
        self.mark_status(project_name, status)


class EnhancedDataAdapter:
    def __init__(
        self,
        data_dir: str,
        default_source: DataSource = DataSource.AUTO,
        github_manager: Optional[GitHubProjectManager] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.default_source = default_source
        self.github_manager = github_manager
        self.finshare = None
        self.finshare_bridge_python = None
        self.finshare_bridge_script = Path(__file__).with_name("finshare_bridge.py")
        self.finshare_bridge_home = self.data_dir.parent / "state" / "finshare_home"
        self.symbol_map = {
            "XAUUSD": "gold_daily",
            "COMEX_Gold": "gold_daily",
            "QQQ": "nasdaq_daily",
            "NDX": "nasdaq_daily",
            "LME_Copper": "copper_daily",
            "HG": "copper_daily",
        }
        self.proxy_map = {
            "AAPL": ("nasdaq_daily", 0.92, 1.05),
            "MSFT": ("nasdaq_daily", 0.98, 1.15),
            "GOOGL": ("nasdaq_daily", 0.88, 0.95),
            "NVDA": ("nasdaq_daily", 1.32, 1.35),
            "600000": ("nasdaq_daily", 0.65, 0.22),
        }
        self.finshare_symbol_map = {
            "600000": ["600000.SH", "600000"],
            "000001": ["000001.SZ", "000001"],
            "XAUUSD": ["AU0", "GC=F"],
            "COMEX_Gold": ["AU0", "GC=F"],
            "LME_Copper": ["CU0", "HG=F"],
            "QQQ": ["QQQ"],
            "AAPL": ["AAPL"],
            "MSFT": ["MSFT"],
            "GOOGL": ["GOOGL"],
            "NVDA": ["NVDA"],
        }
        self._init_finshare()

    def _init_finshare(self) -> None:
        if sys.version_info >= (3, 10):
            try:
                self.finshare = importlib.import_module("finshare")
                if self.github_manager is not None:
                    self.github_manager.mark_status("finshare", GitHubProjectStatus.INTEGRATED)
            except ImportError:
                self.finshare = None
            except Exception:
                self.finshare = None
        self.finshare_bridge_python = self._find_finshare_bridge_python()
        if self.finshare_bridge_python and self.github_manager is not None:
            self.github_manager.mark_status("finshare", GitHubProjectStatus.INTEGRATED)

    def _find_finshare_bridge_python(self) -> Optional[str]:
        if not self.finshare_bridge_script.exists():
            return None
        candidates = []
        env_python = os.environ.get("FINSHARE_BRIDGE_PYTHON")
        if env_python:
            candidates.append(env_python)
        candidates.extend(
            [
                "/opt/homebrew/bin/python3.12",
                "/opt/homebrew/bin/python3.11",
                "/opt/homebrew/bin/python3.10",
                shutil.which("python3.12"),
                shutil.which("python3.11"),
                shutil.which("python3.10"),
            ]
        )
        for candidate in candidates:
            if not candidate:
                continue
            try:
                output = subprocess.run(
                    [candidate, str(self.finshare_bridge_script), "status"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    env=self._finshare_bridge_env(),
                    check=False,
                )
            except Exception:
                continue
            if output.returncode == 0:
                return candidate
        return None

    def _finshare_bridge_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        self.finshare_bridge_home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(self.finshare_bridge_home)
        return env

    def _run_finshare_bridge(self, command: str, payload: Dict[str, Any]) -> Optional[Any]:
        if not self.finshare_bridge_python:
            return None
        try:
            output = subprocess.run(
                [self.finshare_bridge_python, str(self.finshare_bridge_script), command],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=45,
                env=self._finshare_bridge_env(),
                check=False,
            )
        except Exception:
            return None
        if output.returncode != 0 or not output.stdout.strip():
            return None
        try:
            result = json.loads(output.stdout)
        except json.JSONDecodeError:
            return None
        if result.get("ok"):
            if self.github_manager is not None:
                self.github_manager.mark_status("finshare", GitHubProjectStatus.TESTED)
            return result.get("data")
        return None

    def _candidate_symbols(self, symbol: str) -> List[str]:
        return self.finshare_symbol_map.get(symbol, [symbol])

    def _period_to_dates(self, period: str) -> tuple[str, str]:
        end = pd.Timestamp.utcnow().normalize()
        mapping = {
            "1mo": 31,
            "3mo": 92,
            "6mo": 184,
            "1y": 366,
            "max": 3650,
        }
        days = mapping.get(period, 184)
        start = end - pd.Timedelta(days=days)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def _normalize_finshare_frame(self, frame: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        if frame is None or frame.empty:
            return None
        rename_map = {}
        for column in frame.columns:
            lower = column.lower()
            if lower in {"date", "datetime", "time", "timestamp"}:
                rename_map[column] = "Date"
            elif lower in {"open", "open_price"}:
                rename_map[column] = "Open"
            elif lower in {"high", "high_price"}:
                rename_map[column] = "High"
            elif lower in {"low", "low_price"}:
                rename_map[column] = "Low"
            elif lower in {"close", "close_price", "adj_close", "price"}:
                rename_map[column] = "Close"
            elif lower in {"volume", "vol"}:
                rename_map[column] = "Volume"
        normalized = frame.rename(columns=rename_map).copy()
        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        if not all(column in normalized.columns for column in required):
            return None
        normalized = normalized[required].copy()
        normalized["Date"] = pd.to_datetime(normalized["Date"]).dt.strftime("%Y-%m-%d")
        for column in ["Open", "High", "Low", "Close", "Volume"]:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized = normalized.dropna().reset_index(drop=True)
        if normalized.empty:
            return None
        normalized["Symbol"] = symbol
        normalized["Source"] = "finshare"
        return normalized

    def _fetch_finshare_history(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        bridge_data = self._run_finshare_bridge("history", {"symbol": symbol, "period": period})
        if bridge_data:
            frame = pd.DataFrame(bridge_data)
            normalized = self._normalize_finshare_frame(frame, symbol)
            if normalized is not None:
                return self._apply_period(normalized, period)
        if self.finshare is None:
            return None
        fn = getattr(self.finshare, "get_historical_data", None)
        if fn is None:
            return None
        start, end = self._period_to_dates(period)
        signature = inspect.signature(fn)
        accepted = set(signature.parameters.keys())
        for candidate in self._candidate_symbols(symbol):
            attempts = [
                {"symbol": candidate, "start": start, "end": end, "adjust": "qfq"},
                {"code": candidate, "start": start, "end": end, "adjust": "qfq"},
                {"ticker": candidate, "start": start, "end": end, "adjust": "qfq"},
                {"symbol": candidate, "period": period},
                {"code": candidate, "period": period},
            ]
            for payload in attempts:
                kwargs = {key: value for key, value in payload.items() if key in accepted}
                if not kwargs:
                    continue
                try:
                    frame = fn(**kwargs)
                except Exception:
                    continue
                normalized = self._normalize_finshare_frame(frame, symbol)
                if normalized is not None:
                    if self.github_manager is not None:
                        self.github_manager.mark_status("finshare", GitHubProjectStatus.TESTED)
                    return self._apply_period(normalized, period)
        return None

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        bridge_snapshot = self._run_finshare_bridge("snapshot", {"symbol": symbol})
        if isinstance(bridge_snapshot, dict) and bridge_snapshot:
            bridge_snapshot["symbol"] = symbol
            bridge_snapshot["source"] = "finshare"
            return bridge_snapshot
        if self.finshare is None:
            return {"symbol": symbol, "source": DataSource.SYNTHETIC.value}
        fn = getattr(self.finshare, "get_snapshot_data", None)
        if fn is None:
            return {"symbol": symbol, "source": DataSource.SYNTHETIC.value}
        for candidate in self._candidate_symbols(symbol):
            try:
                snapshot = fn(candidate)
            except Exception:
                continue
            if isinstance(snapshot, dict):
                snapshot = dict(snapshot)
                snapshot["symbol"] = symbol
                snapshot["source"] = "finshare"
                if self.github_manager is not None:
                    self.github_manager.mark_status("finshare", GitHubProjectStatus.TESTED)
                return snapshot
            if isinstance(snapshot, pd.DataFrame) and not snapshot.empty:
                row = snapshot.iloc[-1].to_dict()
                row["symbol"] = symbol
                row["source"] = "finshare"
                if self.github_manager is not None:
                    self.github_manager.mark_status("finshare", GitHubProjectStatus.TESTED)
                return row
        return {"symbol": symbol, "source": DataSource.SYNTHETIC.value}

    def _load_local_dataset(self, dataset_name: str) -> pd.DataFrame:
        frame = pd.read_csv(self.data_dir / f"{dataset_name}.csv")
        frame = frame.rename(columns={"timestamp": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        return frame[["Date", "Open", "High", "Low", "Close", "Volume"]]

    def _generate_proxy_dataset(self, symbol: str, base_dataset: str, beta: float, scale: float) -> pd.DataFrame:
        base = self._load_local_dataset(base_dataset).copy()
        seed = int(hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        close = base["Close"].to_numpy(dtype=float)
        returns = pd.Series(close).pct_change().fillna(0.0).to_numpy()
        transformed = np.cumsum(returns * beta + rng.normal(0.0001, 0.0045, len(returns)))
        proxy_close = np.maximum(scale * 100 * np.exp(transformed), 1.0)
        base["Close"] = np.round(proxy_close, 4)
        base["Open"] = np.round(base["Close"] * (1 + rng.normal(0, 0.0025, len(base))), 4)
        base["High"] = np.round(np.maximum(base["Open"], base["Close"]) * (1 + np.abs(rng.normal(0.003, 0.002, len(base)))), 4)
        base["Low"] = np.round(np.minimum(base["Open"], base["Close"]) * (1 - np.abs(rng.normal(0.003, 0.002, len(base)))), 4)
        base["Volume"] = np.maximum(base["Volume"].to_numpy() * abs(beta) * scale * 0.8, 50_000).astype(int)
        return base

    def get_symbol_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        finshare_frame = self._fetch_finshare_history(symbol, period)
        if finshare_frame is not None:
            return finshare_frame
        if symbol in self.symbol_map:
            frame = self._load_local_dataset(self.symbol_map[symbol]).copy()
            frame["Symbol"] = symbol
            frame["Source"] = DataSource.LOCAL_CSV.value
            return self._apply_period(frame, period)
        if symbol in self.proxy_map:
            dataset_name, beta, scale = self.proxy_map[symbol]
            frame = self._generate_proxy_dataset(symbol, dataset_name, beta, scale)
            frame["Symbol"] = symbol
            frame["Source"] = DataSource.SYNTHETIC.value
            return self._apply_period(frame, period)
        frame = self._generate_proxy_dataset(symbol, "nasdaq_daily", 0.75, 0.8)
        frame["Symbol"] = symbol
        frame["Source"] = DataSource.SYNTHETIC.value
        return self._apply_period(frame, period)

    def _apply_period(self, frame: pd.DataFrame, period: str) -> pd.DataFrame:
        mapping = {
            "1mo": 21,
            "3mo": 63,
            "6mo": 126,
            "1y": 252,
            "max": len(frame),
        }
        count = mapping.get(period, len(frame))
        return frame.tail(count).reset_index(drop=True)

    def get_batch_data(self, symbols: List[str], period: str = "1mo") -> Dict[str, pd.DataFrame]:
        return {symbol: self.get_symbol_data(symbol, period=period) for symbol in symbols}

    def get_batch_snapshots(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        bridge_batch = self._run_finshare_bridge("batch_snapshots", {"symbols": symbols})
        if isinstance(bridge_batch, dict) and bridge_batch:
            return bridge_batch
        batch_fn = getattr(self.finshare, "get_batch_snapshots", None) if self.finshare is not None else None
        if batch_fn is not None:
            for candidate_group in [symbols, [self._candidate_symbols(symbol)[0] for symbol in symbols]]:
                try:
                    snapshots = batch_fn(candidate_group)
                except Exception:
                    continue
                if isinstance(snapshots, dict) and snapshots:
                    if self.github_manager is not None:
                        self.github_manager.mark_status("finshare", GitHubProjectStatus.TESTED)
                    return snapshots
        return {symbol: self.get_snapshot(symbol) for symbol in symbols}


class AccountHealthMonitor:
    def __init__(self, initial_capital: float) -> None:
        self.initial_capital = float(initial_capital)
        self.current_equity = float(initial_capital)
        self.current_cash = float(initial_capital)
        self.current_drawdown = 0.0
        self.gross_exposure = 0.0
        self.updated_at = datetime.utcnow().isoformat()

    def sync(self, snapshot: Dict[str, Any]) -> None:
        self.current_equity = float(snapshot.get("equity", self.current_equity))
        self.current_cash = float(snapshot.get("cash", self.current_cash))
        self.current_drawdown = float(snapshot.get("drawdown", self.current_drawdown))
        self.gross_exposure = float(snapshot.get("gross_exposure", self.gross_exposure))
        self.updated_at = datetime.utcnow().isoformat()

    def can_trade(self, proposed_notional: float) -> tuple[bool, str]:
        if self.current_drawdown <= -0.12:
            return False, f"账户回撤 {self.current_drawdown:.2%} 已超过 12% 熔断线"
        if self.current_cash < 0:
            return False, "账户现金为负，禁止继续开新仓"
        if proposed_notional > 0 and self.current_cash < proposed_notional * 0.1:
            return False, "账户可用现金不足以承接新交易"
        return True, "账户健康，允许交易"

    def status(self) -> Dict[str, Any]:
        return {
            "initial_capital": round(self.initial_capital, 2),
            "equity": round(self.current_equity, 2),
            "cash": round(self.current_cash, 2),
            "drawdown": round(self.current_drawdown, 6),
            "gross_exposure": round(self.gross_exposure, 2),
            "updated_at": self.updated_at,
        }


class EnhancedCausalInferenceEngine:
    def __init__(self, use_novaaware: bool = True) -> None:
        self.use_novaaware = use_novaaware
        self.novaaware_engine = None
        self.pcmci_engine = None
        self._init_engines()

    def _init_engines(self) -> None:
        if self.use_novaaware:
            try:
                module = importlib.import_module("novaaware")
                engine_cls = getattr(module, "NovaAwareEngine", None)
                if engine_cls is not None:
                    self.novaaware_engine = engine_cls()
            except ImportError:
                self.novaaware_engine = None
        if self.novaaware_engine is None:
            try:
                from tigramite.pcmci import PCMCI
                from tigramite.independence_tests.parcorr import ParCorr

                self.pcmci_engine = {"PCMCI": PCMCI, "ParCorr": ParCorr}
            except ImportError:
                self.pcmci_engine = None

    def discover_causal_graph(self, data: pd.DataFrame, tau_max: int = 2, pc_alpha: float = 0.01) -> Dict[str, Any]:
        clean = data.dropna().copy()
        if clean.empty:
            return {"engine": "none", "method": "unavailable", "results": {}, "edges": [], "summary": {"edge_count": 0}}
        if self.novaaware_engine is not None:
            return self._discover_with_novaaware(clean, tau_max, pc_alpha)
        if self.pcmci_engine is not None:
            return self._discover_with_pcmci(clean, tau_max, pc_alpha)
        return self._discover_with_fallback(clean, tau_max, pc_alpha)

    def _discover_with_novaaware(self, data: pd.DataFrame, tau_max: int, pc_alpha: float) -> Dict[str, Any]:
        try:
            results = self.novaaware_engine.discover_causal_graph(
                data=data.values,
                var_names=data.columns.tolist(),
                tau_max=tau_max,
                alpha=pc_alpha,
            )
            return {
                "engine": "novaaware",
                "method": "digital_consciousness",
                "results": results,
                "edges": self._fallback_edges_from_matrix(data, tau_max),
                "summary": {"edge_count": len(self._fallback_edges_from_matrix(data, tau_max))},
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "engine": "novaaware_failed",
                "method": "fallback_correlation",
                "error": str(exc),
                **self._discover_with_fallback(data, tau_max, pc_alpha),
            }

    def _discover_with_pcmci(self, data: pd.DataFrame, tau_max: int, pc_alpha: float) -> Dict[str, Any]:
        try:
            from tigramite import data_processing as pp

            dataframe = pp.DataFrame(data.values, datatime=np.arange(len(data)), var_names=data.columns.tolist())
            pcmci = self.pcmci_engine["PCMCI"](
                dataframe=dataframe,
                cond_ind_test=self.pcmci_engine["ParCorr"](significance="analytic"),
                verbosity=0,
            )
            results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
            edges = self._extract_pcmci_edges(results, data.columns.tolist(), pc_alpha)
            return {
                "engine": "PCMCI",
                "method": "partial_correlation",
                "results": results,
                "edges": edges,
                "summary": {"edge_count": len(edges)},
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "engine": "pcmci_failed",
                "method": "fallback_correlation",
                "error": str(exc),
                **self._discover_with_fallback(data, tau_max, pc_alpha),
            }

    def _discover_with_fallback(self, data: pd.DataFrame, tau_max: int, pc_alpha: float) -> Dict[str, Any]:
        edges: List[Dict[str, Any]] = []
        columns = data.columns.tolist()
        for source in columns:
            for target in columns:
                if source == target:
                    continue
                for lag in range(1, tau_max + 1):
                    shifted_source = data[source].shift(lag)
                    aligned = pd.concat([shifted_source, data[target]], axis=1).dropna()
                    if len(aligned) < 10:
                        continue
                    strength = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                    pseudo_p = max(0.0001, 1 - min(abs(strength) * math.sqrt(len(aligned)), 0.9999))
                    if abs(strength) >= max(pc_alpha * 5, 0.18):
                        edges.append(
                            {
                                "source": source,
                                "target": target,
                                "lag": lag,
                                "strength": round(strength, 4),
                                "p_value": round(pseudo_p, 4),
                            }
                        )
        edges.sort(key=lambda item: abs(item["strength"]), reverse=True)
        return {
            "engine": "heuristic_fallback",
            "method": "lagged_correlation_screen",
            "results": {"edge_count": len(edges), "tau_max": tau_max},
            "edges": edges[:20],
            "summary": {
                "edge_count": len(edges),
                "top_driver": edges[0]["source"] if edges else None,
                "strongest_edge": edges[0] if edges else None,
            },
        }

    def _extract_pcmci_edges(self, results: Dict[str, Any], var_names: List[str], pc_alpha: float) -> List[Dict[str, Any]]:
        p_matrix = results.get("p_matrix")
        val_matrix = results.get("val_matrix")
        edges: List[Dict[str, Any]] = []
        if p_matrix is None or val_matrix is None:
            return edges
        for target_idx, target in enumerate(var_names):
            for source_idx, source in enumerate(var_names):
                if source_idx == target_idx:
                    continue
                for lag_idx in range(1, p_matrix.shape[2]):
                    p_value = float(p_matrix[source_idx, target_idx, lag_idx])
                    strength = float(val_matrix[source_idx, target_idx, lag_idx])
                    if p_value <= pc_alpha:
                        edges.append(
                            {
                                "source": source,
                                "target": target,
                                "lag": lag_idx,
                                "strength": round(strength, 4),
                                "p_value": round(p_value, 4),
                            }
                        )
        edges.sort(key=lambda item: abs(item["strength"]), reverse=True)
        return edges[:20]

    def _fallback_edges_from_matrix(self, data: pd.DataFrame, tau_max: int) -> List[Dict[str, Any]]:
        return self._discover_with_fallback(data, tau_max, 0.01)["edges"]


class EnhancedCausalTradingAgent:
    def __init__(self, account_monitor: AccountHealthMonitor, use_causal_ai_agent: bool = True) -> None:
        self.account_monitor = account_monitor
        self.use_causal_ai_agent = use_causal_ai_agent
        self.causal_ai_agent = None
        self.positions: Dict[str, Any] = {}
        self.causal_graph: Optional[Dict[str, Any]] = None
        self.event_calendar = self._build_event_calendar()
        self._init_causal_ai_agent()

    def _init_causal_ai_agent(self) -> None:
        if self.use_causal_ai_agent:
            try:
                module = importlib.import_module("causal_ai_agent")
                agent_cls = getattr(module, "CausalAIAgent", None)
                if agent_cls is not None:
                    self.causal_ai_agent = agent_cls()
            except ImportError:
                self.causal_ai_agent = None

    def _build_event_calendar(self) -> Dict[str, str]:
        return {
            "2026-02-13": "节前最后交易日 - 监测期权IV",
            "2026-02-15": "春节休市开始 - 监测内外盘基差",
            "2026-02-18": "美国FOMC纪要 - 监测美债收益率冲击",
            "2026-02-23": "春节休市结束",
            "2026-02-25": "英伟达财报 - 判断算力基建对金属需求",
        }

    def update_causal_graph(self, causal_results: Dict[str, Any]) -> None:
        self.causal_graph = causal_results
        if self.causal_ai_agent is not None:
            try:
                self.causal_ai_agent.update_causal_graph(causal_results)
            except Exception:
                pass

    def execute_decision(self, current_date: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.causal_ai_agent is not None:
            decision = self._execute_with_causal_ai_agent(current_date, market_data)
            if decision:
                return decision
        return self._execute_with_base_logic(current_date, market_data)

    def _execute_with_causal_ai_agent(self, current_date: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        can_trade, message = self.account_monitor.can_trade(0)
        if not can_trade:
            return {
                "actions": [
                    {
                        "action": "FORCE_LIQUIDATE",
                        "symbol": "ALL",
                        "reason": message,
                        "priority": "CRITICAL",
                    }
                ],
                "timestamp": current_date,
                "engine": "Causal-AI-Agent",
            }
        try:
            decision = self.causal_ai_agent.make_decision(
                current_date=current_date,
                market_data=market_data,
                causal_graph=self.causal_graph,
            )
            if isinstance(decision, dict):
                decision.setdefault("engine", "Causal-AI-Agent")
                decision.setdefault("timestamp", current_date)
                if "actions" not in decision and "action" in decision:
                    decision["actions"] = [decision]
                return decision
        except Exception:
            return {}
        return {}

    def _execute_with_base_logic(self, current_date: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        can_trade, message = self.account_monitor.can_trade(0)
        if not can_trade:
            return {
                "actions": [
                    {
                        "action": "FORCE_LIQUIDATE",
                        "symbol": "ALL",
                        "reason": message,
                        "priority": "CRITICAL",
                    }
                ],
                "timestamp": current_date,
                "engine": "Base Logic",
            }

        actions: List[Dict[str, Any]] = []
        if self._check_gold_causal_trigger(market_data):
            actions.append(
                {
                    "action": "LONG",
                    "symbol": "COMEX_Gold",
                    "reason": "美债信用坍塌风险 + 全球央行购金支撑",
                    "priority": "HIGH",
                    "confidence": 0.78,
                }
            )
        if self._check_copper_causal_trigger(market_data):
            actions.append(
                {
                    "action": "LONG",
                    "symbol": "LME_Copper",
                    "reason": "AI产业刚性需求 + 库存低位逼空预警",
                    "priority": "HIGH",
                    "confidence": 0.71,
                }
            )
        if not actions:
            actions.append(
                {
                    "action": "HOLD",
                    "symbol": "CASH",
                    "reason": "未触发显著因果节点，维持观察",
                    "priority": "LOW",
                    "confidence": 0.55,
                }
            )
        return {"actions": actions, "timestamp": current_date, "engine": "Base Logic"}

    def _check_gold_causal_trigger(self, market_data: Dict[str, Any]) -> bool:
        us_debt = market_data.get("US_Debt", {}).get("value", 0)
        central_bank_gold = market_data.get("Central_Bank_Gold_Purchase", {}).get("value", 0)
        on_rrp = market_data.get("ON_RRP_Balance", {}).get("value", 0)
        return bool(us_debt > 38_000_000_000_000 or central_bank_gold > 83.33 or on_rrp < 500_000_000_000)

    def _check_copper_causal_trigger(self, market_data: Dict[str, Any]) -> bool:
        lme_inventory = market_data.get("LME_Inventory_Days", {}).get("value", 0)
        ai_capex = market_data.get("AI_DataCenter_Capex", {}).get("growth", 0)
        return bool(lme_inventory < 3 or ai_capex > 0.2)


class CausalTradingSystemV4:
    def __init__(self, base_dir: str, initial_capital: float = 1_000_000.0) -> None:
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.github_manager = GitHubProjectManager()
        self.data_adapter = EnhancedDataAdapter(
            str(self.data_dir),
            default_source=DataSource.AUTO,
            github_manager=self.github_manager,
        )
        self.ecosystem = EcosystemIntegrationManager(str(self.base_dir), self.github_manager)
        self.causal_inference_engine = EnhancedCausalInferenceEngine(use_novaaware=True)
        self.account_monitor = AccountHealthMonitor(initial_capital)
        self.trading_agent = EnhancedCausalTradingAgent(self.account_monitor, use_causal_ai_agent=True)

    def sync_account_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.account_monitor.sync(snapshot)

    def get_system_status(self) -> Dict[str, Any]:
        return {
            "version": "V4.0",
            "system_name": "因果AI量化交易系统",
            "data_adapter": "Enhanced V2.0",
            "causal_engine": "novaaware" if self.causal_inference_engine.novaaware_engine else (
                "PCMCI" if self.causal_inference_engine.pcmci_engine else "heuristic_fallback"
            ),
            "trading_agent": "Causal-AI-Agent" if self.trading_agent.causal_ai_agent else "Base Logic",
            "account_monitor": self.account_monitor.status(),
            "github_projects": self.github_manager.get_project_status(),
            "ecosystem": self.ecosystem.status(),
            "event_calendar": self.trading_agent.event_calendar,
        }

    def build_market_data(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        gold = datasets.get("gold_daily")
        copper = datasets.get("copper_daily")
        nasdaq = datasets.get("nasdaq_daily")

        def pct(frame: Optional[pd.DataFrame], window: int) -> float:
            if frame is None or len(frame) <= window:
                return 0.0
            close = frame["close"].astype(float)
            return float(close.iloc[-1] / close.iloc[-window - 1] - 1)

        gold_mom = pct(gold, 60)
        copper_mom = pct(copper, 30)
        nasdaq_mom = pct(nasdaq, 60)
        market_data = {
            "US_Debt": {"value": 38_500_000_000_000, "source": "macro_baseline"},
            "Central_Bank_Gold_Purchase": {"value": round(78 + max(gold_mom, 0) * 120, 2), "source": "gold_proxy"},
            "ON_RRP_Balance": {"value": max(350_000_000_000, 620_000_000_000 - max(nasdaq_mom, 0) * 400_000_000_000), "source": "liquidity_proxy"},
            "LME_Inventory_Days": {"value": round(max(1.5, 5.5 - max(copper_mom, 0) * 14), 2), "source": "copper_proxy"},
            "AI_DataCenter_Capex": {"growth": round(max(0.08, 0.12 + max(nasdaq_mom, 0) * 0.9), 4), "source": "tech_proxy"},
        }
        finshare_snapshots = self.data_adapter.get_batch_snapshots(["AAPL", "MSFT", "600000"])
        market_data["finshare_snapshots"] = finshare_snapshots
        market_data["openbb_market_context"] = self.ecosystem.fetch_openbb_market_context(["AAPL", "MSFT", "GOOGL"])
        return market_data

    def full_analysis_pipeline_v4(
        self,
        symbols: List[str],
        raw_datasets: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        data = self.data_adapter.get_batch_data(symbols, period="6mo")
        technical_pack = self.ecosystem.compute_technical_pack(data)
        shortlist = self.ecosystem.shortlist_symbols(data, technical_pack, limit=min(4, max(1, len(symbols))))
        shortlisted_data = {symbol: data[symbol] for symbol in shortlist if symbol in data}
        results["data"] = {
            symbol: {
                "rows": int(len(frame)),
                "source": frame["Source"].iloc[-1],
                "latest_close": float(frame["Close"].iloc[-1]),
            }
            for symbol, frame in data.items()
        }
        results["technical_pack"] = technical_pack
        results["shortlist"] = shortlist

        if shortlisted_data:
            combined = pd.concat([frame.set_index("Date")[["Close"]] for frame in shortlisted_data.values()], axis=1)
            combined.columns = list(shortlisted_data.keys())
            causal_graph = self.causal_inference_engine.discover_causal_graph(combined, tau_max=2, pc_alpha=0.01)
            self.trading_agent.update_causal_graph(causal_graph)
            results["causal_graph"] = causal_graph

        if raw_datasets is None:
            raw_datasets = {}
        market_data = self.build_market_data(raw_datasets)
        results["market_data"] = market_data
        committee = self.ecosystem.run_multi_agent_committee(
            technical_pack=technical_pack,
            causal_graph=results.get("causal_graph", {}),
            market_data=market_data,
            account_status=self.account_monitor.status(),
        )
        results["committee"] = committee
        decision = self.trading_agent.execute_decision(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            market_data=market_data,
        )
        results["decision"] = decision
        compact_report = self.ecosystem.build_compact_evidence_pack(
            shortlist,
            technical_pack,
            results.get("causal_graph", {}),
            market_data,
        )
        results["optimized_reasoning"] = compact_report
        results["evaluation"] = self.ecosystem.evaluate_reasoning_quality(results, compact_report)
        return results
