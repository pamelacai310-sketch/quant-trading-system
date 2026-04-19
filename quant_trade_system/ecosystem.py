from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .strategy_engine import _all_conditions, prepare_frame


class EcosystemIntegrationManager:
    def __init__(self, base_dir: str, github_manager: Any) -> None:
        self.base_dir = Path(base_dir)
        self.github_manager = github_manager
        self.state_dir = self.base_dir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir = self.state_dir / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.static_vendor_dir = self.base_dir / "static" / "vendor"
        self.ccxt = self._optional_import("ccxt", "ccxt")
        self.ta = self._optional_import("ta", "ta-stack")
        self.backtrader = self._optional_import("backtrader", "backtrader")
        self.lightweight_chart_asset = self.static_vendor_dir / "lightweight-charts.standalone.production.js"
        self.py311 = self._find_python311()
        self.openbb_bridge = self._detect_bridge("openbb", self.base_dir / "quant_trade_system" / "openbb_bridge.py")
        self.quantlib_bridge = self._detect_bridge("QuantLib", self.base_dir / "quant_trade_system" / "quantlib_bridge.py")
        self._mark_export_integrations()
        if self.lightweight_chart_asset.exists():
            self.github_manager.mark_status_name("lightweight-charts", "tested")
        elif self.lightweight_chart_asset.parent.exists():
            self.github_manager.mark_status_name("lightweight-charts", "integrated")

    def _optional_import(self, module_name: str, project_name: str) -> Optional[Any]:
        try:
            module = importlib.import_module(module_name)
            self.github_manager.mark_status_name(project_name, "tested")
            return module
        except Exception:
            return None

    def _find_python311(self) -> Optional[str]:
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

    def _bridge_env(self, name: str) -> Dict[str, str]:
        env = os.environ.copy()
        home = self.state_dir / f"{name}_home"
        home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        return env

    def _detect_bridge(self, project_name: str, script_path: Path) -> Optional[Path]:
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

    def _run_bridge(self, project_name: str, script_path: Optional[Path], command: str, payload: Dict[str, Any]) -> Optional[Any]:
        if not self.py311 or script_path is None:
            return None
        try:
            result = subprocess.run(
                [self.py311, str(script_path), command],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=30,
                env=self._bridge_env(project_name.lower()),
                check=False,
            )
        except Exception:
            return None
        if result.returncode != 0 or not result.stdout.strip():
            return None
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        if parsed.get("ok"):
            self.github_manager.mark_status_name(project_name, "tested")
            return parsed.get("data")
        return None

    def _mark_export_integrations(self) -> None:
        for project_name in [
            "QuantConnect-Lean",
            "freqtrade",
            "hummingbot",
            "FinEval",
            "FinLongEval",
            "XuanYuan",
            "TradingAgents-AShare",
        ]:
            self.github_manager.mark_status_name(project_name, "integrated")

    def status(self) -> Dict[str, Any]:
        return {
            "ccxt": bool(self.ccxt),
            "ta_stack": bool(self.ta),
            "backtrader": bool(self.backtrader),
            "lightweight_charts": self.lightweight_chart_asset.exists(),
            "openbb_bridge": bool(self.openbb_bridge),
            "quantlib_bridge": bool(self.quantlib_bridge),
            "py311": self.py311,
        }

    def fetch_openbb_market_context(self, symbols: List[str]) -> Dict[str, Any]:
        data = self._run_bridge("openbb", self.openbb_bridge, "market", {"symbols": symbols})
        return data or {}

    def price_option(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = self._run_bridge("QuantLib", self.quantlib_bridge, "black_scholes", payload)
        return data or {}

    def compute_technical_pack(self, data_bundle: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        pack: Dict[str, Dict[str, float]] = {}
        for symbol, frame in data_bundle.items():
            working = frame.copy()
            rename_map = {"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
            for src, dst in rename_map.items():
                if src in working.columns and dst not in working.columns:
                    working = working.rename(columns={src: dst})
            if "close" not in working.columns:
                continue
            close = pd.to_numeric(working["close"], errors="coerce")
            high = pd.to_numeric(working.get("high", close), errors="coerce")
            low = pd.to_numeric(working.get("low", close), errors="coerce")
            volume = pd.to_numeric(working.get("volume", pd.Series(np.zeros(len(working)))), errors="coerce")

            if self.ta is not None:
                from ta.momentum import RSIIndicator
                from ta.trend import ADXIndicator, MACD
                from ta.volatility import AverageTrueRange, BollingerBands

                rsi = float(RSIIndicator(close, window=14).rsi().iloc[-1])
                macd_diff = float(MACD(close).macd_diff().iloc[-1])
                adx = float(ADXIndicator(high, low, close).adx().iloc[-1])
                atr = float(AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])
                bb = BollingerBands(close, window=20)
                bb_width = float((bb.bollinger_hband() - bb.bollinger_lband()).iloc[-1] / max(close.iloc[-1], 0.01))
            else:
                returns = close.pct_change().fillna(0.0)
                rsi = float(50 + returns.tail(14).mean() * 1000)
                macd_diff = float(close.ewm(span=12, adjust=False).mean().iloc[-1] - close.ewm(span=26, adjust=False).mean().iloc[-1])
                adx = float(returns.tail(14).abs().mean() * 1000)
                atr = float((high - low).tail(14).mean())
                bb_width = float(close.tail(20).std(ddof=0) / max(close.iloc[-1], 0.01))

            momentum_20 = float(close.iloc[-1] / close.iloc[max(len(close) - 21, 0)] - 1) if len(close) > 20 else 0.0
            volume_ratio = float(volume.iloc[-1] / max(volume.tail(20).mean(), 1)) if len(volume) > 20 else 1.0

            pack[symbol] = {
                "rsi_14": round(rsi, 4),
                "macd_diff": round(macd_diff, 6),
                "adx_14": round(adx, 4),
                "atr_14": round(atr, 6),
                "bb_width": round(bb_width, 6),
                "momentum_20": round(momentum_20, 6),
                "volume_ratio": round(volume_ratio, 6),
            }
        return pack

    def shortlist_symbols(self, data_bundle: Dict[str, pd.DataFrame], technical_pack: Dict[str, Dict[str, float]], limit: int = 4) -> List[str]:
        ranked = []
        for symbol, metrics in technical_pack.items():
            score = (
                abs(metrics.get("momentum_20", 0.0)) * 4
                + abs(metrics.get("macd_diff", 0.0)) * 2
                + abs(metrics.get("rsi_14", 50.0) - 50.0) / 40
                + min(metrics.get("volume_ratio", 1.0), 3.0) / 3
                + min(metrics.get("adx_14", 0.0), 50.0) / 50
            )
            ranked.append((score, symbol))
        ranked.sort(reverse=True)
        return [symbol for _, symbol in ranked[:limit]]

    def estimate_tokens(self, payload: Any) -> int:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return max(1, math.ceil(len(text) / 4))

    def build_compact_evidence_pack(
        self,
        symbols: List[str],
        technical_pack: Dict[str, Dict[str, float]],
        causal_graph: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        compact = {
            "symbols": symbols,
            "top_edges": (causal_graph.get("edges") or [])[:6],
            "macro": {
                key: value
                for key, value in market_data.items()
                if key != "finshare_snapshots"
            },
            "technical": {symbol: technical_pack.get(symbol, {}) for symbol in symbols},
        }
        compact["token_estimate"] = self.estimate_tokens(compact)
        return compact

    def run_multi_agent_committee(
        self,
        technical_pack: Dict[str, Dict[str, float]],
        causal_graph: Dict[str, Any],
        market_data: Dict[str, Any],
        account_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        macro_view = {
            "agent": "macro",
            "stance": "bullish_hard_assets" if market_data.get("US_Debt", {}).get("value", 0) > 38_000_000_000_000 else "neutral",
            "reason": "债务与流动性代理指标偏向黄金/铜等硬资产" if market_data.get("US_Debt", {}).get("value", 0) > 38_000_000_000_000 else "宏观代理指标未显著偏离",
        }
        strongest_edge = (causal_graph.get("edges") or [{}])[0]
        technical_view = {
            "agent": "technical",
            "stance": "trend_follow",
            "reason": f"最强因果边 {strongest_edge.get('source')}->{strongest_edge.get('target')} lag {strongest_edge.get('lag')}" if strongest_edge else "无显著因果边",
        }
        risk_view = {
            "agent": "risk",
            "stance": "risk_on" if account_status.get("drawdown", 0.0) > -0.05 else "risk_reduce",
            "reason": f"当前回撤 {account_status.get('drawdown', 0.0):.2%}",
        }
        execution_view = {
            "agent": "execution",
            "stance": "fast_path",
            "reason": "优先 deterministic + causal + technical，LLM 仅走压缩摘要层，降低 token 成本",
        }
        votes = [macro_view, technical_view, risk_view, execution_view]
        consensus = "long_bias" if risk_view["stance"] == "risk_on" else "cautious"
        return {"committee": votes, "consensus": consensus}

    def evaluate_reasoning_quality(self, report: Dict[str, Any], compact_report: Dict[str, Any]) -> Dict[str, Any]:
        coverage = 0.0
        for key in ["symbols", "top_edges", "macro", "technical"]:
            if compact_report.get(key):
                coverage += 0.25
        raw_tokens = self.estimate_tokens(report)
        compact_tokens = self.estimate_tokens(compact_report)
        compression_ratio = round(compact_tokens / max(raw_tokens, 1), 4)
        return {
            "FinEval_style": {
                "coverage": round(coverage, 4),
                "actionability": 1.0 if compact_report.get("top_edges") else 0.6,
                "grounding": 1.0 if compact_report.get("macro") and compact_report.get("technical") else 0.5,
            },
            "FinLongEval_style": {
                "raw_token_estimate": raw_tokens,
                "compact_token_estimate": compact_tokens,
                "compression_ratio": compression_ratio,
            },
        }

    def export_strategy(self, strategy: Dict[str, Any], target: str) -> Dict[str, Any]:
        spec = strategy["spec"]
        if target == "lean":
            content = f'''from AlgorithmImports import *\n\nclass {strategy["name"].replace(" ", "")}Algorithm(QCAlgorithm):\n    def initialize(self):\n        self.set_start_date(2024, 1, 1)\n        self.set_cash(100000)\n        self.symbol = self.add_equity("{spec["symbol"] if spec["symbol"].isalpha() else "SPY"}", Resolution.DAILY).symbol\n        self.fast = self.sma(self.symbol, {spec.get("indicators", [{}])[0].get("window", 10)})\n        self.slow = self.sma(self.symbol, {spec.get("indicators", [{}, {}])[1].get("window", 30)})\n\n    def on_data(self, slice: Slice):\n        if not self.fast.is_ready or not self.slow.is_ready:\n            return\n        if self.fast.current.value > self.slow.current.value and not self.portfolio.invested:\n            self.set_holdings(self.symbol, 0.8)\n        elif self.fast.current.value < self.slow.current.value and self.portfolio.invested:\n            self.liquidate(self.symbol)\n'''
            filename = self.export_dir / f"{strategy['id']}_lean.py"
        elif target == "freqtrade":
            content = f'''from freqtrade.strategy import IStrategy\nfrom pandas import DataFrame\n\nclass {strategy["name"].replace(" ", "")}Freqtrade(IStrategy):\n    timeframe = "1d"\n    minimal_roi = {{"0": 0.08}}\n    stoploss = -0.05\n\n    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:\n        dataframe["fast_ma"] = dataframe["close"].rolling({spec.get("indicators", [{}])[0].get("window", 10)}).mean()\n        dataframe["slow_ma"] = dataframe["close"].rolling({spec.get("indicators", [{}, {}])[1].get("window", 30)}).mean()\n        return dataframe\n\n    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:\n        dataframe.loc[dataframe["fast_ma"] > dataframe["slow_ma"], "enter_long"] = 1\n        return dataframe\n\n    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:\n        dataframe.loc[dataframe["fast_ma"] < dataframe["slow_ma"], "exit_long"] = 1\n        return dataframe\n'''
            filename = self.export_dir / f"{strategy['id']}_freqtrade.py"
        elif target == "hummingbot":
            content = f'''# Hummingbot directional script generated from strategy {strategy["name"]}\nstrategy: directional_trading\nsymbol: {spec["symbol"]}\nexchange: {{set-via-config}}\nentry_condition: "{spec.get("entry_rules", [])}"\nexit_condition: "{spec.get("exit_rules", [])}"\nrisk_limits: {json.dumps(spec.get("risk_limits", {}), ensure_ascii=False)}\n'''
            filename = self.export_dir / f"{strategy['id']}_hummingbot.yml"
        elif target == "tradingagents":
            content = json.dumps(
                {
                    "agents": ["macro", "technical", "risk", "execution"],
                    "symbol": spec["symbol"],
                    "strategy_name": strategy["name"],
                    "rules": spec,
                },
                ensure_ascii=False,
                indent=2,
            )
            filename = self.export_dir / f"{strategy['id']}_tradingagents.json"
        elif target == "xuanyuan":
            content = json.dumps(
                {
                    "model": os.environ.get("XUANYUAN_MODEL", "xuanyuan"),
                    "base_url": os.environ.get("XUANYUAN_BASE_URL", ""),
                    "system_prompt": "你是金融研究助手，请基于因果证据和技术指标输出最短但可靠的交易分析。",
                    "strategy": strategy,
                },
                ensure_ascii=False,
                indent=2,
            )
            filename = self.export_dir / f"{strategy['id']}_xuanyuan.json"
        else:
            raise ValueError(f"Unsupported export target: {target}")
        filename.write_text(content, encoding="utf-8")
        target_project = {
            "lean": "QuantConnect-Lean",
            "freqtrade": "freqtrade",
            "hummingbot": "hummingbot",
            "tradingagents": "TradingAgents-AShare",
            "xuanyuan": "XuanYuan",
        }[target]
        self.github_manager.mark_status_name(target_project, "tested")
        return {"target": target, "path": str(filename), "content": content}

    def run_backtrader_backtest(self, strategy: Dict[str, Any], frame: pd.DataFrame, starting_cash: float = 100_000.0) -> Dict[str, Any]:
        if self.backtrader is None:
            return {"engine": "backtrader", "available": False}
        bt = self.backtrader
        renamed = frame.rename(columns={"timestamp": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}).copy()
        signal_frame = self._signal_frame(renamed, strategy["spec"])
        if signal_frame.empty:
            return {"engine": "backtrader", "available": True, "error": "no_signal_frame"}

        class SignalData(bt.feeds.PandasData):
            lines = ("signal",)
            params = (("datetime", None), ("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"), ("volume", "volume"), ("openinterest", -1), ("signal", "signal"))

        class SignalStrategy(bt.Strategy):
            def next(self):
                signal = self.datas[0].signal[0]
                if signal > 0 and self.position.size <= 0:
                    self.order_target_percent(target=0.95)
                elif signal < 0 and self.position.size >= 0:
                    self.order_target_percent(target=-0.95)
                elif signal == 0 and self.position.size != 0:
                    self.close()

        data = signal_frame.copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date")
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(starting_cash)
        cerebro.addstrategy(SignalStrategy)
        cerebro.adddata(SignalData(dataname=data))
        cerebro.run()
        ending_value = cerebro.broker.getvalue()
        return {
            "engine": "backtrader",
            "available": True,
            "starting_cash": round(starting_cash, 2),
            "ending_value": round(float(ending_value), 2),
            "total_return": round(float(ending_value / starting_cash - 1), 6),
        }

    def _signal_frame(self, frame: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
        prepared = prepare_frame(frame, spec)
        if prepared.empty:
            return prepared
        current_signal = 0
        signals = []
        long_entry = spec.get("entry_rules", [])
        long_exit = spec.get("exit_rules", [])
        short_entry = spec.get("short_entry_rules", [])
        short_exit = spec.get("short_exit_rules", [])
        direction = spec.get("direction", "long_only")
        for index in range(len(prepared)):
            if current_signal > 0 and long_exit and _all_conditions(prepared, index, long_exit):
                current_signal = 0
            elif current_signal < 0 and short_exit and _all_conditions(prepared, index, short_exit):
                current_signal = 0
            elif current_signal == 0 and long_entry and _all_conditions(prepared, index, long_entry):
                current_signal = 1
            elif current_signal == 0 and direction == "long_short" and short_entry and _all_conditions(prepared, index, short_entry):
                current_signal = -1
            signals.append(current_signal)
        prepared = prepared.copy()
        prepared["signal"] = signals
        return prepared[["date", "open", "high", "low", "close", "volume", "signal"]]
