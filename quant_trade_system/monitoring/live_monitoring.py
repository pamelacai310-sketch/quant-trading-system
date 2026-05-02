"""
实盘监控与熔断机制

功能：
1. 实盘监控MAE/MFE指标
2. 策略健康度评估
3. 自动熔断机制
4. 周报/月报生成
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import json

import numpy as np
import pandas as pd


class CircuitBreakerStatus(Enum):
    """熔断器状态"""
    NORMAL = "normal"              # 正常运行
    WARNING = "warning"            # 警告状态
    PAUSED = "paused"              # 暂停新开仓
    TRIPPED = "tripped"            # 熔断触发（完全停止）
    RECOVERING = "recovering"      # 恢复中


class HealthMetric(Enum):
    """健康度指标"""
    MAE_CONTROL = "mae_control"
    MFE_UTILIZATION = "mfe_utilization"
    WIN_RATE = "win_rate"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    TRADE_FREQUENCY = "trade_frequency"


@dataclass
class MonitoringThreshold:
    """监控阈值"""
    metric: HealthMetric
    warning_level: float          # 警告阈值
    critical_level: float         # 严重阈值
    trip_level: float             # 熔断阈值


@dataclass
class LivePosition:
    """实时持仓"""
    symbol: str
    entry_date: datetime
    entry_price: float
    current_price: float
    position_size: float
    unrealized_pnl_pct: float
    current_mae_pct: float
    current_mfe_pct: float
    holding_days: int


@dataclass
class MonitoringReport:
    """监控报告"""
    timestamp: datetime
    status: CircuitBreakerStatus
    health_scores: Dict[HealthMetric, float]
    triggered_thresholds: List[MonitoringThreshold]
    open_positions: List[LivePosition]
    recent_trades: List[Dict]
    recommendations: List[str]
    summary: str


class LiveMonitor:
    """
    实盘监控器

    持续监控策略运行状态，评估健康度，触发熔断
    """

    def __init__(
        self,
        strategy_id: str,
        monitoring_intervals: Dict[HealthMetric, int] = None,
        circuit_breaker_cooldown_hours: int = 24,
    ):
        """
        Args:
            strategy_id: 策略ID
            monitoring_intervals: 各指标的监控间隔（小时）
            circuit_breaker_cooldown_hours: 熔断冷却期（小时）
        """
        self.strategy_id = strategy_id
        self.monitoring_intervals = monitoring_intervals or {
            HealthMetric.MAE_CONTROL: 1,        # 每小时检查MAE
            HealthMetric.MFE_UTILIZATION: 4,    # 每4小时检查MFE利用率
            HealthMetric.WIN_RATE: 24,          # 每天检查胜率
            HealthMetric.DRAWDOWN: 1,           # 每小时检查回撤
            HealthMetric.VOLATILITY: 24,        # 每天检查波动率
        }
        self.circuit_breaker_cooldown_hours = circuit_breaker_cooldown_hours

        # 状态
        self.status = CircuitBreakerStatus.NORMAL
        self.last_check_time: Dict[HealthMetric, datetime] = {}
        self.circuit_breaker_tripped_time: Optional[datetime] = None
        self.monitoring_history: List[MonitoringReport] = []

        # 阈值配置
        self.thresholds = self._get_default_thresholds()

        # 数据存储
        self.live_positions: Dict[str, LivePosition] = {}
        self.trade_history: List[Dict] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

    def _get_default_thresholds(self) -> List[MonitoringThreshold]:
        """获取默认监控阈值"""

        return [
            # MAE 控制：平均MAE不应超过-8%
            MonitoringThreshold(
                metric=HealthMetric.MAE_CONTROL,
                warning_level=-5.0,
                critical_level=-8.0,
                trip_level=-10.0,
            ),
            # MFE 利用率：不应低于40%
            MonitoringThreshold(
                metric=HealthMetric.MFE_UTILIZATION,
                warning_level=0.50,
                critical_level=0.40,
                trip_level=0.30,
            ),
            # 胜率：不应低于40%
            MonitoringThreshold(
                metric=HealthMetric.WIN_RATE,
                warning_level=0.50,
                critical_level=0.40,
                trip_level=0.30,
            ),
            # 回撤：不应超过15%
            MonitoringThreshold(
                metric=HealthMetric.DRAWDOWN,
                warning_level=0.10,
                critical_level=0.15,
                trip_level=0.20,
            ),
        ]

    def update_position(
        self,
        symbol: str,
        current_price: float,
    ) -> Optional[MonitoringReport]:
        """
        更新持仓数据

        Args:
            symbol: 标的代码
            current_price: 当前价格

        Returns:
            如果触发阈值，返回监控报告
        """

        if symbol not in self.live_positions:
            return None

        position = self.live_positions[symbol]
        position.current_price = current_price

        # 更新MAE/MFE
        if position.position_size > 0:  # 多头
            position.current_mae_pct = min(position.current_mae_pct,
                                           (current_price / position.entry_price - 1) * 100)
            position.current_mfe_pct = max(position.current_mfe_pct,
                                           (current_price / position.entry_price - 1) * 100)
        else:  # 空头
            position.current_mae_pct = min(position.current_mae_pct,
                                           (position.entry_price / current_price - 1) * 100)
            position.current_mfe_pct = max(position.current_mfe_pct,
                                           (position.entry_price / current_price - 1) * 100)

        # 更新持仓天数
        position.holding_days = (datetime.now() - position.entry_date).days

        # 检查是否需要监控
        return self._check_monitoring_conditions()

    def add_position(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
        side: str = 'long',
    ):
        """添加新持仓"""
        self.live_positions[symbol] = LivePosition(
            symbol=symbol,
            entry_date=datetime.now(),
            entry_price=entry_price,
            current_price=entry_price,
            position_size=position_size if side == 'long' else -position_size,
            unrealized_pnl_pct=0.0,
            current_mae_pct=0.0,
            current_mfe_pct=0.0,
            holding_days=0,
        )

    def close_position(
        self,
        symbol: str,
        exit_price: float,
    ) -> Optional[Dict]:
        """平仓"""
        if symbol not in self.live_positions:
            return None

        position = self.live_positions[symbol]

        # 计算最终盈亏
        if position.position_size > 0:  # 多头
            final_pnl_pct = (exit_price / position.entry_price - 1) * 100
        else:  # 空头
            final_pnl_pct = (position.entry_price / exit_price - 1) * 100

        # 记录交易
        trade_record = {
            'symbol': symbol,
            'entry_date': position.entry_date,
            'exit_date': datetime.now(),
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'position_size': abs(position.position_size),
            'final_pnl_pct': final_pnl_pct,
            'mae_pct': position.current_mae_pct,
            'mfe_pct': position.current_mfe_pct,
            'holding_days': position.holding_days,
        }

        self.trade_history.append(trade_record)

        # 移除持仓
        del self.live_positions[symbol]

        return trade_record

    def _check_monitoring_conditions(self) -> Optional[MonitoringReport]:
        """检查监控条件"""

        now = datetime.now()
        triggered_thresholds = []
        health_scores = {}

        # 检查各指标
        for metric, interval_hours in self.monitoring_intervals.items():
            # 检查是否到了监控时间
            last_check = self.last_check_time.get(metric)
            if last_check and (now - last_check).total_seconds() < interval_hours * 3600:
                continue

            # 计算健康度
            health_score = self._calculate_health_score(metric)
            health_scores[metric] = health_score

            # 检查阈值
            for threshold in self.thresholds:
                if threshold.metric == metric:
                    if self._is_threshold_triggered(threshold, health_score):
                        triggered_thresholds.append(threshold)

            self.last_check_time[metric] = now

        # 如果有触发阈值，生成报告
        if triggered_thresholds or self._should_generate_report():
            return self._generate_report(
                health_scores,
                triggered_thresholds,
            )

        return None

    def _calculate_health_score(self, metric: HealthMetric) -> float:
        """计算健康度得分"""

        if metric == HealthMetric.MAE_CONTROL:
            # MAE 控制：计算当前持仓的平均MAE
            if not self.live_positions:
                return 0.0
            maes = [p.current_mae_pct for p in self.live_positions.values()]
            return np.mean(maes) if maes else 0.0

        elif metric == HealthMetric.MFE_UTILIZATION:
            # MFE 利用率：计算最近盈利交易的MFE利用率
            profitable_trades = [
                t for t in self.trade_history[-20:]  # 最近20笔
                if t['final_pnl_pct'] > 0 and t['mfe_pct'] > 0
            ]
            if not profitable_trades:
                return 1.0

            mfe_utils = [
                t['final_pnl_pct'] / t['mfe_pct']
                for t in profitable_trades
            ]
            return np.mean(mfe_utils)

        elif metric == HealthMetric.WIN_RATE:
            # 胜率：最近20笔交易
            recent_trades = self.trade_history[-20:]
            if not recent_trades:
                return 1.0

            wins = sum(1 for t in recent_trades if t['final_pnl_pct'] > 0)
            return wins / len(recent_trades)

        elif metric == HealthMetric.DRAWDOWN:
            # 回撤
            if len(self.equity_curve) < 2:
                return 0.0

            equity_values = [e[1] for e in self.equity_curve]
            peak = max(equity_values)
            current = equity_values[-1]
            return (current / peak - 1)

        else:
            return 0.0

    def _is_threshold_triggered(
        self,
        threshold: MonitoringThreshold,
        current_value: float,
    ) -> bool:
        """判断阈值是否被触发"""

        if threshold.metric in [HealthMetric.MAE_CONTROL, HealthMetric.DRAWDOWN]:
            # 负值指标：越小越不好
            return current_value <= threshold.trip_level
        else:
            # 正值指标：越大越好，所以触发条件是小于trip_level
            return current_value <= threshold.trip_level

    def _should_generate_report(self) -> bool:
        """判断是否应该生成报告"""

        # 定期生成报告（每天一次）
        now = datetime.now()
        if not self.monitoring_history:
            return True

        last_report = self.monitoring_history[-1]
        if (now - last_report.timestamp).days >= 1:
            return True

        return False

    def _generate_report(
        self,
        health_scores: Dict[HealthMetric, float],
        triggered_thresholds: List[MonitoringThreshold],
    ) -> MonitoringReport:
        """生成监控报告"""

        # 评估状态
        status = self._assess_circuit_breaker_status(triggered_thresholds)

        # 生成建议
        recommendations = self._generate_recommendations(
            health_scores, triggered_thresholds, status
        )

        # 生成摘要
        summary = self._generate_summary(
            health_scores, triggered_thresholds, status
        )

        report = MonitoringReport(
            timestamp=datetime.now(),
            status=status,
            health_scores=health_scores,
            triggered_thresholds=triggered_thresholds,
            open_positions=list(self.live_positions.values()),
            recent_trades=self.trade_history[-10:],
            recommendations=recommendations,
            summary=summary,
        )

        self.monitoring_history.append(report)

        return report

    def _assess_circuit_breaker_status(
        self,
        triggered_thresholds: List[MonitoringThreshold],
    ) -> CircuitBreakerStatus:
        """评估熔断器状态"""

        if self.status == CircuitBreakerStatus.TRIPPED:
            # 检查是否可以恢复
            if self.circuit_breaker_tripped_time:
                cooldown_end = (
                    self.circuit_breaker_tripped_time +
                    timedelta(hours=self.circuit_breaker_cooldown_hours)
                )
                if datetime.now() >= cooldown_end:
                    return CircuitBreakerStatus.RECOVERING
            return CircuitBreakerStatus.TRIPPED

        if not triggered_thresholds:
            return CircuitBreakerStatus.NORMAL

        # 检查严重程度
        critical_count = sum(
            1 for t in triggered_thresholds
            if any(abs(t.trip_level - threshold.critical_level) < 0.01
                   for threshold in self.thresholds
                   if threshold.metric == t.metric)
        )

        if critical_count >= 2:
            # 2个以上严重阈值触发：熔断
            self.circuit_breaker_tripped_time = datetime.now()
            return CircuitBreakerStatus.TRIPPED
        elif critical_count >= 1:
            # 1个严重阈值触发：暂停新开仓
            return CircuitBreakerStatus.PAUSED
        else:
            # 警告级别触发
            return CircuitBreakerStatus.WARNING

    def _generate_recommendations(
        self,
        health_scores: Dict[HealthMetric, float],
        triggered_thresholds: List[MonitoringThreshold],
        status: CircuitBreakerStatus,
    ) -> List[str]:
        """生成优化建议"""

        recommendations = []

        for threshold in triggered_thresholds:
            metric = threshold.metric

            if metric == HealthMetric.MAE_CONTROL:
                current_mae = health_scores.get(metric, 0)
                recommendations.append(
                    f"⚠️ 平均MAE达到{current_mae:.1f}%，持仓体验极差。\n"
                    f"   建议：检查买入条件是否过于激进，考虑增加右侧确认信号"
                )

            elif metric == HealthMetric.MFE_UTILIZATION:
                current_util = health_scores.get(metric, 0) * 100
                recommendations.append(
                    f"⚠️ MFE利用率仅{current_util:.1f}%，大量利润回撤。\n"
                    f"   建议：引入或优化动态跟踪止盈机制"
                )

            elif metric == HealthMetric.WIN_RATE:
                current_wr = health_scores.get(metric, 0) * 100
                recommendations.append(
                    f"⚠️ 胜率降至{current_wr:.1f}%，策略可能失效。\n"
                    f"   建议：检查市场环境是否发生变化，考虑暂停交易或优化参数"
                )

            elif metric == HealthMetric.DRAWDOWN:
                current_dd = health_scores.get(metric, 0) * 100
                recommendations.append(
                    f"⚠️ 回撤达到{current_dd:.1f}%，接近风险极限。\n"
                    f"   建议：降低仓位规模或收紧止损"
                )

        # 根据状态给出建议
        if status == CircuitBreakerStatus.TRIPPED:
            recommendations.insert(
                0, "🛑 熔断已触发！立即停止所有交易，重新评估策略有效性。\n"
            )
        elif status == CircuitBreakerStatus.PAUSED:
            recommendations.insert(
                0, "⏸️ 暂停新开仓，继续持有现有仓位直至信号平仓。\n"
            )
        elif status == CircuitBreakerStatus.WARNING:
            recommendations.insert(
                0, "⚠️ 策略进入警告状态，建议降低仓位规模并密切监控。\n"
            )

        return recommendations

    def _generate_summary(
        self,
        health_scores: Dict[HealthMetric, float],
        triggered_thresholds: List[MonitoringThreshold],
        status: CircuitBreakerStatus,
    ) -> str:
        """生成摘要"""

        summary_parts = [
            f"【策略状态】: {status.value.upper()}",
            f"【当前持仓】: {len(self.live_positions)} 个",
            f"【历史交易】: {len(self.trade_history)} 笔",
        ]

        # 添加关键指标
        if HealthMetric.WIN_RATE in health_scores:
            wr = health_scores[HealthMetric.WIN_RATE] * 100
            summary_parts.append(f"【胜率】: {wr:.1f}%")

        if HealthMetric.MAE_CONTROL in health_scores:
            mae = health_scores[HealthMetric.MAE_CONTROL]
            summary_parts.append(f"【平均MAE】: {mae:.1f}%")

        if HealthMetric.MFE_UTILIZATION in health_scores:
            mfe_util = health_scores[HealthMetric.MFE_UTILIZATION] * 100
            summary_parts.append(f"【MFE利用率】: {mfe_util:.1f}%")

        # 添加触发信息
        if triggered_thresholds:
            summary_parts.append(f"【触发阈值】: {len(triggered_thresholds)} 个")

        return " | ".join(summary_parts)

    def should_allow_new_position(self) -> Tuple[bool, str]:
        """
        判断是否允许新开仓

        Returns:
            (是否允许, 原因)
        """

        if self.status == CircuitBreakerStatus.TRIPPED:
            return False, "熔断已触发，禁止新开仓"

        if self.status == CircuitBreakerStatus.PAUSED:
            return False, "策略暂停中，禁止新开仓"

        if self.status == CircuitBreakerStatus.RECOVERING:
            return True, "策略恢复中，可谨慎新开仓"

        if self.status == CircuitBreakerStatus.WARNING:
            return True, "策略警告状态，建议降低仓位"

        return True, "策略正常运行，可以新开仓"

    def generate_weekly_report(self) -> Dict[str, Any]:
        """生成周报"""

        # 最近7天的交易
        one_week_ago = datetime.now() - timedelta(days=7)
        recent_trades = [
            t for t in self.trade_history
            if t['exit_date'] >= one_week_ago
        ]

        if not recent_trades:
            return {'message': '过去一周无交易'}

        # 计算统计
        pnls = [t['final_pnl_pct'] for t in recent_trades]
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        avg_pnl = np.mean(pnls)
        total_return = np.prod([1 + p/100 for p in pnls]) - 1

        # MAE/MFE分析
        avg_mae = np.mean([t['mae_pct'] for t in recent_trades])
        avg_mfe = np.mean([t['mfe_pct'] for t in recent_trades])

        # MFE利用率
        profitable_trades = [t for t in recent_trades if t['final_pnl_pct'] > 0 and t['mfe_pct'] > 0]
        mfe_util = np.mean([
            t['final_pnl_pct'] / t['mfe_pct']
            for t in profitable_trades
        ]) if profitable_trades else 0

        return {
            'period': 'weekly',
            'start_date': one_week_ago.strftime('%Y-%m-%d'),
            'end_date': datetime.now().strftime('%Y-%m-%d'),
            'trade_count': len(recent_trades),
            'win_rate': win_rate,
            'avg_pnl_pct': avg_pnl,
            'total_return_pct': total_return * 100,
            'avg_mae_pct': avg_mae,
            'avg_mfe_pct': avg_mfe,
            'mfe_utilization_rate': mfe_util,
            'status': self.status.value,
        }

    def export_monitoring_data(self, filepath: str):
        """导出监控数据到JSON"""
        data = {
            'strategy_id': self.strategy_id,
            'status': self.status.value,
            'positions': [
                {
                    'symbol': p.symbol,
                    'entry_date': p.entry_date.isoformat(),
                    'entry_price': p.entry_price,
                    'current_price': p.current_price,
                    'position_size': p.position_size,
                    'unrealized_pnl_pct': p.unrealized_pnl_pct,
                    'current_mae_pct': p.current_mae_pct,
                    'current_mfe_pct': p.current_mfe_pct,
                    'holding_days': p.holding_days,
                }
                for p in self.live_positions.values()
            ],
            'trade_history': [
                {
                    'symbol': t['symbol'],
                    'entry_date': t['entry_date'].isoformat(),
                    'exit_date': t['exit_date'].isoformat(),
                    'entry_price': t['entry_price'],
                    'exit_price': t['exit_price'],
                    'final_pnl_pct': t['final_pnl_pct'],
                    'mae_pct': t['mae_pct'],
                    'mfe_pct': t['mfe_pct'],
                    'holding_days': t['holding_days'],
                }
                for t in self.trade_history[-100:]  # 最近100笔
            ],
            'monitoring_reports': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'status': r.status.value,
                    'health_scores': {
                        k.value: v for k, v in r.health_scores.items()
                    },
                    'summary': r.summary,
                }
                for r in self.monitoring_history[-10:]  # 最近10条
            ],
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
