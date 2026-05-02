"""
极端交易自动复盘工具

功能：
1. 自动提取极端交易（大盈/大亏）
2. 生成K线图和买卖点标记
3. 归因分析
4. 识别共性模式
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


@dataclass
class ExtremeTrade:
    """极端交易记录"""

    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    final_pnl_pct: float
    mae_pct: float
    mfe_pct: float
    position_size: float
    trade_type: str

    # 分类
    extreme_type: str  # 'huge_profit', 'huge_loss', 'deep_dive_profit', 'roller_coaster'

    # 归因
    entry_reasoning: str = ""
    exit_reasoning: str = ""
    market_context: str = ""
    lessons: str = ""


class ExtremeTradeAnalyzer:
    """
    极端交易分析器

    自动提取并分析极端交易，生成可操作的洞察
    """

    def __init__(
        self,
        extreme_threshold: float = 10.0,  # 盈亏超过10%视为极端
        mae_threshold: float = -10.0,      # MAE超过-10%视为深套
        mfe_threshold: float = 15.0,       # MFE超过15%视为大浮盈
    ):
        """
        Args:
            extreme_threshold: 极端盈亏阈值（百分比）
            mae_threshold: 深套阈值（百分比）
            mfe_threshold: 大浮盈阈值（百分比）
        """
        self.extreme_threshold = extreme_threshold
        self.mae_threshold = mae_threshold
        self.mfe_threshold = mfe_threshold
        self.extreme_trades: List[ExtremeTrade] = []

    def extract_extreme_trades(
        self,
        backtest_result: Any,  # BacktestResult or similar
        price_data: pd.DataFrame,
    ) -> List[ExtremeTrade]:
        """
        从回测结果中提取极端交易

        Args:
            backtest_result: 回测结果
            price_data: 价格数据

        Returns:
            极端交易列表
        """

        trades = backtest_result.trades if hasattr(backtest_result, 'trades') else []

        for trade in trades:
            extreme_trade = self._classify_extreme_trade(trade, price_data)
            if extreme_trade:
                self.extreme_trades.append(extreme_trade)

        # 按最终盈亏排序
        self.extreme_trades.sort(key=lambda x: x.final_pnl_pct, reverse=True)

        return self.extreme_trades

    def _classify_extreme_trade(
        self,
        trade: Dict,
        price_data: pd.DataFrame,
    ) -> Optional[ExtremeTrade]:
        """判断并分类极端交易"""

        try:
            entry_date = pd.to_datetime(trade['entry_date'])
            exit_date = pd.to_datetime(trade['exit_date'])
            entry_price = float(trade['entry_price'])
            exit_price = float(trade['exit_price'])

            # 计算 MAE/MFE
            period_data = price_data[
                (price_data.index >= entry_date) &
                (price_data.index <= exit_date)
            ]

            if len(period_data) < 2:
                return None

            # 计算MAE/MFE
            mae_price = period_data['low'].min()
            mfe_price = period_data['high'].max()
            mae_pct = (mae_price / entry_price - 1) * 100
            mfe_pct = (mfe_price / entry_price - 1) * 100

            final_pnl_pct = (exit_price / entry_price - 1) * 100

            # 判断是否为极端交易
            extreme_type = None

            # 大盈利交易
            if final_pnl_pct >= self.extreme_threshold:
                if mae_pct < self.mae_threshold:
                    extreme_type = 'deep_dive_profit'  # 深套后大盈
                else:
                    extreme_type = 'huge_profit'  # 顺利大盈

            # 大亏损交易
            elif final_pnl_pct <= -self.extreme_threshold:
                extreme_type = 'huge_loss'

            # 过山车交易
            elif mfe_pct >= self.mfe_threshold and final_pnl_pct < 3:
                extreme_type = 'roller_coaster'  # 大浮盈变小盈

            # 深套交易
            elif mae_pct < self.mae_threshold:
                if final_pnl_pct > 0:
                    extreme_type = 'deep_dive_profit'  # 深套后回本

            if extreme_type:
                return ExtremeTrade(
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    final_pnl_pct=final_pnl_pct,
                    mae_pct=mae_pct,
                    mfe_pct=mfe_pct,
                    position_size=float(trade.get('quantity', 1.0)),
                    trade_type=trade.get('side', 'long'),
                    extreme_type=extreme_type,
                )

        except Exception as e:
            print(f"分析交易时出错: {e}")

        return None

    def analyze_patterns(self) -> Dict[str, Any]:
        """
        分析极端交易的共性模式

        Returns:
            模式分析结果
        """

        if not self.extreme_trades:
            return {'warning': '没有极端交易记录'}

        # 按类型分组
        type_groups: Dict[str, List[ExtremeTrade]] = {
            'huge_profit': [],
            'huge_loss': [],
            'deep_dive_profit': [],
            'roller_coaster': [],
        }

        for trade in self.extreme_trades:
            type_groups[trade.extreme_type].append(trade)

        # 分析每种类型的模式
        patterns = {}

        for trade_type, trades in type_groups.items():
            if not trades:
                continue

            # 统计特征
            avg_holding_days = np.mean([
                (t.exit_date - t.entry_date).days for t in trades
            ])
            avg_mae = np.mean([t.mae_pct for t in trades])
            avg_mfe = np.mean([t.mfe_pct for t in trades])

            # 周几分析
            weekdays = [t.entry_date.weekday() for t in trades]
            weekday_counts = pd.Series(weekdays).value_counts()

            # 月份分析
            months = [t.entry_date.month for t in trades]
            month_counts = pd.Series(months).value_counts()

            patterns[trade_type] = {
                'count': len(trades),
                'avg_holding_days': avg_holding_days,
                'avg_mae_pct': avg_mae,
                'avg_mfe_pct': avg_mfe,
                'entry_weekday_distribution': weekday_counts.to_dict(),
                'entry_month_distribution': month_counts.to_dict(),
                'examples': trades[:3],  # 前3个例子
            }

        return {
            'total_extreme_trades': len(self.extreme_trades),
            'patterns': patterns,
        }

    def plot_extreme_trade(
        self,
        trade: ExtremeTrade,
        price_data: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        """
        绘制极端交易的K线图

        Args:
            trade: 极端交易
            price_data: 价格数据
            save_path: 保存路径
            figsize: 图表大小

        Returns:
            matplotlib Figure 对象
        """

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 获取交易前后各10天的数据
        start_date = trade.entry_date - timedelta(days=10)
        end_date = trade.exit_date + timedelta(days=10)

        plot_data = price_data[
            (price_data.index >= start_date) &
            (price_data.index <= end_date)
        ].copy()

        if len(plot_data) == 0:
            print(f"无法获取交易期间的价格数据: {trade.entry_date} - {trade.exit_date}")
            return None

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

        # 绘制K线图（简化版）
        colors = ['red' if close >= open_ else 'green' for open_, close in zip(plot_data['open'], plot_data['close'])]

        ax1.bar(plot_data.index, plot_data['close'] - plot_data['open'], bottom=plot_data['open'],
               color=colors, alpha=0.6, width=0.8)
        ax1.vlines(plot_data.index, plot_data['low'], plot_data['high'], color=colors, linewidth=1)

        # 标记买卖点
        entry_idx = plot_data.index.get_indexer([trade.entry_date], method='nearest')[0]
        exit_idx = plot_data.index.get_indexer([trade.exit_date], method='nearest')[0]

        if entry_idx >= 0:
            ax1.scatter(plot_data.index[entry_idx], trade.entry_price,
                       color='blue', s=200, marker='^', zorder=5, label='买入', edgecolors='black', linewidth=2)

        if exit_idx >= 0:
            ax1.scatter(plot_data.index[exit_idx], trade.exit_price,
                       color='orange', s=200, marker='v', zorder=5, label='卖出', edgecolors='black', linewidth=2)

        # 标记MAE和MFE点
        period_mask = (
            (plot_data.index >= trade.entry_date) &
            (plot_data.index <= trade.exit_date)
        )
        period_data = plot_data[period_mask]

        if len(period_data) > 0:
            # MAE点（最低价）
            mae_date = period_data['low'].idxmin()
            mae_price = period_data['low'].min()
            ax1.scatter(mae_date, mae_price, color='red', s=150, marker='x',
                       zorder=5, label=f'MAE ({trade.mae_pct:.1f}%)', linewidth=3)

            # MFE点（最高价）
            mfe_date = period_data['high'].idxmax()
            mfe_price = period_data['high'].max()
            ax1.scatter(mfe_date, mfe_price, color='green', s=150, marker='*',
                       zorder=5, label=f'MFE ({trade.mfe_pct:.1f}%)', linewidth=3)

        ax1.set_ylabel('价格', fontsize=11)
        ax1.set_title(
            f'极端交易分析: {trade.extreme_type}\n'
            f'买入: {trade.entry_date.strftime("%Y-%m-%d")} @ {trade.entry_price:.2f} | '
            f'卖出: {trade.exit_date.strftime("%Y-%m-%d")} @ {trade.exit_price:.2f}\n'
            f'MAE: {trade.mae_pct:.1f}% | MFE: {trade.mfe_pct:.1f}% | 最终盈亏: {trade.final_pnl_pct:.1f}%',
            fontsize=12, fontweight='bold'
        )
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 绘制成交量
        ax2.bar(plot_data.index, plot_data['volume'], color='steelblue', alpha=0.6, width=0.8)
        ax2.axvline(x=trade.entry_date, color='blue', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axvline(x=trade.exit_date, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_ylabel('成交量', fontsize=11)
        ax2.set_xlabel('日期', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"K线图已保存至: {save_path}")

        return fig

    def plot_all_extreme_trades(
        self,
        price_data: pd.DataFrame,
        save_dir: Optional[str] = None,
    ) -> List[str]:
        """
        绘制所有极端交易的K线图

        Args:
            price_data: 价格数据
            save_dir: 保存目录

        Returns:
            保存的文件路径列表
        """

        import os
        saved_paths = []

        for i, trade in enumerate(self.extreme_trades):
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                filename = f"extreme_trade_{i+1}_{trade.extreme_type}_{trade.entry_date.strftime('%Y%m%d')}.png"
                save_path = os.path.join(save_dir, filename)
            else:
                save_path = None

            fig = self.plot_extreme_trade(trade, price_data, save_path)
            if fig:
                plt.close(fig)
                if save_path:
                    saved_paths.append(save_path)

        return saved_paths

    def generate_insights_report(self) -> str:
        """生成洞察报告"""

        if not self.extreme_trades:
            return "没有极端交易记录"

        patterns = self.analyze_patterns()
        report = []
        report.append("=" * 80)
        report.append(" " * 20 + "极端交易分析报告")
        report.append("=" * 80)
        report.append(f"\n总极端交易数: {len(self.extreme_trades)}")

        # 按类型统计
        type_counts = {}
        for trade in self.extreme_trades:
            type_counts[trade.extreme_type] = type_counts.get(trade.extreme_type, 0) + 1

        report.append("\n📊 极端交易类型分布:")
        for trade_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            report.append(f"   {trade_type}: {count} 笔")

        # 详细分析每种类型
        if 'patterns' in patterns:
            for trade_type, pattern in patterns['patterns'].items():
                report.append(f"\n🔍 {trade_type.upper()} 分析:")
                report.append(f"   数量: {pattern['count']} 笔")
                report.append(f"   平均持仓天数: {pattern['avg_holding_days']:.1f} 天")
                report.append(f"   平均 MAE: {pattern['avg_mae_pct']:.2f}%")
                report.append(f"   平均 MFE: {pattern['avg_mfe_pct']:.2f}%")

                # 入场日期分析
                if pattern['entry_weekday_distribution']:
                    report.append(f"   入场日期分布（周几）:")
                    for day, count in sorted(pattern['entry_weekday_distribution'].items()):
                        weekdays = ['一', '二', '三', '四', '五', '六', '日']
                        report.append(f"      周{weekdays[day]}: {count} 笔")

                # 典型案例
                if pattern['examples']:
                    report.append(f"\n   典型案例:")
                    for i, example in enumerate(pattern['examples'][:3], 1):
                        report.append(
                            f"      {i}. {example.entry_date.strftime('%Y-%m-%d')} -> "
                            f"{example.exit_date.strftime('%Y-%m-%d')}, "
                            f"盈亏: {example.final_pnl_pct:.1f}%, "
                            f"MAE: {example.mae_pct:.1f}%, "
                            f"MFE: {example.mfe_pct:.1f}%"
                        )

        # 生成建议
        report.append("\n💡 优化建议:")

        # 大盈利交易分析
        huge_profit_trades = [t for t in self.extreme_trades if t.extreme_type == 'huge_profit']
        if huge_profit_trades:
            report.append("\n✅ 大盈利交易特征（可复制）:")
            common_features = self._extract_common_features(huge_profit_trades)
            for feature in common_features:
                report.append(f"   - {feature}")

        # 大亏损交易分析
        huge_loss_trades = [t for t in self.extreme_trades if t.extreme_type == 'huge_loss']
        if huge_loss_trades:
            report.append("\n⚠️ 大亏损交易特征（需避免）:")
            common_features = self._extract_common_features(huge_loss_trades)
            for feature in common_features:
                report.append(f"   - {feature}")

        # 过山车交易分析
        roller_coaster_trades = [t for t in self.extreme_trades if t.extreme_type == 'roller_coaster']
        if roller_coaster_trades:
            report.append("\n⚠️ 过山车交易分析（需优化止盈）:")
            avg_mfe = np.mean([t.mfe_pct for t in roller_coaster_trades])
            avg_pnl = np.mean([t.final_pnl_pct for t in roller_coaster_trades])
            report.append(f"   - 平均MFE达到{avg_mfe:.1f}%，但最终仅盈利{avg_pnl:.1f}%")
            report.append(f"   - 建议: 引入动态跟踪止盈，避免利润大幅回撤")

        # 深套后回本交易分析
        deep_dive_trades = [t for t in self.extreme_trades if t.extreme_type == 'deep_dive_profit']
        if deep_dive_trades:
            report.append("\n⚠️ 深套后回本交易分析（需优化买点）:")
            avg_mae = np.mean([t.mae_pct for t in deep_dive_trades])
            avg_holding = np.mean([(t.exit_date - t.entry_date).days for t in deep_dive_trades])
            report.append(f"   - 平均MAE达到{avg_mae:.1f}%，平均持仓{avg_holding:.0f}天")
            report.append(f"   - 建议: 增加右侧确认信号，避免半山腰抄底")

        report.append("\n" + "=" * 80 + "\n")

        return "\n".join(report)

    def _extract_common_features(self, trades: List[ExtremeTrade]) -> List[str]:
        """提取交易的共性特征"""

        features = []

        # 时间特征
        weekdays = [t.entry_date.weekday() for t in trades]
        weekday_counts = pd.Series(weekdays).value_counts()
        most_common_weekday = weekday_counts.index[0]
        weekdays = ['一', '二', '三', '四', '五', '六', '日']
        features.append(f"最常在周{weekdays[most_common_weekday]}入场")

        # 持仓天数
        holding_days = [(t.exit_date - t.entry_date).days for t in trades]
        avg_holding = np.mean(holding_days)
        features.append(f"平均持仓{avg_holding:.1f}天")

        # MAE/MFE特征
        avg_mae = np.mean([t.mae_pct for t in trades])
        avg_mfe = np.mean([t.mfe_pct for t in trades])
        features.append(f"平均MAE: {avg_mae:.1f}%, 平均MFE: {avg_mfe:.1f}%")

        return features


def analyze_extreme_trades_auto(
    backtest_result: Any,
    price_data: pd.DataFrame,
    save_plots: bool = True,
    save_dir: Optional[str] = None,
) -> Tuple[List[ExtremeTrade], str]:
    """
    自动分析极端交易

    Args:
        backtest_result: 回测结果
        price_data: 价格数据
        save_plots: 是否保存K线图
        save_dir: 保存目录

    Returns:
        (极端交易列表, 洞察报告)
    """

    analyzer = ExtremeTradeAnalyzer(
        extreme_threshold=10.0,
        mae_threshold=-10.0,
        mfe_threshold=15.0,
    )

    # 提取极端交易
    extreme_trades = analyzer.extract_extreme_trades(backtest_result, price_data)

    # 生成K线图
    if save_plots and extreme_trades:
        analyzer.plot_all_extreme_trades(price_data, save_dir)

    # 生成洞察报告
    report = analyzer.generate_insights_report()

    return extreme_trades, report
