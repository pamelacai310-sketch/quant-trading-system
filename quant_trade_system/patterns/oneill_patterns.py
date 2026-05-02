"""
欧奈尔形态识别模块

实现欧奈尔交易体系中的经典价格形态识别，包括：
1. 杯柄形态 (Cup with Handle)
2. 双底/三底 (Double/Triple Bottom)
3. 平底基 (Flat Base)
4. 波动收缩形态 (VCP - Volatility Contraction Pattern)
5. 旗形整理 (Flag)
6. 高标旗 (High Tight Flag)

所有形态识别都基于欧奈尔原著和Minervini等顶级交易员的规则。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class PatternType(Enum):
    """形态类型"""
    CUP_WITH_HANDLE = "cup_with_handle"          # 杯柄形态
    DOUBLE_BOTTOM = "double_bottom"             # 双底
    TRIPLE_BOTTOM = "triple_bottom"             # 三底
    FLAT_BASE = "flat_base"                     # 平底基
    VCP = "vcp"                                # 波动收缩形态
    FLAG = "flag"                              # 旗形
    HIGH_TIGHT_FLAG = "high_tight_flag"        # 高标旗
    ASCENDING_BASE = "ascending_base"          # 上升基底


class PatternQuality(Enum):
    """形态质量"""
    EXCELLENT = "excellent"    # 优秀：所有条件都满足
    GOOD = "good"             # 良好：大部分条件满足
    ACCEPTABLE = "acceptable"  # 可接受：基本条件满足
    POOR = "poor"            # 差：条件不满足


@dataclass
class PatternInfo:
    """形态信息"""
    pattern_type: PatternType
    quality: PatternQuality

    # 形态边界
    start_date: datetime
    end_date: datetime
    pivot_price: float            # 枢轴点价格（突破点）
    stop_loss_price: float         # 止损价格

    # 形态统计
    depth_pct: float              # 形态深度（百分比）
    width_days: int               # 形态宽度（天数）
    symmetry_score: float         # 对称性得分（0-1）

    # 杯柄形态特有
    cup_depth_pct: Optional[float] = None      # 杯底深度
    cup_width_days: Optional[int] = None       # 杯底宽度
    handle_depth_pct: Optional[float] = None   # 柄部深度
    handle_width_days: Optional[int] = None    # 柄部宽度

    # VCP形态特有
    vcp_contractions: Optional[List[float]] = None  # 各次收缩幅度

    # 成交量特征
    volume_contraction: bool = True            # 底部成交量收缩
    required_volume_multiplier: float = 1.5    # 突破所需成交量倍数

    # 附加信息
    description: str = ""
    warnings: List[str] = field(default_factory=list)


class ONeillPatternDetector:
    """
    欧奈尔形态识别器

    识别CANSLIM体系中的各种经典价格形态。
    严格遵循欧奈尔原著和Minervini等顶级交易员的规则。
    """

    def __init__(
        self,
        min_base_days: int = 21,      # 最小基底天数（约1个月）
        max_base_days: int = 252,     # 最大基底天数（约1年）
        min_cup_depth: float = 15.0,  # 最小杯底深度（%）
        max_cup_depth: float = 50.0,  # 最大杯底深度（%）
        min_handle_depth: float = 5.0, # 最小柄部深度（%）
        max_handle_depth: float = 15.0, # 最大柄部深度（%）
        min_rs_rating: int = 70,     # 最小相对强度评级
    ):
        """
        Args:
            min_base_days: 最小基底宽度（天）
            max_base_days: 最大基底宽度（天）
            min_cup_depth: 最小杯底深度（百分比）
            max_cup_depth: 最大杯底深度（百分比）
            min_handle_depth: 最小柄部深度（百分比）
            max_handle_depth: 最大柄部深度（百分比）
            min_rs_rating: 最小相对强度评级（RS Rating）
        """
        self.min_base_days = min_base_days
        self.max_base_days = max_base_days
        self.min_cup_depth = min_cup_depth
        self.max_cup_depth = max_cup_depth
        self.min_handle_depth = min_handle_depth
        self.max_handle_depth = max_handle_depth
        self.min_rs_rating = min_rs_rating

    def detect_all_patterns(
        self,
        data: pd.DataFrame,
        lookback_days: int = 252,
    ) -> List[PatternInfo]:
        """
        检测所有可能的欧奈尔形态

        Args:
            data: 价格数据（必须包含OHLCV）
            lookback_days: 回溯天数

        Returns:
            检测到的形态列表
        """
        patterns = []

        # 检测各种形态
        patterns.extend(self.detect_cup_with_handle(data))
        patterns.extend(self.detect_double_bottom(data))
        patterns.extend(self.detect_triple_bottom(data))
        patterns.extend(self.detect_flat_base(data))
        patterns.extend(self.detect_vcp(data))
        patterns.extend(self.detect_flag(data))
        patterns.extend(self.detect_high_tight_flag(data))

        # 按质量排序
        patterns.sort(key=lambda p: (
            3 if p.quality == PatternQuality.EXCELLENT else
            2 if p.quality == PatternQuality.GOOD else
            1 if p.quality == PatternQuality.ACCEPTABLE else 0
        ), reverse=True)

        return patterns

    def detect_cup_with_handle(
        self,
        data: pd.DataFrame,
        min_cup_days: int = 21,
        max_cup_days: int = 168,
        min_handle_days: int = 5,
        max_handle_days: int = 21,
    ) -> List[PatternInfo]:
        """
        检测杯柄形态

        形态特征：
        1. 前期上涨至少30%
        2. 杯底：U形或碗形调整，深度15-50%，通常持续7-65周
        3. 柄部：杯顶右侧的小幅调整，深度<杯底深度的1/3，持续1-4周
        4. 枢轴点：柄部顶点
        5. 成交量：杯底收缩，柄部收缩，突破放量

        Args:
            data: 价格数据
            min_cup_days: 最小杯底天数
            max_cup_days: 最大杯底天数
            min_handle_days: 最小柄部天数
            max_handle_days: 最大柄部天数

        Returns:
            检测到的杯柄形态列表
        """
        patterns = []

        if len(data) < min_cup_days + min_handle_days:
            return patterns

        # 计算移动平均线
        data = data.copy()
        data['ma50'] = data['close'].rolling(window=50).mean()
        data['ma200'] = data['close'].rolling(window=200).mean()

        # 寻找可能的杯柄形态
        for i in range(max_cup_days + max_handle_days, len(data)):
            # 检查当前是否处于上升趋势（价格在50日和200日均线上方）
            if (
                data['close'].iloc[i] < data['ma50'].iloc[i] or
                data['close'].iloc[i] < data['ma200'].iloc[i]
            ):
                continue

            # 尝试不同的柄部宽度
            for handle_days in range(min_handle_days, max_handle_days + 1):
                handle_start = i - handle_days
                handle_end = i

                # 尝试不同的杯底宽度
                for cup_days in range(min_cup_days, min(max_cup_days, handle_start) + 1):
                    cup_start = handle_start - cup_days
                    cup_end = handle_start

                    if cup_start < self.min_base_days:
                        continue

                    # 检查前期上涨
                    pre_cup_high = data['close'].iloc[cup_start-30:cup_start].max()
                    current_price = data['close'].iloc[i]

                    if current_price / pre_cup_high < 1.3:  # 前期涨幅至少30%
                        continue

                    # 提取杯底阶段
                    cup_data = data.iloc[cup_start:cup_end]
                    cup_high = cup_data['high'].max()
                    cup_low = cup_data['low'].min()
                    cup_depth = (cup_high - cup_low) / cup_high * 100

                    # 检查杯底深度
                    if cup_depth < self.min_cup_depth or cup_depth > self.max_cup_depth:
                        continue

                    # 提取柄部阶段
                    handle_data = data.iloc[handle_start:handle_end]
                    handle_high = handle_data['high'].max()
                    handle_low = handle_data['low'].min()
                    handle_depth = (handle_high - handle_low) / handle_high * 100

                    # 检查柄部深度（应小于杯底深度的1/3）
                    if handle_depth > cup_depth / 3:
                        continue

                    if handle_depth < self.min_handle_depth or handle_depth > self.max_handle_depth:
                        continue

                    # 检查柄部位置（应在杯底上半部分）
                    cup_mid = (cup_high + cup_low) / 2
                    if handle_low < cup_mid:
                        continue

                    # 检查成交量收缩
                    if not self._check_volume_contraction(handle_data):
                        continue

                    # 计算对称性得分
                    symmetry_score = self._calculate_symmetry_score(cup_data)

                    # 计算枢轴点和止损点
                    pivot_price = handle_high
                    stop_loss_price = pivot_price * (1 - 0.08)  # 8%止损

                    # 评估质量
                    quality = self._assess_cup_with_handle_quality(
                        cup_depth, handle_depth, symmetry_score,
                        len(cup_data), len(handle_data)
                    )

                    # 创建形态信息
                    pattern = PatternInfo(
                        pattern_type=PatternType.CUP_WITH_HANDLE,
                        quality=quality,
                        start_date=data.index[cup_start],
                        end_date=data.index[i],
                        pivot_price=pivot_price,
                        stop_loss_price=stop_loss_price,
                        depth_pct=cup_depth,
                        width_days=cup_days + handle_days,
                        symmetry_score=symmetry_score,
                        cup_depth_pct=cup_depth,
                        cup_width_days=cup_days,
                        handle_depth_pct=handle_depth,
                        handle_width_days=handle_days,
                        description=self._generate_cup_with_handle_description(
                            cup_depth, cup_days, handle_depth, handle_days
                        ),
                        warnings=self._generate_pattern_warnings(
                            cup_depth, handle_depth, symmetry_score
                        ),
                    )

                    patterns.append(pattern)

        return patterns

    def detect_double_bottom(
        self,
        data: pd.DataFrame,
        min_base_days: int = 35,
        max_base_days: int = 252,
        depth_tolerance: float = 3.0,
    ) -> List[PatternInfo]:
        """
        检测双底形态（W底）

        形态特征：
        1. 两个明显的低点，价格相近（容差3%以内）
        2. 中间有一个显著的高点（颈线）
        3. 宽度通常7-65周
        4. 突破颈线时成交量放大

        Args:
            data: 价格数据
            min_base_days: 最小基底天数
            max_base_days: 最大基底天数
            depth_tolerance: 两个低点容差（百分比）

        Returns:
            检测到的双底形态列表
        """
        patterns = []

        if len(data) < min_base_days:
            return patterns

        # 寻找局部低点
        local_lows = self._find_local_extrema(data, mode='low', window=10)

        # 寻找双底组合
        for i in range(1, len(local_lows)):
            first_low_idx, first_low_price = local_lows[i-1]
            second_low_idx, second_low_price = local_lows[i]

            # 检查间隔
            days_between = second_low_idx - first_low_idx
            if days_between < 21 or days_between > max_base_days:
                continue

            # 检查价格相近（容差内）
            if abs(first_low_price - second_low_price) / first_low_price * 100 > depth_tolerance:
                continue

            # 检查是否有中间高点（颈线）
            middle_data = data.iloc[first_low_idx:second_low_idx]
            neckline_high = middle_data['high'].max()
            neckline_low = middle_data['low'].min()
            neckline_depth = (neckline_high - neckline_low) / neckline_high * 100

            if neckline_depth < 10:  # 颈线深度至少10%
                continue

            # 检查当前是否突破颈线
            current_price = data['close'].iloc[-1]
            if current_price <= neckline_high:
                continue

            # 检查成交量
            if not self._check_breakout_volume(data, second_low_idx):
                continue

            # 计算形态参数
            total_width = second_low_idx - first_low_idx
            depth_pct = (neckline_high - min(first_low_price, second_low_price)) / neckline_high * 100

            pivot_price = neckline_high
            stop_loss_price = min(first_low_price, second_low_price)

            # 评估质量
            quality = self._assess_double_bottom_quality(
                depth_pct, total_width, neckline_depth
            )

            pattern = PatternInfo(
                pattern_type=PatternType.DOUBLE_BOTTOM,
                quality=quality,
                start_date=data.index[first_low_idx],
                end_date=data.index[-1],
                pivot_price=pivot_price,
                stop_loss_price=stop_loss_price,
                depth_pct=depth_pct,
                width_days=total_width,
                symmetry_score=self._calculate_symmetry_score(data.iloc[first_low_idx:second_low_idx]),
                description=f"双底形态，深度{depth_pct:.1f}%，宽度{total_width}天",
                required_volume_multiplier=1.5,
            )

            patterns.append(pattern)

        return patterns

    def detect_triple_bottom(
        self,
        data: pd.DataFrame,
        min_base_days: int = 42,
        max_base_days: int = 252,
        depth_tolerance: float = 3.0,
    ) -> List[PatternInfo]:
        """
        检测三底形态

        形态特征：
        1. 三个明显的低点，价格相近
        2. 宽度通常比双底更长
        3. 突破颈线时成交量放大
        """
        patterns = []

        if len(data) < min_base_days:
            return patterns

        local_lows = self._find_local_extrema(data, mode='low', window=10)

        # 寻找三底组合
        for i in range(2, len(local_lows)):
            first_low_idx, first_low_price = local_lows[i-2]
            second_low_idx, second_low_price = local_lows[i-1]
            third_low_idx, third_low_price = local_lows[i]

            # 检查总宽度
            total_width = third_low_idx - first_low_idx
            if total_width < min_base_days or total_width > max_base_days:
                continue

            # 检查三个低点价格相近
            low_prices = [first_low_price, second_low_price, third_low_price]
            avg_low = np.mean(low_prices)

            if any(abs(p - avg_low) / avg_low * 100 > depth_tolerance for p in low_prices):
                continue

            # 检查当前是否突破颈线
            all_data = data.iloc[first_low_idx:third_low_idx]
            neckline_high = all_data['high'].max()
            current_price = data['close'].iloc[-1]

            if current_price <= neckline_high:
                continue

            # 计算形态参数
            depth_pct = (neckline_high - avg_low) / neckline_high * 100

            pivot_price = neckline_high
            stop_loss_price = avg_low

            quality = PatternQuality.GOOD  # 三底通常质量较好

            pattern = PatternInfo(
                pattern_type=PatternType.TRIPLE_BOTTOM,
                quality=quality,
                start_date=data.index[first_low_idx],
                end_date=data.index[-1],
                pivot_price=pivot_price,
                stop_loss_price=stop_loss_price,
                depth_pct=depth_pct,
                width_days=total_width,
                symmetry_score=0.7,  # 三底对称性通常较好
                description=f"三底形态，深度{depth_pct:.1f}%，宽度{total_width}天",
                required_volume_multiplier=1.5,
            )

            patterns.append(pattern)

        return patterns

    def detect_flat_base(
        self,
        data: pd.DataFrame,
        min_days: int = 21,
        max_days: int = 84,
        max_range_pct: float = 15.0,
    ) -> List[PatternInfo]:
        """
        检测平底基形态

        形态特征：
        1. 价格在一个窄幅区间横盘整理
        2. 波动幅度通常<15%
        3. 持续时间3-12周
        4. 突破时成交量放大
        """
        patterns = []

        if len(data) < min_days:
            return patterns

        # 寻找横盘区间
        for end_idx in range(min_days, min(len(data), max_days)):
            start_idx = max(0, end_idx - max_days)

            base_data = data.iloc[start_idx:end_idx]
            base_high = base_data['high'].max()
            base_low = base_data['low'].min()

            # 检查波动幅度
            range_pct = (base_high - base_low) / base_high * 100
            if range_pct > max_range_pct:
                continue

            # 检查宽度
            width_days = end_idx - start_idx
            if width_days < min_days:
                continue

            # 检查当前是否突破
            current_price = data['close'].iloc[-1]
            if current_price <= base_high * 1.02:  # 至少突破2%
                continue

            # 检查成交量
            if not self._check_breakout_volume(data, end_idx):
                continue

            pivot_price = base_high
            stop_loss_price = base_low

            quality = PatternQuality.GOOD

            pattern = PatternInfo(
                pattern_type=PatternType.FLAT_BASE,
                quality=quality,
                start_date=data.index[start_idx],
                end_date=data.index[-1],
                pivot_price=pivot_price,
                stop_loss_price=stop_loss_price,
                depth_pct=range_pct,
                width_days=width_days,
                symmetry_score=0.8,
                description=f"平底基形态，波动{range_pct:.1f}%，宽度{width_days}天",
                required_volume_multiplier=1.5,
            )

            patterns.append(pattern)

        return patterns

    def detect_vcp(
        self,
        data: pd.DataFrame,
        min_contractions: int = 3,
        max_contractions: int = 4,
        min_contraction_pct: float = 10.0,
        max_contraction_pct: float = 30.0,
        final_contraction_max: float = 10.0,
    ) -> List[PatternInfo]:
        """
        检测波动收缩形态（VCP）

        形态特征（Minervini SEPA框架）：
        1. 经历3-4次连续的收缩回调
        2. 每次收缩幅度依次递减（如18% -> 12% -> 6%）
        3. 最后一次收缩幅度通常<10%
        4. 表明机构资金持续吸筹，卖压逐渐耗尽
        5. 突破时成交量放大（1.4-1.5倍）

        Args:
            data: 价格数据
            min_contractions: 最小收缩次数
            max_contractions: 最大收缩次数
            min_contraction_pct: 最小收缩幅度（%）
            max_contraction_pct: 最大收缩幅度（%）
            final_contraction_max: 最后一次收缩最大幅度（%）

        Returns:
            检测到的VCP形态列表
        """
        patterns = []

        if len(data) < min_contractions * 21:  # 每次收缩至少3周
            return patterns

        # 寻找局部高点和低点
        peaks = self._find_local_extrema(data, mode='high', window=10)
        troughs = self._find_local_extrema(data, mode='low', window=10)

        # 尝试构建VCP形态
        for i in range(min_contractions, min(max_contractions + 1, len(peaks))):
            # 取最后i个峰和谷
            recent_peaks = peaks[-i:]
            recent_troughs = troughs[-i:]  # 可能比峰少1个

            if len(recent_troughs) < i:
                continue

            # 计算每次收缩幅度
            contractions = []
            for j in range(i):
                peak_idx, peak_price = recent_peaks[j]
                trough_idx, trough_price = recent_troughs[j]

                # 确保谷在峰之后
                if trough_idx <= peak_idx:
                    break

                contraction_pct = (peak_price - trough_price) / peak_price * 100
                contractions.append(contraction_pct)

            if len(contractions) < min_contractions:
                continue

            # 检查收缩是否依次递减
            is_decreasing = all(
                contractions[j] > contractions[j+1]
                for j in range(len(contractions) - 1)
            )

            if not is_decreasing:
                continue

            # 检查收缩幅度范围
            if any(c < min_contraction_pct or c > max_contraction_pct for c in contractions):
                continue

            # 检查最后一次收缩幅度
            if contractions[-1] > final_contraction_max:
                continue

            # 检查当前是否突破最后一个峰
            last_peak_idx, last_peak_price = recent_peaks[-1]
            current_price = data['close'].iloc[-1]

            if current_price <= last_peak_price:
                continue

            # 检查成交量（需要1.4-1.5倍）
            if not self._check_breakout_volume(data, last_peak_idx, min_multiplier=1.4):
                continue

            # 计算形态参数
            total_width = recent_peaks[-1][0] - recent_peaks[0][0]
            total_depth = sum(contractions)

            pivot_price = last_peak_price
            stop_loss_price = recent_troughs[-1][1]

            # VCP通常质量很高
            quality = PatternQuality.EXCELLENT

            pattern = PatternInfo(
                pattern_type=PatternType.VCP,
                quality=quality,
                start_date=data.index[recent_peaks[0][0]],
                end_date=data.index[-1],
                pivot_price=pivot_price,
                stop_loss_price=stop_loss_price,
                depth_pct=total_depth,
                width_days=total_width,
                symmetry_score=0.9,  # VCP收缩性好，对称性高
                vcp_contractions=contractions,
                description=f"VCP形态，{len(contractions)}次收缩：{[f'{c:.1f}%' for c in contractions]}",
                required_volume_multiplier=1.5,
            )

            patterns.append(pattern)

        return patterns

    def detect_flag(
        self,
        data: pd.DataFrame,
        min_days: int = 7,
        max_days: int = 21,
        max_consolidation_pct: float = 10.0,
    ) -> List[PatternInfo]:
        """
        检测旗形整理形态

        形态特征：
        1. 前期强劲上涨（通常>50%）
        2. 短期横盘或小幅回调整理
        3. 整理幅度<10%
        4. 持续1-3周
        5. 突破时继续上涨
        """
        patterns = []

        if len(data) < min_days:
            return patterns

        # 寻找前期上涨
        for end_idx in range(min_days, min(len(data), max_days)):
            start_idx = end_idx - min_days

            # 检查前期涨幅
            pre_flag_high = data['close'].iloc[start_idx-30:start_idx].max()
            flag_start_price = data['close'].iloc[start_idx]

            if flag_start_price / pre_flag_high < 1.5:  # 前期涨幅至少50%
                continue

            # 检查旗形整理
            flag_data = data.iloc[start_idx:end_idx]
            flag_high = flag_data['high'].max()
            flag_low = flag_data['low'].min()

            consolidation_pct = (flag_high - flag_low) / flag_high * 100
            if consolidation_pct > max_consolidation_pct:
                continue

            # 检查当前是否突破
            current_price = data['close'].iloc[-1]
            if current_price <= flag_high:
                continue

            pivot_price = flag_high
            stop_loss_price = flag_low

            quality = PatternQuality.GOOD

            pattern = PatternInfo(
                pattern_type=PatternType.FLAG,
                quality=quality,
                start_date=data.index[start_idx],
                end_date=data.index[-1],
                pivot_price=pivot_price,
                stop_loss_price=stop_loss_price,
                depth_pct=consolidation_pct,
                width_days=end_idx - start_idx,
                symmetry_score=0.7,
                description=f"旗形形态，整理{consolidation_pct:.1f}%，宽度{end_idx - start_idx}天",
                required_volume_multiplier=1.5,
            )

            patterns.append(pattern)

        return patterns

    def detect_high_tight_flag(
        self,
        data: pd.DataFrame,
        min_prior_gain_pct: float = 100.0,  # 前期涨幅至少100%
        consolidation_days: int = 5,          # 整理5天左右
    ) -> List[PatternInfo]:
        """
        检测高标旗形态（High Tight Flag）

        形态特征：
        1. 前期爆发式上涨（通常>100%）
        2. 在高位形成极窄幅的横盘整理
        3. 整理幅度很小（通常<10%）
        4. 持续时间短（通常3-5周）
        5. 突破后可能继续爆发式上涨
        """
        patterns = []

        if len(data) < consolidation_days + 30:
            return patterns

        # 寻找高标旗
        for i in range(consolidation_days + 30, len(data)):
            # 检查前期涨幅
            pre_high = data['close'].iloc[i-30:i-20].max()
            consolidation_start = i - consolidation_days

            # 检查是否达到前期涨幅要求
            if data['close'].iloc[consolidation_start] / pre_high < (1 + min_prior_gain_pct / 100):
                continue

            # 检查整理阶段
            consolidation_data = data.iloc[consolidation_start:i]
            consolidation_high = consolidation_data['high'].max()
            consolidation_low = consolidation_data['low'].min()

            consolidation_pct = (consolidation_high - consolidation_low) / consolidation_high * 100

            # 检查当前是否突破
            current_price = data['close'].iloc[i]
            if current_price <= consolidation_high:
                continue

            pivot_price = consolidation_high
            stop_loss_price = consolidation_low

            # 高标旗通常质量很高
            quality = PatternQuality.EXCELLENT

            pattern = PatternInfo(
                pattern_type=PatternType.HIGH_TIGHT_FLAG,
                quality=quality,
                start_date=data.index[i-30],
                end_date=data.index[i],
                pivot_price=pivot_price,
                stop_loss_price=stop_loss_price,
                depth_pct=consolidation_pct,
                width_days=consolidation_days,
                symmetry_score=0.8,
                description=f"高标旗形态，前期涨幅{min_prior_gain_pct:.0f}%+，整理{consolidation_pct:.1f}%",
                required_volume_multiplier=1.5,
            )

            patterns.append(pattern)

        return patterns

    # ==================== 辅助方法 ====================

    def _find_local_extrema(
        self,
        data: pd.DataFrame,
        mode: str = 'high',
        window: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        查找局部极值点

        Args:
            data: 价格数据
            mode: 'high' 或 'low'
            window: 窗口大小

        Returns:
            [(索引, 价格)] 列表
        """
        extrema = []

        for i in range(window, len(data) - window):
            if mode == 'high':
                current = data['high'].iloc[i]
                before_max = data['high'].iloc[i-window:i].max()
                after_max = data['high'].iloc[i+1:i+window+1].max()

                if current >= before_max and current >= after_max:
                    extrema.append((i, current))

            else:  # mode == 'low'
                current = data['low'].iloc[i]
                before_min = data['low'].iloc[i-window:i].min()
                after_min = data['low'].iloc[i+1:i+window+1].min()

                if current <= before_min and current <= after_min:
                    extrema.append((i, current))

        return extrema

    def _check_volume_contraction(
        self,
        data: pd.DataFrame,
        window: int = 10,
        threshold: float = 0.8,
    ) -> bool:
        """
        检查成交量是否收缩

        Args:
            data: 价格数据
            window: 比较窗口
            threshold: 收缩阈值（与前期均值相比）

        Returns:
            True if volume contracted
        """
        if 'volume' not in data.columns:
            return True  # 如果没有成交量数据，默认通过

        recent_vol = data['volume'].iloc[-window:].mean()
        prior_vol = data['volume'].iloc[-window*2:-window].mean()

        if prior_vol == 0:
            return True

        return recent_vol < prior_vol * threshold

    def _check_breakout_volume(
        self,
        data: pd.DataFrame,
        breakout_idx: int,
        lookback_days: int = 50,
        min_multiplier: float = 1.5,
    ) -> bool:
        """
        检查突破成交量是否放大

        Args:
            data: 价格数据
            breakout_idx: 突破点索引
            lookback_days: 回溯天数
            min_multiplier: 最小成交量倍数

        Returns:
            True if volume expanded sufficiently
        """
        if 'volume' not in data.columns or breakout_idx >= len(data):
            return True  # 如果没有成交量数据，默认通过

        breakout_vol = data['volume'].iloc[breakout_idx]
        avg_vol = data['volume'].iloc[max(0, breakout_idx-lookback_days):breakout_idx].mean()

        if avg_vol == 0:
            return True

        return breakout_vol >= avg_vol * min_multiplier

    def _calculate_symmetry_score(
        self,
        data: pd.DataFrame,
    ) -> float:
        """
        计算形态对称性得分（0-1）

        方法：比较前后半程的深度和时间
        """
        if len(data) < 10:
            return 0.5

        mid = len(data) // 2
        first_half = data.iloc[:mid]
        second_half = data.iloc[mid:]

        # 价格范围
        first_range = first_half['high'].max() - first_half['low'].min()
        second_range = second_half['high'].max() - second_half['low'].min()

        if first_range == 0 or second_range == 0:
            return 0.5

        range_ratio = min(first_range, second_range) / max(first_range, second_range)

        # 时间长度
        time_ratio = min(mid, len(data) - mid) / max(mid, len(data) - mid)

        return (range_ratio + time_ratio) / 2

    def _assess_cup_with_handle_quality(
        self,
        cup_depth: float,
        handle_depth: float,
        symmetry_score: float,
        cup_days: int,
        handle_days: int,
    ) -> PatternQuality:
        """评估杯柄形态质量"""

        score = 0

        # 杯底深度：15-30%为理想
        if 15 <= cup_depth <= 30:
            score += 3
        elif 30 < cup_depth <= 40:
            score += 2
        elif cup_depth <= 15 or cup_depth > 40:
            score += 1

        # 柄部深度：5-10%为理想
        if 5 <= handle_depth <= 10:
            score += 3
        elif 10 < handle_depth <= 15:
            score += 2
        else:
            score += 1

        # 对称性
        if symmetry_score >= 0.8:
            score += 3
        elif symmetry_score >= 0.6:
            score += 2
        else:
            score += 1

        # 时间比例
        if 7 <= handle_days <= 21 and 21 <= cup_days <= 168:
            score += 1

        # 评级
        if score >= 10:
            return PatternQuality.EXCELLENT
        elif score >= 7:
            return PatternQuality.GOOD
        elif score >= 5:
            return PatternQuality.ACCEPTABLE
        else:
            return PatternQuality.POOR

    def _assess_double_bottom_quality(
        self,
        depth_pct: float,
        width_days: int,
        neckline_depth: float,
    ) -> PatternQuality:
        """评估双底形态质量"""

        score = 0

        # 深度：15-40%为理想
        if 15 <= depth_pct <= 40:
            score += 3
        elif depth_pct > 40:
            score += 2
        else:
            score += 1

        # 宽度：35-126天为理想
        if 35 <= width_days <= 126:
            score += 3
        elif width_days > 126:
            score += 2
        else:
            score += 1

        # 颈线深度
        if neckline_depth >= 15:
            score += 3
        elif neckline_depth >= 10:
            score += 2
        else:
            score += 1

        if score >= 7:
            return PatternQuality.EXCELLENT
        elif score >= 5:
            return PatternQuality.GOOD
        elif score >= 3:
            return PatternQuality.ACCEPTABLE
        else:
            return PatternQuality.POOR

    def _generate_cup_with_handle_description(
        self,
        cup_depth: float,
        cup_days: int,
        handle_depth: float,
        handle_days: int,
    ) -> str:
        """生成杯柄形态描述"""
        return (
            f"杯柄形态：杯底深度{cup_depth:.1f}%（{cup_days}天），"
            f"柄部深度{handle_depth:.1f}%（{handle_days}天）"
        )

    def _generate_pattern_warnings(
        self,
        cup_depth: float,
        handle_depth: float,
        symmetry_score: float,
    ) -> List[str]:
        """生成形态警告"""
        warnings = []

        if cup_depth > 40:
            warnings.append(f"杯底过深（{cup_depth:.1f}%），可能回调过度")

        if handle_depth > cup_depth / 2:
            warnings.append(f"柄部过深（{handle_depth:.1f}%），超过杯底深度的一半")

        if symmetry_score < 0.5:
            warnings.append("形态对称性较差，可能不够理想")

        return warnings
