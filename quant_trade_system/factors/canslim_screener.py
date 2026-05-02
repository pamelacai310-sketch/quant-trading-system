"""
CANSLIM 选股系统

实现欧奈尔CANSLIM七要素选股框架：
C - Current Earnings (当季收益增长)
A - Annual Earnings (年化收益增长)
N - New (新产品、新管理层、新高)
S - Supply and Demand (供需关系)
L - Leader (市场领军股票)
I - Institutional Sponsorship (机构资金)
M - Market Direction (大盘趋势)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CANSLIM_Score:
    """CANSLIM评分结果"""
    total_score: float              # 总分（0-100）
    c_score: float                  # 当季收益得分
    a_score: float                  # 年化收益得分
    n_score: float                  # 新产品得分
    s_score: float                  # 供需关系得分
    l_score: float                  # 领涨股得分
    i_score: float                  # 机构资金得分
    m_score: float                  # 大盘趋势得分

    # 详细信息
    eps_growth_current: float       # 当季EPS增长率
    eps_growth_annual: float        # 年化EPS增长率
    rs_rating: float                # 相对强度评级（0-100）
    institutional_ownership: float   # 机构持股比例
    market_trend: str               # 大盘趋势（Uptrend/Downtrend/Sideways）

    # 通过/未通过的项目
    passed_criteria: List[str] = field(default_factory=list)
    failed_criteria: List[str] = field(default_factory=list)

    # 建议
    recommendation: str = ""         # 总体建议
    warnings: List[str] = field(default_factory=list)


class CANSLIM_Screener:
    """
    CANSLIM选股器

    根据欧奈尔CANSLIM框架筛选股票。
    每个要素都有明确的量化标准和权重。
    """

    def __init__(
        self,
        # C - 当季收益阈值
        min_eps_growth_current: float = 20.0,      # 最小当季EPS增长率（%）
        # A - 年化收益阈值
        min_eps_growth_annual: float = 25.0,       # 最小年化EPS增长率（%）
        min_annual_growth_years: int = 3,          # 最小连续增长年数
        # N - 新产品/新高
        min_weeks_high_52w: int = 4,               # 52周新高最小周数
        # L - 领涨股
        min_rs_rating: float = 70.0,               # 最小相对强度评级
        # I - 机构资金
        min_institutional_ownership: float = 5.0,  # 最小机构持股比例（%）
        max_institutional_ownership: float = 95.0, # 最大机构持股比例（%）
        # M - 大盘趋势
        market_trend_lookback: int = 50,           # 大盘趋势回溯天数
    ):
        """
        Args:
            min_eps_growth_current: 最小当季EPS增长率（%），欧奈尔建议至少20-25%
            min_eps_growth_annual: 最小年化EPS增长率（%），欧奈尔建议至少25-30%
            min_annual_growth_years: 最小连续增长年数，欧奈尔建议至少3年
            min_weeks_high_52w: 距离52周高点的最小周数
            min_rs_rating: 最小相对强度评级（0-100），欧奈尔建议至少70-80
            min_institutional_ownership: 最小机构持股比例（%），需要有机构支持
            max_institutional_ownership: 最大机构持股比例（%），避免过度持有
            market_trend_lookback: 大盘趋势判断回溯天数
        """
        self.min_eps_growth_current = min_eps_growth_current
        self.min_eps_growth_annual = min_eps_growth_annual
        self.min_annual_growth_years = min_annual_growth_years
        self.min_weeks_high_52w = min_weeks_high_52w
        self.min_rs_rating = min_rs_rating
        self.min_institutional_ownership = min_institutional_ownership
        self.max_institutional_ownership = max_institutional_ownership
        self.market_trend_lookback = market_trend_lookback

    def screen_stock(
        self,
        stock_data: pd.DataFrame,
        fundamental_data: Dict[str, Any],
        market_index_data: pd.DataFrame,
        symbol: str = "",
    ) -> CANSLIM_Score:
        """
        对单只股票进行CANSLIM筛选

        Args:
            stock_data: 股票价格数据（OHLCV）
            fundamental_data: 基本面数据字典，包含：
                - eps: 当季EPS
                - eps_last_quarter: 上一季度EPS
                - annual_eps: 过去4个季度EPS（列表）
                - historical_annual_eps: 过去几年年化EPS（列表）
                - rs_rating: 相对强度评级（0-100）
                - institutional_ownership: 机构持股比例（%）
                - institutional_count: 机构数量
                - shares_outstanding: 流通股数
                - volume_avg: 平均成交量
                - new_products: 新产品/创新相关新闻（可选）
                - all_time_high: 是否创历史新高（可选）
                - week_high_52w: 距离52周高点的周数（可选）
            market_index_data: 大盘指数数据（如标普500、沪深300）
            symbol: 股票代码

        Returns:
            CANSLIM评分结果
        """
        scores = {}
        passed = []
        failed = []
        warnings = []

        # C - Current Earnings (当季收益)
        c_score, c_passed, c_warnings = self._score_c(fundamental_data)
        scores['c_score'] = c_score
        if c_passed:
            passed.append("C: 当季收益增长")
        else:
            failed.append("C: 当季收益增长")
        warnings.extend(c_warnings)

        # A - Annual Earnings (年化收益)
        a_score, a_passed, a_warnings = self._score_a(fundamental_data)
        scores['a_score'] = a_score
        if a_passed:
            passed.append("A: 年化收益增长")
        else:
            failed.append("A: 年化收益增长")
        warnings.extend(a_warnings)

        # N - New (新产品、新高)
        n_score, n_passed, n_warnings = self._score_n(stock_data, fundamental_data)
        scores['n_score'] = n_score
        if n_passed:
            passed.append("N: 新产品/新高")
        else:
            failed.append("N: 新产品/新高")
        warnings.extend(n_warnings)

        # S - Supply and Demand (供需关系)
        s_score, s_passed, s_warnings = self._score_s(fundamental_data, stock_data)
        scores['s_score'] = s_score
        if s_passed:
            passed.append("S: 供需关系")
        else:
            failed.append("S: 供需关系")
        warnings.extend(s_warnings)

        # L - Leader (领涨股)
        l_score, l_passed, l_warnings = self._score_l(fundamental_data)
        scores['l_score'] = l_score
        if l_passed:
            passed.append("L: 市场领涨股")
        else:
            failed.append("L: 市场领涨股")
        warnings.extend(l_warnings)

        # I - Institutional Sponsorship (机构资金)
        i_score, i_passed, i_warnings = self._score_i(fundamental_data)
        scores['i_score'] = i_score
        if i_passed:
            passed.append("I: 机构资金支持")
        else:
            failed.append("I: 机构资金支持")
        warnings.extend(i_warnings)

        # M - Market Direction (大盘趋势)
        m_score, m_passed, m_warnings, market_trend = self._score_m(market_index_data)
        scores['m_score'] = m_score
        if m_passed:
            passed.append("M: 大盘上升趋势")
        else:
            failed.append("M: 大盘上升趋势")
        warnings.extend(m_warnings)

        # 计算总分（加权平均）
        weights = {'c_score': 0.2, 'a_score': 0.15, 'n_score': 0.1, 's_score': 0.1,
                  'l_score': 0.15, 'i_score': 0.15, 'm_score': 0.15}
        total_score = sum(scores[k] * weights[k] for k in scores) * 100

        # 生成建议
        recommendation = self._generate_recommendation(total_score, passed, failed)

        # 创建评分对象
        result = CANSLIM_Score(
            total_score=total_score,
            c_score=c_score,
            a_score=a_score,
            n_score=n_score,
            s_score=s_score,
            l_score=l_score,
            i_score=i_score,
            m_score=m_score,
            eps_growth_current=fundamental_data.get('eps_growth_current', 0),
            eps_growth_annual=fundamental_data.get('eps_growth_annual', 0),
            rs_rating=fundamental_data.get('rs_rating', 0),
            institutional_ownership=fundamental_data.get('institutional_ownership', 0),
            market_trend=market_trend,
            passed_criteria=passed,
            failed_criteria=failed,
            recommendation=recommendation,
            warnings=warnings,
        )

        return result

    def _score_c(
        self,
        fundamental_data: Dict[str, Any],
    ) -> Tuple[float, bool, List[str]]:
        """
        C - Current Earnings (当季收益)

        标准：当季EPS增长率至少20-25%
        """
        eps_growth = fundamental_data.get('eps_growth_current', 0)

        if eps_growth >= 25:
            score = 1.0
        elif eps_growth >= 20:
            score = 0.8
        elif eps_growth >= 15:
            score = 0.5
        else:
            score = 0.2

        passed = eps_growth >= self.min_eps_growth_current
        warnings = []

        if eps_growth < 0:
            warnings.append(f"当季收益下降{abs(eps_growth):.1f}%")
        elif eps_growth < self.min_eps_growth_current:
            warnings.append(f"当季收益增长{eps_growth:.1f}%，低于{self.min_eps_growth_current}%标准")

        return score, passed, warnings

    def _score_a(
        self,
        fundamental_data: Dict[str, Any],
    ) -> Tuple[float, bool, List[str]]:
        """
        A - Annual Earnings (年化收益)

        标准：
        - 年化EPS增长率至少25-30%
        - 连续3年以上增长
        """
        annual_growth = fundamental_data.get('eps_growth_annual', 0)
        historical_eps = fundamental_data.get('historical_annual_eps', [])

        # 检查连续增长年数
        consecutive_growth = 0
        if len(historical_eps) > 1:
            for i in range(1, len(historical_eps)):
                if historical_eps[i] > historical_eps[i-1]:
                    consecutive_growth += 1
                else:
                    break

        # 评分
        if annual_growth >= 30 and consecutive_growth >= 5:
            score = 1.0
        elif annual_growth >= 25 and consecutive_growth >= 3:
            score = 0.9
        elif annual_growth >= 20 and consecutive_growth >= 3:
            score = 0.7
        elif consecutive_growth >= 3:
            score = 0.5
        else:
            score = 0.2

        passed = (
            annual_growth >= self.min_eps_growth_annual and
            consecutive_growth >= self.min_annual_growth_years
        )

        warnings = []
        if annual_growth < 0:
            warnings.append(f"年化收益下降{abs(annual_growth):.1f}%")
        elif annual_growth < self.min_eps_growth_annual:
            warnings.append(f"年化收益增长{annual_growth:.1f}%，低于{self.min_eps_growth_annual}%标准")

        if consecutive_growth < self.min_annual_growth_years:
            warnings.append(f"仅连续增长{consecutive_growth}年，低于{self.min_annual_growth_years}年标准")

        return score, passed, warnings

    def _score_n(
        self,
        stock_data: pd.DataFrame,
        fundamental_data: Dict[str, Any],
    ) -> Tuple[float, bool, List[str]]:
        """
        N - New (新产品、新高)

        标准：
        - 有新产品、新管理层等催化剂
        - 股价接近或创52周新高
        - 或创历史新高
        """
        score = 0.0
        warnings = []

        # 检查是否接近52周新高
        current_price = stock_data['close'].iloc[-1]
        high_52w = stock_data['close'].iloc[-252:].max() if len(stock_data) >= 252 else stock_data['close'].max()

        pct_from_high = (current_price / high_52w - 1) * 100

        # 是否创历史新高
        all_time_high = fundamental_data.get('all_time_high', False)
        is_near_high = pct_from_high >= -5  # 在5%以内

        # 新产品/创新
        has_new_products = fundamental_data.get('new_products', False)

        # 评分
        if all_time_high:
            score = 1.0
        elif is_near_high:
            score = 0.8
        elif pct_from_high >= -15:
            score = 0.5
        else:
            score = 0.2

        if has_new_products:
            score = min(score + 0.2, 1.0)

        passed = score >= 0.5  # 至少接近高点或有新产品

        if pct_from_high < -20:
            warnings.append(f"股价距52周高点{abs(pct_from_high):.1f}%，较远")

        if not has_new_products and pct_from_high < -10:
            warnings.append("缺乏明确催化剂（新产品/创新）")

        return score, passed, warnings

    def _score_s(
        self,
        fundamental_data: Dict[str, Any],
        stock_data: pd.DataFrame,
    ) -> Tuple[float, bool, List[str]]:
        """
        S - Supply and Demand (供需关系)

        标准：
        - 流通盘不过大（通常<10亿股）
        - 成交量活跃
        - 有大量买盘需求
        """
        warnings = []

        # 流通股数
        shares_outstanding = fundamental_data.get('shares_outstanding', 0)
        if shares_outstanding > 0:
            # 通常中小盘股更易有供需矛盾
            if shares_outstanding < 50_000_000:  # <5000万股
                size_score = 1.0
            elif shares_outstanding < 200_000_000:  # <2亿股
                size_score = 0.7
            elif shares_outstanding < 1_000_000_000:  # <10亿股
                size_score = 0.4
            else:
                size_score = 0.1
                warnings.append(f"流通盘过大（{shares_outstanding/1_000_000:.0f}百万股）")
        else:
            size_score = 0.5  # 无数据时给中性评分

        # 成交量活跃度
        if len(stock_data) >= 20:
            recent_vol = stock_data['volume'].iloc[-20:].mean()
            prior_vol = stock_data['volume'].iloc[-40:-20].mean() if len(stock_data) >= 40 else recent_vol

            if prior_vol > 0:
                vol_ratio = recent_vol / prior_vol
                if vol_ratio >= 1.5:  # 成交量放大50%以上
                    vol_score = 1.0
                elif vol_ratio >= 1.2:
                    vol_score = 0.7
                elif vol_ratio >= 1.0:
                    vol_score = 0.5
                else:
                    vol_score = 0.2
                    warnings.append("成交量萎缩")
            else:
                vol_score = 0.5
        else:
            vol_score = 0.5

        # 综合评分
        score = (size_score + vol_score) / 2
        passed = score >= 0.5

        return score, passed, warnings

    def _score_l(
        self,
        fundamental_data: Dict[str, Any],
    ) -> Tuple[float, bool, List[str]]:
        """
        L - Leader (市场领军股票)

        标准：
        - 相对强度评级(RS Rating)至少70-80
        - 领先于大盘和同行业
        """
        rs_rating = fundamental_data.get('rs_rating', 0)

        if rs_rating >= 90:
            score = 1.0
        elif rs_rating >= 80:
            score = 0.9
        elif rs_rating >= self.min_rs_rating:
            score = 0.7
        elif rs_rating >= 50:
            score = 0.4
        else:
            score = 0.1

        passed = rs_rating >= self.min_rs_rating
        warnings = []

        if rs_rating < 50:
            warnings.append(f"相对强度评级过低（{rs_rating}），市场表现落后")
        elif rs_rating < self.min_rs_rating:
            warnings.append(f"相对强度评级{rs_rating}，低于{self.min_rs_rating}标准")

        return score, passed, warnings

    def _score_i(
        self,
        fundamental_data: Dict[str, Any],
    ) -> Tuple[float, bool, List[str]]:
        """
        I - Institutional Sponsorship (机构资金)

        标准：
        - 有机构持股（至少5-10家机构）
        - 机构持股比例适中（5-70%）
        - 最近有机构增持
        """
        inst_ownership = fundamental_data.get('institutional_ownership', 0)
        inst_count = fundamental_data.get('institutional_count', 0)

        warnings = []

        # 机构持股比例评分
        if self.min_institutional_ownership <= inst_ownership <= 70:
            ownership_score = 1.0
        elif inst_ownership > 70 and inst_ownership <= self.max_institutional_ownership:
            ownership_score = 0.7
            warnings.append(f"机构持股比例较高（{inst_ownership:.1f}%），可能缺乏上升空间")
        elif inst_ownership < self.min_institutional_ownership:
            ownership_score = 0.3
            warnings.append(f"机构持股比例过低（{inst_ownership:.1f}%），缺乏机构支持")
        else:
            ownership_score = 0.1

        # 机构数量评分
        if inst_count >= 20:
            count_score = 1.0
        elif inst_count >= 10:
            count_score = 0.7
        elif inst_count >= 5:
            count_score = 0.5
        elif inst_count > 0:
            count_score = 0.3
            warnings.append(f"机构数量较少（{inst_count}家）")
        else:
            count_score = 0.0
            warnings.append("无机构持股数据")

        # 综合评分
        score = (ownership_score + count_score) / 2
        passed = (
            inst_ownership >= self.min_institutional_ownership and
            inst_ownership <= self.max_institutional_ownership and
            inst_count >= 5
        )

        return score, passed, warnings

    def _score_m(
        self,
        market_index_data: pd.DataFrame,
    ) -> Tuple[float, bool, List[str], str]:
        """
        M - Market Direction (大盘趋势)

        标准：
        - 大盘处于上升趋势（指数在50日/200日均线上方）
        - 或出现Follow Through Day信号
        """
        warnings = []

        if len(market_index_data) < 200:
            # 数据不足
            return 0.5, True, [], "Unknown"

        # 计算均线
        index_data = market_index_data.iloc[-self.market_trend_lookback:]
        ma50 = index_data['close'].rolling(window=50).mean().iloc[-1]
        ma200 = index_data['close'].rolling(window=200).mean().iloc[-1]
        current = index_data['close'].iloc[-1]

        # 判断趋势
        if current > ma50 > ma200:
            # 明确上升趋势
            score = 1.0
            trend = "Uptrend"
            passed = True
        elif current > ma50:
            # 中性偏多
            score = 0.6
            trend = "Sideways_Up"
            passed = True
        elif current > ma200:
            # 中性偏空
            score = 0.4
            trend = "Sideways_Down"
            passed = False
            warnings.append("大盘处于下跌趋势，建议谨慎")
        else:
            # 明确下降趋势
            score = 0.1
            trend = "Downtrend"
            passed = False
            warnings.append("大盘处于明确下降趋势，建议观望")

        return score, passed, warnings, trend

    def _generate_recommendation(
        self,
        total_score: float,
        passed: List[str],
        failed: List[str],
    ) -> str:
        """生成投资建议"""

        if total_score >= 80:
            return "强烈买入 - 符合欧奈尔CANSLIM所有核心要素"
        elif total_score >= 70:
            return "买入 - 符合大部分CANSLIM要素"
        elif total_score >= 60:
            return "谨慎买入 - 符合部分CANSLIM要素，需注意风险"
        elif total_score >= 50:
            return "观望 - CANSLIM得分中等，等待更好买点"
        else:
            return "避免 - 不符合CANSLIM标准"

    def screen_multiple_stocks(
        self,
        stocks_data: Dict[str, pd.DataFrame],
        fundamentals_dict: Dict[str, Dict[str, Any]],
        market_index_data: pd.DataFrame,
        min_score: float = 60.0,
    ) -> List[Tuple[str, CANSLIM_Score]]:
        """
        批量筛选股票

        Args:
            stocks_data: {股票代码: 价格数据}
            fundamentals_dict: {股票代码: 基本面数据}
            market_index_data: 大盘指数数据
            min_score: 最小CANSLIM总分

        Returns:
            [(股票代码, CANSLIM评分)] 列表，按总分降序排列
        """
        results = []

        for symbol, stock_data in stocks_data.items():
            if symbol not in fundamentals_dict:
                continue

            try:
                score = self.screen_stock(
                    stock_data,
                    fundamentals_dict[symbol],
                    market_index_data,
                    symbol,
                )

                if score.total_score >= min_score:
                    results.append((symbol, score))

            except Exception as e:
                print(f"筛选{symbol}时出错: {e}")
                continue

        # 按总分降序排列
        results.sort(key=lambda x: x[1].total_score, reverse=True)

        return results


def calculate_relative_strength(
    stock_data: pd.DataFrame,
    index_data: pd.DataFrame,
    period_days: int = 252,  # 1年
) -> float:
    """
    计算相对强度评级（RS Rating）

    方法：比较股票和大盘的涨跌幅
    RS Rating = (股票涨跌幅 - 大盘涨跌幅 + 100) / 2

    这样，如果股票和大盘涨幅相同，RS=50；
    如果股票涨幅超过大盘，RS>50；最高100。

    Args:
        stock_data: 股票价格数据
        index_data: 大盘指数数据
        period_days: 比较周期

    Returns:
        RS Rating (0-100)
    """
    if len(stock_data) < period_days or len(index_data) < period_days:
        return 50.0  # 数据不足，返回中性值

    # 计算涨跌幅
    stock_return = (stock_data['close'].iloc[-1] / stock_data['close'].iloc[-period_days] - 1) * 100
    index_return = (index_data['close'].iloc[-1] / index_data['close'].iloc[-period_days] - 1) * 100

    # 计算相对强度
    relative_performance = stock_return - index_return

    # 映射到0-100
    # 假设：相对表现+50%为最好，-50%为最差
    rs_rating = 50 + relative_performance

    return max(0, min(100, rs_rating))


def detect_follow_through_day(
    market_index_data: pd.DataFrame,
    min_days_in_downtrend: int = 4,
    min_gain_pct: float = 1.5,
    volume_multiplier: float = 1.5,
) -> Tuple[bool, datetime, str]:
    """
    检测后续交易日（Follow Through Day）

    欧奈尔的FTD信号：
    1. 大盘处于下跌趋势至少4天
    2. 某日大盘涨幅≥1.5%（标普）或≥1.7%（纳指）
    3. 成交量显著放大（>前期均值1.5倍）
    4. 发生在第4-7天

    Args:
        market_index_data: 大盘指数数据
        min_days_in_downtrend: 最小下跌天数
        min_gain_pct: 最小涨幅（%）
        volume_multiplier: 成交量倍数

    Returns:
        (是否为FTD, 日期, 说明)
    """
    if len(market_index_data) < min_days_in_downtrend + 1:
        return False, None, "数据不足"

    # 寻找可能的FTD
    for i in range(min_days_in_downtrend, min(len(market_index_data), 7)):
        daily_data = market_index_data.iloc[i]
        prior_data = market_index_data.iloc[i-min_days_in_downtrend:i]

        # 检查前几天是否下跌
        if prior_data['close'].iloc[-1] >= prior_data['close'].iloc[0]:
            continue  # 不是下跌趋势

        # 检查当日涨幅
        daily_gain = (daily_data['close'] / daily_data['open'] - 1) * 100
        if daily_gain < min_gain_pct:
            continue

        # 检查成交量
        current_vol = daily_data['volume']
        avg_vol = prior_data['volume'].mean()

        if avg_vol > 0 and current_vol < avg_vol * volume_multiplier:
            continue  # 成交量不够

        # 找到FTD
        ftd_date = market_index_data.index[i]
        description = (
            f"后续交易日（FTD）：{ftd_date.strftime('%Y-%m-%d')}，"
            f"涨幅{daily_gain:.2f}%，成交量{current_vol/avg_vol:.1f}倍"
        )

        return True, ftd_date, description

    return False, None, "未检测到后续交易日"
