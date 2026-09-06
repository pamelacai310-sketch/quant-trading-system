"""
特征工程层 - Feature Engineering Layer

将因果因素转化为可计算的宽特征矩阵（500+特征）

核心功能：
1. 从分钟级Level 2数据低频化出500+特征
2. 将因果因素量化为可计算的数学表达式
3. 另类数据对齐（新闻情感、卫星图像等）
4. 模型自动筛选特征（保留金融含义和独立解释力）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from quant_trade_system.core.causal import (
    CausalFactorLibrary,
    CausalFactor,
    CausalEdge,
    CausalType,
    FactorCategory,
    AssetClass,
)


# ============================================================================
# 枚举定义
# ============================================================================

class FeatureGranularity(Enum):
    """特征粒度"""
    TICK = "tick"                    # 逐笔
    SECOND = "second"                # 秒级
    MINUTE_1 = "1min"                # 1分钟
    MINUTE_5 = "5min"                # 5分钟
    MINUTE_15 = "15min"              # 15分钟
    HOUR_1 = "1h"                    # 1小时
    DAILY = "daily"                  # 日级
    WEEKLY = "weekly"                # 周级
    MONTHLY = "monthly"              # 月级
    QUARTERLY = "quarterly"          # 季级


class DataSource(Enum):
    """数据源"""
    LEVEL2_ORDERBOOK = "level2_orderbook"     # Level 2订单簿
    LEVEL2_TRADES = "level2_trades"           # Level 2逐笔成交
    MINUTE_BAR = "minute_bar"                 # 分钟K线
    NEWS_SENTIMENT = "news_sentiment"         # 新闻情感
    SATELLITE_IMAGE = "satellite_image"       # 卫星图像
    ALTERNATIVE_DATA = "alternative_data"     # 另类数据


class FeatureDomain(Enum):
    """特征域"""
    PRICE = "price"                  # 价格域
    VOLUME = "volume"                # 成交量域
    VOLATILITY = "volatility"        # 波动率域
    MOMENTUM = "momentum"            # 动量域
    MACRO = "macro"                  # 宏观域
    MICROSTRUCTURE = "microstructure" # 微观结构域
    SENTIMENT = "sentiment"          # 情绪域
    CAUSAL = "causal"                # 因果域
    FUNDAMENTAL = "fundamental"      # 基本面域
    TECHNICAL = "technical"          # 技术面域
    QUALITY = "quality"              # 质量域
    VALUE = "value"                  # 价值域
    GROWTH = "growth"                # 成长域


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class QuantizedCausalFeature:
    """量化因果特征"""
    feature_id: str                       # 特征ID
    feature_name: str                     # 特征名称
    causal_factor_id: str                 # 源因果因素ID
    formula: str                          # 数学表达式
    formula_description: str               # 公式描述
    financial_meaning: str                 # 金融含义
    expected_sign: int                    # 预期符号 (+1/-1)
    category: FeatureDomain               # 特征域
    granularity: FeatureGranularity       # 数据粒度
    data_requirements: List[str]          # 数据需求
    computational_cost: float             # 计算成本 (0-1)
    interpretability: float               # 可解释性 (0-1)
    independent_power: float              # 独立解释力 (0-1)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class FeatureMatrix:
    """特征矩阵"""
    data: pd.DataFrame                    # 特征数据 (T x N)
    feature_metadata: Dict[str, Dict]     # 特征元数据
    timestamps: List[datetime]            # 时间戳
    symbols: List[str]                    # 标的物
    granularity: FeatureGranularity       # 数据粒度
    sampling_start: datetime              # 采样开始
    sampling_end: datetime                # 采样结束


@dataclass
class FeatureSelectionResult:
    """特征选择结果"""
    selected_features: List[str]         # 选中的特征
    feature_importance: Dict[str, float] # 特征重要性
    feature_correlation: pd.DataFrame     # 特征相关性矩阵
    financial_meaning: Dict[str, str]    # 金融含义
    independent_power: Dict[str, float]  # 独立解释力
    selection_criteria: str               # 选择标准


# ============================================================================
# 核心类：特征工程层
# ============================================================================

class FeatureEngineeringLayer:
    """
    特征工程层

    核心功能：
    1. 因果因素量化为500+可计算特征
    2. 分钟级数据低频化处理
    3. 另类数据对齐采样
    4. 模型自动筛选特征
    """

    def __init__(
        self,
        causal_library: Optional[CausalFactorLibrary] = None,
        target_features: int = 500,
        min_interpretability: float = 0.6,
        min_independent_power: float = 0.5,
    ):
        """
        初始化特征工程层

        参数:
            causal_library: 因果因素库
            target_features: 目标特征数量
            min_interpretability: 最小可解释性阈值
            min_independent_power: 最小独立解释力阈值
        """
        self.causal_library = causal_library or CausalFactorLibrary()
        self.target_features = target_features
        self.min_interpretability = min_interpretability
        self.min_independent_power = min_independent_power

        # 特征注册表
        self.feature_registry: Dict[str, QuantizedCausalFeature] = {}

        # 初始化特征
        self._initialize_quantized_features()

    def _initialize_quantized_features(self):
        """初始化量化因果特征"""
        # 从因果因素库生成量化特征
        for factor_id, causal_factor in self.causal_library.factors.items():
            quantized_features = self._quantize_causal_factor(causal_factor)
            for qf in quantized_features:
                self.feature_registry[qf.feature_id] = qf

    def _quantize_causal_factor(self, causal_factor: CausalFactor) -> List[QuantizedCausalFeature]:
        """
        将因果因素量化为可计算特征

        参数:
            causal_factor: 因果因素

        返回:
            量化特征列表
        """
        features = []

        # 根据因果因素类别生成不同的量化表达式
        if causal_factor.category == FactorCategory.MACRO_POLICY:
            features.extend(self._quantize_macro_factors(causal_factor))
        elif causal_factor.category == FactorCategory.MICROSTRUCTURE:
            features.extend(self._quantize_microstructure_factors(causal_factor))
        elif causal_factor.category == FactorCategory.FUNDAMENTAL:
            features.extend(self._quantize_fundamental_factors(causal_factor))
        elif causal_factor.category == FactorCategory.SUPPLY_DEMAND:
            features.extend(self._quantize_supply_demand_factors(causal_factor))
        elif causal_factor.category == FactorCategory.FUTURES_PRICING:
            features.extend(self._quantize_futures_pricing_factors(causal_factor))
        elif causal_factor.category == FactorCategory.VALUATION:
            features.extend(self._quantize_valuation_factors(causal_factor))
        elif causal_factor.category == FactorCategory.QUANT_STRATEGY:
            features.extend(self._quantize_strategy_factors(causal_factor))
        elif causal_factor.category == FactorCategory.EQUITY_PREMIUM:
            features.extend(self._quantize_equity_premium_factors(causal_factor))
        elif causal_factor.category == FactorCategory.COMMODITY_PREMIUM:
            features.extend(self._quantize_commodity_premium_factors(causal_factor))
        else:
            # 默认处理
            features.extend(self._quantize_default_factors(causal_factor))

        return features

    def _quantize_macro_factors(self, causal_factor: CausalFactor) -> List[QuantizedCausalFeature]:
        """量化宏观因素"""
        features = []

        factor_id = causal_factor.factor_id

        # 示例：利率溢价
        if factor_id == "interest_rate_premium":
            # 特征1: 利率变化率
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_change_rate",
                feature_name="利率变化率",
                causal_factor_id=factor_id,
                formula="(interest_rate_t - interest_rate_t-1) / interest_rate_t-1",
                formula_description="利率的环比变化率",
                financial_meaning="利率上升 → 贴现率上升 → 资产价格下降（负向）",
                expected_sign=-1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["interest_rate"],
                computational_cost=0.1,
                interpretability=0.95,
                independent_power=0.85,
            ))

            # 特征2: 利率绝对水平
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_level",
                feature_name="利率绝对水平",
                causal_factor_id=factor_id,
                formula="interest_rate_t",
                formula_description="当前利率的绝对值",
                financial_meaning="利率水平影响资产折现率",
                expected_sign=-1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["interest_rate"],
                computational_cost=0.05,
                interpretability=0.98,
                independent_power=0.90,
            ))

            # 特征3: 利率波动率
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_volatility",
                feature_name="利率波动率",
                causal_factor_id=factor_id,
                formula="std(interest_rate_t-20:t) / mean(interest_rate_t-20:t)",
                formula_description="过去20天的利率波动率（变异系数）",
                financial_meaning="利率波动反映不确定性，高波动要求更高风险溢价",
                expected_sign=1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["interest_rate"],
                computational_cost=0.2,
                interpretability=0.85,
                independent_power=0.75,
            ))

        # 通胀溢价
        elif factor_id == "inflation_premium":
            # 特征1: 通胀变化率
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_change_rate",
                feature_name="通胀变化率",
                causal_factor_id=factor_id,
                formula="(cpi_t - cpi_t-1) / cpi_t-1",
                formula_description="CPI环比变化率",
                financial_meaning="通胀上升 → 实际利率下降 → 风险资产受益",
                expected_sign=1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.MONTHLY,
                data_requirements=["cpi"],
                computational_cost=0.1,
                interpretability=0.95,
                independent_power=0.85,
            ))

            # 特征2: 核心通胀
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_core",
                feature_name="核心通胀",
                causal_factor_id=factor_id,
                formula="core_cpi_t",
                formula_description="剔除食品能源后的CPI",
                financial_meaning="核心通胀反映持续通胀趋势",
                expected_sign=1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.MONTHLY,
                data_requirements=["core_cpi"],
                computational_cost=0.05,
                interpretability=0.92,
                independent_power=0.80,
            ))

            # 特征3: 通胀超预期
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_surprise",
                feature_name="通胀超预期",
                causal_factor_id=factor_id,
                formula="cpi_t - cpi_expected_t",
                formula_description="实际CPI与预期CPI的差值",
                financial_meaning="通胀超预期 → 市场重新定价",
                expected_sign=1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.MONTHLY,
                data_requirements=["cpi", "cpi_expected"],
                computational_cost=0.15,
                interpretability=0.88,
                independent_power=0.90,
            ))

        # GDP增长
        elif factor_id == "gdp_growth":
            # 特征1: GDP同比增长率
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_yoy",
                feature_name="GDP同比增长率",
                causal_factor_id=factor_id,
                formula="(gdp_t - gdp_t-4) / gdp_t-4",
                formula_description="GDP相对于去年同期的增长率",
                financial_meaning="GDP增长 → 企业盈利增长 → 股价上涨",
                expected_sign=1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.QUARTERLY,
                data_requirements=["gdp"],
                computational_cost=0.1,
                interpretability=0.95,
                independent_power=0.85,
            ))

            # 特征2: GDP环比增长率
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_qoq",
                feature_name="GDP环比增长率",
                causal_factor_id=factor_id,
                formula="(gdp_t - gdp_t-1) / gdp_t-1",
                formula_description="GDP相对于上季度的增长率（季节性调整后）",
                financial_meaning="GDP环比加速 → 经济复苏 → 风险资产受益",
                expected_sign=1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.QUARTERLY,
                data_requirements=["gdp_sa"],
                computational_cost=0.15,
                interpretability=0.90,
                independent_power=0.80,
            ))

            # 特征3: GDP趋势
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_trend",
                feature_name="GDP趋势",
                causal_factor_id=factor_id,
                formula="slope(linear_regression(gdp_t-8:t))",
                formula_description="过去2年GDP的线性回归斜率",
                financial_meaning="GDP上升趋势 → 经济扩张 → 股价牛市",
                expected_sign=1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.QUARTERLY,
                data_requirements=["gdp"],
                computational_cost=0.25,
                interpretability=0.82,
                independent_power=0.75,
            ))

        # 货币政策
        elif factor_id == "monetary_policy":
            # 特征1: M2增速
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_m2_growth",
                feature_name="M2增速",
                causal_factor_id=factor_id,
                formula="(m2_t - m2_t-12) / m2_t-12",
                formula_description="M2同比增速",
                financial_meaning="M2增速 → 流动性充裕程度",
                expected_sign=1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.MONTHLY,
                data_requirements=["m2"],
                computational_cost=0.1,
                interpretability=0.92,
                independent_power=0.85,
            ))

            # 特征2: 政策利率变化
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_rate_change",
                feature_name="政策利率变化",
                causal_factor_id=factor_id,
                formula="policy_rate_t - policy_rate_t-1",
                formula_description="政策利率的变动幅度（基点）",
                financial_meaning="降息 → 宽松政策 → 利好股市",
                expected_sign=-1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["policy_rate"],
                computational_cost=0.05,
                interpretability=0.98,
                independent_power=0.90,
            ))

            # 特征3: 银行间利率
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_interbank_rate",
                feature_name="银行间利率",
                causal_factor_id=factor_id,
                formula="shibor_overnight",
                formula_description="隔夜SHIBOR利率",
                financial_meaning="银行间利率反映资金面松紧",
                expected_sign=-1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["shibor"],
                computational_cost=0.05,
                interpretability=0.95,
                independent_power=0.88,
            ))

        return features

    def _quantize_microstructure_factors(self, causal_factor: CausalFactor) -> List[QuantizedCausalFeature]:
        """量化微观结构因素"""
        features = []

        factor_id = causal_factor.factor_id

        # 机构持仓
        if factor_id == "institutional_ownership":
            # 特征1: 机构持仓比例
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_ratio",
                feature_name="机构持仓比例",
                causal_factor_id=factor_id,
                formula="institutional_holdings / total_float",
                formula_description="机构持股数量占总流通股本的比例",
                financial_meaning="机构持仓比例高 → 流动性提升 → 波动率下降",
                expected_sign=-1,
                category=FeatureDomain.MICROSTRUCTURE,
                granularity=FeatureGranularity.WEEKLY,
                data_requirements=["institutional_holdings", "total_float"],
                computational_cost=0.2,
                interpretability=0.92,
                independent_power=0.80,
            ))

            # 特征2: 机构持仓变化
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_change",
                feature_name="机构持仓变化",
                causal_factor_id=factor_id,
                formula="(institutional_holdings_t - institutional_holdings_t-1) / total_float",
                formula_description="机构持仓比例的周变化",
                financial_meaning="机构增持 → 买压增强 → 价格上涨",
                expected_sign=1,
                category=FeatureDomain.MICROSTRUCTURE,
                granularity=FeatureGranularity.WEEKLY,
                data_requirements=["institutional_holdings", "total_float"],
                computational_cost=0.15,
                interpretability=0.88,
                independent_power=0.85,
            ))

            # 特征3: 机构集中度
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_concentration",
                feature_name="机构集中度",
                causal_factor_id=factor_id,
                formula="sum(top_5_institutional_holdings) / total_institutional_holdings",
                formula_description="前5大机构持仓占机构总持仓的比例",
                financial_meaning="机构集中度高 → 观点一致 → 动量更强",
                expected_sign=1,
                category=FeatureDomain.MICROSTRUCTURE,
                granularity=FeatureGranularity.WEEKLY,
                data_requirements=["institutional_holdings"],
                computational_cost=0.25,
                interpretability=0.75,
                independent_power=0.70,
            ))

        # 流动性
        elif factor_id == "liquidity":
            # 特征1: 买卖价差
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_bid_ask_spread",
                feature_name="买卖价差",
                causal_factor_id=factor_id,
                formula="(ask - bid) / ((ask + bid) / 2)",
                formula_description="买卖价差相对中间价的百分比",
                financial_meaning="价差越小 → 流动性越好 → 交易成本越低",
                expected_sign=-1,
                category=FeatureDomain.MICROSTRUCTURE,
                granularity=FeatureGranularity.MINUTE_1,
                data_requirements=["bid", "ask"],
                computational_cost=0.15,
                interpretability=0.95,
                independent_power=0.88,
            ))

            # 特征2: 成交量
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_volume",
                feature_name="成交量",
                causal_factor_id=factor_id,
                formula="volume_t",
                formula_description="当日成交量",
                financial_meaning="成交量越大 → 流动性越充裕",
                expected_sign=1,
                category=FeatureDomain.MICROSTRUCTURE,
                granularity=FeatureGranularity.MINUTE_5,
                data_requirements=["volume"],
                computational_cost=0.05,
                interpretability=0.98,
                independent_power=0.90,
            ))

            # 特征3: 换手率
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_turnover",
                feature_name="换手率",
                causal_factor_id=factor_id,
                formula="volume_t / float_shares_outstanding",
                formula_description="成交量占流通股本的比例",
                financial_meaning="换手率高 → 流动性好，但过高可能是投机",
                expected_sign=1,
                category=FeatureDomain.MICROSTRUCTURE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["volume", "float_shares_outstanding"],
                computational_cost=0.1,
                interpretability=0.95,
                independent_power=0.85,
            ))

        # 市场情绪
        elif factor_id == "market_sentiment":
            # 特征1: VIX指数
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_vix",
                feature_name="VIX波动率指数",
                causal_factor_id=factor_id,
                formula="vix_index",
                formula_description="CBOE波动率指数",
                financial_meaning="VIX高 → 恐慌情绪 → 风险资产承压",
                expected_sign=-1,
                category=FeatureDomain.SENTIMENT,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["vix_index"],
                computational_cost=0.05,
                interpretability=0.95,
                independent_power=0.88,
            ))

            # 特征2: 看涨看跌比率
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_put_call_ratio",
                feature_name="看跌看涨比率",
                causal_factor_id=factor_id,
                formula="put_volume / call_volume",
                formula_description="看跌期权成交量与看涨期权成交量的比率",
                financial_meaning="比率高 → 看空情绪浓 → 市场承压",
                expected_sign=-1,
                category=FeatureDomain.SENTIMENT,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["put_volume", "call_volume"],
                computational_cost=0.1,
                interpretability=0.90,
                independent_power=0.80,
            ))

            # 特征3: 新增开户数
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_new_accounts",
                feature_name="新增开户数",
                causal_factor_id=factor_id,
                formula="new_accounts_growth_rate",
                formula_description="新增开户数的增长率",
                financial_meaning="新增开户数增长 → 散户入场 → 市场情绪高涨",
                expected_sign=1,
                category=FeatureDomain.SENTIMENT,
                granularity=FeatureGranularity.WEEKLY,
                data_requirements=["new_accounts"],
                computational_cost=0.1,
                interpretability=0.88,
                independent_power=0.75,
            ))

        # 波动率
        elif factor_id == "volatility":
            # 特征1: 历史波动率（20日）
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_hist_20d",
                feature_name="历史波动率20日",
                causal_factor_id=factor_id,
                formula="std(returns_t-20:t) * sqrt(252)",
                formula_description="过去20天的年化波动率",
                financial_meaning="历史波动率反映风险水平",
                expected_sign=1,
                category=FeatureDomain.VOLATILITY,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["close"],
                computational_cost=0.15,
                interpretability=0.95,
                independent_power=0.90,
            ))

            # 特征2: GARCH波动率
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_garch",
                feature_name="GARCH条件波动率",
                causal_factor_id=factor_id,
                formula="garch_conditional_volatility_t",
                formula_description="GARCH模型估计的条件波动率",
                financial_meaning="条件波动率反映当前不确定性",
                expected_sign=1,
                category=FeatureDomain.VOLATILITY,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["close"],
                computational_cost=0.4,
                interpretability=0.82,
                independent_power=0.85,
            ))

            # 特征3: 隐含波动率（如果有期权）
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_implied",
                feature_name="隐含波动率",
                causal_factor_id=factor_id,
                formula="option_implied_volatility",
                formula_description="从期权价格反推的隐含波动率",
                financial_meaning="隐含波动率反映市场预期波动",
                expected_sign=1,
                category=FeatureDomain.VOLATILITY,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["option_prices"],
                computational_cost=0.3,
                interpretability=0.85,
                independent_power=0.80,
            ))

        return features

    def _quantize_fundamental_factors(self, causal_factor: CausalFactor) -> List[QuantizedCausalFeature]:
        """量化基本面因素"""
        features = []

        factor_id = causal_factor.factor_id

        # EPS增长
        if factor_id == "eps_growth":
            # 特征1: 当季EPS增速
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_quarterly",
                feature_name="当季EPS增速",
                causal_factor_id=factor_id,
                formula="(eps_t - eps_t-4) / eps_t-4",
                formula_description="当季EPS相对于去年同期的增长率",
                financial_meaning="EPS增长 → 盈利能力提升 → 股价上涨",
                expected_sign=1,
                category=FeatureDomain.FUNDAMENTAL,
                granularity=FeatureGranularity.QUARTERLY,
                data_requirements=["eps"],
                computational_cost=0.1,
                interpretability=0.95,
                independent_power=0.85,
            ))

            # 特征2: EPS超预期
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_surprise",
                feature_name="EPS超预期幅度",
                causal_factor_id=factor_id,
                formula="(eps_t - eps_expected_t) / eps_expected_t",
                formula_description="实际EPS与预期EPS的差值比率",
                financial_meaning="EPS超预期 → 股价跳空上涨",
                expected_sign=1,
                category=FeatureDomain.FUNDAMENTAL,
                granularity=FeatureGranularity.QUARTERLY,
                data_requirements=["eps", "eps_expected"],
                computational_cost=0.15,
                interpretability=0.92,
                independent_power=0.88,
            ))

            # 特征3: EPS趋势
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_trend",
                feature_name="EPS趋势",
                causal_factor_id=factor_id,
                formula="slope(linear_regression(log(eps_t-8:t)))",
                formula_description="过去2年EPS对数的线性回归斜率",
                financial_meaning="EPS上升趋势 → 盈利持续改善 → 股价牛市",
                expected_sign=1,
                category=FeatureDomain.FUNDAMENTAL,
                granularity=FeatureGranularity.QUARTERLY,
                data_requirements=["eps"],
                computational_cost=0.25,
                interpretability=0.85,
                independent_power=0.80,
            ))

        # ROIC
        elif factor_id == "roic":
            # 特征1: ROIC水平
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_level",
                feature_name="ROIC水平",
                causal_factor_id=factor_id,
                formula="nopat / invested_capital",
                formula_description="税后营业净利润与投入资本的比率",
                financial_meaning="ROIC高 → 资本使用效率高 → 企业价值高",
                expected_sign=1,
                category=FeatureDomain.FUNDAMENTAL,
                granularity=FeatureGranularity.QUARTERLY,
                data_requirements=["nopat", "invested_capital"],
                computational_cost=0.2,
                interpretability=0.90,
                independent_power=0.82,
            ))

            # 特征2: ROIC变化
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_change",
                feature_name="ROIC变化",
                causal_factor_id=factor_id,
                formula="roic_t - roic_t-4",
                formula_description="ROIC相对于去年同期的变化（百分点）",
                financial_meaning="ROIC提升 → 资本效率改善 → 股价上涨",
                expected_sign=1,
                category=FeatureDomain.FUNDAMENTAL,
                granularity=FeatureGranularity.QUARTERLY,
                data_requirements=["roic"],
                computational_cost=0.15,
                interpretability=0.88,
                independent_power=0.85,
            ))

            # 特征3: ROIC vs WACC spread
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_spread",
                feature_name="ROIC-WACC利差",
                causal_factor_id=factor_id,
                formula="roic - wacc",
                formula_description="ROIC与WACC的利差（百分点）",
                financial_meaning="ROIC > WACC → 创造价值 → 股价上涨",
                expected_sign=1,
                category=FeatureDomain.FUNDAMENTAL,
                granularity=FeatureGranularity.QUARTERLY,
                data_requirements=["roic", "wacc"],
                computational_cost=0.25,
                interpretability=0.85,
                independent_power=0.80,
            ))

        return features

    def _quantize_supply_demand_factors(self, causal_factor: CausalFactor) -> List[QuantizedCausalFeature]:
        """量化供需因素"""
        features = []

        factor_id = causal_factor.factor_id

        # 供需平衡
        if factor_id == "supply_demand_balance":
            # 特征1: 库存消费比
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_inventory_consumption",
                feature_name="库存消费比",
                causal_factor_id=factor_id,
                formula="inventory / annual_consumption",
                formula_description="当前库存与年消费量的比率",
                financial_meaning="库存消费比低 → 供应紧张 → 价格上涨",
                expected_sign=-1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.WEEKLY,
                data_requirements=["inventory", "annual_consumption"],
                computational_cost=0.2,
                interpretability=0.90,
                independent_power=0.85,
            ))

            # 特征2: 供需缺口
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_gap",
                feature_name="供需缺口",
                causal_factor_id=factor_id,
                formula="(demand - supply) / demand",
                formula_description="需求与供应的缺口相对需求的比率",
                financial_meaning="需求>供应 → 缺口扩大 → 价格上涨",
                expected_sign=1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.MONTHLY,
                data_requirements=["demand", "supply"],
                computational_cost=0.25,
                interpretability=0.88,
                independent_power=0.80,
            ))

            # 特征3: 产能利用率
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_capacity_utilization",
                feature_name="产能利用率",
                causal_factor_id=factor_id,
                formula="current_production / max_production_capacity",
                formula_description="当前产量与最大产能的比率",
                financial_meaning="产能利用率高 → 供应紧张 → 价格上涨",
                expected_sign=1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.MONTHLY,
                data_requirements=["current_production", "max_production_capacity"],
                computational_cost=0.2,
                interpretability=0.92,
                independent_power=0.85,
            ))

        return features

    def _quantize_futures_pricing_factors(self, causal_factor: CausalFactor) -> List[QuantizedCausalFeature]:
        """量化期货定价因素"""
        features = []

        factor_id = causal_factor.factor_id

        # 基差
        if factor_id == "basis":
            # 特征1: 基差强度
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_strength",
                feature_name="基差强度",
                causal_factor_id=factor_id,
                formula="(spot_price - futures_price) / spot_price",
                formula_description="现货价格与期货价格的价差相对现货的比率",
                financial_meaning="基差>0（升水）→ 现货紧张，基差<0（贴水）→ 现货宽松",
                expected_sign=1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["spot_price", "futures_price"],
                computational_cost=0.1,
                interpretability=0.92,
                independent_power=0.88,
            ))

            # 特征2: 基差历史分位数
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_percentile",
                feature_name="基差历史分位数",
                causal_factor_id=factor_id,
                formula="percentile_rank(basis_t-60:t, current_basis)",
                formula_description="当前基差在过去60天的分位数（0-100）",
                financial_meaning="基差处于高位 → 现货异常紧张 → 价格上涨",
                expected_sign=1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["spot_price", "futures_price"],
                computational_cost=0.3,
                interpretability=0.85,
                independent_power=0.80,
            ))

        # 便利收益
        elif factor_id == "convenience_yield":
            # 特征1: 隐含便利收益
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_implied",
                feature_name="隐含便利收益",
                causal_factor_id=factor_id,
                formula="spot_price - futures_price - cost_of_carry",
                formula_description="现货价格减去期货价格和持有成本",
                financial_meaning="便利收益>0 → 持有现货的便利性高 → 贴水",
                expected_sign=1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["spot_price", "futures_price", "cost_of_carry"],
                computational_cost=0.3,
                interpretability=0.75,
                independent_power=0.70,
            ))

            # 特征2: 便利收益历史分位数
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_percentile",
                feature_name="便利收益历史分位数",
                causal_factor_id=factor_id,
                formula="percentile_rank(convenience_yield_t-60:t, current_cy)",
                formula_description="当前便利收益在过去60天的分位数（0-100）",
                financial_meaning="便利收益高位 → 现货极度紧张 → 价格上涨",
                expected_sign=1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["spot_price", "futures_price", "cost_of_carry"],
                computational_cost=0.35,
                interpretability=0.80,
                independent_power=0.75,
            ))

        return features

    def _quantize_valuation_factors(self, causal_factor: CausalFactor) -> List[QuantizedCausalFeature]:
        """量化估值因素"""
        features = []

        factor_id = causal_factor.factor_id

        # PE比率
        if factor_id == "pe_ratio" or "pe" in factor_id:
            # 特征1: PE水平
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_level",
                feature_name="PE水平",
                causal_factor_id=factor_id,
                formula="price / eps",
                formula_description="当前股价与每股收益的比率",
                financial_meaning="PE低 → 价值被低估 → 股价上涨空间大",
                expected_sign=-1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["price", "eps"],
                computational_cost=0.1,
                interpretability=0.98,
                independent_power=0.85,
            ))

            # 特征2: PE相对历史分位数
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_percentile",
                feature_name="PE历史分位数",
                causal_factor_id=factor_id,
                formula="percentile_rank(pe_t-252:t, current_pe)",
                formula_description="当前PE在过去一年的分位数（0-100）",
                financial_meaning="PE处于历史低位 → 估值便宜 → 上涨空间",
                expected_sign=-1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["price", "eps"],
                computational_cost=0.3,
                interpretability=0.92,
                independent_power=0.80,
            ))

            # 特征3: PE相对行业
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_vs_industry",
                feature_name="PE相对行业",
                causal_factor_id=factor_id,
                formula="pe_stock / pe_industry",
                formula_description="个股PE与行业PE的比率",
                financial_meaning="PE低于行业 → 相对低估 → 跑赢行业",
                expected_sign=-1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["pe_stock", "pe_industry"],
                computational_cost=0.2,
                interpretability=0.90,
                independent_power=0.82,
            ))

        # PB比率
        elif factor_id == "pb_ratio" or "pb" in factor_id:
            # 特征1: PB水平
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_level",
                feature_name="PB水平",
                causal_factor_id=factor_id,
                formula="price / book_value_per_share",
                formula_description="当前股价与每股净资产的比率",
                financial_meaning="PB低 → 价值被低估 → 股价上涨空间大",
                expected_sign=-1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["price", "book_value_per_share"],
                computational_cost=0.1,
                interpretability=0.98,
                independent_power=0.85,
            ))

            # 特征2: PB相对历史分位数
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_percentile",
                feature_name="PB历史分位数",
                causal_factor_id=factor_id,
                formula="percentile_rank(pb_t-252:t, current_pb)",
                formula_description="当前PB在过去一年的分位数（0-100）",
                financial_meaning="PB处于历史低位 → 估值便宜 → 上涨空间",
                expected_sign=-1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["price", "book_value_per_share"],
                computational_cost=0.3,
                interpretability=0.92,
                independent_power=0.80,
            ))

            # 特征3: PB相对行业
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_vs_industry",
                feature_name="PB相对行业",
                causal_factor_id=factor_id,
                formula="pb_stock / pb_industry",
                formula_description="个股PB与行业PB的比率",
                financial_meaning="PB低于行业 → 相对低估 → 跑赢行业",
                expected_sign=-1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["pb_stock", "pb_industry"],
                computational_cost=0.2,
                interpretability=0.90,
                independent_power=0.82,
            ))

        # 股息率
        elif factor_id == "dividend_yield" or "dividend" in factor_id:
            # 特征1: 股息率水平
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_level",
                feature_name="股息率水平",
                causal_factor_id=factor_id,
                formula="annual_dividend / price",
                formula_description="年度股息与股价的比率",
                financial_meaning="股息率高 → 收益率高 → 吸引价值投资",
                expected_sign=1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["annual_dividend", "price"],
                computational_cost=0.1,
                interpretability=0.98,
                independent_power=0.80,
            ))

            # 特征2: 股息率变化
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_change",
                feature_name="股息率变化",
                causal_factor_id=factor_id,
                formula="dividend_yield_t - dividend_yield_t-1",
                formula_description="股息率的变化",
                financial_meaning="股息率上升 → 分红增加 → 股价上涨",
                expected_sign=1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["annual_dividend", "price"],
                computational_cost=0.15,
                interpretability=0.90,
                independent_power=0.75,
            ))

        # EV/EBITDA
        elif factor_id == "ev_ebitda":
            # 特征1: EV/EBITDA水平
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_level",
                feature_name="EV/EBITDA水平",
                causal_factor_id=factor_id,
                formula="enterprise_value / ebitda",
                formula_description="企业价值与EBITDA的比率",
                financial_meaning="EV/EBITDA低 → 价值被低估 → 股价上涨空间大",
                expected_sign=-1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["enterprise_value", "ebitda"],
                computational_cost=0.15,
                interpretability=0.92,
                independent_power=0.85,
            ))

            # 特征2: EV/EBITDA相对历史分位数
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_percentile",
                feature_name="EV/EBITDA历史分位数",
                causal_factor_id=factor_id,
                formula="percentile_rank(ev_ebitda_t-252:t, current_ev_ebitda)",
                formula_description="当前EV/EBITDA在过去一年的分位数（0-100）",
                financial_meaning="EV/EBITDA处于历史低位 → 估值便宜 → 上涨空间",
                expected_sign=-1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["enterprise_value", "ebitda"],
                computational_cost=0.3,
                interpretability=0.88,
                independent_power=0.82,
            ))

        return features

    def _quantize_strategy_factors(self, causal_factor: CausalFactor) -> List[QuantizedCausalFeature]:
        """量化策略因素（动量、反转、趋势等）"""
        features = []

        factor_id = causal_factor.factor_id

        # 动量
        if "momentum" in factor_id:
            # 特征1: 1月动量
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_1m",
                feature_name="1月动量",
                causal_factor_id=factor_id,
                formula="(close_t - close_t-21) / close_t-21",
                formula_description="过去21天的收益率",
                financial_meaning="正动量 → 趋势延续 → 股价继续上涨",
                expected_sign=1,
                category=FeatureDomain.MOMENTUM,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["close"],
                computational_cost=0.1,
                interpretability=0.95,
                independent_power=0.85,
            ))

            # 特征2: 3月动量
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_3m",
                feature_name="3月动量",
                causal_factor_id=factor_id,
                formula="(close_t - close_t-63) / close_t-63",
                formula_description="过去63天的收益率",
                financial_meaning="正动量 → 趋势延续 → 股价继续上涨",
                expected_sign=1,
                category=FeatureDomain.MOMENTUM,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["close"],
                computational_cost=0.1,
                interpretability=0.95,
                independent_power=0.88,
            ))

            # 特征3: 6月动量
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_6m",
                feature_name="6月动量",
                causal_factor_id=factor_id,
                formula="(close_t - close_t-126) / close_t-126",
                formula_description="过去126天的收益率",
                financial_meaning="正动量 → 趋势延续 → 股价继续上涨",
                expected_sign=1,
                category=FeatureDomain.MOMENTUM,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["close"],
                computational_cost=0.1,
                interpretability=0.95,
                independent_power=0.90,
            ))

            # 特征4: 12月动量
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_12m",
                feature_name="12月动量",
                causal_factor_id=factor_id,
                formula="(close_t - close_t-252) / close_t-252",
                formula_description="过去252天的收益率",
                financial_meaning="正动量 → 趋势延续 → 股价继续上涨",
                expected_sign=1,
                category=FeatureDomain.MOMENTUM,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["close"],
                computational_cost=0.1,
                interpretability=0.95,
                independent_power=0.92,
            ))

            # 特征5: 动量加速度
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_acceleration",
                feature_name="动量加速度",
                causal_factor_id=factor_id,
                formula="momentum_3m - momentum_6m",
                formula_description="3月动量减去6月动量",
                financial_meaning="动量加速 → 上涨加速 → 强买入信号",
                expected_sign=1,
                category=FeatureDomain.MOMENTUM,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["close"],
                computational_cost=0.2,
                interpretability=0.85,
                independent_power=0.80,
            ))

        # 反转
        elif "reversal" in factor_id or "reverse" in factor_id:
            # 特征1: 1日反转
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_1d",
                feature_name="1日反转",
                causal_factor_id=factor_id,
                formula="-return_t-1",
                formula_description="负的前一日收益率",
                financial_meaning="负反转 → 昨日下跌 → 今日反弹",
                expected_sign=1,
                category=FeatureDomain.MOMENTUM,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["close"],
                computational_cost=0.05,
                interpretability=0.88,
                independent_power=0.70,
            ))

            # 特征2: 5日反转
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_5d",
                feature_name="5日反转",
                causal_factor_id=factor_id,
                formula="-return_t-5:t",
                formula_description="负的过去5天累计收益率",
                financial_meaning="负反转 → 近期下跌 → 短期反弹",
                expected_sign=1,
                category=FeatureDomain.MOMENTUM,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["close"],
                computational_cost=0.1,
                interpretability=0.85,
                independent_power=0.75,
            ))

            # 特征3: 20日反转
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_20d",
                feature_name="20日反转",
                causal_factor_id=factor_id,
                formula="-return_t-20:t",
                formula_description="负的过去20天累计收益率",
                financial_meaning="负反转 → 近期下跌 → 中期反弹",
                expected_sign=1,
                category=FeatureDomain.MOMENTUM,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["close"],
                computational_cost=0.1,
                interpretability=0.85,
                independent_power=0.78,
            ))

        # 趋势强度
        elif "trend" in factor_id or "trend_strength" in factor_id:
            # 特征1: 移动平均线斜率
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_ma_slope",
                feature_name="移动平均线斜率",
                causal_factor_id=factor_id,
                formula="slope(ma_50)",
                formula_description="50日移动平均线的线性回归斜率",
                financial_meaning="MA上升 → 趋势向上 → 股价上涨",
                expected_sign=1,
                category=FeatureDomain.MOMENTUM,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["close"],
                computational_cost=0.2,
                interpretability=0.90,
                independent_power=0.82,
            ))

            # 特征2: ADX趋势强度
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_adx",
                feature_name="ADX趋势强度",
                causal_factor_id=factor_id,
                formula="adx",
                formula_description="平均趋向指数（ADX）",
                financial_meaning="ADX高 → 趋势强度高 → 趋势策略有效",
                expected_sign=1,
                category=FeatureDomain.MOMENTUM,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["high", "low", "close"],
                computational_cost=0.3,
                interpretability=0.82,
                independent_power=0.75,
            ))

        return features

    def _quantize_equity_premium_factors(self, causal_factor: CausalFactor) -> List[QuantizedCausalFeature]:
        """量化股票溢价因素"""
        features = []

        factor_id = causal_factor.factor_id

        # 股票风险溢价
        if "equity_risk_premium" in factor_id or "erp" in factor_id:
            # 特征1: 股票风险溢价水平
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_level",
                feature_name="股票风险溢价水平",
                causal_factor_id=factor_id,
                formula="expected_equity_return - risk_free_rate",
                formula_description="预期股票收益率减去无风险利率",
                financial_meaning="ERP高 → 股票相对吸引力高 → 股价上涨",
                expected_sign=1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["expected_equity_return", "risk_free_rate"],
                computational_cost=0.15,
                interpretability=0.90,
                independent_power=0.80,
            ))

            # 特征2: ERP历史分位数
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_percentile",
                feature_name="ERP历史分位数",
                causal_factor_id=factor_id,
                formula="percentile_rank(erp_t-252:t, current_erp)",
                formula_description="当前ERP在过去一年的分位数（0-100）",
                financial_meaning="ERP处于高位 → 股票极具吸引力 → 牛市信号",
                expected_sign=1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["expected_equity_return", "risk_free_rate"],
                computational_cost=0.3,
                interpretability=0.85,
                independent_power=0.78,
            ))

        # 市场风险偏好
        elif "risk_appetite" in factor_id or "risk_aversion" in factor_id:
            # 特征1: 高收益债券利差
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_high_yield_spread",
                feature_name="高收益债券利差",
                causal_factor_id=factor_id,
                formula="high_yield_bond_yield - treasury_yield",
                formula_description="高收益债券收益率与国债收益率的利差",
                financial_meaning="利差收窄 → 风险偏好上升 → 股价上涨",
                expected_sign=-1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["high_yield_bond_yield", "treasury_yield"],
                computational_cost=0.1,
                interpretability=0.88,
                independent_power=0.80,
            ))

            # 特征2: 信用利差
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_credit_spread",
                feature_name="信用利差",
                causal_factor_id=factor_id,
                formula="bbb_bond_yield - aaa_bond_yield",
                formula_description="BBB级债券与AAA级债券的收益率利差",
                financial_meaning="信用利差收窄 → 风险偏好上升 → 股价上涨",
                expected_sign=-1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["bbb_bond_yield", "aaa_bond_yield"],
                computational_cost=0.1,
                interpretability=0.90,
                independent_power=0.82,
            ))

        return features

    def _quantize_commodity_premium_factors(self, causal_factor: CausalFactor) -> List[QuantizedCausalFeature]:
        """量化商品溢价因素"""
        features = []

        factor_id = causal_factor.factor_id

        # 商品风险溢价
        if "commodity_risk_premium" in factor_id or "crp" in factor_id:
            # 特征1: 商品风险溢价水平
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_level",
                feature_name="商品风险溢价水平",
                causal_factor_id=factor_id,
                formula="expected_commodity_return - risk_free_rate",
                formula_description="预期商品收益率减去无风险利率",
                financial_meaning="CRP高 → 商品相对吸引力高 → 期货价格上涨",
                expected_sign=1,
                category=FeatureDomain.MACRO,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["expected_commodity_return", "risk_free_rate"],
                computational_cost=0.15,
                interpretability=0.85,
                independent_power=0.75,
            ))

        # 滚动收益
        elif "roll_yield" in factor_id or "roll_return" in factor_id:
            # 特征1: 滚动收益
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_level",
                feature_name="滚动收益",
                causal_factor_id=factor_id,
                formula="(near_month_futures - far_month_futures) / near_month_futures",
                formula_description="近月合约与远月合约的价差率",
                financial_meaning="正滚动收益 → 期货贴水 → 做多收益高",
                expected_sign=1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["near_month_futures", "far_month_futures"],
                computational_cost=0.1,
                interpretability=0.82,
                independent_power=0.78,
            ))

            # 特征2: 滚动收益历史分位数
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_percentile",
                feature_name="滚动收益历史分位数",
                causal_factor_id=factor_id,
                formula="percentile_rank(roll_yield_t-60:t, current_roll_yield)",
                formula_description="当前滚动收益在过去60天的分位数（0-100）",
                financial_meaning="滚动收益处于高位 → 贴水深度 → 强做多信号",
                expected_sign=1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["near_month_futures", "far_month_futures"],
                computational_cost=0.3,
                interpretability=0.80,
                independent_power=0.75,
            ))

        # 期限结构
        elif "term_structure" in factor_id or "curve" in factor_id:
            # 特征1: 期货曲线斜率
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_slope",
                feature_name="期货曲线斜率",
                causal_factor_id=factor_id,
                formula="(futures_12m - futures_1m) / futures_1m",
                formula_description="12月合约与1月合约的价差率",
                financial_meaning="曲线陡峭 → 远期溢价高 → 做多近月",
                expected_sign=-1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["futures_12m", "futures_1m"],
                computational_cost=0.1,
                interpretability=0.85,
                independent_power=0.80,
            ))

            # 特征2: 曲线凹凸度
            features.append(QuantizedCausalFeature(
                feature_id=f"{factor_id}_curvature",
                feature_name="期货曲线凹凸度",
                causal_factor_id=factor_id,
                formula="(2 * futures_6m - futures_1m - futures_12m) / futures_6m",
                formula_description="曲线的二阶导数（凹凸度）",
                financial_meaning="曲线凸性 → 期限结构异常 → 交易机会",
                expected_sign=1,
                category=FeatureDomain.PRICE,
                granularity=FeatureGranularity.DAILY,
                data_requirements=["futures_1m", "futures_6m", "futures_12m"],
                computational_cost=0.2,
                interpretability=0.70,
                independent_power=0.65,
            ))

        return features

    def _quantize_default_factors(self, causal_factor: CausalFactor) -> List[QuantizedCausalFeature]:
        """默认量化处理（未知类别的因果因素）"""
        features = []

        factor_id = causal_factor.factor_id
        name = causal_factor.name

        # 默认创建3个基础特征
        # 特征1: 原始值
        features.append(QuantizedCausalFeature(
            feature_id=f"{factor_id}_raw",
            feature_name=f"{name}-原始值",
            causal_factor_id=factor_id,
            formula=f"{factor_id}",
            formula_description=f"{name}的原始值",
            financial_meaning=f"{causal_factor.description}",
            expected_sign=1,
            category=FeatureDomain.CAUSAL,
            granularity=FeatureGranularity.DAILY,
            data_requirements=[factor_id],
            computational_cost=0.1,
            interpretability=0.70,
            independent_power=0.60,
        ))

        # 特征2: 变化率
        features.append(QuantizedCausalFeature(
            feature_id=f"{factor_id}_change",
            feature_name=f"{name}-变化率",
            causal_factor_id=factor_id,
            formula=f"({factor_id}_t - {factor_id}_t-1) / {factor_id}_t-1",
            formula_description=f"{name}的环比变化率",
            financial_meaning=f"{name}变化 → 影响资产价格",
            expected_sign=1,
            category=FeatureDomain.CAUSAL,
            granularity=FeatureGranularity.DAILY,
            data_requirements=[factor_id],
            computational_cost=0.15,
            interpretability=0.75,
            independent_power=0.65,
        ))

        # 特征3: 历史分位数
        features.append(QuantizedCausalFeature(
            feature_id=f"{factor_id}_percentile",
            feature_name=f"{name}-历史分位数",
            causal_factor_id=factor_id,
            formula=f"percentile_rank({factor_id}_t-60:t, current_{factor_id})",
            formula_description=f"{name}在过去60天的分位数（0-100）",
            financial_meaning=f"{name}处于高位 → 极端值 → 反转风险",
            expected_sign=1,
            category=FeatureDomain.CAUSAL,
            granularity=FeatureGranularity.DAILY,
            data_requirements=[factor_id],
            computational_cost=0.25,
            interpretability=0.70,
            independent_power=0.60,
        ))

        return features

    def generate_feature_matrix(
        self,
        market_data: Dict[str, pd.DataFrame],
        granularity: FeatureGranularity = FeatureGranularity.DAILY,
        feature_limit: int = 500,
    ) -> FeatureMatrix:
        """
        生成特征矩阵

        参数:
            market_data: 市场数据字典 {symbol: dataframe}
            granularity: 数据粒度
            feature_limit: 特征数量限制

        返回:
            FeatureMatrix
        """
        if len(market_data) != 1:
            raise ValueError("Use generate_feature_panels for multiple symbols")
        self.feature_errors = {}
        # 1. 选择特征
        selected_features = self._select_top_features(
            limit=feature_limit,
            min_interpretability=self.min_interpretability,
            min_independent_power=self.min_independent_power,
        )

        # 2. 计算特征
        feature_data = {}
        metadata = {}

        for feature in selected_features:
            try:
                # 计算特征值
                computed = self._compute_feature(
                    feature.feature_id,
                    market_data,
                    granularity,
                )

                if computed is None or computed.isna().all():
                    raise ValueError("Feature has no finite observations")
                if computed is not None:
                    feature_data[feature.feature_id] = computed
                    metadata[feature.feature_id] = {
                        "name": feature.feature_name,
                        "causal_factor_id": feature.causal_factor_id,
                        "formula": feature.formula,
                        "financial_meaning": feature.financial_meaning,
                        "expected_sign": feature.expected_sign,
                        "category": feature.category.value,
                        "interpretability": feature.interpretability,
                        "independent_power": feature.independent_power,
                    }
            except Exception as e:
                self.feature_errors[feature.feature_id] = str(e)
                continue

        # 3. 对齐时间戳
        timestamps = self._align_timestamps(feature_data)

        # 4. 创建特征矩阵
        df = pd.DataFrame(feature_data, index=timestamps)

        return FeatureMatrix(
            data=df,
            feature_metadata=metadata,
            timestamps=timestamps,
            symbols=list(market_data.keys()),
            granularity=granularity,
            sampling_start=timestamps[0] if timestamps else datetime.now(),
            sampling_end=timestamps[-1] if timestamps else datetime.now(),
        )

    def generate_feature_panels(self, market_data, **kwargs):
        """Preserve instrument identity, units and errors in independent matrices."""
        result = {}
        errors = {}
        for symbol, frame in market_data.items():
            result[symbol] = self.generate_feature_matrix({symbol: frame}, **kwargs)
            errors[symbol] = dict(self.feature_errors)
        self.feature_errors = errors
        return result

    def _select_top_features(
        self,
        limit: int = 500,
        min_interpretability: float = 0.6,
        min_independent_power: float = 0.5,
    ) -> List[QuantizedCausalFeature]:
        """选择Top特征"""
        # 筛选条件
        candidates = [
            f for f in self.feature_registry.values()
            if f.interpretability >= min_interpretability
            and f.independent_power >= min_independent_power
        ]

        # 排序：按（可解释性 + 独立解释力）/2排序
        candidates.sort(
            key=lambda f: (f.interpretability + f.independent_power) / 2,
            reverse=True,
        )

        return candidates[:limit]

    def _compute_feature(
        self,
        feature_id: str,
        market_data: Dict[str, pd.DataFrame],
        granularity: FeatureGranularity,
    ) -> Optional[pd.Series]:
        """
        计算单个特征

        参数:
            feature_id: 特征ID
            market_data: 市场数据
            granularity: 数据粒度

        返回:
            特征值序列
        """
        feature = self.feature_registry.get(feature_id)
        if not feature:
            return None

        if len(market_data) != 1:
            raise ValueError("Compute features per symbol; cross-symbol averaging is prohibited")
        frame = next(iter(market_data.values()))
        inputs = {name: frame[name] for name in feature.data_requirements if name in frame}
        return self._evaluate_formula(feature.formula, inputs, granularity)

    def _evaluate_formula(
        self,
        formula: str,
        data: Dict[str, pd.Series],
        granularity: FeatureGranularity,
    ) -> pd.Series:
        """
        评估公式表达式

        参数:
            formula: 公式表达式
            data: 数据字典
            granularity: 数据粒度

        返回:
            计算结果序列
        """
        from .formulas import evaluate_formula
        return evaluate_formula(formula, data)

    def _align_timestamps(
        self,
        feature_data: Dict[str, pd.Series],
    ) -> List[datetime]:
        """对齐时间戳"""
        if not feature_data:
            return []

        # 找到最短长度
        min_length = min(len(series) for series in feature_data.values())

        # 截取对齐
        for key in feature_data:
            feature_data[key] = feature_data[key].iloc[-min_length:]

        # 获取对齐后的时间戳
        first_series = next(iter(feature_data.values()))
        return list(first_series.index)

    def process_level2_orderbook(
        self,
        orderbook_data: pd.DataFrame,
        granularity: FeatureGranularity = FeatureGranularity.MINUTE_1,
    ) -> Dict[str, pd.Series]:
        """
        处理Level 2订单簿数据，低频化出特征

        参数:
            orderbook_data: 订单簿数据，包含 bid_price_1-5, bid_volume_1-5, ask_price_1-5, ask_volume_1-5
            granularity: 目标粒度

        返回:
            特征字典 {feature_name: series}
        """
        features = {}

        try:
            # 确保时间戳为索引
            if 'timestamp' in orderbook_data.columns:
                orderbook_data = orderbook_data.set_index('timestamp')

            # 重采样到目标粒度
            resampled = orderbook_data.resample(granularity.value).last()

            # 特征1: 买卖价差（相对中间价）
            if 'bid_price_1' in resampled.columns and 'ask_price_1' in resampled.columns:
                mid_price = (resampled['bid_price_1'] + resampled['ask_price_1']) / 2
                features['bid_ask_spread'] = (resampled['ask_price_1'] - resampled['bid_price_1']) / mid_price

            # 特征2: 买卖价差绝对值
            if 'bid_price_1' in resampled.columns and 'ask_price_1' in resampled.columns:
                features['bid_ask_spread_abs'] = resampled['ask_price_1'] - resampled['bid_price_1']

            # 特征3: 订单簿不平衡
            if 'bid_volume_1' in resampled.columns and 'ask_volume_1' in resampled.columns:
                features['order_imbalance_1'] = (
                    (resampled['bid_volume_1'] - resampled['ask_volume_1']) /
                    (resampled['bid_volume_1'] + resampled['ask_volume_1'])
                )

            # 特征4: 前5档买卖量不平衡
            bid_vol_cols = [f'bid_volume_{i}' for i in range(1, 6) if f'bid_volume_{i}' in resampled.columns]
            ask_vol_cols = [f'ask_volume_{i}' for i in range(1, 6) if f'ask_volume_{i}' in resampled.columns]

            if bid_vol_cols and ask_vol_cols:
                total_bid_vol = resampled[bid_vol_cols].sum(axis=1)
                total_ask_vol = resampled[ask_vol_cols].sum(axis=1)
                features['order_imbalance_5'] = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)

            # 特征5: 中间价
            if 'bid_price_1' in resampled.columns and 'ask_price_1' in resampled.columns:
                features['mid_price'] = mid_price

            # 特征6: 中间价变化率
            if 'bid_price_1' in resampled.columns and 'ask_price_1' in resampled.columns:
                features['mid_price_change'] = mid_price.pct_change()

            # 特征7: 买卖价差波动率
            if 'bid_ask_spread' in features:
                features['bid_ask_spread_volatility'] = features['bid_ask_spread'].rolling(window=20).std()

            # 特征8: 深度加权买卖价
            if 'bid_price_1' in resampled.columns and 'bid_volume_1' in resampled.columns:
                features['vwap_bid'] = resampled['bid_price_1'] * resampled['bid_volume_1']

            if 'ask_price_1' in resampled.columns and 'ask_volume_1' in resampled.columns:
                features['vwap_ask'] = resampled['ask_price_1'] * resampled['ask_volume_1']

            # 特征9: 压力指标（大单压力）
            if 'bid_volume_1' in resampled.columns and 'ask_volume_1' in resampled.columns:
                features['pressure_ratio'] = resampled['bid_volume_1'] / resampled['ask_volume_1']

            # 特征10: 订单簿斜率（价格/量关系）
            if all(col in resampled.columns for col in ['bid_price_1', 'bid_price_2', 'bid_volume_1', 'bid_volume_2']):
                features['bid_slope'] = (
                    (resampled['bid_price_1'] - resampled['bid_price_2']) /
                    (resampled['bid_volume_1'] - resampled['bid_volume_2']).replace(0, np.nan)
                )

            if all(col in resampled.columns for col in ['ask_price_1', 'ask_price_2', 'ask_volume_1', 'ask_volume_2']):
                features['ask_slope'] = (
                    (resampled['ask_price_2'] - resampled['ask_price_1']) /
                    (resampled['ask_volume_2'] - resampled['ask_volume_1']).replace(0, np.nan)
                )

            # 特征11: 累计深度
            if bid_vol_cols:
                features['cumulative_bid_depth'] = resampled[bid_vol_cols].sum(axis=1)

            if ask_vol_cols:
                features['cumulative_ask_depth'] = resampled[ask_vol_cols].sum(axis=1)

            # 特征12: 深度比率
            if 'cumulative_bid_depth' in features and 'cumulative_ask_depth' in features:
                features['depth_ratio'] = (
                    features['cumulative_bid_depth'] /
                    features['cumulative_ask_depth']
                )

            # 特征13: 价格冲击（假设有大单交易）
            if 'bid_price_1' in resampled.columns and 'bid_price_5' in resampled.columns:
                features['price_impact_bid'] = (resampled['bid_price_1'] - resampled['bid_price_5']) / resampled['bid_price_5']

            if 'ask_price_1' in resampled.columns and 'ask_price_5' in resampled.columns:
                features['price_impact_ask'] = (resampled['ask_price_5'] - resampled['ask_price_1']) / resampled['ask_price_1']

        except Exception as e:
            print(f"处理订单簿数据时出错: {e}")

        return features

    def process_level2_trades(
        self,
        trades_data: pd.DataFrame,
        granularity: FeatureGranularity = FeatureGranularity.MINUTE_1,
    ) -> Dict[str, pd.Series]:
        """
        处理Level 2逐笔成交数据，低频化出特征

        参数:
            trades_data: 逐笔成交数据，包含 timestamp, price, volume, direction(buy/sell)
            granularity: 目标粒度

        返回:
            特征字典 {feature_name: series}
        """
        features = {}

        try:
            # 确保时间戳为索引
            if 'timestamp' in trades_data.columns:
                trades_data = trades_data.set_index('timestamp')

            # 重采样到目标粒度
            resampled = trades_data.resample(granularity.value)

            # 特征1: 成交量
            if 'volume' in trades_data.columns:
                features['trade_volume'] = resampled['volume'].sum()

            # 特征2: 成交额
            if 'price' in trades_data.columns and 'volume' in trades_data.columns:
                features['trade_value'] = (trades_data['price'] * trades_data['volume']).resample(granularity.value).sum()

            # 特征3: 成交笔数
            features['trade_count'] = resampled['price'].count()

            # 特征4: VWAP（成交量加权平均价）
            if 'price' in trades_data.columns and 'volume' in trades_data.columns:
                features['vwap'] = (
                    (trades_data['price'] * trades_data['volume']).resample(granularity.value).sum() /
                    resampled['volume'].sum()
                )

            # 特征5: 买卖成交量
            if 'direction' in trades_data.columns:
                buy_trades = trades_data[trades_data['direction'] == 'buy']
                sell_trades = trades_data[trades_data['direction'] == 'sell']

                if 'volume' in trades_data.columns:
                    features['buy_volume'] = buy_trades['volume'].resample(granularity.value).sum()
                    features['sell_volume'] = sell_trades['volume'].resample(granularity.value).sum()

                    # 特征6: 买卖比率
                    features['buy_sell_ratio'] = (
                        features['buy_volume'] / features['sell_volume'].replace(0, np.nan)
                    )

                    # 特征7: 净买入量
                    features['net_buy_volume'] = features['buy_volume'] - features['sell_volume']

            # 特征8: 成交量波动率
            if 'volume' in trades_data.columns:
                features['volume_volatility'] = resampled['volume'].std()

            # 特征9: 价格波动率
            if 'price' in trades_data.columns:
                features['price_volatility'] = resampled['price'].std()

            # 特征10: 最大单笔成交量
            if 'volume' in trades_data.columns:
                features['max_trade_volume'] = resampled['volume'].max()

            # 特征11: 平均单笔成交量
            if 'volume' in trades_data.columns:
                features['avg_trade_volume'] = resampled['volume'].mean()

            # 特征12: 大单占比（假设大于平均值的为大单）
            if 'volume' in trades_data.columns:
                avg_volume = trades_data['volume'].mean()
                large_trades = trades_data[trades_data['volume'] > avg_volume]
                features['large_trade_ratio'] = (
                    large_trades['volume'].resample(granularity.value).sum() /
                    resampled['volume'].sum()
                )

            # 特征13: 价格变化幅度
            if 'price' in trades_data.columns:
                features['price_range'] = (
                    resampled['price'].max() - resampled['price'].min()
                )

            # 特征14: 成交量变化率
            if 'volume' in trades_data.columns:
                features['volume_change'] = resampled['volume'].sum().pct_change()

        except Exception as e:
            print(f"处理逐笔成交数据时出错: {e}")

        return features

    def align_news_sentiment(
        self,
        news_data: pd.DataFrame,
        market_timestamps: List[datetime],
        window_minutes: int = 60,
    ) -> pd.DataFrame:
        """
        对齐新闻情感数据到市场时间戳

        参数:
            news_data: 新闻数据，包含 timestamp, sentiment_score, relevance_score
            market_timestamps: 市场时间戳列表
            window_minutes: 情感聚合窗口（分钟）

        返回:
            对齐后的情感特征DataFrame
        """
        try:
            # 确保时间戳列存在
            if 'timestamp' not in news_data.columns:
                return pd.DataFrame(index=market_timestamps)

            # 转换为datetime
            news_data['timestamp'] = pd.to_datetime(news_data['timestamp'])

            # 创建结果DataFrame
            result = pd.DataFrame(index=market_timestamps)

            # 为每个市场时间戳聚合窗口内的新闻情感
            sentiment_features = []
            for ts in market_timestamps:
                # 定义窗口
                window_start = ts - timedelta(minutes=window_minutes)
                window_end = ts

                # 筛选窗口内的新闻
                window_news = news_data[
                    (news_data['timestamp'] >= window_start) &
                    (news_data['timestamp'] <= window_end)
                ]

                if not window_news.empty:
                    # 特征1: 平均情感得分
                    if 'sentiment_score' in window_news.columns:
                        avg_sentiment = window_news['sentiment_score'].mean()
                        sentiment_features.append({
                            'timestamp': ts,
                            'news_sentiment_avg': avg_sentiment,
                        })

                    # 特征2: 加权情感得分（按相关性加权）
                    if 'sentiment_score' in window_news.columns and 'relevance_score' in window_news.columns:
                        weighted_sentiment = (
                            window_news['sentiment_score'] * window_news['relevance_score']
                        ).sum() / window_news['relevance_score'].sum()
                        sentiment_features[-1]['news_sentiment_weighted'] = weighted_sentiment

                    # 特征3: 新闻数量
                    sentiment_features[-1]['news_count'] = len(window_news)

                    # 特征4: 正面新闻占比
                    if 'sentiment_score' in window_news.columns:
                        positive_ratio = (window_news['sentiment_score'] > 0).sum() / len(window_news)
                        sentiment_features[-1]['news_positive_ratio'] = positive_ratio

                    # 特征5: 负面新闻占比
                    if 'sentiment_score' in window_news.columns:
                        negative_ratio = (window_news['sentiment_score'] < 0).sum() / len(window_news)
                        sentiment_features[-1]['news_negative_ratio'] = negative_ratio

                    # 特征6: 情感极性（正负新闻得分的差值）
                    if 'sentiment_score' in window_news.columns:
                        positive_score = window_news[window_news['sentiment_score'] > 0]['sentiment_score'].sum()
                        negative_score = window_news[window_news['sentiment_score'] < 0]['sentiment_score'].sum()
                        sentiment_features[-1]['news_sentiment_polarity'] = positive_score - abs(negative_score)

                    # 特征7: 情感波动率
                    if 'sentiment_score' in window_news.columns and len(window_news) > 1:
                        sentiment_volatility = window_news['sentiment_score'].std()
                        sentiment_features[-1]['news_sentiment_volatility'] = sentiment_volatility

                    # 特征8: 最新新闻情感
                    if 'sentiment_score' in window_news.columns:
                        latest_sentiment = window_news.iloc[-1]['sentiment_score']
                        sentiment_features[-1]['news_sentiment_latest'] = latest_sentiment

                else:
                    # 没有新闻时填充NaN
                    sentiment_features.append({
                        'timestamp': ts,
                        'news_sentiment_avg': np.nan,
                        'news_sentiment_weighted': np.nan,
                        'news_count': 0,
                        'news_positive_ratio': np.nan,
                        'news_negative_ratio': np.nan,
                        'news_sentiment_polarity': np.nan,
                        'news_sentiment_volatility': np.nan,
                        'news_sentiment_latest': np.nan,
                    })

            # 转换为DataFrame
            sentiment_df = pd.DataFrame(sentiment_features)
            sentiment_df = sentiment_df.set_index('timestamp')

            return sentiment_df

        except Exception as e:
            print(f"对齐新闻情感数据时出错: {e}")
            return pd.DataFrame(index=market_timestamps)

    def align_satellite_data(
        self,
        satellite_data: pd.DataFrame,
        market_timestamps: List[datetime],
        location: str,
        window_hours: int = 24,
    ) -> pd.DataFrame:
        """
        对齐卫星图像数据到市场时间戳

        参数:
            satellite_data: 卫星数据，包含 timestamp, location, feature_values(dict)
            market_timestamps: 市场时间戳列表
            location: 地点标识
            window_hours: 卫星数据聚合窗口（小时）

        返回:
            对齐后的卫星特征DataFrame
        """
        try:
            # 筛选指定地点的数据
            if 'location' in satellite_data.columns:
                satellite_data = satellite_data[satellite_data['location'] == location]

            # 确保时间戳列存在
            if 'timestamp' not in satellite_data.columns:
                return pd.DataFrame(index=market_timestamps)

            # 转换为datetime
            satellite_data['timestamp'] = pd.to_datetime(satellite_data['timestamp'])

            # 创建结果DataFrame
            result = pd.DataFrame(index=market_timestamps)

            # 为每个市场时间戳聚合窗口内的卫星数据
            satellite_features = []
            for ts in market_timestamps:
                # 定义窗口
                window_start = ts - timedelta(hours=window_hours)
                window_end = ts

                # 筛选窗口内的卫星数据
                window_data = satellite_data[
                    (satellite_data['timestamp'] >= window_start) &
                    (satellite_data['timestamp'] <= window_end)
                ]

                if not window_data.empty:
                    # 假设feature_values是一个字典，包含各种卫星提取的特征
                    # 例如：{'ndvi': 0.6, 'night_lights': 100, 'construction_area': 500}

                    # 聚合所有特征的最新值和平均值
                    if 'feature_values' in window_data.columns:
                        all_features = set()
                        for features in window_data['feature_values']:
                            if isinstance(features, dict):
                                all_features.update(features.keys())

                        feature_row = {'timestamp': ts}

                        for feature_name in all_features:
                            # 提取该特征的值
                            feature_values = [
                                f.get(feature_name, np.nan)
                                for f in window_data['feature_values']
                                if isinstance(f, dict) and feature_name in f
                            ]

                            if feature_values:
                                # 特征1: 最新值
                                feature_row[f'satellite_{feature_name}_latest'] = feature_values[-1]

                                # 特征2: 平均值
                                feature_row[f'satellite_{feature_name}_avg'] = np.mean(feature_values)

                                # 特征3: 变化率
                                if len(feature_values) > 1:
                                    feature_row[f'satellite_{feature_name}_change'] = (
                                        feature_values[-1] - feature_values[0]
                                    ) / feature_values[0] if feature_values[0] != 0 else np.nan

                                # 特征5: 标准差
                                if len(feature_values) > 1:
                                    feature_row[f'satellite_{feature_name}_std'] = np.std(feature_values)

                        # 特征6: 卫星数据点数量（反映云层覆盖等）
                        feature_row['satellite_data_count'] = len(window_data)

                        satellite_features.append(feature_row)

                else:
                    # 没有卫星数据时填充NaN
                    satellite_features.append({
                        'timestamp': ts,
                        'satellite_data_count': 0,
                    })

            # 转换为DataFrame
            satellite_df = pd.DataFrame(satellite_features)
            satellite_df = satellite_df.set_index('timestamp')

            return satellite_df

        except Exception as e:
            print(f"对齐卫星数据时出错: {e}")
            return pd.DataFrame(index=market_timestamps)


# ============================================================================
# 工厂函数
# ============================================================================

def create_feature_engineering_layer(
    target_features: int = 500,
    min_interpretability: float = 0.6,
    min_independent_power: float = 0.5,
) -> FeatureEngineeringLayer:
    """创建特征工程层"""
    return FeatureEngineeringLayer(
        target_features=target_features,
        min_interpretability=min_interpretability,
        min_independent_power=min_independent_power,
    )


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 创建特征工程层
    layer = create_feature_engineering_layer()

    # 生成特征矩阵
    market_data = {}  # 加载市场数据

    feature_matrix = layer.generate_feature_matrix(
        market_data=market_data,
        granularity=FeatureGranularity.DAILY,
        feature_limit=500,
    )

    print(f"✅ 特征矩阵生成成功")
    print(f"  特征数量: {len(feature_matrix.data.columns)}")
    print(f"  时间范围: {feature_matrix.sampling_start} - {feature_matrix.sampling_end}")
    print(f"  数据粒度: {feature_matrix.granularity.value}")
