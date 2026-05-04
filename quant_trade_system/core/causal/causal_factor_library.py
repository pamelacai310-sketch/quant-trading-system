"""
因果因素库 (Causal Factor Library)

定义所有股票和商品期货的因果因素。

这些因素是"非结构化、定性、主观、复杂的因果关系"，
底层逻辑偏向因果关系 (A导致B)，而不是量化因子。
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
import numpy as np


# ============================================================================
# 枚举定义
# ============================================================================

class FactorCategory(Enum):
    """因素类别"""
    # 共有因素
    MACRO_POLICY = "macro_policy"              # 宏观与政策
    MICROSTRUCTURE = "microstructure"          # 市场微观结构
    QUANT_STRATEGY = "quant_strategy"          # 量化策略因子

    # 股票专属
    FUNDAMENTAL = "fundamental"                # 公司基本面
    VALUATION = "valuation"                    # 估值
    EQUITY_PREMIUM = "equity_premium"          # 股票特有溢价

    # 期货专属
    SUPPLY_DEMAND = "supply_demand"            # 供需
    FUTURES_PRICING = "futures_pricing"        # 期货定价
    COMMODITY_PREMIUM = "commodity_premium"    # 商品溢价


class AssetClass(Enum):
    """资产类别"""
    ALL = "all"                               # 所有资产
    EQUITY = "equity"                         # 股票
    COMMODITY = "commodity"                   # 商品期货
    FIXED_INCOME = "fixed_income"             # 固定收益
    FX = "fx"                                 # 外汇
    CRYPTO = "crypto"                         # 加密货币


class CausalType(Enum):
    """因果类型"""
    DIRECT_POSITIVE = "direct_positive"       # 正向因果 (A↑ → B↑)
    DIRECT_NEGATIVE = "direct_negative"       # 负向因果 (A↑ → B↓)
    INDIRECT = "indirect"                     # 间接因果 (A → C → B)
    CONFOUNDED = "confounded"                 # 混杂因果
    BIDIRECTIONAL = "bidirectional"           # 双向因果
    THRESHOLD = "threshold"                   # 阈值因果
    TEMPORAL = "temporal"                     # 时间因果


class EvidenceType(Enum):
    """证据类型"""
    EMPIRICAL = "empirical"                   # 实证证据
    THEORETICAL = "theoretical"               # 理论证据
    EXPERT = "expert"                         # 专家判断
    SIMULATION = "simulation"                 # 模拟结果
    BACKTEST = "backtest"                     # 回测验证


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class CausalFactor:
    """因果因素"""
    factor_id: str                          # 因素ID
    name: str                              # 因素名称
    category: FactorCategory               # 因素类别
    asset_class: AssetClass                # 资产类别
    description: str                       # 因素描述
    causal_mechanism: str                  # 因果机制说明
    data_sources: List[str]                # 数据来源
    measurement_methods: List[str]         # 测量方法
    update_frequency: str                  # 更新频率
    reliability: float                     # 可靠性评分 (0-1)
    confidence: float                      # 置信度 (0-1)
    created_at: datetime
    updated_at: datetime
    version: int                           # 版本号

    # 可选字段
    parent_factors: List[str] = field(default_factory=list)  # 父级因素
    child_factors: List[str] = field(default_factory=list)   # 子级因素
    tags: List[str] = field(default_factory=list)           # 标签
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


@dataclass
class CausalEvidence:
    """因果证据"""
    evidence_id: str
    evidence_type: EvidenceType
    description: str
    data_period: Optional[Tuple[datetime, datetime]] = None
    statistical_significance: Optional[float] = None
    effect_size: Optional[float] = None
    source: str = ""
    url: Optional[str] = None
    validated: bool = False
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CausalEdge:
    """因果边"""
    edge_id: str                           # 边ID
    source_factor_id: str                  # 源因素ID
    target_factor_id: str                  # 目标因素ID
    causal_type: CausalType                # 因果类型
    causal_strength: float                 # 因果强度 (0-1)
    lag_days: int                          # 滞后天数
    confidence: float                      # 置信度 (0-1)
    direction: str                         # 因果方向 (forward/backward/bidirectional)
    mechanism: str                         # 因果机制说明
    evidence: List[CausalEvidence]         # 支持证据
    market_regime: Optional[str]           # 适用市场制度
    created_at: datetime
    updated_at: datetime = field(default_factory=datetime.now)
    validated_count: int = 0               # 验证次数
    success_count: int = 0                 # 成功次数

    # 可选字段
    threshold: Optional[float] = None      # 阈值（如果是THRESHOLD类型）
    conditions: List[str] = field(default_factory=list)  # 条件


# ============================================================================
# 因果因素库定义
# ============================================================================

class CausalFactorLibrary:
    """因果因素库"""

    def __init__(self):
        self.factors: Dict[str, CausalFactor] = {}
        self._initialize_common_factors()
        self._initialize_equity_factors()
        self._initialize_commodity_factors()

    def _initialize_common_factors(self):
        """初始化共有因素"""

        # ========================================
        # 1. 宏观与政策环境
        # ========================================

        self.add_factor(CausalFactor(
            factor_id="gdp_growth",
            name="经济增长",
            category=FactorCategory.MACRO_POLICY,
            asset_class=AssetClass.ALL,
            description="经济规模大 → 风险资产需求上升 → 股价/商品价格上涨",
            causal_mechanism="GDP增长 → 企业盈利增长 → 股价上涨；GDP增长 → 大宗商品需求上升 → 商品价格上涨",
            data_sources=["国家统计局", "GDP数据", "经济指标"],
            measurement_methods=["同比增长率", "环比增长率"],
            update_frequency="quarterly",
            reliability=0.95,
            confidence=0.90,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["macro", "fundamental", "growth"],
        ))

        self.add_factor(CausalFactor(
            factor_id="monetary_policy",
            name="货币政策",
            category=FactorCategory.MACRO_POLICY,
            asset_class=AssetClass.ALL,
            description="货币政策宽松 → 流动性充裕 → 资产价格上涨",
            causal_mechanism="降息/降准 → 市场利率下降 → 贴现率下降 → 资产现值上升 → 价格上涨",
            data_sources=["央行政策", "利率数据", "M2增速"],
            measurement_methods=["政策利率", "M2增速", "社融规模"],
            update_frequency="monthly",
            reliability=0.90,
            confidence=0.85,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["macro", "policy", "liquidity"],
        ))

        self.add_factor(CausalFactor(
            factor_id="inflation_premium",
            name="通胀溢价",
            category=FactorCategory.MACRO_POLICY,
            asset_class=AssetClass.ALL,
            description="通胀上升 → 实际利率下降 → 风险资产价格上涨",
            causal_mechanism="通胀上升 → 名义利率上升但实际利率下降 → 持有现金成本上升 → 风险资产需求上升 → 价格上涨",
            data_sources=["CPI", "PPI", "通胀数据"],
            measurement_methods=["CPI同比", "核心CPI", "PPI"],
            update_frequency="monthly",
            reliability=0.88,
            confidence=0.82,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["macro", "inflation"],
        ))

        self.add_factor(CausalFactor(
            factor_id="interest_rate_premium",
            name="利率溢价",
            category=FactorCategory.MACRO_POLICY,
            asset_class=AssetClass.ALL,
            description="利率上升 → 贴现率上升 → 资产价格下降（负向因果）",
            causal_mechanism="利率↑ → 贴现率↑ → 未来现金流现值↓ → 资产价格↓",
            data_sources=["央行政策利率", "国债收益率"],
            measurement_methods=["10年期国债收益率", "SHIBOR", "LIBOR"],
            update_frequency="daily",
            reliability=0.92,
            confidence=0.88,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["macro", "interest_rate"],
        ))

        # ========================================
        # 2. 市场微观结构
        # ========================================

        self.add_factor(CausalFactor(
            factor_id="institutional_ownership",
            name="机构持仓",
            category=FactorCategory.MICROSTRUCTURE,
            asset_class=AssetClass.ALL,
            description="机构持仓重 → 流动性提升 → 价格发现更有效 → 波动率下降",
            causal_mechanism="机构买入 → 成交量上升 → 买卖价差缩小 → 流动性提升 → 价格发现更有效",
            data_sources=["持仓报告", "交易所数据"],
            measurement_methods=["机构持仓比例", "持仓集中度"],
            update_frequency="weekly",
            reliability=0.85,
            confidence=0.80,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["microstructure", "liquidity"],
        ))

        self.add_factor(CausalFactor(
            factor_id="liquidity",
            name="流动性",
            category=FactorCategory.MICROSTRUCTURE,
            asset_class=AssetClass.ALL,
            description="流动性充裕 → 买卖价差小 → 交易成本低 → 价格更有效",
            causal_mechanism="流动性↑ → 买卖价差↓ → 交易成本↓ → 价格更有效率",
            data_sources=["成交量", "买卖价差"],
            measurement_methods=["买卖价差", "成交量", "换手率"],
            update_frequency="daily",
            reliability=0.90,
            confidence=0.85,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["microstructure", "liquidity"],
        ))

        self.add_factor(CausalFactor(
            factor_id="market_sentiment",
            name="市场情绪",
            category=FactorCategory.MICROSTRUCTURE,
            asset_class=AssetClass.ALL,
            description="市场情绪乐观 → 风险偏好上升 → 风险资产价格上涨",
            causal_mechanism="情绪乐观 → 风险偏好↑ → 风险资产需求↑ → 价格上涨",
            data_sources=["VIX", "投资者情绪指数", "资金流向"],
            measurement_methods=["VIX", "看涨看跌比率", "新增开户数"],
            update_frequency="daily",
            reliability=0.82,
            confidence=0.78,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["microstructure", "sentiment"],
        ))

        self.add_factor(CausalFactor(
            factor_id="volatility",
            name="波动率",
            category=FactorCategory.MICROSTRUCTURE,
            asset_class=AssetClass.ALL,
            description="波动率上升 → 不确定性增加 → 风险溢价上升 → 期望收益率上升",
            causal_mechanism="波动率↑ → 不确定性↑ → 风险溢价↑ → 期望收益率↑",
            data_sources=["历史波动率", "VIX", "隐含波动率"],
            measurement_methods=["历史波动率", "GARCH模型"],
            update_frequency="daily",
            reliability=0.88,
            confidence=0.85,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["microstructure", "volatility", "risk"],
        ))

        # ========================================
        # 3. 量化策略因子
        # ========================================

        self.add_factor(CausalFactor(
            factor_id="momentum_premium",
            name="动量溢价",
            category=FactorCategory.QUANT_STRATEGY,
            asset_class=AssetClass.ALL,
            description="过去表现好的资产 → 未来继续表现好（正反馈）",
            causal_mechanism="价格上涨 → 投资者FOMO → 追涨 → 继续上涨 → 正反馈循环",
            data_sources=["价格数据"],
            measurement_methods=["过去3-12个月收益率"],
            update_frequency="daily",
            reliability=0.80,
            confidence=0.75,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["factor", "momentum", "anomaly"],
        ))

        self.add_factor(CausalFactor(
            factor_id="volatility_risk_premium",
            name="波动率风险溢价",
            category=FactorCategory.QUANT_STRATEGY,
            asset_class=AssetClass.ALL,
            description="高波动率资产 → 风险溢价高 → 期望收益率高",
            causal_mechanism="高波动率 → 高风险 → 投资者要求高回报 → 风险溢价高",
            data_sources=["历史波动率", "收益率数据"],
            measurement_methods=["波动率", "下行风险"],
            update_frequency="daily",
            reliability=0.78,
            confidence=0.72,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["factor", "volatility", "risk_premium"],
        ))

    def _initialize_equity_factors(self):
        """初始化股票专属因素"""

        # ========================================
        # 1. 公司基本面
        # ========================================

        self.add_factor(CausalFactor(
            factor_id="eps_growth",
            name="EPS增长",
            category=FactorCategory.FUNDAMENTAL,
            asset_class=AssetClass.EQUITY,
            description="EPS增长 → 盈利能力提升 → 股价上涨",
            causal_mechanism="EPS↑ → 每股收益↑ → 股票内在价值↑ → 股价↑",
            data_sources=["财报", "盈利预测"],
            measurement_methods=["EPS同比增长率", "预期EPS增长率"],
            update_frequency="quarterly",
            reliability=0.92,
            confidence=0.88,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["fundamental", "earnings", "growth"],
        ))

        self.add_factor(CausalFactor(
            factor_id="roic",
            name="ROIC（资本回报率）",
            category=FactorCategory.FUNDAMENTAL,
            asset_class=AssetClass.EQUITY,
            description="ROIC高 → 资本使用效率高 → 企业价值高 → 股价上涨",
            causal_mechanism="ROIC↑ → 资本使用效率↑ → 企业创造价值能力↑ → 股价↑",
            data_sources=["财报"],
            measurement_methods=["ROIC = NOPAT / 投入资本"],
            update_frequency="quarterly",
            reliability=0.90,
            confidence=0.85,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["fundamental", "profitability", "efficiency"],
        ))

        self.add_factor(CausalFactor(
            factor_id="credit_rating",
            name="信用评级",
            category=FactorCategory.FUNDAMENTAL,
            asset_class=AssetClass.EQUITY,
            description="信用评级高 → 违约风险低 → 融资成本低 → 企业价值高",
            causal_mechanism="评级↑ → 违约风险↓ → 融资成本↓ → 利润↑ → 股价↑",
            data_sources=["标普", "穆迪", "惠誉"],
            measurement_methods=["信用评级等级"],
            update_frequency="quarterly",
            reliability=0.88,
            confidence=0.82,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["fundamental", "credit", "risk"],
        ))

        # ========================================
        # 2. 估值与股东回报
        # ========================================

        self.add_factor(CausalFactor(
            factor_id="valuation_level",
            name="估值水平",
            category=FactorCategory.VALUATION,
            asset_class=AssetClass.EQUITY,
            description="估值低 → 期望收益率高 → 股价上涨（价值回归）",
            causal_mechanism="估值低（P/E低）→ 相对价值高 → 买入压力↑ → 股价↑ → 估值回归",
            data_sources=["行情数据", "估值数据"],
            measurement_methods=["P/E", "P/B", "EV/EBITDA"],
            update_frequency="daily",
            reliability=0.85,
            confidence=0.78,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["valuation", "value"],
        ))

        self.add_factor(CausalFactor(
            factor_id="shareholder_yield",
            name="股东回报率",
            category=FactorCategory.VALUATION,
            asset_class=AssetClass.EQUITY,
            description="分红/回购高 → 股东回报高 → 股价上涨",
            causal_mechanism="分红/回购↑ → 股东现金回报↑ → 吸引力↑ → 买入压力↑ → 股价↑",
            data_sources=["分红数据", "回购数据"],
            measurement_methods=["股息率", "回购比例"],
            update_frequency="quarterly",
            reliability=0.82,
            confidence=0.75,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["valuation", "dividend", "buyback"],
        ))

        # ========================================
        # 3. 股票特有因子溢价
        # ========================================

        self.add_factor(CausalFactor(
            factor_id="equity_risk_premium",
            name="股权风险溢价",
            category=FactorCategory.EQUITY_PREMIUM,
            asset_class=AssetClass.EQUITY,
            description="ERP高 → 股票相对债券更有吸引力 → 股价上涨",
            causal_mechanism="ERP↑ → 股票期望收益率↑ → 债券相对吸引力↓ → 资金流向股市 → 股价↑",
            data_sources=["股票收益率", "债券收益率"],
            measurement_methods=["股票收益率 - 无风险收益率"],
            update_frequency="daily",
            reliability=0.80,
            confidence=0.72,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["premium", "risk_premium"],
        ))

        self.add_factor(CausalFactor(
            factor_id="hml_value_premium",
            name="HML价值溢价",
            category=FactorCategory.EQUITY_PREMIUM,
            asset_class=AssetClass.EQUITY,
            description="价值股（高B/M）→ 长期跑赢成长股",
            causal_mechanism="价值股被低估 → 估值回归 → 长期超额收益",
            data_sources=["财报", "行情"],
            measurement_methods=["高B/M组合 - 低B/M组合收益率"],
            update_frequency="monthly",
            reliability=0.75,
            confidence=0.68,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["premium", "value", "fama_french"],
        ))

        self.add_factor(CausalFactor(
            factor_id="smb_size_premium",
            name="SMB规模溢价",
            category=FactorCategory.EQUITY_PREMIUM,
            asset_class=AssetClass.EQUITY,
            description="小盘股 → 长期跑赢大盘股",
            causal_mechanism="小盘股流动性差 → 风险高 → 风险溢价高 → 长期超额收益",
            data_sources=["市值数据"],
            measurement_methods=["小盘股组合 - 大盘股组合收益率"],
            update_frequency="monthly",
            reliability=0.72,
            confidence=0.65,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["premium", "size", "fama_french"],
        ))

        self.add_factor(CausalFactor(
            factor_id="buyback_premium",
            name="回购溢价",
            category=FactorCategory.EQUITY_PREMIUM,
            asset_class=AssetClass.EQUITY,
            description="回购比例高 → EPS稀释减少 → 股价上涨",
            causal_mechanism="回购↑ → 股数↓ → EPS↑ → 每股价值↑ → 股价↑",
            data_sources=["回购公告", "股本数据"],
            measurement_methods=["回购股数 / 总股本"],
            update_frequency="quarterly",
            reliability=0.78,
            confidence=0.70,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["premium", "buyback"],
        ))

    def _initialize_commodity_factors(self):
        """初始化商品期货专属因素"""

        # ========================================
        # 1. 现货基本面与供需
        # ========================================

        self.add_factor(CausalFactor(
            factor_id="supply_demand_balance",
            name="供需平衡",
            category=FactorCategory.SUPPLY_DEMAND,
            asset_class=AssetClass.COMMODITY,
            description="供应短缺 → 价格上涨；供应过剩 → 价格下跌",
            causal_mechanism="供应<需求 → 库存下降 → 供不应求 → 价格上涨；反之亦然",
            data_sources=["库存数据", "产量数据", "需求数据"],
            measurement_methods=["库存消费比", "供需缺口"],
            update_frequency="monthly",
            reliability=0.92,
            confidence=0.88,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["fundamental", "supply_demand"],
        ))

        self.add_factor(CausalFactor(
            factor_id="basis",
            name="基差",
            category=FactorCategory.SUPPLY_DEMAND,
            asset_class=AssetClass.COMMODITY,
            description="基差（现货-期货）→ 反映供需紧张程度 → 价格预期",
            causal_mechanism="基差>0（现货升水）→ 现货紧张 → 价格上涨预期；基差<0 → 现货宽松 → 价格下跌预期",
            data_sources=["现货价格", "期货价格"],
            measurement_methods=["现货价格 - 期货价格"],
            update_frequency="daily",
            reliability=0.88,
            confidence=0.82,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["fundamental", "pricing", "basis"],
        ))

        self.add_factor(CausalFactor(
            factor_id="inventory_level",
            name="库存水平",
            category=FactorCategory.SUPPLY_DEMAND,
            asset_class=AssetClass.COMMODITY,
            description="库存高 → 供应充足 → 价格下跌；库存低 → 供应紧张 → 价格上涨",
            causal_mechanism="库存↑ → 供应充足 → 价格下行压力；库存↓ → 供应紧张 → 价格上行压力",
            data_sources=["库存数据", "交易所库存"],
            measurement_methods=["库存量", "库存消费比"],
            update_frequency="weekly",
            reliability=0.90,
            confidence=0.85,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["fundamental", "inventory", "supply"],
        ))

        self.add_factor(CausalFactor(
            factor_id="substitution_effect",
            name="替代品价差",
            category=FactorCategory.SUPPLY_DEMAND,
            asset_class=AssetClass.COMMODITY,
            description="替代品价差 → 需求转移 → 价格收敛",
            causal_mechanism="A商品价格↑ → 替代为B商品 → B需求↑ → B价格↑ → 价格收敛",
            data_sources=["现货价格", "替代关系"],
            measurement_methods=["价差", "替代弹性"],
            update_frequency="daily",
            reliability=0.82,
            confidence=0.75,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["fundamental", "substitution", "demand"],
        ))

        # ========================================
        # 2. 期货定价机制
        # ========================================

        self.add_factor(CausalFactor(
            factor_id="cost_of_carry",
            name="持有成本",
            category=FactorCategory.FUTURES_PRICING,
            asset_class=AssetClass.COMMODITY,
            description="持有成本高 → 期货价格高于现货（升水）",
            causal_mechanism="期货价格 = 现货价格 + 持有成本（仓储、保险、资金成本）",
            data_sources=["利率", "仓储费", "保险费"],
            measurement_methods=["持有成本模型"],
            update_frequency="daily",
            reliability=0.95,
            confidence=0.92,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["pricing", "cost_of_carry"],
        ))

        self.add_factor(CausalFactor(
            factor_id="storage_premium",
            name="存储成本溢价",
            category=FactorCategory.FUTURES_PRICING,
            asset_class=AssetClass.COMMODITY,
            description="存储成本高 → 升水大；存储便利 → 贴水可能",
            causal_mechanism="存储成本↑ → 持有成本↑ → 期货升水↑；存储便利（便利收益）→ 可能贴水",
            data_sources=["仓储费率", "库存数据"],
            measurement_methods=["仓储费", "库存便利收益"],
            update_frequency="monthly",
            reliability=0.85,
            confidence=0.78,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["pricing", "storage", "convenience_yield"],
        ))

        self.add_factor(CausalFactor(
            factor_id="convenience_yield",
            name="便利收益",
            category=FactorCategory.FUTURES_PRICING,
            asset_class=AssetClass.COMMODITY,
            description="现货紧张 → 便利收益高 → 期货贴水",
            causal_mechanism="现货紧张 → 持有现货的便利性↑ → 便利收益↑ → 期货价格<现货价格（贴水）",
            data_sources=["现货供需", "库存"],
            measurement_methods=["隐含便利收益 = 现货价格 - 期货价格 - 持有成本"],
            update_frequency="daily",
            reliability=0.80,
            confidence=0.72,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["pricing", "convenience_yield", "backwardation"],
        ))

        self.add_factor(CausalFactor(
            factor_id="location_premium",
            name="地区溢价",
            category=FactorCategory.FUTURES_PRICING,
            asset_class=AssetClass.COMMODITY,
            description="地区溢价 → 反映运输成本和地区供需差异",
            causal_mechanism="地区A需求旺盛/运输成本高 → 地区A现货价格↑ → 相对交割地溢价↑",
            data_sources=["地区现货价格", "运输成本"],
            measurement_methods=["地区价差"],
            update_frequency="daily",
            reliability=0.82,
            confidence=0.75,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["pricing", "location", "regional"],
        ))

        # ========================================
        # 3. 商品特有溢价
        # ========================================

        self.add_factor(CausalFactor(
            factor_id="development_risk_premium",
            name="开发风险溢价",
            category=FactorCategory.COMMODITY_PREMIUM,
            asset_class=AssetClass.COMMODITY,
            description="开发风险高（矿产开采/农作物种植）→ 远期溢价高",
            causal_mechanism="开发风险↑ → 未来供应不确定性↑ → 远期风险溢价↑",
            data_sources=["矿山/农场数据", "天气", "政策"],
            measurement_methods=["开发风险评分"],
            update_frequency="monthly",
            reliability=0.75,
            confidence=0.68,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["premium", "development_risk"],
        ))

        self.add_factor(CausalFactor(
            factor_id="seasonal_premium",
            name="季节性溢价",
            category=FactorCategory.COMMODITY_PREMIUM,
            asset_class=AssetClass.COMMODITY,
            description="季节性供需 → 季节性价格波动 → 季节性溢价",
            causal_mechanism="播种季节 → 需求↑ → 价格↑；收获季节 → 供应↑ → 价格↓",
            data_sources=["历史价格", "季节性数据"],
            measurement_methods=["季节性因子"],
            update_frequency="monthly",
            reliability=0.78,
            confidence=0.70,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1,
            tags=["premium", "seasonality"],
        ))

    def add_factor(self, factor: CausalFactor):
        """添加因素"""
        self.factors[factor.factor_id] = factor

    def get_factor(self, factor_id: str) -> Optional[CausalFactor]:
        """获取因素"""
        return self.factors.get(factor_id)

    def get_factors_by_category(
        self,
        category: FactorCategory
    ) -> List[CausalFactor]:
        """按类别获取因素"""
        return [
            f for f in self.factors.values()
            if f.category == category
        ]

    def get_factors_by_asset_class(
        self,
        asset_class: AssetClass
    ) -> List[CausalFactor]:
        """按资产类别获取因素"""
        if asset_class == AssetClass.ALL:
            return list(self.factors.values())
        return [
            f for f in self.factors.values()
            if f.asset_class in [asset_class, AssetClass.ALL]
        ]

    def search_factors(self, keyword: str) -> List[CausalFactor]:
        """搜索因素"""
        keyword = keyword.lower()
        return [
            f for f in self.factors.values()
            if (keyword in f.name.lower() or
                keyword in f.description.lower() or
                keyword in f.factor_id.lower() or
                any(keyword in tag.lower() for tag in f.tags))
        ]

    def generate_report(self) -> str:
        """生成报告"""
        report = []
        report.append("\n" + "="*80)
        report.append(" " * 25 + "因果因素库报告")
        report.append("="*80)

        # 总体统计
        total_factors = len(self.factors)
        by_category = {}
        by_asset_class = {}

        for factor in self.factors.values():
            cat = factor.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

            asset = factor.asset_class.value
            by_asset_class[asset] = by_asset_class.get(asset, 0) + 1

        report.append(f"\n📊 总体统计:")
        report.append(f"  因素总数: {total_factors}")

        report.append(f"\n📈 按类别分布:")
        for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {cat}: {count}")

        report.append(f"\n💼 按资产类别分布:")
        for asset, count in sorted(by_asset_class.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {asset}: {count}")

        # 高可靠性因素
        high_reliability = [f for f in self.factors.values() if f.reliability > 0.85]
        report.append(f"\n⭐ 高可靠性因素 (>0.85): {len(high_reliability)}个")
        for factor in sorted(high_reliability, key=lambda f: f.reliability, reverse=True)[:10]:
            report.append(f"  - {factor.name} ({factor.reliability:.2f})")

        return "\n".join(report)


# ============================================================================
# 工厂函数
# ============================================================================

def create_causal_factor_library() -> CausalFactorLibrary:
    """创建因果因素库"""
    return CausalFactorLibrary()


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 创建因素库
    library = create_causal_factor_library()

    # 生成报告
    print(library.generate_report())

    # 搜索示例
    print("\n\n🔍 搜索示例:")
    print("\n搜索'利率':")
    for factor in library.search_factors("利率"):
        print(f"  - {factor.name}: {factor.description[:60]}...")

    print("\n搜索'价值':")
    for factor in library.search_factors("价值"):
        print(f"  - {factor.name}: {factor.description[:60]}...")
