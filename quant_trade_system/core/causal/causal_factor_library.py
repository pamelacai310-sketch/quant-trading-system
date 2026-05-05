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


@dataclass
class QuantizedCausalFactor:
    """由因果关系量化而来的结构化因子定义。"""
    quant_factor_id: str
    source_factor_id: str
    name: str
    formula: str
    formula_family: str
    financial_meaning: str
    required_inputs: List[str]
    expected_sign: int
    lag_days: int
    category: FactorCategory
    asset_class: AssetClass
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        self._initialize_cross_asset_extensions()
        self._initialize_equity_extensions()
        self._initialize_commodity_extensions()
        self.quantized_factors: Dict[str, QuantizedCausalFactor] = self._build_quantized_factor_catalog()

    def _bulk_add_factor_specs(self, specs: List[Dict[str, Any]]) -> None:
        """批量添加因素规格。"""
        for spec in specs:
            payload = dict(spec)
            created_at = payload.pop("created_at", datetime.now())
            updated_at = payload.pop("updated_at", created_at)
            version = payload.pop("version", 1)
            self.add_factor(CausalFactor(
                created_at=created_at,
                updated_at=updated_at,
                version=version,
                **payload,
            ))

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

    def _initialize_cross_asset_extensions(self) -> None:
        """扩展跨资产共有因素。"""
        self._bulk_add_factor_specs([
            {
                "factor_id": "country_premium",
                "name": "地区/国别溢价",
                "category": FactorCategory.MACRO_POLICY,
                "asset_class": AssetClass.ALL,
                "description": "国家信用、制度质量与地缘环境差异会改变风险资产与商品的折价/溢价。",
                "causal_mechanism": "制度稳定性/国家风险差异 → 资本要求更高或更低的风险补偿 → 估值与远期风险溢价调整",
                "data_sources": ["主权评级", "CDS利差", "资本流动数据"],
                "measurement_methods": ["主权CDS", "国家风险评分", "资本净流入"],
                "update_frequency": "weekly",
                "reliability": 0.86,
                "confidence": 0.80,
                "tags": ["macro", "country", "risk_premium"],
            },
            {
                "factor_id": "fx_volatility",
                "name": "汇率波动",
                "category": FactorCategory.MACRO_POLICY,
                "asset_class": AssetClass.ALL,
                "description": "汇率大幅波动会通过输入成本、出口竞争力和美元计价链条传导到股票与商品。",
                "causal_mechanism": "汇率波动 ↑ → 现金流折算不确定性/进口成本变化/美元金融条件变化 → 资产估值与商品价格重定价",
                "data_sources": ["外汇市场数据", "DXY", "跨境结算数据"],
                "measurement_methods": ["滚动汇率波动率", "美元指数变动", "套保成本"],
                "update_frequency": "daily",
                "reliability": 0.88,
                "confidence": 0.82,
                "tags": ["macro", "fx", "usd_liquidity"],
            },
            {
                "factor_id": "fiscal_industry_policy",
                "name": "财政/行业政策干预",
                "category": FactorCategory.MACRO_POLICY,
                "asset_class": AssetClass.ALL,
                "description": "财政刺激、补贴、限产或产业扶持会直接改变需求曲线、供给弹性和盈利预期。",
                "causal_mechanism": "财政刺激/行业补贴/限产政策 → 需求或供给约束变化 → 盈利与商品平衡表变化 → 价格重估",
                "data_sources": ["财政预算", "行业政策文件", "补贴公告"],
                "measurement_methods": ["财政赤字率", "专项债投放", "政策事件评分"],
                "update_frequency": "monthly",
                "reliability": 0.87,
                "confidence": 0.81,
                "tags": ["macro", "policy", "industry"],
            },
            {
                "factor_id": "policy_risk",
                "name": "政策风险",
                "category": FactorCategory.MACRO_POLICY,
                "asset_class": AssetClass.ALL,
                "description": "政策不确定性上升会抬高贴现率并压缩风险偏好。",
                "causal_mechanism": "监管/政策路径不确定性 ↑ → 风险偏好 ↓ → 资本开支与估值压缩 → 资产价格承压",
                "data_sources": ["政策不确定性指数", "监管公告", "新闻文本"],
                "measurement_methods": ["EPU指数", "政策情绪评分", "事件跳变频率"],
                "update_frequency": "daily",
                "reliability": 0.84,
                "confidence": 0.79,
                "tags": ["macro", "policy_risk", "uncertainty"],
            },
            {
                "factor_id": "funding_conditions",
                "name": "资金面",
                "category": FactorCategory.MACRO_POLICY,
                "asset_class": AssetClass.ALL,
                "description": "资金宽松会抬升风险承担能力，资金紧张会压制股票和高杠杆商品头寸。",
                "causal_mechanism": "融资成本/可得性变化 → 杠杆资金与做市能力变化 → 风险资产需求与基差结构变化",
                "data_sources": ["回购利率", "社融数据", "融资融券余额"],
                "measurement_methods": ["隔夜回购利率", "融资余额变化", "社融增速"],
                "update_frequency": "daily",
                "reliability": 0.90,
                "confidence": 0.86,
                "tags": ["macro", "liquidity", "leverage"],
            },
            {
                "factor_id": "government_debt_overhang",
                "name": "政府负债与宏观因子耦合",
                "category": FactorCategory.MACRO_POLICY,
                "asset_class": AssetClass.ALL,
                "description": "高政府负债会改变利率、通胀与信用因子的相关结构。",
                "causal_mechanism": "政府负债高企 → 财政可持续性与发债压力变化 → 利率/通胀/信用因子共振增强 → 风险溢价结构改变",
                "data_sources": ["政府债务数据", "财政收支", "主权收益率"],
                "measurement_methods": ["债务/GDP", "财政赤字率", "期限利差"],
                "update_frequency": "quarterly",
                "reliability": 0.81,
                "confidence": 0.76,
                "tags": ["macro", "debt", "correlation"],
            },
            {
                "factor_id": "tax_premium",
                "name": "税收溢价",
                "category": FactorCategory.MACRO_POLICY,
                "asset_class": AssetClass.ALL,
                "description": "税率和税制差异会影响净现金流、套利约束与持有结构。",
                "causal_mechanism": "税负变化 → 税后回报变化 → 资本配置再平衡 → 资产估值与期限结构调整",
                "data_sources": ["税制政策", "上市公司税率", "商品进口税"],
                "measurement_methods": ["有效税率", "税收变动事件", "税后收益差"],
                "update_frequency": "quarterly",
                "reliability": 0.78,
                "confidence": 0.72,
                "tags": ["macro", "tax", "after_tax_return"],
            },
            {
                "factor_id": "market_maturity",
                "name": "市场成熟度",
                "category": FactorCategory.MICROSTRUCTURE,
                "asset_class": AssetClass.ALL,
                "description": "成熟市场通常有更稳定的信息扩散、风险定价和做市深度。",
                "causal_mechanism": "投资者结构与制度成熟度提升 → 信息扩散更快/治理更稳定 → 波动与错误定价下降",
                "data_sources": ["市场结构数据", "机构持有比例", "监管规则"],
                "measurement_methods": ["机构投资者占比", "日均成交额", "上市年限结构"],
                "update_frequency": "monthly",
                "reliability": 0.79,
                "confidence": 0.74,
                "tags": ["microstructure", "maturity", "efficiency"],
            },
            {
                "factor_id": "market_efficiency",
                "name": "市场效率和信息不对称",
                "category": FactorCategory.MICROSTRUCTURE,
                "asset_class": AssetClass.ALL,
                "description": "信息不对称越高，价格发现越慢，因果链传导的时滞和噪声越大。",
                "causal_mechanism": "信息不对称 ↑ → 逆向选择成本 ↑ → 价格发现效率 ↓ → 因果信号兑现更慢",
                "data_sources": ["订单簿", "公告反应", "分析师覆盖"],
                "measurement_methods": ["Amihud指标", "公告后漂移", "分析师覆盖度"],
                "update_frequency": "daily",
                "reliability": 0.85,
                "confidence": 0.80,
                "tags": ["microstructure", "efficiency", "asymmetry"],
            },
            {
                "factor_id": "turnover_activity",
                "name": "交易活跃度/换手率",
                "category": FactorCategory.MICROSTRUCTURE,
                "asset_class": AssetClass.ALL,
                "description": "活跃度提升通常伴随信息快速入价，也可能意味着拥挤交易上升。",
                "causal_mechanism": "换手率/成交额 ↑ → 信息反应加速或拥挤度上升 → 趋势强化或反转风险提升",
                "data_sources": ["成交量", "换手率", "持仓数据"],
                "measurement_methods": ["换手率", "成交额/自由流通市值", "持仓增量"],
                "update_frequency": "daily",
                "reliability": 0.86,
                "confidence": 0.82,
                "tags": ["microstructure", "turnover", "crowding"],
            },
            {
                "factor_id": "standardization_level",
                "name": "交易标准化程度",
                "category": FactorCategory.MICROSTRUCTURE,
                "asset_class": AssetClass.ALL,
                "description": "标准化合约与规则越清晰，流动性越易聚集，跨期和跨市场比较更稳定。",
                "causal_mechanism": "标准化程度 ↑ → 参与者理解成本 ↓/可替代性 ↑ → 流动性和价格一致性提升",
                "data_sources": ["交易所规则", "合约说明书"],
                "measurement_methods": ["合约标准化评分", "交割规则复杂度"],
                "update_frequency": "quarterly",
                "reliability": 0.77,
                "confidence": 0.70,
                "tags": ["microstructure", "contract_design"],
            },
            {
                "factor_id": "minimum_trading_unit",
                "name": "最小交易单位",
                "category": FactorCategory.MICROSTRUCTURE,
                "asset_class": AssetClass.ALL,
                "description": "最小交易单位越大，散户可参与性越弱，流动性和冲击成本结构会不同。",
                "causal_mechanism": "交易单位约束变化 → 可参与资金池变化 → 订单拆分与冲击成本变化",
                "data_sources": ["交易所规则"],
                "measurement_methods": ["最小手数", "最小变动价位对应名义金额"],
                "update_frequency": "quarterly",
                "reliability": 0.74,
                "confidence": 0.68,
                "tags": ["microstructure", "lot_size"],
            },
            {
                "factor_id": "minimum_holding_period",
                "name": "最低持有期",
                "category": FactorCategory.MICROSTRUCTURE,
                "asset_class": AssetClass.ALL,
                "description": "最低持有期会改变资金周转、套利速度和行为风格。",
                "causal_mechanism": "持有期限制 ↑ → 短线套利受限 → 定价偏差修复更慢/波动形态变化",
                "data_sources": ["监管规则", "基金合约"],
                "measurement_methods": ["法定持有期", "产品锁定期"],
                "update_frequency": "quarterly",
                "reliability": 0.73,
                "confidence": 0.67,
                "tags": ["microstructure", "holding_period"],
            },
            {
                "factor_id": "price_limit_rule",
                "name": "涨跌幅限制",
                "category": FactorCategory.MICROSTRUCTURE,
                "asset_class": AssetClass.ALL,
                "description": "涨跌停制度会改变波动释放路径、流动性真空和价格发现节奏。",
                "causal_mechanism": "价格限制触发 → 流动性冻结/订单堆积 → 波动推迟释放 → 因果兑现路径改变",
                "data_sources": ["交易所规则", "盘口数据"],
                "measurement_methods": ["涨跌停幅度", "封板持续时间"],
                "update_frequency": "daily",
                "reliability": 0.84,
                "confidence": 0.78,
                "tags": ["microstructure", "limit_rule", "volatility"],
            },
            {
                "factor_id": "matching_rule",
                "name": "撮合规则",
                "category": FactorCategory.MICROSTRUCTURE,
                "asset_class": AssetClass.ALL,
                "description": "撮合优先级和集合竞价机制决定订单簿中的信息传导路径。",
                "causal_mechanism": "价格优先/时间优先/集合竞价差异 → 成交概率与冲击路径变化 → 交易信号执行偏差变化",
                "data_sources": ["交易所规则", "逐笔成交数据"],
                "measurement_methods": ["竞价规则分类", "成交等待时间"],
                "update_frequency": "quarterly",
                "reliability": 0.76,
                "confidence": 0.70,
                "tags": ["microstructure", "matching_engine"],
            },
            {
                "factor_id": "order_type_flexibility",
                "name": "订单类型丰富度",
                "category": FactorCategory.MICROSTRUCTURE,
                "asset_class": AssetClass.ALL,
                "description": "订单类型越丰富，执行策略越能细分风险，但也可能加剧选择权不对称。",
                "causal_mechanism": "订单类型选择空间 ↑ → 执行精度 ↑/逆向选择复杂度 ↑ → 冲击成本结构改变",
                "data_sources": ["交易所规则", "经纪商接口"],
                "measurement_methods": ["订单类型数量", "冰山/TWAP/VWAP可用性"],
                "update_frequency": "quarterly",
                "reliability": 0.75,
                "confidence": 0.69,
                "tags": ["microstructure", "execution"],
            },
            {
                "factor_id": "margin_mechanism",
                "name": "保证金管理",
                "category": FactorCategory.MICROSTRUCTURE,
                "asset_class": AssetClass.ALL,
                "description": "保证金水平变化会直接影响杠杆上限、平仓压力与拥挤交易。",
                "causal_mechanism": "保证金上调 → 杠杆资金被动收缩 → 强平/减仓 ↑ → 价格波动放大",
                "data_sources": ["交易所保证金通知", "融资融券数据"],
                "measurement_methods": ["初始保证金率", "维持保证金率", "融资折算率"],
                "update_frequency": "daily",
                "reliability": 0.89,
                "confidence": 0.84,
                "tags": ["microstructure", "margin", "leverage"],
            },
            {
                "factor_id": "volatility_arbitrage_premium",
                "name": "波动率套利溢价",
                "category": FactorCategory.QUANT_STRATEGY,
                "asset_class": AssetClass.ALL,
                "description": "隐含与实现波动率偏离会吸引波动率卖方/买方资本，从而改变横截面定价。",
                "causal_mechanism": "隐含波动率偏高/偏低 → 套利资本入场 → 波动率曲面与相关资产风险补偿收敛",
                "data_sources": ["期权隐含波动率", "历史波动率"],
                "measurement_methods": ["IV-RV差", "波动率期限结构"],
                "update_frequency": "daily",
                "reliability": 0.79,
                "confidence": 0.73,
                "tags": ["quant", "vol_arb", "options"],
            },
            {
                "factor_id": "drawdown_repair_days_avg_7d",
                "name": "至少持有7天后的平均回撤修复天数",
                "category": FactorCategory.QUANT_STRATEGY,
                "asset_class": AssetClass.ALL,
                "description": "衡量因果链兑现过程中，深度回撤修复所需的平均时间。",
                "causal_mechanism": "回撤修复越慢 → 说明信号兑现路径更曲折/拥挤更强 → 资金耐心与风控阈值需要调整",
                "data_sources": ["历史收益率", "交易记录"],
                "measurement_methods": ["持有7天后MAE修复时长均值"],
                "update_frequency": "weekly",
                "reliability": 0.77,
                "confidence": 0.71,
                "tags": ["quant", "drawdown", "path_dependency"],
            },
            {
                "factor_id": "drawdown_repair_days_max",
                "name": "最长回撤修复天数",
                "category": FactorCategory.QUANT_STRATEGY,
                "asset_class": AssetClass.ALL,
                "description": "极端修复时长反映了因果信号面对制度切换时的脆弱度。",
                "causal_mechanism": "最长修复期越长 → 表明信号在极端状态下更易失效或传导阻塞",
                "data_sources": ["历史收益率", "交易记录"],
                "measurement_methods": ["最长净值回撤修复期"],
                "update_frequency": "weekly",
                "reliability": 0.76,
                "confidence": 0.70,
                "tags": ["quant", "drawdown", "tail_risk"],
            },
            {
                "factor_id": "risk_adjusted_return",
                "name": "风险调整收益指标",
                "category": FactorCategory.QUANT_STRATEGY,
                "asset_class": AssetClass.ALL,
                "description": "风险调整后的收益能更真实地表达因果链的可交易价值，而不是只看名义收益。",
                "causal_mechanism": "风险调整收益更高 → 单位风险下的因果兑现质量更好 → 更适合纳入风险预算",
                "data_sources": ["收益率序列", "回撤数据"],
                "measurement_methods": ["Sharpe", "Sortino", "Calmar"],
                "update_frequency": "daily",
                "reliability": 0.91,
                "confidence": 0.87,
                "tags": ["quant", "risk_adjusted_return", "portfolio"],
            },
        ])

    def _initialize_equity_extensions(self) -> None:
        """扩展股票专属因素。"""
        self._bulk_add_factor_specs([
            {
                "factor_id": "industry_company_characteristics",
                "name": "行业和公司特性",
                "category": FactorCategory.FUNDAMENTAL,
                "asset_class": AssetClass.EQUITY,
                "description": "不同行业资本密集度、景气度和竞争格局决定同样宏观冲击下的收益分化。",
                "causal_mechanism": "行业结构/商业模式差异 → 对增长、利率和成本冲击的敏感度不同 → 股票收益横截面分化",
                "data_sources": ["行业分类", "公司年报", "卖方研究"],
                "measurement_methods": ["行业Beta", "资本开支强度", "商业模式标签"],
                "update_frequency": "quarterly",
                "reliability": 0.87,
                "confidence": 0.82,
                "tags": ["equity", "industry", "business_model"],
            },
            {
                "factor_id": "profitability_quality",
                "name": "盈利能力质量",
                "category": FactorCategory.FUNDAMENTAL,
                "asset_class": AssetClass.EQUITY,
                "description": "高质量盈利更容易穿越周期，现金流折现的稳定性更高。",
                "causal_mechanism": "利润率/现金转化率高且稳定 → 自由现金流可见性更强 → 估值折价下降",
                "data_sources": ["财报", "现金流量表"],
                "measurement_methods": ["毛利率", "净利率", "经营现金流/净利润"],
                "update_frequency": "quarterly",
                "reliability": 0.90,
                "confidence": 0.86,
                "tags": ["equity", "quality", "profitability"],
            },
            {
                "factor_id": "industry_position",
                "name": "行业地位",
                "category": FactorCategory.FUNDAMENTAL,
                "asset_class": AssetClass.EQUITY,
                "description": "龙头公司通常拥有更强定价权、融资能力和供应链韧性。",
                "causal_mechanism": "行业份额/品牌力/渠道优势 ↑ → 盈利韧性与议价能力 ↑ → 估值中枢提升",
                "data_sources": ["行业报告", "市占率统计"],
                "measurement_methods": ["市场份额", "CR3/CR5", "品牌溢价评分"],
                "update_frequency": "quarterly",
                "reliability": 0.85,
                "confidence": 0.80,
                "tags": ["equity", "leader", "competitive_advantage"],
            },
            {
                "factor_id": "internal_risk_management",
                "name": "企业风险管理能力",
                "category": FactorCategory.FUNDAMENTAL,
                "asset_class": AssetClass.EQUITY,
                "description": "企业的套保、资本配置和治理能力会显著改变极端情景下的现金流损失。",
                "causal_mechanism": "风控治理更强 → 经营波动与尾部损失更小 → 估值折价收窄",
                "data_sources": ["年报", "治理披露", "套保公告"],
                "measurement_methods": ["套保覆盖率", "董事会独立性", "风险事件频率"],
                "update_frequency": "quarterly",
                "reliability": 0.79,
                "confidence": 0.73,
                "tags": ["equity", "governance", "risk_management"],
            },
            {
                "factor_id": "demand_risk_guarantee",
                "name": "需求风险担保",
                "category": FactorCategory.FUNDAMENTAL,
                "asset_class": AssetClass.EQUITY,
                "description": "最低交通量、保底采购等协议会直接改变现金流下限和估值弹性。",
                "causal_mechanism": "保底条款存在 → 下行情景现金流更稳 → 折现率下降/估值提升",
                "data_sources": ["合同披露", "REITs/基建公告"],
                "measurement_methods": ["保底覆盖比例", "合同剩余期限"],
                "update_frequency": "quarterly",
                "reliability": 0.76,
                "confidence": 0.70,
                "tags": ["equity", "contract", "downside_protection"],
            },
            {
                "factor_id": "refinancing_cost",
                "name": "再融资成本",
                "category": FactorCategory.VALUATION,
                "asset_class": AssetClass.EQUITY,
                "description": "再融资成本上升会压缩股东自由现金流和项目IRR。",
                "causal_mechanism": "债务/股权再融资成本 ↑ → 贴现率/WACC ↑ → 项目净现值 ↓ → 股价承压",
                "data_sources": ["债券利差", "定增/配股数据", "贷款利率"],
                "measurement_methods": ["项目债利差", "WACC", "再融资利差"],
                "update_frequency": "monthly",
                "reliability": 0.88,
                "confidence": 0.83,
                "tags": ["equity", "refinancing", "discount_rate"],
            },
            {
                "factor_id": "growth_premium",
                "name": "成长溢价",
                "category": FactorCategory.EQUITY_PREMIUM,
                "asset_class": AssetClass.EQUITY,
                "description": "高成长预期公司在流动性宽松或技术扩散期常获得额外估值溢价。",
                "causal_mechanism": "长期成长可见性 ↑ → 远期现金流权重上升 → 在低贴现率环境下估值弹性更高",
                "data_sources": ["一致预期", "行业景气度", "研发投入"],
                "measurement_methods": ["收入增速", "EPS长期增速", "PEG"],
                "update_frequency": "monthly",
                "reliability": 0.78,
                "confidence": 0.73,
                "tags": ["equity", "growth", "valuation"],
            },
        ])

    def _initialize_commodity_extensions(self) -> None:
        """扩展商品期货专属因素。"""
        self._bulk_add_factor_specs([
            {
                "factor_id": "import_export_balance",
                "name": "进出口贸易差额",
                "category": FactorCategory.SUPPLY_DEMAND,
                "asset_class": AssetClass.COMMODITY,
                "description": "净进口或净出口结构变化会改变本地供需平衡与地区升贴水。",
                "causal_mechanism": "进口依赖度/出口强度变化 → 本地库存与供需缺口变化 → 现货升贴水和期货价格重估",
                "data_sources": ["海关数据", "贸易统计", "港口库存"],
                "measurement_methods": ["净进口量", "贸易差额/GDP", "到港节奏"],
                "update_frequency": "monthly",
                "reliability": 0.86,
                "confidence": 0.80,
                "tags": ["commodity", "trade", "regional_balance"],
            },
            {
                "factor_id": "strategic_reserves",
                "name": "商品储备",
                "category": FactorCategory.SUPPLY_DEMAND,
                "asset_class": AssetClass.COMMODITY,
                "description": "商业库存之外的战略储备释放或收储会改变供需缺口的缓冲能力。",
                "causal_mechanism": "储备投放/收储 → 有效供给变化 → 现货紧张度与远期曲线调整",
                "data_sources": ["国家储备公告", "行业库存"],
                "measurement_methods": ["储备投放量", "储备覆盖天数"],
                "update_frequency": "monthly",
                "reliability": 0.83,
                "confidence": 0.77,
                "tags": ["commodity", "reserve", "buffer_stock"],
            },
            {
                "factor_id": "opportunity_cost_spread",
                "name": "机会成本/替代品价差",
                "category": FactorCategory.SUPPLY_DEMAND,
                "asset_class": AssetClass.COMMODITY,
                "description": "不同品种间的替代与切换会改变需求分流和利润传导。",
                "causal_mechanism": "替代品价差扩大 → 下游切换需求 → 相对强弱与跨品种套利收敛",
                "data_sources": ["现货价差", "产业链价差"],
                "measurement_methods": ["跨品种价差", "利润比价", "替代弹性评分"],
                "update_frequency": "daily",
                "reliability": 0.84,
                "confidence": 0.78,
                "tags": ["commodity", "substitution", "spread"],
            },
        ])

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
                keyword in f.causal_mechanism.lower() or
                keyword in f.factor_id.lower() or
                any(keyword in tag.lower() for tag in f.tags))
        ]

    def get_cross_asset_factors(self) -> List[CausalFactor]:
        """获取跨资产共有因素。"""
        return [
            factor for factor in self.factors.values()
            if factor.asset_class == AssetClass.ALL
        ]

    def get_factor_ids(self) -> List[str]:
        """获取全部因素ID。"""
        return sorted(self.factors.keys())

    def get_quantized_factor_ids(self) -> List[str]:
        """获取全部量化因子ID。"""
        return sorted(self.quantized_factors.keys())

    def get_quantized_factor(self, quant_factor_id: str) -> Optional[QuantizedCausalFactor]:
        """获取单个量化因子定义。"""
        return self.quantized_factors.get(quant_factor_id)

    def get_quantized_factor_catalog(self) -> Dict[str, QuantizedCausalFactor]:
        """获取量化因子定义目录。"""
        return dict(self.quantized_factors)

    def get_factor_coverage_summary(self) -> Dict[str, Any]:
        """获取因果知识库覆盖摘要。"""
        summary = {
            "total_factors": len(self.factors),
            "quantized_factor_count": len(self.quantized_factors),
            "cross_asset_factors": 0,
            "equity_only_factors": 0,
            "commodity_only_factors": 0,
            "by_category": {},
            "high_reliability_factor_ids": [],
        }

        for factor in self.factors.values():
            summary["by_category"][factor.category.value] = (
                summary["by_category"].get(factor.category.value, 0) + 1
            )
            if factor.asset_class == AssetClass.ALL:
                summary["cross_asset_factors"] += 1
            elif factor.asset_class == AssetClass.EQUITY:
                summary["equity_only_factors"] += 1
            elif factor.asset_class == AssetClass.COMMODITY:
                summary["commodity_only_factors"] += 1

        summary["high_reliability_factor_ids"] = sorted([
            factor.factor_id for factor in self.factors.values()
            if factor.reliability >= 0.85
        ])
        return summary

    def _build_quantized_factor_catalog(self) -> Dict[str, QuantizedCausalFactor]:
        """把语义化因果因素映射为可计算量化因子定义。"""
        catalog: Dict[str, QuantizedCausalFactor] = {}
        for factor in self.factors.values():
            family = self._infer_formula_family(factor)
            formula, inputs, expected_sign = self._formula_template_for_family(factor, family)
            quant_factor_id = f"causal_quant_{factor.factor_id}"
            catalog[quant_factor_id] = QuantizedCausalFactor(
                quant_factor_id=quant_factor_id,
                source_factor_id=factor.factor_id,
                name=f"{factor.name}量化因子",
                formula=formula,
                formula_family=family,
                financial_meaning=f"{factor.name}: {factor.causal_mechanism}",
                required_inputs=inputs,
                expected_sign=expected_sign,
                lag_days=max(1, self._default_lag_for_frequency(factor.update_frequency)),
                category=factor.category,
                asset_class=factor.asset_class,
                metadata={
                    "description": factor.description,
                    "measurement_methods": list(factor.measurement_methods),
                    "reliability": factor.reliability,
                    "confidence": factor.confidence,
                    "tags": list(factor.tags),
                },
            )
        return catalog

    def _infer_formula_family(self, factor: CausalFactor) -> str:
        """基于因果标签推断量化公式族。"""
        text = " ".join(
            [
                factor.factor_id,
                factor.name,
                factor.description,
                factor.causal_mechanism,
                " ".join(factor.tags),
            ]
        ).lower()
        if any(token in text for token in ["interest", "rate", "re融资", "贴现", "carry"]):
            return "rate_discount"
        if any(token in text for token in ["inflation", "cpi", "ppi", "gold", "precious"]):
            return "inflation_real_asset"
        if any(token in text for token in ["monetary", "liquidity", "资金面", "policy", "财政"]):
            return "liquidity_policy"
        if any(token in text for token in ["institution", "ownership", "holder", "microstructure", "flow"]):
            return "participation_flow"
        if any(token in text for token in ["sentiment", "risk_preference", "risk aversion", "情绪"]):
            return "sentiment_regime"
        if any(token in text for token in ["volatility", "drawdown", "repair", "tail", "crisis"]):
            return "volatility_tail"
        if any(token in text for token in ["momentum", "trend", "leader", "rs", "strength"]):
            return "trend_strength"
        if any(token in text for token in ["valuation", "pb", "pe", "hml", "smb", "growth", "buyback"]):
            return "valuation_quality"
        if any(token in text for token in ["roic", "eps", "profit", "rating", "governance", "demand guarantee"]):
            return "fundamental_quality"
        if any(token in text for token in ["inventory", "storage", "supply", "demand", "reserve", "trade balance"]):
            return "supply_demand_balance"
        if any(token in text for token in ["basis", "convenience", "location premium", "basis", "cost of carry"]):
            return "carry_curve"
        if any(token in text for token in ["season", "seasonal"]):
            return "seasonality"
        return "broad_causal"

    def _formula_template_for_family(
        self,
        factor: CausalFactor,
        family: str,
    ) -> Tuple[str, List[str], int]:
        """返回公式字符串、必需输入和预期方向。"""
        default_inputs = ["close", "high", "low", "volume"]
        templates: Dict[str, Tuple[str, List[str], int]] = {
            "rate_discount": (
                "-zscore(delta(rate_proxy_5d)) + 0.35*zscore(cashflow_yield_proxy) - 0.20*zscore(duration_sensitive_drawdown)",
                ["close", "rate_proxy", "cashflow_yield_proxy"],
                -1,
            ),
            "inflation_real_asset": (
                "zscore(delta(inflation_proxy_20d)) + 0.5*zscore(real_asset_momentum_20d) - 0.25*zscore(real_rate_proxy)",
                ["close", "inflation_proxy", "real_rate_proxy"],
                1,
            ),
            "liquidity_policy": (
                "zscore(liquidity_growth_20d - funding_stress_20d) + 0.25*zscore(volume_participation_20d)",
                ["close", "volume", "liquidity_growth_20d"],
                1,
            ),
            "participation_flow": (
                "zscore(volume_participation_20d) - 0.5*zscore(spread_proxy_20d) - 0.25*zscore(realized_vol_20d)",
                ["close", "high", "low", "volume"],
                1,
            ),
            "sentiment_regime": (
                "zscore(short_term_momentum_5d) + 0.5*zscore(volume_surprise_5d) - 0.5*zscore(downside_vol_20d)",
                ["close", "volume"],
                1,
            ),
            "volatility_tail": (
                "-zscore(realized_vol_20d + downside_vol_20d) + 0.35*zscore(drawdown_repair_proxy)",
                ["close", "high", "low"],
                -1,
            ),
            "trend_strength": (
                "zscore(momentum_20d / (realized_vol_20d + 1e-6)) + 0.4*zscore(trend_r2_20d)",
                ["close"],
                1,
            ),
            "valuation_quality": (
                "-zscore(valuation_multiple_proxy) + 0.6*zscore(cashflow_yield_proxy) + 0.2*zscore(relative_strength_20d)",
                ["close", "valuation_multiple_proxy"],
                1,
            ),
            "fundamental_quality": (
                "zscore(profitability_proxy) + 0.45*zscore(stability_proxy) + 0.25*zscore(relative_strength_20d)",
                ["close", "volume", "profitability_proxy"],
                1,
            ),
            "supply_demand_balance": (
                "zscore((demand_proxy_20d - supply_proxy_20d) / (abs(supply_proxy_20d) + 1e-6)) + 0.25*zscore(inventory_tightness_proxy)",
                ["close", "volume", "inventory_proxy"],
                1,
            ),
            "carry_curve": (
                "zscore(basis_proxy_20d) + 0.4*zscore(term_structure_proxy_20d) - 0.2*zscore(storage_pressure_proxy)",
                ["close", "high", "low"],
                1,
            ),
            "seasonality": (
                "zscore(seasonal_return_same_period) + 0.3*zscore(calendar_flow_proxy)",
                ["close", "volume"],
                1,
            ),
            "broad_causal": (
                "zscore(relative_strength_20d) + 0.35*zscore(volume_participation_20d) - 0.25*zscore(drawdown_20d)",
                default_inputs,
                1,
            ),
        }
        formula, inputs, expected_sign = templates.get(family, templates["broad_causal"])
        if factor.category in {FactorCategory.VALUATION, FactorCategory.EQUITY_PREMIUM}:
            expected_sign = 1
        elif factor.category == FactorCategory.MACRO_POLICY and "interest" in factor.factor_id:
            expected_sign = -1
        elif factor.category in {FactorCategory.SUPPLY_DEMAND, FactorCategory.FUTURES_PRICING} and "storage" in factor.factor_id:
            expected_sign = -1
        return formula, inputs, expected_sign

    def _default_lag_for_frequency(self, frequency: str) -> int:
        """把更新频率转成默认滞后天数。"""
        freq = frequency.lower()
        if "daily" in freq:
            return 1
        if "weekly" in freq:
            return 5
        if "monthly" in freq:
            return 21
        if "quarter" in freq:
            return 63
        return 5

    def generate_report(self) -> str:
        """生成报告"""
        report = []
        report.append("\n" + "="*80)
        report.append(" " * 25 + "因果因素库报告")
        report.append("="*80)

        summary = self.get_factor_coverage_summary()
        total_factors = summary["total_factors"]
        by_category = summary["by_category"]
        by_asset_class = {
            "cross_asset": summary["cross_asset_factors"],
            "equity_only": summary["equity_only_factors"],
            "commodity_only": summary["commodity_only_factors"],
        }

        report.append(f"\n📊 总体统计:")
        report.append(f"  因素总数: {total_factors}")
        report.append(f"  量化因子总数: {summary['quantized_factor_count']}")
        report.append(f"  跨资产共有因素: {summary['cross_asset_factors']}")
        report.append(f"  股票专属因素: {summary['equity_only_factors']}")
        report.append(f"  商品期货专属因素: {summary['commodity_only_factors']}")

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
