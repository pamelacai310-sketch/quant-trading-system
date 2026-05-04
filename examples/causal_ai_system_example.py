"""
因果AI量化系统 - 使用示例

展示如何构建和使用自迭代的因果AI量化交易系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quant_trade_system.core.causal import (
    CausalFactorLibrary,
    FactorCategory,
    AssetClass,
)


# ============================================================================
# 示例1：创建因果因素库
# ============================================================================

def example1_create_factor_library():
    """示例1：创建因果因素库"""
    print("\n" + "="*80)
    print("示例1：创建因果因素库")
    print("="*80)

    # 创建因果因素库
    library = CausalFactorLibrary()

    # 查看报告
    print(library.generate_report())

    # 按类别查询
    print("\n\n📊 股票专属因素:")
    equity_factors = library.get_factors_by_asset_class(AssetClass.EQUITY)
    for factor in equity_factors[:5]:
        print(f"  - {factor.name} ({factor.category.value})")
        print(f"    机制: {factor.causal_mechanism[:60]}...")

    print("\n\n📊 商品期货专属因素:")
    commodity_factors = library.get_factors_by_asset_class(AssetClass.COMMODITY)
    for factor in commodity_factors[:5]:
        print(f"  - {factor.name} ({factor.category.value})")
        print(f"    机制: {factor.causal_mechanism[:60]}...")


# ============================================================================
# 示例2：搜索因果因素
# ============================================================================

def example2_search_factors():
    """示例2：搜索因果因素"""
    print("\n" + "="*80)
    print("示例2：搜索因果因素")
    print("="*80)

    library = CausalFactorLibrary()

    # 搜索"利率"
    print("\n🔍 搜索'利率':")
    for factor in library.search_factors("利率"):
        print(f"  - {factor.name}")
        print(f"    描述: {factor.description}")
        print(f"    可靠性: {factor.reliability:.2f}")

    # 搜索"供需"
    print("\n🔍 搜索'供需':")
    for factor in library.search_factors("供需"):
        print(f"  - {factor.name}")
        print(f"    描述: {factor.description}")

    # 搜索"溢价"
    print("\n🔍 搜索'溢价':")
    for factor in library.search_factors("溢价")[:5]:
        print(f"  - {factor.name} ({factor.asset_class.value})")


# ============================================================================
# 示例3：构建因果知识图谱
# ============================================================================

def example3_build_causal_graph():
    """示例3：构建因果知识图谱"""
    print("\n" + "="*80)
    print("示例3：构建因果知识图谱")
    print("="*80)

    from quant_trade_system.core.causal.causal_factor_library import CausalEdge, CausalType

    library = CausalFactorLibrary()

    # 定义因果边（示例）
    causal_edges = [
        # 利率 → 股价（负向）
        CausalEdge(
            edge_id="interest_rate_stock_price",
            source_factor_id="interest_rate_premium",
            target_factor_id="valuation_level",
            causal_type=CausalType.DIRECT_NEGATIVE,
            causal_strength=0.85,
            lag_days=7,
            confidence=0.88,
            direction="forward",
            mechanism="利率↑ → 贴现率↑ → 现值↓ → 股价↓",
            evidence=[],
            market_regime="normal",
            created_at=datetime.now(),
        ),

        # 货币政策 → 流动性 → 股价（间接因果）
        CausalEdge(
            edge_id="monetary_policy_liquidity",
            source_factor_id="monetary_policy",
            target_factor_id="liquidity",
            causal_type=CausalType.DIRECT_POSITIVE,
            causal_strength=0.90,
            lag_days=14,
            confidence=0.85,
            direction="forward",
            mechanism="货币政策宽松 → 流动性充裕",
            evidence=[],
            market_regime="all",
            created_at=datetime.now(),
        ),

        # 供需平衡 → 商品价格
        CausalEdge(
            edge_id="supply_demand_commodity_price",
            source_factor_id="supply_demand_balance",
            target_factor_id="basis",
            causal_type=CausalType.DIRECT_POSITIVE,
            causal_strength=0.92,
            lag_days=5,
            confidence=0.88,
            direction="forward",
            mechanism="供应<需求 → 库存↓ → 基差↑ → 价格↑",
            evidence=[],
            market_regime="all",
            created_at=datetime.now(),
        ),

        # 库存 → 基差
        CausalEdge(
            edge_id="inventory_basis",
            source_factor_id="inventory_level",
            target_factor_id="basis",
            causal_type=CausalType.DIRECT_NEGATIVE,
            causal_strength=0.88,
            lag_days=3,
            confidence=0.82,
            direction="forward",
            mechanism="库存↑ → 供应充足 → 基差↓",
            evidence=[],
            market_regime="normal",
            created_at=datetime.now(),
        ),
    ]

    # 打印因果链
    print("\n🔗 因果链示例:")
    for edge in causal_edges:
        source = library.get_factor(edge.source_factor_id)
        target = library.get_factor(edge.target_factor_id)

        if source and target:
            type_str = "→" if edge.causal_type == CausalType.DIRECT_POSITIVE else "↓"
            print(f"\n  {source.name} {type_str} {target.name}")
            print(f"    因果强度: {edge.causal_strength:.2f}")
            print(f"    滞后天数: {edge.lag_days}天")
            print(f"    机制: {edge.mechanism}")


# ============================================================================
# 示例4：应用因果分析到交易策略
# ============================================================================

def example4_apply_to_trading():
    """示例4：应用因果分析到交易策略"""
    print("\n" + "="*80)
    print("示例4：应用因果分析到交易策略")
    print("="*80)

    library = CausalFactorLibrary()

    # 模拟市场状态分析
    print("\n📊 当前市场状态（假设）:")
    market_state = {
        "interest_rate": 0.05,      # 利率5%
        "gdp_growth": 0.06,         # GDP增长6%
        "inflation": 0.03,          # 通胀3%
        "monetary_policy": "宽松",   # 货币政策宽松
        "volatility": 0.20,         # 波动率20%
    }

    for key, value in market_state.items():
        print(f"  {key}: {value}")

    # 因果推理：预测影响
    print("\n🔮 因果推理预测:")

    # 推理1：利率上升 → 股价下跌
    interest_rate_factor = library.search_factors("利率")[0]
    print(f"\n1. {interest_rate_factor.name}的影响:")
    print(f"   当前: {market_state['interest_rate']*100:.1f}%")
    print(f"   因果机制: {interest_rate_factor.causal_mechanism}")
    print(f"   预测: 利率上升 → 高估值股票承压")

    # 推理2：GDP增长 → 企业盈利
    gdp_factor = library.search_factors("经济增长")[0]
    print(f"\n2. {gdp_factor.name}的影响:")
    print(f"   当前: {market_state['gdp_growth']*100:.1f}%")
    print(f"   因果机制: {gdp_factor.causal_mechanism}")
    print(f"   预测: 经济增长 → 周期性股票受益")

    # 推理3：货币政策 → 流动性
    monetary_factor = library.search_factors("货币")[0]
    print(f"\n3. {monetary_factor.name}的影响:")
    print(f"   当前: {market_state['monetary_policy']}")
    print(f"   因果机制: {monetary_factor.causal_mechanism}")
    print(f"   预测: 宽松政策 → 流动性充裕 → 风险资产受益")

    # 综合建议
    print("\n💡 综合交易建议:")
    print("  1. 利率上升 → 减少高估值成长股配置")
    print("  2. GDP增长 → 增加周期性股票配置（原材料、工业）")
    print("  3. 宽松政策 → 增加风险资产配置（股票、商品）")
    print("  4. 通胀温和 → 关注通胀受益资产（大宗商品、REITs）")


# ============================================================================
# 示例5：构建因果驱动的交易信号
# ============================================================================

def example5_causal_trading_signals():
    """示例5：构建因果驱动的交易信号"""
    print("\n" + "="*80)
    print("示例5：构建因果驱动的交易信号")
    print("="*80)

    library = CausalFactorLibrary()

    # 模拟因子变化
    factor_changes = {
        "monetary_policy": +0.2,      # 货币政策宽松（+0.2）
        "gdp_growth": +0.1,           # GDP增长加速（+0.1）
        "inflation_premium": +0.05,   # 通胀上升（+0.05）
        "interest_rate_premium": -0.1, # 利率下降（-0.1）
    }

    print("\n📈 因子变化:")
    for factor_id, change in factor_changes.items():
        factor = library.get_factor(factor_id)
        if factor:
            direction = "↑" if change > 0 else "↓"
            print(f"  {factor.name} {direction} {abs(change):.1%}")

    # 因果推理：生成交易信号
    print("\n🎯 因果驱动的交易信号:")

    signals = []

    # 信号1：货币政策宽松 → 做多股票
    if factor_changes.get("monetary_policy", 0) > 0.1:
        signals.append({
            "asset": "股票",
            "direction": "做多",
            "confidence": 0.85,
            "reason": "货币政策宽松 → 流动性充裕 → 股价上涨",
            "target": "周期性股票、金融股",
        })

    # 信号2：GDP增长加速 → 做多大宗商品
    if factor_changes.get("gdp_growth", 0) > 0.05:
        signals.append({
            "asset": "大宗商品",
            "direction": "做多",
            "confidence": 0.82,
            "reason": "GDP增长 → 需求上升 → 商品价格上涨",
            "target": "铜、原油、螺纹钢",
        })

    # 信号3：利率下降 → 做多成长股
    if factor_changes.get("interest_rate_premium", 0) < -0.05:
        signals.append({
            "asset": "成长股",
            "direction": "做多",
            "confidence": 0.80,
            "reason": "利率下降 → 贴现率下降 → 成长股估值提升",
            "target": "科技股、生物医药",
        })

    # 信号4：通胀上升 → 做多通胀受益资产
    if factor_changes.get("inflation_premium", 0) > 0.03:
        signals.append({
            "asset": "通胀受益资产",
            "direction": "做多",
            "confidence": 0.75,
            "reason": "通胀上升 → 实际利率下降 → 抗通胀资产受益",
            "target": "黄金、大宗商品、REITs",
        })

    # 打印信号
    for i, signal in enumerate(signals, 1):
        print(f"\n  信号{i}: {signal['asset']} - {signal['direction']}")
        print(f"    信心度: {signal['confidence']:.0%}")
        print(f"    理由: {signal['reason']}")
        print(f"    标的: {signal['target']}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print(" " * 20 + "因果AI量化系统 - 使用示例")
    print("="*80)

    example1_create_factor_library()
    example2_search_factors()
    example3_build_causal_graph()
    example4_apply_to_trading()
    example5_causal_trading_signals()

    print("\n" + "="*80)
    print("✅ 所有示例运行完成！")
    print("="*80)

    print("\n💡 因果AI量化系统核心优势：")
    print("  1. 【因果推理】理解市场底层逻辑，不是简单的统计相关")
    print("  2. 【知识积累】因果知识库可持续积累和演化")
    print("  3. 【可解释性】每个预测都有明确的因果机制")
    print("  4. 【自迭代】系统可自动发现、验证、优化因果知识")
    print("  5. 【跨资产】统一的因果框架适用于股票、期货、等多资产")

    print("\n🚀 下一步：")
    print("  1. 扩展因果因素库（添加更多行业、品种特定因素）")
    print("  2. 实现因果发现引擎（自动发现新因果关系）")
    print("  3. 实现因果验证引擎（样本外、反事实验证）")
    print("  4. 实现自迭代系统（学习-验证-优化循环）")
    print("  5. 集成到交易策略（因果驱动信号生成）")
    print("\n")


if __name__ == "__main__":
    main()
