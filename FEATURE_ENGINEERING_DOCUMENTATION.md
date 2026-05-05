# 特征工程层实现文档

## 概述

本文档说明**第二步：特征工程层**的完整实现，该层将因果因素转化为可计算的宽特征矩阵（目标500+特征）。

## 实现状态

### ✅ 已完成功能

#### 1. 核心数据结构
- **FeatureGranularity**: 9种数据粒度（tick到quarterly）
- **DataSource**: 6种数据源类型
- **FeatureDomain**: 14种特征域
- **QuantizedCausalFeature**: 量化因果特征数据类
- **FeatureMatrix**: 特征矩阵数据类
- **FeatureSelectionResult**: 特征选择结果数据类

#### 2. 因果因素量化
已实现将以下类别的因果因素量化为可计算特征：

| 类别 | 因素数量 | 特征数量 | 示例因素 |
|------|---------|---------|---------|
| MACRO_POLICY | 4 | 11 | 利率溢价、通胀溢价、GDP增长、货币政策 |
| MICROSTRUCTURE | 4 | 11 | 机构持仓、流动性、市场情绪、波动率 |
| FUNDAMENTAL | 2 | 6 | EPS增长、ROIC |
| SUPPLY_DEMAND | 1 | 3 | 供需平衡 |
| FUTURES_PRICING | 2 | 4 | 基差、便利收益 |
| VALUATION | 4 | 11 | PE、PB、股息率、EV/EBITDA |
| QUANT_STRATEGY | 4 | 17 | 动量、反转、趋势强度 |
| EQUITY_PREMIUM | 2 | 4 | 股票风险溢价、风险偏好 |
| COMMODITY_PREMIUM | 3 | 6 | 商品风险溢价、滚动收益、期限结构 |
| **总计** | **26** | **73** | - |

**每个因果因素量化为2-3个特征**，包括：
- 原始值/水平值
- 变化率/增长率
- 历史分位数
- 相对指标（相对行业、相对历史）
- 组合指标（如ROIC-WACC利差）

#### 3. Level 2数据处理

##### 3.1 订单簿特征（17个）
```python
- bid_ask_spread: 买卖价差（相对中间价）
- bid_ask_spread_abs: 买卖价差绝对值
- order_imbalance_1: 第1档订单不平衡
- order_imbalance_5: 前5档订单不平衡
- mid_price: 中间价
- mid_price_change: 中间价变化率
- bid_ask_spread_volatility: 买卖价差波动率
- vwap_bid: 买方成交量加权价
- vwap_ask: 卖方成交量加权价
- pressure_ratio: 压力比率
- bid_slope: 买方订单斜率
- ask_slope: 卖方订单斜率
- cumulative_bid_depth: 累计买方深度
- cumulative_ask_depth: 累计卖方深度
- depth_ratio: 深度比率
- price_impact_bid: 买方价格冲击
- price_impact_ask: 卖方价格冲击
```

##### 3.2 逐笔成交特征（15个）
```python
- trade_volume: 成交量
- trade_value: 成交额
- trade_count: 成交笔数
- vwap: 成交量加权平均价
- buy_volume: 买入成交量
- sell_volume: 卖出成交量
- buy_sell_ratio: 买卖比率
- net_buy_volume: 净买入量
- volume_volatility: 成交量波动率
- price_volatility: 价格波动率
- max_trade_volume: 最大单笔成交量
- avg_trade_volume: 平均单笔成交量
- large_trade_ratio: 大单占比
- price_range: 价格变化幅度
- volume_change: 成交量变化率
```

#### 4. 另类数据对齐

##### 4.1 新闻情感特征（8个）
```python
- news_sentiment_avg: 平均情感得分
- news_sentiment_weighted: 加权情感得分（按相关性）
- news_count: 新闻数量
- news_positive_ratio: 正面新闻占比
- news_negative_ratio: 负面新闻占比
- news_sentiment_polarity: 情感极性
- news_sentiment_volatility: 情感波动率
- news_sentiment_latest: 最新新闻情感
```

##### 4.2 卫星数据特征（每类4个）
对每个卫星提取的特征（如NDVI、夜间灯光、建筑面积等）生成：
```python
- satellite_{feature}_latest: 最新值
- satellite_{feature}_avg: 平均值
- satellite_{feature}_change: 变化率
- satellite_{feature}_std: 标准差
```

#### 5. 公式计算引擎
实现了简化的公式解析器，支持：
- 简单列引用（如 `interest_rate_t`）
- 变化率计算（如 `(rate_t - rate_t-1) / rate_t-1`）
- 标准差计算（如 `std(returns_t-20:t)`）
- 比率计算（如 `roic - wacc`）
- 同比/环比增长（如 `_t-12`, `_t-4`）
- 历史波动率（如 `std(returns) * sqrt(252)`）
- 百分位排名（如 `percentile_rank(series, value)`）

## 当前特征统计

| 数据源 | 特征数量 | 说明 |
|-------|---------|------|
| 因果因素量化 | 73 | 从26个因果因素量化而来 |
| Level 2订单簿 | 17 | 微观结构特征 |
| Level 2逐笔成交 | 15 | 高频交易特征 |
| 新闻情感 | 8 | 另类数据特征 |
| 卫星数据 | 17+ | 取决于提取的卫星特征数量 |
| **总计** | **130+** | 当前可生成特征总数 |

## 如何扩展到500+特征

要达到500+特征的目标，有以下几种方法：

### 方法1: 扩展因果因素库（推荐）

在 `quant_trade_system/core/causal/causal_factor_library.py` 中添加更多因果因素。

#### 当前因素数量：30个
#### 目标因素数量：150-200个
#### 预期生成特征：300-600个

#### 添加新因素的步骤：

1. **在 `_initialize_common_factors()` 中添加共有因素**：
   - 宏观经济：PMI、消费者信心、贸易收支、财政政策等
   - 市场微观结构：换手率、amihud非流动性、pastor-stambaugh流动性等
   - 量化策略：规模因子、价值因子、质量因子等

2. **在 `_initialize_equity_factors()` 中添加股票专属因素**：
   - 基本面：净利润增长率、营收增长率、毛利率、净利率等
   - 估值：PEG、PS、EV/Sales、FCF yield等
   - 质量：资产负债率、流动比率、速动比率、毛利率稳定性等
   - 成长：资本支出增长率、研发支出占比等

3. **在 `_initialize_commodity_factors()` 中添加商品期货因素**：
   - 供需：产量数据、消费数据、进口/出口数据
   - 库存：库存水平、库存消费比
   - 天气：降水、温度、极端天气事件
   - 成本：生产成本、运输成本

#### 示例代码：

```python
# 在 _initialize_common_factors() 中添加
self.add_factor(CausalFactor(
    factor_id="pmi_manufacturing",
    name="制造业PMI",
    category=FactorCategory.MACRO_POLICY,
    asset_class=AssetClass.ALL,
    description="PMI上升 → 经济扩张 → 风险资产价格上涨",
    causal_mechanism="PMI↑ → 经济扩张 → 企业盈利↑ → 股价↑",
    data_sources=["统计局", "PMI数据"],
    measurement_methods=["PMI指数", "PMI新订单指数"],
    update_frequency="monthly",
    reliability=0.90,
    confidence=0.85,
    created_at=datetime.now(),
    updated_at=datetime.now(),
    version=1,
    tags=["macro", "economic_indicator"],
))
```

每个新增的因果因素会自动被量化为2-3个特征。

### 方法2: 添加更多技术指标特征

在 `_quantize_technical_factors()` 方法中添加更多技术指标：

```python
# 移动平均线
- MA_5, MA_10, MA_20, MA_50, MA_200
- EMA_12, EMA_26
- MACD, MACD_signal, MACD_histogram

# 动量指标
- RSI_6, RSI_12, RSI_24
- 随机指标(KDJ)
- 威廉指标(WR)

# 成交量指标
- OBV(能量潮)
- Volume_MA_5, Volume_MA_20
- Volume_Ratio

# 波动率指标
- ATR(真实波幅)
- Bollinger_Bands, Bollinger_Width

# 价格形态
- 高低点比率
- 跳空缺口
- 价格通道
```

每个技术指标可以生成3个特征：原始值、变化率、历史分位数。

### 方法3: 添加更多Level 2特征

扩展现有的Level 2处理方法，添加：

```python
# 订单簿动态特征
- 订单簿斜率变化率
- 订单流毒性(POV)
- 大单占比
- 中单占比
- 小单占比

# 时间维度特征
- 开盘后30分钟特征
- 收盘前30分钟特征
- 盘中特征
- 隔夜跳空特征

# 跨时间特征
- 5分钟、15分钟、30分钟、60分钟聚合
- 不同时间尺度的特征对比
```

### 方法4: 添加更多另类数据源

```python
# 社交媒体情感
- Twitter情感
- StockTwits情感
- 东方财富股吧情感

# 卖方分析
- 分析师评级
- 目标价变化
- 盈利预测调整

# 宏观事件
-央行会议
- 经济数据发布
- 地缘政治事件

# 行业数据
- 行业景气度
- 上游/下游数据
- 竞争对手数据
```

## 使用示例

完整的示例代码见 `examples/feature_engineering_example.py`，包含6个示例：

1. **基础特征生成**：从因果因素生成特征矩阵
2. **Level 2订单簿处理**：从订单簿数据提取微观结构特征
3. **Level 2逐笔成交处理**：从逐笔成交数据提取交易特征
4. **新闻情感对齐**：将新闻情感数据对齐到市场时间戳
5. **卫星数据对齐**：将卫星数据对齐到市场时间戳
6. **综合特征矩阵**：整合所有数据源生成完整特征矩阵

### 运行示例：

```bash
cd /Users/caijiawen/Downloads/insurance-crawler-push/quant-trading-system
python3 examples/feature_engineering_example.py
```

## 输出示例

```
✅ 特征工程层创建成功
  注册特征数量: 42

✅ 特征矩阵生成成功
  特征数量: 20
  时间范围: 2024-01-01 至 2024-03-31
  数据粒度: daily
  矩阵形状: (91, 20)

前10个特征:
  1. interest_rate_premium_level
     名称: 利率绝对水平
     因素ID: interest_rate_premium
     金融含义: 利率水平影响资产折现率
     可解释性: 0.98
     独立解释力: 0.90
  ...
```

## 特征质量保证

所有特征都通过以下质量指标评估：

1. **可解释性（Interpretability）**: 0-1，默认要求≥0.6
   - 反映特征是否符合金融理论
   - 公式是否清晰易懂
   - 金融含义是否明确

2. **独立解释力（Independent Power）**: 0-1，默认要求≥0.5
   - 反映特征是否具有独立的预测能力
   - 不完全依赖于其他特征
   - 在不同市场制度下稳定性好

3. **计算成本（Computational Cost）**: 0-1
   - 反映特征计算所需的资源
   - 包括数据获取、计算复杂度、延迟等

4. **预期符号（Expected Sign）**: +1/-1
   - 根据金融理论确定特征与收益的预期关系
   - 用于后续模型训练的约束条件

## 下一步工作

### 第三步：统计学习模型
- 创建 `StatisticalLearningLayer` 类
- 实现 Transformer/LightGBM 模型
- 定义自定义损失函数（胜率、赔率、弹性）
- 实现模型训练pipeline
- 特征选择与模型训练一体化

### 第四步：联合优化层
- 创建 `JointOptimizationLayer` 类
- 将Sharpe比率和CVaR嵌入训练目标
- 实现端到端优化
- 将仓位计算内嵌入训练目标
- 集成Taleb杠铃约束

## 文件结构

```
quant_trade_system/core/
├── feature_engineering_layer.py    # 特征工程层（新增，2300+行）
├── causal/
│   ├── __init__.py                 # 因果模块导出
│   └── causal_factor_library.py    # 因果因素库（30个因素）

examples/
└── feature_engineering_example.py  # 特征工程示例（新增，650+行）
```

## 技术要点

1. **模块化设计**：每个数据处理方法独立，便于测试和扩展
2. **类型安全**：使用Enum和dataclass确保类型安全
3. **可配置性**：所有阈值和参数都可配置
4. **性能优化**：使用pandas向量化操作
5. **错误处理**：每个方法都有异常处理
6. **文档完善**：每个方法都有详细的docstring

## 总结

特征工程层的核心实现已完成，当前可以生成130+特征。要达到500+特征的目标，需要：

1. **扩展因果因素库**：从30个增加到150-200个（推荐）
2. **添加更多技术指标**：50-100个
3. **扩展Level 2特征**：50-100个
4. **集成更多另类数据**：50-100个

通过这些扩展，可以轻松达到500+特征的目标，同时保持每个特征的金融含义和独立解释力。

---

**最后更新**: 2026-05-04
**作者**: Claude Code
**版本**: 1.0
