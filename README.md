# 因果AI量化交易系统

在原量化执行 MVP 上升级出的”因果AI量化交易系统”，现在已经把前三个梯队里对当前系统真正有价值的项目接进来，形成了一套可本地运行、可回测、可执行、可导出、可因果分析的研究与交易底座。

## 📚 完整文档

- **[API文档](API_DOCUMENTATION.md)** - 完整的API接口文档和使用说明
- **[配置指南](CONFIGURATION_GUIDE.md)** - 详细的系统配置和部署指南
- **[使用示例](USAGE_EXAMPLES.md)** - 丰富的代码示例和常见使用场景
- **[故障排除](TROUBLESHOOTING.md)** - 问题诊断和解决方案
- **[性能分析](PERFORMANCE_ANALYSIS.md)** - 系统性能分析和优化建议

## 快速导航

### 🚀 快速开始
```bash
# 克隆仓库
git clone https://github.com/pamelacai310-sketch/quant-trading-system.git
cd quant-trading-system

# 启动系统
python3 run.py
```

访问: [http://127.0.0.1:8108](http://127.0.0.1:8108)

### 📖 核心功能
- 🎯 **策略管理** - JSON配置驱动的策略定义和执行
- 📊 **回测引擎** - 历史数据回测和性能分析
- 📈 **MAE/MFE 诊断** - 基于最大潜亏/潜盈的策略诊断与优化
- 🏆 **欧奈尔CANSLIM体系** - 完整实现欧奈尔7要素选股框架
- 📐 **形态识别** - 杯柄、VCP、双底等经典形态自动识别
- 💎 **口袋支点** - 提前买入信号检测（Pocket Pivot）
- 📉 **波动收缩VCP** - Minervini SEPA框架的VCP形态
- 🤖 **因果AI** - 因果关系发现和智能决策
- 🔒 **风险控制** - 预交易风控和实时监控
- 🛡️ **动态止损止盈** - 智能跟踪止损和硬止损机制
- 📉 **极端交易分析** - 自动识别并分析极端交易
- ⚠️ **实盘监控熔断** - 实时监控策略健康度并触发熔断
- 🌉 **生态集成** - 多个量化项目无缝集成

这版系统的设计目标不是”堆最多项目”，而是围绕三个优化准则做工程收敛：

- 推理逻辑最强：先因子和技术面预筛，再做因果发现，再做多智能体裁决
- 分析效率最大化：优先跑便宜的统计和技术特征，把昂贵分析只留给 shortlist
- token 成本最小化：输出 compact evidence pack，减少重复上下文和长文本展开

---

## 🏆 欧奈尔CANSLIM交易体系

### 系统概述

完整实现威廉·欧奈尔（William J. O'Neil）的CANSLIM交易体系及其演化变体，包括：

- ✅ **CANSLIM 选股器** - 7要素基本面筛选系统
- ✅ **形态识别器** - 杯柄、VCP、双底等经典形态自动识别
- ✅ **口袋支点检测器** - Gil Morales & Chris Kacher的提前买入信号
- ✅ **波动收缩VCP** - Mark Minervini的SEPA框架实现
- ✅ **策略引擎** - 整合扫描、执行、风险管理的完整系统

### CANSLIM 七要素

| 要素 | 名称 | 标准 | 实现 |
|------|------|------|------|
| **C** | Current Earnings | 当季EPS增长≥20-25% | `CANSLIM_Screener._score_c()` |
| **A** | Annual Earnings | 年化EPS增长≥25-30%，连续3年 | `CANSLIM_Screener._score_a()` |
| **N** | New | 新产品/催化剂，接近52周新高 | `CANSLIM_Screener._score_n()` |
| **S** | Supply/Demand | 流通盘适中，成交量活跃 | `CANSLIM_Screener._score_s()` |
| **L** | Leader | 相对强度评级(RS)≥70 | `CANSLIM_Screener._score_l()` |
| **I** | Institutional | 机构持股5-70% | `CANSLIM_Screener._score_i()` |
| **M** | Market | 大盘处于上升趋势 | `CANSLIM_Screener._score_m()` |

### 经典形态识别

#### 1. 杯柄形态 (Cup with Handle)
- 前期上涨≥30%
- 杯底深度15-50%
- 柄部深度<杯底的1/3
- 枢轴点：柄部顶部
- 成交量：底部收缩，突破放大

#### 2. 波动收缩VCP (Minervini)
- 3-4次收缩，幅度递减（18% → 12% → 6%）
- 最后收缩<10%
- 突破时成交量放大1.4-1.5倍
- 风险回报比常达1:10+

#### 3. 口袋支点 (Pocket Pivot)
- 突破10日均线
- 成交量>过去10日所有下跌日
- 提前买入，成本更低
- 止损在10日均线下方

### 快速使用

```python
from quant_trade_system.strategies import run_oneill_strategy

# 运行完整欧奈尔策略
engine = run_oneill_strategy(
    stocks_data=stocks_data,
    fundamentals_dict=fundamentals_dict,
    market_index_data=market_data,
    initial_capital=100_000,
    max_positions=5,
)
```

### 详细文档

- **[欧奈尔交易体系指南](docs/欧奈尔交易体系指南.md)** - 完整使用手册
- **[使用示例](examples/oneill_strategy_example.py)** - 6个完整示例

---

## ⚡ 塔勒布杠铃式尾部全天候量化模型

### 系统概述

完整实现纳西姆·塔勒布（Nassim Taleb）的黑天鹅保护策略，采用杠铃式结构：
- ✅ **90-95% 安全资产**：国债逆回购、货币市场基金，产生4-6%稳定收益
- ✅ **5-10% 尾部期权**：深度虚值看跌期权，黑天鹅时爆发10-50倍收益
- ✅ **动态展期机制**：自动时间展期 + 现价展期，避免Theta加速衰减
- ✅ **危机止盈系统**：Delta激增或VIX飙升时自动分批兑现
- ✅ **永续预算机制**：模块A利息反哺模块B，确保策略永不中断

### 核心原理

**为什么需要尾部保护？**
- 传统量化策略依赖正态分布假设，但市场实际有肥尾
- 2008年金融危机、2020年疫情暴跌，都是"黑天鹅"事件
- 一次黑天鹅就能毁灭10年的收益

**塔勒布策略的优势：**
- **反脆弱**：从混乱和压力中获益
- **有限下行**：最大损失就是年度Theta成本（3-6%）
- **无限上行**：黑天鹅时收益可达10-50倍
- **心理舒适**：不需要预测市场，只需坚持执行

### 策略架构

```
┌─────────────────────────────────────────────────────┐
│  模块A：安全资产（90-95%）                            │
│  ├─ 国债逆回购（T-Bills）                             │
│  ├─ 货币市场基金                                       │
│  └─ 目标收益：4-6%年化                                │
│                                                      │
│  作用：产生利息，全额支付模块B的"保险费"                │
└─────────────────────────────────────────────────────┘
                         ↓ 利息转账
┌─────────────────────────────────────────────────────┐
│  模块B：尾部期权（5-10%）                             │
│  ├─ 深度虚值看跌期权（Delta -0.05至-0.10）             │
│  ├─ DTE 90-180天                                     │
│  └─ 标的：SPY/QQQ/沪深300                            │
│                                                      │
│  平时：休眠状态，缓慢Theta失血（年化3-6%）              │
│  危机：爆发式盈利（10-50倍），单骑救主                  │
└─────────────────────────────────────────────────────┘
```

### 三大核心机制

#### 1. 动态展期
- **时间展期**：DTE < 45天时自动展期，避开Theta加速衰减区
- **现价展期**：Delta < 0.02时重新在当前价格下方建仓

#### 2. 危机止盈
- **触发条件**：Delta > -0.50 或 VIX > 40
- **操作**：卖出50%锁定利润，剩余50%下移重置

#### 3. 预算管理
- **月度预算**：账户净值的0.3-0.5%
- **年度最大损失**：3.6-6%（即使所有期权归零）

### 收益情景

**情景A：长牛/震荡市（80%的时间）**
```
模块A：+5%
模块B：-4%（Theta失血）
合计：+1%（账户存活）
```

**情景B：黑天鹅暴跌（20%的时间，但贡献100%利润）**
```
模块A：+5%
股票市场：-30%
模块B：+2000%（期权翻了20倍）
合计：+35%到+65%（在股灾中大赚）
```

### 快速使用

```python
from quant_trade_system.strategies import TalebBarbellStrategy, simulate_taleb_barbell

# 创建策略
strategy = TalebBarbellStrategy(
    initial_capital=1_000_000,
    safe_allocation=0.90,
    tail_allocation=0.10,
    monthly_budget_pct=0.004,
)

# 初始化
portfolio = strategy.initialize_portfolio()

# 运行3年模拟
simulated_strategy = simulate_taleb_barbell(
    initial_capital=1_000_000,
    days=252 * 3,
)

# 生成报告
print(simulated_strategy.generate_report())
```

### 详细文档

- **[塔勒布杠铃策略指南](docs/塔勒布杠铃策略指南.md)** - 完整使用手册
- **[使用示例](examples/taleb_barbell_example.py)** - 6个完整示例

---

## 当前支持能力

- JSON 直接定义策略
- `pandas` 驱动的指标计算和规则策略
- 技术指标扩展：`ta / TA-Lib 风格指标兼容层`
- 回测、Sharpe / 回撤 / 近似 p-value
- 高级回测：`backtrader` 事件驱动回测
- 预交易风控
- Paper Trading 即时成交
- Live Webhook 券商接口预留
- SQLite 持久化订单、持仓、回测、风险事件
- 内置黄金 / 纳指 / 铜演示数据和前端控制台
- 因果图发现：`novaaware -> PCMCI -> 本地启发式回退`
- 因果智能体：`Causal-AI-Agent -> Base Logic` 自动降级
- `finshare` 数据桥接：Python 3.11 子进程拉取数据，失败后自动回退
- OpenBB 桥接预留：宏观/市场上下文桥接接口
- QuantLib 桥接预留：期权定价与风险分析接口
- `ccxt` 接入：多交易所统一执行适配基础
- `lightweight-charts` 前端图表终端
- 多智能体委员会输出：宏观 / 技术 / 风险 / 执行
- 研究压缩输出：compact evidence pack + token estimate
- 策略导出：`Lean / Freqtrade / Hummingbot / TradingAgents / XuanYuan`
- 推理质量评估：`FinEval / FinLongEval` 风格评分

## 快速启动

```bash
cd "/Users/caijiawen/Documents/New project/quant-trading-system"
python3 run.py
```

打开 [http://127.0.0.1:8108](http://127.0.0.1:8108)

---

## 🎯 MAE/MFE 策略诊断与优化系统

### 系统概述

基于 **MAE (Maximum Adverse Excursion，最大潜亏)** 和 **MFE (Maximum Favorable Excursion，最大潜盈)** 分析，提供了一套完整的量化策略诊断与优化工具链。

传统策略评估只关注胜率、盈亏比和年化收益，但无法反映**持仓体验**和**执行质量**。MAE/MFE 散点图能够精准诊断：

1. **买点精确度** - 通过 MAE 分布诊断
2. **止损坚决度** - 通过 MAE 截断诊断
3. **止盈合理性** - 通过 MFE 利用率诊断

### 核心功能

#### 1. MAE/MFE 诊断散点图
- 可视化分析每笔交易的持仓体验
- 识别"半山腰抄底"、"利润过山车"等问题
- 自动生成优化建议

#### 2. 动态止损止盈
- **硬止损**：-6% 或 -8% 无条件止损
- **动态跟踪止盈**：浮盈 8% 激活，回撤 3% 止盈
- **时间止损**：最长持仓 30 天

#### 3. 极端交易自动复盘
- 自动提取极端交易（盈亏超过 ±10%）
- 生成 K 线图和买卖点标记
- 归因分析，识别共性模式

#### 4. 实盘监控与熔断
- 实时监控 MAE/MFE 指标
- 策略健康度评估
- 自动熔断机制（异常时停止交易）

### 快速使用

```python
from quant_trade_system.diagnostics import MAE_MFE_Diagnostics
from quant_trade_system.execution import (
    create_recommended_stop_manager,
    create_recommended_entry_filter,
)
from quant_trade_system.monitoring import LiveMonitor

# 1. 诊断策略
diagnostics = MAE_MFE_Diagnostics()
diagnosis = diagnostics.calculate_from_backtest(
    backtest_result, price_data
)
diagnostics.print_diagnosis_report()

# 2. 优化配置（推荐）
stop_manager = create_recommended_stop_manager()  # 硬止损-6% + 跟踪止盈8%/3%
entry_filter = create_recommended_entry_filter()  # 均线+MACD+放量

# 3. 实盘监控
monitor = LiveMonitor(strategy_id="my_strategy")
report = monitor.update_position(symbol="AAPL", current_price=150.0)
```

### 详细文档

- **[MAE/MFE 优化系统指南](docs/MAE_MFE_优化系统指南.md)** - 完整使用文档
- **[使用示例](examples/mae_mfe_example.py)** - 6 个完整示例

### 优化效果示例

- ✅ 胜率从 45% 提升到 62%
- ✅ 平均 MAE 从 -8.2% 改善到 -4.1%
- ✅ MFE 利用率从 35% 提升到 68%
- ✅ 最大回撤从 22% 降至 9%
- ⚠️  年化收益可能从 85% 降至 28%（可接受的权衡）

---

## 前三梯队项目集成情况

### 第一梯队：已原生接入或桥接接入

- `OpenBB`
  - 作用：补充宏观和市场上下文
  - 状态：已做 Python 3.11 bridge，未安装时自动回退
- `ccxt`
  - 作用：统一交易所 API，扩展执行层
  - 状态：已原生集成
- `TradingView Lightweight Charts`
  - 作用：升级前端研究与监控界面
  - 状态：已 vendored 到本地静态资源
- `ta / TA-Lib 风格指标栈`
  - 作用：补 RSI / MACD / ADX / ATR / Bollinger 等技术因子
  - 状态：已原生集成

### 第二梯队：已作为高级能力或导出目标接入

- `Backtrader`
  - 作用：高级事件驱动回测
  - 状态：已原生集成
- `QuantConnect Lean`
  - 作用：策略导出目标
  - 状态：已导出集成
- `Freqtrade`
  - 作用：加密策略执行框架导出目标
  - 状态：已导出集成
- `Hummingbot`
  - 作用：做市/套利执行模板导出
  - 状态：已导出集成
- `QuantLib`
  - 作用：期权 Black-Scholes 定价与风险扩展
  - 状态：已做 Python 3.11 bridge，未安装时自动回退

### 第三梯队：已作为评测或智能体扩展接入

- `FinEval`
  - 作用：金融推理质量评测
  - 状态：已评测集成
- `FinLongEval`
  - 作用：长文本研究报告压缩评测
  - 状态：已评测集成
- `XuanYuan`
  - 作用：中文金融模型导出模板
  - 状态：已导出集成
- `TradingAgents-AShare`
  - 作用：多智能体研究/裁决导出模板
  - 状态：已导出集成

## 优化后的默认推理流水线

系统现在默认走一条“先便宜、后昂贵”的推理路径：

1. 数据获取
   - 优先 `finshare`
   - 再看本地/演示数据
   - 补充 OpenBB 宏观上下文
2. 技术预筛选
   - 计算 RSI / MACD / ADX / ATR / BB Width / Momentum / Volume Ratio
   - 快速打分，筛掉大部分无效标的
3. shortlist 选择
   - 只把少量候选送入因果发现与深推理
4. 因果发现
   - `novaaware -> PCMCI -> 本地启发式`
5. 多智能体委员会
   - 宏观、技术、风险、执行四个角色分别给观点
6. 决策与执行
   - 走因果智能体或基础决策逻辑
   - 经过风控后执行
7. 压缩研究输出
   - 生成 compact evidence pack
   - 估算 token 成本
   - 输出 FinEval / FinLongEval 风格评分

这条链路的核心是：把最贵的推理只留给最值得分析的对象。

## 主要接口

- `GET /api/dashboard`：总览、策略、持仓、风险事件
- `GET /api/causal/status`：因果系统状态与 GitHub 项目集成状态
- `GET /api/ecosystem/status`：前三梯队生态集成状态
- `POST /api/causal/pipeline`：运行完整因果分析流水线
- `GET /api/causal/market`：生成当前市场因果快照
- `GET /api/causal/decision`：生成因果智能体决策
- `POST /api/causal/execute`：把因果决策转成订单执行
- `POST /api/strategies`：保存策略
- `POST /api/backtest`：运行回测
- `POST /api/backtest/advanced`：运行 `backtrader` 高级回测
- `POST /api/execute`：按策略生成信号并下单
- `POST /api/orders`：手动下单
- `GET /api/research`：回测摘要和 BH FDR 调整
- `POST /api/export/strategy`：导出到 `Lean / Freqtrade / Hummingbot / TradingAgents / XuanYuan`
- `POST /api/options/price`：QuantLib bridge 期权定价
- `GET /api/data/series?dataset=gold_daily`：图表时序和订单标记数据

## 策略 JSON 示例

```json
{
  "symbol": "XAUUSD",
  "direction": "long_only",
  "indicators": [
    { "name": "fast_ma", "type": "sma", "window": 12 },
    { "name": "slow_ma", "type": "sma", "window": 36 }
  ],
  "entry_rules": [
    { "left": "fast_ma", "op": "crosses_above", "right": "slow_ma" }
  ],
  "exit_rules": [
    { "left": "fast_ma", "op": "crosses_below", "right": "slow_ma" }
  ],
  "position_sizing": {
    "mode": "fixed_fraction",
    "risk_fraction": 0.1,
    "max_units": 120
  },
  "risk_limits": {
    "max_order_notional": 50000,
    "max_position_per_symbol": 120,
    "max_gross_exposure": 250000,
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.12
  }
}
```

## 因果系统设计

- `quant_trade_system/causal_ai.py`
  - `GitHubProjectManager`：检测外部项目安装状态
  - `EnhancedDataAdapter`：优先 `finshare`，失败时回退本地 CSV / 代理行情
  - `EnhancedCausalInferenceEngine`：优先 `novaaware`，其次 `PCMCI`，最后本地滞后相关启发式
  - `EnhancedCausalTradingAgent`：优先 `Causal-AI-Agent`，否则执行黄金/铜因果触发逻辑
  - `CausalTradingSystemV4`：整合成完整因果分析流水线
- `quant_trade_system/ecosystem.py`
  - `EcosystemIntegrationManager`：管理前三梯队项目的原生集成、桥接集成、导出集成和评测集成
  - 技术因子 pack、shortlist、token 估算、compact report、多智能体委员会、策略导出和高级回测
- `quant_trade_system/finshare_bridge.py`
  - 用 Python 3.11 子进程桥接 `finshare`
  - 解决主系统仍运行在 Python 3.9 时的兼容问题
  - 自动把 `HOME` 重定向到项目内可写目录，避免 `finshare` 日志初始化写用户目录失败
- `quant_trade_system/openbb_bridge.py`
  - 用 Python 3.11 子进程桥接 `OpenBB`
  - 提供外部市场上下文
- `quant_trade_system/quantlib_bridge.py`
  - 用 Python 3.11 子进程桥接 `QuantLib`
  - 提供期权定价扩展

## 前端与控制台

- 前端控制台位于 `static/index.html`
- 已集成 `lightweight-charts`
- 支持查看：
  - 因果状态
  - 生态集成状态
  - 回测与高级回测
  - compact pipeline 输出
  - 图表与交易标记
  - 一键导出策略

## Live 券商接入

默认 `broker_mode=paper`。如要转到外部执行端，可设置：

```bash
export QUANT_BROKER_WEBHOOK="https://your-broker-gateway.example.com/orders"
```

然后把下单请求的 `broker_mode` 设为 `live`。系统会把订单 JSON 推送到 webhook。

## 测试

```bash
cd "/Users/caijiawen/Documents/New project/quant-trading-system"
python3 -m unittest discover -s tests
```

## 当前边界

- 这是离线本地系统，不含真实实时行情、FIX、撮合回报、合规报送
- `finshare / novaaware / Causal-AI-Agent / OpenBB / QuantLib` 当前都是可选集成；未安装或不可用时自动降级
- 当前 `finshare` 通过 Python 3.11 bridge 使用；主服务仍保持 Python 3.9
- 当前 `OpenBB / QuantLib` 也通过 Python 3.11 bridge 预留
- 启发式因果发现适合 MVP 演示与研究筛选，不等于机构级因果识别
- p-value 使用近似方法，适合第一版研究筛选，不等于完整研究统计栈
- `ccxt` 已接入统一接口层，但还没有扩成完整多交易所实盘 OMS
- `Lean / Freqtrade / Hummingbot / TradingAgents / XuanYuan` 当前是导出集成，不是内嵌运行时
- live 模式当前是 webhook 适配层，方便后续接券商网关或 OMS
