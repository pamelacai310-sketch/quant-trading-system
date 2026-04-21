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
- 🤖 **因果AI** - 因果关系发现和智能决策
- 🔒 **风险控制** - 预交易风控和实时监控
- 🌉 **生态集成** - 多个量化项目无缝集成

这版系统的设计目标不是“堆最多项目”，而是围绕三个优化准则做工程收敛：

- 推理逻辑最强：先因子和技术面预筛，再做因果发现，再做多智能体裁决
- 分析效率最大化：优先跑便宜的统计和技术特征，把昂贵分析只留给 shortlist
- token 成本最小化：输出 compact evidence pack，减少重复上下文和长文本展开

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
