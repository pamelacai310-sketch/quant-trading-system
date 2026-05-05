# quant-trading-system 项目完整总结

## 🎯 项目概述

**quant-trading-system** 是一个**因果AI量化交易系统**，核心特性：
- ✅ JSON配置驱动的策略定义和执行
- ✅ 从数据输入到交易指令输出的完整链路
- ✅ 多种策略类型（欧奈尔CANSLIM、塔勒布杠铃、融合策略、因果AI）
- ✅ 完整的风险管理和订单执行系统
- ✅ 回测引擎和实盘执行

---

## 📊 系统架构全景图

```
┌─────────────────────────────────────────────────────────────┐
│                 quant-trading-system 架构                    │
└─────────────────────────────────────────────────────────────┘

Layer 1: 数据层 (Data Layer)
├─ 数据源 (CSV文件、API、实时行情)
├─ 数据存储 (SQLite数据库)
└─ 数据管理 (Storage类)

Layer 2: 策略层 (Strategy Layer)
├─ JSON策略配置
├─ 策略引擎 (Strategy Engine)
├─ 指标计算 (Indicators)
└─ 多种策略实现
   ├─ 欧奈尔CANSLIM策略
   ├─ 塔勒布杠铃策略
   ├─ 融合策略（股票+期货）
   ├─ 远期期货策略
   ├─ 周波段T0-T5策略
   ├─ 增强融合策略（+因果AI）
   └─ 因果AI知识库

Layer 3: 执行层 (Execution Layer)
├─ 风险管理 (Risk Manager)
├─ 订单执行 (Broker)
│  ├─ PaperBroker (模拟)
│  └─ WebhookBroker (实盘)
└─ 持仓管理 (Position Management)

Layer 4: 服务层 (Service Layer)
├─ HTTP服务器 (FastAPI/HTTPServer)
├─ RESTful API
└─ 前端界面 (Web UI)

Layer 5: 分析层 (Analysis Layer)
├─ 回测引擎 (Backtest Engine)
├─ MAE/MFE分析
├─ 因果AI分析
└─ 性能报告
```

---

## 🔄 完整工作流程

### 流程图

```
┌─────────────────────────────────────────────────────────────┐
│            Step 1: 数据输入 (Data Input)                     │
└─────────────────────────────────────────────────────────────┘
    ├─ CSV文件 (data/*.csv)
    │  ├─ gold_daily.csv
    │  ├─ nasdaq_daily.csv
    │  └─ copper_daily.csv
    ├─ API接口 (OpenBB、QLib、FinRL)
    └─ 实时行情 (WebSocket)

              ↓
┌─────────────────────────────────────────────────────────────┐
│         Step 2: 策略定义 (Strategy Definition)              │
└─────────────────────────────────────────────────────────────┘
    ├─ JSON配置 (通过API或直接配置)
    │  {
    │    "name": "Gold Momentum Crossover",
    │    "dataset": "gold_daily",
    │    "spec": {
    │      "symbol": "XAUUSD",
    │      "indicators": [...],
    │      "entry_rules": [...],
    │      "exit_rules": [...],
    │      "position_sizing": {...},
    │      "risk_limits": {...}
    │    }
    │  }
    └─ 或Python代码策略类
       ├─ HybridSwingStrategy
       ├─ ONeillStrategyEngine
       └─ EnhancedHybridSwingStrategy

              ↓
┌─────────────────────────────────────────────────────────────┐
│        Step 3: 指标计算 (Indicator Computation)             │
└─────────────────────────────────────────────────────────────┘
    └─ quant_trade_system/indicators.py
       ├─ 技术指标
       │  ├─ SMA/EMA (移动平均)
       │  ├─ RSI (相对强弱指标)
       │  ├─ MACD
       │  ├─ 布林带
       │  ├─ 波动率 (Volatility)
       │  ├─ 动量 (Momentum)
       │  └─ Z-Score
       └─ 自定义指标
          └─ 用户可扩展

              ↓
┌─────────────────────────────────────────────────────────────┐
│         Step 4: 策略执行 (Strategy Execution)               │
└─────────────────────────────────────────────────────────────┘
    └─ quant_trade_system/strategy_engine.py
       run_strategy_once(frame, strategy_id, spec, capital, position)

       4.1 准备数据
           ├─ 加载历史数据
           ├─ 计算技术指标
           └─ 生成信号

       4.2 评估信号
           ├─ 入场条件 (entry_rules)
           │  ├─ fast_ma crosses_above slow_ma
           │  ├─ close > slow_ma
           │  └─ volume > vol_sma_20
           ├─ 出场条件 (exit_rules)
           │  └─ fast_ma crosses_below slow_ma
           └─ 持仓状态判断
              ├─ 无持仓 → 检查入场
              ├─ 多头持仓 → 检查出场
              └─ 空头持仓 → 检查出场

       4.3 计算仓位
           ├─ position_sizing.mode
           │  ├─ fixed_units (固定单位)
           │  └─ fixed_fraction (固定比例)
           ├─ max_units限制
           └─ capital限制

       4.4 生成信号
           └─ 返回 StrategyRunResult
              ├─ signal: 1 (多头), -1 (空头), 0 (观望)
              ├─ side: "buy" / "sell" / None
              ├─ quantity: 数量
              ├─ last_price: 最新价格
              ├─ reason: 原因
              └─ indicators: 指标快照

              ↓
┌─────────────────────────────────────────────────────────────┐
│          Step 5: 风险管理 (Risk Management)                │
└─────────────────────────────────────────────────────────────┘
    └─ quant_trade_system/risk.py (RiskManager)

       5.1 预交易风控 (Pre-trade Risk Check)
           ├─ max_order_notional
           │  └─ 单笔订单名义价值 ≤ 限制
           ├─ max_position_per_symbol
           │  └─ 单品种持仓 ≤ 限制
           ├─ max_gross_exposure
           │  └─ 总敞口 ≤ 限制
           ├─ drawdown_limit
           │  └─ 回撤 ≤ 限制
           └─ 持仓集中度
              └─ 单品种持仓占比 ≤ 限制

       5.2 风险评估
           ├─ 通过 → 继续执行
           └─ 不通过 → 拒绝订单，记录风险事件

              ↓
┌─────────────────────────────────────────────────────────────┐
│           Step 6: 订单执行 (Order Execution)                 │
└─────────────────────────────────────────────────────────────┘
    └─ quant_trade_system/broker.py

       6.1 订单路由 (Order Routing)
           ├─ PaperBroker (模拟交易)
           │  └─ 立即成交，添加滑点
           └─ WebhookBroker (实盘交易)
              └─ 发送到配置的Webhook端点

       6.2 订单成交 (Order Fill)
           ├─ PaperBroker
           │  ├─ 计算成交价 (last_price ± slippage)
           │  ├─ 更新持仓
           │  │  ├─ 计算新的持仓数量
           │  │  ├─ 计算新的平均成本
           │  │  └─ 计算已实现盈亏
           │  └─ 更新现金
           │     └─ cash -= signed_qty * fill_price
           └─ WebhookBroker
              └─ 发送到外部券商执行

       6.3 订单记录 (Order Recording)
           └─ 保存到数据库
              ├─ order_id
              ├─ symbol
              ├─ side (buy/sell)
              ├─ quantity
              ├─ fill_price
              ├─ status (filled/pending)
              ├─ requested_at
              ├─ filled_at
              └─ reason

              ↓
┌─────────────────────────────────────────────────────────────┐
│          Step 7: 持仓管理 (Position Management)              │
└─────────────────────────────────────────────────────────────┘
    └─ quant_trade_system/storage.py (Storage)

       7.1 持仓更新 (Position Update)
           ├─ upsert_position(symbol, quantity, avg_price, realized_pnl)
           ├─ 计算新持仓
           │  ├─ 加仓：加权平均
           │  ├─ 减仓：部分平仓
           │  └─ 反向：先平仓，再开仓
           └─ 记录已实现盈亏

       7.2 账户快照 (Portfolio Snapshot)
           ├─ equity (总权益)
           │  = cash + net_exposure
           ├─ cash (现金)
           ├─ gross_exposure (总敞口)
           │  = Σ|持仓市值|
           ├─ net_exposure (净敞口)
           │  = Σ(持仓市值)
           ├─ realized_pnl (已实现盈亏)
           └─ unrealized_pnl (未实现盈亏)
              = Σ(持仓数量 × (当前价 - 平均价))

              ↓
┌─────────────────────────────────────────────────────────────┐
│          Step 8: 信号输出 (Signal Output)                   │
└─────────────────────────────────────────────────────────────┘
    输出格式：

    {
      "status": "accepted" / "rejected" / "no_action",
      "run": {
        "strategy_id": "xxx",
        "strategy_name": "Gold Momentum Crossover",
        "signal": 1,  # 1=多头, -1=空头, 0=观望
        "side": "buy",  # buy/sell/None
        "quantity": 120.5,
        "last_price": 2350.50,
        "reason": "enter_long",
        "indicators": {
          "fast_ma": 2345.20,
          "slow_ma": 2338.80,
          "close": 2350.50,
          ...
        }
      },
      "execution": {
        "order_id": "xxx-xxx-xxx",
        "symbol": "XAUUSD",
        "side": "buy",
        "quantity": 120.5,
        "fill_price": 2350.45,
        "status": "filled"
      },
      "portfolio": {
        "equity": 1000234.56,
        "cash": 987654.32,
        "gross_exposure": 12580.24,
        "net_exposure": 12580.24,
        "realized_pnl": 234.56,
        "unrealized_pnl": 45.67,
        "drawdown": -0.0023
      },
      "risk": {
        "passed": true,
        "violations": [],
        "context": {...}
      }
    }

              ↓
┌─────────────────────────────────────────────────────────────┐
│          Step 9: 持续监控 (Continuous Monitoring)            │
└─────────────────────────────────────────────────────────────┘
    9.1 实时监控 (通过API轮询或WebSocket)
        ├─ 获取最新价格
        ├─ 检查止损止盈
        │  ├─ stop_loss_pct: -5%
        │  └─ take_profit_pct: +10%
        └─ 自动触发平仓

    9.2 风险监控
        ├─ 实时回撤监控
        ├─ 持仓集中度监控
        └─ 触发熔断机制

    9.3 定期复盘
        ├─ 每日账户快照
        ├─ 每周策略评估
        └─ 每月性能分析
```

---

## 💻 核心代码执行流程

### 1. 启动系统

```bash
# 启动HTTP服务器
python3 run.py --host 127.0.0.1 --port 8108

# 访问 Web UI
http://127.0.0.1:8108
```

### 2. 定义策略（JSON配置）

```python
# 通过API POST /api/strategies
{
  "name": "Gold Momentum Crossover",
  "dataset": "gold_daily",
  "status": "active",
  "spec": {
    "symbol": "XAUUSD",
    "direction": "long_only",

    # 指标定义
    "indicators": [
      {"name": "fast_ma", "type": "sma", "window": 10},
      {"name": "slow_ma", "type": "sma", "window": 30},
      {"name": "vol_20", "type": "volatility", "window": 20},
    ],

    # 入场规则
    "entry_rules": [
      {"left": "fast_ma", "op": "crosses_above", "right": "slow_ma"},
      {"left": "close", "op": ">", "right": "slow_ma"},
    ],

    # 出场规则
    "exit_rules": [
      {"left": "fast_ma", "op": "crosses_below", "right": "slow_ma"},
    ],

    # 仓位管理
    "position_sizing": {
      "mode": "fixed_fraction",
      "risk_fraction": 0.12,  # 12%的风险
      "max_units": 180
    },

    # 风险限制
    "risk_limits": {
      "max_order_notional": 60000,
      "max_position_per_symbol": 180,
      "max_gross_exposure": 250000,
      "stop_loss_pct": 0.05,
      "take_profit_pct": 0.14,
    }
  }
}
```

### 3. 执行策略（代码执行流程）

```python
# quant_trade_system/service.py

def execute_strategy(self, strategy_id: str, broker_mode: str = "paper"):
    # Step 1: 加载策略配置
    strategy = self.storage.get_strategy(strategy_id)

    # Step 2: 加载数据
    frame = self.load_dataset(strategy["dataset"])

    # Step 3: 获取当前持仓
    positions = self.storage.get_positions()
    current_position = ...

    # Step 4: 获取可用资金
    capital = float(self.storage.get_portfolio_state()["cash"])

    # Step 5: 运行策略引擎
    run = run_strategy_once(
        frame,           # 历史数据
        strategy_id,
        strategy["name"],
        strategy["spec"], # 策略配置
        capital,
        current_position
    )

    # Step 6: 检查是否需要交易
    if run.side is None or run.quantity <= 0:
        return {"status": "no_action", ...}

    # Step 7: 提交订单
    order_payload = {
        "strategy_id": strategy_id,
        "symbol": strategy["spec"]["symbol"],
        "side": run.side,  # "buy" or "sell"
        "quantity": run.quantity,
        "reason": run.reason,
        ...
    }
    execution = self.submit_order(order_payload)

    return {"status": execution["status"], ...}
```

### 4. 策略引擎（核心逻辑）

```python
# quant_trade_system/strategy_engine.py

def run_strategy_once(frame, strategy_id, strategy_name, spec, capital, current_position):
    # Step 1: 计算指标
    enriched = compute_indicators(frame, spec["indicators"])

    # Step 2: 获取最新数据
    index = len(enriched) - 1
    latest_row = enriched.iloc[index]

    # Step 3: 评估入场条件
    long_entry = spec.get("entry_rules", [])
    if long_entry and _all_conditions(enriched, index, long_entry):
        position_signal = 1  # 多头信号
        reason = "enter_long"

    # Step 4: 评估出场条件
    long_exit = spec.get("exit_rules", [])
    if current_position > 0 and long_exit and _all_conditions(...):
        position_signal = 0  # 平仓信号
        reason = "exit_long"

    # Step 5: 计算仓位
    units = _position_units(spec["position_sizing"], capital, last_price)
    target_quantity = position_signal * units
    delta = target_quantity - current_position

    # Step 6: 确定交易方向
    side = None
    if delta > 0:
        side = "buy"
    elif delta < 0:
        side = "sell"

    # Step 7: 返回结果
    return StrategyRunResult(
        strategy_id=strategy_id,
        signal=position_signal,
        side=side,
        quantity=abs(delta),
        last_price=latest_price,
        reason=reason,
        indicators=latest_row.to_dict()
    )
```

### 5. 风险管理

```python
# quant_trade_system/risk.py

def check_order(self, order, portfolio, positions, last_price, strategy_limits, latest_drawdown):
    violations = []

    # 检查1: 单笔订单名义价值
    notional = order.quantity * last_price
    max_notional = strategy_limits.get("max_order_notional", float('inf'))
    if notional > max_notional:
        violations.append(f"订单名义价值 ${notional:,.0f} 超过限制 ${max_notional:,.0f}")

    # 检查2: 单品种持仓
    current_qty = sum(abs(p["quantity"]) for p in positions if p["symbol"] == order.symbol)
    new_qty = current_qty + order.quantity
    max_position = strategy_limits.get("max_position_per_symbol", float('inf'))
    if abs(new_qty) > max_position:
        violations.append(f"持仓数量 {abs(new_qty)} 超过限制 {max_position}")

    # 检查3: 总敞口
    gross_exposure = sum(abs(p["quantity"]) * last_price for p in positions)
    new_gross = gross_exposure + order.quantity * last_price
    max_gross = strategy_limits.get("max_gross_exposure", float('inf'))
    if new_gross > max_gross:
        violations.append(f"总敞口 ${new_gross:,.0f} 超过限制 ${max_gross:,.0f}")

    # 检查4: 回撤限制
    if latest_drawdown < -0.10:  # 回撤超过10%
        violations.append(f"回撤 {latest_drawdown:.1%} 超过限制 -10%")

    # 返回风险检查结果
    return RiskCheck(
        passed=len(violations) == 0,
        violations=violations,
        context={...}
    )
```

### 6. 订单执行

```python
# quant_trade_system/broker.py

class PaperBroker:
    def execute(self, order, last_price):
        # Step 1: 计算成交价（添加滑点）
        slippage_bps = 3.0  # 3个基点
        signed_qty = order.quantity if order.side == "buy" else -order.quantity
        fill_price = last_price * (1 + slippage_bps / 10000 * (1 if signed_qty > 0 else -1))

        # Step 2: 获取当前持仓
        current_position = ...
        current_qty = current_position["quantity"]
        current_avg_price = current_position["avg_price"]

        # Step 3: 应用成交
        target_qty, avg_price, realized_pnl = self._apply_fill(
            current_qty,
            current_avg_price,
            current_realized_pnl,
            signed_qty,
            fill_price
        )

        # Step 4: 更新持仓
        self.storage.upsert_position(order.symbol, target_qty, avg_price, realized_pnl)

        # Step 5: 更新现金
        self.storage.update_cash(cash - signed_qty * fill_price)

        # Step 6: 记录订单
        order_record = self.storage.add_order({
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "fill_price": fill_price,
            "status": "filled",
            ...
        })

        return order_record
```

---

## 🎯 多种策略类型

### 1. JSON配置策略（简单技术指标）

**适用场景**：简单技术指标策略

**示例**：黄金双均线交叉

```json
{
  "name": "Gold Momentum Crossover",
  "indicators": [
    {"name": "fast_ma", "type": "sma", "window": 10},
    {"name": "slow_ma", "type": "sma", "window": 30}
  ],
  "entry_rules": [
    {"left": "fast_ma", "op": "crosses_above", "right": "slow_ma"}
  ],
  "exit_rules": [
    {"left": "fast_ma", "op": "crosses_below", "right": "slow_ma"}
  ]
}
```

### 2. Python策略类（复杂逻辑）

**适用场景**：复杂策略逻辑

**示例**：融合策略

```python
from quant_trade_system.strategies import HybridSwingStrategy

# 创建策略
strategy = HybridSwingStrategy(
    initial_capital=1_000_000,
    weekly_target=20_000,
)

# 扫描机会
market_data = load_market_data()
stock_ops = strategy.scan_stock_opportunities(market_data, date)
futures = strategy.scan_futures_contracts(date)

# 分析情绪
sentiment = strategy.analyze_market_sentiment(market_data, date)

# 开仓
if sentiment.confidence > 0.65:
    position = strategy.enter_futures_position(
        futures[0], side, date, capital
    )
```

### 3. 欧奈尔CANSLIM策略

**适用场景**：基本面+技术面结合的股票选股

```python
from quant_trade_system.strategies import run_oneill_strategy

# 运行完整策略
result = run_oneill_strategy(
    market_data=market_data,
    fundamental_data=fundamental_data,
    initial_capital=100_000,
)

# 自动执行：
# 1. CANSLIM筛选（7要素评分）
# 2. 形态识别（杯柄、VCP、双底）
# 3. 口袋支点检测
# 4. 生成交易信号
# 5. 执行订单
```

### 4. 因果AI策略

**适用场景**：基于因果关系的智能决策

```python
from quant_trade_system.strategies import CausalHybridStrategy

strategy = CausalHybridStrategy(
    initial_capital=1_000_000,
    base_oneill_allocation=0.60,
    base_taleb_allocation=0.40,
)

# 分析因果信号
causal_signals = strategy.analyze_causal_signals(market_data)

# 根据因果强度动态配置
allocation = strategy.get_dynamic_allocation(causal_signals)

# 执行交易
# 欧奈尔部分（牛市）+ 塔勒布部分（危机保护）
```

---

## 📊 数据流图

```
数据源 (Data Sources)
    ├─ CSV文件 (data/*.csv)
    ├─ API (OpenBB、QLib、FinRL)
    └─ 实时行情
           ↓
    加载 & 存储
    ├─ Storage.load_dataset()
    └─ SQLite数据库
           ↓
    指标计算
    ├─ compute_indicators()
    ├─ SMA/EMA、RSI、MACD等
    └─ 自定义指标
           ↓
    策略执行
    ├─ run_strategy_once()  # JSON策略
    ├─ HybridSwingStrategy  # Python策略
    └─ ONeillStrategyEngine # 欧奈尔策略
           ↓
    信号生成
    ├─ signal: 1/-1/0
    ├─ side: buy/sell/None
    ├─ quantity: 数量
    └─ reason: 原因
           ↓
    风险管理
    ├─ RiskManager.check_order()
    ├─ 多重风险限制检查
    └─ 返回passed/rejected
           ↓
    订单执行
    ├─ PaperBroker.execute()
    ├─ WebhookBroker.execute()
    └─ 更新持仓和现金
           ↓
    持仓管理
    ├─ 更新positions表
    ├─ 更新portfolio_state表
    └─ 保存portfolio_snapshots
           ↓
    输出结果
    ├─ HTTP API响应
    ├─ JSON格式
    └─ 包含订单、持仓、账户信息
```

---

## 🔧 API接口

### 核心API端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/strategies` | POST | 保存/更新策略配置 |
| `/api/strategies` | GET | 列出所有策略 |
| `/api/execute` | POST | 执行策略并生成订单 |
| `/api/orders` | POST | 手动提交订单 |
| `/api/orders` | GET | 列出所有订单 |
| `/api/backtest` | POST | 回测策略 |
| `/api/backtests` | GET | 列出回测结果 |
| `/api/dashboard` | GET | 获取仪表盘数据 |
| `/api/causal/decision` | GET | 获取因果AI决策 |
| `/api/causal/execute` | POST | 执行因果AI决策 |

### 执行流程示例

```bash
# 1. 保存策略
curl -X POST http://127.0.0.1:8108/api/strategies \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Strategy",
    "dataset": "gold_daily",
    "spec": {...}
  }'

# 2. 执行策略
curl -X POST http://127.0.0.1:8108/api/execute \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "xxx-xxx-xxx",
    "broker_mode": "paper"
  }'

# 3. 查看结果
{
  "status": "accepted",
  "run": {
    "signal": 1,
    "side": "buy",
    "quantity": 120.5,
    "reason": "enter_long"
  },
  "execution": {
    "order_id": "yyy-yyy-yyy",
    "fill_price": 2350.45,
    "status": "filled"
  },
  "portfolio": {
    "equity": 1000234.56,
    "cash": 987654.32,
    ...
  }
}
```

---

## 💾 数据存储

### SQLite数据库表结构

```sql
-- 策略表
strategies (
  id TEXT PRIMARY KEY,
  name TEXT,
  status TEXT,  -- active, draft, archived
  dataset TEXT,
  spec_json TEXT,
  created_at TEXT,
  updated_at TEXT
)

-- 回测表
backtests (
  id TEXT PRIMARY KEY,
  strategy_id TEXT,
  report_json TEXT,
  created_at TEXT
)

-- 订单表
orders (
  id TEXT PRIMARY KEY,
  strategy_id TEXT,
  symbol TEXT,
  side TEXT,  -- buy/sell
  quantity REAL,
  order_type TEXT,  -- market/limit
  limit_price REAL,
  status TEXT,  -- filled/pending/cancelled
  broker_mode TEXT,  -- paper/live
  requested_at TEXT,
  filled_at TEXT,
  fill_price REAL,
  reason TEXT,
  metadata_json TEXT
)

-- 持仓表
positions (
  symbol TEXT PRIMARY KEY,
  quantity REAL,
  avg_price REAL,
  realized_pnl REAL,
  updated_at TEXT
)

-- 风险事件表
risk_events (
  id TEXT PRIMARY KEY,
  strategy_id TEXT,
  event_type TEXT,  -- pre_trade, post_trade
  severity TEXT,  -- low/medium/high
  message TEXT,
  metadata_json TEXT,
  created_at TEXT
)

-- 账户状态表
portfolio_state (
  account_id TEXT PRIMARY KEY,
  cash REAL,
  starting_cash REAL,
  updated_at TEXT
)

-- 账户快照表
portfolio_snapshots (
  timestamp TEXT PRIMARY KEY,
  equity REAL,
  cash REAL,
  gross_exposure REAL,
  net_exposure REAL,
  realized_pnl REAL,
  unrealized_pnl REAL,
  drawdown REAL
)
```

---

## 🚀 完整执行示例

### 示例：黄金双均线策略

```python
# ========================================
# Step 1: 定义策略（JSON配置）
# ========================================
strategy_spec = {
    "name": "Gold Momentum Crossover",
    "dataset": "gold_daily",
    "spec": {
        "symbol": "XAUUSD",
        "direction": "long_only",
        "indicators": [
            {"name": "fast_ma", "type": "sma", "window": 10},
            {"name": "slow_ma", "type": "sma", "window": 30},
        ],
        "entry_rules": [
            {"left": "fast_ma", "op": "crosses_above", "right": "slow_ma"},
        ],
        "exit_rules": [
            {"left": "fast_ma", "op": "crosses_below", "right": "slow_ma"},
        ],
        "position_sizing": {
            "mode": "fixed_fraction",
            "risk_fraction": 0.12,
            "max_units": 180,
        },
        "risk_limits": {
            "max_order_notional": 60000,
            "max_position_per_symbol": 180,
            "max_gross_exposure": 250000,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.14,
        },
    },
}

# ========================================
# Step 2: 加载数据
# ========================================
import pandas as pd
data = pd.read_csv("data/gold_daily.csv")
# timestamp, open, high, low, close, volume

# ========================================
# Step 3: 计算指标
# ========================================
from quant_trade_system.indicators import compute_indicators

enriched = compute_indicators(data, strategy_spec["spec"]["indicators"])
# 新增列: fast_ma, slow_ma

# ========================================
# Step 4: 执行策略
# ========================================
from quant_trade_system.strategy_engine import run_strategy_once

run = run_strategy_once(
    frame=enriched,
    strategy_id="strategy_001",
    strategy_name="Gold Momentum Crossover",
    spec=strategy_spec["spec"],
    capital=100000.0,
    current_position=0.0,
)

# ========================================
# Step 5: 检查信号
# ========================================
print(f"信号: {run.signal}")  # 1=多头, -1=空头, 0=观望
print(f"方向: {run.side}")  # buy/sell/None
print(f"数量: {run.quantity}")
print(f"价格: {run.last_price}")
print(f"原因: {run.reason}")
print(f"指标: {run.indicators}")

# 输出示例：
# 信号: 1
# 方向: buy
# 数量: 120.5
# 价格: 2350.50
# 原因: enter_long
# 指标: {'fast_ma': 2345.20, 'slow_ma': 2338.80, 'close': 2350.50}

# ========================================
# Step 6: 风险检查
# ========================================
from quant_trade_system.risk import RiskManager
from quant_trade_system.models import OrderRequest

order = OrderRequest(
    symbol="XAUUSD",
    side="buy",
    quantity=120.5,
    strategy_id="strategy_001",
    broker_mode="paper",
    reason="enter_long",
)

risk_manager = RiskManager()
risk_check = risk_manager.check_order(
    order=order,
    portfolio={"cash": 100000.0, ...},
    positions=[],
    last_price=2350.50,
    strategy_limits=strategy_spec["spec"]["risk_limits"],
    latest_drawdown=0.0,
)

if risk_check.passed:
    print("✅ 风险检查通过")
else:
    print(f"❌ 风险检查失败: {risk_check.violations}")

# ========================================
# Step 7: 执行订单
# ========================================
from quant_trade_system.broker import PaperBroker

broker = PaperBroker(storage)
result = broker.execute(order, last_price=2350.50)

print(f"订单ID: {result['id']}")
print(f"成交价: {result['fill_price']}")
print(f"状态: {result['status']}")

# ========================================
# Step 8: 更新持仓
# ========================================
# broker.execute() 已自动调用:
# storage.upsert_position(
#     symbol="XAUUSD",
#     quantity=120.5,
#     avg_price=2350.45,
#     realized_pnl=0.0
# )

# ========================================
# Step 9: 查看账户
# ========================================
portfolio_state = storage.get_portfolio_state()
print(f"现金: ${portfolio_state['cash']:,.2f}")

positions = storage.get_positions()
for pos in positions:
    print(f"持仓: {pos['symbol']}, 数量: {pos['quantity']}, 成本: {pos['avg_price']}")
```

---

## 🎯 总结

### 核心流程

1. **数据输入** → CSV文件或API
2. **策略定义** → JSON配置或Python类
3. **指标计算** → 技术指标+自定义指标
4. **策略执行** → 评估入场/出场条件
5. **风险管理** → 多重风险限制检查
6. **订单执行** → PaperBroker或WebhookBroker
7. **持仓管理** → 更新持仓和账户状态
8. **信号输出** → HTTP API JSON响应
9. **持续监控** → 止损止盈、风险监控

### 核心优势

- ✅ **灵活性**：支持JSON配置和Python类两种方式
- ✅ **可扩展**：轻松添加新指标和策略
- ✅ **风险可控**：多层风险管理机制
- ✅ **完整链路**：从数据到订单的完整流程
- ✅ **多策略**：欧奈尔、塔勒布、融合策略、因果AI
- ✅ **回测+实盘**：统一框架支持回测和实盘

---

**文档版本**：1.0.0
**最后更新**：2026-05-05
**维护者**：quant-trading-system团队

🎯 **核心原则：数据驱动、策略灵活、风险可控、完整链路！**
