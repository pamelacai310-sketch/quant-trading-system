# FinceptTerminal 集成指南

## 概述

FinceptTerminal是一个现代化的C++20 Qt6金融终端，提供100+数据连接器、Python 3.11嵌入、AI代理和实时数据分析功能。本集成模块将quant-trading-system与FinceptTerminal无缝连接，实现数据共享、策略互通和信号推送。

---

## 🎯 集成功能

### 1. 数据桥接

**FinceptDataBridge** - 从FinceptTerminal获取市场和经济数据

- **支持的数据源**：
  - Yahoo Finance（股票、ETF、指数）
  - FRED（美国经济数据）
  - Polygon（实时行情）
  - DBnomics（国际数据）
  - 100+ 其他数据源

- **数据类型**：
  - 历史价格数据（OHLCV）
  - 宏观经济指标（GDP、CPI、失业率等）
  - 实时行情数据
  - 公司财务数据

### 2. 策略导出

**FinceptStrategyExporter** - 将系统策略导出为FinceptTerminal格式

- **导出格式**：
  - **fincept_json**：Fincept配置文件格式
  - **fincept_python**：可执行的Python脚本
  - **fincept_workflow**：可视化工作流定义

### 3. 信号推送

**FinceptSignalPusher** - 将交易信号推送到FinceptTerminal

- **推送方式**：
  - API推送（直接推送到Fincept REST API）
  - 文件推送（保存为JSON文件供导入）

### 4. 智能回退

- FinceptTerminal未安装时，自动使用yfinance、pandas-datareader等库
- 确保系统在无Fincept环境下也能正常运行

---

## 🚀 快速开始

### 安装FinceptTerminal（可选）

```bash
# 克隆FinceptTerminal仓库
git clone https://github.com/Fincept-Corporation/FinceptTerminal.git
cd FinceptTerminal

# 按照README进行编译安装
# 需要C++20、Qt6、Python 3.11等依赖
```

### 基本使用

```python
from quant_trade_system.integrations import create_fincept_integration

# 1. 创建集成实例
integrator = create_fincept_integration()

# 2. 检查集成状态
status = integrator.check_integration_status()
print(f"Fincept已安装: {status['fincept_installed']}")
print(f"可用连接器: {status['available_connectors']}")

# 3. 获取市场数据
data = integrator.data_bridge.fetch_market_data(
    symbol="AAPL",
    start_date="2024-01-01",
    end_date="2024-12-31",
)

# 4. 导出策略
strategy_config = {
    "name": "My Strategy",
    "symbol": "AAPL",
    "indicators": [...],
    "entry_rules": [...],
    "exit_rules": [...],
}
integrator.export_strategy_to_fincept(strategy_config)

# 5. 推送信号
integrator.signal_pusher.push_signal(
    symbol="AAPL",
    action="buy",
    quantity=100,
    price=150.25,
    reason="Golden cross",
)
```

---

## 📖 详细示例

### 示例1: 获取市场数据

```python
from quant_trade_system.integrations import FinceptIntegrator, FinceptConfig

# 创建配置
config = FinceptConfig(
    fincept_path="/path/to/fincept",  # 可选
    data_connectors=["yahoo_finance", "fred"],
)

integrator = FinceptIntegrator(config)

# 获取股票数据
stock_data = integrator.data_bridge.fetch_market_data(
    symbol="AAPL",
    start_date="2024-01-01",
    end_date="2024-12-31",
)
print(stock_data.head())

# 获取经济数据
gdp_data = integrator.data_bridge.fetch_economic_data(
    indicator="GDP",
    start_date="2020-01-01",
    end_date="2024-12-31",
)
print(gdp_data.tail())
```

### 示例2: 导出策略为JSON

```python
from quant_trade_system.integrations import FinceptStrategyExporter

exporter = FinceptStrategyExporter()

strategy = {
    "name": "Dual MA",
    "description": "双均线策略",
    "dataset": "yahoo_finance",
    "symbol": "AAPL",
    "indicators": [
        {"name": "fast_ma", "type": "sma", "window": 12},
        {"name": "slow_ma", "type": "sma", "window": 26},
    ],
    "entry_rules": [
        {"left": "fast_ma", "op": "crosses_above", "right": "slow_ma"},
    ],
    "exit_rules": [
        {"left": "fast_ma", "op": "crosses_below", "right": "slow_ma"},
    ],
}

# 导出为JSON格式
json_output = exporter.export_strategy(strategy, "fincept_json")
print(json_output)
```

### 示例3: 导出策略为Python脚本

```python
# 导出为Python脚本
python_output = exporter.export_strategy(strategy, "fincept_python")

# 保存到文件
with open("my_strategy.py", "w") as f:
    f.write(python_output)
```

生成的Python脚本可以直接在FinceptTerminal的嵌入Python环境中运行。

### 示例4: 批量导出多个策略

```python
strategies = [
    {"name": "Strategy 1", ...},
    {"name": "Strategy 2", ...},
    {"name": "Strategy 3", ...},
]

results = integrator.sync_strategies(strategies)
for strategy_id, result in results.items():
    print(f"{strategy_id}: {result}")
```

---

## 🔧 高级配置

### 自定义Fincept路径

```python
from quant_trade_system.integrations import FinceptConfig, FinceptIntegrator

config = FinceptConfig(
    fincept_path="/usr/local/fincept",
    python_executable="/usr/bin/python3.11",
    api_enabled=True,
    api_port=8080,
    data_connectors=[
        "yahoo_finance",
        "polygon",
        "fred",
        "dbnomics",
    ],
)

integrator = FinceptIntegrator(config)
```

### 启用API推送

```python
config = FinceptConfig(
    api_enabled=True,
    api_port=8080,
)

integrator = FinceptIntegrator(config)

# 推送信号（通过API）
success = integrator.signal_pusher.push_signal(
    symbol="AAPL",
    action="buy",
    quantity=100,
    price=150.25,
    reason="Entry signal",
)
```

---

## 📊 支持的策略格式

Fincept集成支持以下策略配置：

### 基本字段

```json
{
  "name": "策略名称",
  "description": "策略描述",
  "dataset": "数据源（yahoo_finance等）",
  "symbol": "标的代码"
}
```

### 指标定义

```json
{
  "indicators": [
    {
      "name": "指标名称",
      "type": "指标类型（sma/ema/rsi/bollinger等）",
      "window": "窗口大小"
    }
  ]
}
```

### 交易规则

```json
{
  "entry_rules": [
    {
      "left": "左侧变量",
      "op": "操作符（crosses_above/crosses_below/>/</>=/<=）",
      "right": "右侧变量"
    }
  ],
  "exit_rules": [...]
}
```

### 风险管理

```json
{
  "position_sizing": {
    "mode": "fixed_fraction/volatility_target",
    "risk_fraction": 0.02
  },
  "risk_limits": {
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.12
  }
}
```

---

## 🔄 回退机制

当FinceptTerminal未安装时，系统会自动回退到替代方案：

| 功能 | Fincept模式 | 回退模式 |
|------|------------|---------|
| 股票数据 | Fincept数据API | yfinance |
| 经济数据 | FRED连接器 | pandas-datareader |
| 策略导出 | 正常导出 | 正常导出 |
| 信号推送 | API推送 | 文件保存 |

---

## 💡 最佳实践

### 1. 数据获取

- 优先使用Fincept的丰富数据源
- 定期缓存数据以提升性能
- 处理缺失数据和异常值

### 2. 策略导出

- 为策略添加清晰的描述和文档
- 使用版本控制管理策略文件
- 在Fincept中验证导出的策略

### 3. 信号推送

- 确保信号格式符合Fincept API规范
- 实现错误处理和重试机制
- 记录信号推送日志

---

## 📚 相关文档

- **[FinceptTerminal GitHub](https://github.com/Fincept-Corporation/FinceptTerminal)** - FinceptTerminal项目主页
- **[使用示例](../examples/fincept_integration_example.py)** - 10个完整示例
- **[API文档](../API_DOCUMENTATION.md)** - 完整API文档

---

## ⚠️ 注意事项

1. **FinceptTerminal是可选依赖**：系统可以在没有Fincept的情况下正常运行
2. **数据延迟**：使用免费数据源可能有15-20分钟延迟
3. **API限制**：某些数据源有API调用频率限制
4. **Python版本**：FinceptTerminal使用Python 3.11，主系统使用Python 3.9

---

## 🐛 故障排除

### 问题1: Fincept路径未找到

**解决方法**：
- 确认FinceptTerminal安装路径
- 在FinceptConfig中设置正确的fincept_path
- 或让系统使用回退模式

### 问题2: yfinance导入失败

**解决方法**：
```bash
pip install yfinance pandas-datareader
```

### 问题3: API推送失败

**解决方法**：
- 检查Fincept API是否启用
- 确认端口配置正确
- 使用文件模式作为备选方案

---

**更新日期**：2026-05-02
**版本**：1.0.0
**维护者**：quant-trading-system团队
