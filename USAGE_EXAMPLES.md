# 量化交易系统使用示例

## 目录

- [快速开始](#快速开始)
- [策略创建示例](#策略创建示例)
- [回测示例](#回测示例)
- [交易执行示例](#交易执行示例)
- [因果AI使用](#因果ai使用)
- [高级功能](#高级功能)
- [API集成示例](#api集成示例)
- [常见使用场景](#常见使用场景)

## 快速开始

### 1. 启动系统

```bash
# 基础启动
python3 run.py

# 自定义端口
python3 run.py --port 8080

# 自定义主机
python3 run.py --host 0.0.0.0 --port 8108
```

### 2. 访问Web界面

打开浏览器访问: `http://127.0.0.1:8108`

### 3. 使用curl测试

```bash
# 健康检查
curl http://127.0.0.1:8108/api/health

# 获取仪表板
curl http://127.0.0.1:8108/api/dashboard
```

## 策略创建示例

### 示例1: 双均线策略

```json
{
  "name": "双均线黄金策略",
  "dataset": "gold_daily",
  "status": "active",
  "spec": {
    "symbol": "XAUUSD",
    "direction": "long_only",
    "indicators": [
      {"name": "fast_ma", "type": "sma", "window": 10},
      {"name": "slow_ma", "type": "sma", "window": 30}
    ],
    "entry_rules": [
      {"left": "fast_ma", "op": "crosses_above", "right": "slow_ma"}
    ],
    "exit_rules": [
      {"left": "fast_ma", "op": "crosses_below", "right": "slow_ma"}
    ],
    "position_sizing": {
      "mode": "fixed_fraction",
      "risk_fraction": 0.1,
      "max_units": 100
    },
    "risk_limits": {
      "max_order_notional": 50000,
      "max_position_per_symbol": 100,
      "max_gross_exposure": 200000,
      "stop_loss_pct": 0.05,
      "take_profit_pct": 0.12
    }
  }
}
```

### 示例2: RSI超买超卖策略

```json
{
  "name": "RSI均值回归策略",
  "dataset": "nasdaq_daily",
  "status": "active",
  "spec": {
    "symbol": "QQQ",
    "direction": "long_short",
    "indicators": [
      {"name": "rsi_14", "type": "rsi", "window": 14},
      {"name": "bb_upper", "type": "bollinger_upper", "window": 20, "num_std": 2},
      {"name": "bb_lower", "type": "bollinger_lower", "window": 20, "num_std": 2}
    ],
    "entry_rules": [
      {"left": "rsi_14", "op": "<", "right": 30}
    ],
    "exit_rules": [
      {"left": "rsi_14", "op": ">", "right": 70}
    ],
    "short_entry_rules": [
      {"left": "rsi_14", "op": ">", "right": 70}
    ],
    "short_exit_rules": [
      {"left": "rsi_14", "op": "<", "right": 30}
    ],
    "position_sizing": {
      "mode": "fixed_units",
      "units": 50
    },
    "risk_limits": {
      "max_order_notional": 30000,
      "stop_loss_pct": 0.03
    }
  }
}
```

### 示例3: 动量突破策略

```json
{
  "name": "动量突破策略",
  "dataset": "copper_daily",
  "status": "active",
  "spec": {
    "symbol": "HG",
    "direction": "long_only",
    "indicators": [
      {"name": "momentum_20", "type": "momentum", "window": 20},
      {"name": "atr_14", "type": "atr", "window": 14},
      {"name": "volume_sma", "type": "volume_sma", "window": 20}
    ],
    "entry_rules": [
      {"left": "momentum_20", "op": ">", "right": 0},
      {"left": "volume", "op": ">", "right": "volume_sma"}
    ],
    "exit_rules": [
      {"left": "close", "op": "<", "right": "close", "shift": 1}
    ],
    "position_sizing": {
      "mode": "volatility_target",
      "target_volatility": 0.15,
      "max_units": 200
    },
    "risk_limits": {
      "max_order_notional": 80000,
      "stop_loss_pct": 0.08
    }
  }
}
```

### Python客户端示例

```python
import requests
import json

class QuantTradingClient:
    def __init__(self, base_url="http://127.0.0.1:8108"):
        self.base_url = base_url

    def create_strategy(self, strategy_spec):
        """创建策略"""
        response = requests.post(
            f"{self.base_url}/api/strategies",
            json=strategy_spec
        )
        return response.json()

    def get_strategies(self):
        """获取所有策略"""
        response = requests.get(f"{self.base_url}/api/strategies")
        return response.json()

    def run_backtest(self, strategy_id, start_date, end_date):
        """运行回测"""
        response = requests.post(
            f"{self.base_url}/api/backtest",
            json={
                "strategy_id": strategy_id,
                "start_date": start_date,
                "end_date": end_date
            }
        )
        return response.json()

# 使用示例
client = QuantTradingClient()

# 创建双均线策略
strategy = {
    "name": "测试策略",
    "dataset": "gold_daily",
    "status": "active",
    "spec": {
        "symbol": "XAUUSD",
        "direction": "long_only",
        "indicators": [
            {"name": "fast_ma", "type": "sma", "window": 10},
            {"name": "slow_ma", "type": "sma", "window": 30}
        ],
        "entry_rules": [
            {"left": "fast_ma", "op": "crosses_above", "right": "slow_ma"}
        ],
        "exit_rules": [
            {"left": "fast_ma", "op": "crosses_below", "right": "slow_ma"}
        ],
        "position_sizing": {
            "mode": "fixed_fraction",
            "risk_fraction": 0.1
        },
        "risk_limits": {
            "max_order_notional": 50000,
            "stop_loss_pct": 0.05
        }
    }
}

result = client.create_strategy(strategy)
print(f"策略创建成功: {result['strategy_id']}")
```

## 回测示例

### 基础回测

```bash
# 使用curl
curl -X POST http://127.0.0.1:8108/api/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "STRAT_001",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
  }'
```

### Python回测脚本

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt

def run_backtest_and_analyze():
    # 运行回测
    response = requests.post(
        "http://127.0.0.1:8108/api/backtest",
        json={
            "strategy_id": "STRAT_001",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        }
    )

    result = response.json()

    if not result.get("success"):
        print(f"回测失败: {result.get('message')}")
        return

    # 分析结果
    summary = result["summary"]
    print(f"总交易次数: {summary['total_trades']}")
    print(f"胜率: {summary['win_rate']:.2%}")
    print(f"总收益率: {summary['total_return']:.2%}")
    print(f"夏普比率: {summary['sharpe_ratio']:.2f}")
    print(f"最大回撤: {summary['max_drawdown']:.2%}")

    # 绘制资金曲线
    equity_curve = pd.DataFrame(result["equity_curve"])
    equity_curve['date'] = pd.to_datetime(equity_curve['date'])

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve['date'], equity_curve['value'])
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('equity_curve.png')
    print("资金曲线已保存到 equity_curve.png")

    # 分析交易记录
    trades = pd.DataFrame(result["trades"])
    if not trades.empty:
        trades['entry_date'] = pd.to_datetime(trades['entry_date'])
        trades['exit_date'] = pd.to_datetime(trades['exit_date'])

        # 计算持仓时间
        trades['holding_days'] = (trades['exit_date'] - trades['entry_date']).dt.days

        print(f"\n平均持仓天数: {trades['holding_days'].mean():.1f}")
        print(f"最长持仓天数: {trades['holding_days'].max()}")
        print(f"最短持仓天数: {trades['holding_days'].min()}")

        # 按月统计收益
        trades['month'] = trades['exit_date'].dt.to_period('M')
        monthly_returns = trades.groupby('month')['pnl'].sum()
        print("\n月度收益:")
        print(monthly_returns)

if __name__ == "__main__":
    run_backtest_and_analyze()
```

### 批量回测优化

```python
from concurrent.futures import ThreadPoolExecutor
import requests

def run_single_backtest(strategy_params):
    """运行单个回测"""
    try:
        response = requests.post(
            "http://127.0.0.1:8108/api/backtest",
            json=strategy_params,
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def batch_backtest(strategy_variations):
    """批量回测多个参数组合"""
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_single_backtest, params)
            for params in strategy_variations
        ]

        for future in futures:
            result = future.result()
            results.append(result)

    return results

# 参数优化示例
def optimize_strategy_parameters():
    base_strategy = {
        "strategy_id": "STRAT_001",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    }

    # 测试不同的均线参数组合
    variations = []
    for fast_window in [5, 10, 15]:
        for slow_window in [20, 30, 40]:
            if fast_window < slow_window:
                variation = base_strategy.copy()
                variation['fast_ma'] = fast_window
                variation['slow_ma'] = slow_window
                variations.append(variation)

    # 批量运行回测
    results = batch_backtest(variations)

    # 找出最佳参数组合
    best_result = max(
        [r for r in results if r.get("success")],
        key=lambda x: x["summary"]["sharpe_ratio"]
    )

    print(f"最佳参数组合:")
    print(f"夏普比率: {best_result['summary']['sharpe_ratio']:.2f}")
    print(f"收益率: {best_result['summary']['total_return']:.2%}")

if __name__ == "__main__":
    optimize_strategy_parameters()
```

## 交易执行示例

### 实时信号执行

```python
import requests
import time
from datetime import datetime

class LiveTrader:
    def __init__(self, strategy_id, base_url="http://127.0.0.1:8108"):
        self.strategy_id = strategy_id
        self.base_url = base_url
        self.running = False

    def execute_strategy(self):
        """执行策略生成交易信号"""
        response = requests.post(
            f"{self.base_url}/api/execute",
            json={
                "strategy_id": self.strategy_id,
                "broker_mode": "paper"
            }
        )
        return response.json()

    def run_live(self, interval_minutes=60):
        """运行实时交易"""
        self.running = True

        while self.running:
            try:
                print(f"\n[{datetime.now()}] 检查交易信号...")

                result = self.execute_strategy()

                if result.get("success"):
                    signals = result.get("signals", [])

                    if signals:
                        print(f"发现 {len(signals)} 个交易信号:")
                        for signal in signals:
                            print(f"  {signal['symbol']}: {signal['action']} "
                                  f"{signal['quantity']} @ {signal['price']}")
                            print(f"  原因: {signal['reason']}")
                    else:
                        print("暂无交易信号")
                else:
                    print(f"执行失败: {result.get('message')}")

                # 等待下一次检查
                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                print("\n停止实时交易")
                self.running = False
            except Exception as e:
                print(f"发生错误: {e}")
                time.sleep(60)

# 使用示例
trader = LiveTrader("STRAT_001")
trader.run_live(interval_minutes=30)
```

### 手动交易

```bash
# 创建买入订单
curl -X POST http://127.0.0.1:8108/api/orders \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "XAUUSD",
    "side": "buy",
    "quantity": 50,
    "order_type": "market",
    "broker_mode": "paper"
  }'

# 查看订单状态
curl http://127.0.0.1:8108/api/orders?status=filled&limit=10
```

## 因果AI使用

### 因果分析流水线

```python
import requests
import json

def run_causal_analysis():
    """运行完整因果分析"""
    response = requests.post(
        "http://127.0.0.1:8108/api/causal/pipeline",
        json={
            "symbol": "XAUUSD",
            "lookback_days": 30,
            "include_features": True
        }
    )

    result = response.json()

    if result.get("success"):
        # 显示因果图
        causal_graph = result["causal_graph"]
        print("因果关系图:")
        print(f"节点: {causal_graph['nodes']}")

        print("\n因果关系边:")
        for edge in causal_graph['edges']:
            print(f"  {edge['from']} -> {edge['to']}: "
                  f"强度={edge['strength']:.2f}")

        # 显示洞察
        print("\n因果洞察:")
        for insight in result["insights"]:
            print(f"  • {insight}")

        # 显示推荐
        print("\n交易推荐:")
        for rec in result["recommendations"]:
            print(f"  • {rec['action']} (置信度: {rec['confidence']:.2f})")
            print(f"    原因: {rec['reason']}")
    else:
        print(f"因果分析失败: {result.get('message')}")

if __name__ == "__main__":
    run_causal_analysis()
```

### 市场状态监控

```python
import requests
import time
from datetime import datetime

def monitor_market_regime():
    """监控市场状态"""
    while True:
        try:
            response = requests.get(
                "http://127.0.0.1:8108/api/causal/market"
            )
            result = response.json()

            regime = result["market_regime"]
            probability = result["regime_probability"]

            print(f"[{datetime.now()}] 市场状态: {regime} "
                  f"(概率: {probability:.2%})")

            # 显示关键因果因素
            print("关键因果因素:")
            for factor in result["key_causal_factors"]:
                direction = "↑" if factor["direction"] == "positive" else "↓"
                print(f"  {direction} {factor['factor']}: "
                      f"影响={factor['impact']:.2f}")

            time.sleep(300)  # 每5分钟检查一次

        except KeyboardInterrupt:
            print("\n停止监控")
            break
        except Exception as e:
            print(f"监控错误: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_market_regime()
```

## 高级功能

### 策略导出

```python
import requests

def export_strategy(strategy_id, target_platform):
    """导出策略到其他平台"""
    response = requests.post(
        "http://127.0.0.1:8108/api/export/strategy",
        json={
            "strategy_id": strategy_id,
            "target": target_platform  # lean, freqtrade, hummingbot
        }
    )

    result = response.json()

    if result.get("success"):
        print(f"策略已导出到 {target_platform}")
        print(f"文件: {result['export_file']}")

        # 保存导出文件
        with open(result['export_file'], 'w') as f:
            f.write(result['exported_code'])
    else:
        print(f"导出失败: {result.get('message')}")

# 导出到Freqtrade
export_strategy("STRAT_001", "freqtrade")

# 导出到Lean
export_strategy("STRAT_001", "lean")
```

### 期权定价

```python
import requests

def price_option(
    underlying_price,
    strike_price,
    time_to_expiry,
    volatility,
    risk_free_rate=0.02
):
    """使用QuantLib定价期权"""
    response = requests.post(
        "http://127.0.0.1:8108/api/options/price",
        json={
            "underlying_price": underlying_price,
            "strike_price": strike_price,
            "time_to_expiry": time_to_expiry,
            "volatility": volatility,
            "risk_free_rate": risk_free_rate,
            "option_type": "call"
        }
    )

    result = response.json()

    if result.get("success"):
        print(f"期权价格: {result['price']:.2f}")
        print(f"Delta: {result['greeks']['delta']:.4f}")
        print(f"Gamma: {result['greeks']['gamma']:.4f}")
        print(f"Theta: {result['greeks']['theta']:.4f}")
        print(f"Vega: {result['greeks']['vega']:.4f}")
    else:
        print(f"定价失败: {result.get('message')}")

# 示例：定价一个看涨期权
price_option(
    underlying_price=1890.5,    # 标的价格
    strike_price=1900,          # 行权价
    time_to_expiry=0.25,        # 3个月
    volatility=0.18             # 18%波动率
)
```

## API集成示例

### JavaScript集成

```javascript
const axios = require('axios');

class QuantTradingClient {
    constructor(baseURL = 'http://127.0.0.1:8108') {
        this.baseURL = baseURL;
    }

    async createStrategy(strategySpec) {
        const response = await axios.post(
            `${this.baseURL}/api/strategies`,
            strategySpec
        );
        return response.data;
    }

    async runBacktest(strategyId, startDate, endDate) {
        const response = await axios.post(
            `${this.baseURL}/api/backtest`,
            {
                strategy_id: strategyId,
                start_date: startDate,
                end_date: endDate
            }
        );
        return response.data;
    }

    async executeStrategy(strategyId) {
        const response = await axios.post(
            `${this.baseURL}/api/execute`,
            {
                strategy_id: strategyId,
                broker_mode: "paper"
            }
        );
        return response.data;
    }
}

// 使用示例
(async () => {
    const client = new QuantTradingClient();

    // 创建策略
    const strategy = await client.createStrategy({
        name: "测试策略",
        dataset: "gold_daily",
        spec: {
            symbol: "XAUUSD",
            direction: "long_only",
            indicators: [
                { name: "fast_ma", type: "sma", window: 10 },
                { name: "slow_ma", type: "sma", window: 30 }
            ],
            entry_rules: [
                { left: "fast_ma", op: "crosses_above", right: "slow_ma" }
            ]
        }
    });

    console.log(`策略创建成功: ${strategy.strategy_id}`);

    // 运行回测
    const backtest = await client.runBacktest(
        strategy.strategy_id,
        "2023-01-01",
        "2023-12-31"
    );

    console.log(`回测完成，夏普比率: ${backtest.summary.sharpe_ratio}`);
})();
```

## 常见使用场景

### 场景1: 日内交易

```python
# 短周期均值回归策略
intraday_strategy = {
    "name": "日内均值回归",
    "dataset": "gold_hourly",  # 需要准备小时数据
    "spec": {
        "symbol": "XAUUSD",
        "direction": "long_short",
        "indicators": [
            {"name": "ema_9", "type": "ema", "window": 9},
            {"name": "ema_21", "type": "ema", "window": 21},
            {"name": "rsi_14", "type": "rsi", "window": 14}
        ],
        "entry_rules": [
            {"left": "close", "op": "<", "right": "ema_21"},
            {"left": "rsi_14", "op": "<", "right": 30}
        ],
        "exit_rules": [
            {"left": "close", "op": ">", "right": "ema_9"}
        ],
        "position_sizing": {
            "mode": "fixed_units",
            "units": 10
        }
    }
}
```

### 场景2: 趋势跟踪

```python
# 长周期趋势策略
trend_strategy = {
    "name": "趋势跟踪策略",
    "dataset": "nasdaq_daily",
    "spec": {
        "symbol": "QQQ",
        "direction": "long_only",
        "indicators": [
            {"name": "trend_ma", "type": "sma", "window": 200},
            {"name": "signal_ma", "type": "ema", "window": 50},
            {"name": "macd", "type": "macd", "fast": 12, "slow": 26, "signal": 9}
        ],
        "entry_rules": [
            {"left": "close", "op": ">", "right": "trend_ma"},
            {"left": "signal_ma", "op": "crosses_above", "right": "trend_ma"}
        ],
        "exit_rules": [
            {"left": "close", "op": "<", "right": "trend_ma"}
        ],
        "position_sizing": {
            "mode": "volatility_target",
            "target_volatility": 0.12,
            "max_units": 500
        }
    }
}
```

### 场景3: 多资产组合

```python
# 创建多个相关策略
symbols = ["XAUUSD", "QQQ", "HG"]  # 黄金、纳指、铜

for symbol in symbols:
    strategy = {
        "name": f"{symbol}动量策略",
        "dataset": f"{symbol.lower()}_daily",
        "spec": {
            "symbol": symbol,
            "direction": "long_only",
            "indicators": [
                {"name": "momentum", "type": "momentum", "window": 20}
            ],
            "entry_rules": [
                {"left": "momentum", "op": ">", "right": 0}
            ]
        }
    }
    # 创建并运行策略
    client.create_strategy(strategy)
```

### 场景4: 风险平价组合

```python
def create_risk_parity_portfolio():
    """创建风险平价组合"""
    # 为每个资产分配等风险预算
    assets = ["XAUUSD", "QQQ", "HG"]
    risk_budget = 1.0 / len(assets)  # 每个资产33%风险预算

    strategies = []
    for asset in assets:
        strategy = {
            "name": f"{asset}风险平价",
            "spec": {
                "symbol": asset,
                "position_sizing": {
                    "mode": "risk_parity",
                    "risk_budget": risk_budget,
                    "target_volatility": 0.15
                }
            }
        }
        strategies.append(strategy)

    return strategies

# 创建组合
portfolio = create_risk_parity_portfolio()
for strategy in portfolio:
    client.create_strategy(strategy)
```

## 调试和监控

### 启用详细日志

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# 在代码中使用
logger = logging.getLogger(__name__)
logger.info("启动交易系统")
logger.debug(f"策略参数: {strategy_params}")
```

### 性能监控

```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time

        logger.info(f"{func.__name__} 耗时: {elapsed_time:.3f}秒")

        # 如果耗时过长，发出警告
        if elapsed_time > 5.0:
            logger.warning(f"{func.__name__} 执行时间过长!")

        return result
    return wrapper

# 使用装饰器
@monitor_performance
def run_backtest(strategy_params):
    # 回测逻辑
    pass
```

## 最佳实践

1. **策略测试**: 在实盘前充分回测
2. **风险管理**: 设置合理的止损和仓位限制
3. **参数优化**: 定期重新优化策略参数
4. **监控日志**: 定期检查系统日志和性能指标
5. **备份数据**: 定期备份交易数据和策略配置

## 故障排除

1. **API连接失败**: 检查系统是否启动，端口是否正确
2. **策略创建失败**: 检查策略JSON格式是否正确
3. **回测无数据**: 确认数据文件存在且格式正确
4. **执行无信号**: 检查市场条件是否满足入场条件

更多帮助请参考: `TROUBLESHOOTING.md`
