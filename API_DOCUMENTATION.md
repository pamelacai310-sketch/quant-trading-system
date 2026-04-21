# 量化交易系统 API 文档

## 基础信息

**Base URL**: `http://127.0.0.1:8108`
**API版本**: v1.0
**数据格式**: JSON
**字符编码**: UTF-8

## 目录

- [概览](#概览)
- [认证](#认证)
- [通用接口](#通用接口)
- [策略管理](#策略管理)
- [回测接口](#回测接口)
- [交易执行](#交易执行)
- [数据接口](#数据接口)
- [因果AI](#因果ai)
- [系统监控](#系统监控)
- [错误处理](#错误处理)

## 概览

本API提供完整的量化交易功能，包括策略定义、回测、执行、风险管理以及因果AI分析。

### 核心功能

- 🎯 **策略管理**: 创建、更新、删除交易策略
- 📊 **回测引擎**: 历史数据回测和性能分析
- 🤖 **智能执行**: 自动化交易执行和风控
- 🧠 **因果AI**: 因果关系发现和智能决策
- 📈 **实时监控**: 持仓、风险、性能监控

## 认证

当前版本无需认证（开发模式）。生产环境建议添加API密钥认证。

```http
Authorization: Bearer YOUR_API_KEY
```

## 通用接口

### GET /

系统健康检查

**响应示例**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET /api/dashboard

获取系统总览信息

**响应示例**:
```json
{
  "portfolio": {
    "starting_cash": 100000.0,
    "current_value": 125430.5,
    "unrealized_pnl": 25430.5,
    "realized_pnl": 8520.3,
    "total_pnl": 33950.8
  },
  "positions": [
    {
      "symbol": "XAUUSD",
      "quantity": 150.0,
      "avg_price": 1850.25,
      "current_price": 1890.50,
      "unrealized_pnl": 6037.5
    }
  ],
  "recent_orders": [
    {
      "id": "ORD_20240115_001",
      "symbol": "XAUUSD",
      "side": "buy",
      "quantity": 50,
      "price": 1885.3,
      "status": "filled",
      "timestamp": "2024-01-15T10:25:00Z"
    }
  ],
  "risk_events": []
}
```

## 策略管理

### GET /api/strategies

获取所有策略列表

**响应示例**:
```json
{
  "strategies": [
    {
      "id": "STRAT_001",
      "name": "Gold Momentum Crossover",
      "dataset": "gold_daily",
      "status": "active",
      "created_at": "2024-01-10T08:00:00Z",
      "updated_at": "2024-01-15T10:00:00Z"
    }
  ]
}
```

### POST /api/strategies

创建新策略

**请求体**:
```json
{
  "name": "My Strategy",
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

**响应示例**:
```json
{
  "success": true,
  "strategy_id": "STRAT_002",
  "message": "策略创建成功"
}
```

### PUT /api/strategies/{strategy_id}

更新策略

**请求体**: 同POST /api/strategies

**响应示例**:
```json
{
  "success": true,
  "message": "策略更新成功"
}
```

### DELETE /api/strategies/{strategy_id}

删除策略

**响应示例**:
```json
{
  "success": true,
  "message": "策略删除成功"
}
```

## 回测接口

### POST /api/backtest

执行基础回测

**请求体**:
```json
{
  "strategy_id": "STRAT_001",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31"
}
```

**响应示例**:
```json
{
  "success": true,
  "backtest_id": "BT_20240115_001",
  "summary": {
    "total_trades": 45,
    "win_rate": 0.62,
    "total_return": 0.254,
    "sharpe_ratio": 1.85,
    "max_drawdown": -0.082,
    "avg_win": 0.032,
    "avg_loss": -0.018
  },
  "equity_curve": [
    {"date": "2023-01-01", "value": 100000},
    {"date": "2023-01-02", "value": 100250}
  ],
  "trades": [
    {
      "entry_date": "2023-01-05",
      "exit_date": "2023-01-12",
      "side": "long",
      "entry_price": 1850.3,
      "exit_price": 1890.5,
      "quantity": 50,
      "pnl": 2010.0,
      "return": 0.022
    }
  ]
}
```

### POST /api/backtest/advanced

执行高级回测（使用backtrader）

**请求体**:
```json
{
  "strategy_id": "STRAT_001",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "initial_cash": 100000,
  "commission": 0.001
}
```

**响应示例**:
```json
{
  "success": true,
  "backtest_id": "BT_ADV_20240115_001",
  "sharpe": 1.92,
  "max_drawdown": 0.078,
  "total_return": 0.265,
  "win_rate": 0.64,
  "trades_count": 48
}
```

## 交易执行

### POST /api/execute

按策略执行交易

**请求体**:
```json
{
  "strategy_id": "STRAT_001",
  "broker_mode": "paper"
}
```

**响应示例**:
```json
{
  "success": true,
  "signals": [
    {
      "symbol": "XAUUSD",
      "action": "buy",
      "quantity": 50,
      "price": 1885.3,
      "reason": "Fast MA crosses above Slow MA",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "orders_created": 1
}
```

### POST /api/orders

手动创建订单

**请求体**:
```json
{
  "symbol": "XAUUSD",
  "side": "buy",
  "quantity": 50,
  "order_type": "market",
  "broker_mode": "paper"
}
```

**响应示例**:
```json
{
  "success": true,
  "order_id": "ORD_20240115_002",
  "status": "filled",
  "filled_price": 1885.3,
  "filled_quantity": 50
}
```

### GET /api/orders

获取订单列表

**查询参数**:
- `status`: 订单状态过滤 (pending/filled/cancelled)
- `limit`: 返回数量限制 (默认: 50)

**响应示例**:
```json
{
  "orders": [
    {
      "id": "ORD_20240115_001",
      "symbol": "XAUUSD",
      "side": "buy",
      "quantity": 50,
      "price": 1885.3,
      "status": "filled",
      "created_at": "2024-01-15T10:25:00Z"
    }
  ],
  "total": 128
}
```

## 数据接口

### GET /api/data/series

获取时序数据

**查询参数**:
- `dataset`: 数据集名称 (gold_daily/nasdaq_daily/copper_daily)
- `start_date`: 开始日期 (YYYY-MM-DD)
- `end_date`: 结束日期 (YYYY-MM-DD)

**响应示例**:
```json
{
  "dataset": "gold_daily",
  "data": [
    {
      "date": "2024-01-15",
      "open": 1880.5,
      "high": 1895.3,
      "low": 1875.2,
      "close": 1890.5,
      "volume": 125000
    }
  ],
  "annotations": [
    {
      "date": "2024-01-15",
      "type": "order",
      "side": "buy",
      "price": 1885.3,
      "quantity": 50
    }
  ]
}
```

## 因果AI

### GET /api/causal/status

获取因果系统状态

**响应示例**:
```json
{
  "causal_system": {
    "enabled": true,
    "engine": "pcmci",
    "status": "ready"
  },
  "integrations": {
    "novaaware": "installed",
    "Causal-AI-Agent": "not_installed",
    "finshare": "installed"
  }
}
```

### POST /api/causal/pipeline

运行完整因果分析流水线

**请求体**:
```json
{
  "symbol": "XAUUSD",
  "lookback_days": 30,
  "include_features": true
}
```

**响应示例**:
```json
{
  "success": true,
  "causal_graph": {
    "nodes": ["price", "volume", "volatility", "momentum"],
    "edges": [
      {"from": "momentum", "to": "price", "strength": 0.82},
      {"from": "volume", "to": "volatility", "strength": 0.65}
    ]
  },
  "insights": [
    "动量指标对价格有强因果关系 (0.82)",
    "成交量对波动率有中等强度因果关系 (0.65)"
  ],
  "recommendations": [
    {
      "action": "hold_long",
      "confidence": 0.78,
      "reason": "动量指标显示持续上升趋势"
    }
  ]
}
```

### GET /api/causal/market

获取市场因果快照

**响应示例**:
```json
{
  "market_regime": "trending",
  "key_causal_factors": [
    {"factor": "momentum", "impact": 0.82, "direction": "positive"},
    {"factor": "volatility", "impact": 0.45, "direction": "negative"}
  ],
  "regime_probability": 0.76,
  "updated_at": "2024-01-15T10:30:00Z"
}
```

## 系统监控

### GET /api/performance

获取系统性能指标

**响应示例**:
```json
{
  "api": {
    "avg_response_time_ms": 285,
    "requests_per_second": 12.5,
    "success_rate": 0.998
  },
  "database": {
    "query_time_ms": 35,
    "connection_pool_size": 5,
    "active_connections": 2
  },
  "memory": {
    "used_mb": 850,
    "peak_mb": 1200,
    "percentage": 42
  }
}
```

### GET /api/health

详细健康检查

**响应示例**:
```json
{
  "status": "healthy",
  "components": {
    "database": "healthy",
    "storage": "healthy",
    "causal_engine": "healthy",
    "bridges": {
      "qlib": "not_installed",
      "finrl": "not_installed",
      "hftbacktest": "not_installed"
    }
  },
  "version": "1.0.0"
}
```

## 错误处理

### 错误响应格式

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "参数验证失败",
    "details": {
      "field": "symbol",
      "issue": "symbol is required"
    }
  }
}
```

### HTTP状态码

- `200 OK`: 请求成功
- `400 Bad Request`: 请求参数错误
- `404 Not Found`: 资源不存在
- `500 Internal Server Error`: 服务器内部错误

### 常见错误代码

| 错误代码 | 描述 | 解决方案 |
|---------|------|----------|
| VALIDATION_ERROR | 参数验证失败 | 检查请求参数格式 |
| STRATEGY_NOT_FOUND | 策略不存在 | 确认策略ID正确 |
| INSUFFICIENT_DATA | 数据不足 | 提供更长时间范围 |
| RISK_LIMIT_EXCEEDED | 超过风险限制 | 调整仓位大小 |
| CAUSAL_ENGINE_ERROR | 因果分析失败 | 检查数据质量和参数 |

## 速率限制

默认限制:
- 每分钟: 100请求
- 每小时: 1000请求

超过限制返回 `429 Too Many Requests`

## 数据格式

### 日期时间
所有时间使用ISO 8601格式: `YYYY-MM-DDTHH:MM:SSZ`

### 数字格式
- 价格: 保留2位小数
- 百分比: 0-1之间的小数
- 数量: 整数或2位小数

### 枚举值

**订单方向**: `buy`, `sell`
**订单状态**: `pending`, `filled`, `cancelled`, `rejected`
**策略状态**: `active`, `draft`, `archived`
**经纪商模式**: `paper`, `live`

## 版本更新

### v1.0 (2024-01-15)
- 初始版本发布
- 核心交易功能
- 因果AI集成
- 回测引擎

## 支持和反馈

- GitHub Issues: https://github.com/pamelacai310-sketch/quant-trading-system/issues
- 文档: https://github.com/pamelacai310-sketch/quant-trading-system#readme
