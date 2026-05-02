# MAE/MFE 量化策略诊断与优化系统

## 📖 系统概述

本系统基于 **MAE (Maximum Adverse Excursion，最大潜亏)** 和 **MFE (Maximum Favorable Excursion，最大潜盈)** 分析，提供了一套完整的量化策略诊断与优化工具链。

### 核心价值

传统策略评估只关注胜率、盈亏比和年化收益，但这些指标无法反映策略的**持仓体验**和**执行质量**。高收益可能伴随着：
- 深度套牢后回本（买点质量差）
- 利润大幅回吐（止盈能力差）
- 过山车式持仓（心理压力大）

MAE/MFE 散点图能够精准诊断策略在以下三个维度的缺陷：
1. **买点精确度** - 通过 MAE 分布诊断
2. **止损坚决度** - 通过 MAE 截断诊断
3. **止盈合理性** - 通过 MFE 利用率诊断

---

## 🎯 系统架构

```
量化交易系统
├── diagnostics/               # 诊断模块
│   ├── mae_mfe_diagnostics.py    # MAE/MFE 核心诊断
│   └── extreme_trade_analyzer.py # 极端交易分析
├── execution/                 # 执行模块
│   └── dynamic_stops.py          # 动态止损止盈
└── monitoring/                # 监控模块
    └── live_monitoring.py        # 实盘监控熔断
```

---

## 🚀 快速开始

### 安装依赖

```bash
pip install pandas numpy matplotlib seaborn
```

### 基础使用

```python
from quant_trade_system.diagnostics import MAE_MFE_Diagnostics
from quant_trade_system.execution import create_recommended_stop_manager

# 1. 创建诊断器
diagnostics = MAE_MFE_Diagnostics()

# 2. 从回测结果计算 MAE/MFE
diagnosis = diagnostics.calculate_from_backtest(
    backtest_result=backtest_result,
    price_data=price_data,
)

# 3. 打印诊断报告
diagnostics.print_diagnosis_report()

# 4. 绘制散点图
fig = diagnostics.plot_mae_mfe_scatter(save_path='diagnosis.png')

# 5. 应用动态止损止盈
stop_manager = create_recommended_stop_manager()
# 配置: 硬止损 -6%, 动态跟踪止盈 8%/3%
```

---

## 📊 四步优化流程

### 第一步：MFE 诊断与止盈机制优化

**问题诊断**：通过 MFE vs 最终盈亏散点图识别：
- **神仙线**（对角线）：完美止盈
- **傻白甜区**（右下角）：有浮盈没止盈，跌回成本
- **过山车区**（偏离对角线）：利润大幅回撤

**解决方案**：引入动态跟踪止盈

```python
from quant_trade_system.execution import DynamicStopManager, StopLossConfig, StopType

# 配置动态跟踪止盈
stop_manager = DynamicStopManager()
stop_manager.add_trailing_profit_stop(
    activation_threshold=8.0,  # 浮盈8%激活
    trail_pct=3.0,             # 回撤3%止盈
)
```

**实战规则**：
- 当单笔交易浮盈达到 8% 后，激活跟踪止盈
- 激活后，从最高点回撤超过 3%，立即止盈平仓
- 目标：将"傻白甜"和"过山车"交易转化为"神仙线"交易

---

### 第二步：MAE 诊断与止损/买点机制优化

**问题诊断**：通过 MAE vs 最终盈亏散点图识别：
- **完美交易区**（右上角）：MAE < -5%, 盈利 > 5%
- **艰难盈利区**（左上角）：MAE < -10%, 最终盈利（半山腰抄底）
- **糟糕交易区**（左下角）：MAE < -10%, 最终亏损（缺乏止损）

**解决方案 1**：设定硬止损

```python
# 配置硬止损
stop_manager.add_hard_stop(
    threshold_pct=-6.0,  # 亏损6%无条件止损
)
```

**解决方案 2**：优化买入过滤条件

```python
from quant_trade_system.execution import EntryFilter

# 创建买入过滤器
entry_filter = EntryFilter()
entry_filter.add_ma_trend_filter(
    fast_period=20,
    slow_period=60,
)
entry_filter.add_macd_confirmation()
entry_filter.add_volume_breakout_filter(min_volume_ratio=1.5)

# 使用
should_enter, reason = entry_filter.should_enter(price_data)
```

**实战规则**：
- 硬止损：任何持仓浮亏达到 -6% 或 -8%，立即止损
- 买入过滤：增加右侧确认信号（均线多头、MACD金叉、放量突破）
- 目标：切断左侧深度亏损，提升持仓体验

---

### 第三步：极端交易自动复盘

**目的**：从宏观散点图回归到微观 K 线图，找到策略底层逻辑漏洞

**使用方法**：

```python
from quant_trade_system.diagnostics import ExtremeTradeAnalyzer

# 创建分析器
analyzer = ExtremeTradeAnalyzer(
    extreme_threshold=10.0,   # 盈亏超过10%
    mae_threshold=-10.0,      # MAE超过-10%
    mfe_threshold=15.0,       # MFE超过15%
)

# 提取极端交易
extreme_trades = analyzer.extract_extreme_trades(
    backtest_result, price_data
)

# 分析模式
patterns = analyzer.analyze_patterns()

# 生成K线图
analyzer.plot_extreme_trade(
    trade=extreme_trades[0],
    price_data=price_data,
    save_path='extreme_trade.png',
)

# 生成洞察报告
report = analyzer.generate_insights_report()
print(report)
```

**归因分析**：
- 大赚交易有什么共性？（板块主升浪？均线支撑？）
- 大亏交易为什么止损这么迟？（追高？加速赶顶？）

---

### 第四步：实盘监控与熔断机制

**目的**：回测无法完全模拟实盘，必须建立基于 MAE/MFE 的实盘监控体系

**核心原则**：
- 牺牲部分理论收益，换取夏普比率提升和最大回撤下降
- 例如：年化从 85% 降至 28%，但最大回撤从 22% 降至 9% 是极其成功的优化

**使用方法**：

```python
from quant_trade_system.monitoring import LiveMonitor, HealthMetric

# 创建监控器
monitor = LiveMonitor(
    strategy_id="my_strategy",
    circuit_breaker_cooldown_hours=24,
)

# 更新持仓价格
report = monitor.update_position(
    symbol="AAPL",
    current_price=150.0,
)

# 检查是否允许新开仓
allowed, reason = monitor.should_allow_new_position()

# 生成周报
weekly_report = monitor.generate_weekly_report()

# 导出监控数据
monitor.export_monitoring_data('monitoring.json')
```

**熔断规则**：
- 每周复盘实盘交易的 MAE/MFE
- 如果盈利单的 MFE 始终比最终利润大很多（浮盈化为乌有）→ 止盈失效
- 如果 MAE 动不动突破 -10%（买点失效，止损失效）→ 停止实盘
- 结论：市场环境改变或策略失效，重新回测修改参数

---

## 📈 完整工作流示例

```python
# examples/mae_mfe_example.py

from quant_trade_system.diagnostics import MAE_MFE_Diagnostics
from quant_trade_system.execution import (
    OptimizedBacktester,
    create_recommended_stop_manager,
    create_recommended_entry_filter,
)
from quant_trade_system.monitoring import LiveMonitor

# 1. 初始诊断
diagnostics = MAE_MFE_Diagnostics()
diagnosis = diagnostics.calculate_from_backtest(
    backtest_result, price_data
)
diagnostics.print_diagnosis_report()

# 2. 优化回测
backtester = OptimizedBacktester(original_trades, price_data)

# 使用推荐配置
stop_manager = create_recommended_stop_manager()  # 硬止损-6% + 跟踪止盈8%/3%
entry_filter = create_recommended_entry_filter()  # 均线+MACD+放量

comparison = backtester.run_with_stops(
    stop_configs=stop_manager.stop_configs,
    entry_filter=entry_filter,
)

print(f"优化前胜率: {comparison['original']['win_rate']*100:.1f}%")
print(f"优化后胜率: {comparison['optimized']['win_rate']*100:.1f}%")
print(f"总收益变化: {comparison['improvement']['total_return_delta']*100:+.1f}%")

# 3. 建立监控
live_monitor = LiveMonitor(strategy_id="optimized_strategy")
```

---

## 🎨 散点图解读指南

### MAE vs 最终盈亏散点图

```
最终盈亏 (%)
     ↑
  20 |        ▲ 完美交易区
     |      ✗✗✗
     |    ✗✗✗✗✗
   0 |━━━━━━━━━━━━━━━━━
     | ✗✗✗✗✗
 -20 | ✗✗✗        ▲ 糟糕交易区
     |____________________→
     -15  -10   -5   0   MAE (%)
                ▲
            艰难盈利区
```

- **完美交易区**（右上）：MAE > -5%, 盈利 > 5%
  - 买点精准，持仓体验极佳
  - 继续保持当前买入策略

- **艰难盈利区**（左上）：MAE < -10%, 最终盈利
  - 半山腰抄底，深套后回本
  - 建议：增加右侧确认信号

- **糟糕交易区**（左下）：MAE < -10%, 最终亏损
  - 缺乏止损，无限亏损
  - 建议：立即设立硬止损

### MFE vs 最终盈亏散点图

```
最终盈亏 (%)
     ↑
  20 |      / 神仙线（完美止盈）
     |    /
     |  /
   0 |/_______________
     |
 -20 |  ▲ 傻白甜区
     |____________________→
       0   10   20   30  MFE (%)
           ▲
        过山车区
```

- **神仙线**（对角线）：MFE = 最终盈亏
  - 在最高点精准卖出，无利润回撤

- **傻白甜区**（右下角）：MFE > 10%, 最终亏损/小盈
  - 有浮盈没止盈，跌回成本
  - 建议：引入动态跟踪止盈

- **过山车区**（偏离对角线）：MFE 40%, 最终只赚 20%
  - 利润回撤极大
  - 建议：优化止盈时机

---

## 🔧 高级配置

### 自定义止损配置

```python
from quant_trade_system.execution import (
    DynamicStopManager,
    StopLossConfig,
    StopType,
    StopAction,
)

# 创建自定义配置
custom_config = StopLossConfig(
    stop_type=StopType.PROFIT_PROTECT,
    threshold_pct=5.0,              # 回撤5%止盈
    activation_threshold=10.0,      # 浮盈10%激活
    trail_pct=5.0,                  # 跟踪回撤5%
    action=StopAction.SELL_HALF,    # 平仓一半
)

stop_manager = DynamicStopManager()
stop_manager.stop_configs.append(custom_config)
```

### 自定义监控阈值

```python
from quant_trade_system.monitoring import (
    LiveMonitor,
    HealthMetric,
    MonitoringThreshold,
)

monitor = LiveMonitor(strategy_id="my_strategy")

# 添加自定义阈值
custom_threshold = MonitoringThreshold(
    metric=HealthMetric.MAE_CONTROL,
    warning_level=-3.0,
    critical_level=-6.0,
    trip_level=-8.0,
)

monitor.thresholds.append(custom_threshold)
```

---

## 📚 API 文档

### MAE_MFE_Diagnostics

```python
class MAE_MFE_Diagnostics:
    def calculate_from_backtest(
        self,
        backtest_result: BacktestResult,
        price_data: pd.DataFrame,
    ) -> MAE_MFE_Diagnosis:
        """从回测结果计算 MAE/MFE"""

    def plot_mae_mfe_scatter(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> Figure:
        """绘制 MAE/MFE 散点图"""

    def print_diagnosis_report(self):
        """打印诊断报告"""
```

### DynamicStopManager

```python
class DynamicStopManager:
    def add_hard_stop(
        self,
        threshold_pct: float = -6.0,
        action: StopAction = StopAction.SELL_ALL,
    ) -> DynamicStopManager:
        """添加硬止损"""

    def add_trailing_profit_stop(
        self,
        activation_threshold: float = 8.0,
        trail_pct: float = 3.0,
        action: StopAction = StopAction.SELL_ALL,
    ) -> DynamicStopManager:
        """添加动态跟踪止盈"""

    def check_stops(
        self,
        current_price: float,
        entry_price: float,
        current_date: datetime,
        position_type: str = 'long',
    ) -> Optional[StopTrigger]:
        """检查是否触发止损"""
```

### LiveMonitor

```python
class LiveMonitor:
    def update_position(
        self,
        symbol: str,
        current_price: float,
    ) -> Optional[MonitoringReport]:
        """更新持仓数据"""

    def should_allow_new_position(self) -> Tuple[bool, str]:
        """判断是否允许新开仓"""

    def generate_weekly_report(self) -> Dict[str, Any]:
        """生成周报"""
```

---

## 🛡️ 最佳实践

### 1. 诊断优先原则

永远先诊断再优化：
- ✅ 正确：运行 MAE/MFE 诊断 → 识别问题 → 针对性优化
- ❌ 错误：盲目添加止损止盈 → 可能适得其反

### 2. 持仓体验导向

接受"收益降低，体验提升"：
- 年化 85% + 回撤 22% → 年化 28% + 回撤 9% ✅
- 夏普比率从 1.2 提升到 2.5
- 实盘心理压力大幅降低

### 3. 熔断机制必须启用

回测表现好 ≠ 实盘能赚钱：
- 每周复盘 MAE/MFE
- 发现异常立即熔断
- 重新回测后再恢复实盘

### 4. 极端交易归因

不要只看统计数据：
- 大赚交易：找共性（可复制）
- 大亏交易：找原因（可避免）
- 结合 K 线图进行微观分析

---

## 📞 支持与反馈

- GitHub Issues: [提交问题](https://github.com/pamelacai310-sketch/quant-trading-system/issues)
- 文档：查看 `examples/mae_mfe_example.py` 获取完整示例

---

## 📄 许可证

MIT License

---

**祝您的策略优化顺利，实盘长红！** 🚀
