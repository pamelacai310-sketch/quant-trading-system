# 量化交易系统故障排除指南

## 目录

- [常见问题](#常见问题)
- [安装问题](#安装问题)
- [运行时问题](#运行时问题)
- [性能问题](#性能问题)
- [数据问题](#数据问题)
- [API问题](#api问题)
- [因果AI问题](#因果ai问题)
- [桥接模块问题](#桥接模块问题)
- [调试技巧](#调试技巧)
- [日志分析](#日志分析)

## 常见问题

### 问题1: 系统无法启动

**症状**: 运行 `python3 run.py` 后立即退出

**可能原因**:
1. 端口被占用
2. 依赖未安装
3. 配置文件错误

**解决方案**:
```bash
# 检查端口占用
lsof -i :8108
# 更换端口
python3 run.py --port 8109

# 检查依赖
pip install -r requirements.txt

# 查看错误日志
tail -f logs/trading.log
```

### 问题2: 数据库错误

**症状**: SQLite错误或数据库锁定

**解决方案**:
```bash
# 检查数据库文件
ls -la state/quant.db

# 删除锁定文件
rm state/quant.db-wal state/quant.db-shm

# 重新初始化数据库
rm state/quant.db
python3 run.py  # 系统会自动创建新数据库
```

### 问题3: 内存不足

**症状**: 系统运行缓慢或崩溃

**解决方案**:
```python
# 查看内存使用
import psutil
import os

process = psutil.Process(os.getpid())
print(f"内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# 清理缓存
import gc
gc.collect()

# 减少数据加载量
# 在配置中设置 CHUNK_SIZE = 5000
```

## 安装问题

### 问题: Python版本不兼容

**症状**: 导入错误或语法错误

**解决方案**:
```bash
# 检查Python版本
python3 --version  # 应该 >= 3.9

# 使用正确的Python版本
python3.9 run.py
# 或
python3.10 run.py

# 创建虚拟环境
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 问题: 依赖安装失败

**症状**: pip install 出错

**解决方案**:
```bash
# 升级pip
pip install --upgrade pip

# 清理缓存
pip cache purge

# 单独安装失败的包
pip install pandas==1.3.0
pip install numpy==1.21.0

# 使用conda（可选）
conda install pandas numpy flask
```

### 问题: 权限错误

**症状**: 无法创建文件或目录

**解决方案**:
```bash
# 修复目录权限
chmod -R 755 /path/to/quant-trading-system

# 修复数据库文件权限
chmod 644 state/quant.db

# 如果使用sudo运行
sudo python3 run.py
# 但不推荐，最好修复权限问题
```

## 运行时问题

### 问题1: 策略执行失败

**症状**: API返回错误或无交易信号

**诊断步骤**:
```bash
# 1. 检查策略是否存在
curl http://127.0.0.1:8108/api/strategies

# 2. 查看策略详情
# 从策略列表获取ID后
curl http://127.0.0.1:8108/api/strategies/STRAT_001

# 3. 手动测试策略
curl -X POST http://127.0.0.1:8108/api/execute \
  -H "Content-Type: application/json" \
  -d '{"strategy_id": "STRAT_001", "broker_mode": "paper"}'
```

**常见原因和解决方案**:

1. **数据不足**
```python
# 检查数据可用性
import pandas as pd
data = pd.read_csv("data/gold_daily.csv")
print(f"数据行数: {len(data)}")
print(f"日期范围: {data['date'].min()} 到 {data['date'].max()}")

# 确保有足够的历史数据（至少需要指标窗口期的2倍）
```

2. **指标计算错误**
```python
# 测试指标计算
from quant_trade_system.indicators import calculate_indicators
import pandas as pd

data = pd.read_csv("data/gold_daily.csv")
indicators = calculate_indicators(data, {
    "fast_ma": {"type": "sma", "window": 10},
    "slow_ma": {"type": "sma", "window": 30}
})

print(indicators.tail())
```

3. **入场条件未满足**
```python
# 检查最近的信号
from quant_trade_system.strategy_engine import run_strategy_once

result = run_strategy_once(strategy_spec, data)
print(f"当前状态: {result['current_position']}")
print(f"最新信号: {result['latest_signal']}")
print(f"入场条件满足: {result['entry_conditions_met']}")
```

### 问题2: 回测结果异常

**症状**: 回测收益过高或过低，交易次数不合理

**诊断步骤**:
```python
# 1. 检查回测参数
backtest_params = {
    "strategy_id": "STRAT_001",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
}

# 确保日期范围在数据范围内
# 2. 检查数据完整性
data = pd.read_csv("data/gold_daily.csv")
data['date'] = pd.to_datetime(data['date'])

# 检查缺失值
print(data.isnull().sum())

# 检查数据频率
print("数据频率分析:")
print(data['date'].diff().describe())
```

**常见问题**:

1. **前视偏差 (Look-ahead Bias)**
```python
# 错误：使用了未来数据
# indicators['signal'] = indicators['close'].shift(-1) > indicators['ma']

# 正确：只使用历史数据
indicators['signal'] = indicators['close'] > indicators['ma']
```

2. **交易成本未考虑**
```python
# 在回测中添加交易成本
def adjust_for_commission(trades, commission_rate=0.001):
    for trade in trades:
        trade['pnl'] -= trade['notional'] * commission_rate
    return trades
```

3. **滑点未考虑**
```python
# 添加滑点模拟
def add_slippage(orders, slippage_bps=5):
    for order in orders:
        if order['side'] == 'buy':
            order['filled_price'] *= (1 + slippage_bps/10000)
        else:
            order['filled_price'] *= (1 - slippage_bps/10000)
    return orders
```

## 性能问题

### 问题1: API响应缓慢

**症状**: API请求耗时超过1秒

**诊断**:
```python
import time
from functools import wraps

def time_api_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper

# 在server.py中使用
@app.route('/api/dashboard')
@time_api_call
def get_dashboard():
    # ...
```

**优化方案**:
```python
# 1. 启用查询缓存
from functools import lru_cache

@lru_cache(maxsize=100)
def get_portfolio_state_cached():
    return storage.get_portfolio_state()

# 2. 批量数据库操作
def batch_get_orders(order_ids):
    # 一次性获取多个订单
    pass

# 3. 异步处理
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.route('/api/backtest', methods=['POST'])
def async_backtest():
    data = request.json
    future = executor.submit(run_backtest, data)
    return jsonify({'task_id': id(future)})
```

### 问题2: 内存使用过高

**症状**: 内存占用超过2GB

**解决方案**:
```python
# 1. 优化数据类型
dtypes = {
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',
    'volume': 'int32'
}

data = pd.read_csv("data/gold_daily.csv", dtype=dtypes)

# 2. 分块处理
def process_large_data(file_path, chunk_size=10000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield process_chunk(chunk)

# 3. 及时释放内存
import gc
def free_memory():
    gc.collect()
```

### 问题3: 因果AI计算缓慢

**症状**: 因果分析耗时超过10秒

**优化方案**:
```python
# 1. 减少数据量
def run_causal_analysis_optimized(symbol, lookback_days=15):
    # 使用更短的回看期
    pass

# 2. 缓存结果
from functools import lru_cache

@lru_cache(maxsize=10)
def get_causal_graph_cached(symbol, days):
    return compute_causal_graph(symbol, days)

# 3. 并行计算
from concurrent.futures import ProcessPoolExecutor

def parallel_causal_discovery(symbols):
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(discover_causal, symbols)
    return list(results)
```

## 数据问题

### 问题1: 数据格式错误

**症状**: 无法读取CSV文件

**解决方案**:
```python
# 检查数据格式
import pandas as pd

try:
    data = pd.read_csv("data/gold_daily.csv")
    print(data.head())
    print(data.dtypes)
except Exception as e:
    print(f"读取错误: {e}")

# 修复常见问题
# 1. 日期格式
data['date'] = pd.to_datetime(data['date'])

# 2. 列名
data.columns = data.columns.str.lower()

# 3. 缺失值
data = data.dropna()
# 或
data = data.fillna(method='ffill')
```

### 问题2: 数据质量差

**症状**: 价格异常、成交量异常

**检测方法**:
```python
# 检测异常值
def detect_outliers(data):
    # 价格异常
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        z_scores = (data[col] - data[col].mean()) / data[col].std()
        outliers = data[abs(z_scores) > 3]
        if not outliers.empty:
            print(f"发现{col}异常值:")
            print(outliers[['date', col]])

    # 逻辑异常
    # High不能低于Low
    invalid = data[data['high'] < data['low']]
    if not invalid.empty:
        print("High < Low的记录:")
        print(invalid)

    # 成交量异常
    vol_outliers = data[data['volume'] <= 0]
    if not vol_outliers.empty:
        print("成交量异常:")
        print(vol_outliers)

# 使用
data = pd.read_csv("data/gold_daily.csv")
detect_outliers(data)
```

## API问题

### 问题1: 认证失败

**症状**: 401 Unauthorized

**解决方案**:
```python
# 检查API密钥
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

response = requests.get(
    'http://127.0.0.1:8108/api/strategies',
    headers=headers
)
```

### 问题2: 请求格式错误

**症状**: 400 Bad Request

**调试方法**:
```python
import json

# 验证JSON格式
try:
    json_str = json.dumps(strategy_spec)
    json.loads(json_str)  # 验证是否有效
except json.JSONDecodeError as e:
    print(f"JSON格式错误: {e}")

# 检查必需字段
required_fields = ['name', 'dataset', 'spec']
missing = [f for f in required_fields if f not in strategy_spec]
if missing:
    print(f"缺少必需字段: {missing}")
```

## 因果AI问题

### 问题1: 因果发现失败

**症状**: 因果图分析错误

**解决方案**:
```python
# 1. 检查数据质量
def check_data_for_causal(data):
    # 检查缺失值
    print(f"缺失值比例: {data.isnull().sum() / len(data)}")

    # 检查数据长度
    if len(data) < 50:
        print("数据量不足，建议至少50个数据点")

    # 检查方差
    for col in data.select_dtypes(include=[np.number]).columns:
        if data[col].var() == 0:
            print(f"警告: {col} 方差为0")

# 2. 简化模型
def run_simple_causal(data):
    # 使用更简单的因果关系测试
    from scipy import stats

    correlations = {}
    for i, col1 in enumerate(data.columns):
        for col2 in data.columns[i+1:]:
            corr, p_value = stats.pearsonr(data[col1], data[col2])
            correlations[(col1, col2)] = (corr, p_value)

    return correlations
```

### 问题2: 因果AI智能体未安装

**症状**: Causal-AI-Agent 不可用

**解决方案**:
```bash
# 1. 检查安装状态
curl http://127.0.0.1:8108/api/causal/status

# 2. 安装依赖（可选）
# causal_ai_agent 不是必需的，系统会自动降级到基础逻辑

# 3. 手动安装（如果需要）
git clone https://github.com/jdubbert/Causal-AI-Agent.git
cd Causal-AI-Agent
pip install -r requirements.txt
```

## 桥接模块问题

### 问题1: Python 3.11未找到

**症状**: 桥接模块无法启动

**解决方案**:
```bash
# 1. 检查Python 3.11是否安装
which python3.11

# 2. 安装Python 3.11
# macOS
brew install python@3.11

# Ubuntu
sudo apt install python3.11

# 3. 设置环境变量
export PROJECT_BRIDGE_PYTHON=/opt/homebrew/bin/python3.11

# 4. 验证
$PROJECT_BRIDGE_PYTHON --version
```

### 问题2: Qlib/FinRL未安装

**症状**: 桥接模块状态显示未安装

**解决方案**:
```bash
# 运行升级安装脚本
./install_upgrades.sh

# 或手动安装
pip install pyqlib        # Qlib
pip install finrl         # FinRL-X
pip install hftbacktest   # hftbacktest
```

## 调试技巧

### 启用调试模式

```python
# 在run.py中添加
import logging
logging.basicConfig(level=logging.DEBUG)

# 或设置环境变量
export DEBUG=true
python3 run.py
```

### 使用Python调试器

```python
# 在代码中插入断点
import pdb; pdb.set_trace()

# 使用ipdb（更友好）
import ipdb; ipdb.set_trace()

# 或使用breakpoint()（Python 3.7+）
breakpoint()
```

### 性能分析

```python
import cProfile
import pstats

# 分析函数性能
def profile_function(func):
    profiler = cProfile.Profile()
    profiler.enable()

    result = func()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # 打印前20个最耗时的函数

    return result

# 使用
profile_function(run_backtest)
```

### 内存分析

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # 你的代码
    pass

# 运行分析
python -m memory_profiler your_script.py
```

## 日志分析

### 查看日志

```bash
# 实时查看日志
tail -f logs/trading.log

# 查看错误日志
grep ERROR logs/trading.log

# 查看特定时间段的日志
grep "2024-01-15" logs/trading.log
```

### 日志分析脚本

```python
import re
from collections import Counter

def analyze_logs(log_file):
    with open(log_file, 'r') as f:
        logs = f.readlines()

    # 统计错误
    errors = [line for line in logs if 'ERROR' in line]
    print(f"错误数量: {len(errors)}")

    # 统计警告
    warnings = [line for line in logs if 'WARNING' in line]
    print(f"警告数量: {len(warnings)}")

    # 分析常见错误
    error_patterns = Counter()
    for error in errors:
        # 提取错误类型
        match = re.search(r'ERROR \s+ (\w+)', error)
        if match:
            error_patterns[match.group(1)] += 1

    print("常见错误类型:")
    for error_type, count in error_patterns.most_common(10):
        print(f"  {error_type}: {count}")

# 使用
analyze_logs('logs/trading.log')
```

## 监控脚本

### 系统健康检查

```bash
#!/bin/bash
# health_check.sh

echo "系统健康检查"
echo "================"

# 检查API
if curl -f http://127.0.0.1:8108/api/health > /dev/null 2>&1; then
    echo "✓ API运行正常"
else
    echo "✗ API无响应"
fi

# 检查数据库
if [ -f "state/quant.db" ]; then
    echo "✓ 数据库文件存在"
else
    echo "✗ 数据库文件缺失"
fi

# 检查数据文件
for file in data/*.csv; do
    if [ -f "$file" ]; then
        echo "✓ 数据文件: $file"
    else
        echo "✗ 数据文件缺失: $file"
    fi
done

# 检查磁盘空间
df -h | grep -E '/$|/home'

# 检查内存
free -h

echo "================"
echo "检查完成"
```

### 自动重启脚本

```bash
#!/bin/bash
# auto_restart.sh

while true; do
    if ! curl -f http://127.0.0.1:8108/api/health > /dev/null 2>&1; then
        echo "$(date): 系统无响应，正在重启..."
        pkill -f "python3 run.py"
        sleep 5
        nohup python3 run.py > logs/output.log 2>&1 &
        echo "$(date): 系统已重启"
    fi
    sleep 60
done
```

## 获取帮助

### 收集诊断信息

```bash
#!/bin/bash
# collect_info.sh

echo "=== 系统信息 ===" > diagnostic.txt
uname -a >> diagnostic.txt
python3 --version >> diagnostic.txt

echo -e "\n=== 依赖版本 ===" >> diagnostic.txt
pip list >> diagnostic.txt

echo -e "\n=== 系统状态 ===" >> diagnostic.txt
curl http://127.0.0.1:8108/api/health >> diagnostic.txt 2>&1

echo -e "\n=== 最近错误 ===" >> diagnostic.txt
tail -n 50 logs/trading.log | grep ERROR >> diagnostic.txt

echo -e "\n诊断信息已保存到 diagnostic.txt"
```

### 联系支持

如果问题仍未解决，请：

1. 检查日志文件: `logs/trading.log`
2. 运行诊断脚本: `bash collect_info.sh`
3. 提交GitHub Issue: https://github.com/pamelacai310-sketch/quant-trading-system/issues
4. 包含以下信息:
   - 系统环境（OS, Python版本）
   - 错误信息和日志
   - 重现步骤
   - 期望行为和实际行为

## 预防性维护

### 定期任务

```bash
#!/bin/bash
# maintenance.sh

# 1. 清理旧日志
find logs/ -name "*.log" -mtime +30 -delete

# 2. 备份数据库
cp state/quant.db backups/quant_$(date +%Y%m%d).db

# 3. 清理临时文件
rm -rf /tmp/quant_*

# 4. 优化数据库
sqlite3 state/quant.db "VACUUM;"

# 5. 检查磁盘空间
df -h | grep -E '/$|/home'

echo "维护任务完成"
```

### 定期检查清单

- [ ] 每日检查系统状态
- [ ] 每周检查日志错误
- [ ] 每月优化数据库
- [ ] 每月备份重要数据
- [ ] 每季度更新依赖版本
- [ ] 每季度审查策略性能

记住：预防胜于治疗！
