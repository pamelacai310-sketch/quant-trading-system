# 量化交易系统配置指南

## 目录

- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [基本配置](#基本配置)
- [高级配置](#高级配置)
- [桥接模块配置](#桥接模块配置)
- [数据源配置](#数据源配置)
- [性能调优](#性能调优)
- [安全配置](#安全配置)
- [生产环境部署](#生产环境部署)

## 环境要求

### 系统要求
- **操作系统**: macOS, Linux, Windows
- **Python版本**: Python 3.9+
- **内存**: 最小4GB，推荐8GB+
- **磁盘空间**: 最小2GB，推荐10GB+

### Python依赖
```bash
# 核心依赖
pandas>=1.3.0
numpy>=1.21.0
flask>=2.0.0
sqlite3

# 可选依赖
ccxt>=1.0.0          # 交易所接口
ta>=0.10.0           # 技术分析
backtrader>=1.9.0    # 回测框架
```

## 安装步骤

### 1. 克隆仓库
```bash
git clone https://github.com/pamelacai310-sketch/quant-trading-system.git
cd quant-trading-system
```

### 2. 安装Python依赖
```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 安装核心依赖
pip install -r requirements.txt

# 安装V2可选模块
./install_upgrades.sh
```

### 3. 验证安装
```bash
# 运行测试
python3 -m unittest discover -s tests

# 启动系统
python3 run.py
```

## 基本配置

### 1. 环境变量配置

创建 `.env` 文件：
```bash
# 系统配置
PORT=8108
HOST=127.0.0.1
DEBUG=false

# 数据库配置
DATABASE_PATH=state/quant.db

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/trading.log

# 交易配置
INITIAL_CAPITAL=100000
DEFAULT_DATASET=gold_daily

# 风险管理
MAX_POSITION_SIZE=100000
MAX_DAILY_LOSS=10000
```

### 2. 数据目录配置

系统自动创建以下目录结构：
```
quant-trading-system/
├── data/               # 市场数据
│   ├── gold_daily.csv
│   ├── nasdaq_daily.csv
│   └── copper_daily.csv
├── state/              # 运行时状态
│   ├── quant.db       # SQLite数据库
│   ├── qlib_home/     # Qlib状态
│   ├── finrl_home/    # FinRL状态
│   └── exports/       # 导出文件
└── logs/               # 日志文件
```

### 3. 策略配置

策略JSON配置示例：
```json
{
  "name": "策略名称",
  "dataset": "数据集",
  "status": "active",
  "spec": {
    "symbol": "交易标的",
    "direction": "long_only",
    "indicators": [
      {"name": "指标名", "type": "指标类型", "window": 窗口期}
    ],
    "entry_rules": [
      {"left": "左侧值", "op": "操作符", "right": "右侧值"}
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
```

## 高级配置

### 1. 数据库配置

#### SQLite配置（默认）
```python
# 在 service.py 中配置
self.storage = Storage("state/quant.db")

# 数据库优化
storage.connection_pool_size = 5
storage.query_timeout = 30
storage.batch_size = 1000
```

#### PostgreSQL配置（可选）
```python
# 需要安装 psycopg2
import psycopg2
from psycopg2 import pool

# 创建连接池
connection_pool = psycopg2.pool.SimpleConnectionPool(
    1,  # 最小连接数
    10, # 最大连接数
    host="localhost",
    database="quant_trading",
    user="trader",
    password="secure_password"
)
```

### 2. 缓存配置

```python
# 启用查询缓存
CACHE_ENABLED = True
CACHE_TTL = 60  # 缓存过期时间（秒）
CACHE_MAX_SIZE = 1000  # 最大缓存条目

# Redis缓存（可选）
import redis
cache = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)
```

### 3. 并发配置

```python
# Flask配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# 线程池配置
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)
```

## 桥接模块配置

### 1. Python 3.11桥接配置

```bash
# 设置Python 3.11解释器路径
export PROJECT_BRIDGE_PYTHON=/opt/homebrew/bin/python3.11

# 或在配置文件中设置
PROJECT_BRIDGE_PYTHON=/usr/local/bin/python3.11
```

### 2. Qlib配置

```bash
# 安装Qlib
pip install pyqlib

# 配置数据目录
export QLIB_DATA_PATH=~/.qlib/qlib_data/cn_data

# 在系统中配置
qlib_config = {
    "provider_uri": "~/.qlib/qlib_data/cn_data",
    "region": "cn",  # 或 "us"
    "log_level": "WARNING"
}
```

### 3. FinRL-X配置

```bash
# 安装FinRL-X
pip install finrl

# 配置环境
export FINRL_DATA_DIR=./data/finrl
export WANDB_DISABLED=true  # 禁用wandb日志
```

### 4. hftbacktest配置

```bash
# 安装hftbacktest
pip install hftbacktest

# 配置
export HFTBACKTEST_DATA_DIR=./data/hft
export NUMBA_DISABLE_JIT=1  # 加快启动速度
```

### 5. Bloomberg配置（需要订阅）

```python
# Bloomberg API配置
bloomberg_config = {
    "host": "localhost",
    "port": 8194,
    "api_version": "v3",
    "authentication_required": True
}
```

## 数据源配置

### 1. 本地数据源

```python
# CSV数据配置
data_config = {
    "gold_daily": {
        "path": "data/gold_daily.csv",
        "date_column": "date",
        "columns": ["open", "high", "low", "close", "volume"]
    },
    "nasdaq_daily": {
        "path": "data/nasdaq_daily.csv",
        "date_column": "date",
        "columns": ["open", "high", "low", "close", "volume"]
    }
}
```

### 2. finshare数据源

```bash
# 安装finshare
pip install finshare

# 配置
export FINSHARE_CACHE_DIR=./data/finshare
export FINSHARE_MAX_CACHE_DAYS=30
```

### 3. OpenBB数据源

```bash
# 安装OpenBB
pip install openbb

# 配置
openbb_config = {
    "cache_enabled": True,
    "cache_dir": "./data/openbb",
    "provider": "yfinance"  # 数据提供商
}
```

### 4. ccxt交易所配置

```python
import ccxt

# 配置交易所
exchange_config = {
    "binance": {
        "api_key": "your_api_key",
        "secret": "your_secret",
        "enable_rate_limit": True,
        "options": {
            "defaultType": "spot"
        }
    }
}

# 初始化交易所
exchange = ccxt.binance(exchange_config['binance'])
```

## 性能调优

### 1. 数据库性能

```sql
-- 创建索引
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_orders_date ON orders(created_at);
CREATE INDEX idx_trades_date ON trades(entry_date);

-- 定期维护
VACUUM;
ANALYZE;
```

### 2. 应用性能

```python
# 启用查询缓存
CACHE_ENABLED = True

# 批量操作
BATCH_SIZE = 100

# 连接池
POOL_SIZE = 5

# 异步处理
ASYNC_ENABLED = True
MAX_WORKERS = 4
```

### 3. 内存优化

```python
# 数据类型优化
dtypes = {
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',
    'volume': 'int32'
}

# 数据分块
CHUNK_SIZE = 10000

# 内存限制
MAX_MEMORY_USAGE = 0.8  # 最大使用80%内存
```

## 安全配置

### 1. API认证

```python
# API密钥认证
API_KEYS = {
    "user1": "key1_secret",
    "user2": "key2_secret"
}

# JWT认证（推荐）
import jwt
JWT_SECRET = "your_secret_key"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 3600  # 1小时
```

### 2. 数据加密

```python
# 敏感数据加密
from cryptography.fernet import Fernet

# 生成加密密钥
key = Fernet.generate_key()
cipher = Fernet(key)

# 加密配置数据
encrypted_config = cipher.encrypt(config_data.encode())
```

### 3. 访问控制

```python
# IP白名单
ALLOWED_IPS = [
    "127.0.0.1",
    "192.168.1.0/24"
]

# 速率限制
RATE_LIMIT = {
    "per_minute": 100,
    "per_hour": 1000
}
```

### 4. 日志安全

```python
# 日志脱敏
import logging

def sanitize_log(record):
    # 移除敏感信息
    record.msg = record.msg.replace(API_KEY, "***")
    return record

logging.getLogger().addFilter(sanitize_log)
```

## 生产环境部署

### 1. 使用Gunicorn部署

```bash
# 安装Gunicorn
pip install gunicorn

# 启动服务
gunicorn -w 4 -b 0.0.0.0:8108 "quant_trade_system.server:create_app()"
```

### 2. 使用Systemd服务

创建 `/etc/systemd/system/quant-trading.service`:
```ini
[Unit]
Description=Quant Trading System
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/path/to/quant-trading-system
ExecStart=/path/to/venv/bin/python run.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl start quant-trading
sudo systemctl enable quant-trading
```

### 3. Nginx反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8108;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 4. Docker部署

创建 `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8108

CMD ["python", "run.py"]
```

构建和运行：
```bash
docker build -t quant-trading-system .
docker run -d -p 8108:8108 quant-trading-system
```

### 5. 监控和日志

```bash
# 日志轮转
cat > /etc/logrotate.d/quant-trading << EOF
/path/to/quant-trading-system/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 trader trader
}
EOF

# 监控脚本
#!/bin/bash
# check_system.sh
if ! curl -f http://localhost:8108/api/health > /dev/null 2>&1; then
    echo "System down! Restarting..."
    systemctl restart quant-trading
fi
```

## 配置检查清单

### 基础配置
- [ ] Python环境设置
- [ ] 依赖安装完成
- [ ] 数据目录创建
- [ ] 环境变量配置
- [ ] 基本功能测试通过

### 高级配置
- [ ] 数据库优化配置
- [ ] 缓存系统配置
- [ ] 并发处理配置
- [ ] 安全认证配置

### 生产部署
- [ ] 反向代理配置
- [ ] SSL证书配置
- [ ] 监控系统配置
- [ ] 日志轮转配置
- [ ] 自动备份配置

## 故障排除

### 常见问题

1. **端口占用**
```bash
# 检查端口占用
lsof -i :8108
# 修改端口
export PORT=8109
```

2. **权限错误**
```bash
# 修复权限
chmod +x run.py
chmod -R 755 state/
```

3. **依赖冲突**
```bash
# 重新安装依赖
pip install --force-reinstall -r requirements.txt
```

4. **内存不足**
```bash
# 增加交换空间
sudo swapon /swapfile
```

## 配置文件示例

完整配置示例可参考 `config/` 目录：
- `config.development.json` - 开发环境
- `config.production.json` - 生产环境
- `config.testing.json` - 测试环境

## 支持和帮助

- 查看日志: `tail -f logs/trading.log`
- 运行诊断: `python3 -m pytest tests/ -v`
- 提交问题: GitHub Issues
- 查看文档: `README.md`, `API_DOCUMENTATION.md`
