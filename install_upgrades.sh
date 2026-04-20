#!/bin/bash
# 量化交易系统升级安装脚本
# 安装V2新增的桥接模块依赖

set -e

echo "================================"
echo "量化交易系统 V2 升级安装"
echo "================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 项目目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "📁 项目目录: $PROJECT_DIR"
echo ""

# Python 3.11检测
echo "🔍 检测Python 3.11..."
PYTHON311_PATHS=(
    "/opt/homebrew/bin/python3.11"
    "/usr/local/bin/python3.11"
    "$(which python3.11)"
)

PYTHON311=""
for path in "${PYTHON311_PATHS[@]}"; do
    if [ -n "$path" ] && [ -f "$path" ]; then
        PYTHON311="$path"
        break
    fi
done

if [ -z "$PYTHON311" ]; then
    echo -e "${RED}✗ 未找到Python 3.11${NC}"
    echo "  请安装Python 3.11："
    echo "  brew install python@3.11"
    exit 1
fi

echo -e "${GREEN}✓ 找到Python 3.11: $PYTHON311${NC}"
echo ""

# 设置环境变量
export PROJECT_BRIDGE_PYTHON="$PYTHON311"

# 检查当前Python版本
CURRENT_PYTHON=$(python3 --version 2>&1 | awk '{print $2}')
echo "📊 当前Python版本: $CURRENT_PYTHON"
echo "🔧 桥接Python版本: $($PYTHON311 --version 2>&1 | awk '{print $2}')"
echo ""

# 安装函数
install_package() {
    local package_name="$1"
    local display_name="$2"
    local optional="$3"

    echo -e "\n📦 安装 $display_name..."

    if [ "$optional" = "true" ]; then
        echo -e "${YELLOW}⚠️  $display_name 需要额外订阅或权限${NC}"
        read -p "是否继续安装? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "跳过 $display_name"
            return
        fi
    fi

    if $PYTHON311 -m pip show "$package_name" &> /dev/null; then
        echo -e "${GREEN}✓ $display_name 已安装${NC}"
        $PYTHON311 -m pip show "$package_name" | grep Version
    else
        echo "正在安装 $package_name..."
        if $PYTHON311 -m pip install "$package_name"; then
            echo -e "${GREEN}✓ $display_name 安装成功${NC}"
        else
            echo -e "${RED}✗ $display_name 安装失败${NC}"
            if [ "$optional" = "false" ]; then
                return 1
            fi
        fi
    fi
}

# ========== V2新增模块 ==========

echo "================================"
echo "安装V2新增模块"
echo "================================"

# 1. Qlib (微软量化投资平台)
echo ""
echo "================================"
echo "1. Microsoft Qlib"
echo "================================"
echo "GitHub: https://github.com/microsoft/qlib"
echo "文档: https://qlib.readthedocs.io/"
echo ""
install_package "pyqlib" "Qlib" "false"

# 2. FinRL-X (强化学习交易框架)
echo ""
echo "================================"
echo "2. FinRL-X"
echo "================================"
echo "GitHub: https://github.com/AI4Finance-Foundation/FinRL-X"
echo "文档: https://finrl-x.readthedocs.io/"
echo ""
install_package "finrl" "FinRL-X" "false"

# 3. hftbacktest (高频交易回测)
echo ""
echo "================================"
echo "3. hftbacktest"
echo "================================"
echo "GitHub: https://github.com/notorious6/hftbacktest"
echo "文档: https://hftbacktest.readthedocs.io/"
echo ""
install_package "hftbacktest" "hftbacktest" "false"

# 4. Bloomberg API (需要订阅)
echo ""
echo "================================"
echo "4. Bloomberg API"
echo "================================"
echo "官网: https://www.bloomberg.com/professional/"
echo "API: https://developer.bloomberg.com/"
echo ""
install_package "blpapi" "Bloomberg API" "true"

# ========== 原有模块 ==========

echo ""
echo "================================"
echo "安装原有模块（如果未安装）"
echo "================================"

# 5. ccxt (交易所API)
echo ""
echo "================================"
echo "5. CCXT"
echo "================================"
echo "GitHub: https://github.com/ccxt/ccxt"
echo ""
install_package "ccxt" "CCXT" "false"

# 6. ta (技术分析库)
echo ""
echo "================================"
echo "6. TA-Lib / ta"
echo "================================"
echo "GitHub: https://github.com/bukosabino/ta"
echo ""
install_package "ta" "ta (Technical Analysis)" "false"

# 7. backtrader (回测框架)
echo ""
echo "================================"
echo "7. Backtrader"
echo "================================"
echo "GitHub: https://github.com/backtrader/backtrader"
echo ""
install_package "backtrader" "Backtrader" "false"

# 8. QuantLib (金融工程库)
echo ""
echo "================================"
echo "8. QuantLib"
echo "================================"
echo "GitHub: https://github.com/lballabio/QuantLib-SWIG"
echo ""
install_package "QuantLib-Python" "QuantLib" "false"

# ========== 可选模块 ==========

echo ""
echo "================================"
echo "可选模块"
echo "================================"

# 9. OpenBB (金融数据平台)
echo ""
echo "================================"
echo "9. OpenBB"
echo "================================"
echo "GitHub: https://github.com/OpenBB-finance/OpenBB"
echo ""
read -p "是否安装OpenBB? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    install_package "openbb" "OpenBB" "false"
fi

# ========== 测试安装 ==========

echo ""
echo "================================"
echo "测试安装"
echo "================================"

echo ""
echo "运行测试脚本..."
python3 -m pytest tests/test_upgrades.py -v || python3 tests/test_upgrades.py

# ========== 总结 ==========

echo ""
echo "================================"
echo "安装总结"
echo "================================"

echo ""
echo "已安装的桥接模块："

for bridge in qlib_bridge finrl_bridge hftbacktest_bridge bloomberg_bridge; do
    if [ -f "quant_trade_system/$bridge.py" ]; then
        echo -e "  ${GREEN}✓${NC} $bridge"
    fi
done

echo ""
echo "生态系统配置文件："

if [ -f "quant_trade_system/ecosystem_v2.py" ]; then
    echo -e "  ${GREEN}✓${NC} ecosystem_v2.py"
fi

if [ -f "install_upgrades.sh" ]; then
    echo -e "  ${GREEN}✓${NC} install_upgrades.sh"
fi

if [ -f "tests/test_upgrades.py" ]; then
    echo -e "  ${GREEN}✓${NC} tests/test_upgrades.py"
fi

echo ""
echo -e "${GREEN}================================"
echo "✓ V2升级安装完成！"
echo "================================${NC}"
echo ""
echo "下一步："
echo "  1. 运行测试: python3 -m pytest tests/test_upgrades.py"
echo "  2. 启动系统: python3 run.py"
echo "  3. 访问: http://127.0.0.1:8108"
echo ""
echo "提示："
echo "  - 使用Python 3.11运行桥接模块"
echo "  - 某些模块需要额外的API密钥或订阅"
echo "  - 参考各模块文档获取配置信息"
echo ""
