"""
Renaissance Technologies风格统计套利模块

核心功能：
1. 统计套利算法（均值回归）
2. 因子正交化（消除多重共线性）
3. 市场中性策略
4. 高频交易信号
5. 风险管理

参考：
- Simons, J. (Renaissance Technologies): Medallion Fund
- Brown, M. (Renaissance Technologies): Statistical Arbitrage
- Barglia, O. et al.: Statistical Arbitrage in the U.S. Equity Market

核心技术：
1. 因子模型（Factor Model）
2. 协整关系（Cointegration）
3. 配对交易（Pairs Trading）
4. 多因子择时（Multi-Factor Timing）
5. 风险归因（Risk Attribution）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")

try:
    from statsmodels.tsa.stattools import coint
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Install with: pip install statsmodels")


# ============================================================================
# 枚举定义
# ============================================================================

class StatArbStrategy(Enum):
    """统计套利策略"""
    PAIRS_TRADING = "pairs_trading"         # 配对交易
    FACTACT_NEUTRAL = "factor_neutral"       # 因子中性
    MEAN_REVERSION = "mean_reversion"       # 均值回归
    MOMENTUM = "momentum"                    # 动量
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"  # 统计套利
    MARKET_MAKING = "market_making"          # 做市


class OrthogonalizationMethod(Enum):
    """正交化方法"""
    PCA = "pca"                            # 主成分分析
    GRAM_SCHMIDT = "gram_schmidt"          # Gram-Schmidt正交化
    RESIDUAL = "residual"                   # 残差正交化
    SEQUENTIAL = "sequential"               # 序贯回归


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class PairsTradingSignal:
    """配对交易信号"""
    pair_id: str                           # 配对ID
    asset1: str                             # 资产1
    asset2: str                             # 资产2
    z_score: float                          # Z-score
    signal: str                             # 信号（long_short/neutral）
    confidence: float                       # 置信度
    expected_return: float                  # 预期收益
    stop_loss: float                        # 止损
    take_profit: float                      # 止盈
    entry_time: datetime                    # 入场时间
    exit_time: Optional[datetime]          # 出场时间


@dataclass
class FactorNeutralPosition:
    """因子中性仓位"""
    asset_id: str                           # 资产ID
    raw_exposure: float                     # 原始敞口
    factor_exposures: Dict[str, float]      # 因子敞口
    hedge_positions: Dict[str, float]       # 对冲仓位
    net_exposure: float                     # 净敞口
    risk_contribution: float                # 风险贡献


@dataclass
class OrthogonalFactor:
    """正交化因子"""
    factor_id: str                          # 因子ID
    factor_name: str                        # 因子名称
    orthogonal_factors: List[str]           # 正交化后的因子
    orthogonal_weights: List[float]         # 正交化权重
    explained_variance: float               # 解释方差比
    orthogonalization_method: OrthogonalizationMethod  # 正交化方法


@dataclass
class StatArbPosition:
    """统计套利仓位"""
    position_id: str                        # 仓位ID
    strategy: StatArbStrategy               # 策略
    assets: List[str]                        # 资产
    weights: np.ndarray                      # 权重
    entry_time: datetime                     # 入场时间
    exit_time: Optional[datetime]            # 出场时间
    entry_price: Dict[str, float]            # 入场价格
    current_price: Dict[str, float]          # 当前价格
    unrealized_pnl: float                   # 未实现盈亏
    risk_metrics: Dict[str, float]           # 风险指标


# ============================================================================
# Renaissance Technologies风格统计套利引擎
# ============================================================================

class RenaissanceStatArbEngine:
    """
    Renaissance Technologies风格统计套利引擎

    核心技术：
    1. 因子模型（多因子回归）
    2. 因子正交化（消除多重共线性）
    3. 协整关系（长期均衡关系）
    4. 均值回归（短期偏离回归）
    5. 风险管理（投资组合优化）
    """

    def __init__(
        self,
        lookback_period: int = 252,          # 回望期（1年）
        rebalance_frequency: str = "daily",   # 再平衡频率
        significance_level: float = 0.05,    # 显著性水平
    ):
        """
        初始化统计套利引擎

        参数:
            lookback_period: 回望期（交易日）
            rebalance_frequency: 再平衡频率
            significance_level: 显著性水平
        """
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.significance_level = significance_level

        # 因子模型
        self.factor_model = None
        self.factor_loadings = None
        self.factor_returns = None

        # 正交化因子
        self.orthogonal_factors = []

        # 配对关系
        self.cointegrated_pairs = []

        # 当前仓位
        self.current_positions = []

    def build_factor_model(
        self,
        returns: pd.DataFrame,
        factor_data: pd.DataFrame,
        method: str = "ols",
    ) -> Dict[str, Any]:
        """
        构建因子模型

        参数:
            returns: 资产收益率 (T x N)
            factor_data: 因子数据 (T x K)
            method: 回归方法（ols, ridge, lasso）

        返回:
            因子模型结果
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not installed")

        n_assets = returns.shape[1]
        n_factors = factor_data.shape[1]

        # 对每个资产进行时间序列回归
        factor_loadings = {}
        factor_residuals = {}

        for asset in returns.columns:
            y = returns[asset].values

            if method == "ols":
                model = LinearRegression()
            elif method == "ridge":
                model = Ridge(alpha=1.0)
            elif method == "lasso":
                model = Lasso(alpha=0.1)
            else:
                raise ValueError(f"Unknown method: {method}")

            # 训练模型
            model.fit(factor_data.values, y)

            # 保存因子载荷
            factor_loadings[asset] = model.coef_

            # 计算残差（特质收益）
            y_pred = model.predict(factor_data.values)
            residuals = y - y_pred
            factor_residuals[asset] = residuals

        # 计算因子收益（从截面回归）
        # 使用特异收益作为因变量
        residual_matrix = pd.DataFrame(factor_residuals)

        factor_returns = {}
        for factor in factor_data.columns:
            # 简化：使用因子值作为因子收益的代理
            factor_returns[factor] = factor_data[factor].values

        self.factor_model = {
            "loadings": factor_loadings,
            "residuals": factor_residuals,
            "factor_returns": factor_returns,
            "n_assets": n_assets,
            "n_factors": n_factors,
            "method": method,
        }

        # 保存到实例变量
        self.factor_loadings = factor_loadings

        return {
            "factor_loadings": factor_loadings,
            "residuals": factor_residuals,
            "r_squared": self._calculate_r_squared(returns, factor_data, factor_loadings),
        }

    def orthogonalize_factors(
        self,
        factor_data: pd.DataFrame,
        method: OrthogonalizationMethod = OrthogonalizationMethod.PCA,
        n_components: Optional[int] = None,
    ) -> List[OrthogonalFactor]:
        """
        因子正交化

        消除因子之间的多重共线性，提取独立的信号源。

        参数:
            factor_data: 原始因子数据 (T x K)
            method: 正交化方法
            n_components: 主成分数量

        返回:
            正交化因子列表
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not installed")

        orthogonal_factors = []

        if method == OrthogonalizationMethod.PCA:
            # 主成分分析
            n_components = n_components or min(factor_data.shape[1], factor_data.shape[0])

            pca = PCA(n_components=n_components)
            pca.fit(factor_data)

            # 提取主成分
            components = pca.transform(factor_data)

            for i in range(n_components):
                # 获取载荷
                loadings = pca.components_[i]

                # 计算解释方差比
                explained_var = pca.explained_variance_ratio_[i]

                # 选择重要因子（|loading| > 0.3）
                important_indices = np.where(np.abs(loadings) > 0.3)[0]

                orthogonal_factors.append(OrthogonalFactor(
                    factor_id=f"PC{i+1}",
                    factor_name=f"Principal Component {i+1}",
                    orthogonal_factors=[factor_data.columns[idx] for idx in important_indices],
                    orthogonal_weights=[loadings[idx] for idx in important_indices],
                    explained_variance=explained_var,
                    orthogonalization_method=method,
                ))

        elif method == OrthogonalizationMethod.GRAM_SCHMIDT:
            # Gram-Schmidt正交化
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(factor_data)

            # 执行Gram-Schmidt正交化
            orthogonal_matrix = np.zeros_like(scaled_data)
            orthogonal_vectors = []

            for i in range(scaled_data.shape[1]):
                vector = scaled_data[:, i]

                # 减去在所有前面正交向量上的投影
                for ortho_vector in orthogonal_vectors:
                    projection = np.dot(vector, ortho_vector) * ortho_vector
                    vector = vector - projection

                # 归一化
                norm = np.linalg.norm(vector)
                if norm > 1e-10:
                    vector = vector / norm

                orthogonal_matrix[:, i] = vector
                orthogonal_vectors.append(vector)

                # 计算每个原始因子的权重
                orthogonal_factors.append(OrthogonalFactor(
                    factor_id=f"GS{i+1}",
                    factor_name=f"Gram-Schmidt Factor {i+1}",
                    orthogonal_factors=list(factor_data.columns),
                    orthogonal_weights=list(vector),
                    explained_variance=1.0 / (i + 1),  # 简化
                    orthogonalization_method=method,
                ))

        elif method == OrthogonalizationMethod.RESIDUAL:
            # 残差正交化
            # 逐步回归，取残差

            remaining_factors = factor_data.columns.tolist()
            orthogonal_factors_list = []

            for i, factor in enumerate(factor_data.columns):
                # 使用前序因子回归当前因子
                if i == 0:
                    residuals = factor_data[factor].values
                else:
                    # 使用前面的正交因子预测
                    if orthogonal_factors_list:
                        X = np.column_stack([
                            of.orthogonal_weights
                            for of in orthogonal_factors_list
                        ])

                        model = LinearRegression()
                        model.fit(X, factor_data[factor].values)
                        predicted = model.predict(X)
                        residuals = factor_data[factor].values - predicted
                    else:
                        residuals = factor_data[factor].values

                # 计算残差的方差
                residual_var = np.var(residuals)
                original_var = np.var(factor_data[factor].values)
                explained_var = 1 - residual_var / original_var

                orthogonal_factors_list.append(OrthogonalFactor(
                    factor_id=f"Residual{i+1}",
                    factor_name=f"Residual Factor {i+1} ({factor})",
                    orthogonal_factors=[factor],
                    orthogonal_weights=[1.0],
                    explained_variance=explained_var,
                    orthogonalization_method=method,
                ))

            orthogonal_factors = orthogonal_factors_list

        self.orthogonal_factors = orthogonal_factors

        return orthogonal_factors

    def find_cointegrated_pairs(
        self,
        price_data: pd.DataFrame,
        price_type: str = "close",
        max_pairs: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        寻找协整对

        参数:
            price_data: 价格数据 (T x N)
            price_type: 价格类型（close/ midpoint）
            max_pairs: 最大配对数量

        返回:
            卐整对列表
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels not installed")

        pairs = []
        n_assets = len(price_data.columns)

        # 对所有资产对进行协整检验
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                asset1 = price_data.columns[i]
                asset2 = price_data.columns[j]

                # 提取价格数据
                S1 = price_data[asset1].values
                S2 = price_data[asset2].values

                # 协整检验
                try:
                    score, pvalue, _ = coint(S1, S2)

                    if pvalue < self.significance_level:
                        # 协整关系存在
                        # 计算对冲比率
                        model = LinearRegression()
                        model.fit(S1.reshape(-1, 1), S2)
                        hedge_ratio = model.coef_[0]

                        # 计算残差（价差）
                        spread = S2 - hedge_ratio * S1
                        spread_mean = spread.mean()
                        spread_std = spread.std()

                        pairs.append({
                            "pair_id": f"{asset1}_{asset2}",
                            "asset1": asset1,
                            "asset2": asset2,
                            "hedge_ratio": hedge_ratio,
                            "coint_statistic": score,
                            "p_value": pvalue,
                            "spread_mean": spread_mean,
                            "spread_std": spread_std,
                            "half_life": self._calculate_half_life(spread),
                        })

                        if len(pairs) >= max_pairs:
                            return pairs

                except Exception:
                    continue

        # 按p值排序
        pairs.sort(key=lambda x: x["p_value"])

        self.cointegrated_pairs = pairs

        return pairs

    def generate_pairs_trading_signals(
        self,
        current_prices: Dict[str, float],
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss: float = 3.0,
    ) -> List[PairsTradingSignal]:
        """
        生成配对交易信号

        参数:
            current_prices: 当前价格
            entry_threshold: 入场阈值（标准差倍数）
            exit_threshold: 出场阈值（标准差倍数）
            stop_loss: 止损（标准差倍数）

        返回:
            配对交易信号列表
        """
        signals = []

        for pair in self.cointegrated_pairs:
            asset1 = pair["asset1"]
            asset2 = pair["asset2"]

            if asset1 not in current_prices or asset2 not in current_prices:
                continue

            # 计算当前价差
            S1_current = current_prices[asset1]
            S2_current = current_prices[asset2]
            spread_current = S2_current - pair["hedge_ratio"] * S1_current

            # 计算Z-score
            z_score = (spread_current - pair["spread_mean"]) / pair["spread_std"]

            # 生成信号
            signal = None
            if z_score < -entry_threshold:
                # 价差过低，买入价差（做多asset2，做空asset1）
                signal = "long_short"
            elif z_score > entry_threshold:
                # 价差过高，卖出价差（做空asset2，做多asset1）
                signal = "short_long"
            elif abs(z_score) < exit_threshold:
                # 价差回归，平仓
                signal = "neutral"

            if signal:
                # 计算预期收益（回归到均值）
                expected_spread = pair["spread_mean"]
                expected_return = abs(spread_current - expected_spread) / S1_current

                pairs_signal = PairsTradingSignal(
                    pair_id=pair["pair_id"],
                    asset1=asset1,
                    asset2=asset2,
                    z_score=z_score,
                    signal=signal,
                    confidence=min(abs(z_score) / 3.0, 1.0),
                    expected_return=expected_return,
                    stop_loss=stop_loss,
                    take_profit=abs(z_score) * 0.8,
                    entry_time=datetime.now(),
                    exit_time=None,
                )

                signals.append(pairs_signal)

        return signals

    def execute_mean_reversion(
        self,
        prices: pd.Series,
        window: int = 20,
        std_multiplier: float = 2.0,
    ) -> Dict[str, Any]:
        """
        执行均值回归策略

        参数:
            prices: 价格序列
            window: 移动窗口
            std_multiplier: 标准差倍数

        返回:
            策略结果
        """
        # 计算移动平均和标准差
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()

        # 计算Z-score
        z_scores = (prices - rolling_mean) / rolling_std

        # 生成信号
        signals = pd.Series(index=prices.index, dtype=object)
        signals[z_scores < -std_multiplier] = "long"   # 价格过低，做多
        signals[z_scores > std_multiplier] = "short"  # 价格过高，做空
        signals[abs(z_scores) <= std_multiplier * 0.5] = "neutral"  # 价格回归，平仓

        # 计算收益
        returns = prices.pct_change()
        strategy_returns = returns.shift(-1)  # 提前一天的收益

        # 多头收益：信号为long时的次日收益
        long_mask = signals == "long"
        short_mask = signals == "short"

        long_returns = strategy_returns[long_mask]
        short_returns = -strategy_returns[short_mask]

        total_return = pd.concat([long_returns, short_returns]).sum()

        return {
            "signals": signals,
            "z_scores": z_scores,
            "total_return": total_return,
            "sharpe_ratio": total_return / returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            "win_rate": (strategy_returns[signals == "long"] > 0).sum() / (signals == "long").sum() if (signals == "long").sum() > 0 else 0,
        }

    def build_market_neutral_portfolio(
        self,
        returns: pd.DataFrame,
        factor_data: pd.DataFrame,
        target_risk: float = 0.15,
    ) -> Dict[str, Any]:
        """
        构建市场中性组合

        参数:
            returns: 资产收益率
            factor_data: 因子数据
            target_risk: 目标风险（波动率）

        返回:
            组合权重和性能
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not installed")

        # 构建因子模型
        model_result = self.build_factor_model(returns, factor_data)

        # 对每个资产，计算因子暴露
        asset_factor_exposures = {}
        for asset in returns.columns:
            asset_factor_exposures[asset] = [
                model_result["factor_loadings"][asset][i]
                for i in range(len(model_result["factor_loadings"][asset]))
            ]

        # 计算组合权重（最小化因子暴露）
        # 简化：等权重组合，因子中性通过择股实现

        # 预测特异收益（风险调整后）
        predicted_returns = {}
        for asset in returns.columns:
            # 使用历史平均收益作为预期
            predicted_returns[asset] = returns[asset].mean()

        # 计算协方差矩阵
        cov_matrix = returns.cov().values

        # 均值方差优化（简化：等权重）
        n = len(returns.columns)
        weights = np.ones(n) / n

        # 计算组合收益和风险
        portfolio_return = np.dot(weights, [predicted_returns[asset] for asset in returns.columns])
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

        return {
            "weights": dict(zip(returns.columns, weights)),
            "expected_return": portfolio_return,
            "expected_risk": portfolio_risk,
            "sharpe_ratio": portfolio_return / portfolio_risk if portfolio_risk > 0 else 0,
            "factor_loadings": model_result["factor_loadings"],
            "r_squared": model_result["r_squared"],
        }

    def calculate_half_life(
        self,
        spread: np.ndarray,
    ) -> float:
        """
        计算均值回归的半衰期

        参数:
            spread: 价差序列

        返回:
            半衰期（交易日）
        """
        # 使用Ornstein-Uhlenbeck过程的半衰期公式
        # half_life = -log(0.5) / theta
        # 其中theta是均值回归速度

        # 简化：使用AR(1)模型估计
        from sklearn.linear_model import LinearRegression

        spread_lag = np.roll(spread, 1)
        spread_lag[0] = spread[1]  # 填充第一个值

        model = LinearRegression()
        model.fit(spread_lag[1:].reshape(-1, 1), spread[1:])

        # AR(1)系数
        phi = model.coef_[0]

        # 均值回归速度
        theta = 1 - phi

        if theta > 0:
            half_life = -np.log(0.5) / theta
        else:
            half_life = float('inf')

        return half_life

    def backtest_stat_arb_strategy(
        self,
        price_data: pd.DataFrame,
        start_date: str,
        end_date: str,
        strategy: StatArbStrategy = StatArbStrategy.PAIRS_TRADING,
    ) -> Dict[str, Any]:
        """
        回测统计套利策略

        参数:
            price_data: 价格数据
            start_date: 开始日期
            end_date: 结束日期
            strategy: 策略类型

        返回:
            回测结果
        """
        # 筛选日期范围
        prices = price_data.loc[start_date:end_date]

        if strategy == StatArbStrategy.PAIRS_TRADING:
            # 寻找协整对
            pairs = self.find_cointegrated_pairs(prices)

            # 生成交易信号
            signals_list = []
            for date in prices.index:
                current_prices = prices.loc[date].to_dict()
                signals = self.generate_pairs_trading_signals(current_prices)
                signals_list.append(signals)

            # 计算收益（简化）
            total_return = 0.0
            win_rate = 0.5

        elif strategy == StatArbStrategy.MEAN_REVERSION:
            # 对每个资产执行均值回归
            returns_list = []

            for asset in prices.columns:
                result = self.execute_mean_reversion(prices[asset])
                returns_list.append(result["total_return"])

            total_return = np.mean(returns_list)

        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        return {
            "strategy": strategy,
            "total_return": total_return,
            "annualized_return": total_return * 252 / len(prices),
            "sharpe_ratio": total_return / np.std(prices.pct_change().mean(axis=1)) * np.sqrt(252),
            "start_date": start_date,
            "end_date": end_date,
        }

    def _calculate_r_squared(
        self,
        returns: pd.DataFrame,
        factor_data: pd.DataFrame,
        factor_loadings: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """计算R²"""
        r_squared = {}

        for asset in returns.columns:
            y = returns[asset].values

            # 预测值
            y_pred = np.dot(factor_data.values, factor_loadings[asset])

            # R²
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)

            r_squared[asset] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return r_squared


# ============================================================================
# 工厂函数
# ============================================================================

def create_renaissance_stat_arb_engine(
    lookback_period: int = 252,
    rebalance_frequency: str = "daily",
) -> RenaissanceStatArbEngine:
    """创建Renaissance Technologies风格统计套利引擎"""
    return RenaissanceStatArbEngine(
        lookback_period=lookback_period,
        rebalance_frequency=rebalance_frequency,
    )


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 创建统计套利引擎
    engine = create_renaissance_stat_arb_engine()

    print("✅ Renaissance Technologies风格统计套利引擎创建成功")
    print(f"  回望期: {engine.lookback_period}交易日")
    print(f"  再平衡频率: {engine.rebalance_frequency}")
