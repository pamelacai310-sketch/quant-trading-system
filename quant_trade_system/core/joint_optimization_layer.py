"""
联合优化层 - Joint Optimization Layer

核心功能：
1. 将Sharpe比率和CVaR嵌入训练目标
2. 组合优化和信号生成统一为端到端目标函数
3. 仓位计算内嵌入训练目标（不再固定映射）
4. 集成Taleb杠铃策略约束
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

from quant_trade_system.core.statistical_learning_layer import (
    StatisticalLearningLayer,
    ModelType,
    LearningObjectiveConfig,
    OptimizationMetric,
    TrainingResult,
    PredictionResult,
)
from quant_trade_system.core.feature_engineering_layer import FeatureMatrix

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: CVXPY not installed. Install with: pip install cvxpy")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


# ============================================================================
# 枚举定义
# ============================================================================

class OptimizationStrategy(Enum):
    """优化策略"""
    MEAN_VARIANCE = "mean_variance"          # 均值-方差
    RISK_PARITY = "risk_parity"              # 风险平价
    TALEB_BARBELL = "taleb_barbell"          # 塔勒布杠铃
    EQUAL_WEIGHT = "equal_weight"            # 等权重
    MAX_DIVERSIFICATION = "max_diversification"  # 最大化分散
    MAX_SHARPE = "max_sharpe"                # 最大化夏普
    MIN_CVAR = "min_cvar"                    # 最小化CVaR


class RiskMeasure(Enum):
    """风险度量"""
    VARIANCE = "variance"                    # 方差
    CVAR = "cvar"                           # 条件风险价值
    MAX_DRAWDOWN = "max_drawdown"           # 最大回撤
    DRAWDOWN_DURATION = "drawdown_duration" # 回撤持续时间


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class PortfolioConstraint:
    """组合约束"""
    max_position_size: float = 0.3         # 单个资产最大仓位
    max_total_exposure: float = 1.0         # 最大总敞口
    min_cash_ratio: float = 0.05            # 最小现金比例
    max_sector_exposure: float = 0.5        # 单个行业最大敞口
    turnover_limit: float = 0.5             # 换手率限制
    beta_range: Tuple[float, float] = (0.5, 1.5)  # Beta范围


@dataclass
class TalebBarbellConfig:
    """塔勒布杠铃配置"""
    safe_asset_ratio: float = 0.85          # 安全资产比例（85-90%）
    risky_asset_ratio: float = 0.15         # 风险资产比例（10-15%）

    # 安全资产特征
    safe_asset_types: List[str] = field(default_factory=lambda: [
        "treasury_bonds",
        "high_grade_corporate_bonds",
        "cash_equivalents",
    ])

    # 风险资产特征
    risky_asset_types: List[str] = field(default_factory=lambda: [
        "growth_stocks",
        "emerging_markets",
        "commodities",
        "cryptocurrencies",
    ])

    # 风险资产选择标准
    min_momentum: float = 0.5               # 最小动量得分
    max_correlation_with_safe: float = 0.3  # 与安全资产最大相关性
    min_asymmetric_return: float = 1.5      # 最小非对称收益


@dataclass
class OptimizationResult:
    """优化结果"""
    optimal_weights: np.ndarray             # 最优权重
    expected_return: float                  # 预期收益
    expected_risk: float                    # 预期风险
    sharpe_ratio: float                     # 夏普比率
    cvar: float                             # 条件风险价值
    max_drawdown: float                     # 最大回撤
    turnover: float                         # 换手率
    diversification_ratio: float            # 分散化比率
    safe_asset_weights: Dict[str, float]    # 安全资产权重
    risky_asset_weights: Dict[str, float]   # 风险资产权重
    optimization_time: float                # 优化时间（秒）
    convergence_status: str                 # 收敛状态


@dataclass
class JointTrainingResult:
    """联合训练结果"""
    model: Any                              # 训练好的模型
    optimization_history: List[OptimizationResult]  # 优化历史
    best_sharpe: float                      # 最佳夏普比率
    best_cvar: float                        # 最佳CVaR
    best_weights: np.ndarray                # 最佳权重
    training_epochs: int                    # 训练轮数
    convergence_curve: List[float]          # 收敛曲线


# ============================================================================
# PyTorch模型：端到端优化
# ============================================================================

if PYTORCH_AVAILABLE:
    class EndToEndPortfolioModel(nn.Module):
        """
        端到端组合优化模型

        将信号生成和仓位计算统一为一个优化目标
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] = [256, 128, 64],
            output_dim: int = 10,  # 资产数量
            dropout: float = 0.1,
        ):
            super().__init__()

            self.input_dim = input_dim
            self.output_dim = output_dim

            # 特征编码器
            layers = []
            prev_dim = input_dim
            for dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = dim

            self.encoder = nn.Sequential(*layers)

            # 信号生成头（预测收益）
            self.return_predictor = nn.Sequential(
                nn.Linear(prev_dim, prev_dim // 2),
                nn.ReLU(),
                nn.Linear(prev_dim // 2, output_dim),
            )

            # 风险预测头（预测风险）
            self.risk_predictor = nn.Sequential(
                nn.Linear(prev_dim, prev_dim // 2),
                nn.ReLU(),
                nn.Linear(prev_dim // 2, output_dim),
                nn.Softplus(),  # 确保输出为正
            )

            # 仓位生成头
            self.position_generator = nn.Sequential(
                nn.Linear(prev_dim, prev_dim // 2),
                nn.ReLU(),
                nn.Linear(prev_dim // 2, output_dim),
                nn.Softmax(dim=-1),  # 确保权重和为1
            )

        def forward(self, x):
            """
            前向传播

            参数:
                x: (batch_size, input_dim)

            返回:
                returns: (batch_size, output_dim) 预测收益
                risks: (batch_size, output_dim) 预测风险
                positions: (batch_size, output_dim) 建议仓位
            """
            # 编码
            encoded = self.encoder(x)

            # 预测收益
            returns = self.return_predictor(encoded)

            # 预测风险
            risks = self.risk_predictor(encoded)

            # 生成仓位
            positions = self.position_generator(encoded)

            return returns, risks, positions


    class JointOptimizationLoss(nn.Module):
        """
        联合优化损失函数

        同时优化：
        1. 预测准确性
        2. 组合夏普比率
        3. 组合CVaR
        4. 交易成本
        """

        def __init__(
            self,
            prediction_weight: float = 0.3,
            sharpe_weight: float = 0.3,
            cvar_weight: float = 0.2,
            cost_weight: float = 0.1,
            cvar_confidence: float = 0.95,
        ):
            super().__init__()
            self.prediction_weight = prediction_weight
            self.sharpe_weight = sharpe_weight
            self.cvar_weight = cvar_weight
            self.cost_weight = cost_weight
            self.cvar_confidence = cvar_confidence

        def forward(
            self,
            predicted_returns,
            predicted_risks,
            positions,
            true_returns,
            prev_positions=None,
        ):
            """
            计算联合损失

            参数:
                predicted_returns: (batch_size, n_assets)
                predicted_risks: (batch_size, n_assets)
                positions: (batch_size, n_assets)
                true_returns: (batch_size, n_assets)
                prev_positions: (batch_size, n_assets) 前一期仓位

            返回:
                loss: 标量
            """
            # 1. 预测损失（MSE）
            prediction_loss = nn.MSELoss()(predicted_returns, true_returns)

            # 2. 计算组合收益
            portfolio_returns = (positions * true_returns).sum(dim=1)

            # 3. 夏普比率损失（负的夏普比率）
            if portfolio_returns.std() > 1e-8:
                sharpe = portfolio_returns.mean() / (portfolio_returns.std() + 1e-8)
                sharpe_loss = -sharpe
            else:
                sharpe_loss = torch.tensor(0.0)

            # 4. CVaR损失
            # 计算组合收益分布的VaR和CVaR
            var_index = int((1 - self.cvar_confidence) * len(portfolio_returns))
            sorted_returns, _ = torch.sort(portfolio_returns)
            var = sorted_returns[var_index]
            cvar = sorted_returns[:var_index+1].mean()
            cvar_loss = -cvar  # 最大化CVaR（最小化负CVaR）

            # 5. 交易成本损失
            if prev_positions is not None:
                turnover = (positions - prev_positions).abs().sum(dim=1).mean()
                cost_loss = turnover
            else:
                cost_loss = torch.tensor(0.0)

            # 组合损失
            total_loss = (
                self.prediction_weight * prediction_loss +
                self.sharpe_weight * sharpe_loss +
                self.cvar_weight * cvar_loss +
                self.cost_weight * cost_loss
            )

            return total_loss


# ============================================================================
# 联合优化层核心类
# ============================================================================

class JointOptimizationLayer:
    """
    联合优化层

    核心功能：
    1. 将Sharpe/CVaR嵌入训练目标
    2. 组合优化和信号生成为端到端优化
    3. 仓位计算内嵌入训练
    4. 集成Taleb杠铃约束
    """

    def __init__(
        self,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.TALEB_BARBELL,
        portfolio_constraints: Optional[PortfolioConstraint] = None,
        taleb_config: Optional[TalebBarbellConfig] = None,
        statistical_layer: Optional[StatisticalLearningLayer] = None,
    ):
        """
        初始化联合优化层

        参数:
            optimization_strategy: 优化策略
            portfolio_constraints: 组合约束
            taleb_config: 塔勒布杠铃配置
            statistical_layer: 统计学习层（可选）
        """
        self.optimization_strategy = optimization_strategy
        self.portfolio_constraints = portfolio_constraints or PortfolioConstraint()
        self.taleb_config = taleb_config or TalebBarbellConfig()
        self.statistical_layer = statistical_layer

        # 训练结果
        self.joint_training_result: Optional[JointTrainingResult] = None
        self.is_fitted = False

    def optimize_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        asset_types: List[str],
        current_positions: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """
        优化组合权重

        参数:
            expected_returns: (n_assets,) 预期收益
            covariance_matrix: (n_assets, n_assets) 协方差矩阵
            asset_types: (n_assets,) 资产类型
            current_positions: (n_assets,) 当前仓位

        返回:
            OptimizationResult
        """
        start_time = datetime.now()

        n_assets = len(expected_returns)

        if self.optimization_strategy == OptimizationStrategy.TALEB_BARBELL:
            result = self._optimize_taleb_barbell(
                expected_returns,
                covariance_matrix,
                asset_types,
                current_positions,
            )
        elif self.optimization_strategy == OptimizationStrategy.MAX_SHARPE:
            result = self._optimize_max_sharpe(
                expected_returns,
                covariance_matrix,
                current_positions,
            )
        elif self.optimization_strategy == OptimizationStrategy.MIN_CVAR:
            result = self._optimize_min_cvar(
                expected_returns,
                covariance_matrix,
                current_positions,
            )
        elif self.optimization_strategy == OptimizationStrategy.RISK_PARITY:
            result = self._optimize_risk_parity(
                covariance_matrix,
                current_positions,
            )
        elif self.optimization_strategy == OptimizationStrategy.EQUAL_WEIGHT:
            result = self._optimize_equal_weight(
                n_assets,
                current_positions,
            )
        else:
            result = self._optimize_mean_variance(
                expected_returns,
                covariance_matrix,
                current_positions,
            )

        result.optimization_time = (datetime.now() - start_time).total_seconds()

        return result

    def train_joint_model(
        self,
        feature_matrix: FeatureMatrix,
        returns: np.ndarray,
        asset_types: List[str],
        epochs: int = 100,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> JointTrainingResult:
        """
        训练端到端联合模型

        参数:
            feature_matrix: 特征矩阵
            returns: (T, n_assets) 收益率矩阵
            asset_types: (n_assets,) 资产类型
            epochs: 训练轮数
            learning_rate: 学习率
            validation_split: 验证集比例

        返回:
            JointTrainingResult
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")

        start_time = datetime.now()

        # 准备数据
        X = feature_matrix.data.values
        T, n_assets = returns.shape

        # 归一化
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_normalized = (X - X_mean) / X_std

        returns_mean = returns.mean(axis=0)
        returns_std = returns.std(axis=0) + 1e-8
        returns_normalized = (returns - returns_mean) / returns_std

        # 划分训练集和验证集
        split_idx = int(T * (1 - validation_split))

        X_train = X_normalized[:split_idx]
        returns_train = returns_normalized[:split_idx]
        X_val = X_normalized[split_idx:]
        returns_val = returns_normalized[split_idx:]

        # 创建模型
        model = EndToEndPortfolioModel(
            input_dim=X.shape[1],
            output_dim=n_assets,
        )

        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 损失函数
        criterion = JointOptimizationLoss(
            prediction_weight=0.3,
            sharpe_weight=0.3,
            cvar_weight=0.2,
            cost_weight=0.1,
        )

        # 训练循环
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        optimization_history = []
        convergence_curve = []
        best_sharpe = -np.inf
        best_cvar = np.inf
        best_weights = None

        batch_size = 32
        seq_len = 20

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            # 批量训练
            for i in range(0, len(X_train) - seq_len, batch_size):
                batch_end = min(i + batch_size, len(X_train) - seq_len)

                # 准备序列数据
                batch_x = torch.FloatTensor(
                    X_train[i:i+batch_end]
                ).to(device)

                # 准备目标收益（使用下一期的收益）
                batch_returns = torch.FloatTensor(
                    returns_train[i+seq_len:i+seq_len+batch_end-i]
                ).to(device)

                if len(batch_x) == 0 or len(batch_returns) == 0:
                    continue

                # 前向传播
                predicted_returns, predicted_risks, positions = model(batch_x)

                # 计算损失
                prev_positions = None
                loss = criterion(
                    predicted_returns,
                    predicted_risks,
                    positions,
                    batch_returns,
                    prev_positions,
                )

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= (len(X_train) // batch_size + 1)

            # 验证
            model.eval()
            with torch.no_grad():
                val_x = torch.FloatTensor(X_val).to(device)
                val_returns = torch.FloatTensor(returns_val).to(device)

                pred_returns, pred_risks, val_positions = model(val_x)

                # 计算组合指标
                portfolio_returns = (val_positions * val_returns).sum(dim=1)

                sharpe = (
                    portfolio_returns.mean() /
                    (portfolio_returns.std() + 1e-8)
                )

                var_index = int(0.05 * len(portfolio_returns))
                sorted_returns, _ = torch.sort(portfolio_returns)
                cvar = sorted_returns[:var_index+1].mean()

                # 记录最佳结果
                if sharpe > best_sharpe:
                    best_sharpe = sharpe.item()
                    best_weights = val_positions[-1].cpu().numpy()

                if cvar < best_cvar:
                    best_cvar = cvar.item()

                convergence_curve.append(sharpe.item())

            # 早停
            if epoch > 10 and len(convergence_curve) > 10:
                recent_improvement = (
                    convergence_curve[-1] -
                    min(convergence_curve[-10:-1])
                )
                if recent_improvement < 1e-4:
                    break

        # 创建训练结果
        self.joint_training_result = JointTrainingResult(
            model=model,
            optimization_history=optimization_history,
            best_sharpe=best_sharpe,
            best_cvar=best_cvar,
            best_weights=best_weights,
            training_epochs=epoch + 1,
            convergence_curve=convergence_curve,
        )

        self.is_fitted = True

        return self.joint_training_result

    def predict_portfolio(
        self,
        feature_matrix: FeatureMatrix,
        asset_types: List[str],
    ) -> OptimizationResult:
        """
        预测最优组合

        参数:
            feature_matrix: 特征矩阵
            asset_types: 资产类型

        返回:
            OptimizationResult
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train_joint_model() first.")

        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")

        # 预测
        model = self.joint_training_result.model
        model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            X = torch.FloatTensor(feature_matrix.data.values).to(device)
            predicted_returns, predicted_risks, positions = model(X)

        # 获取最后一个预测
        weights = positions[-1].cpu().numpy()
        pred_returns = predicted_returns[-1].cpu().numpy()
        pred_risks = predicted_risks[-1].cpu().numpy()

        # 计算组合指标
        expected_return = (weights * pred_returns).sum()
        expected_risk = np.sqrt(
            (weights @ np.diag(pred_risks**2) @ weights)
        )
        sharpe_ratio = expected_return / (expected_risk + 1e-8)

        # 分离安全资产和风险资产
        safe_assets = [
            i for i, t in enumerate(asset_types)
            if t in self.taleb_config.safe_asset_types
        ]
        risky_assets = [
            i for i, t in enumerate(asset_types)
            if t in self.taleb_config.risky_asset_types
        ]

        safe_asset_weights = {
            asset_types[i]: weights[i]
            for i in safe_assets
        }
        risky_asset_weights = {
            asset_types[i]: weights[i]
            for i in risky_assets
        }

        return OptimizationResult(
            optimal_weights=weights,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            cvar=0.0,  # 需要历史数据计算
            max_drawdown=0.0,
            turnover=0.0,
            diversification_ratio=0.0,
            safe_asset_weights=safe_asset_weights,
            risky_asset_weights=risky_asset_weights,
            optimization_time=0.0,
            convergence_status="converged",
        )

    def _optimize_taleb_barbell(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        asset_types: List[str],
        current_positions: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """塔勒布杠铃优化"""
        if not CVXPY_AVAILABLE:
            # 简化版本：直接分配权重
            return self._taleb_barbell_simple(
                expected_returns,
                asset_types,
            )

        n_assets = len(expected_returns)

        # 分类资产
        safe_indices = [
            i for i, t in enumerate(asset_types)
            if t in self.taleb_config.safe_asset_types
        ]
        risky_indices = [
            i for i, t in enumerate(asset_types)
            if t in self.taleb_config.risky_asset_types
        ]

        # 决策变量
        weights = cp.Variable(n_assets)

        # 目标：最大化夏普比率
        portfolio_return = expected_returns @ weights
        portfolio_risk = cp.quad_form(weights, covariance_matrix)
        objective = cp.Maximize(portfolio_return / cp.sqrt(portfolio_risk))

        # 约束
        constraints = [
            cp.sum(weights) == 1,  # 权重和为1
            weights >= 0,  # 不允许做空
            weights <= self.portfolio_constraints.max_position_size,  # 单个资产最大仓位
        ]

        # 塔勒布杠铃约束
        constraints.append(
            cp.sum(weights[safe_indices]) >= self.taleb_config.safe_asset_ratio * 0.9
        )
        constraints.append(
            cp.sum(weights[safe_indices]) <= self.taleb_config.safe_asset_ratio * 1.1
        )
        constraints.append(
            cp.sum(weights[risky_indices]) >= self.taleb_config.risky_asset_ratio * 0.9
        )
        constraints.append(
            cp.sum(weights[risky_indices]) <= self.taleb_config.risky_asset_ratio * 1.1
        )

        # 求解
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if weights.value is None:
            # 如果求解失败，使用简化版本
            return self._taleb_barbell_simple(
                expected_returns,
                asset_types,
            )

        optimal_weights = weights.value

        # 计算指标
        portfolio_return = (optimal_weights @ expected_returns)
        portfolio_risk = np.sqrt(optimal_weights @ covariance_matrix @ optimal_weights)
        sharpe_ratio = portfolio_return / (portfolio_risk + 1e-8)

        # 分离安全资产和风险资产
        safe_asset_weights = {
            asset_types[i]: optimal_weights[i]
            for i in safe_indices
        }
        risky_asset_weights = {
            asset_types[i]: optimal_weights[i]
            for i in risky_indices
        }

        return OptimizationResult(
            optimal_weights=optimal_weights,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            cvar=0.0,
            max_drawdown=0.0,
            turnover=0.0,
            diversification_ratio=0.0,
            safe_asset_weights=safe_asset_weights,
            risky_asset_weights=risky_asset_weights,
            optimization_time=0.0,
            convergence_status="optimal",
        )

    def _taleb_barbell_simple(
        self,
        expected_returns: np.ndarray,
        asset_types: List[str],
    ) -> OptimizationResult:
        """塔勒布杠铃简化版本"""
        n_assets = len(expected_returns)
        weights = np.zeros(n_assets)

        # 分类资产
        safe_indices = [
            i for i, t in enumerate(asset_types)
            if t in self.taleb_config.safe_asset_types
        ]
        risky_indices = [
            i for i, t in enumerate(asset_types)
            if t in self.taleb_config.risky_asset_types
        ]

        # 安全资产等权分配
        if safe_indices:
            safe_weight = self.taleb_config.safe_asset_ratio / len(safe_indices)
            weights[safe_indices] = safe_weight

        # 风险资产等权分配
        if risky_indices:
            risky_weight = self.taleb_config.risky_asset_ratio / len(risky_indices)
            weights[risky_indices] = risky_weight

        # 计算指标
        portfolio_return = (weights @ expected_returns)
        portfolio_risk = np.std(weights * expected_returns)
        sharpe_ratio = portfolio_return / (portfolio_risk + 1e-8)

        safe_asset_weights = {
            asset_types[i]: weights[i]
            for i in safe_indices
        }
        risky_asset_weights = {
            asset_types[i]: weights[i]
            for i in risky_indices
        }

        return OptimizationResult(
            optimal_weights=weights,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            cvar=0.0,
            max_drawdown=0.0,
            turnover=0.0,
            diversification_ratio=0.0,
            safe_asset_weights=safe_asset_weights,
            risky_asset_weights=risky_asset_weights,
            optimization_time=0.0,
            convergence_status="heuristic",
        )

    def _optimize_max_sharpe(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_positions: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """最大化夏普比率"""
        if not CVXPY_AVAILABLE:
            # 等权
            n = len(expected_returns)
            weights = np.ones(n) / n
            return OptimizationResult(
                optimal_weights=weights,
                expected_return=weights @ expected_returns,
                expected_risk=np.sqrt(weights @ covariance_matrix @ weights),
                sharpe_ratio=(weights @ expected_returns) / np.sqrt(weights @ covariance_matrix @ weights),
                cvar=0.0,
                max_drawdown=0.0,
                turnover=0.0,
                diversification_ratio=0.0,
                safe_asset_weights={},
                risky_asset_weights={},
                optimization_time=0.0,
                convergence_status="heuristic",
            )

        n = len(expected_returns)
        weights = cp.Variable(n)

        portfolio_return = expected_returns @ weights
        portfolio_risk = cp.quad_form(weights, covariance_matrix)

        objective = cp.Maximize(portfolio_return / cp.sqrt(portfolio_risk))

        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            weights <= self.portfolio_constraints.max_position_size,
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        if weights.value is None:
            weights = np.ones(n) / n
        else:
            weights = weights.value

        return OptimizationResult(
            optimal_weights=weights,
            expected_return=weights @ expected_returns,
            expected_risk=np.sqrt(weights @ covariance_matrix @ weights),
            sharpe_ratio=(weights @ expected_returns) / np.sqrt(weights @ covariance_matrix @ weights),
            cvar=0.0,
            max_drawdown=0.0,
            turnover=0.0,
            diversification_ratio=0.0,
            safe_asset_weights={},
            risky_asset_weights={},
            optimization_time=0.0,
            convergence_status="optimal" if weights.value is not None else "heuristic",
        )

    def _optimize_min_cvar(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_positions: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """最小化CVaR"""
        # 简化实现：使用风险平价
        return self._optimize_risk_parity(covariance_matrix, current_positions)

    def _optimize_risk_parity(
        self,
        covariance_matrix: np.ndarray,
        current_positions: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """风险平价优化"""
        n = len(covariance_matrix)

        # 计算资产风险
        asset_risks = np.sqrt(np.diag(covariance_matrix))

        # 风险平价：权重与风险成反比
        inv_risks = 1.0 / (asset_risks + 1e-8)
        weights = inv_risks / inv_risks.sum()

        return OptimizationResult(
            optimal_weights=weights,
            expected_return=0.0,
            expected_risk=0.0,
            sharpe_ratio=0.0,
            cvar=0.0,
            max_drawdown=0.0,
            turnover=0.0,
            diversification_ratio=0.0,
            safe_asset_weights={},
            risky_asset_weights={},
            optimization_time=0.0,
            convergence_status="heuristic",
        )

    def _optimize_equal_weight(
        self,
        n_assets: int,
        current_positions: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """等权重优化"""
        weights = np.ones(n_assets) / n_assets

        return OptimizationResult(
            optimal_weights=weights,
            expected_return=0.0,
            expected_risk=0.0,
            sharpe_ratio=0.0,
            cvar=0.0,
            max_drawdown=0.0,
            turnover=0.0,
            diversification_ratio=0.0,
            safe_asset_weights={},
            risky_asset_weights={},
            optimization_time=0.0,
            convergence_status="optimal",
        )

    def _optimize_mean_variance(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_positions: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """均值-方差优化"""
        return self._optimize_max_sharpe(expected_returns, covariance_matrix, current_positions)


# ============================================================================
# 工厂函数
# ============================================================================

def create_joint_optimization_layer(
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.TALEB_BARBELL,
    portfolio_constraints: Optional[PortfolioConstraint] = None,
    taleb_config: Optional[TalebBarbellConfig] = None,
) -> JointOptimizationLayer:
    """创建联合优化层"""
    return JointOptimizationLayer(
        optimization_strategy=optimization_strategy,
        portfolio_constraints=portfolio_constraints,
        taleb_config=taleb_config,
    )


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 创建联合优化层
    layer = create_joint_optimization_layer(
        optimization_strategy=OptimizationStrategy.TALEB_BARBELL,
    )

    print("✅ 联合优化层创建成功")
    print(f"  优化策略: {layer.optimization_strategy}")
    print(f"  安全资产比例: {layer.taleb_config.safe_asset_ratio}")
    print(f"  风险资产比例: {layer.taleb_config.risky_asset_ratio}")
