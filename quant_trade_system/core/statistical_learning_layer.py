"""
统计学习模型层 - Statistical Learning Layer

核心功能：
1. 使用Transformer/LightGBM从特征矩阵中学习
2. 替代固定权重打分，将核心从因果推理转移到模型端
3. 优化三个关键指标：胜率、赔率、弹性
4. 特征自动选择（保持金融含义和独立解释力）
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

from quant_trade_system.core.feature_engineering_layer import (
    FeatureMatrix,
    FeatureDomain,
)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    Dataset = None
    DataLoader = None
    print("Warning: PyTorch not installed. Install with: pip install torch")


# ============================================================================
# 枚举定义
# ============================================================================

class ModelType(Enum):
    """模型类型"""
    LIGHTGBM = "lightgbm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"


class OptimizationMetric(Enum):
    """优化指标"""
    WIN_RATE = "win_rate"              # 胜率
    ODDS_RATIO = "odds_ratio"          # 赔率
    ELASTICITY = "elasticity"          # 弹性
    SHARPE_RATIO = "sharpe_ratio"      # 夏普比率
    MAX_DRAWDOWN = "max_drawdown"      # 最大回撤
    CVAR = "cvar"                      # 条件风险价值


class LossFunction(Enum):
    """损失函数类型"""
    WIN_RATE_LOSS = "win_rate_loss"
    ODDS_RATIO_LOSS = "odds_ratio_loss"
    ELASTICITY_LOSS = "elasticity_loss"
    COMBINED_LOSS = "combined_loss"
    SHARPE_LOSS = "sharpe_loss"
    CVAR_LOSS = "cvar_loss"


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class LearningObjectiveConfig:
    """学习目标配置"""
    primary_metric: OptimizationMetric = OptimizationMetric.WIN_RATE
    secondary_metrics: List[OptimizationMetric] = field(default_factory=list)
    metric_weights: Dict[OptimizationMetric, float] = field(default_factory=dict)
    min_win_rate: float = 0.55
    min_odds_ratio: float = 1.5
    min_elasticity: float = 1.2
    target_sharpe: float = 1.5
    max_drawdown_threshold: float = 0.15
    cvar_confidence: float = 0.95


@dataclass
class TrainingResult:
    """训练结果"""
    model: Any                        # 训练好的模型
    model_type: ModelType             # 模型类型
    feature_importance: Dict[str, float]  # 特征重要性
    selected_features: List[str]      # 选中的特征
    training_metrics: Dict[str, float]  # 训练指标
    validation_metrics: Dict[str, float]  # 验证指标
    training_time: float              # 训练时间（秒）
    predictions: np.ndarray           # 预测结果
    positions: np.ndarray             # 仓位序列
    returns: np.ndarray               # 收益序列


@dataclass
class PredictionResult:
    """预测结果"""
    timestamps: List[datetime]        # 时间戳
    symbols: List[str]                # 标的物
    predictions: np.ndarray           # 预测值（概率或分数）
    positions: np.ndarray             # 建议仓位（0-1）
    confidence: np.ndarray            # 预测置信度
    feature_contributions: Dict[str, np.ndarray]  # 特征贡献度
    expected_win_rate: float          # 预期胜率
    expected_odds_ratio: float        # 预期赔率
    expected_elasticity: float        # 预期弹性


@dataclass
class ModelEvaluation:
    """模型评估"""
    win_rate: float                   # 实际胜率
    odds_ratio: float                 # 实际赔率
    elasticity: float                 # 实际弹性
    sharpe_ratio: float               # 夏普比率
    max_drawdown: float               # 最大回撤
    cvar: float                       # 条件风险价值
    total_return: float               # 总收益
    hit_rate: float                   # 命中率
    profit_factor: float              # 盈利因子


# ============================================================================
# PyTorch Transformer模型
# ============================================================================

if PYTORCH_AVAILABLE:
    class FinancialTransformer(nn.Module):
        """
        金融时序Transformer模型

        架构：
        1. 特征嵌入层
        2. 位置编码
        3. Transformer编码器
        4. 预测头（MLP）
        """

        def __init__(
            self,
            input_dim: int,
            d_model: int = 256,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            dim_feedforward: int = 1024,
            dropout: float = 0.1,
            output_dim: int = 1,
        ):
            super().__init__()

            self.input_dim = input_dim
            self.d_model = d_model

            # 特征嵌入
            self.feature_embedding = nn.Linear(input_dim, d_model)

            # 位置编码
            self.pos_encoding = PositionalEncoding(d_model, dropout)

            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_encoder_layers,
            )

            # 预测头
            self.predictor = nn.Sequential(
                nn.Linear(d_model, dim_feedforward // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward // 2, output_dim),
                nn.Sigmoid(),  # 输出0-1之间的概率
            )

        def forward(self, x, mask=None):
            """
            前向传播

            参数:
                x: (batch_size, seq_len, input_dim)
                mask: (batch_size, seq_len)

            返回:
                predictions: (batch_size, 1)
            """
            # 特征嵌入
            x = self.feature_embedding(x)  # (batch_size, seq_len, d_model)

            # 位置编码
            x = self.pos_encoding(x)

            # Transformer编码
            x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

            # 取最后一个时间步
            x = x[:, -1, :]  # (batch_size, d_model)

            # 预测
            predictions = self.predictor(x)  # (batch_size, 1)

            return predictions.squeeze(-1)  # (batch_size,)


    class PositionalEncoding(nn.Module):
        """位置编码"""

        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            # 创建位置编码矩阵
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)

            self.register_buffer('pe', pe)

        def forward(self, x):
            """
            参数:
                x: (batch_size, seq_len, d_model)
            """
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


    class FinancialDataset(Dataset):
        """金融数据集"""

        def __init__(
            self,
            features: np.ndarray,
            labels: np.ndarray,
            seq_len: int = 20,
        ):
            """
            参数:
                features: (T, N) 特征矩阵
                labels: (T,) 标签
                seq_len: 序列长度
            """
            self.features = features
            self.labels = labels
            self.seq_len = seq_len

        def __len__(self):
            return len(self.labels) - self.seq_len

        def __getitem__(self, idx):
            """
            返回序列数据
            """
            # 特征序列
            x = self.features[idx:idx + self.seq_len, :]

            # 标签
            y = self.labels[idx + self.seq_len]

            return (
                torch.FloatTensor(x),
                torch.FloatTensor([y]),
            )


# ============================================================================
# 自定义损失函数
# ============================================================================

if PYTORCH_AVAILABLE:

    class WinRateLoss(nn.Module):
        """
        胜率损失函数

        最大化胜率 = 最小化 (1 - 胜率)
        """

        def __init__(self):
            super().__init__()

        def forward(self, predictions, targets, positions=None):
            """
            参数:
                predictions: (batch_size,) 预测值（0-1）
                targets: (batch_size,) 实际标签（0或1）
                positions: (batch_size,) 仓位（可选）

            返回:
                loss: 标量
            """
            # 计算预测正确性
            correct = (predictions > 0.5) == (targets > 0.5)

            # 胜率 = 正确预测的比例
            win_rate = correct.float().mean()

            # 损失 = 1 - 胜率
            loss = 1.0 - win_rate

            return loss


    class OddsRatioLoss(nn.Module):
        """
        赔率损失函数

        最大化赔率 = 平均盈利 / 平均亏损
        """

        def __init__(self):
            super().__init__()

        def forward(self, predictions, targets, returns=None):
            """
            参数:
                predictions: (batch_size,) 预测值（0-1）
                targets: (batch_size,) 实际标签（0或1）
                returns: (batch_size,) 实际收益（可选）

            返回:
                loss: 标量
            """
            # 如果没有提供returns，使用prediction和target计算
            if returns is None:
                # 假设：预测正确时收益为1，错误时收益为-1
                returns = torch.where(
                    (predictions > 0.5) == (targets > 0.5),
                    torch.ones_like(predictions),
                    -torch.ones_like(predictions),
                )

            # 计算盈利和亏损
            profits = returns[returns > 0]
            losses = returns[returns < 0]

            # 平均盈利和平均亏损
            avg_profit = profits.mean() if len(profits) > 0 else torch.tensor(0.0)
            avg_loss = -losses.mean() if len(losses) > 0 else torch.tensor(1.0)

            # 赔率 = 平均盈利 / 平均亏损
            odds_ratio = avg_profit / (avg_loss + 1e-8)

            # 损失 = 1 / (赔率 + epsilon)
            loss = 1.0 / (odds_ratio + 1e-8)

            return loss


    class ElasticityLoss(nn.Module):
        """
        弹性损失函数

        最大化弹性 = 收益变化幅度 / 基准变化幅度
        """

        def __init__(self, benchmark_return=0.001):
            super().__init__()
            self.benchmark_return = benchmark_return

        def forward(self, predictions, targets, returns=None):
            """
            参数:
                predictions: (batch_size,) 预测值（0-1）
                targets: (batch_size,) 实际标签（0或1）
                returns: (batch_size,) 实际收益（可选）

            返回:
                loss: 标量
            """
            # 如果没有提供returns，使用prediction和target计算
            if returns is None:
                # 假设：预测正确时收益为prediction，错误时收益为-prediction
                returns = torch.where(
                    (predictions > 0.5) == (targets > 0.5),
                    predictions,
                    -predictions,
                )

            # 计算策略收益的绝对变化幅度
            strategy_elasticity = returns.abs().mean()

            # 损失 = 1 / (弹性 + epsilon)
            loss = 1.0 / (strategy_elasticity + 1e-8)

            return loss


    class CombinedLoss(nn.Module):
        """
        组合损失函数

        同时优化胜率、赔率、弹性
        """

        def __init__(
            self,
            win_rate_weight: float = 0.4,
            odds_ratio_weight: float = 0.3,
            elasticity_weight: float = 0.3,
        ):
            super().__init__()
            self.win_rate_weight = win_rate_weight
            self.odds_ratio_weight = odds_ratio_weight
            self.elasticity_weight = elasticity_weight

            self.win_rate_loss = WinRateLoss()
            self.odds_ratio_loss = OddsRatioLoss()
            self.elasticity_loss = ElasticityLoss()

        def forward(self, predictions, targets, returns=None):
            """
            参数:
                predictions: (batch_size,) 预测值（0-1）
                targets: (batch_size,) 实际标签（0或1）
                returns: (batch_size,) 实际收益（可选）

            返回:
                loss: 标量
            """
            # 计算各项损失
            loss_win_rate = self.win_rate_loss(predictions, targets)
            loss_odds_ratio = self.odds_ratio_loss(predictions, targets, returns)
            loss_elasticity = self.elasticity_loss(predictions, targets, returns)

            # 加权组合
            loss = (
                self.win_rate_weight * loss_win_rate +
                self.odds_ratio_weight * loss_odds_ratio +
                self.elasticity_weight * loss_elasticity
            )

            return loss


# ============================================================================
# 统计学习层核心类
# ============================================================================

class StatisticalLearningLayer:
    """
    统计学习模型层

    核心功能：
    1. 使用Transformer/LightGBM从特征矩阵中学习
    2. 替代固定权重打分
    3. 优化胜率、赔率、弹性
    4. 自动特征选择
    """

    def __init__(
        self,
        model_type: ModelType = ModelType.LIGHTGBM,
        objective_config: Optional[LearningObjectiveConfig] = None,
        feature_selection: bool = True,
        min_interpretability: float = 0.6,
        min_independent_power: float = 0.5,
    ):
        """
        初始化统计学习层

        参数:
            model_type: 模型类型
            objective_config: 学习目标配置
            feature_selection: 是否进行特征选择
            min_interpretability: 最小可解释性阈值
            min_independent_power: 最小独立解释力阈值
        """
        self.model_type = model_type
        self.objective_config = objective_config or LearningObjectiveConfig()
        self.feature_selection = feature_selection
        self.min_interpretability = min_interpretability
        self.min_independent_power = min_independent_power

        # 模型
        self.model = None
        self.is_fitted = False

        # 训练结果
        self.training_result: Optional[TrainingResult] = None

    def train(
        self,
        feature_matrix: FeatureMatrix,
        labels: np.ndarray,
        returns: Optional[np.ndarray] = None,
        validation_split: float = 0.2,
        **model_params,
    ) -> TrainingResult:
        """
        训练模型

        参数:
            feature_matrix: 特征矩阵
            labels: 标签（0或1，表示涨跌）
            returns: 收益率（可选，用于计算赔率和弹性）
            validation_split: 验证集比例
            **model_params: 模型参数

        返回:
            TrainingResult
        """
        start_time = datetime.now()

        # 准备数据
        X = feature_matrix.data.values
        y = labels
        timestamps = feature_matrix.timestamps

        # 划分训练集和验证集
        split_idx = int(len(X) * (1 - validation_split))

        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if returns is not None:
            returns_train, returns_val = returns[:split_idx], returns[split_idx:]
        else:
            returns_train, returns_val = None, None

        # 特征选择
        selected_features = self._select_features(
            feature_matrix,
            X_train,
            y_train,
        )

        X_train_selected = X_train[:, [list(feature_matrix.data.columns).index(f) for f in selected_features]]
        X_val_selected = X_val[:, [list(feature_matrix.data.columns).index(f) for f in selected_features]]

        # 训练模型
        if self.model_type == ModelType.LIGHTGBM:
            self.model = self._train_lightgbm(
                X_train_selected, y_train,
                X_val_selected, y_val,
                returns_train,
                **model_params,
            )
        elif self.model_type == ModelType.TRANSFORMER:
            self.model = self._train_transformer(
                X_train_selected, y_train,
                X_val_selected, y_val,
                returns_train,
                **model_params,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.is_fitted = True

        # 训练时间
        training_time = (datetime.now() - start_time).total_seconds()

        # 评估
        train_predictions = self._predict_proba(X_train_selected)
        val_predictions = self._predict_proba(X_val_selected)

        train_metrics = self._compute_metrics(
            y_train,
            train_predictions,
            returns_train,
        )
        val_metrics = self._compute_metrics(
            y_val,
            val_predictions,
            returns_val,
        )

        # 特征重要性
        feature_importance = self._compute_feature_importance(selected_features)

        # 创建训练结果
        self.training_result = TrainingResult(
            model=self.model,
            model_type=self.model_type,
            feature_importance=feature_importance,
            selected_features=selected_features,
            training_metrics=train_metrics,
            validation_metrics=val_metrics,
            training_time=training_time,
            predictions=val_predictions,
            positions=self._compute_positions(val_predictions),
            returns=returns_val if returns_val is not None else None,
        )

        return self.training_result

    def predict(
        self,
        feature_matrix: FeatureMatrix,
    ) -> PredictionResult:
        """
        预测

        参数:
            feature_matrix: 特征矩阵

        返回:
            PredictionResult
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        # 获取选中的特征
        selected_features = self.training_result.selected_features
        feature_names = list(feature_matrix.data.columns)

        # 提取选中的特征
        X = feature_matrix.data.values
        X_selected = X[:, [feature_names.index(f) for f in selected_features]]

        # 预测
        predictions = self._predict_proba(X_selected)

        # 计算仓位
        positions = self._compute_positions(predictions)

        # 计算置信度
        confidence = np.abs(predictions - 0.5) * 2  # 0-1之间

        # 特征贡献度（使用SHAP或梯度）
        feature_contributions = self._compute_feature_contributions(X_selected, selected_features)

        # 预期指标
        expected_win_rate = self.training_result.validation_metrics.get('win_rate', 0.5)
        expected_odds_ratio = self.training_result.validation_metrics.get('odds_ratio', 1.0)
        expected_elasticity = self.training_result.validation_metrics.get('elasticity', 1.0)

        return PredictionResult(
            timestamps=feature_matrix.timestamps,
            symbols=feature_matrix.symbols,
            predictions=predictions,
            positions=positions,
            confidence=confidence,
            feature_contributions=feature_contributions,
            expected_win_rate=expected_win_rate,
            expected_odds_ratio=expected_odds_ratio,
            expected_elasticity=expected_elasticity,
        )

    def _select_features(
        self,
        feature_matrix: FeatureMatrix,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[str]:
        """特征选择"""
        if not self.feature_selection:
            return list(feature_matrix.data.columns)

        # 基于可解释性和独立解释力筛选
        feature_names = list(feature_matrix.data.columns)

        selected = []
        for name in feature_names:
            metadata = feature_matrix.feature_metadata.get(name, {})

            interpretability = metadata.get('interpretability', 0)
            independent_power = metadata.get('independent_power', 0)

            if (
                interpretability >= self.min_interpretability and
                independent_power >= self.min_independent_power
            ):
                selected.append(name)

        # 如果使用LightGBM，可以使用模型内置的特征重要性
        if self.model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
            # 训练一个临时模型获取特征重要性
            temp_model = lgb.LGBMClassifier(
                n_estimators=100,
                verbose=-1,
            )
            temp_model.fit(X, y)

            importance = temp_model.feature_importances_

            # 选择重要性前80%的特征
            threshold = np.percentile(importance, 20)
            important_indices = np.where(importance >= threshold)[0]

            selected = [selected[i] for i in important_indices if i < len(selected)]

        return selected

    def _train_lightgbm(
        self,
        X_train, y_train,
        X_val, y_val,
        returns_train,
        **params,
    ):
        """训练LightGBM模型"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")

        # 默认参数
        default_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 7,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
        }
        default_params.update(params)

        # 创建模型
        model = lgb.LGBMClassifier(**default_params)

        # 训练
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
            ],
        )

        return model

    def _train_transformer(
        self,
        X_train, y_train,
        X_val, y_val,
        returns_train,
        **params,
    ):
        """训练Transformer模型"""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")

        # 默认参数
        default_params = {
            'd_model': 256,
            'nhead': 8,
            'num_encoder_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'seq_len': 20,
            'batch_size': 32,
            'epochs': 100,
            'lr': 0.001,
        }
        default_params.update(params)

        seq_len = default_params.pop('seq_len')
        batch_size = default_params.pop('batch_size')
        epochs = default_params.pop('epochs')
        lr = default_params.pop('lr')

        # 创建数据集
        train_dataset = FinancialDataset(X_train, y_train, seq_len)
        val_dataset = FinancialDataset(X_val, y_val, seq_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 创建模型
        model = FinancialTransformer(
            input_dim=X_train.shape[1],
            **default_params,
        )

        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 损失函数
        criterion = CombinedLoss(
            win_rate_weight=0.4,
            odds_ratio_weight=0.3,
            elasticity_weight=0.3,
        )

        # 训练循环
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).squeeze(-1)

                optimizer.zero_grad()

                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device).squeeze(-1)

                    predictions = model(batch_x)
                    loss = criterion(predictions, batch_y)

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return model

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if self.model_type == ModelType.LIGHTGBM:
            return self.model.predict_proba(X)[:, 1]
        elif self.model_type == ModelType.TRANSFORMER:
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch not installed")

            self.model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 创建序列数据
            seq_len = 20  # 与训练时相同
            dataset = FinancialDataset(X, np.zeros(len(X)), seq_len)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)

            predictions = []
            with torch.no_grad():
                for batch_x, _ in loader:
                    batch_x = batch_x.to(device)
                    pred = self.model(batch_x)
                    predictions.append(pred.cpu().numpy())

            # 填充前面没有预测的部分
            full_predictions = np.zeros(len(X))
            pred_array = np.concatenate(predictions)

            # 最后seq_len个时间点有预测
            full_predictions[seq_len:] = pred_array[:len(X)-seq_len]

            return full_predictions

    def _compute_positions(self, predictions: np.ndarray) -> np.ndarray:
        """计算仓位（0-1）"""
        # 简单策略：预测概率 > 0.5 时满仓，否则空仓
        # 可以改进为：仓位 = 预测概率
        return predictions

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """计算指标"""
        metrics = {}

        # 胜率
        correct = ((y_pred > 0.5) == (y_true > 0.5)).sum()
        total = len(y_true)
        metrics['win_rate'] = correct / total if total > 0 else 0.0

        # 如果有returns，计算赔率和弹性
        if returns is not None:
            # 计算交易收益
            trade_returns = np.where(
                (y_pred > 0.5) == (y_true > 0.5),
                returns,
                -returns,
            )

            # 赔率
            profits = trade_returns[trade_returns > 0]
            losses = trade_returns[trade_returns < 0]

            avg_profit = profits.mean() if len(profits) > 0 else 0
            avg_loss = -losses.mean() if len(losses) > 0 else 0

            metrics['odds_ratio'] = avg_profit / (avg_loss + 1e-8)
            metrics['elasticity'] = np.abs(trade_returns).mean()

            # 夏普比率
            if len(trade_returns) > 1:
                metrics['sharpe_ratio'] = (
                    trade_returns.mean() / (trade_returns.std() + 1e-8) *
                    np.sqrt(252)
                )
            else:
                metrics['sharpe_ratio'] = 0.0

            # 最大回撤
            cumulative = np.cumprod(1 + trade_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()

            # CVaR
            var = np.percentile(trade_returns, 5)
            cvar = trade_returns[trade_returns <= var].mean()
            metrics['cvar'] = cvar

        return metrics

    def _compute_feature_importance(
        self,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """计算特征重要性"""
        if self.model_type == ModelType.LIGHTGBM:
            importance = self.model.feature_importances_
        elif self.model_type == ModelType.TRANSFORMER:
            # Transformer使用特征嵌入的权重
            if PYTORCH_AVAILABLE:
                importance = np.abs(
                    self.model.feature_embedding.weight.data.cpu().numpy()
                ).mean(axis=0)
            else:
                importance = np.ones(len(feature_names))
        else:
            importance = np.ones(len(feature_names))

        return dict(zip(feature_names, importance))

    def _compute_feature_contributions(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, np.ndarray]:
        """计算特征贡献度（简化版）"""
        # 这里简化处理，实际应该使用SHAP或积分梯度
        contributions = {}

        if self.model_type == ModelType.LIGHTGBM:
            # 使用LightGBM的特征重要性作为平均贡献
            importance = self.model.feature_importances_
            for name, imp in zip(feature_names, importance):
                contributions[name] = np.ones(len(X)) * imp

        elif self.model_type == ModelType.TRANSFORMER:
            # 使用注意力权重（简化）
            if PYTORCH_AVAILABLE:
                # 这里简化处理
                for name in feature_names:
                    contributions[name] = np.ones(len(X)) * 0.1

        return contributions


# ============================================================================
# 工厂函数
# ============================================================================

def create_statistical_learning_layer(
    model_type: ModelType = ModelType.LIGHTGBM,
    objective_config: Optional[LearningObjectiveConfig] = None,
    feature_selection: bool = True,
) -> StatisticalLearningLayer:
    """创建统计学习层"""
    return StatisticalLearningLayer(
        model_type=model_type,
        objective_config=objective_config,
        feature_selection=feature_selection,
    )


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 创建统计学习层
    layer = create_statistical_learning_layer(
        model_type=ModelType.LIGHTGBM,
    )

    print("✅ 统计学习层创建成功")
    print(f"  模型类型: {layer.model_type}")
    print(f"  特征选择: {layer.feature_selection}")
