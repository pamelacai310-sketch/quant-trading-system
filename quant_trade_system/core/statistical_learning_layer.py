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
import copy
from quant_trade_system.ledger import position_returns
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
    training_metrics: Dict[str, Any]  # 训练指标
    validation_metrics: Dict[str, Any]  # 验证指标
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
    expected_win_rate: Optional[float]          # 预期胜率
    expected_odds_ratio: Optional[float]        # 预期赔率
    expected_elasticity: Optional[float]        # 预期弹性


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
            return max(0, len(self.labels) - self.seq_len)

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
        """Proper differentiable probability loss; not realized trade win rate."""
        def forward(self, predictions, targets, positions=None):
            return nn.functional.binary_cross_entropy(predictions, targets)


    class OddsRatioLoss(nn.Module):
        """Retired: batch odds are not a proper probability training objective."""
        def forward(self, predictions, targets, returns=None):
            raise ValueError("OddsRatioLoss retired; use BCE and evaluate net ledger PnL")


    class ElasticityLoss(nn.Module):
        def __init__(self, benchmark_return=0.001):
            super().__init__()

        def forward(self, predictions, targets, returns=None):
            raise ValueError("ElasticityLoss retired; absolute return rewards losses")


    class CombinedLoss(nn.Module):
        def __init__(self, win_rate_weight=1.0, odds_ratio_weight=0.0, elasticity_weight=0.0):
            super().__init__()
            if odds_ratio_weight or elasticity_weight or win_rate_weight != 1.0:
                raise ValueError("Only calibrated probability BCE is supported")

        def forward(self, predictions, targets, returns=None):
            return nn.functional.binary_cross_entropy(predictions, targets)


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
        expected_win_rate = None  # No per-trade conditional estimate has been fitted.
        expected_odds_ratio = None
        expected_elasticity = None

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

        self.seq_len = seq_len
        if min(len(X_train), len(X_val)) <= seq_len:
            raise ValueError('Insufficient observations for sequence training/validation')
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
        criterion = nn.BCELoss()  # Probability estimation, not a claimed PnL objective.

        # 训练循环
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_state = None

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
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
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
            seq_len = self.seq_len
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
            pred_array = np.concatenate(predictions) if predictions else np.array([])

            # 最后seq_len个时间点有预测
            full_predictions[seq_len:] = pred_array[:len(X)-seq_len]

            return full_predictions

    def _compute_positions(self, predictions: np.ndarray) -> np.ndarray:
        """Probability-sized long-only exposure; the same mapping is used in metrics."""
        return np.clip(np.asarray(predictions, float), 0, 1)

    def _compute_metrics(self, y_true, y_pred, returns=None, cost_bps=8.0):
        """Period diagnostics. returns[t] must follow the executable entry for w[t].

        Closed-trade win_rate/odds_ratio are deliberately unavailable here: a
        direction classifier has no independent exit/round-trip ledger.
        """
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError('Prediction/label alignment mismatch')
        valid = np.isfinite(y_pred)
        if self.model_type == ModelType.TRANSFORMER:
            valid[:getattr(self, 'seq_len', 20)] = False
        metrics = {'direction_accuracy': float(np.mean((y_pred[valid] > .5) == (y_true[valid] > .5))) if valid.any() else 0.0,
                   'win_rate': None, 'odds_ratio': None,
                   'training_objective': 'binary_cross_entropy',
                   'metric_scope': 'period_diagnostic_not_closed_trades',
                   'production_eligible': False}
        if returns is not None:
            r = np.asarray(returns, float)
            if r.shape != y_pred.shape or not np.isfinite(r).all():
                raise ValueError('Forward return alignment mismatch')
            weights = self._compute_positions(y_pred)
            weights[~valid] = 0
            net = position_returns(weights, r, cost_bps)
            equity = np.r_[1.0, np.cumprod(1 + net)]
            metrics.update(net_total_return=float(equity[-1]-1),
                           period_win_rate=float(np.mean(net[valid] > 0)) if valid.any() else 0.,
                           elasticity=float(np.abs(net).mean()) if len(net) else 0.,
                           sharpe_ratio=float(net.mean()/net.std()*np.sqrt(252)) if len(net) and net.std() > 1e-12 else 0.,
                           max_drawdown=float(np.min(equity/np.maximum.accumulate(equity)-1)),
                           cvar=float(net[net <= np.percentile(net, 5)].mean()) if len(net) else 0.,
                           cost_bps=cost_bps)
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
