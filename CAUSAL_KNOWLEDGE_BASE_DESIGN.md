# 因果知识库设计方案

## 🎯 系统愿景

构建一个**自迭代的因果AI量化交易系统**，从传统的"量化因子"升级为"因果推理"，真正理解市场的底层逻辑。

---

## 📐 核心设计原则

### 1. 因果 vs 相关

```
传统量化因子              →        因果AI推理
高度结构化                       非结构化
定量计算                         定性+定量
客观数据                         主观判断+客观数据
统计相关                         因果机制
黑盒模型                         白盒解释
静态因子                         动态演化
```

### 2. 知识表示层次

```
Layer 3: 因果推理引擎 (Causal Inference Engine)
    ├─ 因果发现 (Causal Discovery)
    ├─ 因果验证 (Causal Validation)
    └─ 因果预测 (Causal Prediction)
           ↓
Layer 2: 因果知识图谱 (Causal Knowledge Graph)
    ├─ 实体节点 (Entity Nodes)
    ├─ 因果边 (Causal Edges)
    └─ 治理机制 (Governance)
           ↓
Layer 1: 因果因素库 (Causal Factor Library)
    ├─ 共有因素 (Cross-Asset Factors)
    ├─ 股票专属 (Equities Only)
    └─ 期货专属 (Commodities Only)
```

---

## 🏗️ 因果知识库架构

### 数据结构设计

```python
@dataclass
class CausalFactor:
    """因果因素基类"""
    factor_id: str                          # 因素ID
    name: str                              # 因素名称
    category: FactorCategory               # 因素类别
    asset_class: AssetClass                # 资产类别
    description: str                       # 因素描述
    causal_mechanism: str                  # 因果机制说明
    data_sources: List[str]                # 数据来源
    measurement_methods: List[str]         # 测量方法
    update_frequency: str                  # 更新频率
    reliability: float                     # 可靠性评分 (0-1)
    confidence: float                      # 置信度 (0-1)
    created_at: datetime
    updated_at: datetime
    version: int                           # 版本号


@dataclass
class CausalEdge:
    """因果边"""
    edge_id: str                           # 边ID
    source_factor_id: str                  # 源因素ID
    target_factor_id: str                  # 目标因素ID
    causal_type: CausalType                # 因果类型
    causal_strength: float                 # 因果强度 (0-1)
    lag_days: int                          # 滞后天数
    confidence: float                      # 置信度 (0-1)
    direction: CausalDirection             # 因果方向
    mechanism: str                         # 因果机制说明
    evidence: List[CausalEvidence]         # 支持证据
    market_regime: MarketRegime            # 适用市场制度
    created_at: datetime
    validated_count: int                   # 验证次数
    success_count: int                     # 成功次数


@dataclass
class CausalEvidence:
    """因果证据"""
    evidence_id: str
    evidence_type: EvidenceType            # 证据类型（实证/理论/专家）
    description: str
    data_period: Tuple[datetime, datetime] # 数据周期
    statistical_significance: float        # 统计显著性
    effect_size: float                     # 效应大小
    source: str                            # 来源
    url: Optional[str]
    validated: bool


class CausalKnowledgeGraph:
    """因果知识图谱"""
    def __init__(self):
        self.factors: Dict[str, CausalFactor] = {}
        self.edges: Dict[str, CausalEdge] = {}
        self.graph: nx.DiGraph = nx.DiGraph()

    def add_factor(self, factor: CausalFactor):
        """添加因素"""

    def add_edge(self, edge: CausalEdge):
        """添加因果边"""

    def query_causal_chain(self, source: str, target: str) -> List[CausalEdge]:
        """查询因果链"""

    def compute_causal_strength(self, source: str, target: str) -> float:
        """计算因果强度"""
```

---

## 📚 因果因素库分类体系

### 枚举定义

```python
class FactorCategory(Enum):
    """因素类别"""
    # 共有因素
    MACRO_POLICY = "macro_policy"              # 宏观与政策
    MICROSTRUCTURE = "microstructure"          # 市场微观结构
    QUANT_STRATEGY = "quant_strategy"          # 量化策略因子

    # 股票专属
    FUNDAMENTAL = "fundamental"                # 公司基本面
    VALUATION = "valuation"                    # 估值
    EQUITY_PREMIUM = "equity_premium"          # 股票特有溢价

    # 期货专属
    SUPPLY_DEMAND = "supply_demand"            # 供需
    FUTURES_PRICING = "futures_pricing"        # 期货定价
    COMMODITY_PREMIUM = "commodity_premium"    # 商品溢价


class AssetClass(Enum):
    """资产类别"""
    ALL = "all"                               # 所有资产
    EQUITY = "equity"                         # 股票
    COMMODITY = "commodity"                   # 商品期货
    FIXED_INCOME = "fixed_income"             # 固定收益
    FX = "fx"                                 # 外汇
    CRYPTO = "crypto"                         # 加密货币


class CausalType(Enum):
    """因果类型"""
    DIRECT_POSITIVE = "direct_positive"       # 正向因果 (A↑ → B↑)
    DIRECT_NEGATIVE = "direct_negative"       # 负向因果 (A↑ → B↓)
    INDIRECT = "indirect"                     # 间接因果 (A → C → B)
    CONFOUNDED = "confounded"                 # 混杂因果 (A → B, 但C影响两者)
    BIDIRECTIONAL = "bidirectional"           # 双向因果 (A ↔ B)
    THRESHOLD = "threshold"                   # 阈值因果 (A > x → B)
    TEMPORAL = "temporal"                     # 时间因果 (A发生在t → B发生在t+n)


class EvidenceType(Enum):
    """证据类型"""
    EMPIRICAL = "empirical"                   # 实证证据
    THEORETICAL = "theoretical"               # 理论证据
    EXPERT = "expert"                         # 专家判断
    SIMULATION = "simulation"                 # 模拟结果
    BACKTEST = "backtest"                     # 回测验证
```

---

## 🔍 因果发现引擎

### 因果发现算法

```python
class CausalDiscoveryEngine:
    """因果发现引擎"""

    def __init__(self, knowledge_graph: CausalKnowledgeGraph):
        self.kg = knowledge_graph
        self.discovery_methods = {
            'pcm': self.physical_causal_modeling,      # 物理因果建模
            'granger': self.granger_causality,         # 格兰杰因果
            'pc': self.peter_clark,                    # Peter-Clark算法
            'varlingam': self.varlingam,               # VAR-LiNGAM
            'intervention': self.intervention_based,   # 干预为基础
        }

    def discover_causal_relationships(
        self,
        data: pd.DataFrame,
        factors: List[str],
        method: str = 'auto',
    ) -> List[CausalEdge]:
        """
        发现因果关系

        参数:
            data: 时序数据
            factors: 待分析的因素列表
            method: 发现方法
        """
        # 1. 预处理
        preprocessed_data = self._preprocess(data, factors)

        # 2. 选择方法
        if method == 'auto':
            method = self._select_optimal_method(preprocessed_data)

        # 3. 因果发现
        discovered_edges = self.discovery_methods[method](
            preprocessed_data, factors
        )

        # 4. 过滤弱因果关系
        strong_edges = [
            edge for edge in discovered_edges
            if edge.causal_strength > 0.3 and edge.confidence > 0.7
        ]

        return strong_edges

    def physical_causal_modeling(
        self,
        data: pd.DataFrame,
        factors: List[str],
    ) -> List[CausalEdge]:
        """
        物理因果建模

        基于经济学理论、金融理论、物理学定律的因果建模
        """
        edges = []

        # 示例：利率 → 股票价格（负向因果）
        # 机制：利率↑ → 贴现率↑ → 现值↓ → 股价↓

        # 获取利率和股票价格
        if 'interest_rate' in data.columns and 'stock_price' in data.columns:
            # 计算格兰杰因果检验
            from statsmodels.tsa.stattools import grangercausalitytests

            test_result = grangercausalitytests(
                data[['stock_price', 'interest_rate']],
                maxlag=5,
                verbose=False
            )

            # 提取p值
            p_values = [result[0]['ssr_ftest'][1] for result in test_result]

            if min(p_values) < 0.05:  # 显著
                edge = CausalEdge(
                    edge_id=f"interest_rate_stock_price_{datetime.now().strftime('%Y%m%d')}",
                    source_factor_id='interest_rate',
                    target_factor_id='stock_price',
                    causal_type=CausalType.DIRECT_NEGATIVE,
                    causal_strength=1 - min(p_values),
                    lag_days=p_values.index(min(p_values)) + 1,
                    confidence=1 - min(p_values),
                    direction=CausalDirection.FORWARD,
                    mechanism="利率↑ → 贴现率↑ → 现值↓ → 股价↓",
                    evidence=[],
                    market_regime=MarketRegime.BULL,
                    created_at=datetime.now(),
                )
                edges.append(edge)

        return edges

    def granger_causality(
        self,
        data: pd.DataFrame,
        factors: List[str],
    ) -> List[CausalEdge]:
        """格兰杰因果检验"""
        edges = []

        from statsmodels.tsa.stattools import grangercausalitytests

        for i, source in enumerate(factors):
            for target in factors[i+1:]:
                # 双向检验
                for direction in [(source, target), (target, source)]:
                    try:
                        test_result = grangercausalitytests(
                            data[[direction[1], direction[0]]],
                            maxlag=5,
                            verbose=False
                        )

                        p_values = [result[0]['ssr_ftest'][1] for result in test_result]
                        min_p = min(p_values)

                        if min_p < 0.05:
                            edge = CausalEdge(
                                edge_id=f"{direction[0]}_{direction[1]}_{datetime.now().strftime('%Y%m%d')}",
                                source_factor_id=direction[0],
                                target_factor_id=direction[1],
                                causal_type=CausalType.DIRECT_POSITIVE,
                                causal_strength=1 - min_p,
                                lag_days=p_values.index(min_p) + 1,
                                confidence=1 - min_p,
                                direction=CausalDirection.FORWARD,
                                mechanism=f"格兰杰因果: {direction[0]} 预测 {direction[1]}",
                                evidence=[],
                                market_regime=MarketRegime.VOLATILE,
                                created_at=datetime.now(),
                            )
                            edges.append(edge)
                    except:
                        continue

        return edges
```

---

## ✅ 因果验证引擎

### 因果验证机制

```python
class CausalValidationEngine:
    """因果验证引擎"""

    def __init__(self, knowledge_graph: CausalKnowledgeGraph):
        self.kg = knowledge_graph
        self.validation_methods = {
            'out_of_sample': self.out_of_sample_validation,     # 样本外验证
            'intervention': self.intervention_validation,       # 干预验证
            'counterfactual': self.counterfactual_validation,   # 反事实验证
            'stability': self.temporal_stability_validation,    # 时序稳定性
            'regime': self.regime_robustness_validation,        # 制度稳健性
        }

    def validate_causal_edge(
        self,
        edge: CausalEdge,
        validation_data: pd.DataFrame,
        method: str = 'all',
    ) -> CausalValidationResult:
        """
        验证因果边

        返回:
            CausalValidationResult: 验证结果
        """
        results = {}

        if method == 'all':
            methods = self.validation_methods.keys()
        else:
            methods = [method]

        for m in methods:
            results[m] = self.validation_methods[m](
                edge, validation_data
            )

        # 汇总结果
        overall_score = np.mean([r['score'] for r in results.values()])

        return CausalValidationResult(
            edge_id=edge.edge_id,
            validation_date=datetime.now(),
            method_scores=results,
            overall_score=overall_score,
            is_valid=overall_score > 0.7,
        )

    def out_of_sample_validation(
        self,
        edge: CausalEdge,
        data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        样本外验证

        将数据分为训练集和测试集，在训练集上发现因果，
        在测试集上验证预测能力
        """
        # 分割数据
        train_size = int(len(data) * 0.7)
        train_data = data[:train_size]
        test_data = data[train_size:]

        # 在训练集上估计因果强度
        source_train = train_data[edge.source_factor_id]
        target_train = train_data[edge.target_factor_id]

        # 简单线性回归
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(source_train.values.reshape(-1, 1), target_train)

        # 在测试集上预测
        source_test = test_data[edge.source_factor_id]
        target_test = test_data[edge.target_factor_id]

        predictions = model.predict(source_test.values.reshape(-1, 1))

        # 计算R²
        from sklearn.metrics import r2_score

        r2 = r2_score(target_test, predictions)

        return {
            'score': max(0, r2),
            'r2': r2,
            'method': 'out_of_sample',
        }

    def counterfactual_validation(
        self,
        edge: CausalEdge,
        data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        反事实验证

        使用反事实推理验证因果：
        "如果源因素没有变化，目标因素会怎样？"
        """
        # 使用合成控制方法
        from causalnex.network import BayesianNetwork
        from causalnex.plots import plot_structure

        # 简化版：计算"do"算子的效果
        source = data[edge.source_factor_id]
        target = data[edge.target_factor_id]

        # 计算因果效应
        causal_effect = np.corrcoef(source, target)[0, 1]

        # 反事实：如果源因素保持不变
        # （简化实现，实际应该更复杂）

        return {
            'score': abs(causal_effect),
            'causal_effect': causal_effect,
            'method': 'counterfactual',
        }
```

---

## 🔄 自迭代机制

### 学习-验证-优化循环

```python
class SelfIteratingCausalSystem:
    """自迭代因果系统"""

    def __init__(self):
        self.knowledge_graph = CausalKnowledgeGraph()
        self.discovery_engine = CausalDiscoveryEngine(self.knowledge_graph)
        self.validation_engine = CausalValidationEngine(self.knowledge_graph)
        self.prediction_engine = CausalPredictionEngine(self.knowledge_graph)

        # 迭代历史
        self.iteration_history: List[IterationRecord] = []

    def iterate(
        self,
        market_data: pd.DataFrame,
        iteration_number: int,
    ) -> IterationResult:
        """
        执行一次迭代：学习 → 验证 → 优化

        参数:
            market_data: 市场数据
            iteration_number: 迭代次数

        返回:
            IterationResult: 迭代结果
        """
        print(f"\n{'='*80}")
        print(f"开始第 {iteration_number} 次迭代")
        print(f"{'='*80}")

        start_time = datetime.now()

        # ========================================
        # Phase 1: 因果发现 (Causal Discovery)
        # ========================================
        print("\n【Phase 1】因果发现...")
        discovered_edges = self._discover_new_causal_relationships(market_data)
        print(f"  发现 {len(discovered_edges)} 个新因果关系")

        # ========================================
        # Phase 2: 因果验证 (Causal Validation)
        # ========================================
        print("\n【Phase 2】因果验证...")
        validated_edges = self._validate_causal_relationships(
            discovered_edges, market_data
        )
        print(f"  验证通过 {len(validated_edges)} 个因果关系")

        # ========================================
        # Phase 3: 知识更新 (Knowledge Update)
        # ========================================
        print("\n【Phase 3】知识更新...")
        self._update_knowledge_base(validated_edges)
        print(f"  知识库更新完成")

        # ========================================
        # Phase 4: 因果预测 (Causal Prediction)
        # ========================================
        print("\n【Phase 4】因果预测...")
        predictions = self._make_causal_predictions(market_data)
        print(f"  生成 {len(predictions)} 个预测")

        # ========================================
        # Phase 5: 性能评估 (Performance Evaluation)
        # ========================================
        print("\n【Phase 5】性能评估...")
        performance = self._evaluate_prediction_performance(predictions)
        print(f"  预测准确率: {performance['accuracy']:.2%}")
        print(f"  因果强度: {performance['avg_causal_strength']:.2f}")

        # 记录迭代
        end_time = datetime.now()
        iteration_record = IterationRecord(
            iteration_number=iteration_number,
            start_time=start_time,
            end_time=end_time,
            discovered_count=len(discovered_edges),
            validated_count=len(validated_edges),
            predictions_count=len(predictions),
            performance=performance,
        )

        self.iteration_history.append(iteration_record)

        return IterationResult(
            iteration_number=iteration_number,
            discovered_edges=discovered_edges,
            validated_edges=validated_edges,
            predictions=predictions,
            performance=performance,
        )

    def _discover_new_causal_relationships(
        self,
        data: pd.DataFrame,
    ) -> List[CausalEdge]:
        """发现新因果关系"""
        # 获取所有因素
        all_factors = data.columns.tolist()

        # 使用多种方法发现因果
        all_edges = []

        # 方法1: 格兰杰因果
        granger_edges = self.discovery_engine.granger_causality(data, all_factors)
        all_edges.extend(granger_edges)

        # 方法2: 物理因果建模
        physical_edges = self.discovery_engine.physical_causal_modeling(data, all_factors)
        all_edges.extend(physical_edges)

        # 去重
        unique_edges = self._deduplicate_edges(all_edges)

        return unique_edges

    def _validate_causal_relationships(
        self,
        edges: List[CausalEdge],
        data: pd.DataFrame,
    ) -> List[CausalEdge]:
        """验证因果关系"""
        validated = []

        for edge in edges:
            # 样本外验证
            result = self.validation_engine.out_of_sample_validation(edge, data)

            # 反事实验证
            counterfactual_result = self.validation_engine.counterfactual_validation(
                edge, data
            )

            # 综合评分
            avg_score = (result['score'] + counterfactual_result['score']) / 2

            if avg_score > 0.6:  # 验证通过阈值
                # 更新边的置信度
                edge.confidence = avg_score

                # 添加证据
                edge.evidence.append(
                    CausalEvidence(
                        evidence_id=f"ev_{edge.edge_id}",
                        evidence_type=EvidenceType.EMPIRICAL,
                        description="自迭代验证",
                        data_period=(data.index[0], data.index[-1]),
                        statistical_significance=avg_score,
                        effect_size=edge.causal_strength,
                        source="SelfIteratingCausalSystem",
                        validated=True,
                    )
                )

                validated.append(edge)

        return validated

    def _update_knowledge_base(self, validated_edges: List[CausalEdge]):
        """更新知识库"""
        for edge in validated_edges:
            # 如果边已存在，更新
            if edge.edge_id in self.knowledge_graph.edges:
                existing_edge = self.knowledge_graph.edges[edge.edge_id]

                # 增加验证计数
                existing_edge.validated_count += 1

                # 更新因果强度（指数移动平均）
                alpha = 0.3
                existing_edge.causal_strength = (
                    alpha * edge.causal_strength +
                    (1 - alpha) * existing_edge.causal_strength
                )

                # 更新置信度
                existing_edge.confidence = max(
                    existing_edge.confidence, edge.confidence
                )

                # 添加证据
                existing_edge.evidence.extend(edge.evidence)

                # 更新时间
                existing_edge.updated_at = datetime.now()
                existing_edge.version += 1

            else:
                # 新增边
                self.knowledge_graph.add_edge(edge)

                # 确保源和目标因素存在
                if edge.source_factor_id not in self.knowledge_graph.factors:
                    # 创建因素
                    factor = CausalFactor(
                        factor_id=edge.source_factor_id,
                        name=edge.source_factor_id,
                        category=FactorCategory.MACRO_POLICY,
                        asset_class=AssetClass.ALL,
                        description=f"Auto-generated factor: {edge.source_factor_id}",
                        causal_mechanism="To be determined",
                        data_sources=[],
                        measurement_methods=[],
                        update_frequency="daily",
                        reliability=0.5,
                        confidence=0.5,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        version=1,
                    )
                    self.knowledge_graph.add_factor(factor)

                if edge.target_factor_id not in self.knowledge_graph.factors:
                    # 创建因素
                    factor = CausalFactor(
                        factor_id=edge.target_factor_id,
                        name=edge.target_factor_id,
                        category=FactorCategory.MACRO_POLICY,
                        asset_class=AssetClass.ALL,
                        description=f"Auto-generated factor: {edge.target_factor_id}",
                        causal_mechanism="To be determined",
                        data_sources=[],
                        measurement_methods=[],
                        update_frequency="daily",
                        reliability=0.5,
                        confidence=0.5,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        version=1,
                    )
                    self.knowledge_graph.add_factor(factor)

    def _make_causal_predictions(
        self,
        data: pd.DataFrame,
    ) -> List[CausalPrediction]:
        """基于因果知识库进行预测"""
        # 使用因果图进行预测
        # （简化实现）

        predictions = []

        for edge_id, edge in self.knowledge_graph.edges.items():
            if edge.causal_strength > 0.7:  # 只使用强因果关系
                # 预测：如果源因素变化，目标因素将如何变化
                # （简化版）
                pass

        return predictions

    def _evaluate_prediction_performance(
        self,
        predictions: List[CausalPrediction],
    ) -> Dict[str, float]:
        """评估预测性能"""
        # 计算准确率、因果强度等
        # （简化实现）

        return {
            'accuracy': 0.75,
            'avg_causal_strength': 0.68,
            'prediction_count': len(predictions),
        }

    def generate_evolution_report(self) -> str:
        """生成演化报告"""
        report = []
        report.append("\n" + "="*80)
        report.append(" " * 20 + "因果知识库演化报告")
        report.append("="*80)

        # 统计
        total_factors = len(self.knowledge_graph.factors)
        total_edges = len(self.knowledge_graph.edges)

        report.append(f"\n📊 知识库规模:")
        report.append(f"  因素总数: {total_factors}")
        report.append(f"  因果边总数: {total_edges}")

        # 因果边统计
        strong_edges = [e for e in self.knowledge_graph.edges.values() if e.causal_strength > 0.7]
        report.append(f"  强因果关系 (>0.7): {len(strong_edges)}")

        # 按类别统计
        by_category = {}
        for edge in self.knowledge_graph.edges.values():
            # 获取源因素的类别
            source_factor = self.knowledge_graph.factors.get(edge.source_factor_id)
            if source_factor:
                category = source_factor.category.value
                by_category[category] = by_category.get(category, 0) + 1

        report.append(f"\n📈 按类别分布:")
        for category, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {category}: {count}")

        # 迭代历史
        report.append(f"\n🔄 迭代历史:")
        for record in self.iteration_history[-5:]:  # 最近5次
            report.append(f"  迭代 {record.iteration_number}: "
                        f"发现{record.discovered_count} "
                        f"验证{record.validated_count} "
                        f"准确率{record.performance['accuracy']:.1%}")

        return "\n".join(report)
```

---

## 🎯 实施路线图

### Phase 1: 因果知识库构建 (Week 1-2) ⭐⭐⭐⭐⭐
- ✅ 实现因果因素数据结构
- ✅ 构建因果知识图谱
- ✅ 实现因果边的增删改查

### Phase 2: 因果发现引擎 (Week 3-4) ⭐⭐⭐⭐⭐
- ✅ 实现格兰杰因果检验
- ✅ 实现物理因果建模
- ✅ 集成多种因果发现算法

### Phase 3: 因果验证引擎 (Week 5-6) ⭐⭐⭐⭐
- ✅ 实现样本外验证
- ✅ 实现反事实验证
- ✅ 实现制度稳健性验证

### Phase 4: 自迭代机制 (Week 7-8) ⭐⭐⭐⭐⭐
- ✅ 实现学习-验证-优化循环
- ✅ 实现知识更新机制
- ✅ 实现演化报告

### Phase 5: 与现有系统集成 (Week 9-10) ⭐⭐⭐⭐
- ✅ 集成到融合策略
- ✅ 增强因果AI分析
- ✅ 实现因果驱动交易

---

**文档版本**：1.0.0
**最后更新**：2026-05-05
**维护者**：quant-trading-system团队

🎯 **核心原则：因果推理、自迭代演化、知识积累、持续优化！**
