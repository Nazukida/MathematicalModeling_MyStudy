"""
============================================================
模型验证工作流 (Model Validation Pipeline)
============================================================

【最简单的使用方式 - 一个参数切换模型】

    from workflow.model_validation_pipeline import *
    
    pipeline = ModelValidationPipeline("我的任务")
    pipeline.load_data(my_data, "数据")
    pipeline.set_model(get_model("kmeans"))  # ← 改这个字符串切换模型！
    pipeline.configure_model(n_clusters=4)    # ← 改这里调参数
    pipeline.run()
    result = pipeline.get_model_result()

【可用的模型名称】（传给 get_model() 的字符串参数）

    聚类: "kmeans", "hierarchical"
    分类: "decision_tree", "knn", "naive_bayes", "random_forest"★, "svm", "xgboost_cls"
    回归: "linear", "ridge", "lasso", "polynomial", "xgboost_reg"
    预测: "grey", "arima", "exp_smoothing"
    评价: "topsis"★, "entropy"★, "ahp"
    优化: "dp", "pso"★, "ga"★, "sa", "linear_prog", "integer_prog"
    降维: "pca"
    模拟: "monte_carlo"★
    
    ★ = 调用 models/ 目录下的完整实现，其他为适配器内嵌实现

【各模型常用参数】（configure_model 可以设置的参数）

    kmeans:        n_clusters=3
    hierarchical:  n_clusters=3, linkage='ward'
    decision_tree: max_depth=None, test_size=0.2
    knn:           n_neighbors=5, test_size=0.2
    random_forest: n_estimators=100, max_depth=None
    svm:           C=1.0, test_size=0.2
    xgboost_*:     n_estimators=100, max_depth=6, learning_rate=0.1
    
    linear/ridge:  alpha=1.0 (ridge/lasso专用)
    polynomial:    degree=2
    
    grey:          n_predict=5
    arima:         order=(1,1,1), n_predict=5
    exp_smoothing: alpha=0.3, n_predict=5
    
    topsis:        weights=[...], is_benefit=[True, False, ...]
    entropy:       is_benefit=[True, False, ...]
    ahp:           comparison_matrix=[[1,2,3],[1/2,1,2],[1/3,1/2,1]]
    
    dp:            capacity=10
    pso/ga/sa:     bounds=(-5,5), n_dims=2, max_iter=100 (需要先 set_objective)
    linear_prog:   c=[...], A_ub=[[...]], b_ub=[...]
    integer_prog:  同上，加 integrality=[1,1,0,...]
    
    pca:           n_components=2
    monte_carlo:   n_simulations=10000, confidence=0.95 (需要先 set_simulation)

【模型库调用说明】

    本 pipeline 中的适配器会自动检测并调用 models/ 目录下的模型类：
    
    - TOPSIS/熵权法  → models.evaluation.evaluation_toolkit.TOPSIS/EntropyWeightMethod
    - PSO/遗传算法   → models.optimization.optimization_toolkit.PSO/GeneticAlgorithm
    - 随机森林分类   → models.classification.classification_toolkit.RandomForestModel
    - 蒙特卡洛模拟   → models.probability.monte_carlo_simulation.MonteCarloSimulator
    
    如果导入失败，会自动回退到适配器内的简化实现。

============================================================
【完整使用流程】

    # 1. 导入
    from workflow.model_validation_pipeline import *
    
    # 2. 创建工作流
    pipeline = ModelValidationPipeline("任务名")
    
    # 3. 加载数据 (DataFrame/array/list/dict)
    pipeline.load_data(my_data, "数据描述")
    
    # 4. 预处理（可选，可多个）
    pipeline.add_preprocessing(MissingValueStep('mean'))
    pipeline.add_preprocessing(OutlierRemovalStep('iqr'))
    pipeline.add_preprocessing(NormalizationStep('minmax'))
    
    # 5. 设置模型 ← 核心：改 get_model("xxx") 的参数
    pipeline.set_model(get_model("topsis"))
    pipeline.configure_model(weights=[0.3, 0.3, 0.2, 0.2])
    
    # 6. 可视化（可选）
    pipeline.add_visualization(DPTableVisualization())
    
    # 7. 运行
    pipeline.run()
    
    # 8. 获取结果
    result = pipeline.get_model_result()
    data = pipeline.get_processed_data()

============================================================
【预处理选项】

    MissingValueStep('mean')      - 均值填充
    MissingValueStep('median')    - 中位数填充
    MissingValueStep('knn')       - KNN插补
    MissingValueStep('drop')      - 删除缺失行
    OutlierRemovalStep('iqr')     - IQR异常值处理
    OutlierRemovalStep('zscore')  - Z-score异常值处理
    NormalizationStep('zscore')   - Z-score标准化
    NormalizationStep('minmax')   - Min-Max归一化
    NormalizationStep('robust')   - 稳健标准化

============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

# 添加路径以导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 导入各模块
try:
    from data_preprocessing.preprocessing_toolkit import (
        MissingValueHandler, OutlierDetector, SampleDataGenerator
    )
except ImportError:
    MissingValueHandler = None
    OutlierDetector = None
    SampleDataGenerator = None

try:
    from visualization.plot_config import PlotStyleConfig, PlotTemplates, FigureSaver
except ImportError:
    PlotStyleConfig = None
    PlotTemplates = None
    FigureSaver = None

# ===== 导入模型库 =====
# 评价模型
try:
    from models.evaluation.evaluation_toolkit import (
        EntropyWeightMethod as _EntropyWeightMethod,
        TOPSIS as _TOPSIS
    )
except ImportError:
    _EntropyWeightMethod = None
    _TOPSIS = None

# 优化模型
try:
    from models.optimization.optimization_toolkit import (
        ParticleSwarmOptimization as _PSO,
        GeneticAlgorithm as _GA
    )
except ImportError:
    _PSO = None
    _GA = None

# 动力学模型
try:
    from models.dynamics.dynamics_toolkit import (
        SIRModel as _SIRModel,
        SEIRModel as _SEIRModel,
        LotkaVolterra as _LotkaVolterra,
        PopulationDynamics as _PopulationDynamics
    )
except ImportError:
    _SIRModel = None
    _SEIRModel = None
    _LotkaVolterra = None
    _PopulationDynamics = None

# 分类模型
try:
    from models.classification.classification_toolkit import (
        RandomForestModel as _RandomForestModel,
        EnsembleClassifier as _EnsembleClassifier,
        BaseClassifier as _BaseClassifier
    )
except ImportError:
    _RandomForestModel = None
    _EnsembleClassifier = None
    _BaseClassifier = None

# 预测模型
try:
    from models.prediction.prediction_toolkit import (
        TimeSeriesAnalyzer as _TimeSeriesAnalyzer
    )
except ImportError:
    _TimeSeriesAnalyzer = None

# 概率/模拟模型
try:
    from models.probability.monte_carlo_simulation import (
        MonteCarloSimulator as _MonteCarloSimulator
    )
except ImportError:
    _MonteCarloSimulator = None


# ============================================================
# 第一部分：统一数据格式 (Unified Data Format)
# ============================================================
"""
【PipelineData 是什么？】
- 数据的"包装盒"，让数据能在各模块间传递
- 你不需要直接创建它，pipeline.load_data() 会自动创建

【你可能用到的方法】
- pipeline_data.get_dataframe()  → 获取 pandas DataFrame
- pipeline_data.get_array()      → 获取 numpy array  
- pipeline_data.get_list()       → 获取 Python list
- pipeline_data.summary()        → 打印数据摘要
"""

class PipelineData:
    """
    工作流数据容器 - 统一各模块间的数据传递格式
    
    作用：确保数据在预处理、模型、可视化之间顺利流转
    """
    
    def __init__(self, data=None, name="未命名数据"):
        """
        初始化数据容器
        
        :param data: 原始数据 (DataFrame, ndarray, dict, list)
        :param name: 数据名称
        """
        self.name = name
        self.raw_data = None           # 原始数据
        self.processed_data = None     # 预处理后的数据
        self.model_input = None        # 模型输入格式
        self.model_output = None       # 模型输出结果
        self.metadata = {}             # 元数据（列名、类型等）
        self.history = []              # 处理历史记录
        
        if data is not None:
            self.load(data)
    
    def load(self, data):
        """加载数据"""
        if isinstance(data, pd.DataFrame):
            self.raw_data = data.copy()
            self.metadata['columns'] = list(data.columns)
            self.metadata['dtypes'] = data.dtypes.to_dict()
        elif isinstance(data, np.ndarray):
            self.raw_data = pd.DataFrame(data)
            self.metadata['columns'] = list(self.raw_data.columns)
        elif isinstance(data, dict):
            self.raw_data = pd.DataFrame(data)
            self.metadata['columns'] = list(self.raw_data.columns)
        elif isinstance(data, list):
            self.raw_data = pd.DataFrame(data)
            self.metadata['columns'] = list(self.raw_data.columns)
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")
        
        self.processed_data = self.raw_data.copy()
        self._log("数据加载完成", f"形状: {self.raw_data.shape}")
        return self
    
    def _log(self, operation, details=""):
        """记录操作历史"""
        self.history.append({
            'time': datetime.now().strftime("%H:%M:%S"),
            'operation': operation,
            'details': details
        })
    
    def get_array(self):
        """获取numpy数组格式"""
        return self.processed_data.values if self.processed_data is not None else None
    
    def get_dataframe(self):
        """获取DataFrame格式"""
        return self.processed_data
    
    def get_list(self):
        """获取列表格式（适合动态规划等）"""
        return self.processed_data.values.tolist() if self.processed_data is not None else None
    
    def get_dict(self):
        """获取字典格式"""
        return self.processed_data.to_dict('list') if self.processed_data is not None else None
    
    def set_model_output(self, output, output_type="general"):
        """
        设置模型输出
        
        :param output: 模型输出结果
        :param output_type: 输出类型标签
        """
        self.model_output = {
            'result': output,
            'type': output_type,
            'timestamp': datetime.now()
        }
        self._log("模型输出已设置", f"类型: {output_type}")
        return self
    
    def summary(self):
        """打印数据摘要"""
        print("\n" + "="*60)
        print(f"📦 数据容器: {self.name}")
        print("="*60)
        print(f"  原始数据形状: {self.raw_data.shape if self.raw_data is not None else 'None'}")
        print(f"  处理后数据形状: {self.processed_data.shape if self.processed_data is not None else 'None'}")
        print(f"  列名: {self.metadata.get('columns', [])}")
        print(f"  模型输出: {'已设置' if self.model_output else '未设置'}")
        print(f"\n  📋 处理历史:")
        for h in self.history[-5:]:  # 只显示最近5条
            print(f"    [{h['time']}] {h['operation']}: {h['details']}")
        print("="*60)


# ============================================================
# 第二部分：预处理步骤 (Preprocessing Steps)
# ============================================================
"""
【预处理步骤是什么？】
- 对数据进行清洗、转换的操作
- 可以添加0个、1个或多个步骤
- 按添加顺序依次执行

【可用的预处理步骤】（在第4步 add_preprocessing 时选择）

1. 缺失值处理 - MissingValueStep(method)
   method 可选值：
   - 'mean'   : 用该列均值填充（推荐用于正态分布数据）
   - 'median' : 用该列中位数填充（推荐用于有偏斜的数据）
   - 'mode'   : 用该列众数填充（推荐用于分类数据）
   - 'knn'    : 用KNN算法插补（推荐用于有相关性的多列数据）
   - 'drop'   : 直接删除含缺失值的行
   
   示例：pipeline.add_preprocessing(MissingValueStep('mean'))

2. 异常值处理 - OutlierRemovalStep(method, threshold)
   method 可选值：
   - 'iqr'    : 四分位距方法，threshold建议1.5（默认）
   - 'zscore' : 标准差方法，threshold建议2或3
   
   示例：pipeline.add_preprocessing(OutlierRemovalStep('iqr', 1.5))

3. 标准化 - NormalizationStep(method)
   method 可选值：
   - 'zscore' : Z-score标准化，结果均值0标准差1
   - 'minmax' : Min-Max归一化，结果在[0,1]之间
   - 'robust' : 稳健标准化，对异常值不敏感
   
   示例：pipeline.add_preprocessing(NormalizationStep('minmax'))

【组合示例】
   # 先填充缺失值，再处理异常值，最后标准化
   pipeline.add_preprocessing(MissingValueStep('mean'))
   pipeline.add_preprocessing(OutlierRemovalStep('iqr', 1.5))
   pipeline.add_preprocessing(NormalizationStep('minmax'))
"""

class PreprocessingStep:
    """预处理步骤基类"""
    
    def __init__(self, name="预处理步骤"):
        self.name = name
        self.params = {}
    
    def apply(self, pipeline_data: PipelineData) -> PipelineData:
        """应用预处理步骤"""
        raise NotImplementedError


class MissingValueStep(PreprocessingStep):
    """缺失值处理步骤"""
    
    def __init__(self, method='mean', **kwargs):
        """
        :param method: 'mean', 'median', 'mode', 'knn', 'drop'
        """
        super().__init__(f"缺失值处理({method})")
        self.method = method
        self.params = kwargs
    
    def apply(self, pipeline_data: PipelineData) -> PipelineData:
        data = pipeline_data.get_dataframe()
        
        if data.isnull().sum().sum() == 0:
            pipeline_data._log(self.name, "无缺失值，跳过处理")
            return pipeline_data
        
        if MissingValueHandler is not None:
            handler = MissingValueHandler(verbose=False)
            if self.method == 'drop':
                filled = handler.drop_missing(data, **self.params)
            else:
                filled = handler.fill_missing(data, method=self.method, **self.params)
        else:
            # 备用实现
            if self.method == 'mean':
                filled = data.fillna(data.mean())
            elif self.method == 'median':
                filled = data.fillna(data.median())
            elif self.method == 'drop':
                filled = data.dropna()
            else:
                filled = data.fillna(0)
        
        pipeline_data.processed_data = filled
        pipeline_data._log(self.name, f"处理了 {data.isnull().sum().sum()} 个缺失值")
        return pipeline_data


class OutlierRemovalStep(PreprocessingStep):
    """异常值处理步骤"""
    
    def __init__(self, method='iqr', threshold=1.5):
        """
        :param method: 'iqr' 或 'zscore'
        :param threshold: IQR倍数 或 Z-score阈值
        """
        super().__init__(f"异常值处理({method})")
        self.method = method
        self.threshold = threshold
    
    def apply(self, pipeline_data: PipelineData) -> PipelineData:
        data = pipeline_data.get_dataframe().copy()
        outlier_count = 0
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if self.method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.threshold * IQR
                upper = Q3 + self.threshold * IQR
                mask = (data[col] < lower) | (data[col] > upper)
            else:  # zscore
                z = np.abs((data[col] - data[col].mean()) / data[col].std())
                mask = z > self.threshold
            
            outlier_count += mask.sum()
            # 用边界值替换
            if self.method == 'iqr':
                data.loc[data[col] < lower, col] = lower
                data.loc[data[col] > upper, col] = upper
        
        pipeline_data.processed_data = data
        pipeline_data._log(self.name, f"处理了 {outlier_count} 个异常值")
        return pipeline_data


class NormalizationStep(PreprocessingStep):
    """数据标准化步骤"""
    
    def __init__(self, method='zscore'):
        """
        :param method: 'zscore', 'minmax', 'robust'
        """
        super().__init__(f"标准化({method})")
        self.method = method
    
    def apply(self, pipeline_data: PipelineData) -> PipelineData:
        data = pipeline_data.get_dataframe().copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.method == 'zscore':
                data[col] = (data[col] - data[col].mean()) / data[col].std()
            elif self.method == 'minmax':
                min_val = data[col].min()
                max_val = data[col].max()
                data[col] = (data[col] - min_val) / (max_val - min_val + 1e-10)
            elif self.method == 'robust':
                median = data[col].median()
                iqr = data[col].quantile(0.75) - data[col].quantile(0.25)
                data[col] = (data[col] - median) / (iqr + 1e-10)
        
        pipeline_data.processed_data = data
        pipeline_data._log(self.name, f"标准化了 {len(numeric_cols)} 列")
        return pipeline_data


# ============================================================
# 第三部分：模型适配器 (Model Adapters)
# ============================================================
"""
【模型适配器是什么？】
- 把你的模型"包装"成工作流能识别的格式
- 必须设置一个模型（用 set_model）
- 可以设置参数（用 configure_model）

============================================================
【已实现的模型适配器】（可直接使用）
============================================================

1. DynamicProgrammingAdapter() - 动态规划（背包问题）
2. OptimizationAdapter('pso')  - 粒子群优化
3. LinearProgrammingAdapter()  - 线性规划
4. GreyPredictionAdapter()     - 灰色预测
5. TOPSISAdapter()             - TOPSIS评价

============================================================
【models/ 目录下可接入的模型】（需要自己写适配器）
============================================================

optimization/ 优化类：
    - linear_programming.py      线性规划
    - integer_programming.py     整数规划
    - zero_one_programming.py    0-1规划
    - nonlinear_programming.py   非线性规划
    - simulated_annealing.py     模拟退火
    - nsga2_multi_objective.py   NSGA2多目标优化

prediction/ 预测类：
    - grey_prediction.py         灰色预测
    - arma_prediction.py         ARMA时间序列
    - logistic_prediction.py     Logistic增长预测
    - markov_prediction.py       马尔可夫预测
    - prophet_forecast.py        Prophet预测
    - xgboost_regression.py      XGBoost回归

clustering/ 聚类类：
    - kmeans_clustering.py       K-means聚类
    - hierarchical_clustering.py 层次聚类
    - som_clustering.py          SOM自组织映射

classification/ 分类类：
    - decision_tree_classification.py  决策树
    - knn_classification.py            KNN
    - naive_bayes_classification.py    朴素贝叶斯

evaluation/ 评价类：
    - evaluation_toolkit.py      熵权法 + TOPSIS

dynamics/ 动力学：
    - glv_ecosystem_model.py     GLV生态系统模型
    - war_model.py               战争模型

============================================================
【如何接入上述任意模型？复制这个模板】
============================================================

假设你要用 models/prediction/grey_prediction.py 里的灰色预测：

```python
# 在你的 main.py 中：

from workflow.model_validation_pipeline import ModelAdapter, PipelineData
# 导入你要用的模型（根据实际路径调整）
# from models.prediction.grey_prediction import GreyPredictor

class GreyPredictionAdapter(ModelAdapter):
    '''灰色预测模型适配器'''
    
    def __init__(self):
        super().__init__("灰色预测GM(1,1)")
        self.params = {
            'n_predict': 5,  # 预测未来5个时间点
        }
    
    def run(self, pipeline_data):
        # 1. 获取数据
        data = pipeline_data.get_array().flatten()  # 一维时间序列
        
        # 2. 获取参数
        n_predict = self.params['n_predict']
        
        # 3. 调用你的模型
        # ============ 把模型代码放这里 ============
        # predictor = GreyPredictor()
        # predictor.fit(data)
        # predictions = predictor.predict(n_predict)
        
        # 示例：简单实现
        predictions = [data[-1] * 1.1 ** i for i in range(1, n_predict+1)]
        # ==========================================
        
        # 4. 保存结果
        self.result = {
            'predictions': predictions,
            'original_data': data,
        }
        
        # 5. 设置输出
        pipeline_data.set_model_output(self.result, "grey_prediction")
        return pipeline_data

# 使用：
# pipeline.set_model(GreyPredictionAdapter())
# pipeline.configure_model(n_predict=10)
```

============================================================
【更多适配器模板示例】
============================================================

--- 聚类模型适配器模板 ---
```python
class KMeansAdapter(ModelAdapter):
    def __init__(self):
        super().__init__("K-Means聚类")
        self.params = {'n_clusters': 3}
    
    def run(self, pipeline_data):
        from sklearn.cluster import KMeans
        data = pipeline_data.get_array()
        
        kmeans = KMeans(n_clusters=self.params['n_clusters'])
        labels = kmeans.fit_predict(data)
        
        self.result = {
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
        }
        pipeline_data.set_model_output(self.result, "clustering")
        return pipeline_data
```

--- 回归模型适配器模板 ---
```python
class RegressionAdapter(ModelAdapter):
    def __init__(self, method='linear'):
        super().__init__(f"{method}回归")
        self.method = method
        self.params = {}
    
    def run(self, pipeline_data):
        from sklearn.linear_model import LinearRegression, Ridge
        data = pipeline_data.get_dataframe()
        
        X = data.iloc[:, :-1].values  # 前n-1列为特征
        y = data.iloc[:, -1].values   # 最后一列为目标
        
        if self.method == 'linear':
            model = LinearRegression()
        elif self.method == 'ridge':
            model = Ridge(alpha=self.params.get('alpha', 1.0))
        
        model.fit(X, y)
        
        self.result = {
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'r2_score': model.score(X, y),
        }
        pipeline_data.set_model_output(self.result, "regression")
        return pipeline_data
```

--- TOPSIS评价适配器模板 ---
```python
class TOPSISAdapter(ModelAdapter):
    def __init__(self):
        super().__init__("TOPSIS评价")
        self.params = {
            'weights': None,      # 权重，None则等权
            'is_benefit': None,   # 各指标是否为效益型
        }
    
    def run(self, pipeline_data):
        data = pipeline_data.get_array()
        n_samples, n_features = data.shape
        
        # 权重
        weights = self.params['weights']
        if weights is None:
            weights = np.ones(n_features) / n_features
        
        # 归一化
        norm_data = data / np.sqrt(np.sum(data**2, axis=0))
        weighted = norm_data * weights
        
        # 理想解
        is_benefit = self.params['is_benefit']
        if is_benefit is None:
            is_benefit = [True] * n_features
        
        ideal_best = np.array([weighted[:, j].max() if is_benefit[j] else weighted[:, j].min() 
                               for j in range(n_features)])
        ideal_worst = np.array([weighted[:, j].min() if is_benefit[j] else weighted[:, j].max() 
                                for j in range(n_features)])
        
        # 距离和得分
        d_best = np.sqrt(np.sum((weighted - ideal_best)**2, axis=1))
        d_worst = np.sqrt(np.sum((weighted - ideal_worst)**2, axis=1))
        scores = d_worst / (d_best + d_worst + 1e-10)
        
        self.result = {
            'scores': scores,
            'ranking': np.argsort(-scores) + 1,  # 排名
            'weights': weights,
        }
        pipeline_data.set_model_output(self.result, "evaluation")
        return pipeline_data
```

--- 分类模型适配器模板 ---
```python
class ClassificationAdapter(ModelAdapter):
    def __init__(self, method='decision_tree'):
        super().__init__(f"{method}分类")
        self.method = method
        self.params = {'test_size': 0.2}
    
    def run(self, pipeline_data):
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score
        
        data = pipeline_data.get_dataframe()
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.params['test_size'])
        
        if self.method == 'decision_tree':
            model = DecisionTreeClassifier()
        elif self.method == 'knn':
            model = KNeighborsClassifier(n_neighbors=self.params.get('k', 5))
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.result = {
            'accuracy': accuracy_score(y_test, y_pred),
            'predictions': y_pred,
            'model': model,
        }
        pipeline_data.set_model_output(self.result, "classification")
        return pipeline_data
```

============================================================
【总结：接入任意模型的步骤】
============================================================

1. 创建类，继承 ModelAdapter
2. __init__ 中定义 self.params = {...} 参数
3. run() 中：
   - 用 pipeline_data.get_array() 或 get_dataframe() 获取数据
   - 用 self.params['xxx'] 获取参数
   - 运行你的模型逻辑
   - 把结果存入 self.result = {...}
   - 调用 pipeline_data.set_model_output(self.result, "类型名")
   - return pipeline_data

就这么简单！
"""

class ModelAdapter:
    """模型适配器基类 - 将不同模型统一为相同接口"""
    
    def __init__(self, name="模型"):
        self.name = name
        self.params = {}
        self.result = None
    
    def set_params(self, **kwargs):
        """设置模型参数"""
        self.params.update(kwargs)
        return self
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        """运行模型"""
        raise NotImplementedError
    
    def get_result(self):
        """获取结果"""
        return self.result


class DynamicProgrammingAdapter(ModelAdapter):
    """动态规划模型适配器 - 背包问题示例"""
    
    def __init__(self):
        super().__init__("动态规划-背包问题")
        self.params = {
            'capacity': 10,  # 背包容量
        }
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        """
        运行动态规划
        
        期望输入数据格式：每行一个物品，列为 [重量, 价值] 或 DataFrame
        """
        data = pipeline_data.get_dataframe()
        
        # 提取重量和价值
        if data.shape[1] >= 2:
            weights = data.iloc[:, 0].values.astype(int)
            values = data.iloc[:, 1].values.astype(int)
        else:
            raise ValueError("数据格式错误：需要至少两列（重量和价值）")
        
        capacity = self.params.get('capacity', 10)
        n = len(weights)
        
        # DP求解
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for j in range(capacity + 1):
                if weights[i-1] <= j:
                    dp[i][j] = max(
                        values[i-1] + dp[i-1][j-weights[i-1]],
                        dp[i-1][j]
                    )
                else:
                    dp[i][j] = dp[i-1][j]
        
        # 回溯
        selected = []
        j = capacity
        for i in range(n, 0, -1):
            if dp[i][j] != dp[i-1][j]:
                selected.append(i-1)
                j -= weights[i-1]
        
        self.result = {
            'max_value': dp[n][capacity],
            'selected_items': selected,
            'total_weight': sum(weights[i] for i in selected),
            'dp_table': np.array(dp),
            'weights': weights,
            'values': values,
            'capacity': capacity
        }
        
        pipeline_data.set_model_output(self.result, "dynamic_programming")
        pipeline_data._log(self.name, f"最大价值: {self.result['max_value']}")
        return pipeline_data


class OptimizationAdapter(ModelAdapter):
    """
    优化算法适配器 - 调用 models.optimization.optimization_toolkit
    
    支持 PSO (粒子群优化)
    
    用法：
        model = get_model("pso")
        model.set_objective(lambda x: x[0]**2 + x[1]**2)  # 设置目标函数
        model.set_params(bounds=(-5, 5), n_dims=2, max_iter=100)
    """
    
    def __init__(self, algorithm='pso'):
        super().__init__(f"优化算法-{algorithm.upper()}")
        self.algorithm = algorithm
        self.params = {
            'n_particles': 30,  # PSO: 粒子数
            'pop_size': 30,     # GA: 种群大小
            'max_iter': 100,
            'bounds': (-5, 5),  # 搜索范围
            'n_dims': 2,        # 变量维度
        }
        self.objective_func = None
        self._optimizer = None
    
    def set_objective(self, func):
        """设置目标函数 f(x) -> float"""
        self.objective_func = func
        return self
    
    def run(self, pipeline_data: PipelineData = None) -> PipelineData:
        """运行优化"""
        if self.objective_func is None:
            raise ValueError("请先调用 set_objective(func) 设置目标函数")
        
        bounds = self.params['bounds']
        n_dims = self.params['n_dims']
        max_iter = self.params['max_iter']
        
        # 转换 bounds 格式: (min, max) -> [(min, max), (min, max), ...]
        if isinstance(bounds, tuple) and len(bounds) == 2:
            bounds_list = [bounds] * n_dims
        else:
            bounds_list = list(bounds)
        
        # 优先使用库中的优化器
        if self.algorithm == 'pso' and _PSO is not None:
            optimizer = _PSO(
                objective_func=self.objective_func,
                bounds=bounds_list,
                n_dims=n_dims,
                pop_size=self.params.get('n_particles', 30),
                max_iter=max_iter,
                verbose=False
            )
            optimizer.optimize()
            self._optimizer = optimizer
            
            self.result = {
                'best_position': optimizer.best_position,
                'best_value': optimizer.best_value,
                'convergence_history': optimizer.history,
            }
        elif self.algorithm == 'ga' and _GA is not None:
            optimizer = _GA(
                objective_func=self.objective_func,
                bounds=bounds_list,
                n_dims=n_dims,
                pop_size=self.params.get('pop_size', 50),
                max_iter=max_iter,
                verbose=False
            )
            optimizer.optimize()
            self._optimizer = optimizer
            
            self.result = {
                'best_position': optimizer.best_position,
                'best_value': optimizer.best_value,
                'convergence_history': optimizer.history,
            }
        else:
            # 回退到内置 PSO 实现
            n_particles = self.params['n_particles']
            
            lb = np.array([b[0] for b in bounds_list])
            ub = np.array([b[1] for b in bounds_list])
            
            positions = np.random.uniform(lb, ub, (n_particles, n_dims))
            velocities = np.random.uniform(-1, 1, (n_particles, n_dims))
            pbest_pos = positions.copy()
            pbest_val = np.array([self.objective_func(p) for p in positions])
            gbest_idx = np.argmin(pbest_val)
            gbest_pos = pbest_pos[gbest_idx].copy()
            gbest_val = pbest_val[gbest_idx]
            
            history = [gbest_val]
            w, c1, c2 = 0.7, 1.5, 1.5
            
            for _ in range(max_iter):
                r1, r2 = np.random.rand(n_particles, n_dims), np.random.rand(n_particles, n_dims)
                velocities = w * velocities + c1*r1*(pbest_pos - positions) + c2*r2*(gbest_pos - positions)
                positions = positions + velocities
                positions = np.clip(positions, lb, ub)
                
                fitness = np.array([self.objective_func(p) for p in positions])
                improved = fitness < pbest_val
                pbest_pos[improved] = positions[improved]
                pbest_val[improved] = fitness[improved]
                
                if np.min(pbest_val) < gbest_val:
                    gbest_idx = np.argmin(pbest_val)
                    gbest_pos = pbest_pos[gbest_idx].copy()
                    gbest_val = pbest_val[gbest_idx]
                
                history.append(gbest_val)
            
            self.result = {
                'best_position': gbest_pos,
                'best_value': gbest_val,
                'convergence_history': history
            }
        
        if pipeline_data:
            pipeline_data.set_model_output(self.result, "optimization")
            pipeline_data._log(self.name, f"最优值: {self.result['best_value']:.6f}")
        
        return pipeline_data


# ============================================================
# 更多内置模型适配器
# ============================================================

class LinearProgrammingAdapter(ModelAdapter):
    """
    线性规划适配器
    
    用法：
        pipeline.set_model(LinearProgrammingAdapter())
        pipeline.configure_model(
            c=[-2, -3],           # 目标函数系数（最小化 c^T x）
            A_ub=[[1, 1], [2, 1]],# 不等式约束矩阵
            b_ub=[4, 5],          # 不等式约束右端
            bounds=[(0, None), (0, None)]  # 变量范围
        )
    """
    
    def __init__(self):
        super().__init__("线性规划")
        self.params = {
            'c': None,        # 目标函数系数
            'A_ub': None,     # 不等式约束 Ax <= b
            'b_ub': None,
            'A_eq': None,     # 等式约束 Ax = b
            'b_eq': None,
            'bounds': None,   # 变量范围
        }
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        from scipy.optimize import linprog
        
        result = linprog(
            c=self.params['c'],
            A_ub=self.params.get('A_ub'),
            b_ub=self.params.get('b_ub'),
            A_eq=self.params.get('A_eq'),
            b_eq=self.params.get('b_eq'),
            bounds=self.params.get('bounds'),
            method='highs'
        )
        
        self.result = {
            'optimal_value': -result.fun if result.success else None,  # 转为最大化
            'optimal_solution': result.x,
            'success': result.success,
            'message': result.message,
        }
        
        pipeline_data.set_model_output(self.result, "linear_programming")
        pipeline_data._log(self.name, f"最优值: {self.result['optimal_value']}")
        return pipeline_data


class GreyPredictionAdapter(ModelAdapter):
    """
    灰色预测GM(1,1)适配器
    
    数据格式：一列时间序列数据
    
    用法：
        pipeline.set_model(GreyPredictionAdapter())
        pipeline.configure_model(n_predict=5)  # 预测未来5个点
    """
    
    def __init__(self):
        super().__init__("灰色预测GM(1,1)")
        self.params = {
            'n_predict': 5,  # 预测步数
        }
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        data = pipeline_data.get_array().flatten()
        n = len(data)
        n_predict = self.params['n_predict']
        
        # 累加生成
        x1 = np.cumsum(data)
        
        # 构建矩阵
        B = np.zeros((n-1, 2))
        Y = np.zeros((n-1, 1))
        for i in range(n-1):
            B[i, 0] = -0.5 * (x1[i] + x1[i+1])
            B[i, 1] = 1
            Y[i, 0] = data[i+1]
        
        # 最小二乘求参数
        params = np.linalg.lstsq(B, Y, rcond=None)[0]
        a, b = params[0, 0], params[1, 0]
        
        # 预测
        predictions = []
        for k in range(1, n + n_predict + 1):
            x1_pred = (data[0] - b/a) * np.exp(-a * (k-1)) + b/a
            predictions.append(x1_pred)
        
        # 累减还原
        predictions = np.diff(np.array([0] + predictions))
        
        self.result = {
            'fitted': predictions[:n],
            'predictions': predictions[n:],
            'a': a,
            'b': b,
            'original': data,
        }
        
        pipeline_data.set_model_output(self.result, "grey_prediction")
        pipeline_data._log(self.name, f"预测了 {n_predict} 个点")
        return pipeline_data


class KMeansAdapter(ModelAdapter):
    """
    K-Means聚类适配器
    
    用法：
        pipeline.set_model(KMeansAdapter())
        pipeline.configure_model(n_clusters=3)
    """
    
    def __init__(self):
        super().__init__("K-Means聚类")
        self.params = {
            'n_clusters': 3,
            'random_state': 42,
        }
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        from sklearn.cluster import KMeans
        
        data = pipeline_data.get_array()
        
        kmeans = KMeans(
            n_clusters=self.params['n_clusters'],
            random_state=self.params.get('random_state', 42),
            n_init=10
        )
        labels = kmeans.fit_predict(data)
        
        self.result = {
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'n_clusters': self.params['n_clusters'],
        }
        
        pipeline_data.set_model_output(self.result, "clustering")
        pipeline_data._log(self.name, f"聚成 {self.params['n_clusters']} 类")
        return pipeline_data


class TOPSISAdapter(ModelAdapter):
    """
    TOPSIS综合评价适配器 - 调用 models.evaluation.evaluation_toolkit.TOPSIS
    
    数据格式：每行一个评价对象，每列一个指标
    
    用法：
        pipeline.set_model(TOPSISAdapter())
        pipeline.configure_model(
            weights=[0.3, 0.3, 0.2, 0.2],  # 权重，None则等权
            is_benefit=[True, True, False, False]  # 是否效益型指标
        )
    """
    
    def __init__(self):
        super().__init__("TOPSIS评价")
        self.params = {
            'weights': None,
            'is_benefit': None,  # True=效益型(越大越好), False=成本型(越小越好)
        }
        self._topsis_model = None  # 保存原始模型实例
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        data = pipeline_data.get_dataframe()
        n_samples, n_features = data.shape
        
        weights = self.params.get('weights')
        is_benefit = self.params.get('is_benefit')
        
        # 转换 is_benefit 为 indicator_types 格式
        indicator_types = None
        if is_benefit is not None:
            indicator_types = ['positive' if b else 'negative' for b in is_benefit]
        
        # 优先使用库中的 TOPSIS 类
        if _TOPSIS is not None:
            topsis = _TOPSIS(verbose=False)
            topsis.fit(data, weights=weights, indicator_types=indicator_types)
            self._topsis_model = topsis
            
            self.result = {
                'scores': topsis.closeness,
                'ranking': np.argsort(-topsis.closeness) + 1,
                'weights': topsis.weights,
                'd_best': topsis.distances_positive,
                'd_worst': topsis.distances_negative,
                'results_df': topsis.get_results(),
            }
        else:
            # 回退到内置实现
            data_arr = data.values if isinstance(data, pd.DataFrame) else data
            if weights is None:
                weights = np.ones(n_features) / n_features
            weights = np.array(weights)
            
            norm_data = data_arr / np.sqrt(np.sum(data_arr**2, axis=0) + 1e-10)
            weighted = norm_data * weights
            
            if is_benefit is None:
                is_benefit = [True] * n_features
            
            ideal_best = np.array([
                weighted[:, j].max() if is_benefit[j] else weighted[:, j].min()
                for j in range(n_features)
            ])
            ideal_worst = np.array([
                weighted[:, j].min() if is_benefit[j] else weighted[:, j].max()
                for j in range(n_features)
            ])
            
            d_best = np.sqrt(np.sum((weighted - ideal_best)**2, axis=1))
            d_worst = np.sqrt(np.sum((weighted - ideal_worst)**2, axis=1))
            scores = d_worst / (d_best + d_worst + 1e-10)
            
            self.result = {
                'scores': scores,
                'ranking': np.argsort(-scores) + 1,
                'weights': weights,
                'd_best': d_best,
                'd_worst': d_worst,
            }
        
        pipeline_data.set_model_output(self.result, "evaluation")
        pipeline_data._log(self.name, f"评价了 {n_samples} 个对象")
        return pipeline_data


class RegressionAdapter(ModelAdapter):
    """
    回归模型适配器
    
    数据格式：前n-1列为特征X，最后一列为目标y
    
    用法：
        pipeline.set_model(RegressionAdapter('linear'))  # 或 'ridge', 'lasso'
        pipeline.configure_model(alpha=1.0)  # Ridge/Lasso的正则化参数
    """
    
    def __init__(self, method='linear'):
        """
        :param method: 'linear', 'ridge', 'lasso', 'polynomial'
        """
        super().__init__(f"{method.capitalize()}回归")
        self.method = method
        self.params = {
            'alpha': 1.0,
            'degree': 2,  # 多项式回归的阶数
        }
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import r2_score, mean_squared_error
        
        data = pipeline_data.get_dataframe()
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        if self.method == 'polynomial':
            poly = PolynomialFeatures(degree=self.params['degree'])
            X = poly.fit_transform(X)
            model = LinearRegression()
        elif self.method == 'ridge':
            model = Ridge(alpha=self.params['alpha'])
        elif self.method == 'lasso':
            model = Lasso(alpha=self.params['alpha'])
        else:
            model = LinearRegression()
        
        model.fit(X, y)
        y_pred = model.predict(X)
        
        self.result = {
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'r2_score': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'predictions': y_pred,
        }
        
        pipeline_data.set_model_output(self.result, "regression")
        pipeline_data._log(self.name, f"R² = {self.result['r2_score']:.4f}")
        return pipeline_data


# ============================================================
# 更多模型适配器 (More Model Adapters)
# ============================================================

class HierarchicalClusteringAdapter(ModelAdapter):
    """
    层次聚类适配器
    
    数据格式：每行一个样本，每列一个特征
    
    参数：
        n_clusters: 聚类数量，默认3
        linkage: 连接方式 'ward'/'complete'/'average'/'single'，默认'ward'
    """
    def __init__(self):
        super().__init__("层次聚类")
        self.params = {'n_clusters': 3, 'linkage': 'ward'}
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        from sklearn.cluster import AgglomerativeClustering
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        data = pipeline_data.get_array()
        
        model = AgglomerativeClustering(
            n_clusters=self.params['n_clusters'],
            linkage=self.params['linkage']
        )
        labels = model.fit_predict(data)
        linkage_matrix = linkage(data, method=self.params['linkage'])
        
        self.result = {
            'labels': labels,
            'linkage_matrix': linkage_matrix,
            'n_clusters': self.params['n_clusters'],
        }
        pipeline_data.set_model_output(self.result, "clustering")
        pipeline_data._log(self.name, f"层次聚类完成，{self.params['n_clusters']}类")
        return pipeline_data


class DecisionTreeAdapter(ModelAdapter):
    """
    决策树分类适配器
    
    数据格式：前n-1列为特征X，最后一列为标签y
    
    参数：
        max_depth: 最大深度，None表示不限制
        test_size: 测试集比例，默认0.2
    """
    def __init__(self):
        super().__init__("决策树分类")
        self.params = {'max_depth': None, 'test_size': 0.2, 'random_state': 42}
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        data = pipeline_data.get_dataframe()
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.params['test_size'], 
            random_state=self.params['random_state']
        )
        
        model = DecisionTreeClassifier(
            max_depth=self.params['max_depth'],
            random_state=self.params['random_state']
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.result = {
            'accuracy': accuracy_score(y_test, y_pred),
            'predictions': y_pred,
            'feature_importance': model.feature_importances_,
            'model': model,
        }
        pipeline_data.set_model_output(self.result, "classification")
        pipeline_data._log(self.name, f"准确率: {self.result['accuracy']:.4f}")
        return pipeline_data


class KNNAdapter(ModelAdapter):
    """
    KNN分类适配器
    
    数据格式：前n-1列为特征X，最后一列为标签y
    
    参数：
        n_neighbors: 邻居数量，默认5
        test_size: 测试集比例，默认0.2
    """
    def __init__(self):
        super().__init__("KNN分类")
        self.params = {'n_neighbors': 5, 'test_size': 0.2, 'random_state': 42}
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        data = pipeline_data.get_dataframe()
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.params['test_size'],
            random_state=self.params['random_state']
        )
        
        model = KNeighborsClassifier(n_neighbors=self.params['n_neighbors'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.result = {
            'accuracy': accuracy_score(y_test, y_pred),
            'predictions': y_pred,
            'model': model,
        }
        pipeline_data.set_model_output(self.result, "classification")
        pipeline_data._log(self.name, f"准确率: {self.result['accuracy']:.4f}")
        return pipeline_data


class NaiveBayesAdapter(ModelAdapter):
    """
    朴素贝叶斯分类适配器
    
    数据格式：前n-1列为特征X，最后一列为标签y
    
    参数：
        method: 'gaussian'/'multinomial'/'bernoulli'，默认'gaussian'
        test_size: 测试集比例
    """
    def __init__(self, method='gaussian'):
        super().__init__(f"朴素贝叶斯({method})")
        self.method = method
        self.params = {'test_size': 0.2, 'random_state': 42}
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        data = pipeline_data.get_dataframe()
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.params['test_size'],
            random_state=self.params['random_state']
        )
        
        if self.method == 'multinomial':
            model = MultinomialNB()
        elif self.method == 'bernoulli':
            model = BernoulliNB()
        else:
            model = GaussianNB()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.result = {
            'accuracy': accuracy_score(y_test, y_pred),
            'predictions': y_pred,
            'model': model,
        }
        pipeline_data.set_model_output(self.result, "classification")
        pipeline_data._log(self.name, f"准确率: {self.result['accuracy']:.4f}")
        return pipeline_data


class RandomForestAdapter(ModelAdapter):
    """
    随机森林分类适配器 - 调用 models.classification.classification_toolkit.RandomForestModel
    
    数据格式：前n-1列为特征X，最后一列为标签y
    
    参数：
        n_estimators: 树的数量，默认100
        max_depth: 最大深度
        test_size: 测试集比例
    """
    def __init__(self):
        super().__init__("随机森林分类")
        self.params = {'n_estimators': 100, 'max_depth': None, 'test_size': 0.2, 'random_state': 42}
        self._rf_model = None
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        data = pipeline_data.get_dataframe()
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # 优先使用库中的 RandomForestModel
        if _RandomForestModel is not None:
            rf = _RandomForestModel(
                n_estimators=self.params['n_estimators'],
                max_depth=self.params['max_depth'],
                verbose=False
            )
            rf.fit(X, y, test_size=self.params['test_size'])
            self._rf_model = rf
            
            self.result = {
                'accuracy': rf.metrics['test']['accuracy'],
                'metrics': rf.metrics,
                'feature_importance': rf.feature_importance,
                'confusion_matrix': rf.confusion_matrix,
                'model': rf.model,
            }
        else:
            # 回退到 sklearn 直接调用
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            X_arr = X.values
            y_arr = y.values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_arr, y_arr, test_size=self.params['test_size'],
                random_state=self.params['random_state']
            )
            
            model = RandomForestClassifier(
                n_estimators=self.params['n_estimators'],
                max_depth=self.params['max_depth'],
                random_state=self.params['random_state']
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            self.result = {
                'accuracy': accuracy_score(y_test, y_pred),
                'predictions': y_pred,
                'feature_importance': model.feature_importances_,
                'model': model,
            }
        
        pipeline_data.set_model_output(self.result, "classification")
        pipeline_data._log(self.name, f"准确率: {self.result['accuracy']:.4f}")
        return pipeline_data


class SVMAdapter(ModelAdapter):
    """
    支持向量机分类适配器
    
    数据格式：前n-1列为特征X，最后一列为标签y
    
    参数：
        kernel: 核函数 'linear'/'rbf'/'poly'/'sigmoid'，默认'rbf'
        C: 正则化参数，默认1.0
        test_size: 测试集比例
    """
    def __init__(self, kernel='rbf'):
        super().__init__(f"SVM({kernel})")
        self.kernel = kernel
        self.params = {'C': 1.0, 'test_size': 0.2, 'random_state': 42}
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import StandardScaler
        
        data = pipeline_data.get_dataframe()
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        # SVM需要标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.params['test_size'],
            random_state=self.params['random_state']
        )
        
        model = SVC(kernel=self.kernel, C=self.params['C'], random_state=self.params['random_state'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.result = {
            'accuracy': accuracy_score(y_test, y_pred),
            'predictions': y_pred,
            'model': model,
            'scaler': scaler,
        }
        pipeline_data.set_model_output(self.result, "classification")
        pipeline_data._log(self.name, f"准确率: {self.result['accuracy']:.4f}")
        return pipeline_data


class XGBoostAdapter(ModelAdapter):
    """
    XGBoost回归/分类适配器
    
    数据格式：前n-1列为特征X，最后一列为目标y
    
    参数：
        task: 'regression'/'classification'，默认'regression'
        n_estimators: 树的数量
        max_depth: 最大深度
        learning_rate: 学习率
    """
    def __init__(self, task='regression'):
        super().__init__(f"XGBoost({task})")
        self.task = task
        self.params = {
            'n_estimators': 100, 'max_depth': 6, 
            'learning_rate': 0.1, 'test_size': 0.2, 'random_state': 42
        }
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        try:
            from xgboost import XGBRegressor, XGBClassifier
        except ImportError:
            raise ImportError("请先安装xgboost: pip install xgboost")
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
        
        data = pipeline_data.get_dataframe()
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.params['test_size'],
            random_state=self.params['random_state']
        )
        
        if self.task == 'classification':
            model = XGBClassifier(
                n_estimators=self.params['n_estimators'],
                max_depth=self.params['max_depth'],
                learning_rate=self.params['learning_rate'],
                random_state=self.params['random_state'],
                use_label_encoder=False, eval_metric='logloss'
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            self.result = {
                'accuracy': accuracy_score(y_test, y_pred),
                'predictions': y_pred,
                'feature_importance': model.feature_importances_,
                'model': model,
            }
            pipeline_data._log(self.name, f"准确率: {self.result['accuracy']:.4f}")
        else:
            model = XGBRegressor(
                n_estimators=self.params['n_estimators'],
                max_depth=self.params['max_depth'],
                learning_rate=self.params['learning_rate'],
                random_state=self.params['random_state']
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            self.result = {
                'r2_score': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'predictions': y_pred,
                'feature_importance': model.feature_importances_,
                'model': model,
            }
            pipeline_data._log(self.name, f"R² = {self.result['r2_score']:.4f}")
        
        pipeline_data.set_model_output(self.result, self.task)
        return pipeline_data


class ARIMAAdapter(ModelAdapter):
    """
    ARIMA时间序列预测适配器
    
    数据格式：一列时间序列数据
    
    参数：
        order: (p,d,q) ARIMA阶数，默认(1,1,1)
        n_predict: 预测步数
    """
    def __init__(self):
        super().__init__("ARIMA预测")
        self.params = {'order': (1, 1, 1), 'n_predict': 5}
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            raise ImportError("请先安装statsmodels: pip install statsmodels")
        
        data = pipeline_data.get_array().flatten()
        n_predict = self.params['n_predict']
        
        model = ARIMA(data, order=self.params['order'])
        fitted = model.fit()
        forecast = fitted.forecast(steps=n_predict)
        
        self.result = {
            'fitted_values': fitted.fittedvalues,
            'predictions': forecast,
            'aic': fitted.aic,
            'bic': fitted.bic,
            'original': data,
        }
        pipeline_data.set_model_output(self.result, "prediction")
        pipeline_data._log(self.name, f"预测了 {n_predict} 个点, AIC={fitted.aic:.2f}")
        return pipeline_data


class ExponentialSmoothingAdapter(ModelAdapter):
    """
    指数平滑预测适配器
    
    数据格式：一列时间序列数据
    
    参数：
        alpha: 平滑系数，0-1之间
        n_predict: 预测步数
    """
    def __init__(self):
        super().__init__("指数平滑预测")
        self.params = {'alpha': 0.3, 'n_predict': 5}
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        data = pipeline_data.get_array().flatten()
        alpha = self.params['alpha']
        n_predict = self.params['n_predict']
        
        # 简单指数平滑
        smoothed = np.zeros(len(data))
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        
        # 预测
        predictions = [smoothed[-1]] * n_predict
        
        self.result = {
            'smoothed': smoothed,
            'predictions': np.array(predictions),
            'alpha': alpha,
            'original': data,
        }
        pipeline_data.set_model_output(self.result, "prediction")
        pipeline_data._log(self.name, f"平滑系数α={alpha}, 预测{n_predict}步")
        return pipeline_data


class MonteCarloAdapter(ModelAdapter):
    """
    蒙特卡洛模拟适配器 - 调用 models.probability.monte_carlo_simulation.MonteCarloSimulator
    
    需要设置模拟函数
    
    参数：
        n_simulations: 模拟次数，默认10000
        confidence: 置信水平，默认0.95
    """
    def __init__(self):
        super().__init__("蒙特卡洛模拟")
        self.params = {'n_simulations': 10000, 'confidence': 0.95}
        self.simulation_func = None
        self._mc_simulator = None
    
    def set_simulation(self, func):
        """设置模拟函数 f() -> float"""
        self.simulation_func = func
        return self
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        if self.simulation_func is None:
            raise ValueError("请先设置模拟函数: model.set_simulation(func)")
        
        n = self.params['n_simulations']
        conf = self.params['confidence']
        
        # 优先使用库中的 MonteCarloSimulator
        if _MonteCarloSimulator is not None:
            simulator = _MonteCarloSimulator(n_simulations=n, verbose=False)
            results = simulator.simulate(self.simulation_func)
            self._mc_simulator = simulator
            
            z = 1.96 if conf == 0.95 else 2.576
            
            self.result = {
                'mean': simulator.mean,
                'std': simulator.std,
                'ci_lower': simulator.ci_lower if hasattr(simulator, 'ci_lower') else simulator.mean - z * simulator.std / np.sqrt(n),
                'ci_upper': simulator.ci_upper if hasattr(simulator, 'ci_upper') else simulator.mean + z * simulator.std / np.sqrt(n),
                'percentile_5': np.percentile(results, 5),
                'percentile_95': np.percentile(results, 95),
                'var_95': np.percentile(results, 5),
                'simulations': results,
            }
        else:
            # 回退到内置实现
            results = np.array([self.simulation_func() for _ in range(n)])
            
            mean = np.mean(results)
            std = np.std(results)
            se = std / np.sqrt(n)
            z = 1.96 if conf == 0.95 else 2.576
            
            self.result = {
                'mean': mean,
                'std': std,
                'ci_lower': mean - z * se,
                'ci_upper': mean + z * se,
                'percentile_5': np.percentile(results, 5),
                'percentile_95': np.percentile(results, 95),
                'var_95': np.percentile(results, 5),
                'simulations': results,
            }
        
        pipeline_data.set_model_output(self.result, "simulation")
        pipeline_data._log(self.name, f"均值={self.result['mean']:.4f}, 标准差={self.result['std']:.4f}")
        return pipeline_data


class PCAAdapter(ModelAdapter):
    """
    PCA降维适配器
    
    数据格式：每行一个样本，每列一个特征
    
    参数：
        n_components: 保留的主成分数量，默认2
    """
    def __init__(self):
        super().__init__("PCA降维")
        self.params = {'n_components': 2}
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        data = pipeline_data.get_array()
        
        # 标准化
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        pca = PCA(n_components=self.params['n_components'])
        transformed = pca.fit_transform(data_scaled)
        
        self.result = {
            'transformed_data': transformed,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_,
            'n_components': self.params['n_components'],
        }
        pipeline_data.set_model_output(self.result, "dimensionality_reduction")
        pipeline_data._log(self.name, f"保留{self.params['n_components']}个主成分, 解释方差{sum(pca.explained_variance_ratio_)*100:.1f}%")
        return pipeline_data


class EntropyWeightAdapter(ModelAdapter):
    """
    熵权法适配器 - 调用 models.evaluation.evaluation_toolkit.EntropyWeightMethod
    
    数据格式：每行一个评价对象，每列一个指标
    
    参数：
        is_benefit: 各指标是否为效益型（越大越好），默认全为True
    """
    def __init__(self):
        super().__init__("熵权法")
        self.params = {'is_benefit': None}
        self._entropy_model = None
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        data = pipeline_data.get_dataframe()
        n_samples, n_features = data.shape
        
        is_benefit = self.params.get('is_benefit')
        
        # 转换 is_benefit 为 indicator_types 格式
        indicator_types = None
        if is_benefit is not None:
            indicator_types = ['positive' if b else 'negative' for b in is_benefit]
        
        # 优先使用库中的 EntropyWeightMethod 类
        if _EntropyWeightMethod is not None:
            entropy_model = _EntropyWeightMethod(verbose=False)
            entropy_model.fit(data, indicator_types=indicator_types)
            self._entropy_model = entropy_model
            
            self.result = {
                'weights': entropy_model.weights,
                'entropy': entropy_model.entropy,
                'difference_coefficient': 1 - entropy_model.entropy,
                'weights_series': entropy_model.get_weights(),
            }
        else:
            # 回退到内置实现
            data_arr = data.values if isinstance(data, pd.DataFrame) else data
            
            if is_benefit is None:
                is_benefit = [True] * n_features
            
            data_pos = data_arr.copy()
            for j in range(n_features):
                if not is_benefit[j]:
                    data_pos[:, j] = data_pos[:, j].max() - data_pos[:, j]
            
            data_norm = data_pos / (data_pos.sum(axis=0) + 1e-10)
            
            entropy = np.zeros(n_features)
            for j in range(n_features):
                p = data_norm[:, j]
                p = p[p > 0]
                entropy[j] = -np.sum(p * np.log(p + 1e-10)) / np.log(n_samples)
            
            d = 1 - entropy
            weights = d / (d.sum() + 1e-10)
            
            self.result = {
                'weights': weights,
                'entropy': entropy,
                'difference_coefficient': d,
            }
        
        pipeline_data.set_model_output(self.result, "evaluation")
        pipeline_data._log(self.name, f"计算了 {n_features} 个指标的权重")
        return pipeline_data


class AHPAdapter(ModelAdapter):
    """
    层次分析法(AHP)适配器
    
    需要输入判断矩阵
    
    参数：
        comparison_matrix: 判断矩阵（需要用户设置）
    """
    def __init__(self):
        super().__init__("层次分析法AHP")
        self.params = {'comparison_matrix': None}
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        # 使用传入的判断矩阵或从数据中获取
        matrix = self.params.get('comparison_matrix')
        if matrix is None:
            matrix = pipeline_data.get_array()
        matrix = np.array(matrix)
        
        n = matrix.shape[0]
        
        # 特征值法求权重
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        max_idx = np.argmax(eigenvalues.real)
        lambda_max = eigenvalues[max_idx].real
        weights = eigenvectors[:, max_idx].real
        weights = weights / weights.sum()  # 归一化
        
        # 一致性检验
        CI = (lambda_max - n) / (n - 1) if n > 1 else 0
        RI_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
        RI = RI_table.get(n, 1.45)
        CR = CI / RI if RI > 0 else 0
        
        self.result = {
            'weights': np.abs(weights),
            'lambda_max': lambda_max,
            'CI': CI,
            'CR': CR,
            'is_consistent': CR < 0.1,
        }
        pipeline_data.set_model_output(self.result, "evaluation")
        status = "通过" if CR < 0.1 else "未通过"
        pipeline_data._log(self.name, f"一致性检验{status}, CR={CR:.4f}")
        return pipeline_data


class SimulatedAnnealingAdapter(ModelAdapter):
    """
    模拟退火优化适配器
    
    需要设置目标函数
    
    参数：
        T0: 初始温度，默认1000
        T_min: 最低温度，默认1e-8
        alpha: 降温系数，默认0.95
        max_iter: 每个温度的迭代次数
        bounds: 搜索范围
        n_dims: 变量维度
    """
    def __init__(self):
        super().__init__("模拟退火优化")
        self.params = {
            'T0': 1000, 'T_min': 1e-8, 'alpha': 0.95,
            'max_iter': 100, 'bounds': (-5, 5), 'n_dims': 2
        }
        self.objective_func = None
    
    def set_objective(self, func):
        """设置目标函数（最小化）"""
        self.objective_func = func
        return self
    
    def run(self, pipeline_data: PipelineData = None) -> PipelineData:
        if self.objective_func is None:
            raise ValueError("请先设置目标函数: model.set_objective(func)")
        
        T = self.params['T0']
        T_min = self.params['T_min']
        alpha = self.params['alpha']
        bounds = self.params['bounds']
        n_dims = self.params['n_dims']
        max_iter = self.params['max_iter']
        
        # 初始解
        x = np.random.uniform(bounds[0], bounds[1], n_dims)
        f = self.objective_func(x)
        best_x, best_f = x.copy(), f
        history = [best_f]
        
        while T > T_min:
            for _ in range(max_iter):
                # 生成新解
                x_new = x + np.random.normal(0, T * 0.01, n_dims)
                x_new = np.clip(x_new, bounds[0], bounds[1])
                f_new = self.objective_func(x_new)
                
                # Metropolis准则
                delta = f_new - f
                if delta < 0 or np.random.rand() < np.exp(-delta / T):
                    x, f = x_new, f_new
                    if f < best_f:
                        best_x, best_f = x.copy(), f
            
            history.append(best_f)
            T *= alpha
        
        self.result = {
            'best_position': best_x,
            'best_value': best_f,
            'convergence_history': history,
        }
        
        if pipeline_data:
            pipeline_data.set_model_output(self.result, "optimization")
            pipeline_data._log(self.name, f"最优值: {best_f:.6f}")
        
        return pipeline_data


class GeneticAlgorithmAdapter(ModelAdapter):
    """
    遗传算法优化适配器 - 调用 models.optimization.optimization_toolkit.GeneticAlgorithm
    
    需要设置目标函数
    
    参数：
        pop_size: 种群大小，默认50
        max_gen: 最大代数，默认100
        crossover_rate: 交叉概率
        mutation_rate: 变异概率
        bounds: 搜索范围
        n_dims: 变量维度
    """
    def __init__(self):
        super().__init__("遗传算法优化")
        self.params = {
            'pop_size': 50, 'max_gen': 100,
            'crossover_rate': 0.8, 'mutation_rate': 0.1,
            'bounds': (-5, 5), 'n_dims': 2
        }
        self.objective_func = None
        self._ga_optimizer = None
    
    def set_objective(self, func):
        """设置目标函数（最小化）"""
        self.objective_func = func
        return self
    
    def run(self, pipeline_data: PipelineData = None) -> PipelineData:
        if self.objective_func is None:
            raise ValueError("请先设置目标函数: model.set_objective(func)")
        
        pop_size = self.params['pop_size']
        max_gen = self.params['max_gen']
        cr = self.params['crossover_rate']
        mr = self.params['mutation_rate']
        bounds = self.params['bounds']
        n_dims = self.params['n_dims']
        
        # 转换 bounds 格式
        if isinstance(bounds, tuple) and len(bounds) == 2:
            bounds_list = [bounds] * n_dims
        else:
            bounds_list = list(bounds)
        
        # 优先使用库中的 GeneticAlgorithm
        if _GA is not None:
            ga = _GA(
                objective_func=self.objective_func,
                bounds=bounds_list,
                n_dims=n_dims,
                pop_size=pop_size,
                max_iter=max_gen,
                crossover_rate=cr,
                mutation_rate=mr,
                verbose=False
            )
            ga.optimize()
            self._ga_optimizer = ga
            
            self.result = {
                'best_position': ga.best_position,
                'best_value': ga.best_value,
                'convergence_history': ga.history,
            }
        else:
            # 回退到内置实现
            lb = bounds[0] if isinstance(bounds, tuple) else min(b[0] for b in bounds_list)
            ub = bounds[1] if isinstance(bounds, tuple) else max(b[1] for b in bounds_list)
            
            pop = np.random.uniform(lb, ub, (pop_size, n_dims))
            fitness = np.array([self.objective_func(ind) for ind in pop])
            best_idx = np.argmin(fitness)
            best_x, best_f = pop[best_idx].copy(), fitness[best_idx]
            history = [best_f]
            
            for gen in range(max_gen):
                fit_inv = 1 / (fitness + 1e-10)
                probs = fit_inv / fit_inv.sum()
                indices = np.random.choice(pop_size, pop_size, p=probs)
                new_pop = pop[indices].copy()
                
                for i in range(0, pop_size-1, 2):
                    if np.random.rand() < cr:
                        point = np.random.randint(1, n_dims)
                        new_pop[i, point:], new_pop[i+1, point:] = \
                            new_pop[i+1, point:].copy(), new_pop[i, point:].copy()
                
                for i in range(pop_size):
                    if np.random.rand() < mr:
                        j = np.random.randint(n_dims)
                        new_pop[i, j] += np.random.normal(0, (ub-lb)*0.1)
                
                new_pop = np.clip(new_pop, lb, ub)
                pop = new_pop
                fitness = np.array([self.objective_func(ind) for ind in pop])
                
                if fitness.min() < best_f:
                    best_idx = np.argmin(fitness)
                    best_x, best_f = pop[best_idx].copy(), fitness[best_idx]
                history.append(best_f)
            
            self.result = {
                'best_position': best_x,
                'best_value': best_f,
                'convergence_history': history,
            }
        
        if pipeline_data:
            pipeline_data.set_model_output(self.result, "optimization")
            pipeline_data._log(self.name, f"最优值: {self.result['best_value']:.6f}")
        
        return pipeline_data


class IntegerProgrammingAdapter(ModelAdapter):
    """
    整数规划适配器
    
    参数：
        c: 目标函数系数（最大化时取负）
        A_ub, b_ub: 不等式约束 Ax <= b
        A_eq, b_eq: 等式约束
        bounds: 变量范围
        integrality: 整数约束 (1=整数, 0=连续)
    """
    def __init__(self):
        super().__init__("整数规划")
        self.params = {
            'c': None, 'A_ub': None, 'b_ub': None,
            'A_eq': None, 'b_eq': None, 'bounds': None,
            'integrality': None  # 1表示整数变量
        }
    
    def run(self, pipeline_data: PipelineData) -> PipelineData:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        c = np.array(self.params['c'])
        
        constraints = []
        if self.params['A_ub'] is not None:
            A_ub = np.array(self.params['A_ub'])
            b_ub = np.array(self.params['b_ub'])
            constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))
        
        if self.params['A_eq'] is not None:
            A_eq = np.array(self.params['A_eq'])
            b_eq = np.array(self.params['b_eq'])
            constraints.append(LinearConstraint(A_eq, b_eq, b_eq))
        
        bounds_param = self.params.get('bounds')
        if bounds_param:
            lb = [b[0] if b[0] is not None else -np.inf for b in bounds_param]
            ub = [b[1] if b[1] is not None else np.inf for b in bounds_param]
            bounds = Bounds(lb, ub)
        else:
            bounds = None
        
        integrality = self.params.get('integrality')
        
        result = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)
        
        self.result = {
            'optimal_value': -result.fun if result.success else None,
            'optimal_solution': result.x,
            'success': result.success,
            'message': result.message,
        }
        pipeline_data.set_model_output(self.result, "optimization")
        pipeline_data._log(self.name, f"最优值: {self.result['optimal_value']}")
        return pipeline_data


# ============================================================
# 【核心】模型工厂 - 一个参数切换所有模型
# ============================================================
"""
【最简单的使用方式】

只需要一行代码切换模型：
    pipeline.set_model(get_model("kmeans"))
    pipeline.set_model(get_model("topsis"))
    pipeline.set_model(get_model("grey"))

支持的模型名称（字符串参数）：

【聚类类】
    "kmeans"           - K-Means聚类
    "hierarchical"     - 层次聚类
    
【分类类】
    "decision_tree"    - 决策树分类
    "knn"              - KNN分类
    "naive_bayes"      - 朴素贝叶斯
    "random_forest"    - 随机森林
    "svm"              - 支持向量机
    "xgboost_cls"      - XGBoost分类

【回归类】
    "linear"           - 线性回归
    "ridge"            - 岭回归
    "lasso"            - Lasso回归
    "polynomial"       - 多项式回归
    "xgboost_reg"      - XGBoost回归

【预测类】
    "grey"             - 灰色预测GM(1,1)
    "arima"            - ARIMA时间序列
    "exp_smoothing"    - 指数平滑

【评价类】
    "topsis"           - TOPSIS综合评价
    "entropy"          - 熵权法
    "ahp"              - 层次分析法AHP

【优化类】
    "dp"               - 动态规划(背包)
    "pso"              - 粒子群优化
    "ga"               - 遗传算法
    "sa"               - 模拟退火
    "linear_prog"      - 线性规划
    "integer_prog"     - 整数规划

【降维类】
    "pca"              - PCA降维

【模拟类】
    "monte_carlo"      - 蒙特卡洛模拟
"""

# 模型注册表
MODEL_REGISTRY = {
    # ===== 聚类 =====
    "kmeans": KMeansAdapter,
    "hierarchical": HierarchicalClusteringAdapter,
    
    # ===== 分类 =====
    "decision_tree": DecisionTreeAdapter,
    "knn": KNNAdapter,
    "naive_bayes": lambda: NaiveBayesAdapter('gaussian'),
    "random_forest": RandomForestAdapter,
    "svm": lambda: SVMAdapter('rbf'),
    "xgboost_cls": lambda: XGBoostAdapter('classification'),
    
    # ===== 回归 =====
    "linear": lambda: RegressionAdapter('linear'),
    "ridge": lambda: RegressionAdapter('ridge'),
    "lasso": lambda: RegressionAdapter('lasso'),
    "polynomial": lambda: RegressionAdapter('polynomial'),
    "xgboost_reg": lambda: XGBoostAdapter('regression'),
    
    # ===== 预测 =====
    "grey": GreyPredictionAdapter,
    "arima": ARIMAAdapter,
    "exp_smoothing": ExponentialSmoothingAdapter,
    
    # ===== 评价 =====
    "topsis": TOPSISAdapter,
    "entropy": EntropyWeightAdapter,
    "ahp": AHPAdapter,
    
    # ===== 优化 =====
    "dp": DynamicProgrammingAdapter,
    "pso": lambda: OptimizationAdapter('pso'),
    "ga": GeneticAlgorithmAdapter,
    "sa": SimulatedAnnealingAdapter,
    "linear_prog": LinearProgrammingAdapter,
    "integer_prog": IntegerProgrammingAdapter,
    
    # ===== 降维 =====
    "pca": PCAAdapter,
    
    # ===== 模拟 =====
    "monte_carlo": MonteCarloAdapter,
}


def get_model(name: str) -> ModelAdapter:
    """
    【一键获取模型】
    
    用法：
        model = get_model("kmeans")      # 获取K-Means聚类
        model = get_model("topsis")      # 获取TOPSIS评价
        model = get_model("grey")        # 获取灰色预测
    
    :param name: 模型名称，见上方支持列表
    :return: 对应的模型适配器实例
    
    完整示例：
        pipeline = ModelValidationPipeline("我的任务")
        pipeline.load_data(my_data, "数据")
        pipeline.set_model(get_model("kmeans"))  # ← 改这里切换模型
        pipeline.configure_model(n_clusters=4)    # ← 改这里调参数
        pipeline.run()
        result = pipeline.get_model_result()
    """
    name = name.lower().strip()
    
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"未知模型: '{name}'\n"
            f"可用模型: {available}\n"
            f"使用方法: get_model('kmeans')"
        )
    
    model_cls = MODEL_REGISTRY[name]
    
    # 如果是lambda函数则调用，否则实例化类
    if callable(model_cls) and not isinstance(model_cls, type):
        return model_cls()
    else:
        return model_cls()


def list_models():
    """列出所有可用模型"""
    print("\n" + "="*60)
    print("📚 可用模型列表")
    print("="*60)
    
    categories = {
        "聚类": ["kmeans", "hierarchical"],
        "分类": ["decision_tree", "knn", "naive_bayes", "random_forest", "svm", "xgboost_cls"],
        "回归": ["linear", "ridge", "lasso", "polynomial", "xgboost_reg"],
        "预测": ["grey", "arima", "exp_smoothing"],
        "评价": ["topsis", "entropy", "ahp"],
        "优化": ["dp", "pso", "ga", "sa", "linear_prog", "integer_prog"],
        "降维": ["pca"],
        "模拟": ["monte_carlo"],
    }
    
    for cat, models in categories.items():
        print(f"\n【{cat}类】")
        for m in models:
            adapter = get_model(m)
            print(f"    '{m}' → {adapter.name}")
    
    print("\n" + "="*60)
    print("用法: pipeline.set_model(get_model('模型名'))")
    print("="*60)


# ============================================================
# 第四部分：可视化步骤 (Visualization Steps)
# ============================================================
"""
【可视化步骤是什么？】
- 根据模型结果自动生成图表
- 可以添加0个、1个或多个
- 运行后可以用 show_figures() 显示，save_figures() 保存

【可用的可视化】（在第6步 add_visualization 时选择）

1. DPTableVisualization()
   用途：动态规划结果可视化
   生成：DP表格热力图 + 物品选择柱状图
   
   示例：pipeline.add_visualization(DPTableVisualization())

2. ConvergenceVisualization()
   用途：优化算法结果可视化
   生成：收敛曲线图
   
   示例：pipeline.add_visualization(ConvergenceVisualization())

3. DataComparisonVisualization()
   用途：对比预处理前后的数据分布
   生成：各列数据的直方图对比
   
   示例：pipeline.add_visualization(DataComparisonVisualization())

4. 自定义可视化 - 继承 VisualizationStep

【如何添加你自己的可视化？】
复制下面的模板：

```python
class MyVisualization(VisualizationStep):
    def __init__(self):
        super().__init__("图表名称")
    
    def plot(self, pipeline_data):
        # 获取数据
        data = pipeline_data.get_dataframe()
        
        # 获取模型结果（如果需要）
        if pipeline_data.model_output:
            result = pipeline_data.model_output['result']
        
        # 创建图表
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # ========== 你的绑图代码 ==========
        self.ax.plot(data.iloc[:, 0], data.iloc[:, 1])
        self.ax.set_title('我的图表')
        # ==================================
        
        return self.fig
```

使用：
    pipeline.add_visualization(MyVisualization())
"""

class VisualizationStep:
    """可视化步骤基类"""
    
    def __init__(self, name="可视化"):
        self.name = name
        self.fig = None
        self.ax = None
    
    def plot(self, pipeline_data: PipelineData):
        """生成图表"""
        raise NotImplementedError
    
    def save(self, filepath):
        """保存图表"""
        if self.fig:
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  📊 图表已保存: {filepath}")


class DPTableVisualization(VisualizationStep):
    """动态规划表格可视化"""
    
    def __init__(self):
        super().__init__("DP表格热力图")
    
    def plot(self, pipeline_data: PipelineData):
        if PlotStyleConfig:
            PlotStyleConfig.setup_style()
        
        output = pipeline_data.model_output
        if not output or output.get('type') != 'dynamic_programming':
            print("⚠️ 无动态规划结果可视化")
            return None
        
        result = output['result']
        dp_table = result['dp_table']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. DP表格热力图
        ax1 = axes[0]
        im = ax1.imshow(dp_table, cmap='Blues', aspect='auto')
        ax1.set_xlabel('背包容量', fontweight='bold')
        ax1.set_ylabel('物品索引', fontweight='bold')
        ax1.set_title('动态规划表格 (DP Table)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='最大价值')
        
        # 标注选中路径
        selected = result['selected_items']
        j = result['capacity']
        for i in range(len(result['weights']), 0, -1):
            if i-1 in selected:
                ax1.plot(j, i, 'r*', markersize=15)
                j -= result['weights'][i-1]
        
        # 2. 物品选择对比
        ax2 = axes[1]
        x = np.arange(len(result['weights']))
        width = 0.35
        
        colors = ['#27AE60' if i in selected else '#CCCCCC' for i in range(len(x))]
        ax2.bar(x - width/2, result['weights'], width, label='重量', color=colors, alpha=0.8)
        ax2.bar(x + width/2, result['values'], width, label='价值', color=colors, edgecolor='black')
        
        ax2.set_xlabel('物品索引', fontweight='bold')
        ax2.set_ylabel('数值', fontweight='bold')
        ax2.set_title('物品选择结果', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.legend()
        
        # 添加选中标记
        for i in selected:
            ax2.annotate('✓', (i, max(result['weights'][i], result['values'][i]) + 0.5),
                        ha='center', fontsize=16, color='green', fontweight='bold')
        
        plt.tight_layout()
        self.fig = fig
        self.ax = axes
        
        # 添加结果文本
        result_text = f"最大价值: {result['max_value']} | 总重量: {result['total_weight']}/{result['capacity']}"
        fig.suptitle(result_text, y=1.02, fontsize=12, fontweight='bold', color='#2E86AB')
        
        return fig


class ConvergenceVisualization(VisualizationStep):
    """收敛曲线可视化"""
    
    def __init__(self):
        super().__init__("收敛曲线")
    
    def plot(self, pipeline_data: PipelineData):
        if PlotStyleConfig:
            PlotStyleConfig.setup_style()
        
        output = pipeline_data.model_output
        if not output or output.get('type') != 'optimization':
            print("⚠️ 无优化结果可视化")
            return None
        
        result = output['result']
        history = result['convergence_history']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(history, color='#2E86AB', linewidth=2.5, label='最优适应度')
        ax.fill_between(range(len(history)), history, alpha=0.2, color='#2E86AB')
        ax.scatter([len(history)-1], [history[-1]], color='#C73E1D', s=100, zorder=5, label=f'最终: {history[-1]:.6f}')
        
        ax.set_xlabel('迭代次数', fontweight='bold')
        ax.set_ylabel('适应度值', fontweight='bold')
        ax.set_title('优化算法收敛曲线', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig = fig
        self.ax = ax
        return fig


class DataComparisonVisualization(VisualizationStep):
    """数据对比可视化（预处理前后）"""
    
    def __init__(self):
        super().__init__("数据对比")
    
    def plot(self, pipeline_data: PipelineData):
        if PlotStyleConfig:
            PlotStyleConfig.setup_style()
        
        raw = pipeline_data.raw_data
        processed = pipeline_data.processed_data
        
        if raw is None or processed is None:
            print("⚠️ 无数据可对比")
            return None
        
        n_cols = min(4, len(raw.select_dtypes(include=[np.number]).columns))
        fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))
        
        numeric_cols = raw.select_dtypes(include=[np.number]).columns[:n_cols]
        
        for i, col in enumerate(numeric_cols):
            # 原始数据
            axes[0, i].hist(raw[col].dropna(), bins=20, color='#A23B72', alpha=0.7, edgecolor='white')
            axes[0, i].set_title(f'{col} (原始)', fontweight='bold')
            axes[0, i].set_xlabel('值')
            axes[0, i].set_ylabel('频数')
            
            # 处理后数据
            axes[1, i].hist(processed[col].dropna(), bins=20, color='#2E86AB', alpha=0.7, edgecolor='white')
            axes[1, i].set_title(f'{col} (处理后)', fontweight='bold')
            axes[1, i].set_xlabel('值')
            axes[1, i].set_ylabel('频数')
        
        fig.suptitle('数据预处理前后对比', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.fig = fig
        self.ax = axes
        return fig


# ============================================================
# 第五部分：主工作流类 (Main Pipeline)
# ============================================================
"""
【ModelValidationPipeline 完整使用指南】

这是工作流的主类，按以下步骤使用：

步骤1: 创建工作流
    pipeline = ModelValidationPipeline("任务名称")
    # "任务名称"会显示在输出和保存的文件名中

步骤2: 加载数据
    pipeline.load_data(data, "数据描述")
    # data 可以是：
    #   - pandas DataFrame（推荐）
    #   - numpy array
    #   - Python list
    #   - dict

步骤3: 添加预处理（可选，可跳过）
    pipeline.add_preprocessing(MissingValueStep('mean'))
    pipeline.add_preprocessing(OutlierRemovalStep('iqr'))
    # 可以添加多个，按顺序执行
    # 不需要预处理就跳过这步

步骤4: 设置模型（必须）
    pipeline.set_model(DynamicProgrammingAdapter())
    pipeline.configure_model(capacity=10)
    # 必须设置一个模型
    # configure_model 设置模型参数

步骤5: 添加可视化（可选，可跳过）
    pipeline.add_visualization(DPTableVisualization())
    # 可以添加多个
    # 不需要可视化就跳过这步

步骤6: 运行
    pipeline.run()
    # 这一步会依次执行：预处理 → 模型 → 可视化

步骤7: 获取结果
    result = pipeline.get_model_result()     # 获取模型结果（字典）
    data = pipeline.get_processed_data()     # 获取处理后的数据
    pipeline.show_results()                  # 打印结果摘要
    pipeline.show_figures()                  # 显示图表
    pipeline.save_figures('./output/')       # 保存图表
"""

class ModelValidationPipeline:
    """
    模型验证工作流 - 串联所有模块
    
    【完整使用示例】
    
    ```python
    from workflow.model_validation_pipeline import *
    
    # === 步骤1: 创建工作流 ===
    pipeline = ModelValidationPipeline("背包问题验证")
    
    # === 步骤2: 加载数据 ===
    items_data = [[2, 6], [2, 3], [6, 5], [5, 4], [4, 6]]
    pipeline.load_data(items_data, "物品列表")
    
    # === 步骤3: 添加预处理（可选）===
    # 如果数据干净可以跳过这步
    pipeline.add_preprocessing(MissingValueStep('mean'))
    pipeline.add_preprocessing(OutlierRemovalStep('iqr'))
    
    # === 步骤4: 设置模型 ===
    pipeline.set_model(DynamicProgrammingAdapter())
    pipeline.configure_model(capacity=10)
    
    # === 步骤5: 添加可视化（可选）===
    pipeline.add_visualization(DPTableVisualization())
    
    # === 步骤6: 运行 ===
    pipeline.run()
    
    # === 步骤7: 获取结果 ===
    result = pipeline.get_model_result()
    print(f"最大价值: {result['max_value']}")
    
    pipeline.show_figures()  # 显示图表
    ```
    """
    
    def __init__(self, name="模型验证工作流", save_dir='./figures'):
        """
        创建工作流
        
        :param name: 工作流名称（会显示在输出中）
        :param save_dir: 图表保存目录
        """
        self.name = name
        self.pipeline_data = None
        self.preprocessing_steps = []
        self.model = None
        self.visualizations = []
        self.save_dir = save_dir
        self.completed = False
        
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print(f"🚀 初始化工作流: {name}")
        print("="*60)
    
    def load_data(self, data, name="输入数据"):
        """
        【步骤2】加载数据
        
        :param data: 数据，支持以下格式：
            - pandas DataFrame（推荐）
            - numpy array  
            - Python list，如 [[1,2], [3,4]]
            - dict，如 {'col1': [1,2], 'col2': [3,4]}
        :param name: 数据描述（显示用）
        
        示例：
            pipeline.load_data(my_dataframe, "实验数据")
            pipeline.load_data([[1,2], [3,4]], "物品列表")
        """
        self.pipeline_data = PipelineData(data, name)
        print(f"✅ 数据已加载: {name}")
        return self
    
    def add_preprocessing(self, step: PreprocessingStep):
        """
        【步骤3】添加预处理步骤（可选，可多次调用）
        
        :param step: 预处理步骤，可选：
            - MissingValueStep('mean')      均值填充缺失值
            - MissingValueStep('median')    中位数填充
            - MissingValueStep('drop')      删除缺失行
            - OutlierRemovalStep('iqr')     IQR异常值处理
            - OutlierRemovalStep('zscore')  Z-score异常值处理
            - NormalizationStep('zscore')   Z-score标准化
            - NormalizationStep('minmax')   Min-Max归一化
        
        示例：
            pipeline.add_preprocessing(MissingValueStep('mean'))
            pipeline.add_preprocessing(OutlierRemovalStep('iqr', 1.5))
        """
        self.preprocessing_steps.append(step)
        print(f"  ➕ 预处理步骤: {step.name}")
        return self
    
    def set_model(self, model: ModelAdapter):
        """
        【步骤4】设置模型（必须）
        
        :param model: 模型适配器，可选：
            - DynamicProgrammingAdapter()  动态规划（背包问题）
            - OptimizationAdapter('pso')   粒子群优化
            - 自定义模型（继承ModelAdapter）
        
        示例：
            pipeline.set_model(DynamicProgrammingAdapter())
            
            # 优化问题需要先设置目标函数
            model = OptimizationAdapter('pso')
            model.set_objective(my_func)
            pipeline.set_model(model)
        """
        self.model = model
        print(f"✅ 模型已设置: {model.name}")
        return self
    
    def configure_model(self, **kwargs):
        """
        【步骤4续】配置模型参数
        
        :param kwargs: 模型参数，取决于使用的模型：
        
        DynamicProgrammingAdapter 参数：
            - capacity: 背包容量（整数）
        
        OptimizationAdapter 参数：
            - bounds: 搜索范围，如 (-5, 5)
            - n_dims: 变量维度
            - max_iter: 最大迭代次数
            - n_particles: 粒子数量（仅PSO）
        
        示例：
            pipeline.configure_model(capacity=15)
            pipeline.configure_model(bounds=(-10, 10), n_dims=3, max_iter=100)
        """
        if self.model:
            self.model.set_params(**kwargs)
            print(f"  ⚙️ 模型参数更新: {kwargs}")
        return self
    
    def add_visualization(self, viz: VisualizationStep):
        """
        【步骤5】添加可视化（可选，可多次调用）
        
        :param viz: 可视化步骤，可选：
            - DPTableVisualization()        动态规划表格热力图
            - ConvergenceVisualization()    优化收敛曲线
            - DataComparisonVisualization() 预处理前后对比
            - 自定义可视化（继承VisualizationStep）
        
        示例：
            pipeline.add_visualization(DPTableVisualization())
        """
        self.visualizations.append(viz)
        print(f"  📊 可视化: {viz.name}")
        return self
    
    def run(self):
        """
        【步骤6】运行工作流
        
        执行顺序：预处理 → 模型 → 可视化
        
        运行后可以：
            - get_model_result()    获取模型结果
            - get_processed_data()  获取处理后的数据
            - show_results()        打印结果摘要
            - show_figures()        显示图表
            - save_figures(path)    保存图表
        """
        print("\n" + "-"*60)
        print("▶️ 开始运行工作流...")
        print("-"*60)
        
        if self.pipeline_data is None:
            raise ValueError("请先加载数据！")
        
        # 1. 预处理
        print("\n📦 Step 1: 数据预处理")
        for step in self.preprocessing_steps:
            self.pipeline_data = step.apply(self.pipeline_data)
            print(f"    ✓ {step.name} 完成")
        
        # 2. 运行模型
        print("\n🔧 Step 2: 运行模型")
        if self.model:
            self.pipeline_data = self.model.run(self.pipeline_data)
            print(f"    ✓ {self.model.name} 完成")
        
        # 3. 生成可视化
        print("\n📊 Step 3: 生成可视化")
        for viz in self.visualizations:
            viz.plot(self.pipeline_data)
            print(f"    ✓ {viz.name} 生成完成")
        
        self.completed = True
        print("\n" + "="*60)
        print("✅ 工作流运行完成!")
        print("="*60)
        return self
    
    def show_results(self):
        """
        【步骤7】显示结果摘要
        打印数据信息和模型结果
        """
        if not self.completed:
            print("⚠️ 请先运行工作流 (run())")
            return
        
        self.pipeline_data.summary()
        
        if self.model and self.model.result:
            print("\n📋 模型结果:")
            for k, v in self.model.result.items():
                if isinstance(v, np.ndarray):
                    print(f"    {k}: shape={v.shape}")
                elif isinstance(v, list) and len(v) > 10:
                    print(f"    {k}: list(len={len(v)})")
                else:
                    print(f"    {k}: {v}")
    
    def show_figures(self):
        """显示所有图表"""
        for viz in self.visualizations:
            if viz.fig:
                plt.figure(viz.fig.number)
                plt.show()
    
    def save_figures(self, directory=None):
        """保存所有图表"""
        save_dir = directory or self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, viz in enumerate(self.visualizations):
            if viz.fig:
                filename = f"{self.name}_{viz.name}_{timestamp}.png"
                filepath = os.path.join(save_dir, filename)
                viz.save(filepath)
    
    def get_model_result(self):
        """
        【步骤7】获取模型结果
        
        :return: dict，模型输出的结果字典
        
        示例：
            result = pipeline.get_model_result()
            print(result['max_value'])  # 动态规划的最大价值
            print(result['best_position'])  # 优化的最优解
        """
        return self.model.result if self.model else None
    
    def get_processed_data(self):
        """
        【步骤7】获取预处理后的数据
        
        :return: pandas DataFrame
        
        示例：
            clean_data = pipeline.get_processed_data()
            clean_data.to_csv('clean_data.csv')
        """
        return self.pipeline_data.get_dataframe() if self.pipeline_data else None


# ============================================================
# 第六部分：快速工厂函数 (Quick Factory Functions)
# ============================================================
"""
【快速函数】
如果你只是想快速验证，不需要自定义配置，可以直接用这些函数：

1. quick_dp_validation(items_data, capacity=10)
   - 快速运行动态规划背包问题
   - items_data: [[重量, 价值], ...] 格式的数据
   - capacity: 背包容量
   
   示例：
       items = [[2, 6], [2, 3], [6, 5], [5, 4], [4, 6]]
       result = quick_dp_validation(items, capacity=10)
       print(f"最大价值: {result['max_value']}")

2. quick_optimization_validation(objective_func, bounds, n_dims, max_iter)
   - 快速运行优化算法求函数最小值
   
   示例：
       def sphere(x): return sum(xi**2 for xi in x)
       result = quick_optimization_validation(sphere, bounds=(-5, 5), n_dims=2)
       print(f"最优解: {result['best_position']}")
"""

def quick_dp_validation(items_data, capacity=10, save_dir='./figures'):
    """
    快速动态规划验证
    
    :param items_data: 物品数据 [[重量, 价值], ...]
    :param capacity: 背包容量
    :param save_dir: 图片保存目录
    
    示例：
        items = [[2, 6], [2, 3], [6, 5], [5, 4], [4, 6]]
        result = quick_dp_validation(items, capacity=10)
    """
    pipeline = ModelValidationPipeline("背包问题", save_dir)
    pipeline.load_data(items_data, "物品数据")
    pipeline.set_model(DynamicProgrammingAdapter())
    pipeline.configure_model(capacity=capacity)
    pipeline.add_visualization(DPTableVisualization())
    pipeline.run()
    pipeline.show_results()
    pipeline.show_figures()
    return pipeline.get_model_result()


def quick_optimization_validation(objective_func, bounds=(-5, 5), n_dims=2, 
                                   max_iter=100, save_dir='./figures'):
    """
    快速优化算法验证
    
    :param objective_func: 目标函数
    :param bounds: 变量范围
    :param n_dims: 维度
    :param max_iter: 迭代次数
    
    示例：
        def sphere(x): return np.sum(x**2)
        result = quick_optimization_validation(sphere, bounds=(-5, 5), n_dims=3)
    """
    pipeline = ModelValidationPipeline("优化验证", save_dir)
    pipeline.pipeline_data = PipelineData(name="优化问题")  # 优化问题可能不需要外部数据
    
    model = OptimizationAdapter('pso')
    model.set_objective(objective_func)
    model.set_params(bounds=bounds, n_dims=n_dims, max_iter=max_iter)
    
    pipeline.set_model(model)
    pipeline.add_visualization(ConvergenceVisualization())
    pipeline.run()
    pipeline.show_results()
    pipeline.show_figures()
    return pipeline.get_model_result()


# ============================================================
# 演示
# ============================================================
"""
【运行演示】
直接运行这个文件可以看到工作流的效果：
    python model_validation_pipeline.py

【常见问题】

Q: 我不需要预处理，可以跳过吗？
A: 可以，不调用 add_preprocessing() 就行。

Q: 我不需要可视化，可以跳过吗？
A: 可以，不调用 add_visualization() 就行。

Q: 我想只做数据清洗，不需要模型？
A: 可以，不调用 set_model()，run() 后用 get_processed_data() 获取数据。

Q: 我的模型需要特殊格式的数据怎么办？
A: 在你的 ModelAdapter.run() 方法中，用 pipeline_data.get_xxx() 获取数据后自己转换。

Q: 怎么对比不同参数的效果？
A: 创建多个 pipeline，每个用不同参数，分别运行。

Q: 怎么保存中间结果？
A: pipeline_data.metadata['my_key'] = value  # 保存
   value = pipeline_data.metadata['my_key']  # 读取
"""

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 模型验证工作流演示 - 一个参数切换模型")
    print("="*70)
    
    # 列出所有可用模型
    list_models()
    
    # ============================================================
    # 示例1: 聚类分析
    # ============================================================
    print("\n\n📌 示例1: K-Means聚类")
    print("-"*50)
    
    # 生成测试数据
    np.random.seed(42)
    cluster_data = np.vstack([
        np.random.randn(30, 2) + [0, 0],
        np.random.randn(30, 2) + [5, 5],
        np.random.randn(30, 2) + [10, 0],
    ])
    
    pipeline = ModelValidationPipeline("聚类分析", save_dir='./modelCode/figures')
    pipeline.load_data(cluster_data, "聚类数据")
    pipeline.set_model(get_model("kmeans"))       # ← 只需改这里！
    pipeline.configure_model(n_clusters=3)
    pipeline.run()
    
    result = pipeline.get_model_result()
    print(f"聚类标签: {result['labels'][:10]}...")
    
    # ============================================================
    # 示例2: TOPSIS评价
    # ============================================================
    print("\n\n📌 示例2: TOPSIS综合评价")
    print("-"*50)
    
    eval_data = pd.DataFrame({
        '质量': [90, 85, 70, 95, 80],
        '价格': [100, 150, 80, 200, 120],  # 成本型
        '交货期': [5, 10, 3, 15, 7],       # 成本型
        '服务': [85, 80, 90, 75, 88],
    })
    
    pipeline2 = ModelValidationPipeline("供应商评价", save_dir='./modelCode/figures')
    pipeline2.load_data(eval_data, "供应商数据")
    pipeline2.set_model(get_model("topsis"))     # ← 只需改这里！
    pipeline2.configure_model(is_benefit=[True, False, False, True])
    pipeline2.run()
    
    result2 = pipeline2.get_model_result()
    print(f"评分: {result2['scores']}")
    print(f"排名: {result2['ranking']}")
    
    # ============================================================
    # 示例3: 灰色预测
    # ============================================================
    print("\n\n📌 示例3: 灰色预测GM(1,1)")
    print("-"*50)
    
    time_series = pd.DataFrame({'值': [100, 112, 125, 138, 150, 165]})
    
    pipeline3 = ModelValidationPipeline("销量预测", save_dir='./modelCode/figures')
    pipeline3.load_data(time_series, "历史销量")
    pipeline3.set_model(get_model("grey"))        # ← 只需改这里！
    pipeline3.configure_model(n_predict=3)
    pipeline3.run()
    
    result3 = pipeline3.get_model_result()
    print(f"预测值: {result3['predictions']}")
    
    # ============================================================
    # 示例4: 动态规划背包问题
    # ============================================================
    print("\n\n📌 示例4: 动态规划背包问题")
    print("-"*50)
    
    items = pd.DataFrame({'重量': [2, 2, 6, 5, 4], '价值': [6, 3, 5, 4, 6]})
    
    pipeline4 = ModelValidationPipeline("背包问题", save_dir='./modelCode/figures')
    pipeline4.load_data(items, "物品列表")
    pipeline4.set_model(get_model("dp"))          # ← 只需改这里！
    pipeline4.configure_model(capacity=10)
    pipeline4.add_visualization(DPTableVisualization())
    pipeline4.run()
    
    result4 = pipeline4.get_model_result()
    print(f"最大价值: {result4['max_value']}")
    print(f"选中物品: {result4['selected_items']}")
    
    # ============================================================
    # 示例5: 分类（随机森林）
    # ============================================================
    print("\n\n📌 示例5: 随机森林分类")
    print("-"*50)
    
    from sklearn.datasets import load_iris
    iris = load_iris()
    clf_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    clf_data['label'] = iris.target
    
    pipeline5 = ModelValidationPipeline("鸢尾花分类", save_dir='./modelCode/figures')
    pipeline5.load_data(clf_data, "鸢尾花数据")
    pipeline5.set_model(get_model("random_forest"))  # ← 只需改这里！
    pipeline5.configure_model(n_estimators=50)
    pipeline5.run()
    
    result5 = pipeline5.get_model_result()
    print(f"准确率: {result5['accuracy']:.4f}")
    
    print("\n\n" + "="*70)
    print("✅ 演示完成！")
    print("="*70)
    print("""
【使用方法总结】

    pipeline.set_model(get_model("模型名"))
    
    把 "模型名" 换成你需要的模型即可：
    
    聚类: "kmeans", "hierarchical"
    分类: "decision_tree", "knn", "naive_bayes", "random_forest", "svm"
    回归: "linear", "ridge", "lasso", "polynomial"
    预测: "grey", "arima", "exp_smoothing"
    评价: "topsis", "entropy", "ahp"
    优化: "dp", "pso", "ga", "sa", "linear_prog"
    降维: "pca"
    模拟: "monte_carlo"
    """)
    
    plt.show()
