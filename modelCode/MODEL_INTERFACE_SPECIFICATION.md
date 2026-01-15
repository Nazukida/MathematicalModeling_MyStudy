# 模型接口规范文档
## Model Interface Specification for MCM/ICM

> 本文档为 `modelCode/models` 目录下所有模型提供标准化的接口描述，便于后续创建工作流适配器。

---

## 目录

1. [分类模型 (Classification)](#1-分类模型-classification)
2. [聚类模型 (Clustering)](#2-聚类模型-clustering)
3. [相关性分析 (Correlation)](#3-相关性分析-correlation)
4. [动力学模型 (Dynamics)](#4-动力学模型-dynamics)
5. [评价模型 (Evaluation)](#5-评价模型-evaluation)
6. [图算法 (Graph Algorithms)](#6-图算法-graph-algorithms)
7. [插值方法 (Interpolation)](#7-插值方法-interpolation)
8. [神经网络 (Neural Networks)](#8-神经网络-neural-networks)
9. [优化算法 (Optimization)](#9-优化算法-optimization)
10. [预测模型 (Prediction)](#10-预测模型-prediction)
11. [概率模型 (Probability)](#11-概率模型-probability)
12. [回归分析 (Regression)](#12-回归分析-regression)

---

## 1. 分类模型 (Classification)

### 1.1 决策树分类 (Decision Tree Classification)

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/classification/decision_tree_classification.py` |
| **模型类型** | 脚本示例（使用sklearn） |
| **主要依赖** | `sklearn.tree.DecisionTreeClassifier` |

**核心函数/类：**
```python
from sklearn.tree import DecisionTreeClassifier

# 初始化参数
dt_model = DecisionTreeClassifier(
    max_depth=5,           # 最大树深度
    criterion='gini',      # 分裂准则：'gini' 或 'entropy'
    random_state=42        # 随机种子
)

# 主要方法
dt_model.fit(X_train, y_train)           # 训练模型
y_pred = dt_model.predict(X_test)        # 预测
importance = dt_model.feature_importances_  # 特征重要性
```

**返回值结构：**
- `predict()`: ndarray, shape=(n_samples,) - 预测的类别标签
- `feature_importances_`: ndarray, shape=(n_features,) - 各特征的重要性分数

---

### 1.2 KNN分类 (K-Nearest Neighbors)

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/classification/knn_classification.py` |
| **模型类型** | 脚本示例（使用sklearn） |
| **主要依赖** | `sklearn.neighbors.KNeighborsClassifier` |

**核心函数/类：**
```python
from sklearn.neighbors import KNeighborsClassifier

# 初始化参数
knn = KNeighborsClassifier(
    n_neighbors=5,         # K值（近邻数量）
    metric='euclidean'     # 距离度量：'euclidean', 'manhattan', 'minkowski'
)

# 主要方法
knn.fit(X_train_scaled, y_train)         # 训练（需要标准化）
y_pred = knn.predict(X_test_scaled)      # 预测
```

**返回值结构：**
- `predict()`: ndarray, shape=(n_samples,) - 预测的类别标签

---

### 1.3 朴素贝叶斯分类 (Naive Bayes)

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/classification/naive_bayes_classification.py` |
| **模型类型** | 脚本示例（使用sklearn） |
| **主要依赖** | `sklearn.naive_bayes.GaussianNB`, `MultinomialNB` |

**核心函数/类：**
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# 高斯朴素贝叶斯（连续特征）
gnb = GaussianNB()

# 多项式朴素贝叶斯（离散/文本特征）
mnb = MultinomialNB()

# 主要方法
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)     # 预测概率
```

**返回值结构：**
- `predict()`: ndarray, shape=(n_samples,) - 预测的类别标签
- `predict_proba()`: ndarray, shape=(n_samples, n_classes) - 各类别的概率

---

## 2. 聚类模型 (Clustering)

### 2.1 层次聚类 (Hierarchical Clustering)

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/clustering/hierarchical_clustering.py` |
| **模型类型** | 脚本示例（使用scipy） |
| **主要依赖** | `scipy.cluster.hierarchy` |

**核心函数：**
```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# 执行层次聚类
Z = linkage(
    X_scaled,              # 标准化后的数据
    method='ward',         # 链接方法：'ward', 'average', 'complete', 'single'
    metric='euclidean'     # 距离度量
)

# 获取聚类结果
y_pred = fcluster(Z, k, criterion='maxclust')  # 划分为k个簇

# 可视化聚类树
dendrogram(Z, truncate_mode='lastp', p=20)
```

**返回值结构：**
- `linkage()`: ndarray, shape=(n-1, 4) - 聚类合并信息矩阵
- `fcluster()`: ndarray, shape=(n,) - 聚类标签

---

### 2.2 K-Means聚类

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/clustering/kmeans_clustering.py` |
| **模型类型** | 脚本示例（使用sklearn） |
| **主要依赖** | `sklearn.cluster.KMeans` |

**核心函数/类：**
```python
from sklearn.cluster import KMeans

# 初始化参数
kmeans = KMeans(
    n_clusters=3,          # 聚类数量
    init='k-means++',      # 初始化方法
    n_init=10,             # 不同初始化的运行次数
    max_iter=300,          # 最大迭代次数
    random_state=42
)

# 主要方法
y_pred = kmeans.fit_predict(X_scaled)    # 训练并预测
centers = kmeans.cluster_centers_         # 聚类中心
inertia = kmeans.inertia_                 # 簇内平方和
```

**返回值结构：**
- `fit_predict()`: ndarray, shape=(n_samples,) - 聚类标签
- `cluster_centers_`: ndarray, shape=(n_clusters, n_features) - 聚类中心坐标
- `inertia_`: float - 所有样本到其最近聚类中心的距离平方和

---

### 2.3 SOM自组织映射

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/clustering/som_clustering.py` |
| **模型类型** | 脚本示例（使用minisom） |
| **主要依赖** | `minisom.MiniSom` |

**核心函数/类：**
```python
from minisom import MiniSom

# 初始化参数
som = MiniSom(
    x=10, y=10,            # 神经元网格大小
    input_len=n_features,  # 输入维度
    sigma=1.0,             # 初始邻域半径
    learning_rate=0.5,     # 初始学习率
    neighborhood_function='gaussian',
    random_seed=42
)

# 主要方法
som.random_weights_init(X_scaled)        # 初始化权重
som.train_random(X_scaled, 1000)         # 训练
bmus = [som.winner(x) for x in X_scaled] # 获取最佳匹配单元
u_matrix = som.distance_map()            # U矩阵
weights = som.get_weights()              # 权重矩阵
```

**返回值结构：**
- `winner()`: tuple (x, y) - 最佳匹配单元的坐标
- `distance_map()`: ndarray, shape=(x, y) - U矩阵
- `get_weights()`: ndarray, shape=(x, y, input_len) - 权重向量

---

## 3. 相关性分析 (Correlation)

### 3.1 卡方检验 (Chi-Square Test)

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/correlation/chi_square_test.py` |
| **模型类型** | 脚本示例（使用scipy） |
| **适用场景** | 两个离散变量的相关性检验 |

**核心函数：**
```python
from scipy.stats import chi2_contingency
import pandas as pd

# 创建列联表
contingency_table = pd.crosstab(data['var1'], data['var2'])

# 执行卡方检验
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
```

**返回值结构：**
- `chi2`: float - 卡方统计量
- `p_value`: float - P值
- `dof`: int - 自由度
- `expected`: ndarray - 期望频数表

---

### 3.2 连续变量相关性分析

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/correlation/continuous_correlation_analysis.py` |
| **模型类型** | 脚本示例（使用scipy） |
| **适用场景** | 协方差、Pearson、Spearman相关系数 |

**核心函数：**
```python
from scipy.stats import pearsonr, spearmanr
import numpy as np

# Pearson相关系数（线性相关）
pearson_corr, pearson_p = pearsonr(x, y)

# Spearman相关系数（单调相关）
spearman_corr, spearman_p = spearmanr(x, y)

# 协方差
covariance = np.cov(x, y)[0, 1]
```

**返回值结构：**
- 相关系数: float, 范围[-1, 1]
- p_value: float - 显著性检验P值

---

### 3.3 Kendall相关系数

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/correlation/kendall_correlation.py` |
| **模型类型** | 脚本示例（使用scipy） |
| **适用场景** | 小样本、存在平局的单调相关分析 |

**核心函数：**
```python
from scipy.stats import kendalltau

# Kendall τ相关系数
kendall_tau, kendall_p = kendalltau(x, y)
```

**返回值结构：**
- `kendall_tau`: float, 范围[-1, 1] - Kendall相关系数
- `kendall_p`: float - P值

---

## 4. 动力学模型 (Dynamics)

### 4.1 GLV生态系统模型

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/dynamics/glv_ecosystem_model.py` |
| **模型类型** | 类封装 |
| **适用场景** | 生态系统物种动态模拟、种间相互作用 |

**核心类：**
```python
class GLVEcosystemModel:
    def __init__(self, n_species=5, seed=42):
        """
        参数:
            n_species: int - 物种数量
            seed: int - 随机种子
        """
        self.r = ...        # 固有增长率 ndarray
        self.K = ...        # 环境容纳量 ndarray
        self.alpha_base = ...  # 基础相互作用矩阵
    
    def alpha_ij(self, W, i, j):
        """计算水资源依赖的相互作用系数"""
        pass
    
    def water_dynamics(self, t, W_base=0.6, A=0.2, omega=0.5, ...):
        """水资源动态方程"""
        pass
    
    def simulate(self, t_span, B0=None, drought_events=None):
        """
        运行模拟
        参数:
            t_span: array - 时间点数组
            B0: array - 初始生物量
            drought_events: list - 干旱事件 [(start, end, intensity), ...]
        返回:
            solution: 模拟结果
        """
        pass
```

---

### 4.2 兰彻斯特战争模型

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/dynamics/war_model.py` |
| **模型类型** | 函数式脚本 |
| **适用场景** | 战争兵力动态模拟 |

**核心函数：**
```python
from scipy.integrate import solve_ivp

def lanchester_model(t, state, a, b):
    """
    兰彻斯特战争模型微分方程
    
    参数:
        t: 时间
        state: [x, y] 双方兵力
        a: 乙方战斗力系数
        b: 甲方战斗力系数
    返回:
        [dxdt, dydt]: 兵力变化率
    """
    x, y = state
    dxdt = -b * y
    dydt = -a * x
    return [dxdt, dydt]

# 求解
solution = solve_ivp(lanchester_model, t_span, [x0, y0], args=(a, b))
```

---

## 5. 评价模型 (Evaluation)

### 5.1 AHP层次分析法

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/evaluation/evaluation_methods.py` |
| **类名** | `AHP` |

**接口规范：**
```python
class AHP:
    RI_TABLE = [0, 0, 0.58, 0.90, ...]  # 随机一致性指标
    
    def __init__(self, n_criteria=None, criteria_names=None):
        """
        参数:
            n_criteria: int - 评价指标数量
            criteria_names: list - 指标名称列表
        """
    
    def set_matrix(self, matrix):
        """设置判断矩阵（1-9标度法）"""
    
    def calculate_weights(self, method='eigenvalue'):
        """
        计算权重向量
        参数:
            method: 'eigenvalue'/'geometric'/'arithmetic'
        返回:
            weights: ndarray - 归一化权重向量
        """
    
    def get_report(self):
        """
        返回:
            dict: {矩阵阶数, 最大特征值, CI, CR, 是否通过一致性检验, 权重向量}
        """
    
    def plot_weights(self, figsize=(10,6), save_path=None):
        """可视化权重分布"""
```

---

### 5.2 DEA数据包络分析

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/evaluation/evaluation_methods.py` |
| **类名** | `DEA` |

**接口规范：**
```python
class DEA:
    def __init__(self, orientation='input'):
        """
        参数:
            orientation: 'input'(投入导向) / 'output'(产出导向)
        """
    
    def fit(self, inputs, outputs, dmu_names=None):
        """
        拟合DEA-CCR模型
        参数:
            inputs: ndarray (n_dmu × n_inputs) - 投入矩阵
            outputs: ndarray (n_dmu × n_outputs) - 产出矩阵
            dmu_names: list - 决策单元名称
        """
    
    def get_results(self):
        """
        返回:
            DataFrame: {决策单元, 效率值, DEA有效}
        """
    
    def plot_efficiency(self, figsize=(12,5), save_path=None):
        """可视化效率分析结果"""
```

---

### 5.3 模糊综合评价

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/evaluation/evaluation_methods.py` |
| **类名** | `FuzzyComprehensiveEvaluation` |

**接口规范：**
```python
class FuzzyComprehensiveEvaluation:
    def __init__(self, n_levels=3, level_names=None):
        """
        参数:
            n_levels: int - 评价等级数
            level_names: list - 等级名称（从高到低）
        """
    
    def fit(self, data, weights, criteria=None, sample_names=None):
        """
        参数:
            data: ndarray (n_samples × n_criteria) - 评价数据
            weights: ndarray - 指标权重向量
            criteria: ndarray - 评价等级标准矩阵
            sample_names: list - 样本名称
        """
    
    def get_results(self):
        """
        返回:
            DataFrame: {样本, 综合得分, 排名, 最优等级}
        """
```

---

### 5.4 灰色关联分析

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/evaluation/evaluation_methods.py` |
| **类名** | `GreyRelationalAnalysis` |

**接口规范：**
```python
class GreyRelationalAnalysis:
    def __init__(self, rho=0.5):
        """
        参数:
            rho: float - 分辨系数，通常取0.5
        """
    
    def fit(self, data, reference=None, factor_names=None, sample_names=None):
        """
        参数:
            data: ndarray (n_samples × n_factors) - 比较序列矩阵
            reference: ndarray - 参考序列（None则取最优值）
        """
    
    def get_results(self):
        """
        返回:
            DataFrame: {样本, 灰色关联度, 排名}
        """
```

---

### 5.5 熵权法+TOPSIS

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/evaluation/evaluation_toolkit.py` |
| **类名** | `EntropyWeightMethod`, `TOPSIS` |

**熵权法接口：**
```python
class EntropyWeightMethod:
    def __init__(self, verbose=True):
        pass
    
    def fit(self, data, negative_indices=None, indicator_types=None):
        """
        参数:
            data: DataFrame或ndarray - 评价数据
            negative_indices: list - 负向指标的列索引
            indicator_types: list - ['positive', 'negative', ...]
        返回:
            self (weights属性包含计算的权重)
        """
```

---

## 6. 图算法 (Graph Algorithms)

### 6.1 Dijkstra最短路径算法

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/graph_algorithms/dijkstra_algorithm.py` |
| **模型类型** | 函数式脚本 |
| **适用场景** | 单源最短路径（非负边权） |

**核心函数：**
```python
def dijkstra(graph, start):
    """
    Dijkstra算法求解单源最短路径
    
    参数:
        graph: list[list] - 邻接矩阵（0表示无直接连接）
        start: int - 起始节点索引
    
    返回:
        distances: list - 起点到各节点的最短距离
        paths: list[list] - 起点到各节点的最短路径
    """
    pass
```

---

### 6.2 Floyd最短路径算法

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/graph_algorithms/floyd_algorithm.py` |
| **模型类型** | 函数式脚本 |
| **适用场景** | 全源最短路径 |

**核心函数：**
```python
def floyd(graph):
    """
    Floyd算法求解所有节点对之间的最短路径
    
    参数:
        graph: list[list] - 邻接矩阵（np.inf表示无直接连接）
    
    返回:
        dist: list[list] - 最短距离矩阵
        path: list[list] - 路径矩阵（用于重建路径）
    """

def get_path(path, start, end):
    """根据路径矩阵获取具体路径"""
```

---

## 7. 插值方法 (Interpolation)

### 7.1 拉格朗日插值

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/interpolation/lagrange_interpolation.py` |
| **模型类型** | 脚本示例 |

**核心算法：**
```python
def lagrange_interpolate(x_sample, y_sample, x_interp):
    """
    拉格朗日插值
    
    参数:
        x_sample: ndarray - 已知采样点x坐标
        y_sample: ndarray - 已知采样点y值
        x_interp: ndarray - 需要插值的x坐标
    
    返回:
        y_interp: ndarray - 插值结果
    """
    n = len(x_sample)
    y_interp = np.zeros_like(x_interp)
    
    for k in range(len(x_interp)):
        x = x_interp[k]
        y = 0.0
        for i in range(n):
            L = 1.0
            for j in range(n):
                if j != i:
                    L *= (x - x_sample[j]) / (x_sample[i] - x_sample[j])
            y += y_sample[i] * L
        y_interp[k] = y
    
    return y_interp
```

---

## 8. 神经网络 (Neural Networks)

### 8.1 BP神经网络分类

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/neural_networks/bp_classification.py` |
| **模型类型** | 脚本示例（使用sklearn） |

**核心函数/类：**
```python
from sklearn.neural_network import MLPClassifier

# 初始化参数
bp_model = MLPClassifier(
    hidden_layer_sizes=(32,),  # 隐藏层结构，如(32,16)表示两个隐藏层
    activation='relu',          # 激活函数：'relu', 'tanh', 'logistic'
    solver='adam',              # 优化器：'adam', 'sgd', 'lbfgs'
    learning_rate_init=0.001,   # 初始学习率
    max_iter=300,               # 最大迭代次数
    random_state=42
)

# 主要方法
bp_model.fit(X_train_scaled, y_train)
y_pred = bp_model.predict(X_test_scaled)
y_proba = bp_model.predict_proba(X_test_scaled)
loss_curve = bp_model.loss_curve_    # 训练损失曲线
```

---

### 8.2 BP神经网络预测（回归）

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/neural_networks/bp_prediction.py` |
| **模型类型** | 脚本示例（使用sklearn） |

**核心函数/类：**
```python
from sklearn.neural_network import MLPRegressor

# 初始化参数
model = MLPRegressor(
    hidden_layer_sizes=(10,),  # 隐藏层结构
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# 主要方法
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## 9. 优化算法 (Optimization)

### 9.1 线性规划

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/optimization/linear_programming.py` |
| **模型类型** | 脚本示例（使用PuLP） |

**核心函数/类：**
```python
import pulp as pl

# 创建问题
prob = pl.LpProblem("问题名称", pl.LpMaximize)  # 或 pl.LpMinimize

# 定义变量
x = pl.LpVariable("变量名", lowBound=0)

# 目标函数
prob += 目标表达式, "目标名称"

# 约束条件
prob += 约束表达式, "约束名称"

# 求解
prob.solve()

# 获取结果
status = pl.LpStatus[prob.status]
optimal_value = pl.value(prob.objective)
var_value = pl.value(x)
```

---

### 9.2 整数规划

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/optimization/integer_programming.py` |
| **模型类型** | 脚本示例（使用PuLP） |

**与线性规划的区别：**
```python
# 0-1整数变量
x = pl.LpVariable.dicts("变量", 索引集, cat=pl.LpBinary)

# 整数变量
x = pl.LpVariable("变量", cat=pl.LpInteger)
```

---

### 9.3 非线性规划

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/optimization/nonlinear_programming.py` |
| **模型类型** | 脚本示例（使用scipy） |

**核心函数：**
```python
from scipy.optimize import minimize

def objective(x):
    """目标函数（最小化）"""
    return f(x)

def constraint1(x):
    """不等式约束 g(x) >= 0"""
    return g(x)

# 求解
solution = minimize(
    objective,
    x0,                        # 初始猜测
    method='SLSQP',            # 方法：'SLSQP', 'trust-constr', 'COBYLA'
    bounds=[(0, None), ...],   # 变量边界
    constraints=[
        {'type': 'ineq', 'fun': constraint1},
        {'type': 'eq', 'fun': constraint2}
    ]
)

optimal_x = solution.x
optimal_value = -solution.fun  # 如果目标是最大化
```

---

### 9.4 0-1规划

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/optimization/zero_one_programming.py` |
| **模型类型** | 脚本示例（使用PuLP） |
| **适用场景** | 选址问题、投资组合等二元决策 |

---

### 9.5 动态规划

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/optimization/dynamic_programming.py` |
| **模型类型** | 脚本示例 |
| **适用场景** | 背包问题、最短路径等 |

**背包问题示例：**
```python
# dp[i][j] 表示考虑前i个物品，容量为j时的最大价值
dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

for i in range(1, n + 1):
    for j in range(capacity + 1):
        if weights[i-1] <= j:
            dp[i][j] = max(values[i-1] + dp[i-1][j-weights[i-1]], dp[i-1][j])
        else:
            dp[i][j] = dp[i-1][j]

max_value = dp[n][capacity]
```

---

### 9.6 模拟退火算法

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/optimization/simulated_annealing.py` |
| **模型类型** | 函数式脚本 |
| **适用场景** | 全局优化、组合优化 |

**核心函数：**
```python
def simulated_annealing(objective_func, bounds, initial_temp, 
                        cooling_rate, max_iter, step_size):
    """
    参数:
        objective_func: callable - 目标函数
        bounds: list[(low, high)] - 变量边界
        initial_temp: float - 初始温度
        cooling_rate: float - 冷却速率 (0-1)
        max_iter: int - 最大迭代次数
        step_size: float - 邻域搜索步长
    
    返回:
        best_solution: ndarray - 最优解
        best_value: float - 最优值
        history: list - 优化历史
    """
```

---

### 9.7 最速下降法

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/optimization/steepest_descent.py` |
| **模型类型** | 函数式脚本 |
| **适用场景** | 无约束凸优化 |

**核心函数：**
```python
def steepest_descent(initial_point, learning_rate, max_iterations, tolerance):
    """
    参数:
        initial_point: list - 初始点
        learning_rate: float - 学习率（步长）
        max_iterations: int - 最大迭代次数
        tolerance: float - 收敛容差
    
    返回:
        optimal_point: ndarray - 最优解
        path: list - 迭代路径
        iterations: int - 实际迭代次数
    """
```

---

### 9.8 NSGA-II多目标优化

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/optimization/nsga2_multi_objective.py` |
| **类名** | `NSGAII` |
| **适用场景** | 多目标优化、Pareto前沿求解 |

**接口规范：**
```python
class NSGAII:
    def __init__(self, objectives, n_var=2, bounds=(0, 10),
                 pop_size=50, n_generations=100, verbose=True):
        """
        参数:
            objectives: list[callable] - 目标函数列表（均为最小化）
            n_var: int - 决策变量维度
            bounds: tuple或list - 变量范围
            pop_size: int - 种群大小
            n_generations: int - 迭代次数
        """
    
    def optimize(self):
        """
        返回:
            pareto_solutions: ndarray - Pareto最优解集
            pareto_front: ndarray - Pareto前沿（目标值）
        """
    
    def plot_pareto_front(self, save_path=None):
        """可视化Pareto前沿"""
```

---

## 10. 预测模型 (Prediction)

### 10.1 灰色预测 GM(1,1)

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/prediction/grey_prediction.py` |
| **模型类型** | 脚本示例 |
| **适用场景** | 小样本（4-10个数据）、信息不完全 |

**核心算法：**
```python
# 输入数据
x0 = np.array([...], dtype=np.float64)  # 原始数据序列

# 累加生成
x1 = np.cumsum(x0)

# 构造数据矩阵并求解参数
# a: 发展系数, b: 灰作用量

# 预测公式
def predict(x0, a, b, k):
    x1_hat = (x0[0] - b/a) * np.exp(-a * k) + b/a
    # 累减还原...

# 模型检验
C = s2 / s1  # 后验差比 (C < 0.35 为好)
P = ...      # 小误差概率 (P > 0.95 为好)
```

---

### 10.2 ARMA时间序列预测

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/prediction/arma_prediction.py` |
| **模型类型** | 脚本示例（使用statsmodels） |
| **适用场景** | 平稳时间序列预测 |

**核心函数/类：**
```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# 平稳性检验
result = adfuller(data)
is_stationary = result[1] < 0.05

# 拟合ARIMA(p, d, q)模型
model = ARIMA(data, order=(p, d, q))
model_fit = model.fit()

# 预测
forecast = model_fit.get_forecast(steps=n)
pred_mean = forecast.predicted_mean
conf_int = forecast.conf_int()
```

---

### 10.3 Logistic增长预测

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/prediction/logistic_prediction.py` |
| **模型类型** | 脚本示例（使用scipy） |
| **适用场景** | 饱和增长（人口、产品扩散等） |

**核心函数：**
```python
from scipy.optimize import curve_fit

def logistic_func(t, K, r, y0):
    """
    Logistic增长模型
    y(t) = K / (1 + (K/y0 - 1) * exp(-r*t))
    
    参数:
        K: 环境承载力（饱和值）
        r: 增长率
        y0: 初始值
    """
    return K / (1 + (K / y0 - 1) * np.exp(-r * t))

# 拟合
params, covariance = curve_fit(logistic_func, t, y, p0=initial_guess)
K, r, y0 = params
```

---

### 10.4 马尔可夫预测

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/prediction/markov_prediction.py` |
| **模型类型** | 脚本示例 |
| **适用场景** | 状态序列预测（无后效性） |

**核心算法：**
```python
# 统计状态转移次数
transition_counts = np.zeros((n_states, n_states))
for i in range(n - 1):
    transition_counts[states[i], states[i+1]] += 1

# 计算转移概率矩阵
transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)

# 预测未来状态分布
current_dist = initial_distribution
for _ in range(forecast_steps):
    current_dist = np.dot(current_dist, transition_matrix)

# 稳态分布
steady_state = ... # 多次迭代直到收敛
```

---

### 10.5 Prophet时间序列预测

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/prediction/prophet_forecast.py` |
| **类名** | `ProphetPredictor` |
| **适用场景** | 多季节性、假日效应、趋势突变 |

**接口规范：**
```python
class ProphetPredictor:
    def __init__(self, yearly_seasonality=True, weekly_seasonality=True,
                 daily_seasonality=False, changepoint_prior_scale=0.05):
        """
        参数:
            yearly_seasonality: bool - 年季节性
            weekly_seasonality: bool - 周季节性
            daily_seasonality: bool - 日季节性
            changepoint_prior_scale: float - 趋势变化灵活度 (0.001-0.5)
        """
    
    def fit(self, data, ds_col='ds', y_col='y'):
        """
        训练模型
        参数:
            data: DataFrame - 需包含日期列和目标列
        """
    
    def predict(self, periods=30, freq='D'):
        """
        预测未来
        返回:
            forecast: DataFrame - 包含yhat, yhat_lower, yhat_upper
        """
    
    def evaluate(self):
        """返回 {'MAE', 'MAPE', 'RMSE'}"""
    
    def plot_forecast(self, save_path=None): pass
    def plot_components(self, save_path=None): pass
```

---

### 10.6 XGBoost回归预测

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/prediction/xgboost_regression.py` |
| **类名** | `XGBPredictor` |
| **适用场景** | 高精度回归、非线性关系、特征重要性 |

**接口规范：**
```python
class XGBPredictor:
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, random_state=42):
        """
        参数:
            n_estimators: int - 迭代次数
            max_depth: int - 最大深度 (3-10)
            learning_rate: float - 学习率 (0.01-0.3)
            subsample: float - 样本采样比例
            colsample_bytree: float - 特征采样比例
        """
    
    def fit(self, X, y, test_size=0.2, early_stopping=False):
        """训练并评估"""
    
    def predict(self, X):
        """返回预测值"""
    
    def cross_validate(self, X, y, cv=5):
        """交叉验证"""
    
    def plot_feature_importance(self, save_path=None): pass
    def plot_prediction(self, save_path=None): pass
```

---

### 10.7 二次指数平滑预测

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/prediction/quadratic_exponential_smoothing.py` |
| **模型类型** | 脚本示例 |
| **适用场景** | 线性趋势数据 |

**核心算法：**
```python
alpha = 0.3  # 平滑系数

# 一次指数平滑
s1[i] = alpha * y[i] + (1 - alpha) * s1[i-1]

# 二次指数平滑
s2[i] = alpha * s1[i] + (1 - alpha) * s2[i-1]

# 预测系数
a = 2 * s1[-1] - s2[-1]
b = (alpha / (1 - alpha)) * (s1[-1] - s2[-1])

# 预测公式
y_hat(n+k) = a + b * k
```

---

### 10.8 季节指数预测

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/prediction/seasonal_index_prediction.py` |
| **模型类型** | 脚本示例 |
| **适用场景** | 明显季节性波动的数据 |

**核心算法：**
```python
# 计算季节指数
seasonal_mean = data.reshape(n_years, n_seasons).mean(axis=0)
seasonal_index = seasonal_mean / total_mean

# 消除季节影响
deseasonalized = data / seasonal_index[季节索引]

# 拟合趋势（线性回归）
b, a = np.polyfit(t, deseasonalized, 1)

# 预测
future_trend = a + b * future_t
future_values = future_trend * seasonal_index[对应季节]
```

---

## 11. 概率模型 (Probability)

### 11.1 贝叶斯推断

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/probability/bayesian_inference.py` |
| **主要类** | `NormalNormalBayes`, `BetaBinomialBayes` |
| **适用场景** | 由果推因、参数估计、不确定性量化 |

**正态-正态共轭模型接口：**
```python
class NormalNormalBayes(BayesianInference):
    def __init__(self, prior_mu=0, prior_tau=10, known_sigma=1):
        """
        参数:
            prior_mu: float - 先验均值 μ₀
            prior_tau: float - 先验标准差 τ₀
            known_sigma: float - 已知的数据标准差
        """
    
    def fit(self, data):
        """根据数据更新后验分布"""
    
    def credible_interval(self, level=0.95):
        """返回后验可信区间 (lower, upper)"""
    
    def predict(self, n_samples=1000):
        """后验预测分布采样"""
    
    def update(self, new_data):
        """序列贝叶斯更新"""
    
    def plot_distributions(self, save_path=None):
        """绘制先验、似然、后验分布"""
```

---

### 11.2 高斯分布分析

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/probability/gaussian_distribution.py` |
| **类名** | `GaussianDistribution` |
| **适用场景** | 正态分布参数估计、假设检验 |

**接口规范：**
```python
class GaussianDistribution:
    def __init__(self, mu=None, sigma=None):
        """参数可从数据估计或直接指定"""
    
    def fit(self, data):
        """MLE参数估计"""
    
    def pdf(self, x): """概率密度函数"""
    def cdf(self, x): """累积分布函数"""
    def ppf(self, q): """分位点函数"""
    
    def probability_range(self, a, b):
        """P(a ≤ X ≤ b)"""
    
    def confidence_interval(self, confidence=0.95):
        """均值的置信区间"""
    
    def normality_test(self, method='shapiro'):
        """
        正态性检验
        method: 'shapiro'/'ks'/'anderson'/'dagostino'/'all'
        """
    
    def sample(self, n=100):
        """生成随机样本"""
```

---

### 11.3 高斯混合模型 (GMM)

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/probability/gaussian_mixture_model.py` |
| **类名** | `GMMClustering` |
| **适用场景** | 软聚类、密度估计、异常检测 |

**接口规范：**
```python
class GMMClustering:
    def __init__(self, n_components='auto', covariance_type='full',
                 max_components=10, random_state=42):
        """
        参数:
            n_components: int或'auto' - 聚类数
            covariance_type: 'full'/'tied'/'diag'/'spherical'
        """
    
    def fit(self, X, scale=True):
        """拟合GMM，自动选择最佳k"""
    
    def predict(self, X_new): """预测簇标签"""
    def predict_proba(self, X_new): """预测各簇概率"""
    
    def get_cluster_summary(self):
        """返回各簇统计摘要DataFrame"""
    
    def detect_anomalies(self, threshold=0.01):
        """基于密度的异常检测"""
    
    def sample(self, n_samples=100):
        """从拟合的GMM生成新样本"""
```

---

### 11.4 蒙特卡洛模拟

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/probability/monte_carlo_simulation.py` |
| **类名** | `MonteCarloSimulator` |
| **适用场景** | 风险分析、数值积分、随机过程 |

**接口规范：**
```python
class MonteCarloSimulator:
    def __init__(self, n_simulations=10000, random_seed=42):
        """
        参数:
            n_simulations: int - 模拟次数
            random_seed: int - 随机种子
        """
    
    def simulate(self, simulation_func, *args, **kwargs):
        """
        执行模拟
        参数:
            simulation_func: callable - 单次模拟函数
        返回:
            results: ndarray - 模拟结果数组
        """
    
    def simulate_vectorized(self, simulation_func, *args, **kwargs):
        """向量化模拟（更快）"""
    
    def percentile(self, q): """计算分位数"""
    def probability_above(self, threshold): """P(X > threshold)"""
    def probability_below(self, threshold): """P(X < threshold)"""
    def probability_between(self, lower, upper): """P(lower < X < upper)"""
    
    def value_at_risk(self, confidence=0.95):
        """VaR风险价值"""
    
    def conditional_value_at_risk(self, confidence=0.95):
        """CVaR条件风险价值"""
    
    def plot_distribution(self, title='...', save_path=None): pass
    def plot_convergence(self, save_path=None): pass
```

---

## 12. 回归分析 (Regression)

### 12.1 高斯过程回归 (GPR)

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/regression/gaussian_process_regression.py` |
| **模型类型** | 脚本示例（使用sklearn） |
| **适用场景** | 非线性回归、不确定性估计 |

**核心函数/类：**
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 定义核函数
kernel = C(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

# 创建模型
gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=0.01,              # 噪声项
    n_restarts_optimizer=10  # 优化器重启次数
)

# 训练与预测
gpr.fit(X_train, y_train)
y_pred, y_std = gpr.predict(X_test, return_std=True)  # 预测值和标准差
```

**特点：**
- 提供预测的不确定性估计（标准差）
- 可用于置信区间计算

---

### 12.2 多元回归分析

| 属性 | 描述 |
|------|------|
| **文件路径** | `models/regression/regression_analysis.py` |
| **模型类型** | 脚本示例（使用sklearn） |
| **适用场景** | 线性/多项式回归 |

**核心函数/类：**
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 线性回归
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
coef = linear_model.coef_         # 系数
intercept = linear_model.intercept_  # 截距

# 多项式回归
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
```

---

## 通用接口模式总结

### 类封装模型的标准模式

```python
class ModelName:
    def __init__(self, **params):
        """初始化模型参数"""
        self.param1 = params.get('param1', default_value)
        self.fitted = False
        self.results = None
    
    def fit(self, data, **kwargs):
        """训练/拟合模型"""
        # 处理数据
        # 执行算法
        self.fitted = True
        return self
    
    def predict(self, X_new):
        """预测新数据"""
        if not self.fitted:
            raise ValueError("请先调用 fit()")
        return predictions
    
    def get_results(self):
        """获取结果（通常返回DataFrame）"""
        return self.results
    
    def plot_xxx(self, save_path=None):
        """可视化结果"""
        fig, ax = plt.subplots(...)
        # 绑定绘图
        if save_path:
            plt.savefig(save_path)
        plt.show()
        return fig
```

### 数据预处理建议

| 模型类型 | 是否需要标准化 | 推荐方法 |
|---------|---------------|---------|
| KNN | ✅ 必须 | StandardScaler |
| 神经网络 | ✅ 必须 | StandardScaler/MinMaxScaler |
| 决策树 | ❌ 不需要 | - |
| 线性回归 | ⚠️ 建议 | StandardScaler |
| K-Means | ✅ 必须 | StandardScaler |
| SOM | ✅ 必须 | MinMaxScaler (0-1) |
| 高斯过程 | ✅ 建议 | StandardScaler |

---

## 版本信息

- 文档版本: 1.0
- 创建日期: 2026年1月
- 适用于: MCM/ICM 2026

