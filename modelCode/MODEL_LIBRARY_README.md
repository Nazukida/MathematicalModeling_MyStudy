# 数学建模代码库 (MCM/ICM Model Library)

## 📁 目录结构

```
modelCode/
├── data_preprocessing/          # 数据预处理模块
│   ├── __init__.py
│   ├── preprocessing_tools.py   # 数据清洗、标准化、异常值处理
│   ├── preprocessing_toolkit.py # 完整预处理工具集
│   └── pca_reduction.py         # PCA主成分分析降维
│
├── models/                       # 模型库
│   ├── __init__.py
│   ├── probability/             # 概率统计模型
│   │   ├── __init__.py
│   │   ├── gaussian_distribution.py   # 高斯分布分析
│   │   ├── gaussian_mixture_model.py  # GMM聚类
│   │   ├── bayesian_inference.py      # 贝叶斯推断
│   │   └── monte_carlo_simulation.py  # 蒙特卡洛模拟
│   │
│   ├── optimization/            # 优化算法
│   │   ├── __init__.py
│   │   ├── optimization_algorithms.py  # PSO、GA、DE、参数反演
│   │   ├── optimization_toolkit.py     # 优化工具集
│   │   └── dynamic_programming.py      # 动态规划
│   │
│   └── evaluation/              # 评价模型
│       ├── __init__.py
│       └── evaluation_toolkit.py       # 熵权法、TOPSIS
│
├── workflow/                    # 🆕 工作流模块（串联各模块）
│   ├── __init__.py
│   └── model_validation_pipeline.py    # 模型验证工作流
│
├── visualization/               # 可视化模块
│   ├── __init__.py
│   └── plot_config.py          # 统一图表配置
│
├── workflow_tutorial.py         # 工作流使用教程
└── figures/                     # 图表输出目录
```

---

## 🚀 快速开始

### 0. 🆕 工作流模式（推荐）

**一行代码完成模型验证：**
```python
from workflow import quick_dp_validation, quick_optimization_validation

# 动态规划背包问题
items = [[2, 6], [2, 3], [6, 5], [5, 4], [4, 6]]  # [重量, 价值]
result = quick_dp_validation(items, capacity=10)
print(f"最大价值: {result['max_value']}")

# 优化算法验证
def sphere(x): return sum(x**2)
result = quick_optimization_validation(sphere, bounds=(-5, 5), n_dims=3)
```

**完整工作流（预处理 → 模型 → 可视化）：**
```python
from workflow import (
    ModelValidationPipeline,
    MissingValueStep, OutlierRemovalStep, NormalizationStep,
    DynamicProgrammingAdapter,
    DPTableVisualization, DataComparisonVisualization
)

# 创建工作流
pipeline = ModelValidationPipeline("背包问题验证")

# 加载数据
pipeline.load_data(items_data, "物品列表")

# 添加预处理步骤（可链式调用）
pipeline.add_preprocessing(MissingValueStep('mean'))       # 缺失值填充
pipeline.add_preprocessing(OutlierRemovalStep('iqr', 1.5)) # 异常值处理
pipeline.add_preprocessing(NormalizationStep('minmax'))    # 标准化

# 设置模型并配置参数
pipeline.set_model(DynamicProgrammingAdapter())
pipeline.configure_model(capacity=15)

# 添加可视化
pipeline.add_visualization(DPTableVisualization())
pipeline.add_visualization(DataComparisonVisualization())

# 运行并查看结果
pipeline.run()
pipeline.show_results()
pipeline.show_figures()
pipeline.save_figures('./figures/')
```

### 1. 数据预处理

```python
from data_preprocessing import DataCleaner, DataScaler, PCAReducer

# 数据清洗
cleaner = DataCleaner(data)
cleaned_data = cleaner.clean()

# 标准化
scaler = DataScaler(cleaned_data)
scaled_data = scaler.standardize()

# PCA降维
pca = PCAReducer(n_components=3)
pca.fit(scaled_data)
reduced = pca.transform(scaled_data)
pca.plot_explained_variance()
```

### 2. 概率统计模型

#### 高斯分布分析
```python
from models.probability import GaussianDistribution

gauss = GaussianDistribution(data)
gauss.fit()
gauss.plot_distribution()
gauss.plot_qq()
is_normal, stats = gauss.normality_test()
print(f"正态性检验: {'通过' if is_normal else '不通过'}")
```

#### GMM聚类
```python
from models.probability import GMMClustering

gmm = GMMClustering(n_components=3)
gmm.fit(data)
labels = gmm.predict(data)
probs = gmm.predict_proba(data)
gmm.plot_clusters(data)
gmm.plot_component_selection()  # BIC/AIC曲线
```

#### 贝叶斯推断（由果推因）
```python
from models.probability import BayesianParameterEstimation

# 例：从观测数据推断参数
def model(x, a, b, c):
    return a * x**2 + b * x + c

bayes = BayesianParameterEstimation(param_names=['a', 'b', 'c'])
bayes.add_observation(x_data, y_data, sigma=0.1)
bayes.run_mcmc(n_samples=10000)
bayes.plot_posterior()  # 后验分布
bayes.plot_trace()      # MCMC轨迹
```

#### 蒙特卡洛模拟
```python
from models.probability import ProjectRiskSimulator

# 项目风险模拟
sim = ProjectRiskSimulator()
sim.add_task('任务A', min_days=5, mode_days=7, max_days=15)
sim.add_task('任务B', min_days=3, mode_days=5, max_days=10)
results = sim.run_simulation(n_simulations=10000)
sim.plot_distribution()
prob = sim.probability_exceeds(25)
print(f"超过25天的概率: {prob:.1%}")
```

### 3. 优化算法（参数反演）

```python
from models.optimization import PSO, GeneticAlgorithm, ParameterInversion

# 粒子群优化
pso = PSO(n_particles=50, n_dim=2, bounds=[(-5, 5), (-5, 5)])
best_pos, best_val = pso.optimize(objective_func, max_iter=100)
pso.plot_convergence()

# 参数反演（从观测反推参数）
def forward_model(params, x):
    a, b = params
    return a * np.exp(-b * x)

inversion = ParameterInversion(
    forward_model=forward_model,
    param_bounds=[(0, 10), (0, 1)]
)
best_params, error = inversion.invert(x_obs, y_obs, n_trials=10)
inversion.plot_fit(x_obs, y_obs)
```

### 4. 可视化配置

```python
from visualization import PlotStyleConfig, FigureSaver

# 初始化学术论文风格
PlotStyleConfig.setup_style()
colors = PlotStyleConfig.get_palette(5)

# 保存高质量图表
saver = FigureSaver(output_dir='./figures')
fig, ax = plt.subplots()
# ... 绑图代码 ...
saver.save(fig, 'my_figure', formats=['png', 'pdf'])
```

---

## 📊 模型选择指南

| 问题类型 | 推荐模型 | 适用场景 |
|---------|---------|---------|
| 描述随机性 | 高斯分布 | 自然现象、测量误差 |
| 软分类/聚类 | GMM | 重叠群体、异常检测 |
| 由果推因 | 贝叶斯推断 | 参数反演、逆问题 |
| 复杂过程模拟 | 蒙特卡洛 | 风险评估、不确定性 |
| 参数优化 | PSO/GA/DE | 复杂方程求解 |
| 离散决策 | 动态规划 | 背包、路径、资源分配 |
| 多指标评价 | 熵权TOPSIS | 方案排序、综合评估 |

---

## 🔗 模块串联指南

### 问题：各模块如何配合使用？

使用 `workflow` 模块可以轻松串联：

```
数据 → 预处理 → 模型 → 可视化 → 结果
```

### 工作流组件

| 组件类型 | 可用类 | 说明 |
|---------|-------|------|
| 数据容器 | `PipelineData` | 统一数据格式，支持多种转换 |
| 预处理步骤 | `MissingValueStep` | 缺失值处理（mean/median/knn） |
| | `OutlierRemovalStep` | 异常值处理（IQR/Z-score） |
| | `NormalizationStep` | 标准化（zscore/minmax/robust） |
| 模型适配器 | `DynamicProgrammingAdapter` | 动态规划（背包问题） |
| | `OptimizationAdapter` | 优化算法（PSO等） |
| 可视化 | `DPTableVisualization` | DP表格热力图 |
| | `ConvergenceVisualization` | 收敛曲线 |
| | `DataComparisonVisualization` | 预处理前后对比 |

### 自定义扩展

```python
from workflow import ModelAdapter, VisualizationStep

# 自定义模型适配器
class MyModelAdapter(ModelAdapter):
    def __init__(self):
        super().__init__("我的模型")
    
    def run(self, pipeline_data):
        data = pipeline_data.get_array()  # 获取数据
        # ... 你的模型逻辑 ...
        self.result = {'key': value}
        pipeline_data.set_model_output(self.result, "my_model")
        return pipeline_data

# 自定义可视化
class MyVisualization(VisualizationStep):
    def plot(self, pipeline_data):
        # ... 绑图代码 ...
        return self.fig
```

---

## 📝 扩展指南

### 添加新模型
1. 在对应子目录创建 `.py` 文件
2. 继承基类或遵循接口规范
3. 在 `__init__.py` 中导出
4. （可选）创建对应的 `ModelAdapter` 以支持工作流

### 自定义可视化
修改 `visualization/plot_config.py` 中的配置：
- `COLOR_PALETTES`: 配色方案
- `STYLE_PRESETS`: 样式预设
- `PlotTemplates`: 图表模板

### 添加工作流组件
在 `workflow/model_validation_pipeline.py` 中：
- 继承 `PreprocessingStep` 添加预处理步骤
- 继承 `ModelAdapter` 添加模型适配器
- 继承 `VisualizationStep` 添加可视化

---

## ⚠️ 注意事项

1. **路径问题**：使用相对导入时确保从项目根目录运行
2. **依赖安装**：`pip install numpy pandas matplotlib scipy scikit-learn`
3. **中文显示**：已配置SimHei字体，如显示异常请检查字体安装
4. **工作流教程**：运行 `python workflow_tutorial.py` 查看完整示例

---

*美国大学生数学建模竞赛 (MCM/ICM) 代码库*
