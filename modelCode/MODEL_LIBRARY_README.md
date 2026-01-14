# 数学建模代码库 (MCM/ICM Model Library)

## 📁 目录结构

```
modelCode/
├── data_preprocessing/          # 数据预处理模块
│   ├── __init__.py
│   ├── preprocessing_tools.py   # 数据清洗、标准化、异常值处理
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
│   └── optimization/            # 优化算法
│       ├── __init__.py
│       └── optimization_algorithms.py  # PSO、GA、DE、参数反演
│
├── visualization/               # 可视化模块
│   ├── __init__.py
│   └── plot_config.py          # 统一图表配置
│
└── figures/                     # 图表输出目录
```

---

## 🚀 快速开始

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

---

## 📝 扩展指南

### 添加新模型
1. 在对应子目录创建 `.py` 文件
2. 继承基类或遵循接口规范
3. 在 `__init__.py` 中导出

### 自定义可视化
修改 `visualization/plot_config.py` 中的配置：
- `COLOR_PALETTES`: 配色方案
- `STYLE_PRESETS`: 样式预设
- `PlotTemplates`: 图表模板

---

## ⚠️ 注意事项

1. **路径问题**：使用相对导入时确保从项目根目录运行
2. **依赖安装**：`pip install numpy pandas matplotlib scipy scikit-learn`
3. **中文显示**：已配置SimHei字体，如显示异常请检查字体安装

---

*美国大学生数学建模竞赛 (MCM/ICM) 代码库*
