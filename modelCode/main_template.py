"""
============================================================
主程序模板 - 数学建模分析流程
============================================================

使用方法：
1. 复制此文件，重命名为你的问题名称（如 problem_A.py）
2. 修改数据加载和模型部分
3. 运行分析

"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加 modelCode 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modelCode'))


# ============================================================
# 第1步：数据预处理
# ============================================================

# 导入预处理工具
from data_preprocessing.preprocessing_tools import DataCleaner, DataScaler, OutlierDetector

# 加载数据
# data = pd.read_csv('你的数据.csv')
data = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])  # 示例

# 数据清洗
cleaner = DataCleaner(verbose=True)
cleaner.check_quality(data)
data = cleaner.fill_missing(data, method='median')

# 异常值处理（可选）
# detector = OutlierDetector()
# data, _ = detector.detect_zscore(data, handle='clip')

# 标准化（可选）
# scaler = DataScaler()
# data_scaled = scaler.fit_transform(data, method='standard')


# ============================================================
# 第2步：建模求解
# ============================================================

# 根据问题类型选择模型：

# --- 非线性规划 ---
# from models.optimization.advanced_nonlinear_programming import NonlinearProgrammingSolver
# solver = NonlinearProgrammingSolver()
# result = solver.solve(objective=目标函数, bounds=边界, constraints=约束)

# --- 整数规划 ---
# from models.optimization.advanced_integer_programming import IntegerProgrammingSolver
# solver = IntegerProgrammingSolver()
# result = solver.solve_custom_ip(c=目标系数, A_ub=不等式矩阵, b_ub=不等式右端项, ...)

# --- 多目标规划 ---
# from models.optimization.advanced_multi_objective import MultiObjectiveSolver
# solver = MultiObjectiveSolver()
# result = solver.solve_weighted_sum(objectives=目标函数列表, weights=权重, bounds=边界)

# --- 预测模型 ---
# from models.prediction.grey_prediction import GreyPredictor
# predictor = GreyPredictor()
# result = predictor.predict(data, n_predict=5)

# 示例：简单优化
from scipy.optimize import minimize
result = minimize(lambda x: x[0]**2 + x[1]**2, x0=[1, 1])
print(f"优化结果: {result.x}")


# ============================================================
# 第3步：结果可视化
# ============================================================

from visualization.plot_config import PlotStyleConfig, FigureSaver

# 设置论文样式
PlotStyleConfig.setup_style('academic')

# 绑图
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(data['A'], label='A')
ax.plot(data['B'], label='B')
ax.set_xlabel('时间')
ax.set_ylabel('值')
ax.set_title('数据分析结果')
ax.legend()
ax.grid(True, alpha=0.3)

# 保存图片
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/result.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ 分析完成！图片已保存到 figures/ 目录")
