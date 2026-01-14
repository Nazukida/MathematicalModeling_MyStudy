#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
连续变量相关性分析案例说明：
本案例实现三种常用的连续变量相关性分析方法：
1. 协方差：衡量两个变量变化趋势的一致性，值为正表示同向变化，值为负表示反向变化
2. Pearson相关系数：标准化的协方差，取值范围[-1,1]，衡量线性相关程度
3. Spearman相关系数：基于变量秩次的非参数方法，取值范围[-1,1]，衡量单调相关关系

示例使用波士顿房价数据集（或替代数据集），分析房屋平均房间数与房屋价格之间的相关性，
并对比三种方法的结果差异和适用场景。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# 加载数据集（使用seaborn内置数据集）
print("加载数据集...")
# 尝试加载波士顿房价数据集，若不可用则使用替代数据集
try:
    # 波士顿房价数据集
    boston = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
    x = boston['rm']      # 平均房间数
    y = boston['medv']    # 房屋价格
    var1_name = '平均房间数'
    var2_name = '房屋价格(千美元)'
except:
    # 替代数据集：鸢尾花数据集
    iris = sns.load_dataset('iris')
    x = iris['sepal_length']  # 花萼长度
    y = iris['petal_length']  # 花瓣长度
    var1_name = '花萼长度(cm)'
    var2_name = '花瓣长度(cm)'

# 1. 计算协方差
# 协方差公式：cov(X,Y) = E[(X-μX)(Y-μY)]
mean_x = np.mean(x)
mean_y = np.mean(y)
covariance = np.mean((x - mean_x) * (y - mean_y))
# 使用numpy内置函数验证
cov_matrix = np.cov(x, y)
numpy_cov = cov_matrix[0, 1]

# 2. 计算Pearson相关系数
# Pearson公式：r = cov(X,Y)/(σXσY)
std_x = np.std(x, ddof=0)  # 总体标准差
std_y = np.std(y, ddof=0)
pearson_corr = covariance / (std_x * std_y)
# 使用scipy函数验证并获取p值
pearson_corr_scipy, pearson_p = pearsonr(x, y)

# 3. 计算Spearman相关系数（基于秩次）
# Spearman是对变量排序后计算的Pearson相关系数
x_rank = x.rank()
y_rank = y.rank()
mean_rank_x = np.mean(x_rank)
mean_rank_y = np.mean(y_rank)
cov_rank = np.mean((x_rank - mean_rank_x) * (y_rank - mean_rank_y))
std_rank_x = np.std(x_rank, ddof=0)
std_rank_y = np.std(y_rank, ddof=0)
spearman_corr = cov_rank / (std_rank_x * std_rank_y)
# 使用scipy函数验证并获取p值
spearman_corr_scipy, spearman_p = spearmanr(x, y)

# 输出分析结果
print(f"\n{var1_name}与{var2_name}的相关性分析结果：")
print(f"1. 协方差: {covariance:.4f} (numpy验证: {numpy_cov:.4f})")
print(f"2. Pearson相关系数: {pearson_corr:.4f} (scipy验证: {pearson_corr_scipy:.4f})，P值: {pearson_p:.8f}")
print(f"3. Spearman相关系数: {spearman_corr:.4f} (scipy验证: {spearman_corr_scipy:.4f})，P值: {spearman_p:.8f}")

# 结果解释
alpha = 0.05
pearson_significant = "显著" if pearson_p < alpha else "不显著"
spearman_significant = "显著" if spearman_p < alpha else "不显著"

print(f"\n结果解释：")
print(f"- Pearson相关系数表明{var1_name}与{var2_name}存在{pearson_significant}的线性相关")
print(f"- Spearman相关系数表明{var1_name}与{var2_name}存在{spearman_significant}的单调相关")

# 可视化相关性
plt.figure(figsize=(15, 10))

# 1. 散点图与回归线
plt.subplot(2, 2, 1)
sns.regplot(x=x, y=y, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title(f'{var1_name}与{var2_name}的散点图及回归线')
plt.xlabel(var1_name)
plt.ylabel(var2_name)
plt.grid(alpha=0.3)

# 2. 相关性热图
plt.subplot(2, 2, 2)
corr_df = pd.DataFrame({var1_name: x, var2_name: y})
corr_matrix = corr_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            center=0, annot_kws={"size": 12})
plt.title('相关性热图')

# 3. 秩次散点图（用于Spearman）
plt.subplot(2, 2, 3)
plt.scatter(x_rank, y_rank, alpha=0.6, color='green')
sns.regplot(x=x_rank, y=y_rank, scatter=False, line_kws={'color':'darkgreen'})
plt.title(f'{var1_name}与{var2_name}的秩次散点图')
plt.xlabel(f'{var1_name}的秩次')
plt.ylabel(f'{var2_name}的秩次')
plt.grid(alpha=0.3)

# 4. 相关性数值对比
plt.subplot(2, 2, 4)
methods = ['协方差', 'Pearson相关系数', 'Spearman相关系数']
values = [covariance, pearson_corr, spearman_corr]
plt.bar(methods, values, color=['blue', 'orange', 'green'])
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
plt.title('三种相关性度量值对比')
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
    