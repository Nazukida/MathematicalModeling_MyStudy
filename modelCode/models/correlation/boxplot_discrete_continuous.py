#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
箱型图分析离散与连续变量相关性案例说明：
本案例使用箱型图分析离散变量与连续变量之间的关系。箱型图通过展示连续变量在
不同离散类别中的分布差异（如中位数、四分位数、异常值等）来判断两者是否相关。
分布差异越明显，表明两个变量的相关性越强。

示例使用泰坦尼克号数据集，分析"乘客等级"（离散变量）与"年龄"（连续变量）之间
的关系，通过比较不同乘客等级的年龄分布来判断它们是否相关。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 加载泰坦尼克号数据集
print("加载泰坦尼克号数据集...")
titanic = sns.load_dataset('titanic')

# 数据预处理：选择需要的变量并处理缺失值
# 离散变量：pclass（乘客等级：1,2,3）
# 连续变量：age（年龄）
data = titanic[['pclass', 'age']].dropna()
data['pclass'] = data['pclass'].astype('category')  # 确保pclass为分类类型

# 按离散变量分组，获取连续变量数据
groups = [data[data['pclass'] == cls]['age'].values for cls in sorted(data['pclass'].unique())]
class_names = [f'等级 {cls}' for cls in sorted(data['pclass'].unique())]

# 计算各组基本统计量
stats_summary = pd.DataFrame()
for i, cls in enumerate(sorted(data['pclass'].unique())):
    cls_data = data[data['pclass'] == cls]['age']
    stats_summary[f'等级 {cls}'] = [
        len(cls_data),                # 样本量
        cls_data.mean(),              # 均值
        cls_data.median(),            # 中位数
        cls_data.std(),               # 标准差
        cls_data.min(),               # 最小值
        cls_data.quantile(0.25),      # 下四分位数
        cls_data.quantile(0.75),      # 上四分位数
        cls_data.max()                # 最大值
    ]
stats_summary.index = ['样本量', '均值', '中位数', '标准差', '最小值', '25%分位数', '75%分位数', '最大值']
print("\n不同乘客等级的年龄统计量：")
print(stats_summary.round(2))

# 执行单因素方差分析（ANOVA），检验组间差异是否显著
f_val, p_val = stats.f_oneway(*groups)
print(f"\nANOVA检验结果：F值 = {f_val:.4f}, P值 = {p_val:.8f}")

# 结果解释
alpha = 0.05
if p_val < alpha:
    conclusion = f"由于P值({p_val:.8f}) < {alpha}，不同乘客等级的年龄分布存在显著差异，表明两者相关。"
else:
    conclusion = f"由于P值({p_val:.8f}) ≥ {alpha}，不同乘客等级的年龄分布无显著差异，表明两者不相关。"
print("结论：", conclusion)

# 绘制箱型图可视化
plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='age', data=data, palette='Set3', 
            showmeans=True, meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black"})

# 添加散点图展示实际数据分布
sns.stripplot(x='pclass', y='age', data=data, color='black', alpha=0.3, jitter=True, size=3)

plt.title('不同乘客等级的年龄分布箱型图', fontsize=14)
plt.xlabel('乘客等级', fontsize=12)
plt.ylabel('年龄', fontsize=12)
plt.xticks(ticks=[0, 1, 2], labels=['1等舱', '2等舱', '3等舱'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.text(1, data['age'].max()*0.95, 
         f'ANOVA: F={f_val:.2f}, P={p_val:.6f}', 
         ha='center', bbox=dict(facecolor='white', alpha=0.8))
plt.show()

# 补充：绘制小提琴图（展示更详细的分布形状）
plt.figure(figsize=(10, 6))
sns.violinplot(x='pclass', y='age', data=data, palette='Set3', inner='quartile')
plt.title('不同乘客等级的年龄分布小提琴图', fontsize=14)
plt.xlabel('乘客等级', fontsize=12)
plt.ylabel('年龄', fontsize=12)
plt.xticks(ticks=[0, 1, 2], labels=['1等舱', '2等舱', '3等舱'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
    