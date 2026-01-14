#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
卡方检验案例说明：
本案例使用卡方检验分析两个离散变量之间的相关性。卡方检验通过比较观测频数
与期望频数的差异来判断两个分类变量是否独立。值越大，表明两个变量相关性越强。
示例中使用泰坦尼克号数据集，分析"性别"与"生存情况"两个离散变量的相关性。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.datasets import fetch_openml

# 加载泰坦尼克号数据集
print("加载泰坦尼克号数据集...")
titanic = fetch_openml('titanic', version=1, as_frame=True).frame

# 数据预处理：选择需要的离散变量并处理缺失值
data = titanic[['sex', 'survived']].dropna()
data['survived'] = data['survived'].astype(int)  # 将生存情况转换为整数

# 创建列联表（ contingency table）
contingency_table = pd.crosstab(data['sex'], data['survived'])
print("\n列联表（性别 vs 生存情况）:")
print(contingency_table)
print("\n解释：行表示性别（female/male），列表示生存情况（0=未生存，1=生存）")

# 执行卡方检验
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# 输出检验结果
print("\n卡方检验结果：")
print(f"卡方统计量: {chi2:.4f}")
print(f"P值: {p_value:.8f}")
print(f"自由度: {dof}")
print("\n期望频数表:")
print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns).round(2))

# 结果解释
alpha = 0.05
if p_value < alpha:
    conclusion = f"由于P值({p_value:.8f}) < {alpha}，拒绝原假设，表明性别与生存情况显著相关。"
else:
    conclusion = f"由于P值({p_value:.8f}) ≥ {alpha}，不拒绝原假设，表明性别与生存情况无显著相关。"
print("\n结论：", conclusion)

# 可视化列联表数据
plt.figure(figsize=(10, 6))
contingency_table.plot(kind='bar', stacked=False, color=['#ff9999','#66b3ff'])
plt.title('泰坦尼克号乘客性别与生存情况的关系')
plt.xlabel('性别')
plt.ylabel('人数')
plt.xticks(rotation=0)
plt.legend(title='生存情况', labels=['未生存', '生存'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.text(0.5, max(contingency_table.sum(axis=1))*0.9, 
         f'卡方值: {chi2:.2f}, P值: {p_value:.6f}', 
         ha='center', bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.show()
    