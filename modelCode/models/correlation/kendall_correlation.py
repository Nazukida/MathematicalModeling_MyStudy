#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kendall相关系数分析案例说明：
本案例实现Kendall秩相关系数（τ系数）分析，这是一种非参数统计方法，用于衡量
两个变量之间的单调相关程度。与Spearman相关系数相比，Kendall更适用于样本量
较小或存在较多相同秩次（平局）的数据。

Kendall系数取值范围为[-1, 1]：
- 1表示完全正相关
- -1表示完全负相关
- 0表示无单调相关

示例使用汽车数据集，分析汽车重量与油耗之间的相关性，并与Pearson和Spearman
系数进行对比，展示不同相关系数的特点。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, spearmanr

# 加载数据集
print("加载汽车数据集...")
mpg = sns.load_dataset('mpg')

# 选择两个变量进行分析
# 连续变量：weight（汽车重量）和mpg（每加仑英里数，油耗的反向指标）
x = mpg['weight']
y = mpg['mpg']
var1_name = '汽车重量'
var2_name = '每加仑英里数(mpg)'

# 数据预处理：移除缺失值
data = pd.DataFrame({var1_name: x, var2_name: y}).dropna()
x = data[var1_name]
y = data[var2_name]

# 手动计算Kendall相关系数（简化版，处理小样本）
def kendall_correlation(x, y):
    """手动计算Kendall相关系数"""
    n = len(x)
    concordant = 0  # 一致对数量
    discordant = 0  # 不一致对数量
    
    # 遍历所有数据对
    for i in range(n):
        for j in range(i+1, n):
            # 计算x和y的差异符号
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            
            if dx * dy > 0:
                # 符号相同，一致对
                concordant += 1
            elif dx * dy < 0:
                # 符号相反，不一致对
                discordant += 1
            # 若dx或dy为0，视为平局，不纳入计算
    
    # Kendall系数公式：τ = (一致对 - 不一致对) / C(n,2)
    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 0
    return (concordant - discordant) / total_pairs

# 计算三种相关系数
kendall_manual = kendall_correlation(x[:50].values, y[:50].values)  # 手动计算（小样本）
kendall_tau, kendall_p = kendalltau(x, y)  # 使用scipy计算
pearson_r, pearson_p = pearsonr(x, y)
spearman_rho, spearman_p = spearmanr(x, y)

# 输出分析结果
print(f"\n{var1_name}与{var2_name}的相关性分析结果：")
print(f"Kendall相关系数（手动计算，小样本）: {kendall_manual:.4f}")
print(f"Kendall相关系数（τ）: {kendall_tau:.4f}，P值: {kendall_p:.8f}")
print(f"Pearson相关系数（r）: {pearson_r:.4f}，P值: {pearson_p:.8f}")
print(f"Spearman相关系数（ρ）: {spearman_rho:.4f}，P值: {spearman_p:.8f}")

# 结果解释
alpha = 0.05
kendall_significant = "显著" if kendall_p < alpha else "不显著"
print(f"\n结果解释：{var1_name}与{var2_name}存在{kendall_significant}的Kendall单调相关")

# 可视化相关性
plt.figure(figsize=(15, 10))

# 1. 散点图与回归线
plt.subplot(2, 2, 1)
sns.regplot(x=x, y=y, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title(f'{var1_name}与{var2_name}的散点图及回归线')
plt.xlabel(var1_name)
plt.ylabel(var2_name)
plt.grid(alpha=0.3)

# 2. 三种相关系数对比
plt.subplot(2, 2, 2)
methods = ['Kendall τ', 'Pearson r', 'Spearman ρ']
values = [kendall_tau, pearson_r, spearman_rho]
plt.bar(methods, values, color=['blue', 'orange', 'green'])
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
plt.ylim(-1.1, 1.1)
plt.title('三种相关系数对比')
plt.grid(axis='y', alpha=0.3)

# 3. 秩次散点图（用于Kendall和Spearman）
plt.subplot(2, 2, 3)
x_rank = x.rank()
y_rank = y.rank()
plt.scatter(x_rank, y_rank, alpha=0.6, color='purple')
sns.regplot(x=x_rank, y=y_rank, scatter=False, line_kws={'color':'darkred'})
plt.title(f'{var1_name}与{var2_name}的秩次散点图')
plt.xlabel(f'{var1_name}的秩次')
plt.ylabel(f'{var2_name}的秩次')
plt.grid(alpha=0.3)

# 4. 相关性热图
plt.subplot(2, 2, 4)
corr_df = pd.DataFrame({var1_name: x, var2_name: y})
corr_matrix = corr_df.corr(method='kendall')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            center=0, annot_kws={"size": 12})
plt.title('Kendall相关系数热图')

plt.tight_layout()
plt.show()

# 补充说明不同相关系数的适用场景
print("\n相关系数适用场景说明：")
print("- Kendall τ：适用于小样本、存在较多平局或需要更稳健结果的情况")
print("- Pearson r：适用于线性关系且数据近似正态分布的情况")
print("- Spearman ρ：适用于非线性但单调的关系，对异常值较稳健")
    