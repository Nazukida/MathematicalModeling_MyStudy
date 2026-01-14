#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准差法检测异常值案例说明：
本案例使用标准差法检测数据中的异常值。标准差法假设数据服从正态分布，
通常将偏离均值超过3个标准差的数据点视为异常值（3σ原则）。
示例中生成服从正态分布的模拟数据并添加异常值，使用3σ原则进行检测。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 生成示例数据：正态分布数据 + 异常值
np.random.seed(42)  # 设置随机种子，确保结果可复现
mu = 50             # 均值
sigma = 10          # 标准差
normal_data = np.random.normal(mu, sigma, 300)  # 正常数据
outliers = np.array([10, 20, 90, 100, 110, 120])  # 异常值
data = np.concatenate([normal_data, outliers])    # 合并数据

# 计算均值和标准差
mean = np.mean(data)
std = np.std(data)

# 定义异常值阈值（3σ原则）
lower_threshold = mean - 3 * std
upper_threshold = mean + 3 * std

# 检测异常值
outlier_indices = np.where((data < lower_threshold) | (data > upper_threshold))[0]
outlier_values = data[outlier_indices]
normal_values = data[np.where((data >= lower_threshold) & (data <= upper_threshold))[0]]

# 输出结果
print("标准差法(3σ原则)异常值检测结果：")
print(f"数据总量: {len(data)} 个")
print(f"均值(μ): {mean:.2f}")
print(f"标准差(σ): {std:.2f}")
print(f"异常值下限: μ - 3σ = {lower_threshold:.2f}")
print(f"异常值上限: μ + 3σ = {upper_threshold:.2f}")
print(f"检测到异常值数量: {len(outlier_values)} 个")
print(f"异常值: {', '.join([f'{x:.2f}' for x in outlier_values])}")

# 可视化结果
plt.figure(figsize=(12, 6))

# 绘制直方图
n, bins, patches = plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

# 标记正常数据和异常值区域
for patch in patches:
    x = patch.get_x() + patch.get_width()/2
    if x < lower_threshold or x > upper_threshold:
        patch.set_facecolor('red')
    else:
        patch.set_facecolor('skyblue')

# 绘制均值和3σ线
plt.axvline(mean, color='green', linestyle='-', linewidth=2, label=f'均值 (μ = {mean:.2f})')
plt.axvline(lower_threshold, color='orange', linestyle='--', linewidth=2, 
           label=f'μ - 3σ = {lower_threshold:.2f}')
plt.axvline(upper_threshold, color='orange', linestyle='--', linewidth=2, 
           label=f'μ + 3σ = {upper_threshold:.2f}')

# 添加图例和标签
plt.title('标准差法(3σ原则)异常值检测')
plt.xlabel('数据值')
plt.ylabel('频数')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加颜色说明
blue_patch = mpatches.Patch(color='skyblue', label='正常数据')
red_patch = mpatches.Patch(color='red', label='异常值区域')
plt.legend(handles=[blue_patch, red_patch, 
                   plt.Line2D([], [], color='green', linestyle='-', label=f'均值 (μ)'),
                   plt.Line2D([], [], color='orange', linestyle='--', label=f'3σ 边界')])

plt.show()
    