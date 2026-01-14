#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UMAP降维法案例说明：
本案例使用UMAP(均匀流形近似和投影)算法对高维数据进行降维。
UMAP是一种现代的非线性降维方法，相比t-SNE通常保留更多的全局结构，计算速度也更快。
示例中使用Fashion-MNIST数据集(784维)，通过UMAP降维到2维空间进行可视化。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import umap

# 加载Fashion-MNIST数据集（只使用部分数据加快计算）
print("加载Fashion-MNIST数据集...")
X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)

# 只使用1000个样本加快计算
n_samples = 1000
X_sample, y_sample = X[:n_samples], y[:n_samples]
y_sample = y_sample.astype(int)  # 转换为整数标签

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# 应用UMAP降维到2维
print("运行UMAP降维...")
reducer = umap.UMAP(
    n_components=2,        # 降维到2维
    n_neighbors=15,        # 近邻数量
    min_dist=0.1,          # 最小距离
    random_state=42        # 随机种子，确保结果可复现
)
X_umap = reducer.fit_transform(X_scaled)

# 输出UMAP结果信息
print("\nUMAP降维结果：")
print(f"原始数据维度: {X_sample.shape[1]} 维")
print(f"降维后数据维度: {X_umap.shape[1]} 维")
print(f"使用参数: n_neighbors={reducer.n_neighbors}, min_dist={reducer.min_dist}")

# 定义Fashion-MNIST类别名称
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 可视化UMAP降维结果
plt.figure(figsize=(12, 10))
# 使用不同颜色和标记区分不同类别
colors = plt.cm.rainbow(np.linspace(0, 1, 10))
markers = ['o', 's', '^', 'D', '*', 'P', 'X', 'd', 'p', 'H']

for cls in range(10):
    indices = y_sample == cls
    plt.scatter(
        X_umap[indices, 0], 
        X_umap[indices, 1], 
        c=[colors[cls]], 
        label=class_names[cls],
        marker=markers[cls],
        alpha=0.7,
        s=50
    )

plt.legend(title='类别', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('UMAP降维：Fashion-MNIST数据集(784维→2维)')
plt.xlabel('UMAP特征1')
plt.ylabel('UMAP特征2')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
    