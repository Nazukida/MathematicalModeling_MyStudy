#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
层次聚类算法案例说明：
层次聚类是一种无监督聚类方法，通过构建聚类树（ dendrogram ）来展示数据的层次聚类结构。
主要分为两种：
1. 凝聚式（自底向上）：从单个样本开始，逐步合并最相似的簇
2. 分裂式（自顶向下）：从所有样本为一个簇开始，逐步分裂为更小的簇

本案例使用葡萄酒数据集，采用凝聚式层次聚类，通过不同的距离度量（欧氏距离、曼哈顿距离）
和链接方法（ Ward 法、平均距离法）进行聚类，并绘制聚类树和聚类结果可视化。
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# 加载数据集
wine = load_wine()
X = wine.data  # 特征数据
y_true = wine.target  # 真实标签（仅用于对比）
feature_names = wine.feature_names

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 执行层次聚类（使用Ward链接法，基于欧氏距离）
# 常用链接方法：'ward'（最小化簇内方差）、'average'（平均距离）、'complete'（最大距离）
Z = linkage(X_scaled, method='ward', metric='euclidean')

# 输出聚类树信息
print("层次聚类树（前10个合并步骤）：")
print(pd.DataFrame(Z[:10], 
                  columns=['簇1', '簇2', '距离', '样本数']).round(2))

# 绘制聚类树（ dendrogram ）
plt.figure(figsize=(12, 6))
dendrogram(Z, 
           truncate_mode='lastp',  # 只显示最后p个合并
           p=20,                   # 显示的合并数
           leaf_rotation=90,       # 叶子节点旋转角度
           leaf_font_size=10,      # 叶子节点字体大小
           show_contracted=True,   # 显示收缩的簇
           color_threshold=5,      # 颜色阈值
           above_threshold_color='gray')
plt.title('层次聚类树（Ward链接法）')
plt.xlabel('样本或簇')
plt.ylabel('距离')
plt.grid(axis='y', alpha=0.3)
plt.show()

# 根据聚类树确定聚类数量并获取聚类结果
k = 3  # 已知葡萄酒数据集有3类
y_pred = fcluster(Z, k, criterion='maxclust')  # 划分为k个簇
y_pred = y_pred - 1  # 调整标签从0开始

# 输出聚类结果
print(f"\n层次聚类结果 (k={k}):")
for i in range(k):
    print(f"簇 {i}: {np.sum(y_pred == i)} 个样本")

# 使用PCA降维到2维以便可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 可视化聚类结果
plt.figure(figsize=(12, 5))

# 1. 层次聚类结果
plt.subplot(1, 2, 1)
colors = ['navy', 'turquoise', 'darkorange']
for i in range(k):
    plt.scatter(X_pca[y_pred == i, 0], X_pca[y_pred == i, 1], 
                c=colors[i], alpha=0.6, label=f'簇 {i}')
plt.xlabel(f'PCA特征1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PCA特征2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title(f'层次聚类结果 (k={k})')
plt.legend()
plt.grid(alpha=0.3)

# 2. 与真实标签对比
plt.subplot(1, 2, 2)
for i in range(3):
    plt.scatter(X_pca[y_true == i, 0], X_pca[y_true == i, 1], 
                c=colors[i], alpha=0.6, label=wine.target_names[i])
plt.xlabel(f'PCA特征1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PCA特征2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('真实标签分布')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 对比不同链接方法的聚类树
plt.figure(figsize=(15, 5))
methods = ['ward', 'average', 'complete']
for i, method in enumerate(methods):
    plt.subplot(1, 3, i+1)
    Z_method = linkage(X_scaled, method=method, metric='euclidean')
    dendrogram(Z_method, 
               truncate_mode='lastp', 
               p=10, 
               leaf_rotation=90, 
               leaf_font_size=8)
    plt.title(f'链接方法: {method}')
    plt.xlabel('样本或簇')
    plt.ylabel('距离')
    plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
    