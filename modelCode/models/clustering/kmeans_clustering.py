#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
K-means聚类算法案例说明：
K-means是一种常用的无监督聚类算法，通过迭代方式将数据集划分为k个不同的簇。
算法步骤：
1. 随机选择k个初始质心
2. 将每个样本分配到最近的质心所在的簇
3. 重新计算每个簇的质心（平均值）
4. 重复步骤2-3，直到质心不再显著变化或达到最大迭代次数

本案例使用鸢尾花数据集（4维特征），通过K-means聚为3类，
并将结果可视化展示（使用PCA降维到2维以便可视化）。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 加载数据集
iris = load_iris()
X = iris.data  # 特征数据
y_true = iris.target  # 真实标签（仅用于对比，无监督学习中不使用）
feature_names = iris.feature_names

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 设置聚类数量（已知鸢尾花有3个品种）
k = 3

# 执行K-means聚类
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)  # 聚类结果

# 获取聚类中心
centers = kmeans.cluster_centers_

# 输出聚类结果信息
print(f"K-means聚类结果 (k={k}):")
print(f"迭代次数: {kmeans.n_iter_}")
print(f"最终惯性值(簇内平方和): {kmeans.inertia_:.4f}")
print("\n每个簇的样本数量:")
for i in range(k):
    print(f"簇 {i}: {np.sum(y_pred == i)} 个样本")

# 使用PCA降维到2维以便可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(centers)  # 转换质心到PCA空间

# 可视化聚类结果
plt.figure(figsize=(12, 5))

# 1. 聚类结果可视化
plt.subplot(1, 2, 1)
colors = ['navy', 'turquoise', 'darkorange']
for i in range(k):
    plt.scatter(X_pca[y_pred == i, 0], X_pca[y_pred == i, 1], 
                c=colors[i], alpha=0.6, label=f'簇 {i}')
# 绘制质心
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, marker='X', 
            c='red', label='质心', edgecolors='black')
plt.xlabel(f'PCA特征1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PCA特征2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title(f'K-means聚类结果 (k={k})')
plt.legend()
plt.grid(alpha=0.3)

# 2. 与真实标签对比（仅用于演示）
plt.subplot(1, 2, 2)
for i in range(3):
    plt.scatter(X_pca[y_true == i, 0], X_pca[y_true == i, 1], 
                c=colors[i], alpha=0.6, label=iris.target_names[i])
plt.xlabel(f'PCA特征1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PCA特征2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('真实标签分布')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 尝试不同k值，通过肘部法确定最佳k值
inertias = []
k_range = range(1, 11)
for k_test in k_range:
    kmeans_test = KMeans(n_clusters=k_test, n_init=10, random_state=42)
    kmeans_test.fit(X_scaled)
    inertias.append(kmeans_test.inertia_)

# 绘制肘部图
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('k值（聚类数量）')
plt.ylabel('惯性值（簇内平方和）')
plt.title('肘部法确定最佳k值')
plt.grid(alpha=0.3)
plt.show()
    