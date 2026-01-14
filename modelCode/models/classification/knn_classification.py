#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KNN分类模型案例说明：
KNN（K近邻）是一种简单的监督学习算法，其核心思想是：一个样本的类别由其周围k个最近邻样本的类别决定。
KNN没有显式的训练过程，属于"懒惰学习"算法，在预测时才进行计算。

算法步骤：
1. 确定k值（近邻数量）
2. 计算待预测样本与所有训练样本的距离（常用欧氏距离）
3. 选取距离最近的k个样本
4. 这k个样本中出现次数最多的类别即为待预测样本的类别

本案例使用鸢尾花数据集，通过KNN模型进行分类，并对比不同k值对模型性能的影响，
最终选择最优k值并可视化分类结果。
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# 加载数据集
iris = load_iris()
X = iris.data  # 特征数据
y = iris.target  # 标签
feature_names = iris.feature_names
class_names = iris.target_names

# 数据划分：训练集和测试集（7:3）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# 数据标准化（KNN对距离敏感，需要标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 尝试不同的k值，选择最优k
k_range = range(1, 21)
accuracy_scores = []

for k in k_range:
    # 创建并训练KNN模型
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train_scaled, y_train)
    
    # 预测并计算准确率
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print(f"k={k}时，测试集准确率: {accuracy:.4f}")

# 找到最优k值
best_k = k_range[np.argmax(accuracy_scores)]
print(f"\n最优k值为: {best_k}")

# 使用最优k值构建最终模型
knn_best = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
knn_best.fit(X_train_scaled, y_train)
y_pred_best = knn_best.predict(X_test_scaled)

# 输出详细评估结果
print("\n最优模型在测试集上的性能：")
print(f"准确率: {accuracy_score(y_test, y_pred_best):.4f}")
print("\n混淆矩阵：")
print(confusion_matrix(y_test, y_pred_best))
print("\n分类报告：")
print(classification_report(y_test, y_pred_best, target_names=class_names))

# 可视化不同k值的准确率
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracy_scores, 'bo-')
plt.xlabel('k值（近邻数量）')
plt.ylabel('测试集准确率')
plt.title('不同k值对KNN模型性能的影响')
plt.grid(alpha=0.3)
plt.axvline(x=best_k, color='r', linestyle='--', label=f'最优k={best_k}')
plt.legend()
plt.show()

# 使用PCA降维到2D，可视化分类结果
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 绘制决策边界
h = 0.02  # 网格步长
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 使用PCA转换后的训练数据重新训练模型（仅用于可视化）
knn_pca = KNeighborsClassifier(n_neighbors=best_k)
knn_pca.fit(X_train_pca, y_train)
Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界和样本点
plt.figure(figsize=(12, 5))

# 1. 训练集分类结果
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=plt.cm.coolwarm, 
            edgecolors='k', label='训练样本')
plt.xlabel(f'PCA特征1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PCA特征2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title(f'KNN决策边界 (k={best_k}) - 训练集')
plt.legend()

# 2. 测试集分类结果
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=plt.cm.coolwarm, 
            edgecolors='k', marker='s', s=80, label='真实标签')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_best, cmap=plt.cm.coolwarm, 
            edgecolors='y', marker='o', s=40, label='预测标签')
plt.xlabel(f'PCA特征1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PCA特征2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title(f'KNN决策边界 (k={best_k}) - 测试集')
plt.legend()

plt.tight_layout()
plt.show()
    