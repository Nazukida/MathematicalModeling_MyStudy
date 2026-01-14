#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
朴素贝叶斯分类模型案例说明：
朴素贝叶斯是基于贝叶斯定理和特征条件独立假设的分类算法。"朴素"指的是假设各个特征之间相互独立，
这一假设简化了计算，使模型能够高效处理高维数据。

算法步骤：
1. 计算先验概率：每个类别的出现概率
2. 计算似然概率：给定类别时每个特征的条件概率
3. 应用贝叶斯定理计算后验概率：给定特征时样本属于某个类别的概率
4. 选择后验概率最大的类别作为预测结果

本案例使用新闻组文本数据集，构建高斯朴素贝叶斯模型进行文本分类，
并评估模型在不同类别上的分类性能。
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import TruncatedSVD

# 加载数据集（选择4个新闻类别）
categories = ['rec.sport.baseball', 'rec.sport.hockey', 
              'sci.space', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='all', categories=categories,
                                remove=('headers', 'footers', 'quotes'),
                                shuffle=True, random_state=42)

X = newsgroups.data  # 文本数据
y = newsgroups.target  # 标签
class_names = newsgroups.target_names

print(f"数据集包含 {len(X)} 个样本，分为 {len(class_names)} 个类别：")
for i, name in enumerate(class_names):
    print(f"类别 {i}: {name}，样本数: {np.sum(y == i)}")

# 文本特征提取：将文本转换为TF-IDF特征向量
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)  # 稀疏矩阵
X_tfidf_dense = X_tfidf.toarray()  # 转换为稠密矩阵（用于高斯朴素贝叶斯）

# 数据划分：训练集和测试集（7:3）
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf_dense, y, test_size=0.3, random_state=42, stratify=y)

# 构建并训练朴素贝叶斯模型
# 对于文本数据，MultinomialNB通常比GaussianNB表现更好
# 这里同时展示两种模型
print("\n训练朴素贝叶斯模型...")

# 1. 高斯朴素贝叶斯（假设特征服从高斯分布）
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

# 2. 多项式朴素贝叶斯（适用于离散特征，如词频）
mnb = MultinomialNB()
mnb.fit(X_train, y_train)  # 虽然输入是TF-IDF，但多项式NB仍可使用
y_pred_mnb = mnb.predict(X_test)

# 输出评估结果
print("\n高斯朴素贝叶斯在测试集上的性能：")
print(f"准确率: {accuracy_score(y_test, y_pred_gnb):.4f}")

print("\n多项式朴素贝叶斯在测试集上的性能：")
print(f"准确率: {accuracy_score(y_test, y_pred_mnb):.4f}")
print("\n混淆矩阵：")
print(confusion_matrix(y_test, y_pred_mnb))
print("\n分类报告：")
print(classification_report(y_test, y_pred_mnb, target_names=class_names))

# 提取特征重要性（每个类别的高概率特征）
feature_names = vectorizer.get_feature_names_out()
top_n = 10  # 每个类别取前10个重要特征

print("\n每个类别的重要特征（高频词）：")
for i in range(len(class_names)):
    # 多项式朴素贝叶斯的特征重要性可以通过系数表示
    log_prob = mnb.feature_log_prob_[i]
    top_indices = np.argsort(log_prob)[-top_n:][::-1]
    top_features = [feature_names[j] for j in top_indices]
    print(f"{class_names[i]}: {', '.join(top_features)}")

# 使用SVD降维到2D（文本数据常用的降维方法），可视化分类结果
svd = TruncatedSVD(n_components=2, random_state=42)
X_train_svd = svd.fit_transform(X_train)
X_test_svd = svd.transform(X_test)

# 绘制决策边界（使用多项式朴素贝叶斯）
h = 0.5  # 网格步长
x_min, x_max = X_train_svd[:, 0].min() - 1, X_train_svd[:, 0].max() + 1
y_min, y_max = X_train_svd[:, 1].min() - 1, X_train_svd[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 创建一个临时数据集用于决策边界绘制
grid_points = np.c_[xx.ravel(), yy.ravel()]
# 将SVD空间的点映射回原始特征空间（近似）
grid_points_orig = np.dot(grid_points, svd.components_) + svd.mean_
# 确保非负（因为TF-IDF值非负）
grid_points_orig = np.maximum(grid_points_orig, 0)

# 预测网格点类别
Z = mnb.predict(grid_points_orig)
Z = Z.reshape(xx.shape)

# 绘制决策边界和样本点
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
scatter = plt.scatter(X_test_svd[:, 0], X_test_svd[:, 1], c=y_test, 
                     cmap=plt.cm.coolwarm, edgecolors='k', label='真实标签')

# 添加图例
handles, labels = scatter.legend_elements()
legend1 = plt.legend(handles, class_names, loc="upper right", title="类别")
plt.gca().add_artist(legend1)

plt.xlabel(f'SVD特征1 (解释方差比: {svd.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'SVD特征2 (解释方差比: {svd.explained_variance_ratio_[1]:.2%})')
plt.title('多项式朴素贝叶斯决策边界 - 测试集')
plt.grid(alpha=0.3)
plt.show()

# 对比两种朴素贝叶斯模型的准确率
plt.figure(figsize=(8, 5))
models = ['高斯朴素贝叶斯', '多项式朴素贝叶斯']
accuracies = [
    accuracy_score(y_test, y_pred_gnb),
    accuracy_score(y_test, y_pred_mnb)
]
plt.bar(models, accuracies, color=['lightblue', 'lightgreen'])
plt.ylim(0, 1.0)
plt.ylabel('测试集准确率')
plt.title('不同朴素贝叶斯模型的性能对比')
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.grid(axis='y', alpha=0.3)
plt.show()
    