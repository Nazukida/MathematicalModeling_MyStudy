#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
决策树分类模型案例说明：
决策树是一种基于树状结构进行决策的监督学习算法，通过一系列if-then规则对数据进行分类。
每个内部节点表示一个特征的判断，每个分支代表一个判断结果，每个叶节点代表一个类别。

算法步骤：
1. 选择最佳特征作为根节点，根据该特征的不同取值创建分支
2. 对每个分支递归地应用步骤1，选择最佳特征继续分裂
3. 当满足停止条件（如节点样本数小于阈值、树深度达到上限等）时停止分裂
4. 叶节点的类别为该节点中样本数最多的类别

本案例使用红酒质量数据集，构建决策树模型预测红酒质量等级，并可视化决策树结构，
分析特征重要性，对比不同树深度对模型性能的影响。
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# 加载数据集
wine = load_wine()
X = wine.data  # 特征数据
y = wine.target  # 标签（3个类别）
feature_names = wine.feature_names
class_names = wine.target_names

# 数据划分：训练集和测试集（7:3）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# 数据标准化（决策树对数据尺度不敏感，标准化不是必须的，但这里仍进行标准化以便后续可视化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 尝试不同的树深度，选择最优深度
max_depth_range = range(1, 11)
accuracy_scores = []

for depth in max_depth_range:
    # 创建并训练决策树模型
    dt_model = DecisionTreeClassifier(
        max_depth=depth,
        criterion='gini',  # 使用基尼不纯度
        random_state=42
    )
    dt_model.fit(X_train, y_train)  # 决策树不需要标准化数据
    
    # 预测并计算准确率
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print(f"树深度={depth}时，测试集准确率: {accuracy:.4f}")

# 找到最优树深度
best_depth = max_depth_range[np.argmax(accuracy_scores)]
print(f"\n最优树深度为: {best_depth}")

# 使用最优树深度构建最终模型
dt_best = DecisionTreeClassifier(
    max_depth=best_depth,
    criterion='gini',
    random_state=42
)
dt_best.fit(X_train, y_train)
y_pred_best = dt_best.predict(X_test)

# 输出评估结果
print("\n最优决策树模型在测试集上的性能：")
print(f"准确率: {accuracy_score(y_test, y_pred_best):.4f}")
print("\n混淆矩阵：")
print(confusion_matrix(y_test, y_pred_best))
print("\n分类报告：")
print(classification_report(y_test, y_pred_best, target_names=class_names))

# 输出特征重要性
feature_importance = pd.DataFrame({
    '特征': feature_names,
    '重要性': dt_best.feature_importances_
}).sort_values(by='重要性', ascending=False)
print("\n特征重要性排序：")
print(feature_importance.round(4))

# 可视化不同树深度的准确率
plt.figure(figsize=(10, 6))
plt.plot(max_depth_range, accuracy_scores, 'bo-')
plt.xlabel('树深度')
plt.ylabel('测试集准确率')
plt.title('不同树深度对决策树模型性能的影响')
plt.grid(alpha=0.3)
plt.axvline(x=best_depth, color='r', linestyle='--', label=f'最优深度={best_depth}')
plt.legend()
plt.show()

# 可视化决策树结构
plt.figure(figsize=(20, 10))
plot_tree(
    dt_best,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,  # 按类别填充颜色
    rounded=True,  # 圆角
    proportion=True,  # 显示样本比例
    precision=2  # 显示精度
)
plt.title(f'决策树结构 (深度={best_depth})')
plt.show()

# 可视化特征重要性
plt.figure(figsize=(12, 6))
plt.barh(feature_importance['特征'], feature_importance['重要性'], color='skyblue')
plt.xlabel('重要性')
plt.ylabel('特征')
plt.title('决策树特征重要性')
plt.grid(axis='x', alpha=0.3)
plt.gca().invert_yaxis()  # 重要性高的特征在上方
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
dt_pca = DecisionTreeClassifier(
    max_depth=best_depth,
    criterion='gini',
    random_state=42
)
dt_pca.fit(X_train_pca, y_train)
Z = dt_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界和样本点
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=plt.cm.coolwarm, 
            edgecolors='k', marker='s', s=80, label='真实标签')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_best, cmap=plt.cm.coolwarm, 
            edgecolors='y', marker='o', s=40, label='预测标签')
plt.xlabel(f'PCA特征1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PCA特征2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title(f'决策树决策边界 (深度={best_depth}) - 测试集')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
    