#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BP神经网络分类模型案例说明：
BP（反向传播）神经网络是一种多层前馈神经网络，通过反向传播算法调整网络权重来最小化预测误差。
它包含输入层、隐藏层和输出层，各层神经元之间全连接。

算法步骤：
1. 初始化网络权重和偏置
2. 前向传播：计算输入经过各层后的输出
3. 计算损失：比较预测输出与真实标签的差异
4. 反向传播：计算损失对各层权重的梯度
5. 更新权重：使用梯度下降法调整权重以减小损失
6. 重复步骤2-5直到收敛或达到最大迭代次数

本案例使用乳腺癌数据集，构建一个含1个隐藏层的BP神经网络进行二分类，
预测肿瘤是良性还是恶性，并评估模型性能。
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA

# 加载数据集
cancer = load_breast_cancer()
X = cancer.data  # 特征数据（30个特征）
y = cancer.target  # 标签（0=恶性，1=良性）
feature_names = cancer.feature_names
class_names = cancer.target_names

# 数据划分：训练集和测试集（7:3）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# 数据标准化（神经网络对数据尺度敏感）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建BP神经网络模型
# 隐藏层含32个神经元，使用ReLU激活函数，输出层使用sigmoid激活函数（二分类）
bp_model = MLPClassifier(
    hidden_layer_sizes=(32,),  # 隐藏层结构
    activation='relu',         # 隐藏层激活函数
    solver='adam',             # 优化器
    learning_rate_init=0.001,  # 初始学习率
    max_iter=300,              # 最大迭代次数
    random_state=42,           # 随机种子
    verbose=False              # 不打印训练过程
)

# 训练模型
print("训练BP神经网络...")
bp_model.fit(X_train_scaled, y_train)

# 预测
y_pred = bp_model.predict(X_test_scaled)
y_pred_proba = bp_model.predict_proba(X_test_scaled)[:, 1]  # 正类的预测概率

# 输出评估结果
print("\nBP神经网络在测试集上的性能：")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n混淆矩阵：")
print(confusion_matrix(y_test, y_pred))
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=class_names))

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(bp_model.loss_curve_)
plt.xlabel('迭代次数')
plt.ylabel('训练损失')
plt.title('BP神经网络训练损失曲线')
plt.grid(alpha=0.3)
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
bp_pca = MLPClassifier(
    hidden_layer_sizes=(32,), 
    activation='relu', 
    solver='adam',
    max_iter=300, 
    random_state=42
)
bp_pca.fit(X_train_pca, y_train)
Z = bp_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# 可视化结果
plt.figure(figsize=(15, 6))

# 1. 决策边界和样本点
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=plt.cm.coolwarm, 
            edgecolors='k', marker='s', s=80, label='真实标签')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap=plt.cm.coolwarm, 
            edgecolors='y', marker='o', s=40, label='预测标签')
plt.xlabel(f'PCA特征1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PCA特征2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('BP神经网络决策边界 - 测试集')
plt.legend()
plt.grid(alpha=0.3)

# 2. ROC曲线
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (面积 = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 随机猜测的基准线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率 (FPR)')
plt.ylabel('真正例率 (TPR)')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
    