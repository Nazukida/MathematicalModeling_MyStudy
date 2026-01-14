#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T-SNE降维算法案例说明：
本案例使用t-SNE(t分布随机邻域嵌入)算法对高维数据进行降维。
t-SNE是一种非线性降维方法，特别适合高维数据的可视化，能够较好地保留数据的局部结构。
示例中使用MNIST手写数字数据集(784维)，通过t-SNE降维到2维空间进行可视化。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# 加载MNIST数据集（只使用部分数据加快计算）
print("加载MNIST数据集...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# 只使用1000个样本加快计算
n_samples = 1000
X_sample, y_sample = X[:n_samples], y[:n_samples]
y_sample = y_sample.astype(int)  # 转换为整数标签

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# 应用t-SNE降维到2维
print("运行t-SNE降维...")
tsne = TSNE(
    n_components=2,        # 降维到2维
    perplexity=30,         # 困惑度，通常在5-50之间
    learning_rate=200,     # 学习率
    n_iter=1000,           # 迭代次数
    random_state=42        # 随机种子，确保结果可复现
)
X_tsne = tsne.fit_transform(X_scaled)

# 输出t-SNE结果信息
print("\nt-SNE降维结果：")
print(f"原始数据维度: {X_sample.shape[1]} 维")
print(f"降维后数据维度: {X_tsne.shape[1]} 维")
print(f"使用参数: perplexity={tsne.perplexity}, learning_rate={tsne.learning_rate}, "
      f"n_iter={tsne.n_iter}")

# 可视化t-SNE降维结果
plt.figure(figsize=(12, 10))
# 使用不同颜色和标记区分不同数字
colors = plt.cm.rainbow(np.linspace(0, 1, 10))
markers = ['o', 's', '^', 'D', '*', 'P', 'X', 'd', 'p', 'H']

for digit in range(10):
    indices = y_sample == digit
    plt.scatter(
        X_tsne[indices, 0], 
        X_tsne[indices, 1], 
        c=[colors[digit]], 
        label=str(digit),
        marker=markers[digit],
        alpha=0.7,
        s=50
    )

plt.legend(title='数字', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('t-SNE降维：MNIST手写数字数据集(784维→2维)')
plt.xlabel('t-SNE特征1')
plt.ylabel('t-SNE特征2')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
    