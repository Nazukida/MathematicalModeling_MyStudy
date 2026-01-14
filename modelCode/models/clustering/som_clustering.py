#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOM自组织神经网络案例说明：
自组织映射（SOM）是一种无监督的人工神经网络，通过竞争学习将高维数据映射到低维（通常是2D）空间，
同时保留数据的拓扑结构。SOM由网格状排列的神经元组成，每个神经元有一个与输入数据维度相同的权重向量。

算法步骤：
1. 初始化神经元权重向量
2. 从数据集中随机选择一个样本
3. 找到与样本最相似的神经元（最佳匹配单元，BMU）
4. 更新BMU及其邻域神经元的权重，使其更接近样本
5. 逐渐减小邻域大小和学习率
6. 重复步骤2-5直到收敛

本案例使用鸢尾花数据集（4维特征），通过SOM映射到20×20的二维网格，
可视化聚类结果和数据的拓扑结构。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

# 加载数据集
iris = load_iris()
X = iris.data  # 特征数据
y_true = iris.target  # 真实标签（仅用于对比）
feature_names = iris.feature_names

# 数据归一化（SOM对数据尺度敏感，通常归一化到[0,1]）
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# SOM参数设置
grid_size = (10, 10)  # 神经元网格大小
input_dim = X_scaled.shape[1]  # 输入维度（4）
sigma = 1.0  # 初始邻域半径
learning_rate = 0.5  # 初始学习率
num_iterations = 1000  # 迭代次数

# 初始化并训练SOM
som = MiniSom(grid_size[0], grid_size[1], input_dim, 
              sigma=sigma, learning_rate=learning_rate, 
              neighborhood_function='gaussian', random_seed=42)

# 随机初始化权重
som.random_weights_init(X_scaled)
print(f"训练SOM神经网络（{grid_size[0]}×{grid_size[1]}网格）...")
# 训练SOM
som.train_random(X_scaled, num_iterations)

# 获取每个样本的最佳匹配单元（BMU）
bmus = np.array([som.winner(x) for x in X_scaled])  # 每个样本对应的BMU坐标

# 可视化SOM聚类结果
plt.figure(figsize=(12, 10))

# 1. SOM节点激活频率热图
activation_map = np.zeros(grid_size)
for x, y in bmus:
    activation_map[x, y] += 1

plt.subplot(2, 2, 1)
plt.imshow(activation_map, cmap='viridis', origin='lower')
plt.colorbar(label='激活频率')
plt.title('SOM节点激活频率')
plt.xlabel('神经元列索引')
plt.ylabel('神经元行索引')

# 2. 样本在SOM上的分布（按真实标签）
plt.subplot(2, 2, 2)
colors = ['navy', 'turquoise', 'darkorange']
for i in range(3):
    idx = y_true == i
    plt.scatter(bmus[idx, 1], bmus[idx, 0],  # 注意坐标顺序（列，行）
                c=colors[i], alpha=0.7, label=iris.target_names[i], s=50)
plt.title('样本在SOM上的分布（按真实标签）')
plt.xlabel('神经元列索引')
plt.ylabel('神经元行索引')
plt.xlim(-0.5, grid_size[1]-0.5)
plt.ylim(-0.5, grid_size[0]-0.5)
plt.xticks(range(grid_size[1]))
plt.yticks(range(grid_size[0]))
plt.grid(True)
plt.legend()

# 3. SOM权重向量的U矩阵（反映节点间相似度）
u_matrix = som.distance_map()  # U矩阵：每个节点与周围节点的平均距离

plt.subplot(2, 2, 3)
plt.imshow(u_matrix, cmap='bone', origin='lower')
plt.colorbar(label='平均距离')
plt.title('SOM的U矩阵（反映拓扑结构）')
plt.xlabel('神经元列索引')
plt.ylabel('神经元行索引')

# 4. 每个特征的权重分布
plt.subplot(2, 2, 4)
# 选择第一个特征的权重分布进行可视化
feature_idx = 0
weights_map = som.get_weights()[:, :, feature_idx]
plt.imshow(weights_map, cmap='coolwarm', origin='lower')
plt.colorbar(label='权重值')
plt.title(f'特征 "{feature_names[feature_idx]}" 的权重分布')
plt.xlabel('神经元列索引')
plt.ylabel('神经元行索引')

plt.tight_layout()
plt.show()

# 展示所有特征的权重分布
plt.figure(figsize=(15, 10))
for i in range(input_dim):
    plt.subplot(2, 2, i+1)
    weights_map = som.get_weights()[:, :, i]
    plt.imshow(weights_map, cmap='coolwarm', origin='lower')
    plt.colorbar(label='权重值')
    plt.title(f'特征 "{feature_names[i]}" 的权重分布')
    plt.xlabel('神经元列索引')
    plt.ylabel('神经元行索引')

plt.tight_layout()
plt.show()
    