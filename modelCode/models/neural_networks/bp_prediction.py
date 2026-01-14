#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BP神经网络预测模型案例说明：
本案例使用BP神经网络预测某商品的销售量，基于过去12个月的
广告投入、促销活动强度和节假日因素等特征数据。BP神经网络
是一种多层前馈神经网络，通过反向传播算法调整权重，适用于
复杂非线性关系的预测问题。本案例使用1个隐藏层(10个神经元)，
输入层3个神经元(对应3个特征)，输出层1个神经元(预测销售量)。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 准备数据
# 生成模拟数据：12个月的特征和销售量
np.random.seed(42)  # 设置随机种子，确保结果可复现

# 特征数据：[广告投入(万元), 促销强度(0-10), 节假日数量]
X = np.array([
    [5.2, 7, 2], [6.3, 5, 1], [4.8, 8, 3], [7.1, 6, 2],
    [5.9, 9, 1], [8.2, 7, 2], [6.8, 6, 3], [9.1, 8, 2],
    [7.5, 10, 1], [8.9, 7, 2], [9.5, 9, 3], [10.2, 8, 2]
])

# 目标数据：销售量(千件)，与特征呈非线性关系
y = np.array([
    12.5, 11.8, 13.2, 14.5, 
    13.8, 16.2, 15.1, 17.5, 
    16.8, 18.2, 19.5, 20.3
])

# 2. 数据预处理：归一化到[0,1]范围
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# 4. 创建并训练BP神经网络模型
# 定义模型：1个隐藏层(10个神经元)，使用ReLU激活函数，Adam优化器
model = MLPRegressor(
    hidden_layer_sizes=(10,),  # 隐藏层结构
    activation='relu',         # 激活函数
    solver='adam',             # 优化器
    max_iter=1000,             # 最大迭代次数
    random_state=42,           # 随机种子
    verbose=False              # 不输出训练过程
)

# 训练模型
model.fit(X_train, y_train)

# 5. 模型预测
# 对训练集和测试集进行预测
y_train_pred_scaled = model.predict(X_train)
y_test_pred_scaled = model.predict(X_test)

# 反归一化预测结果
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 6. 模型评估
# 计算训练集和测试集的性能指标
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
train_r2 = r2_score(y_train_actual, y_train_pred)
test_r2 = r2_score(y_test_actual, y_test_pred)

# 7. 预测未来数据
# 未来3个月的特征数据
future_X = np.array([
    [11.0, 9, 2], [11.5, 8, 1], [12.0, 10, 3]
])
future_X_scaled = scaler_X.transform(future_X)
future_y_scaled = model.predict(future_X_scaled)
future_y = scaler_y.inverse_transform(future_y_scaled.reshape(-1, 1)).flatten()

# 8. 输出结果
print("BP神经网络预测模型结果：")
print(f"模型结构：输入层{X.shape[1]}个神经元，隐藏层10个神经元，输出层1个神经元")
print(f"训练集RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
print(f"测试集RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

print("\n训练集预测结果：")
for i in range(len(y_train_actual)):
    print(f"实际值: {y_train_actual[i]:.2f}, 预测值: {y_train_pred[i]:.2f}, 误差: {y_train_actual[i]-y_train_pred[i]:.2f}")

print("\n测试集预测结果：")
for i in range(len(y_test_actual)):
    print(f"实际值: {y_test_actual[i]:.2f}, 预测值: {y_test_pred[i]:.2f}, 误差: {y_test_actual[i]-y_test_pred[i]:.2f}")

print("\n未来3个月销售量预测结果：")
for i in range(len(future_y)):
    print(f"第{i+1}个月: 广告投入{future_X[i,0]}万元, 促销强度{future_X[i,1]}, 节假日{future_X[i,2]}天, 预测销售量{future_y[i]:.2f}千件")

# 9. 可视化结果
plt.figure(figsize=(12, 6))
# 绘制训练集结果
plt.plot(range(len(y_train_actual)), y_train_actual, 'bo-', label='训练集实际值')
plt.plot(range(len(y_train_pred)), y_train_pred, 'r--', label='训练集预测值')
# 绘制测试集结果（偏移显示）
test_offset = len(y_train_actual)
plt.plot(range(test_offset, test_offset+len(y_test_actual)), y_test_actual, 'go-', label='测试集实际值')
plt.plot(range(test_offset, test_offset+len(y_test_pred)), y_test_pred, 'm--', label='测试集预测值')
# 绘制未来预测结果（偏移显示）
future_offset = test_offset + len(y_test_actual)
plt.plot(range(future_offset, future_offset+len(future_y)), future_y, 'c*-', label='未来预测值')

plt.xlabel('样本索引')
plt.ylabel('销售量（千件）')
plt.title('BP神经网络销售量预测')
plt.legend()
plt.grid(True)
plt.show()
