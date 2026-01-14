#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高斯回归预测模型案例说明：
本案例使用高斯过程回归(GPR)预测某化学反应的产物浓度，
基于反应温度和反应时间两个特征。高斯过程是一种非参数模型，
能够捕捉数据中的非线性模式，并提供预测的不确定性估计。
本案例使用径向基函数(RBF)作为核函数，适用于具有平滑变化特性的数据。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. 准备数据
# 生成模拟数据：反应温度(℃)、反应时间(小时)与产物浓度(%)的关系
np.random.seed(42)  # 设置随机种子，确保结果可复现

# 特征数据：[反应温度(℃), 反应时间(小时)]
X = np.array([
    [50, 1], [50, 2], [50, 3], [50, 4],
    [60, 1], [60, 2], [60, 3], [60, 4],
    [70, 1], [70, 2], [70, 3], [70, 4],
    [80, 1], [80, 2], [80, 3], [80, 4]
])

# 目标数据：产物浓度(%)，与特征呈非线性关系
y = np.array([
    22.3, 35.6, 45.2, 50.1,
    30.5, 48.2, 60.3, 65.8,
    45.8, 62.5, 72.1, 76.5,
    55.2, 70.3, 80.1, 83.6
])

# 2. 数据预处理：标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 3. 划分训练集和测试集
train_indices = [0, 1, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15]  # 训练集索引
test_indices = [2, 5, 9, 12]                              # 测试集索引
X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
y_train, y_test = y_scaled[train_indices], y_scaled[test_indices]

# 4. 定义并训练高斯过程回归模型
# 定义核函数：常数核 * 径向基函数(RBF)
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

# 创建高斯过程回归模型
gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=0.01,  # 噪声项
    n_restarts_optimizer=10  # 多次重启优化器以找到最优核参数
)

# 训练模型
gpr.fit(X_train, y_train)

# 输出优化后的核参数
print(f"优化后的核参数: {gpr.kernel_}")
print(f"对数边际似然: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}")

# 5. 模型预测
# 对训练集和测试集进行预测，同时获取标准差
y_train_pred, y_train_std = gpr.predict(X_train, return_std=True)
y_test_pred, y_test_std = gpr.predict(X_test, return_std=True)

# 反标准化预测结果
y_train_pred = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
y_test_pred = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 6. 模型评估
# 计算训练集和测试集的性能指标
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
train_r2 = r2_score(y_train_actual, y_train_pred)
test_r2 = r2_score(y_test_actual, y_test_pred)

# 7. 预测新数据点
# 新的反应条件
future_X = np.array([[55, 2.5], [65, 3.5], [75, 2], [85, 3]])
future_X_scaled = scaler_X.transform(future_X)
future_y_pred, future_y_std = gpr.predict(future_X_scaled, return_std=True)
future_y_pred = scaler_y.inverse_transform(future_y_pred.reshape(-1, 1)).flatten()
future_y_std_actual = future_y_std * np.std(y)  # 反标准化标准差

# 8. 输出结果
print("\n高斯过程回归预测模型结果：")
print(f"训练集RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
print(f"测试集RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

print("\n训练集预测结果：")
for i in range(len(y_train_actual)):
    print(f"实际值: {y_train_actual[i]:.2f}%, 预测值: {y_train_pred[i]:.2f}%, "
          f"标准差: {y_train_std[i]*np.std(y):.2f}%")

print("\n测试集预测结果：")
for i in range(len(y_test_actual)):
    print(f"实际值: {y_test_actual[i]:.2f}%, 预测值: {y_test_pred[i]:.2f}%, "
          f"标准差: {y_test_std[i]*np.std(y):.2f}%")

print("\n新反应条件下的产物浓度预测：")
for i in range(len(future_X)):
    print(f"温度{future_X[i,0]}℃, 时间{future_X[i,1]}小时: "
          f"预测浓度{future_y_pred[i]:.2f}%, 标准差{future_y_std_actual[i]:.2f}%")

# 9. 可视化结果（选择温度为特征，固定时间为2小时）
plt.figure(figsize=(10, 6))
# 创建温度序列
temp_range = np.linspace(45, 85, 100)
time_fixed = 2  # 固定时间为2小时
X_plot = np.column_stack((temp_range, np.full_like(temp_range, time_fixed)))
X_plot_scaled = scaler_X.transform(X_plot)

# 预测
y_plot_pred, y_plot_std = gpr.predict(X_plot_scaled, return_std=True)
y_plot_pred = scaler_y.inverse_transform(y_plot_pred.reshape(-1, 1)).flatten()
y_plot_std_actual = y_plot_std * np.std(y)

# 绘制结果
plt.plot(temp_range, y_plot_pred, 'b-', label='预测均值')
plt.fill_between(temp_range, 
                 y_plot_pred - 1.96 * y_plot_std_actual,  # 95%置信区间下限
                 y_plot_pred + 1.96 * y_plot_std_actual,  # 95%置信区间上限
                 alpha=0.2, color='blue', label='95%置信区间')

# 绘制实际数据点（时间=2小时）
mask = X[:, 1] == 2
plt.scatter(X[mask, 0], y[mask], c='red', s=50, label='实际数据点')

plt.xlabel('反应温度 (℃)')
plt.ylabel('产物浓度 (%)')
plt.title('固定反应时间为2小时的产物浓度预测')
plt.legend()
plt.grid(True)
plt.show()
