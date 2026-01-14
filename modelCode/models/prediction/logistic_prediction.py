#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logistic预测模型案例说明：
本案例使用Logistic模型预测某地区未来5年的人口数量，
已知过去10年的人口数据（单位：万人），Logistic模型适用于
具有饱和增长特性的预测问题，如人口增长、产品扩散等。
模型公式：y(t) = K / (1 + (K/y0 - 1) * exp(-r*t))
其中K为环境承载力，r为增长率，y0为初始值。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. 准备数据
# 年份（相对值，0表示起始年份）
t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# 对应年份的人口数量（单位：万人）
y = np.array([32.5, 35.6, 39.0, 42.8, 46.9, 51.3, 55.9, 60.5, 65.0, 69.2])

# 2. 定义Logistic模型
def logistic_func(t, K, r, y0):
    """Logistic增长模型公式"""
    return K / (1 + (K / y0 - 1) * np.exp(-r * t))

# 3. 拟合模型参数
# 初始参数估计（K: 环境承载力，r: 增长率，y0: 初始值）
initial_guess = [100, 0.1, 30]
# 曲线拟合
params, covariance = curve_fit(logistic_func, t, y, p0=initial_guess, maxfev=10000)
K, r, y0 = params  # 提取拟合得到的参数

# 4. 预测未来5年数据
future_t = np.arange(10, 15)  # 未来5年的时间点
future_y = logistic_func(future_t, K, r, y0)  # 预测值

# 5. 计算拟合优度R²
y_pred = logistic_func(t, K, r, y0)
ss_total = np.sum((y - np.mean(y)) **2)
ss_residual = np.sum((y - y_pred)** 2)
r_squared = 1 - (ss_residual / ss_total)

# 6. 输出结果
print("Logistic预测模型结果：")
print(f"模型参数：K={K:.2f}, r={r:.4f}, y0={y0:.2f}")
print(f"拟合优度R²：{r_squared:.4f}")
print("\n历史数据拟合值：")
for i in range(len(t)):
    print(f"年份{t[i]}: 实际值={y[i]}, 拟合值={y_pred[i]:.2f}")
print("\n未来5年预测值：")
for i in range(len(future_t)):
    print(f"年份{future_t[i]}: 预测值={future_y[i]:.2f}")

# 7. 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(t, y, 'bo', label='实际数据')
plt.plot(t, y_pred, 'r-', label='拟合曲线')
plt.plot(future_t, future_y, 'g--', label='预测曲线')
plt.xlabel('年份（相对值）')
plt.ylabel('人口数量（万人）')
plt.title('Logistic模型人口预测')
plt.legend()
plt.grid(True)
plt.show()
