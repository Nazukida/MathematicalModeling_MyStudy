#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
拉格朗日插值法案例说明：
本案例使用拉格朗日插值法对已知数据点进行插值，估算未知点的值。
拉格朗日插值是一种多项式插值方法，通过构造一组基函数来逼近原函数。
示例中使用sin(x)函数的部分采样点进行插值，展示插值效果。
"""

import numpy as np
import matplotlib.pyplot as plt

# 生成样本数据
x_sample = np.array([0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi])  # 采样点
y_sample = np.sin(x_sample)  # 对应函数值

# 定义插值点
x_interp = np.linspace(0, np.pi, 100)  # 需要插值的点
y_interp = np.zeros_like(x_interp)     # 存储插值结果

# 拉格朗日插值核心计算
n = len(x_sample)  # 样本点数量
for k in range(len(x_interp)):
    x = x_interp[k]
    y = 0.0
    for i in range(n):
        # 计算拉格朗日基函数L_i(x)
        L = 1.0
        for j in range(n):
            if j != i:
                L *= (x - x_sample[j]) / (x_sample[i] - x_sample[j])
        # 累加得到插值结果
        y += y_sample[i] * L
    y_interp[k] = y

# 计算真实值用于对比
y_true = np.sin(x_interp)

# 输出部分插值结果
print("拉格朗日插值法结果（部分）：")
print(f"{'x值':<10} {'插值结果':<15} {'真实值':<15} {'误差':<10}")
print("-" * 50)
for i in [0, 20, 40, 60, 80, 99]:
    x_val = x_interp[i]
    interp_val = y_interp[i]
    true_val = y_true[i]
    error = abs(interp_val - true_val)
    print(f"{x_val:<10.4f} {interp_val:<15.6f} {true_val:<15.6f} {error:<10.6e}")

# 可视化插值结果
plt.figure(figsize=(10, 6))
plt.plot(x_interp, y_true, 'b-', label='真实函数 sin(x)')
plt.plot(x_interp, y_interp, 'r--', label='拉格朗日插值')
plt.scatter(x_sample, y_sample, color='green', s=50, label='样本点')
plt.xlabel('x')
plt.ylabel('y')
plt.title('拉格朗日插值法示例')
plt.legend()
plt.grid(True)
plt.show()
    