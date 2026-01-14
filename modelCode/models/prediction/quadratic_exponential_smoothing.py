#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
二次指数平滑预测模型案例说明：
本案例使用二次指数平滑法预测某产品未来4个月的销售量，
已知过去12个月的销售数据（单位：件）。二次指数平滑适用于
具有线性趋势的数据预测，在一次指数平滑基础上增加了对趋势的平滑处理。
平滑系数α的取值范围为0-1，通常通过试算选择使预测误差最小的值。
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据
# 过去12个月的销售量（单位：件）
y = np.array([120, 135, 142, 150, 165, 178, 190, 205, 218, 230, 245, 260])
n = len(y)  # 数据长度

# 2. 设置平滑系数
alpha = 0.3  # 平滑系数，可根据实际情况调整

# 3. 初始化一次和二次指数平滑值
s1 = np.zeros(n)  # 一次指数平滑值
s2 = np.zeros(n)  # 二次指数平滑值

s1[0] = y[0]  # 第一个一次平滑值等于原始数据
s2[0] = y[0]  # 第一个二次平滑值等于原始数据

# 4. 计算一次和二次指数平滑值
for i in range(1, n):
    # 一次指数平滑公式：s1[i] = α*y[i] + (1-α)*s1[i-1]
    s1[i] = alpha * y[i] + (1 - alpha) * s1[i-1]
    # 二次指数平滑公式：s2[i] = α*s1[i] + (1-α)*s2[i-1]
    s2[i] = alpha * s1[i] + (1 - alpha) * s2[i-1]

# 5. 计算平滑系数和预测模型
# 估计当前水平和趋势
a = 2 * s1[-1] - s2[-1]  # 截距项
b = (alpha / (1 - alpha)) * (s1[-1] - s2[-1])  # 趋势项

# 6. 计算历史数据拟合值
y_hat = np.zeros(n)
for t in range(n):
    # 第t+1期的拟合值
    y_hat[t] = a - b * (n - t - 1)

# 7. 预测未来4个月数据
forecast_num = 4  # 预测未来4个月
future_t = np.arange(1, forecast_num + 1)  # 预测步数
future_y = a + b * future_t  # 预测公式：y_hat(n+k) = a + b*k

# 8. 计算预测误差（均方根误差RMSE）
rmse = np.sqrt(np.mean((y - y_hat) **2))

# 9. 输出结果
print("二次指数平滑预测模型结果：")
print(f"平滑系数α = {alpha}")
print(f"模型参数：a = {a:.2f}, b = {b:.2f}")
print(f"均方根误差RMSE = {rmse:.2f}")

print("\n历史数据拟合结果：")
for i in range(n):
    print(f"第{i+1}月: 实际值={y[i]}, 拟合值={y_hat[i]:.2f}, 误差={y[i]-y_hat[i]:.2f}")

print("\n未来4个月预测结果：")
for i in range(forecast_num):
    print(f"第{n+i+1}月预测值: {future_y[i]:.2f}")

# 10. 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(range(1, n+1), y, 'bo-', label='实际销售量')
plt.plot(range(1, n+1), y_hat, 'r--', label='拟合销售量')
plt.plot(range(n+1, n+1+forecast_num), future_y, 'g*-', label='预测销售量')
plt.xlabel('月份')
plt.ylabel('销售量（件）')
plt.title(f'二次指数平滑预测 (α={alpha})')
plt.legend()
plt.grid(True)
plt.show()
