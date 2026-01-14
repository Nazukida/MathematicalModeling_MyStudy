#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
季节指数预测模型案例说明：
本案例使用季节指数法预测某景区未来4个季度的游客数量，
已知过去4年（16个季度）的游客数据（单位：千人）。
季节指数法适用于具有明显季节性波动的数据，通过分离
季节因素和趋势因素，先预测趋势，再用季节指数调整，
得到最终预测结果。
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据
# 过去4年（16个季度）的游客数量（单位：千人）
# 数据呈现季节性：夏季（第2季度）和秋季（第3季度）为旺季
visitors = np.array([
    32, 65, 58, 28,  # 第1年
    35, 70, 62, 30,  # 第2年
    38, 75, 68, 33,  # 第3年
    42, 80, 72, 36   # 第4年
])
n_years = 4  # 年数
n_seasons = 4  # 每年的季节数（季度）
n = len(visitors)  # 数据总长度

# 2. 计算各季节的平均值
seasonal_data = visitors.reshape(n_years, n_seasons)  # 按年和季节整理数据
seasonal_mean = np.mean(seasonal_data, axis=0)  # 各季节的平均值

# 3. 计算总平均值
total_mean = np.mean(visitors)  # 所有数据的总平均值

# 4. 计算季节指数（季节平均值/总平均值）
seasonal_index = seasonal_mean / total_mean

# 5. 计算 deseasonalized 数据（消除季节影响的数据）
deseasonalized = np.zeros(n)
for i in range(n):
    season = i % n_seasons  # 当前数据点所属的季节
    deseasonalized[i] = visitors[i] / seasonal_index[season]

# 6. 拟合趋势线（线性趋势）
t = np.arange(1, n+1)  # 时间变量（1到n）
# 线性回归：y = a + b*t
b, a = np.polyfit(t, deseasonalized, 1)  # 计算斜率b和截距a

# 7. 预测未来趋势值
forecast_num = 4  # 预测未来4个季度（1年）
future_t = np.arange(n+1, n+1+forecast_num)  # 未来时间点
future_trend = a + b * future_t  # 未来趋势预测值

# 8. 用季节指数调整趋势预测值，得到最终预测值
future_seasons = np.arange(n % n_seasons, n % n_seasons + forecast_num) % n_seasons
future_visitors = future_trend * seasonal_index[future_seasons]

# 9. 计算历史数据拟合值
historical_trend = a + b * t  # 历史趋势值
historical_fitted = historical_trend * seasonal_index[[i % n_seasons for i in range(n)]]

# 10. 计算预测误差（均方根误差RMSE）
rmse = np.sqrt(np.mean((visitors - historical_fitted) **2))

# 11. 输出结果
print("季节指数预测模型结果：")
print(f"各季节指数: {seasonal_index.round(4)}")
print(f"趋势方程: y = {a:.4f} + {b:.4f}*t")
print(f"均方根误差RMSE: {rmse:.2f}")

print("\n历史数据拟合结果：")
for i in range(n_years):
    for j in range(n_seasons):
        idx = i * n_seasons + j
        print(f"第{i+1}年第{j+1}季度: 实际值={visitors[idx]}, 拟合值={historical_fitted[idx]:.2f}")

print("\n未来4个季度预测结果：")
for i in range(forecast_num):
    print(f"第{n_years+1}年第{i+1}季度预测值: {future_visitors[i]:.2f}千人")

# 12. 可视化结果
plt.figure(figsize=(12, 6))
# 绘制历史数据
plt.plot(range(1, n+1), visitors, 'bo-', label='实际游客数量')
# 绘制拟合数据
plt.plot(range(1, n+1), historical_fitted, 'r--', label='拟合游客数量')
# 绘制预测数据
plt.plot(range(n+1, n+1+forecast_num), future_visitors, 'g*-', label='预测游客数量')
# 添加网格线和标注
plt.xticks(np.arange(1, n+1+forecast_num, n_seasons), 
           [f'第{i+1}年' for i in range(n_years+1)])
plt.xlabel('时间')
plt.ylabel('游客数量（千人）')
plt.title('季节指数预测模型')
plt.legend()
plt.grid(True)
plt.show()
