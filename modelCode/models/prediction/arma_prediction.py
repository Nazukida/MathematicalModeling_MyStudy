#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ARMA时间序列预测模型案例说明：
本案例使用ARMA(自回归移动平均)模型预测某地区未来5个月的月度降雨量，
已知过去36个月的降雨数据（单位：mm）。ARMA(p,q)模型结合了
自回归模型AR(p)和移动平均模型MA(q)的优点，适用于平稳时间序列的预测。
其中p是自回归项数，q是移动平均项数，通常通过AIC准则确定最优阶数。
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# 1. 准备数据
# 生成模拟的36个月降雨量数据（单位：mm）
np.random.seed(42)  # 设置随机种子，确保结果可复现
months = np.arange(36)
# 生成带有季节性和随机波动的降雨量数据
rainfall = 50 + 10 * np.sin(months * np.pi / 6) + np.random.normal(0, 8, 36)
rainfall = np.maximum(rainfall, 5)  # 确保降雨量不为负

# 2. 平稳性检验（ADF检验）
result = adfuller(rainfall)
print(f'ADF检验结果：统计量={result[0]:.4f}, p值={result[1]:.4f}')
print(f'临界值: {result[4]}')
is_stationary = result[1] < 0.05
print(f'序列{"" if is_stationary else "不"}平稳')

# 如果序列不平稳，进行差分处理（此处假设序列已平稳）
d = 0  # 差分阶数
if not is_stationary:
    rainfall = np.diff(rainfall)
    d = 1
    print("已对序列进行1阶差分使其平稳")

# 3. 确定ARMA模型阶数（通过ACF和PACF图直观判断）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(rainfall, lags=12, ax=ax1)  # 自相关图，用于确定MA(q)的q值
plot_pacf(rainfall, lags=12, ax=ax2)  # 偏自相关图，用于确定AR(p)的p值
plt.show()

# 根据ACF和PACF图，选择ARMA(2,1)模型（p=2, q=1）
p, q = 2, 1

# 4. 拟合ARMA模型
model = ARIMA(rainfall, order=(p, d, q))
model_fit = model.fit()

# 输出模型摘要
print("\nARMA模型拟合摘要：")
print(model_fit.summary())

# 5. 计算历史数据拟合值
y_hat = model_fit.fittedvalues

# 6. 预测未来5个月数据
forecast_num = 5  # 预测未来5个月
forecast_result = model_fit.get_forecast(steps=forecast_num)
future_y = forecast_result.predicted_mean  # 预测值
conf_int = forecast_result.conf_int()  # 置信区间

# 7. 计算预测误差（均方根误差RMSE）
rmse = np.sqrt(np.mean((rainfall[d:] - y_hat[d:]) **2))
print(f"\n模型均方根误差RMSE: {rmse:.2f}")

# 8. 输出结果
print("\n未来5个月降雨量预测结果：")
for i in range(forecast_num):
    print(f"第{len(rainfall)+i+1}个月: 预测值={future_y[i]:.2f}mm, "
          f"95%置信区间=[{conf_int.iloc[i,0]:.2f}, {conf_int.iloc[i,1]:.2f}]")

# 9. 可视化结果
plt.figure(figsize=(12, 6))
# 绘制历史数据
plt.plot(range(1, len(rainfall)+1), rainfall, 'bo-', label='实际降雨量')
# 绘制拟合数据
plt.plot(range(1+d, len(rainfall)+1), y_hat[d:], 'r--', label='拟合降雨量')
# 绘制预测数据
plt.plot(range(len(rainfall)+1, len(rainfall)+1+forecast_num), 
         future_y, 'g*-', label='预测降雨量')
# 绘制置信区间
plt.fill_between(range(len(rainfall)+1, len(rainfall)+1+forecast_num),
                 conf_int.iloc[:,0], conf_int.iloc[:,1], color='green', alpha=0.2)
plt.xlabel('月份')
plt.ylabel('降雨量（mm）')
plt.title(f'ARMA({p},{d},{q})时间序列预测')
plt.legend()
plt.grid(True)
plt.show()
