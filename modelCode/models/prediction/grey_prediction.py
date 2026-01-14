#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
灰色预测模型案例说明：
本案例使用GM(1,1)灰色预测模型预测某企业未来3年的销售额，
已知过去6年的销售额数据（单位：万元）。灰色预测模型适用于
数据量少（通常4-10个数据）、信息不完全的预测问题，
不需要数据满足典型分布特征。
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据
# 原始数据序列（过去6年的销售额，单位：万元）
x0 = np.array([120, 135, 150, 170, 190, 215], dtype=np.float64)
n = len(x0)  # 数据长度

# 2. 累加生成（1-AGO）
x1 = np.cumsum(x0)  # 一次累加序列

# 3. 构造数据矩阵B和数据向量Y
B = np.zeros((n-1, 2))
Y = np.zeros((n-1, 1))

for i in range(n-1):
    B[i, 0] = -0.5 * (x1[i] + x1[i+1])  # 均值生成
    B[i, 1] = 1
    Y[i, 0] = x0[i+1]

# 4. 计算模型参数a和b（最小二乘法）
# 参数估计：(a, b)^T = (B^T B)^(-1) B^T Y
BT = B.T
params = np.dot(np.dot(np.linalg.inv(np.dot(BT, B)), BT), Y)
a = params[0, 0]  # 发展系数
b = params[1, 0]  # 灰作用量

# 5. 建立预测模型
# 时间响应函数：x1_hat(k+1) = (x0(1) - b/a) * exp(-a*k) + b/a
def predict(x0, a, b, k):
    """预测第k+1个值（k从0开始）"""
    x1_hat = (x0[0] - b/a) * np.exp(-a * k) + b/a
    if k == 0:
        return x1_hat  # 第一个预测值
    # 计算k+1时刻的一次累加预测值
    x1_hat_prev = (x0[0] - b/a) * np.exp(-a * (k-1)) + b/a
    return x1_hat - x1_hat_prev  # 累减得到原始序列预测值

# 6. 计算历史数据拟合值
x0_hat = np.zeros(n)
x0_hat[0] = x0[0]  # 第一个值等于原始值
for i in range(1, n):
    x0_hat[i] = predict(x0, a, b, i)

# 7. 预测未来3年数据
forecast_num = 3  # 预测未来3年
future_x0 = np.zeros(forecast_num)
for i in range(forecast_num):
    future_x0[i] = predict(x0, a, b, n + i)

# 8. 模型检验：计算后验差比C和小误差概率P
# 计算残差
epsilon = x0 - x0_hat
# 原始数据标准差
s1 = np.std(x0, ddof=1)
# 残差标准差
s2 = np.std(epsilon, ddof=1)
# 后验差比
C = s2 / s1
# 小误差概率
P = np.mean(np.abs(epsilon - np.mean(epsilon)) < 0.6745 * s1)

# 9. 输出结果
print("GM(1,1)灰色预测模型结果：")
print(f"模型参数：a={a:.6f}, b={b:.6f}")
print(f"后验差比C={C:.4f}, 小误差概率P={P:.4f}")
print(f"模型精度等级：{'好' if C < 0.35 and P > 0.95 else '合格' if C < 0.5 and P > 0.8 else '勉强' if C < 0.65 and P > 0.7 else '不合格'}")

print("\n历史数据拟合结果：")
for i in range(n):
    print(f"第{i+1}年: 实际值={x0[i]}, 拟合值={x0_hat[i]:.2f}, 残差={epsilon[i]:.2f}")

print("\n未来3年预测结果：")
for i in range(forecast_num):
    print(f"第{n+i+1}年预测值: {future_x0[i]:.2f}")

# 10. 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(range(1, n+1), x0, 'bo-', label='实际销售额')
plt.plot(range(1, n+1), x0_hat, 'r--', label='拟合销售额')
plt.plot(range(n+1, n+1+forecast_num), future_x0, 'g*-', label='预测销售额')
plt.xlabel('年份')
plt.ylabel('销售额（万元）')
plt.title('GM(1,1)灰色预测模型')
plt.legend()
plt.grid(True)
plt.show()
