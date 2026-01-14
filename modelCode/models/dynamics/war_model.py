#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
战争模型案例说明：
本案例实现经典的兰彻斯特战争模型（Lanchester's model），该模型用于描述两个敌对部队在战斗中的兵力变化。

基本模型公式（平方律）：
dx/dt = -b*y
dy/dt = -a*x

其中：
- x(t) 表示甲方在t时刻的兵力
- y(t) 表示乙方在t时刻的兵力
- a 表示乙方的战斗力系数（单位时间内每个乙方士兵消灭的甲方士兵数）
- b 表示甲方的战斗力系数（单位时间内每个甲方士兵消灭的乙方士兵数）

模型特点：
- 平方律模型适用于双方都能瞄准并攻击敌方任意目标的现代战争
- 战斗力不仅取决于兵力数量，还取决于武器装备、训练水平等因素（体现在系数a和b中）
- 当一方兵力减为0时，另一方获胜

本案例模拟两种不同初始条件下的战斗过程，展示兵力随时间的变化。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. 定义兰彻斯特战争模型（平方律）
def lanchester_model(t, state, a, b):
    """
    兰彻斯特战争模型微分方程组
    dx/dt = -b*y
    dy/dt = -a*x
    
    参数:
    t: 时间
    state: 状态变量 [x, y]，分别表示甲乙双方的兵力
    a: 乙方战斗力系数
    b: 甲方战斗力系数
    """
    x, y = state
    dxdt = -b * y  # 甲方兵力变化率
    dydt = -a * x  # 乙方兵力变化率
    return [dxdt, dydt]

# 2. 设置模型参数和初始条件
# 情况1：甲方兵力占优
a1 = 0.05       # 乙方战斗力系数
b1 = 0.08       # 甲方战斗力系数
x0_1, y0_1 = 100, 60  # 初始兵力

# 情况2：乙方装备更精良（战斗力系数更高）
a2 = 0.12       # 乙方战斗力系数（更高）
b2 = 0.08       # 甲方战斗力系数
x0_2, y0_2 = 100, 80  # 初始兵力

t_span = [0, 15]  # 时间跨度（天）
t_eval = np.linspace(0, 15, 100)  # 评估时间点

# 3. 求解微分方程（两种情况）
# 情况1
solution1 = solve_ivp(
    lanchester_model,          # 微分方程函数
    t_span,                    # 时间范围
    [x0_1, y0_1],              # 初始条件
    args=(a1, b1),             # 传递给模型的额外参数
    t_eval=t_eval,             # 评估解的时间点
    method='RK45'              # 数值方法
)

# 情况2
solution2 = solve_ivp(
    lanchester_model,
    t_span,
    [x0_2, y0_2],
    args=(a2, b2),
    t_eval=t_eval,
    method='RK45'
)

# 4. 确定每种情况下的战斗结束时间
def find_end_time(t, x, y):
    """找到一方兵力接近0的时间点"""
    # 找到甲方兵力接近0的索引
    x_end_idx = np.argmax(x < 1)
    # 找到乙方兵力接近0的索引
    y_end_idx = np.argmax(y < 1)
    
    if x_end_idx == 0 and x[0] >= 1:
        x_end_idx = len(x) + 1  # 甲方未被消灭
    if y_end_idx == 0 and y[0] >= 1:
        y_end_idx = len(y) + 1  # 乙方未被消灭
    
    if x_end_idx < y_end_idx:
        # 甲方先被消灭
        end_time = t[x_end_idx]
        winner = "乙方"
        remaining = y[x_end_idx]
    elif y_end_idx < x_end_idx:
        # 乙方先被消灭
        end_time = t[y_end_idx]
        winner = "甲方"
        remaining = x[y_end_idx]
    else:
        # 同时被消灭
        end_time = t[-1]
        winner = "双方"
        remaining = 0
    
    return end_time, winner, remaining

# 计算情况1的结果
end_time1, winner1, remaining1 = find_end_time(
    solution1.t, solution1.y[0], solution1.y[1])

# 计算情况2的结果
end_time2, winner2, remaining2 = find_end_time(
    solution2.t, solution2.y[0], solution2.y[1])

# 5. 可视化结果
plt.figure(figsize=(14, 6))

# 绘制情况1
plt.subplot(1, 2, 1)
plt.plot(solution1.t, solution1.y[0], 'b-', linewidth=2, label=f'甲方兵力 (初始: {x0_1})')
plt.plot(solution1.t, solution1.y[1], 'r-', linewidth=2, label=f'乙方兵力 (初始: {y0_1})')
plt.axvline(x=end_time1, color='k', linestyle='--', alpha=0.7, 
            label=f'战斗结束: {end_time1:.1f}天')
plt.xlabel('时间（天）', fontsize=12)
plt.ylabel('兵力数量', fontsize=12)
plt.title(f'情况1: {winner1}获胜，剩余兵力: {remaining1:.0f}', fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.ylim(0, max(x0_1, y0_1) * 1.1)

# 绘制情况2
plt.subplot(1, 2, 2)
plt.plot(solution2.t, solution2.y[0], 'b-', linewidth=2, label=f'甲方兵力 (初始: {x0_2})')
plt.plot(solution2.t, solution2.y[1], 'r-', linewidth=2, label=f'乙方兵力 (初始: {y0_2})')
plt.axvline(x=end_time2, color='k', linestyle='--', alpha=0.7, 
            label=f'战斗结束: {end_time2:.1f}天')
plt.xlabel('时间（天）', fontsize=12)
plt.ylabel('兵力数量', fontsize=12)
plt.title(f'情况2: {winner2}获胜，剩余兵力: {remaining2:.0f}', fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.ylim(0, max(x0_2, y0_2) * 1.1)

plt.tight_layout()
plt.show()

# 6. 输出战斗结果
print("战斗结果摘要：")
print(f"情况1: 甲方初始兵力={x0_1}, 乙方初始兵力={y0_1}, "
      f"乙方战斗力系数={a1}, 甲方战斗力系数={b1}")
print(f"       获胜方: {winner1}, 战斗持续时间: {end_time1:.1f}天, "
      f"剩余兵力: {remaining1:.0f}\n")

print(f"情况2: 甲方初始兵力={x0_2}, 乙方初始兵力={y0_2}, "
      f"乙方战斗力系数={a2}, 甲方战斗力系数={b2}")
print(f"       获胜方: {winner2}, 战斗持续时间: {end_time2:.1f}天, "
      f"剩余兵力: {remaining2:.0f}")
    