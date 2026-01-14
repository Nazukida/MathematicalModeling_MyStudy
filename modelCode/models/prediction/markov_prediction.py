#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
马尔可夫预测模型案例说明：
本案例使用马尔可夫链模型预测某股票价格的涨跌状态，
基于过去30天的价格状态记录。马尔可夫模型适用于预测
具有无后效性的随机过程，即未来状态只与当前状态有关，
与过去状态无关。本案例将股票价格分为3种状态：下跌(0)、
平稳(1)、上涨(2)，通过状态转移概率矩阵预测未来状态。
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据
# 生成模拟数据：过去30天的股票价格状态
# 状态定义：0=下跌，1=平稳，2=上涨
np.random.seed(42)  # 设置随机种子，确保结果可复现

# 生成30天的状态序列（模拟股票价格波动）
states = np.array([
    2, 2, 1, 0, 0, 1, 2, 2, 2, 1,
    0, 0, 1, 1, 2, 2, 1, 1, 0, 0,
    1, 2, 2, 1, 0, 1, 2, 2, 1, 0
])
n_days = len(states)  # 天数
n_states = 3          # 状态数量：0-下跌，1-平稳，2-上涨
state_names = ['下跌', '平稳', '上涨']

# 2. 计算状态转移概率矩阵
# 初始化转移计数矩阵
transition_counts = np.zeros((n_states, n_states), dtype=int)

# 统计状态转移次数
for i in range(n_days - 1):
    current_state = states[i]
    next_state = states[i + 1]
    transition_counts[current_state, next_state] += 1

# 计算转移概率矩阵（行标准化）
transition_matrix = np.zeros((n_states, n_states))
for i in range(n_states):
    total = np.sum(transition_counts[i, :])
    if total > 0:
        transition_matrix[i, :] = transition_counts[i, :] / total

# 3. 计算初始状态分布
initial_state = states[-1]  # 最后一天的状态作为初始状态
current_distribution = np.zeros(n_states)
current_distribution[initial_state] = 1.0  # 初始分布：确定处于最后一个状态

# 4. 预测未来状态
forecast_days = 5  # 预测未来5天
forecast_distributions = []  # 存储每天的状态分布预测

# 迭代计算未来每一天的状态分布
for _ in range(forecast_days):
    # 状态分布 = 当前分布 × 转移矩阵
    current_distribution = np.dot(current_distribution, transition_matrix)
    forecast_distributions.append(current_distribution.copy())

# 确定每天最可能的状态
most_likely_states = [np.argmax(dist) for dist in forecast_distributions]

# 5. 计算各状态的稳态分布（长期预测）
# 通过多次迭代转移矩阵直到收敛
steady_state = current_distribution.copy()
for _ in range(1000):
    steady_state = np.dot(steady_state, transition_matrix)

# 6. 输出结果
print("马尔可夫预测模型结果：")
print("\n状态转移概率矩阵：")
print("行：当前状态，列：下一个状态")
print("      下跌     平稳     上涨")
for i in range(n_states):
    print(f"{state_names[i]}: {transition_matrix[i, 0]:.4f}, {transition_matrix[i, 1]:.4f}, {transition_matrix[i, 2]:.4f}")

print(f"\n最后一天的状态：{state_names[initial_state]}")

print("\n未来5天的状态概率分布：")
for i in range(forecast_days):
    print(f"第{i+1}天:")
    for j in range(n_states):
        print(f"  {state_names[j]}: {forecast_distributions[i][j]:.2%}")
    print(f"  最可能的状态: {state_names[most_likely_states[i]]}")

print("\n长期稳态分布：")
for j in range(n_states):
    print(f"  {state_names[j]}: {steady_state[j]:.2%}")

# 7. 可视化结果
plt.figure(figsize=(12, 6))

# 绘制历史状态
plt.subplot(2, 1, 1)
plt.plot(range(1, n_days+1), states, 'bo-', markersize=6)
plt.yticks([0, 1, 2], state_names)
plt.xlabel('天数')
plt.ylabel('状态')
plt.title('股票价格历史状态')
plt.grid(True)

# 绘制预测概率
plt.subplot(2, 1, 2)
days = range(n_days+1, n_days+1+forecast_days)
# 提取每种状态的预测概率
down_probs = [dist[0] for dist in forecast_distributions]
stable_probs = [dist[1] for dist in forecast_distributions]
up_probs = [dist[2] for dist in forecast_distributions]

plt.stackplot(days, down_probs, stable_probs, up_probs, 
              labels=state_names, alpha=0.8, 
              colors=['red', 'blue', 'green'])
plt.plot(days, most_likely_states, 'ko-', markersize=8, linewidth=2, label='最可能状态')
plt.yticks(np.arange(0, 1.1, 0.2))
plt.xlabel('天数')
plt.ylabel('概率')
plt.title(f'未来{forecast_days}天状态预测概率分布')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
