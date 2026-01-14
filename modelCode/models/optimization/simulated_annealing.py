#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模拟退火算法案例说明：
本案例使用模拟退火算法求解Schwefel函数的最小值。
Schwefel函数是一个具有强多峰特性的复杂函数，全局最优值在(420.9687,420.9687)附近。
模拟退火算法灵感来源于物理退火过程，通过接受一定概率的劣解跳出局部最优，
适合求解复杂的全局优化问题。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Schwefel函数：一个复杂的多峰函数
def schwefel(x):
    n = len(x)
    return 418.9829 * n - sum([xi * np.sin(np.sqrt(np.abs(xi))) for xi in x])

# 生成邻域解
def generate_neighbor(x, bounds, step_size):
    neighbor = x.copy()
    dim = len(x)
    
    # 随机选择一个维度进行扰动
    idx = np.random.randint(dim)
    # 生成随机扰动
    neighbor[idx] += np.random.uniform(-step_size, step_size)
    
    # 边界处理
    neighbor[idx] = max(bounds[idx][0], min(neighbor[idx], bounds[idx][1]))
    
    return neighbor

# 模拟退火算法
def simulated_annealing(objective_func, bounds, initial_temp, cooling_rate, 
                        max_iter, step_size):
    dim = len(bounds)
    
    # 初始化当前解
    current_solution = np.array([np.random.uniform(low, high) for low, high in bounds])
    current_value = objective_func(current_solution)
    
    # 初始化最优解
    best_solution = current_solution.copy()
    best_value = current_value
    
    # 记录历史
    current_temp = initial_temp
    best_history = [best_value]
    
    # 主循环
    for i in range(max_iter):
        # 生成邻域解
        neighbor = generate_neighbor(current_solution, bounds, step_size)
        neighbor_value = objective_func(neighbor)
        
        # 计算能量差（目标函数差）
        delta = neighbor_value - current_value
        
        # 接受准则：如果更优则接受，否则以一定概率接受
        if delta < 0 or np.random.rand() < np.exp(-delta / current_temp):
            current_solution = neighbor.copy()
            current_value = neighbor_value
            
            # 更新最优解
            if current_value < best_value:
                best_solution = current_solution.copy()
                best_value = current_value
        
        # 降温
        current_temp *= cooling_rate
        
        # 记录历史
        best_history.append(best_value)
        
        # 动态调整步长（可选）
        if i % 100 == 0 and i > 0:
            step_size = max(0.1, step_size * 0.95)
    
    return best_solution, best_value, best_history

# 参数设置
bounds = [(-500, 500), (-500, 500)]  # 变量范围
initial_temp = 100.0                 # 初始温度
cooling_rate = 0.95                  # 冷却速率
max_iter = 1000                      # 最大迭代次数
step_size = 50.0                     # 初始步长

# 运行模拟退火算法
best_solution, best_value, best_history = simulated_annealing(
    schwefel, bounds, initial_temp, cooling_rate, max_iter, step_size
)

# 输出结果
print("模拟退火算法求解Schwefel函数结果：")
print(f"最优解: x = {best_solution[0]:.6f}, y = {best_solution[1]:.6f}")
print(f"最优值: f(x,y) = {best_value:.6f}")
print(f"理论最优解附近: (420.9687, 420.9687)")

# 可视化结果（可选）
if True:
    # 绘制函数曲面
    fig = plt.figure(figsize=(15, 6))
    
    # 3D曲面图
    ax1 = fig.add_subplot(121, projection='3d')
    x = np.linspace(bounds[0][0], bounds[0][1], 50)
    y = np.linspace(bounds[1][0], bounds[1][1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.array([schwefel([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax1.scatter(best_solution[0], best_solution[1], best_value, color='red', s=100, label='最优解')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Schwefel函数曲面与最优解')
    ax1.legend()
    
    # 收敛曲线图
    ax2 = fig.add_subplot(122)
    ax2.plot(best_history)
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('最优值')
    ax2.set_title('收敛曲线')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    