#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最速下降法案例说明：
本案例使用最速下降法（梯度下降法）求解无约束优化问题。
目标函数为 f(x,y) = x² + 3y² - 2xy，这是一个凸函数，存在唯一最小值。
最速下降法是一种一阶优化算法，沿着目标函数梯度的反方向（最速下降方向）搜索最优解。
"""

import numpy as np
import matplotlib.pyplot as plt

# 目标函数: f(x,y) = x² + 3y² - 2xy
def objective(x):
    return x[0]**2 + 3*x[1]** 2 - 2*x[0]*x[1]

# 目标函数的梯度: ∇f(x,y) = [2x-2y, 6y-2x]
def gradient(x):
    dx = 2*x[0] - 2*x[1]    # 对x的偏导数
    dy = 6*x[1] - 2*x[0]    # 对y的偏导数
    return np.array([dx, dy])

# 最速下降法
def steepest_descent(initial_point, learning_rate, max_iterations, tolerance):
    x = np.array(initial_point, dtype=np.float64)
    path = [x.copy()]  # 记录迭代路径
    
    for i in range(max_iterations):
        grad = gradient(x)  # 计算梯度
        
        # 判断收敛条件：梯度的模长小于 tolerance
        if np.linalg.norm(grad) < tolerance:
            break
            
        # 沿负梯度方向更新（最速下降方向）
        x = x - learning_rate * grad
        path.append(x.copy())
    
    return x, path, i+1  # 返回最优解、迭代路径和迭代次数

# 参数设置
initial_point = [5.0, 5.0]  # 初始点
learning_rate = 0.1         # 学习率（步长）
max_iterations = 1000       # 最大迭代次数
tolerance = 1e-6            # 收敛容差

# 运行最速下降法
optimal_point, path, iterations = steepest_descent(
    initial_point, learning_rate, max_iterations, tolerance
)

# 计算最优值
optimal_value = objective(optimal_point)

# 输出结果
print("最速下降法求解结果：")
print(f"初始点: {initial_point}")
print(f"迭代次数: {iterations}")
print(f"最优解: x = {optimal_point[0]:.6f}, y = {optimal_point[1]:.6f}")
print(f"最优值: f(x,y) = {optimal_value:.6f}")
print(f"最终梯度模长: {np.linalg.norm(gradient(optimal_point)):.6e}")

# 可视化迭代过程（可选）
if True:
    # 创建网格
    x = np.linspace(-1, 6, 100)
    y = np.linspace(-1, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = objective([X, Y])
    
    # 绘制等高线和迭代路径
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=30, cmap='viridis')
    plt.colorbar(label='函数值')
    
    # 绘制迭代路径
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'ro-', markersize=5, label='迭代路径')
    plt.plot(initial_point[0], initial_point[1], 'go', markersize=8, label='初始点')
    plt.plot(optimal_point[0], optimal_point[1], 'bo', markersize=8, label='最优解')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('最速下降法迭代路径')
    plt.legend()
    plt.grid(True)
    plt.show()
    