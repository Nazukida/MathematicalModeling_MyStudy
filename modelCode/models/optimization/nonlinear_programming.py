#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
非线性规划模型案例说明：
本案例使用非线性规划解决生产优化问题。某工厂生产两种产品A和B，
生产成本和销售价格存在非线性关系，目标是最大化净利润。
非线性规划适用于目标函数或约束条件包含非线性表达式的优化问题。
"""

import numpy as np
from scipy.optimize import minimize

# 定义目标函数：最大化净利润 = 销售收入 - 生产成本
# 注意：这里返回负值，因为scipy的minimize是求最小值
def objective(x):
    x1, x2 = x  # x1:产品A产量, x2:产品B产量
    # 销售收入（非线性）：价格随产量增加而降低
    revenue = (100 - 0.5*x1)*x1 + (150 - 0.8*x2)*x2
    # 生产成本（非线性）：包含固定成本和可变成本
    cost = 500 + 20*x1 + 30*x2 + 0.1*x1**2 + 0.2*x2**2
    return -(revenue - cost)  # 最小化负净利润等价于最大化净利润

# 定义约束条件
def constraint1(x):
    # 原材料约束：2x1 + 3x2 <= 200
    return 200 - (2*x[0] + 3*x[1])

def constraint2(x):
    # 工时约束：5x1 + 4x2 <= 300
    return 300 - (5*x[0] + 4*x[1])

# 初始猜测值
x0 = [10, 10]

# 变量边界：x1 >= 0, x2 >= 0
bounds = [(0, None), (0, None)]

# 约束条件设置
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
constraints = [con1, con2]

# 求解非线性规划
solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
x = solution.x

# 计算结果
x1, x2 = x
revenue = (100 - 0.5*x1)*x1 + (150 - 0.8*x2)*x2
cost = 500 + 20*x1 + 30*x2 + 0.1*x1**2 + 0.2*x2**2
profit = revenue - cost

# 输出结果
print("非线性规划生产优化问题结果：")
print(f"最优解: 产品A生产 {x1:.2f} 件, 产品B生产 {x2:.2f} 件")
print(f"销售收入: {revenue:.2f} 元")
print(f"生产成本: {cost:.2f} 元")
print(f"最大净利润: {profit:.2f} 元")

# 约束条件使用情况
material_used = 2*x1 + 3*x2
labor_used = 5*x1 + 4*x2
print("\n约束条件使用情况:")
print(f"原材料约束: 实际使用 {material_used:.2f}, 限制 200")
print(f"工时约束: 实际使用 {labor_used:.2f}, 限制 300")
    