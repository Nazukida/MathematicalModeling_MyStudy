#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
整数规划模型案例说明：
本案例使用整数规划解决投资组合问题。有5个投资项目可选，
每个项目有初始投资和预期收益，总预算限制为300万元。
目标是最大化总预期收益，且每个项目只能选择投资或不投资（0-1整数变量）。
整数规划适用于决策变量需取整数值的优化问题。
"""

import pulp as pl

# 创建问题实例：最大化总预期收益
prob = pl.LpProblem("投资组合问题", pl.LpMaximize)

# 定义决策变量：5个项目是否投资（0-1整数）
projects = range(5)
x = pl.LpVariable.dicts("投资项目", projects, cat=pl.LpBinary)

# 项目数据：[初始投资(万元), 预期收益(万元)]
project_data = {
    0: [100, 150],
    1: [120, 180],
    2: [80, 120],
    3: [150, 220],
    4: [90, 130]
}

# 定义目标函数：最大化总预期收益
prob += sum(project_data[i][1] * x[i] for i in projects), "总预期收益"

# 定义约束条件：总投资不超过预算
prob += sum(project_data[i][0] * x[i] for i in projects) <= 300, "预算约束"

# 求解问题
prob.solve()

# 输出结果
print(f"求解状态: {pl.LpStatus[prob.status]}")
print("\n最优投资组合:")
total_invest = 0
total_profit = 0
for i in projects:
    if pl.value(x[i]) == 1:
        print(f"投资项目{i}: 投资{project_data[i][0]}万元, 预期收益{project_data[i][1]}万元")
        total_invest += project_data[i][0]
        total_profit += project_data[i][1]

print(f"\n总投资: {total_invest}万元, 总预期收益: {total_profit}万元")
print(f"预算剩余: {300 - total_invest}万元")
