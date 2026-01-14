#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
线性规划模型案例说明：
本案例使用线性规划解决生产资源分配问题。某工厂生产两种产品A和B，
每种产品需要消耗原材料和工时，有最大供应量和市场需求限制。
目标是最大化总利润。线性规划适用于目标函数和约束条件均为线性的问题。
"""

import pulp as pl

# 创建问题实例：最大化总利润
prob = pl.LpProblem("生产资源分配问题", pl.LpMaximize)

# 定义决策变量：产品A和B的生产量（非负）
x1 = pl.LpVariable("产品A产量", lowBound=0)
x2 = pl.LpVariable("产品B产量", lowBound=0)

# 定义目标函数：最大化总利润（A利润50元/件，B利润60元/件）
prob += 50 * x1 + 60 * x2, "总利润"

# 定义约束条件
prob += 2 * x1 + 3 * x2 <= 100, "原材料约束"  # 原材料最大供应量100单位
prob += 4 * x1 + 2 * x2 <= 120, "工时约束"     # 总工时不超过120小时
prob += x1 <= 25, "产品A需求约束"           # 产品A最大需求25件
prob += x2 <= 30, "产品B需求约束"           # 产品B最大需求30件

# 求解问题
prob.solve()

# 输出结果
print(f"求解状态: {pl.LpStatus[prob.status]}")
print(f"最优解: 产品A生产 {pl.value(x1):.2f} 件, 产品B生产 {pl.value(x2):.2f} 件")
print(f"最大总利润: {pl.value(prob.objective):.2f} 元")

# 输出各约束条件的使用情况
print("\n约束条件使用情况:")
for constraint in prob.constraints:
    print(f"{constraint}: 实际使用 {prob.constraints[constraint].value():.2f}, 限制 {prob.constraints[constraint].ub:.2f}")
