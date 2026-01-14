#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0-1规划模型案例说明：
本案例使用0-1规划解决设备选址问题。有4个候选地点可建立仓库，
每个地点的建设成本和覆盖需求不同，总预算限制为500万元。
目标是最大化总覆盖需求，且每个地点只能选择建设或不建设（0-1变量）。
0-1规划是整数规划的特例，决策变量只能取0或1。
"""

import pulp as pl

# 创建问题实例：最大化总覆盖需求
prob = pl.LpProblem("设备选址问题", pl.LpMaximize)

# 定义决策变量：4个地点是否建设仓库（0-1整数）
locations = range(4)
x = pl.LpVariable.dicts("建设仓库", locations, cat=pl.LpBinary)

# 地点数据：[建设成本(万元), 覆盖需求(千户)]
location_data = {
    0: [200, 15],
    1: [180, 12],
    2: [250, 20],
    3: [150, 10]
}

# 定义目标函数：最大化总覆盖需求
prob += sum(location_data[i][1] * x[i] for i in locations), "总覆盖需求"

# 定义约束条件：总建设成本不超过预算
prob += sum(location_data[i][0] * x[i] for i in locations) <= 500, "预算约束"

# 求解问题
prob.solve()

# 输出结果
print(f"求解状态: {pl.LpStatus[prob.status]}")
print("\n最优选址方案:")
total_cost = 0
total_demand = 0
for i in locations:
    if pl.value(x[i]) == 1:
        print(f"在地点{i}建设仓库: 成本{location_data[i][0]}万元, 覆盖需求{location_data[i][1]}千户")
        total_cost += location_data[i][0]
        total_demand += location_data[i][1]

print(f"\n总建设成本: {total_cost}万元, 总覆盖需求: {total_demand}千户")
print(f"预算剩余: {500 - total_cost}万元")
