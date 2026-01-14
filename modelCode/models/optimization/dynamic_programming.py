#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态优化模型（动态规划）案例说明：
本案例使用动态规划解决背包问题。有5个物品，每个物品有重量和价值，
背包最大容量为10公斤。目标是在不超过背包容量的前提下，最大化总价值。
动态规划通过将复杂问题分解为子问题，利用子问题的解构建原问题的解。
"""

# 物品数据：[重量(kg), 价值(元)]
items = [
    [2, 6],   # 物品0
    [2, 3],   # 物品1
    [6, 5],   # 物品2
    [5, 4],   # 物品3
    [4, 6]    # 物品4
]
weights = [item[0] for item in items]
values = [item[1] for item in items]
capacity = 10  # 背包容量(kg)
n = len(items)  # 物品数量

# 创建二维数组dp，dp[i][j]表示考虑前i个物品，背包容量为j时的最大价值
dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

# 填充dp数组
for i in range(1, n + 1):
    for j in range(capacity + 1):
        # 当前物品的重量和价值（i-1是因为物品索引从0开始）
        current_weight = weights[i-1]
        current_value = values[i-1]
        
        if current_weight <= j:
            # 可以放入第i个物品，比较放入和不放入的价值
            # 放入：当前物品价值 + 剩余容量的最佳价值
            # 不放入：前i-1个物品的最佳价值
            dp[i][j] = max(current_value + dp[i-1][j-current_weight], dp[i-1][j])
        else:
            # 不能放入第i个物品，继承前i-1个物品的最佳价值
            dp[i][j] = dp[i-1][j]

# 回溯找出选择的物品
selected_items = []
j = capacity  # 从最大容量开始回溯
for i in range(n, 0, -1):
    # 如果当前状态与不包含第i个物品的状态不同，说明选择了第i个物品
    if dp[i][j] != dp[i-1][j]:
        selected_items.append(i-1)  # 记录物品索引
        j -= weights[i-1]  # 减去该物品的重量

# 输出结果
print("动态规划解决背包问题案例：")
print(f"背包最大容量: {capacity}kg")
print("\n物品列表:")
for i, item in enumerate(items):
    print(f"物品{i}: 重量{item[0]}kg, 价值{item[1]}元")

max_value = dp[n][capacity]
print(f"\n最大总价值: {max_value}元")
print("选择的物品:")
for i in selected_items:
    print(f"物品{i}: 重量{items[i][0]}kg, 价值{items[i][1]}元")

total_weight = sum(items[i][0] for i in selected_items)
print(f"\n总重量: {total_weight}kg, 剩余容量: {capacity - total_weight}kg")
