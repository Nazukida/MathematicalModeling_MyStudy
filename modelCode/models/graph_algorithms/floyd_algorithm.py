#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Floyd算法案例说明：
Floyd算法（弗洛伊德算法）是一种用于寻找图中所有节点对之间最短路径的动态规划算法。
与Dijkstra算法不同，它可以一次性计算出所有节点之间的最短路径，适用于有向图和无向图，
但不适用于包含负权回路的图。

算法步骤：
1. 初始化距离矩阵dist，dist[i][j]表示节点i到节点j的初始距离
2. 对于每个中间节点k（0到n-1）：
   a. 对于每对节点(i, j)：
      i. 计算从i经过k到j的距离：dist[i][k] + dist[k][j]
      ii. 如果该距离小于当前dist[i][j]，则更新dist[i][j]
3. 算法结束后，dist[i][j]即为节点i到节点j的最短距离

本案例使用一个包含5个节点的有向图，演示Floyd算法如何计算所有节点对之间的最短路径，
并以矩阵形式展示结果，同时可视化部分关键路径。
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 定义图的邻接矩阵（表示节点间的距离，inf表示无直接连接）
# 这是一个有向图，节点0-4代表5个不同的地点
graph = [
    [0, 3, np.inf, 7, np.inf],   # 节点0到其他节点的距离
    [8, 0, 2, np.inf, np.inf],   # 节点1到其他节点的距离
    [np.inf, np.inf, 0, 1, 4],   # 节点2到其他节点的距离
    [np.inf, np.inf, np.inf, 0, 5], # 节点3到其他节点的距离
    [np.inf, 6, np.inf, np.inf, 0]  # 节点4到其他节点的距离
]

# 节点名称（用于可视化）
node_names = ['A', 'B', 'C', 'D', 'E']
num_nodes = len(graph)

def floyd(graph):
    """实现Floyd算法，返回所有节点对之间的最短距离矩阵和路径矩阵"""
    # 初始化距离矩阵
    dist = [row[:] for row in graph]
    # 初始化路径矩阵，path[i][j]表示i到j的最短路径中j的前一个节点
    path = [[-1 for _ in range(num_nodes)] for _ in range(num_nodes)]
    
    # 初始化路径矩阵：如果i和j直接相连，则path[i][j] = i
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and dist[i][j] != np.inf:
                path[i][j] = i
    
    # Floyd算法核心
    for k in range(num_nodes):  # 中间节点
        for i in range(num_nodes):  # 起点
            for j in range(num_nodes):  # 终点
                # 如果i到k和k到j都可达，且经过k的路径更短
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    # 更新最短距离
                    dist[i][j] = dist[i][k] + dist[k][j]
                    # 更新路径：j的前一个节点是k的前一个节点
                    path[i][j] = path[k][j]
    
    return dist, path

def get_path(path, start, end):
    """根据路径矩阵获取从start到end的最短路径"""
    if path[start][end] == -1:
        return None  # 不可达
    # 重建路径
    result = [end]
    current = end
    while current != start:
        current = path[start][current]
        if current == -1:  # 路径不存在
            return None
        result.append(current)
    # 反转路径，得到从起点到终点的顺序
    return result[::-1]

# 运行Floyd算法
print("使用Floyd算法计算所有节点对之间的最短路径...")
dist_matrix, path_matrix = floyd(graph)

# 输出所有节点对之间的最短距离矩阵
print("\n所有节点对之间的最短距离矩阵：")
# 打印表头
print("    " + " ".join(f"{name:4}" for name in node_names))
for i in range(num_nodes):
    row = [f"{node_names[i]}:"]
    for j in range(num_nodes):
        if dist_matrix[i][j] == np.inf:
            row.append("  ∞")
        else:
            row.append(f"{dist_matrix[i][j]:4}")
    print(" ".join(row))

# 输出部分关键路径
print("\n部分关键路径详情：")
key_pairs = [(0, 4), (1, 3), (4, 2)]  # 选择几对有代表性的节点
for start, end in key_pairs:
    path = get_path(path_matrix, start, end)
    if path is None:
        print(f"从{node_names[start]}到{node_names[end]}: 不可达")
    else:
        path_names = [node_names[node] for node in path]
        print(f"从{node_names[start]}到{node_names[end]}: "
              f"路径: {' -> '.join(path_names)}, 距离: {dist_matrix[start][end]}")

# 可视化图和部分最短路径
plt.figure(figsize=(12, 8))

# 创建有向图
G = nx.DiGraph()

# 添加节点
for i in range(num_nodes):
    G.add_node(i, name=node_names[i])

# 添加边和权重
for i in range(num_nodes):
    for j in range(num_nodes):
        if graph[i][j] > 0 and graph[i][j] != np.inf:
            G.add_edge(i, j, weight=int(graph[i][j]))

# 定义节点位置（使用圆形布局）
pos = nx.circular_layout(G)

# 绘制所有节点和边
nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue')
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=1, 
                       alpha=0.5, edge_color='gray', arrowstyle='->')

# 绘制关键路径（用红色高亮显示）
for start, end in key_pairs:
    path = get_path(path_matrix, start, end)
    if path is not None and len(path) > 1:
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, 
                              edge_color='red', arrowstyle='->')

# 绘制节点标签和边权重
nx.draw_networkx_labels(G, pos, labels={i: node_names[i] for i in range(num_nodes)}, font_size=12)
edge_labels = {(i, j): graph[i][j] for i, j in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

plt.title('Floyd算法计算的全源最短路径（红色为关键路径）', fontsize=14)
plt.axis('off')
plt.show()
    