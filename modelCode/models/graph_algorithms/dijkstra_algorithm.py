#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dijkstra算法案例说明：
Dijkstra算法是一种用于寻找图中从单个源节点到其他所有节点的最短路径的贪心算法。
它适用于边权值为非负数的图，通过不断选择当前距离源节点最近的未访问节点，
并更新其相邻节点的距离来实现。

算法步骤：
1. 初始化：将源节点到所有其他节点的距离设为无穷大，源节点到自身的距离设为0
2. 选择距离源节点最近的未访问节点u
3. 对于u的每个相邻节点v，计算从源节点经过u到v的距离，若小于当前已知距离则更新
4. 标记u为已访问
5. 重复步骤2-4，直到所有节点都被访问

本案例使用一个包含7个节点的无向图，演示Dijkstra算法如何找到从源节点到其他所有节点的最短路径，
并可视化图结构和最短路径结果。
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 定义图的邻接矩阵（表示节点间的距离，0表示无直接连接）
# 节点0-6代表7个不同的地点
graph = [
    [0, 4, 0, 0, 0, 0, 2],   # 节点0与其他节点的距离
    [4, 0, 6, 0, 0, 0, 0],   # 节点1与其他节点的距离
    [0, 6, 0, 3, 0, 5, 0],   # 节点2与其他节点的距离
    [0, 0, 3, 0, 2, 0, 0],   # 节点3与其他节点的距离
    [0, 0, 0, 2, 0, 1, 0],   # 节点4与其他节点的距离
    [0, 0, 5, 0, 1, 0, 4],   # 节点5与其他节点的距离
    [2, 0, 0, 0, 0, 4, 0]    # 节点6与其他节点的距离
]

# 节点名称（用于可视化）
node_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
num_nodes = len(graph)

def dijkstra(graph, start):
    """实现Dijkstra算法，返回从起点到所有节点的最短距离和路径"""
    # 初始化距离数组，无穷大表示不可达
    distances = [float('inf')] * num_nodes
    # 起点到自身的距离为0
    distances[start] = 0
    # 记录已访问的节点
    visited = [False] * num_nodes
    # 记录最短路径的前驱节点
    predecessors = [-1] * num_nodes
    
    for _ in range(num_nodes):
        # 找到当前未访问节点中距离最小的节点u
        min_distance = float('inf')
        u = -1
        for i in range(num_nodes):
            if not visited[i] and distances[i] < min_distance:
                min_distance = distances[i]
                u = i
        
        # 如果所有节点都已访问或剩下的节点不可达，退出循环
        if u == -1:
            break
        
        # 标记u为已访问
        visited[u] = True
        
        # 更新u的所有邻居节点的距离
        for v in range(num_nodes):
            # 如果v是u的邻居且未被访问
            if graph[u][v] > 0 and not visited[v]:
                # 计算从起点经过u到v的距离
                new_distance = distances[u] + graph[u][v]
                # 如果新距离更小，则更新
                if new_distance < distances[v]:
                    distances[v] = new_distance
                    predecessors[v] = u
    
    # 重建最短路径
    paths = []
    for end in range(num_nodes):
        path = []
        current = end
        # 从终点回溯到起点
        while current != -1:
            path.append(current)
            current = predecessors[current]
        # 反转路径，得到从起点到终点的路径
        path.reverse()
        # 如果路径只有一个节点且不是起点，说明不可达
        if len(path) == 1 and path[0] != start:
            paths.append(None)  # 不可达
        else:
            paths.append(path)
    
    return distances, paths

# 选择起点（例如选择节点0，对应名称'A'）
start_node = 0
print(f"使用Dijkstra算法计算从节点{node_names[start_node]}到其他所有节点的最短路径...")

# 运行Dijkstra算法
distances, paths = dijkstra(graph, start_node)

# 输出结果
print("\n最短路径结果：")
for i in range(num_nodes):
    if i == start_node:
        continue
    if paths[i] is None:
        print(f"从{node_names[start_node]}到{node_names[i]}: 不可达")
    else:
        path_names = [node_names[j] for j in paths[i]]
        print(f"从{node_names[start_node]}到{node_names[i]}: "
              f"路径: {' -> '.join(path_names)}, 距离: {distances[i]}")

# 可视化图和最短路径
plt.figure(figsize=(12, 8))

# 创建无向图
G = nx.Graph()

# 添加节点
for i in range(num_nodes):
    G.add_node(i, name=node_names[i])

# 添加边和权重
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        if graph[i][j] > 0:
            G.add_edge(i, j, weight=graph[i][j])

# 定义节点位置（使用圆形布局）
pos = nx.circular_layout(G)

# 绘制所有节点和边
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=1, alpha=0.5, edge_color='gray')

# 绘制最短路径（用红色高亮显示）
for i in range(num_nodes):
    if i != start_node and paths[i] is not None and len(paths[i]) > 1:
        path_edges = [(paths[i][j], paths[i][j+1]) for j in range(len(paths[i])-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color='red')

# 绘制节点标签和边权重
nx.draw_networkx_labels(G, pos, labels={i: node_names[i] for i in range(num_nodes)}, font_size=12)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

plt.title(f'Dijkstra算法最短路径（从{node_names[start_node]}出发）', fontsize=14)
plt.axis('off')
plt.show()
    