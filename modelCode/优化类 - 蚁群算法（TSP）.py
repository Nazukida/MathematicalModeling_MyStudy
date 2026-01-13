"""
============================================================
蚁群算法 (Ant Colony Optimization, ACO)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：组合优化、TSP旅行商问题、路径规划
原理：模拟蚂蚁觅食行为，通过信息素引导搜索
作者：MCM/ICM Team
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 图表美化设置
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class AntColonyTSP:
    """
    蚁群算法求解TSP问题
    
    核心公式：
    转移概率: P_ij = (τ_ij^α * η_ij^β) / Σ(τ^α * η^β)
    信息素更新: τ_ij = (1-ρ)*τ_ij + Δτ_ij
    
    参数说明：
    - α (alpha): 信息素重要程度（1-2）
    - β (beta): 启发式信息重要程度（2-5）
    - ρ (rho): 信息素挥发系数（0.1-0.5）
    - Q: 信息素增量常数
    """
    
    def __init__(self, cities, n_ants=30, max_iter=100,
                 alpha=1.0, beta=2.0, rho=0.5, Q=100,
                 random_seed=42, verbose=True):
        """
        参数配置
        
        :param cities: 城市坐标 numpy数组 (n_cities, 2)
        :param n_ants: 蚂蚁数量
        :param max_iter: 最大迭代次数
        :param alpha: 信息素重要程度
        :param beta: 启发式因子重要程度
        :param rho: 信息素挥发率
        :param Q: 信息素增量常数
        """
        self.cities = cities
        self.n_cities = len(cities)
        self.n_ants = n_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.verbose = verbose
        
        np.random.seed(random_seed)
        
        # 计算距离矩阵
        self.dist_matrix = self._calc_distance_matrix()
        
        # 结果存储
        self.best_path = None
        self.best_distance = None
        self.history = {'best_distances': [], 'avg_distances': []}
    
    def _calc_distance_matrix(self):
        """计算城市间距离矩阵"""
        n = self.n_cities
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i, j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return dist
    
    def _calc_path_distance(self, path):
        """计算路径总长度"""
        dist = sum(self.dist_matrix[path[i], path[i+1]] for i in range(len(path)-1))
        dist += self.dist_matrix[path[-1], path[0]]  # 回到起点
        return dist
    
    def optimize(self):
        """执行蚁群算法优化"""
        n = self.n_cities
        
        # 初始化信息素矩阵
        tau = np.ones((n, n))
        eta = 1 / (self.dist_matrix + 1e-10)  # 启发式信息（距离倒数）
        
        self.best_path = None
        self.best_distance = float('inf')
        
        if self.verbose:
            print("\n" + "="*50)
            print("🐜 蚁群算法TSP优化开始...")
            print("="*50)
            print(f"  城市数: {n}, 蚂蚁数: {self.n_ants}")
            print(f"  参数: α={self.alpha}, β={self.beta}, ρ={self.rho}")
            print("-"*50)
        
        for it in range(self.max_iter):
            paths = []
            path_dists = []
            
            # 每只蚂蚁构建路径
            for _ in range(self.n_ants):
                path = [np.random.randint(n)]  # 随机起点
                visited = set(path)
                
                while len(path) < n:
                    current = path[-1]
                    unvisited = [i for i in range(n) if i not in visited]
                    
                    # 计算转移概率
                    prob = (tau[current, unvisited] ** self.alpha) * \
                           (eta[current, unvisited] ** self.beta)
                    prob /= prob.sum()
                    
                    # 轮盘赌选择下一个城市
                    next_city = np.random.choice(unvisited, p=prob)
                    path.append(next_city)
                    visited.add(next_city)
                
                path_dist = self._calc_path_distance(path)
                paths.append(path)
                path_dists.append(path_dist)
                
                # 更新全局最优
                if path_dist < self.best_distance:
                    self.best_distance = path_dist
                    self.best_path = path.copy()
            
            # 记录历史
            self.history['best_distances'].append(self.best_distance)
            self.history['avg_distances'].append(np.mean(path_dists))
            
            # 信息素更新
            tau *= (1 - self.rho)  # 挥发
            for i, path in enumerate(paths):
                delta = self.Q / path_dists[i]
                for j in range(n):
                    u, v = path[j], path[(j+1) % n]
                    tau[u, v] += delta
                    tau[v, u] += delta
            
            if self.verbose and (it + 1) % 10 == 0:
                print(f"  迭代 {it+1:3d}: 最短距离 = {self.best_distance:.2f}")
        
        if self.verbose:
            self._print_results()
        
        return self.best_path, self.best_distance
    
    def _print_results(self):
        """打印结果"""
        print("\n" + "="*50)
        print("📊 蚁群算法优化完成")
        print("="*50)
        print(f"  最优路径: {[x+1 for x in self.best_path]}")  # 从1开始编号
        print(f"  最短距离: {self.best_distance:.2f}")
        print("="*50)
    
    def plot_result(self, save_path=None):
        """可视化结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 最优路径图
        ax1 = axes[0]
        path = self.best_path + [self.best_path[0]]  # 闭环
        path_coords = self.cities[path]
        
        ax1.plot(path_coords[:, 0], path_coords[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax1.scatter(self.cities[:, 0], self.cities[:, 1], s=100, c='red', 
                   zorder=5, edgecolors='white', linewidths=2)
        
        for i, city in enumerate(self.cities):
            ax1.annotate(str(i+1), (city[0], city[1]), textcoords="offset points",
                        xytext=(5, 5), fontsize=10, fontweight='bold')
        
        ax1.set_xlabel('X坐标', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Y坐标', fontsize=12, fontweight='bold')
        ax1.set_title(f'(a) 最优路径 (总距离={self.best_distance:.2f})', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 收敛曲线
        ax2 = axes[1]
        ax2.plot(self.history['best_distances'], linewidth=2, 
                color='#27AE60', label='最优距离')
        ax2.plot(self.history['avg_distances'], linewidth=2, 
                color='#E74C3C', alpha=0.7, linestyle='--', label='平均距离')
        ax2.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
        ax2.set_ylabel('路径距离', fontsize=12, fontweight='bold')
        ax2.set_title('(b) 收敛曲线', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("   蚁群算法TSP求解演示")
    print("="*60)
    
    # 1. 生成城市坐标
    np.random.seed(42)
    n_cities = 15
    cities = np.random.uniform(0, 100, (n_cities, 2))
    
    print(f"\n📍 生成 {n_cities} 个城市")
    
    # 2. 蚁群算法求解
    aco = AntColonyTSP(
        cities=cities,
        n_ants=30,
        max_iter=80,
        alpha=1.0,
        beta=3.0,
        rho=0.3,
        Q=100,
        verbose=True
    )
    best_path, best_dist = aco.optimize()
    
    # 3. 可视化
    aco.plot_result()
    
    # 4. 参数敏感性说明
    print("\n" + "="*60)
    print("📖 参数调优建议")
    print("="*60)
    print("""
    α (信息素重要程度): 
      - 增大α → 更依赖历史经验 → 收敛快但易陷入局部最优
      - 建议: 1.0 - 2.0
    
    β (启发式因子重要程度):
      - 增大β → 更贪心选择近距离城市
      - 建议: 2.0 - 5.0
    
    ρ (信息素挥发率):
      - 增大ρ → 遗忘历史更快 → 探索性更强
      - 建议: 0.1 - 0.5
    """)
