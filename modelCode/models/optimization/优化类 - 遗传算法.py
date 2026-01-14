"""
============================================================
遗传算法 (Genetic Algorithm, GA)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：组合优化、参数寻优、函数优化
原理：模拟生物进化过程（选择、交叉、变异）
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


class GeneticAlgorithm:
    """
    遗传算法类
    
    核心操作：
    1. 选择：轮盘赌/锦标赛选择优秀个体
    2. 交叉：单点/双点/均匀交叉产生后代
    3. 变异：高斯变异增加多样性
    
    参数说明：
    - crossover_rate: 交叉概率（0.6-0.9）
    - mutation_rate: 变异概率（0.01-0.1）
    """
    
    def __init__(self, objective_func, bounds, dim=2,
                 pop_size=50, max_iter=100,
                 crossover_rate=0.8, mutation_rate=0.1,
                 selection_method='roulette',
                 random_seed=42, verbose=True):
        """
        参数配置
        
        :param objective_func: 目标函数（最小化）
        :param bounds: 变量范围 [min, max]
        :param dim: 变量维度
        :param pop_size: 种群大小（建议30-100）
        :param max_iter: 迭代代数
        :param crossover_rate: 交叉概率
        :param mutation_rate: 变异概率
        :param selection_method: 选择方法 'roulette'/'tournament'
        """
        self.func = objective_func
        self.bounds = bounds
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.verbose = verbose
        
        np.random.seed(random_seed)
        
        # 结果存储
        self.best_solution = None
        self.best_value = None
        self.history = {'best_values': [], 'avg_values': []}
    
    def _fitness(self, population):
        """计算适应度（最小化问题转换为最大化）"""
        values = np.array([self.func(ind) for ind in population])
        # 适应度 = 1/(目标值+小常数)，值越小适应度越高
        return 1 / (values + 1e-10)
    
    def _selection_roulette(self, population, fitness):
        """轮盘赌选择"""
        prob = fitness / fitness.sum()
        indices = np.random.choice(len(population), size=self.pop_size, p=prob)
        return population[indices]
    
    def _selection_tournament(self, population, fitness, k=3):
        """锦标赛选择"""
        selected = []
        for _ in range(self.pop_size):
            candidates = np.random.choice(len(population), k, replace=False)
            winner = candidates[np.argmax(fitness[candidates])]
            selected.append(population[winner])
        return np.array(selected)
    
    def _crossover(self, population):
        """单点交叉"""
        offspring = population.copy()
        for i in range(0, self.pop_size, 2):
            if i + 1 >= self.pop_size:
                break
            if np.random.rand() < self.crossover_rate:
                cross_point = np.random.randint(1, self.dim)
                offspring[i, cross_point:], offspring[i+1, cross_point:] = \
                    population[i+1, cross_point:].copy(), population[i, cross_point:].copy()
        return offspring
    
    def _mutation(self, population):
        """高斯变异"""
        lb, ub = self.bounds
        for i in range(self.pop_size):
            for j in range(self.dim):
                if np.random.rand() < self.mutation_rate:
                    population[i, j] += np.random.normal(0, (ub - lb) * 0.1)
                    population[i, j] = np.clip(population[i, j], lb, ub)
        return population
    
    def optimize(self):
        """执行遗传算法优化"""
        lb, ub = self.bounds
        
        # 初始化种群
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        
        if self.verbose:
            print("\n" + "="*50)
            print("🧬 遗传算法优化开始...")
            print("="*50)
            print(f"  种群大小: {self.pop_size}, 迭代代数: {self.max_iter}")
            print(f"  交叉率: {self.crossover_rate}, 变异率: {self.mutation_rate}")
            print("-"*50)
        
        for gen in range(self.max_iter):
            # 计算适应度
            fitness = self._fitness(population)
            
            # 记录历史
            values = np.array([self.func(ind) for ind in population])
            self.history['best_values'].append(values.min())
            self.history['avg_values'].append(values.mean())
            
            # 选择
            if self.selection_method == 'roulette':
                selected = self._selection_roulette(population, fitness)
            else:
                selected = self._selection_tournament(population, fitness)
            
            # 交叉
            offspring = self._crossover(selected)
            
            # 变异
            offspring = self._mutation(offspring)
            
            # 精英保留：保留最优个体
            best_idx = np.argmin(values)
            worst_idx = np.argmax([self.func(ind) for ind in offspring])
            offspring[worst_idx] = population[best_idx]
            
            population = offspring
            
            if self.verbose and (gen + 1) % 20 == 0:
                print(f"  代数 {gen+1:3d}: 最优值 = {self.history['best_values'][-1]:.6f}")
        
        # 找到最优解
        final_values = np.array([self.func(ind) for ind in population])
        best_idx = np.argmin(final_values)
        self.best_solution = population[best_idx]
        self.best_value = final_values[best_idx]
        
        if self.verbose:
            self._print_results()
        
        return self.best_solution, self.best_value
    
    def _print_results(self):
        """打印结果"""
        print("\n" + "="*50)
        print("📊 遗传算法优化完成")
        print("="*50)
        print(f"  最优解: {self.best_solution.round(6)}")
        print(f"  最优值: {self.best_value:.6f}")
        print("="*50)
    
    def plot_convergence(self, save_path=None):
        """绘制收敛曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.history['best_values'], linewidth=2, 
               color='#27AE60', label='最优值')
        ax.plot(self.history['avg_values'], linewidth=2, 
               color='#E74C3C', alpha=0.7, linestyle='--', label='平均值')
        
        ax.fill_between(range(len(self.history['best_values'])),
                       self.history['best_values'], alpha=0.2, color='#27AE60')
        
        ax.set_xlabel('迭代代数', fontsize=12, fontweight='bold')
        ax.set_ylabel('函数值', fontsize=12, fontweight='bold')
        ax.set_title('遗传算法收敛曲线', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 测试函数
# ============================================================
def sphere(x):
    """Sphere函数（最小值0）"""
    return sum(xi**2 for xi in x)

def rastrigin(x):
    """Rastrigin函数（最小值0）"""
    A = 10
    return A * len(x) + sum([xi**2 - A * np.cos(2 * np.pi * xi) for xi in x])


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("   遗传算法演示")
    print("="*60)
    
    # 1. 优化Sphere函数
    print("\n📍 测试1: Sphere函数优化 f(x) = x1² + x2²")
    ga = GeneticAlgorithm(
        objective_func=sphere,
        bounds=[-10, 10],
        dim=2,
        pop_size=50,
        max_iter=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        verbose=True
    )
    best_sol, best_val = ga.optimize()
    ga.plot_convergence()
    
    # 2. 优化Rastrigin函数
    print("\n📍 测试2: Rastrigin函数优化（多峰函数）")
    ga2 = GeneticAlgorithm(
        objective_func=rastrigin,
        bounds=[-5.12, 5.12],
        dim=3,
        pop_size=80,
        max_iter=150,
        selection_method='tournament',
        verbose=True
    )
    best_sol2, best_val2 = ga2.optimize()
    ga2.plot_convergence()
    
    print(f"\n✅ 理论最小值均为0")
