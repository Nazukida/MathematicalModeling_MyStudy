"""
============================================================
NSGA-II 多目标优化算法
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：多目标优化、Pareto前沿求解、权衡分析
原理：基于非支配排序和拥挤度的遗传算法
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


class NSGAII:
    """
    NSGA-II 多目标优化器
    
    核心机制：
    1. 非支配排序 - 确定解的优劣等级
    2. 拥挤度计算 - 保持解的多样性
    3. 精英保留策略 - 保留最优解
    
    关键概念：
    - Pareto支配：解A在所有目标上都不差于B，且至少一个目标严格优于B
    - Pareto前沿：所有非支配解的集合
    - 拥挤度：解在目标空间中的分布密度
    """
    
    def __init__(self, objectives, n_var=2, bounds=(0, 10),
                 pop_size=50, n_generations=100, verbose=True):
        """
        参数配置
        
        :param objectives: 目标函数列表 [f1, f2, ...]（均为最小化）
        :param n_var: 决策变量维度
        :param bounds: 变量范围 (min, max) 或 [(min1,max1), ...]
        :param pop_size: 种群大小
        :param n_generations: 迭代次数
        """
        self.objectives = objectives
        self.n_obj = len(objectives)
        self.n_var = n_var
        
        if isinstance(bounds[0], (int, float)):
            self.bounds = [(bounds[0], bounds[1]) for _ in range(n_var)]
        else:
            self.bounds = bounds
        
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.verbose = verbose
        
        self.population = None
        self.pareto_front = None
        self.pareto_solutions = None
        self.history = []
    
    def _evaluate(self, x):
        """评估所有目标"""
        return [obj(x) for obj in self.objectives]
    
    def _non_dominated_sort(self, objectives):
        """非支配排序"""
        n = len(objectives)
        dominated_by = [[] for _ in range(n)]
        domination_count = [0] * n
        ranks = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                i_better = all(objectives[i][k] <= objectives[j][k] for k in range(self.n_obj))
                i_strictly = any(objectives[i][k] < objectives[j][k] for k in range(self.n_obj))
                
                j_better = all(objectives[j][k] <= objectives[i][k] for k in range(self.n_obj))
                j_strictly = any(objectives[j][k] < objectives[i][k] for k in range(self.n_obj))
                
                if i_better and i_strictly:
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif j_better and j_strictly:
                    dominated_by[j].append(i)
                    domination_count[i] += 1
        
        current_rank = 0
        remaining = set(range(n))
        while remaining:
            front = [i for i in remaining if domination_count[i] == 0]
            if not front:
                break
            for i in front:
                ranks[i] = current_rank
                remaining.remove(i)
                for j in dominated_by[i]:
                    domination_count[j] -= 1
            current_rank += 1
        
        return ranks
    
    def _crowding_distance(self, objectives, indices):
        """计算拥挤度"""
        n = len(indices)
        if n <= 2:
            return {i: float('inf') for i in indices}
        
        distance = {i: 0.0 for i in indices}
        
        for k in range(self.n_obj):
            sorted_idx = sorted(indices, key=lambda x: objectives[x][k])
            distance[sorted_idx[0]] = float('inf')
            distance[sorted_idx[-1]] = float('inf')
            
            f_range = objectives[sorted_idx[-1]][k] - objectives[sorted_idx[0]][k]
            if f_range == 0:
                continue
            
            for i in range(1, n - 1):
                distance[sorted_idx[i]] += (
                    objectives[sorted_idx[i + 1]][k] - objectives[sorted_idx[i - 1]][k]
                ) / f_range
        
        return distance
    
    def _crossover(self, p1, p2, prob=0.9, eta=20):
        """模拟二进制交叉 (SBX)"""
        if np.random.rand() > prob:
            return p1.copy(), p2.copy()
        
        c1, c2 = np.zeros(self.n_var), np.zeros(self.n_var)
        for i in range(self.n_var):
            if np.random.rand() < 0.5:
                c1[i], c2[i] = p1[i], p2[i]
            else:
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        
        return c1, c2
    
    def _mutate(self, x, prob=0.1, eta=20):
        """多项式变异"""
        mutated = x.copy()
        for i in range(self.n_var):
            if np.random.rand() < prob:
                low, high = self.bounds[i]
                delta = (high - low)
                u = np.random.rand()
                if u < 0.5:
                    delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                mutated[i] = x[i] + delta_q * delta
                mutated[i] = np.clip(mutated[i], low, high)
        return mutated
    
    def optimize(self):
        """运行优化"""
        self.population = np.array([
            [np.random.uniform(self.bounds[i][0], self.bounds[i][1]) 
             for i in range(self.n_var)]
            for _ in range(self.pop_size)
        ])
        
        for gen in range(self.n_generations):
            objectives = [self._evaluate(ind) for ind in self.population]
            ranks = self._non_dominated_sort(objectives)
            pareto_idx = [i for i in range(len(ranks)) if ranks[i] == 0]
            self.history.append(len(pareto_idx))
            
            offspring = []
            for _ in range(self.pop_size // 2):
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                p1 = self.population[i1] if ranks[i1] < ranks[i2] else self.population[i2]
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                p2 = self.population[i1] if ranks[i1] < ranks[i2] else self.population[i2]
                
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                offspring.extend([c1, c2])
            
            combined = np.vstack([self.population, offspring])
            combined_obj = [self._evaluate(ind) for ind in combined]
            combined_ranks = self._non_dominated_sort(combined_obj)
            
            fronts = {}
            for i, r in enumerate(combined_ranks):
                fronts.setdefault(r, []).append(i)
            
            new_pop = []
            for r in sorted(fronts.keys()):
                if len(new_pop) + len(fronts[r]) <= self.pop_size:
                    new_pop.extend(fronts[r])
                else:
                    crowd = self._crowding_distance(combined_obj, fronts[r])
                    sorted_front = sorted(fronts[r], key=lambda x: -crowd[x])
                    new_pop.extend(sorted_front[:self.pop_size - len(new_pop)])
                    break
            
            self.population = combined[new_pop]
            
            if self.verbose and (gen + 1) % 20 == 0:
                print(f"  第{gen+1}代: Pareto解数量={len(pareto_idx)}")
        
        final_obj = [self._evaluate(ind) for ind in self.population]
        final_ranks = self._non_dominated_sort(final_obj)
        pareto_idx = [i for i in range(len(final_ranks)) if final_ranks[i] == 0]
        
        self.pareto_solutions = self.population[pareto_idx]
        self.pareto_front = np.array([final_obj[i] for i in pareto_idx])
        
        if self.verbose:
            self._print_results()
        
        return self.pareto_solutions, self.pareto_front
    
    def _print_results(self):
        """打印结果"""
        print("\n" + "="*50)
        print("🎯 NSGA-II 多目标优化结果")
        print("="*50)
        print(f"\n  目标数: {self.n_obj}")
        print(f"  决策变量: {self.n_var}")
        print(f"  种群大小: {self.pop_size}")
        print(f"  迭代次数: {self.n_generations}")
        print(f"\n  Pareto最优解数量: {len(self.pareto_solutions)}")
        print(f"\n  部分Pareto解:")
        for i, (sol, obj) in enumerate(zip(self.pareto_solutions[:5], self.pareto_front[:5])):
            obj_str = ", ".join([f"f{j+1}={v:.4f}" for j, v in enumerate(obj)])
            print(f"    解{i+1}: x={sol.round(3)} → {obj_str}")
        print("="*50)
    
    def plot_pareto_front(self, save_path=None):
        """可视化Pareto前沿"""
        if self.pareto_front is None:
            raise ValueError("请先调用optimize()")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.n_obj == 2:
            ax.scatter(self.pareto_front[:, 0], self.pareto_front[:, 1],
                      c='#E94F37', s=80, edgecolor='white', linewidth=2,
                      alpha=0.8, label='Pareto前沿')
            
            sorted_idx = np.argsort(self.pareto_front[:, 0])
            ax.plot(self.pareto_front[sorted_idx, 0], self.pareto_front[sorted_idx, 1],
                   'k--', alpha=0.3, linewidth=1)
            
            ax.set_xlabel('目标1', fontsize=12, fontweight='bold')
            ax.set_ylabel('目标2', fontsize=12, fontweight='bold')
        else:
            for i, obj in enumerate(self.pareto_front):
                ax.plot(range(self.n_obj), obj, 'o-', alpha=0.5)
            ax.set_xticks(range(self.n_obj))
            ax.set_xticklabels([f'目标{i+1}' for i in range(self.n_obj)])
            ax.set_ylabel('目标值', fontsize=12, fontweight='bold')
        
        ax.set_title(f'NSGA-II Pareto前沿 ({len(self.pareto_front)}个解)',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("   NSGA-II 多目标优化演示 - 成本vs效率权衡")
    print("="*60)
    
    def cost(x):
        """成本函数（越小越好）"""
        return x[0]**2 + x[1]**2
    
    def neg_efficiency(x):
        """效率函数取负（转为最小化）"""
        return -(2*x[0] + 3*x[1] - x[0]*x[1]/5)
    
    nsga = NSGAII(
        objectives=[cost, neg_efficiency],
        n_var=2,
        bounds=(1, 10),
        pop_size=100,
        n_generations=100,
        verbose=True
    )
    
    solutions, pareto_front = nsga.optimize()
    nsga.plot_pareto_front()
    
    print("\n【Pareto解决方案（前5个）】")
    print("-"*50)
    for i, (sol, obj) in enumerate(zip(solutions[:5], pareto_front[:5])):
        cost_val = obj[0]
        efficiency_val = -obj[1]
        print(f"方案{i+1}: x1={sol[0]:.2f}, x2={sol[1]:.2f}")
        print(f"        成本={cost_val:.2f}, 效率={efficiency_val:.2f}")
        print("-"*50)
