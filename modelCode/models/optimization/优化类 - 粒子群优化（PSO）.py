"""
============================================================
粒子群优化算法 (Particle Swarm Optimization, PSO)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：连续函数优化、参数调优
原理：模拟鸟群觅食行为，通过群体协作寻找最优解
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


class PSO:
    """
    粒子群优化算法类
    
    核心公式：
    v(t+1) = w*v(t) + c1*r1*(pbest-x) + c2*r2*(gbest-x)
    x(t+1) = x(t) + v(t+1)
    
    参数说明：
    - w: 惯性权重（控制全局/局部搜索平衡）
    - c1: 个体学习因子（向个体最优学习）
    - c2: 社会学习因子（向全局最优学习）
    """
    
    def __init__(self, objective_func, bounds, dim=2,
                 pop_size=30, max_iter=100,
                 w=0.7, c1=2.0, c2=2.0,
                 random_seed=42, verbose=True):
        """
        参数配置
        
        :param objective_func: 目标函数（最小化）
        :param bounds: 变量范围 [min, max]
        :param dim: 变量维度
        :param pop_size: 粒子数量（建议20-50）
        :param max_iter: 最大迭代次数
        :param w: 惯性权重（0.4-0.9，可线性递减）
        :param c1, c2: 学习因子（通常c1=c2=2）
        """
        self.func = objective_func
        self.bounds = bounds
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
        
        np.random.seed(random_seed)
        
        # 结果存储
        self.best_solution = None
        self.best_value = None
        self.history = {'best_values': [], 'positions': []}
    
    def optimize(self):
        """执行PSO优化"""
        lb, ub = self.bounds
        
        # 初始化粒子位置和速度
        x = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        v = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        
        # 个体最优和全局最优
        p_best = x.copy()
        p_best_val = np.array([self.func(ind) for ind in x])
        g_best_idx = np.argmin(p_best_val)
        g_best = x[g_best_idx].copy()
        g_best_val = p_best_val[g_best_idx]
        
        if self.verbose:
            print("\n" + "="*50)
            print("🐦 PSO粒子群优化开始...")
            print("="*50)
            print(f"  粒子数: {self.pop_size}, 迭代次数: {self.max_iter}")
            print(f"  参数: w={self.w}, c1={self.c1}, c2={self.c2}")
            print("-"*50)
        
        # 迭代优化
        for it in range(self.max_iter):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            
            # 速度更新
            v = (self.w * v + 
                 self.c1 * r1 * (p_best - x) + 
                 self.c2 * r2 * (g_best - x))
            
            # 位置更新
            x = np.clip(x + v, lb, ub)
            
            # 更新个体最优
            current_val = np.array([self.func(ind) for ind in x])
            improved = current_val < p_best_val
            p_best[improved] = x[improved]
            p_best_val[improved] = current_val[improved]
            
            # 更新全局最优
            min_idx = np.argmin(p_best_val)
            if p_best_val[min_idx] < g_best_val:
                g_best = p_best[min_idx].copy()
                g_best_val = p_best_val[min_idx]
            
            self.history['best_values'].append(g_best_val)
            
            if self.verbose and (it + 1) % 20 == 0:
                print(f"  迭代 {it+1:3d}: 最优值 = {g_best_val:.6f}")
        
        self.best_solution = g_best
        self.best_value = g_best_val
        
        if self.verbose:
            self._print_results()
        
        return self.best_solution, self.best_value
    
    def _print_results(self):
        """打印结果"""
        print("\n" + "="*50)
        print("📊 PSO优化完成")
        print("="*50)
        print(f"  最优解: {self.best_solution.round(6)}")
        print(f"  最优值: {self.best_value:.6f}")
        print("="*50)
    
    def plot_convergence(self, save_path=None):
        """绘制收敛曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.history['best_values'], linewidth=2, color='#2E86AB')
        ax.fill_between(range(len(self.history['best_values'])), 
                       self.history['best_values'], alpha=0.3, color='#2E86AB')
        
        ax.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
        ax.set_ylabel('最优函数值', fontsize=12, fontweight='bold')
        ax.set_title('PSO收敛曲线', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标注最终值
        ax.annotate(f'最终值: {self.best_value:.6f}',
                   xy=(len(self.history['best_values'])-1, self.best_value),
                   xytext=(-80, 30), textcoords='offset points',
                   fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 测试函数库
# ============================================================
def rastrigin(x):
    """Rastrigin函数（多峰函数，最小值0）"""
    A = 10
    return A * len(x) + sum([xi**2 - A * np.cos(2 * np.pi * xi) for xi in x])

def sphere(x):
    """Sphere函数（单峰函数，最小值0）"""
    return sum(xi**2 for xi in x)

def rosenbrock(x):
    """Rosenbrock函数（香蕉函数，最小值0）"""
    return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("   PSO粒子群优化算法演示")
    print("="*60)
    
    # 1. 优化Rastrigin函数
    print("\n📍 测试1: Rastrigin函数优化")
    pso = PSO(
        objective_func=rastrigin,
        bounds=[-5.12, 5.12],
        dim=2,
        pop_size=30,
        max_iter=100,
        w=0.7, c1=2.0, c2=2.0,
        verbose=True
    )
    best_sol, best_val = pso.optimize()
    pso.plot_convergence()
    
    # 2. 优化Sphere函数
    print("\n📍 测试2: Sphere函数优化")
    pso2 = PSO(
        objective_func=sphere,
        bounds=[-10, 10],
        dim=3,
        pop_size=40,
        max_iter=80,
        verbose=True
    )
    best_sol2, best_val2 = pso2.optimize()
    pso2.plot_convergence()
    
    print(f"\n✅ 理论最小值均为0，算法找到的最优值越接近0效果越好")
