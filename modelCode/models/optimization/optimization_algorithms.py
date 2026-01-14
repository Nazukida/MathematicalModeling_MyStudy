"""
============================================================
优化算法：粒子群优化 (PSO) + 遗传算法 (GA) + 参数反演
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：函数优化、参数估计、逆问题求解、复杂方程参数反演
原理：智能优化算法搜索最优参数
作者：MCM/ICM Team
日期：2026年1月
============================================================

应用场景：
- 复杂函数最优化
- 模型参数标定（逆问题）
- 机器学习超参数调优
- 工程设计优化
- 资源配置问题

核心算法：
1. PSO：群体智能，模拟鸟群觅食
2. GA：进化算法，模拟自然选择
3. 差分进化 (DE)：连续空间优化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import warnings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from visualization.plot_config import PlotStyleConfig, FigureSaver

PlotStyleConfig.setup_style()
warnings.filterwarnings('ignore')


class BaseOptimizer(ABC):
    """优化算法基类"""
    
    def __init__(self, objective_func, bounds, n_dims=None, 
                 max_iter=100, random_seed=42, verbose=True):
        """
        初始化优化器
        
        :param objective_func: 目标函数 f(x) -> scalar（最小化）
        :param bounds: 参数范围 [(low1, high1), (low2, high2), ...] 或 (low, high)
        :param n_dims: 参数维度（如果bounds是元组则需要指定）
        :param max_iter: 最大迭代次数
        :param random_seed: 随机种子
        :param verbose: 是否打印详细信息
        """
        self.objective_func = objective_func
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.verbose = verbose
        
        np.random.seed(random_seed)
        
        # 处理边界
        if isinstance(bounds, tuple) and len(bounds) == 2:
            if n_dims is None:
                raise ValueError("使用(low, high)格式时需要指定n_dims")
            self.bounds = np.array([bounds] * n_dims)
        else:
            self.bounds = np.array(bounds)
        
        self.n_dims = len(self.bounds)
        self.lower_bounds = self.bounds[:, 0]
        self.upper_bounds = self.bounds[:, 1]
        
        # 结果存储
        self.best_solution = None
        self.best_value = np.inf
        self.history = {
            'best_values': [],
            'mean_values': [],
            'solutions': []
        }
        self.n_evaluations = 0
        
    @abstractmethod
    def optimize(self):
        """执行优化"""
        pass
    
    def _clip_to_bounds(self, x):
        """将解限制在边界内"""
        return np.clip(x, self.lower_bounds, self.upper_bounds)
    
    def _random_init(self, n_particles):
        """随机初始化种群"""
        return np.random.uniform(
            self.lower_bounds, self.upper_bounds, 
            size=(n_particles, self.n_dims)
        )
    
    def _evaluate(self, x):
        """评估目标函数"""
        self.n_evaluations += 1
        return self.objective_func(x)
    
    def _print_header(self, algo_name):
        """打印算法头部"""
        print("\n" + "="*60)
        print(f"🔧 {algo_name} 优化开始")
        print("="*60)
        print(f"  参数维度: {self.n_dims}")
        print(f"  最大迭代: {self.max_iter}")
        print("-"*60)
    
    def _print_progress(self, iteration, best_val, mean_val=None):
        """打印进度"""
        if mean_val:
            print(f"  迭代 {iteration:4d}: 最优 = {best_val:.6f}, 平均 = {mean_val:.6f}")
        else:
            print(f"  迭代 {iteration:4d}: 最优 = {best_val:.6f}")
    
    def _print_results(self, algo_name):
        """打印最终结果"""
        print("-"*60)
        print(f"✅ {algo_name} 优化完成")
        print(f"\n  最优解:")
        for i, (val, (low, high)) in enumerate(zip(self.best_solution, self.bounds)):
            print(f"    x[{i}] = {val:.6f}  ∈ [{low}, {high}]")
        print(f"\n  最优目标值: {self.best_value:.6f}")
        print(f"  函数评估次数: {self.n_evaluations}")
        print("="*60)
    
    def plot_convergence(self, save_path=None):
        """绘制收敛曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(1, len(self.history['best_values']) + 1)
        
        ax.plot(iterations, self.history['best_values'], 
               color=PlotStyleConfig.COLORS['primary'], linewidth=2.5, 
               label='最优值')
        
        if self.history['mean_values']:
            ax.plot(iterations, self.history['mean_values'],
                   color=PlotStyleConfig.COLORS['secondary'], linewidth=2,
                   linestyle='--', label='平均值', alpha=0.7)
        
        ax.axhline(self.best_value, color=PlotStyleConfig.COLORS['danger'],
                  linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
        ax.set_ylabel('目标函数值', fontsize=12, fontweight='bold')
        ax.set_title('优化收敛曲线', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax


class PSO(BaseOptimizer):
    """
    粒子群优化算法 (Particle Swarm Optimization)
    
    核心公式：
    v(t+1) = w*v(t) + c1*r1*(pbest-x) + c2*r2*(gbest-x)
    x(t+1) = x(t) + v(t+1)
    
    特点：
    - 收敛速度快
    - 实现简单
    - 适合连续优化问题
    """
    
    def __init__(self, objective_func, bounds, n_dims=None, 
                 pop_size=30, max_iter=100,
                 w=0.7, c1=1.5, c2=1.5, w_decay=True,
                 random_seed=42, verbose=True):
        """
        初始化PSO
        
        :param pop_size: 粒子数量
        :param w: 惯性权重
        :param c1: 个体学习因子
        :param c2: 社会学习因子
        :param w_decay: 是否使用惯性权重线性递减
        """
        super().__init__(objective_func, bounds, n_dims, max_iter, random_seed, verbose)
        
        self.pop_size = pop_size
        self.w = w
        self.w_init = w
        self.c1 = c1
        self.c2 = c2
        self.w_decay = w_decay
        
    def optimize(self):
        """执行PSO优化"""
        if self.verbose:
            self._print_header("粒子群优化 (PSO)")
        
        # 初始化粒子
        positions = self._random_init(self.pop_size)
        velocities = np.zeros_like(positions)
        
        # 个体最优
        personal_best_pos = positions.copy()
        personal_best_val = np.array([self._evaluate(p) for p in positions])
        
        # 全局最优
        global_best_idx = np.argmin(personal_best_val)
        global_best_pos = positions[global_best_idx].copy()
        global_best_val = personal_best_val[global_best_idx]
        
        # 速度限制
        v_max = 0.2 * (self.upper_bounds - self.lower_bounds)
        
        # 迭代优化
        for it in range(self.max_iter):
            # 惯性权重递减
            if self.w_decay:
                self.w = self.w_init - (self.w_init - 0.4) * (it / self.max_iter)
            
            # 更新速度和位置
            r1 = np.random.rand(self.pop_size, self.n_dims)
            r2 = np.random.rand(self.pop_size, self.n_dims)
            
            velocities = (self.w * velocities + 
                         self.c1 * r1 * (personal_best_pos - positions) +
                         self.c2 * r2 * (global_best_pos - positions))
            
            # 限制速度
            velocities = np.clip(velocities, -v_max, v_max)
            
            # 更新位置
            positions = self._clip_to_bounds(positions + velocities)
            
            # 评估
            fitness = np.array([self._evaluate(p) for p in positions])
            
            # 更新个体最优
            improved = fitness < personal_best_val
            personal_best_pos[improved] = positions[improved]
            personal_best_val[improved] = fitness[improved]
            
            # 更新全局最优
            best_idx = np.argmin(personal_best_val)
            if personal_best_val[best_idx] < global_best_val:
                global_best_val = personal_best_val[best_idx]
                global_best_pos = personal_best_pos[best_idx].copy()
            
            # 记录历史
            self.history['best_values'].append(global_best_val)
            self.history['mean_values'].append(np.mean(fitness))
            
            if self.verbose and (it + 1) % max(1, self.max_iter // 10) == 0:
                self._print_progress(it + 1, global_best_val, np.mean(fitness))
        
        self.best_solution = global_best_pos
        self.best_value = global_best_val
        
        if self.verbose:
            self._print_results("PSO")
        
        return self.best_solution, self.best_value


class GeneticAlgorithm(BaseOptimizer):
    """
    遗传算法 (Genetic Algorithm)
    
    核心操作：
    1. 选择：锦标赛/轮盘赌
    2. 交叉：SBX交叉
    3. 变异：多项式变异
    
    特点：
    - 全局搜索能力强
    - 适合离散和连续问题
    - 可并行化
    """
    
    def __init__(self, objective_func, bounds, n_dims=None,
                 pop_size=50, max_iter=100,
                 crossover_rate=0.9, mutation_rate=0.1,
                 tournament_size=3, elitism=True,
                 random_seed=42, verbose=True):
        """
        初始化GA
        
        :param pop_size: 种群大小
        :param crossover_rate: 交叉概率
        :param mutation_rate: 变异概率
        :param tournament_size: 锦标赛大小
        :param elitism: 是否保留精英
        """
        super().__init__(objective_func, bounds, n_dims, max_iter, random_seed, verbose)
        
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        
    def _tournament_selection(self, population, fitness):
        """锦标赛选择"""
        selected = []
        for _ in range(self.pop_size):
            candidates = np.random.choice(self.pop_size, self.tournament_size, replace=False)
            winner = candidates[np.argmin(fitness[candidates])]
            selected.append(population[winner])
        return np.array(selected)
    
    def _sbx_crossover(self, parent1, parent2, eta=20):
        """模拟二进制交叉 (SBX)"""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = parent1.copy(), parent2.copy()
        
        for i in range(self.n_dims):
            if np.random.rand() < 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    
                    child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                    child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
        
        return self._clip_to_bounds(child1), self._clip_to_bounds(child2)
    
    def _polynomial_mutation(self, individual, eta=20):
        """多项式变异"""
        mutant = individual.copy()
        
        for i in range(self.n_dims):
            if np.random.rand() < self.mutation_rate:
                u = np.random.rand()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                
                mutant[i] += delta * (self.upper_bounds[i] - self.lower_bounds[i])
        
        return self._clip_to_bounds(mutant)
    
    def optimize(self):
        """执行GA优化"""
        if self.verbose:
            self._print_header("遗传算法 (GA)")
        
        # 初始化种群
        population = self._random_init(self.pop_size)
        fitness = np.array([self._evaluate(ind) for ind in population])
        
        # 记录最优
        best_idx = np.argmin(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_value = fitness[best_idx]
        
        # 迭代进化
        for it in range(self.max_iter):
            # 选择
            selected = self._tournament_selection(population, fitness)
            
            # 交叉
            offspring = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = selected[i], selected[min(i+1, self.pop_size-1)]
                c1, c2 = self._sbx_crossover(p1, p2)
                offspring.extend([c1, c2])
            offspring = np.array(offspring[:self.pop_size])
            
            # 变异
            offspring = np.array([self._polynomial_mutation(ind) for ind in offspring])
            
            # 评估子代
            offspring_fitness = np.array([self._evaluate(ind) for ind in offspring])
            
            # 精英保留
            if self.elitism:
                worst_idx = np.argmax(offspring_fitness)
                if self.best_value < offspring_fitness[worst_idx]:
                    offspring[worst_idx] = self.best_solution.copy()
                    offspring_fitness[worst_idx] = self.best_value
            
            # 更新种群
            population = offspring
            fitness = offspring_fitness
            
            # 更新最优
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_value:
                self.best_value = fitness[best_idx]
                self.best_solution = population[best_idx].copy()
            
            # 记录历史
            self.history['best_values'].append(self.best_value)
            self.history['mean_values'].append(np.mean(fitness))
            
            if self.verbose and (it + 1) % max(1, self.max_iter // 10) == 0:
                self._print_progress(it + 1, self.best_value, np.mean(fitness))
        
        if self.verbose:
            self._print_results("GA")
        
        return self.best_solution, self.best_value


class DifferentialEvolution(BaseOptimizer):
    """
    差分进化算法 (Differential Evolution)
    
    核心操作：
    变异: v = x_r1 + F * (x_r2 - x_r3)
    交叉: 二项式交叉
    选择: 贪婪选择
    
    特点：
    - 连续优化效果好
    - 参数少，易调节
    - 适合高维问题
    """
    
    def __init__(self, objective_func, bounds, n_dims=None,
                 pop_size=50, max_iter=100,
                 F=0.8, CR=0.9, strategy='best/1/bin',
                 random_seed=42, verbose=True):
        """
        初始化DE
        
        :param F: 缩放因子 (0.4-1.0)
        :param CR: 交叉概率 (0.5-1.0)
        :param strategy: 变异策略 'rand/1/bin', 'best/1/bin', 'rand/2/bin'
        """
        super().__init__(objective_func, bounds, n_dims, max_iter, random_seed, verbose)
        
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.strategy = strategy
        
    def optimize(self):
        """执行DE优化"""
        if self.verbose:
            self._print_header("差分进化 (DE)")
        
        # 初始化种群
        population = self._random_init(self.pop_size)
        fitness = np.array([self._evaluate(ind) for ind in population])
        
        # 记录最优
        best_idx = np.argmin(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_value = fitness[best_idx]
        
        # 迭代进化
        for it in range(self.max_iter):
            for i in range(self.pop_size):
                # 选择变异个体
                idxs = [j for j in range(self.pop_size) if j != i]
                
                if 'best' in self.strategy:
                    base = self.best_solution
                    r = np.random.choice(idxs, 2, replace=False)
                    mutant = base + self.F * (population[r[0]] - population[r[1]])
                else:
                    r = np.random.choice(idxs, 3, replace=False)
                    mutant = population[r[0]] + self.F * (population[r[1]] - population[r[2]])
                
                mutant = self._clip_to_bounds(mutant)
                
                # 交叉
                trial = population[i].copy()
                j_rand = np.random.randint(self.n_dims)
                for j in range(self.n_dims):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # 选择
                trial_fitness = self._evaluate(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_solution = trial.copy()
            
            # 记录历史
            self.history['best_values'].append(self.best_value)
            self.history['mean_values'].append(np.mean(fitness))
            
            if self.verbose and (it + 1) % max(1, self.max_iter // 10) == 0:
                self._print_progress(it + 1, self.best_value, np.mean(fitness))
        
        if self.verbose:
            self._print_results("DE")
        
        return self.best_solution, self.best_value


class ParameterInversion:
    """
    参数反演/标定工具
    
    场景：给定正向模型和观测数据，反推模型参数
    
    应用：
    - 模型参数标定
    - 系统辨识
    - 逆问题求解
    """
    
    def __init__(self, forward_model, param_bounds, verbose=True):
        """
        初始化参数反演器
        
        :param forward_model: 正向模型 f(params) -> predictions
        :param param_bounds: 参数范围 [(low1, high1), ...]
        """
        self.forward_model = forward_model
        self.param_bounds = param_bounds
        self.verbose = verbose
        self.n_params = len(param_bounds)
        
        self.best_params = None
        self.best_rmse = None
        self.optimizer = None
        
    def objective(self, params, observations, weights=None):
        """目标函数：加权RMSE"""
        predictions = self.forward_model(params)
        residuals = observations - predictions
        
        if weights is None:
            return np.sqrt(np.mean(residuals**2))
        else:
            return np.sqrt(np.average(residuals**2, weights=weights))
    
    def fit(self, observations, method='pso', weights=None, **kwargs):
        """
        执行参数反演
        
        :param observations: 观测数据
        :param method: 优化方法 'pso', 'ga', 'de'
        :param weights: 观测权重（可选）
        :param kwargs: 优化器参数
        """
        # 定义目标函数
        def obj_func(params):
            return self.objective(params, observations, weights)
        
        # 选择优化器
        if method.lower() == 'pso':
            self.optimizer = PSO(obj_func, self.param_bounds, verbose=self.verbose, **kwargs)
        elif method.lower() == 'ga':
            self.optimizer = GeneticAlgorithm(obj_func, self.param_bounds, verbose=self.verbose, **kwargs)
        elif method.lower() == 'de':
            self.optimizer = DifferentialEvolution(obj_func, self.param_bounds, verbose=self.verbose, **kwargs)
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        # 执行优化
        self.best_params, self.best_rmse = self.optimizer.optimize()
        
        return self.best_params, self.best_rmse
    
    def predict(self, params=None):
        """使用参数进行预测"""
        if params is None:
            params = self.best_params
        return self.forward_model(params)
    
    def sensitivity_analysis(self, observations, n_samples=100):
        """
        参数敏感性分析
        
        :return: 各参数的敏感性指标
        """
        sensitivities = []
        
        for i in range(self.n_params):
            # 在最优解附近扰动
            param_range = np.linspace(
                self.param_bounds[i][0], 
                self.param_bounds[i][1], 
                n_samples
            )
            
            rmses = []
            for val in param_range:
                params = self.best_params.copy()
                params[i] = val
                rmse = self.objective(params, observations)
                rmses.append(rmse)
            
            # 敏感性 = RMSE变化范围
            sensitivity = max(rmses) - min(rmses)
            sensitivities.append({
                'param_idx': i,
                'sensitivity': sensitivity,
                'rmse_range': (min(rmses), max(rmses))
            })
        
        if self.verbose:
            print("\n" + "="*55)
            print("📊 参数敏感性分析")
            print("="*55)
            for s in sorted(sensitivities, key=lambda x: -x['sensitivity']):
                print(f"  参数 {s['param_idx']}: 敏感性 = {s['sensitivity']:.4f}")
            print("="*55)
        
        return sensitivities
    
    def plot_fit(self, observations, x=None, save_path=None):
        """绘制拟合效果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        predictions = self.predict()
        
        if x is None:
            x = np.arange(len(observations))
        
        # 拟合对比
        axes[0].scatter(x, observations, color=PlotStyleConfig.COLORS['primary'],
                       s=60, alpha=0.7, label='观测值', edgecolors='white')
        axes[0].plot(x, predictions, color=PlotStyleConfig.COLORS['danger'],
                    linewidth=2.5, label='模型预测')
        axes[0].set_xlabel('X', fontweight='bold')
        axes[0].set_ylabel('Y', fontweight='bold')
        axes[0].set_title('模型拟合效果', fontsize=12, fontweight='bold')
        axes[0].legend()
        
        # 残差图
        residuals = observations - predictions
        axes[1].scatter(predictions, residuals, color=PlotStyleConfig.COLORS['secondary'],
                       s=60, alpha=0.7, edgecolors='white')
        axes[1].axhline(0, color='gray', linestyle='--', linewidth=1.5)
        axes[1].set_xlabel('预测值', fontweight='bold')
        axes[1].set_ylabel('残差', fontweight='bold')
        axes[1].set_title('残差分析', fontsize=12, fontweight='bold')
        
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.suptitle(f'参数反演结果 (RMSE = {self.best_rmse:.4f})', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, axes


# ==================== 标准测试函数 ====================

class BenchmarkFunctions:
    """标准测试函数"""
    
    @staticmethod
    def sphere(x):
        """球函数 - 最简单的单峰函数"""
        return np.sum(x**2)
    
    @staticmethod
    def rastrigin(x):
        """Rastrigin函数 - 多峰函数"""
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def rosenbrock(x):
        """Rosenbrock函数 - 香蕉形山谷"""
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    @staticmethod
    def ackley(x):
        """Ackley函数"""
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e


def compare_optimizers(objective_func, bounds, n_dims, max_iter=100, n_runs=5):
    """
    比较不同优化器的性能
    """
    results = {}
    
    for name, Optimizer in [('PSO', PSO), ('GA', GeneticAlgorithm), ('DE', DifferentialEvolution)]:
        values = []
        for seed in range(n_runs):
            opt = Optimizer(objective_func, bounds, n_dims=n_dims, 
                           max_iter=max_iter, random_seed=seed, verbose=False)
            _, best_val = opt.optimize()
            values.append(best_val)
        
        results[name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'best': np.min(values),
            'worst': np.max(values)
        }
    
    print("\n" + "="*60)
    print("📊 优化器性能比较")
    print("="*60)
    print(f"  {'算法':<8} {'平均值':<12} {'标准差':<12} {'最优':<12} {'最差':<12}")
    print("  " + "-"*52)
    for name, r in results.items():
        print(f"  {name:<8} {r['mean']:<12.6f} {r['std']:<12.6f} {r['best']:<12.6f} {r['worst']:<12.6f}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("🔧 优化算法与参数反演演示")
    print("="*60)
    
    # ================== 示例1: 函数优化 ==================
    print("\n" + "="*60)
    print("示例1: Rastrigin函数优化")
    print("="*60)
    
    # 使用PSO优化Rastrigin函数
    pso = PSO(
        objective_func=BenchmarkFunctions.rastrigin,
        bounds=(-5.12, 5.12),
        n_dims=5,
        pop_size=30,
        max_iter=100
    )
    best_x, best_val = pso.optimize()
    
    fig1, ax1 = pso.plot_convergence()
    plt.show()
    
    # ================== 示例2: 参数反演 ==================
    print("\n" + "="*60)
    print("示例2: 参数反演（逆问题）")
    print("="*60)
    
    # 定义正向模型：y = a*sin(b*x + c) + d
    def forward_model(params):
        a, b, c, d = params
        x = np.linspace(0, 2*np.pi, 50)
        return a * np.sin(b * x + c) + d
    
    # 生成观测数据（真实参数：a=3, b=2, c=0.5, d=1）
    true_params = [3, 2, 0.5, 1]
    x = np.linspace(0, 2*np.pi, 50)
    observations = forward_model(true_params) + np.random.normal(0, 0.2, 50)
    
    print(f"真实参数: a={true_params[0]}, b={true_params[1]}, c={true_params[2]}, d={true_params[3]}")
    
    # 参数反演
    inverter = ParameterInversion(
        forward_model=forward_model,
        param_bounds=[(0, 5), (0, 5), (-np.pi, np.pi), (-5, 5)]
    )
    best_params, rmse = inverter.fit(observations, method='de', max_iter=100)
    
    print(f"\n反演参数: a={best_params[0]:.3f}, b={best_params[1]:.3f}, "
          f"c={best_params[2]:.3f}, d={best_params[3]:.3f}")
    
    fig2, axes2 = inverter.plot_fit(observations, x)
    plt.show()
    
    # 敏感性分析
    inverter.sensitivity_analysis(observations)
    
    # ================== 示例3: 算法比较 ==================
    print("\n" + "="*60)
    print("示例3: 优化算法性能比较")
    print("="*60)
    
    compare_optimizers(BenchmarkFunctions.rastrigin, (-5.12, 5.12), n_dims=10, max_iter=100, n_runs=5)
    
    print("\n✅ 优化算法演示完成!")
