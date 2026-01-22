"""
============================================================
优化类模型 (Optimization Models)
包含：粒子群优化(PSO) + 遗传算法(GA) + 蚁群算法(ACO)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：连续/离散优化、TSP问题、调度问题、参数寻优
特点：完整的参数设置、数据预处理、可视化与美化、算法对比
作者：MCM/ICM Team
日期：2026年1月
============================================================

使用场景：
- 函数最优化（单目标/多目标）
- 组合优化（TSP、VRP、调度）
- 参数调优、资源配置
- 路径规划、选址问题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
from datetime import datetime
from abc import ABC, abstractmethod
import time

warnings.filterwarnings('ignore')


# ============================================================
# 第一部分：全局配置与美化设置 (Global Configuration)
# ============================================================

class PlotStyleConfig:
    """图表美化配置类 - 符合学术论文标准"""
    
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'success': '#C73E1D',
        'neutral': '#3B3B3B',
        'background': '#FAFAFA',
        'grid': '#E0E0E0'
    }
    
    PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4C9A', '#1B998B']
    
    # 算法专用配色
    ALGO_COLORS = {
        'PSO': '#2E86AB',
        'GA': '#A23B72',
        'ACO': '#F18F01',
        'SA': '#C73E1D'
    }
    
    @staticmethod
    def setup_style():
        """设置全局绘图风格"""
        plt.style.use('seaborn-v0_8-whitegrid')
        rcParams['figure.figsize'] = (12, 8)
        rcParams['figure.dpi'] = 100
        rcParams['savefig.dpi'] = 300
        rcParams['font.size'] = 11
        rcParams['axes.titlesize'] = 14
        rcParams['axes.labelsize'] = 12
        rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False

PlotStyleConfig.setup_style()


# ============================================================
# 第二部分：测试函数库 (Benchmark Functions)
# ============================================================

class BenchmarkFunctions:
    """标准测试函数库 - 用于算法性能验证"""
    
    @staticmethod
    def sphere(x):
        """球函数 - 最简单的单峰函数
        最优解: f(0,0,...,0) = 0
        """
        return np.sum(x**2)
    
    @staticmethod
    def rastrigin(x):
        """Rastrigin函数 - 多峰函数（测试全局搜索能力）
        最优解: f(0,0,...,0) = 0
        """
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def rosenbrock(x):
        """Rosenbrock函数 - 香蕉形山谷（测试收敛精度）
        最优解: f(1,1,...,1) = 0
        """
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    @staticmethod
    def ackley(x):
        """Ackley函数 - 多峰函数
        最优解: f(0,0,...,0) = 0
        """
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e
    
    @staticmethod
    def griewank(x):
        """Griewank函数
        最优解: f(0,0,...,0) = 0
        """
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
        return sum_term - prod_term + 1
    
    @staticmethod
    def get_function_info():
        """获取函数信息"""
        return {
            'sphere': {'name': 'Sphere', 'bounds': (-5.12, 5.12), 'optimum': 0},
            'rastrigin': {'name': 'Rastrigin', 'bounds': (-5.12, 5.12), 'optimum': 0},
            'rosenbrock': {'name': 'Rosenbrock', 'bounds': (-5, 10), 'optimum': 0},
            'ackley': {'name': 'Ackley', 'bounds': (-32.768, 32.768), 'optimum': 0},
            'griewank': {'name': 'Griewank', 'bounds': (-600, 600), 'optimum': 0}
        }


# ============================================================
# 第三部分：优化算法基类 (Base Optimizer)
# ============================================================

class BaseOptimizer(ABC):
    """优化算法基类"""
    
    def __init__(self, objective_func, bounds, n_dims, 
                 max_iter=100, random_seed=42, verbose=True):
        """
        初始化优化器
        
        :param objective_func: 目标函数（最小化）
        :param bounds: 变量范围 (min, max) 或 [(min1,max1), (min2,max2), ...]
        :param n_dims: 变量维度
        :param max_iter: 最大迭代次数
        :param random_seed: 随机种子
        :param verbose: 是否打印详细信息
        """
        self.objective_func = objective_func
        self.n_dims = n_dims
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.verbose = verbose
        
        # 处理边界
        if isinstance(bounds, tuple):
            self.bounds = np.array([bounds] * n_dims)
        else:
            self.bounds = np.array(bounds)
        
        np.random.seed(random_seed)
        
        # 结果记录
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'iteration': []
        }
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_evaluations': 0
        }
    
    @abstractmethod
    def optimize(self):
        """执行优化（子类实现）"""
        pass
    
    def _evaluate(self, x):
        """评估个体适应度"""
        self.stats['total_evaluations'] += 1
        return self.objective_func(x)
    
    def _record_history(self, iteration, best_fit, mean_fit):
        """记录历史"""
        self.history['iteration'].append(iteration)
        self.history['best_fitness'].append(best_fit)
        self.history['mean_fitness'].append(mean_fit)
    
    def get_results(self):
        """获取优化结果"""
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'stats': self.stats
        }


# ============================================================
# 第四部分：粒子群优化算法 (PSO)
# ============================================================

class ParticleSwarmOptimization(BaseOptimizer):
    """
    粒子群优化算法 (Particle Swarm Optimization)
    
    原理：
    模拟鸟群觅食行为，每个粒子根据自身经验（个体最优）
    和群体经验（全局最优）调整飞行速度和方向。
    
    速度更新公式：
    v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
    
    参数说明：
    - w: 惯性权重，控制搜索惯性
    - c1: 认知系数，个体学习因子
    - c2: 社会系数，群体学习因子
    """
    
    def __init__(self, objective_func, bounds, n_dims,
                 pop_size=30, max_iter=100,
                 w=0.7, c1=2.0, c2=2.0,
                 w_decay=True, w_min=0.4, w_max=0.9,
                 velocity_clamp=0.2,
                 random_seed=42, verbose=True):
        """
        参数配置说明
        
        核心参数：
        :param pop_size: 种群大小（粒子数量）
            - 建议：20-50，复杂问题可增加
            
        :param w: 惯性权重
            - 范围：0.4-0.9
            - 大w：全局搜索能力强
            - 小w：局部搜索能力强
            
        :param c1: 认知系数（个体学习）
            - 通常：1.5-2.5
            
        :param c2: 社会系数（群体学习）
            - 通常：1.5-2.5
            - c1+c2 ≈ 4 效果较好
        
        高级参数：
        :param w_decay: 是否启用权重衰减
        :param velocity_clamp: 速度限制（相对于搜索范围的比例）
        """
        super().__init__(objective_func, bounds, n_dims, max_iter, random_seed, verbose)
        
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.w_decay = w_decay
        self.w_min = w_min
        self.w_max = w_max
        self.velocity_clamp = velocity_clamp
        
        # 粒子状态
        self.positions = None
        self.velocities = None
        self.pbest_positions = None
        self.pbest_fitness = None
        self.gbest_position = None
        self.gbest_fitness = float('inf')
    
    def _initialize(self):
        """初始化粒子群"""
        # 位置初始化
        self.positions = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            (self.pop_size, self.n_dims)
        )
        
        # 速度初始化
        velocity_range = (self.bounds[:, 1] - self.bounds[:, 0]) * self.velocity_clamp
        self.velocities = np.random.uniform(-velocity_range, velocity_range,
                                            (self.pop_size, self.n_dims))
        
        # 个体最优初始化
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.array([self._evaluate(p) for p in self.positions])
        
        # 全局最优初始化
        best_idx = np.argmin(self.pbest_fitness)
        self.gbest_position = self.pbest_positions[best_idx].copy()
        self.gbest_fitness = self.pbest_fitness[best_idx]
    
    def optimize(self):
        """执行PSO优化"""
        self.stats['start_time'] = time.time()
        
        self._initialize()
        
        if self.verbose:
            print("\n" + "="*60)
            print("🔄 粒子群优化 (PSO) 开始...")
            print("="*60)
            print(f"  种群大小: {self.pop_size}")
            print(f"  最大迭代: {self.max_iter}")
            print(f"  惯性权重: {self.w} (衰减: {self.w_decay})")
            print(f"  学习因子: c1={self.c1}, c2={self.c2}")
            print("-"*60)
        
        for iteration in range(self.max_iter):
            # 自适应惯性权重
            if self.w_decay:
                w = self.w_max - (self.w_max - self.w_min) * iteration / self.max_iter
            else:
                w = self.w
            
            # 更新速度和位置
            r1 = np.random.rand(self.pop_size, self.n_dims)
            r2 = np.random.rand(self.pop_size, self.n_dims)
            
            cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
            social = self.c2 * r2 * (self.gbest_position - self.positions)
            self.velocities = w * self.velocities + cognitive + social
            
            # 速度限制
            velocity_range = (self.bounds[:, 1] - self.bounds[:, 0]) * self.velocity_clamp
            self.velocities = np.clip(self.velocities, -velocity_range, velocity_range)
            
            # 更新位置
            self.positions = self.positions + self.velocities
            self.positions = np.clip(self.positions, self.bounds[:, 0], self.bounds[:, 1])
            
            # 评估适应度
            current_fitness = np.array([self._evaluate(p) for p in self.positions])
            
            # 更新个体最优
            improved = current_fitness < self.pbest_fitness
            self.pbest_positions[improved] = self.positions[improved]
            self.pbest_fitness[improved] = current_fitness[improved]
            
            # 更新全局最优
            best_idx = np.argmin(self.pbest_fitness)
            if self.pbest_fitness[best_idx] < self.gbest_fitness:
                self.gbest_position = self.pbest_positions[best_idx].copy()
                self.gbest_fitness = self.pbest_fitness[best_idx]
            
            # 记录历史
            self._record_history(iteration, self.gbest_fitness, current_fitness.mean())
            
            if self.verbose and (iteration + 1) % 20 == 0:
                print(f"  Iter {iteration+1:4d}: Best = {self.gbest_fitness:.6f}, "
                      f"Mean = {current_fitness.mean():.6f}")
        
        self.best_solution = self.gbest_position
        self.best_fitness = self.gbest_fitness
        self.stats['end_time'] = time.time()
        
        if self.verbose:
            self._print_summary()
        
        return self.best_solution, self.best_fitness
    
    def _print_summary(self):
        """打印结果摘要"""
        elapsed = self.stats['end_time'] - self.stats['start_time']
        print("\n" + "="*60)
        print("📊 PSO 优化完成")
        print("="*60)
        print(f"  最优解: {self.best_solution}")
        print(f"  最优值: {self.best_fitness:.8f}")
        print(f"  运行时间: {elapsed:.2f} 秒")
        print(f"  函数评估次数: {self.stats['total_evaluations']}")
        print("="*60)


# ============================================================
# 第五部分：模拟退火算法 (SA)
# ============================================================

class SimulatedAnnealing(BaseOptimizer):
    """
    模拟退火算法 (Simulated Annealing)
    
    原理：
    模拟金属退火过程，在高温时接受较差的解以跳出局部最优，
    随着温度降低逐渐趋于稳定，最终收敛到全局最优解附近。
    
    核心机制：
    - Metropolis准则：以概率 exp(-ΔE/T) 接受劣解
    - 降温策略：T(k+1) = α * T(k)
    
    参数说明：
    - T0: 初始温度，决定初始接受概率
    - T_min: 终止温度，算法终止条件
    - alpha: 降温系数，控制降温速度
    - max_iter_per_temp: 每个温度下的迭代次数
    """
    
    def __init__(self, objective_func, bounds, n_dims,
                 initial_temp=100.0, min_temp=1e-8, cooling_rate=0.95,
                 max_iter=1000, max_iter_per_temp=10,
                 step_size=None, adaptive_step=True,
                 random_seed=42, verbose=True):
        """
        参数配置说明
        
        核心参数：
        :param initial_temp: 初始温度
            - 建议：使初始接受概率约为0.8
            - 经验公式：T0 ≈ -Δf_avg / ln(0.8)
            
        :param min_temp: 最低温度（终止条件）
            - 建议：1e-8 ~ 1e-6
            
        :param cooling_rate: 降温系数 (α)
            - 范围：0.9 ~ 0.99
            - 小α：降温快，可能错过最优
            - 大α：降温慢，精度高但耗时
            
        :param max_iter_per_temp: 每个温度下的迭代次数
            - 建议：10 ~ 100，与问题维度相关
        
        高级参数：
        :param step_size: 扰动步长（None则自动计算）
        :param adaptive_step: 是否自适应调整步长
        """
        super().__init__(objective_func, bounds, n_dims, max_iter, random_seed, verbose)
        
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.max_iter_per_temp = max_iter_per_temp
        self.adaptive_step = adaptive_step
        
        # 自动计算步长
        if step_size is None:
            self.step_size = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
        else:
            self.step_size = np.full(n_dims, step_size) if np.isscalar(step_size) else np.array(step_size)
        
        # 当前状态
        self.current_solution = None
        self.current_fitness = float('inf')
        self.temperature = initial_temp
        
        # 额外统计
        self.stats['accepted_moves'] = 0
        self.stats['rejected_moves'] = 0
        self.stats['temperatures'] = []
    
    def _initialize(self):
        """初始化解"""
        self.current_solution = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], self.n_dims
        )
        self.current_fitness = self._evaluate(self.current_solution)
        
        # 初始化最优解
        self.best_solution = self.current_solution.copy()
        self.best_fitness = self.current_fitness
        self.temperature = self.initial_temp
    
    def _generate_neighbor(self):
        """生成邻域解"""
        neighbor = self.current_solution.copy()
        
        # 随机选择扰动方式
        if np.random.rand() < 0.5:
            # 单维度扰动
            idx = np.random.randint(self.n_dims)
            neighbor[idx] += np.random.uniform(-self.step_size[idx], self.step_size[idx])
        else:
            # 多维度扰动
            perturbation = np.random.uniform(-self.step_size, self.step_size)
            neighbor += perturbation * np.random.rand(self.n_dims)
        
        # 边界处理
        neighbor = np.clip(neighbor, self.bounds[:, 0], self.bounds[:, 1])
        
        return neighbor
    
    def _metropolis_criterion(self, delta):
        """Metropolis接受准则"""
        if delta < 0:
            return True  # 更优解，直接接受
        else:
            # 以概率 exp(-delta/T) 接受劣解
            probability = np.exp(-delta / self.temperature)
            return np.random.rand() < probability
    
    def optimize(self):
        """执行SA优化"""
        self.stats['start_time'] = time.time()
        
        self._initialize()
        
        if self.verbose:
            print("\n" + "="*60)
            print("🔥 模拟退火算法 (SA) 开始...")
            print("="*60)
            print(f"  初始温度: {self.initial_temp}")
            print(f"  终止温度: {self.min_temp}")
            print(f"  降温系数: {self.cooling_rate}")
            print(f"  每温度迭代: {self.max_iter_per_temp}")
            print("-"*60)
        
        iteration = 0
        temp_iteration = 0
        
        while self.temperature > self.min_temp and iteration < self.max_iter:
            for _ in range(self.max_iter_per_temp):
                if iteration >= self.max_iter:
                    break
                
                # 生成邻域解
                neighbor = self._generate_neighbor()
                neighbor_fitness = self._evaluate(neighbor)
                
                # 计算能量差
                delta = neighbor_fitness - self.current_fitness
                
                # Metropolis准则判断
                if self._metropolis_criterion(delta):
                    self.current_solution = neighbor.copy()
                    self.current_fitness = neighbor_fitness
                    self.stats['accepted_moves'] += 1
                    
                    # 更新最优解
                    if self.current_fitness < self.best_fitness:
                        self.best_solution = self.current_solution.copy()
                        self.best_fitness = self.current_fitness
                else:
                    self.stats['rejected_moves'] += 1
                
                iteration += 1
            
            # 记录历史
            self._record_history(temp_iteration, self.best_fitness, self.current_fitness)
            self.stats['temperatures'].append(self.temperature)
            
            # 自适应步长调整
            if self.adaptive_step and temp_iteration > 0 and temp_iteration % 10 == 0:
                accept_ratio = self.stats['accepted_moves'] / (
                    self.stats['accepted_moves'] + self.stats['rejected_moves'] + 1e-10
                )
                if accept_ratio > 0.5:
                    self.step_size *= 1.1  # 接受率高，增大步长
                elif accept_ratio < 0.2:
                    self.step_size *= 0.9  # 接受率低，减小步长
                self.step_size = np.clip(self.step_size, 
                                         (self.bounds[:, 1] - self.bounds[:, 0]) * 0.001,
                                         (self.bounds[:, 1] - self.bounds[:, 0]) * 0.5)
            
            # 降温
            self.temperature *= self.cooling_rate
            temp_iteration += 1
            
            if self.verbose and temp_iteration % 20 == 0:
                print(f"  Temp={self.temperature:.4e}: Best = {self.best_fitness:.6f}, "
                      f"Current = {self.current_fitness:.6f}")
        
        self.stats['end_time'] = time.time()
        
        if self.verbose:
            self._print_summary()
        
        return self.best_solution, self.best_fitness
    
    def _print_summary(self):
        """打印结果摘要"""
        elapsed = self.stats['end_time'] - self.stats['start_time']
        total_moves = self.stats['accepted_moves'] + self.stats['rejected_moves']
        accept_ratio = self.stats['accepted_moves'] / (total_moves + 1e-10) * 100
        
        print("\n" + "="*60)
        print("📊 SA 优化完成")
        print("="*60)
        print(f"  最优解: {self.best_solution}")
        print(f"  最优值: {self.best_fitness:.8f}")
        print(f"  运行时间: {elapsed:.2f} 秒")
        print(f"  函数评估次数: {self.stats['total_evaluations']}")
        print(f"  接受率: {accept_ratio:.1f}% ({self.stats['accepted_moves']}/{total_moves})")
        print(f"  最终温度: {self.temperature:.4e}")
        print("="*60)


# ============================================================
# 第五部分(续)：模拟退火TSP版本 (SA-TSP)
# ============================================================

class SimulatedAnnealingTSP:
    """
    模拟退火算法 - TSP专用版本
    
    原理：
    使用模拟退火求解旅行商问题，邻域操作采用
    2-opt交换或随机插入等方式。
    """
    
    def __init__(self, cities,
                 initial_temp=1000.0, min_temp=1e-6, cooling_rate=0.995,
                 max_iter_per_temp=100,
                 random_seed=42, verbose=True):
        """
        :param cities: 城市坐标 (n_cities, 2)
        :param initial_temp: 初始温度
        :param min_temp: 终止温度
        :param cooling_rate: 降温系数
        :param max_iter_per_temp: 每温度迭代次数
        """
        self.cities = np.array(cities)
        self.n_cities = len(cities)
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.max_iter_per_temp = max_iter_per_temp
        self.random_seed = random_seed
        self.verbose = verbose
        
        np.random.seed(random_seed)
        
        # 计算距离矩阵
        self.distance_matrix = self._compute_distance_matrix()
        
        # 结果
        self.best_path = None
        self.best_distance = float('inf')
        self.history = {
            'best_distance': [],
            'current_distance': [],
            'temperature': [],
            'iteration': []
        }
    
    def _compute_distance_matrix(self):
        """计算距离矩阵"""
        n = self.n_cities
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i, j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return dist
    
    def _calculate_distance(self, path):
        """计算路径总距离"""
        distance = 0
        for i in range(len(path)):
            distance += self.distance_matrix[path[i], path[(i+1) % len(path)]]
        return distance
    
    def _generate_neighbor(self, path):
        """生成邻域解 - 使用多种邻域操作"""
        new_path = path.copy()
        operation = np.random.choice(['2opt', 'insert', 'swap'])
        
        if operation == '2opt':
            # 2-opt: 反转一段路径
            i, j = sorted(np.random.choice(len(path), 2, replace=False))
            new_path[i:j+1] = new_path[i:j+1][::-1]
            
        elif operation == 'insert':
            # 插入操作: 将一个城市移到另一个位置
            i = np.random.randint(len(path))
            j = np.random.randint(len(path))
            city = new_path.pop(i)
            new_path.insert(j, city)
            
        else:  # swap
            # 交换两个城市
            i, j = np.random.choice(len(path), 2, replace=False)
            new_path[i], new_path[j] = new_path[j], new_path[i]
        
        return new_path
    
    def optimize(self):
        """执行SA-TSP优化"""
        if self.verbose:
            print("\n" + "="*60)
            print("🔥 模拟退火算法-TSP (SA-TSP) 开始...")
            print("="*60)
            print(f"  城市数量: {self.n_cities}")
            print(f"  初始温度: {self.initial_temp}")
            print(f"  降温系数: {self.cooling_rate}")
            print("-"*60)
        
        # 初始化：随机路径
        current_path = list(range(self.n_cities))
        np.random.shuffle(current_path)
        current_distance = self._calculate_distance(current_path)
        
        self.best_path = current_path.copy()
        self.best_distance = current_distance
        
        temperature = self.initial_temp
        iteration = 0
        
        while temperature > self.min_temp:
            for _ in range(self.max_iter_per_temp):
                # 生成邻域解
                new_path = self._generate_neighbor(current_path)
                new_distance = self._calculate_distance(new_path)
                
                # 计算能量差
                delta = new_distance - current_distance
                
                # Metropolis准则
                if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                    current_path = new_path
                    current_distance = new_distance
                    
                    if current_distance < self.best_distance:
                        self.best_path = current_path.copy()
                        self.best_distance = current_distance
            
            # 记录历史
            self.history['iteration'].append(iteration)
            self.history['best_distance'].append(self.best_distance)
            self.history['current_distance'].append(current_distance)
            self.history['temperature'].append(temperature)
            
            # 降温
            temperature *= self.cooling_rate
            iteration += 1
            
            if self.verbose and iteration % 50 == 0:
                print(f"  Iter {iteration:4d}: T={temperature:.2e}, "
                      f"Best={self.best_distance:.2f}, Current={current_distance:.2f}")
        
        if self.verbose:
            self._print_summary()
        
        return self.best_path, self.best_distance
    
    def _print_summary(self):
        """打印结果摘要"""
        print("\n" + "="*60)
        print("📊 SA-TSP 优化完成")
        print("="*60)
        print(f"  最优路径: {[x+1 for x in self.best_path]}")
        print(f"  最短距离: {self.best_distance:.4f}")
        print("="*60)
    
    def get_results(self):
        """获取结果"""
        return {
            'best_path': self.best_path,
            'best_distance': self.best_distance,
            'history': self.history
        }


# ============================================================
# 第七部分：遗传算法 (GA)
# ============================================================

class GeneticAlgorithm(BaseOptimizer):
    """
    遗传算法 (Genetic Algorithm)
    
    原理：
    模拟自然选择和遗传变异过程，通过选择、交叉、变异
    操作进化出最优解。
    
    流程：
    1. 初始化种群
    2. 适应度评估
    3. 选择（轮盘赌/锦标赛）
    4. 交叉（单点/两点/均匀）
    5. 变异（高斯/均匀）
    6. 重复2-5直到收敛
    """
    
    def __init__(self, objective_func, bounds, n_dims,
                 pop_size=50, max_iter=100,
                 crossover_rate=0.8, mutation_rate=0.1,
                 selection_method='tournament', tournament_size=3,
                 crossover_method='uniform', mutation_scale=0.1,
                 elitism=True, elite_size=2,
                 random_seed=42, verbose=True):
        """
        参数配置说明
        
        核心参数：
        :param pop_size: 种群大小
            - 建议：50-200
            
        :param crossover_rate: 交叉概率
            - 范围：0.6-0.9
            - 过低：进化缓慢
            - 过高：可能破坏好的基因
            
        :param mutation_rate: 变异概率
            - 范围：0.01-0.2
            - 过低：容易早熟收敛
            - 过高：退化为随机搜索
        
        高级参数：
        :param selection_method: 选择方法
            - 'roulette': 轮盘赌选择
            - 'tournament': 锦标赛选择（推荐）
            
        :param crossover_method: 交叉方法
            - 'single': 单点交叉
            - 'two_point': 两点交叉
            - 'uniform': 均匀交叉
            
        :param elitism: 是否保留精英个体
        """
        super().__init__(objective_func, bounds, n_dims, max_iter, random_seed, verbose)
        
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.crossover_method = crossover_method
        self.mutation_scale = mutation_scale
        self.elitism = elitism
        self.elite_size = elite_size
        
        self.population = None
        self.fitness = None
    
    def _initialize(self):
        """初始化种群"""
        self.population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            (self.pop_size, self.n_dims)
        )
        self.fitness = np.array([self._evaluate(ind) for ind in self.population])
    
    def _selection(self):
        """选择操作"""
        if self.selection_method == 'roulette':
            # 轮盘赌选择（适应度转换为概率）
            fitness_inv = 1 / (self.fitness + 1e-10)
            prob = fitness_inv / fitness_inv.sum()
            selected_idx = np.random.choice(self.pop_size, size=self.pop_size, p=prob)
            
        elif self.selection_method == 'tournament':
            # 锦标赛选择
            selected_idx = []
            for _ in range(self.pop_size):
                candidates = np.random.choice(self.pop_size, size=self.tournament_size, replace=False)
                winner = candidates[np.argmin(self.fitness[candidates])]
                selected_idx.append(winner)
            selected_idx = np.array(selected_idx)
        
        return self.population[selected_idx].copy()
    
    def _crossover(self, parents):
        """交叉操作"""
        offspring = parents.copy()
        
        for i in range(0, self.pop_size - 1, 2):
            if np.random.rand() < self.crossover_rate:
                p1, p2 = offspring[i], offspring[i+1]
                
                if self.crossover_method == 'single':
                    # 单点交叉
                    point = np.random.randint(1, self.n_dims)
                    offspring[i] = np.concatenate([p1[:point], p2[point:]])
                    offspring[i+1] = np.concatenate([p2[:point], p1[point:]])
                    
                elif self.crossover_method == 'two_point':
                    # 两点交叉
                    points = sorted(np.random.choice(self.n_dims, 2, replace=False))
                    offspring[i][points[0]:points[1]] = p2[points[0]:points[1]]
                    offspring[i+1][points[0]:points[1]] = p1[points[0]:points[1]]
                    
                elif self.crossover_method == 'uniform':
                    # 均匀交叉
                    mask = np.random.rand(self.n_dims) < 0.5
                    offspring[i][mask] = p2[mask]
                    offspring[i+1][mask] = p1[mask]
        
        return offspring
    
    def _mutation(self, offspring):
        """变异操作"""
        for i in range(self.pop_size):
            for j in range(self.n_dims):
                if np.random.rand() < self.mutation_rate:
                    # 高斯变异
                    scale = (self.bounds[j, 1] - self.bounds[j, 0]) * self.mutation_scale
                    offspring[i, j] += np.random.normal(0, scale)
                    offspring[i, j] = np.clip(offspring[i, j], 
                                               self.bounds[j, 0], self.bounds[j, 1])
        return offspring
    
    def optimize(self):
        """执行GA优化"""
        self.stats['start_time'] = time.time()
        
        self._initialize()
        
        if self.verbose:
            print("\n" + "="*60)
            print("🧬 遗传算法 (GA) 开始...")
            print("="*60)
            print(f"  种群大小: {self.pop_size}")
            print(f"  最大迭代: {self.max_iter}")
            print(f"  交叉率: {self.crossover_rate}, 变异率: {self.mutation_rate}")
            print(f"  选择方法: {self.selection_method}")
            print("-"*60)
        
        for iteration in range(self.max_iter):
            # 精英保留
            if self.elitism:
                elite_idx = np.argsort(self.fitness)[:self.elite_size]
                elites = self.population[elite_idx].copy()
                elite_fitness = self.fitness[elite_idx].copy()
            
            # 遗传操作
            selected = self._selection()
            offspring = self._crossover(selected)
            offspring = self._mutation(offspring)
            
            # 评估新种群
            new_fitness = np.array([self._evaluate(ind) for ind in offspring])
            
            # 精英替换
            if self.elitism:
                worst_idx = np.argsort(new_fitness)[-self.elite_size:]
                offspring[worst_idx] = elites
                new_fitness[worst_idx] = elite_fitness
            
            self.population = offspring
            self.fitness = new_fitness
            
            # 更新最优
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_solution = self.population[best_idx].copy()
                self.best_fitness = self.fitness[best_idx]
            
            # 记录历史
            self._record_history(iteration, self.best_fitness, self.fitness.mean())
            
            if self.verbose and (iteration + 1) % 20 == 0:
                print(f"  Iter {iteration+1:4d}: Best = {self.best_fitness:.6f}, "
                      f"Mean = {self.fitness.mean():.6f}")
        
        self.stats['end_time'] = time.time()
        
        if self.verbose:
            self._print_summary()
        
        return self.best_solution, self.best_fitness
    
    def _print_summary(self):
        """打印结果摘要"""
        elapsed = self.stats['end_time'] - self.stats['start_time']
        print("\n" + "="*60)
        print("📊 GA 优化完成")
        print("="*60)
        print(f"  最优解: {self.best_solution}")
        print(f"  最优值: {self.best_fitness:.8f}")
        print(f"  运行时间: {elapsed:.2f} 秒")
        print(f"  函数评估次数: {self.stats['total_evaluations']}")
        print("="*60)


# ============================================================
# 第八部分：蚁群算法 (ACO) - TSP专用
# ============================================================

class AntColonyOptimization:
    """
    蚁群算法 (Ant Colony Optimization) - TSP问题
    
    原理：
    模拟蚂蚁觅食行为，通过信息素的释放和蒸发机制
    找到最短路径。
    
    信息素更新：
    τ(t+1) = (1-ρ)*τ(t) + Δτ
    
    转移概率：
    P_ij = [τ_ij^α * η_ij^β] / Σ[τ_ik^α * η_ik^β]
    """
    
    def __init__(self, cities, 
                 n_ants=30, max_iter=100,
                 alpha=1.0, beta=2.0, rho=0.5, Q=100,
                 random_seed=42, verbose=True):
        """
        参数配置说明
        
        :param cities: 城市坐标 (n_cities, 2)
        
        核心参数：
        :param n_ants: 蚂蚁数量
            - 建议：与城市数量相当或更多
            
        :param alpha: 信息素重要性
            - 范围：1-5
            - 大α：更依赖历史经验
            
        :param beta: 启发式因子重要性
            - 范围：2-5
            - 大β：更贪心地选择近距离城市
            
        :param rho: 信息素挥发系数
            - 范围：0.1-0.5
            - 大ρ：更新更快，但可能丢失好路径
            
        :param Q: 信息素增量系数
        """
        self.cities = np.array(cities)
        self.n_cities = len(cities)
        self.n_ants = n_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.random_seed = random_seed
        self.verbose = verbose
        
        np.random.seed(random_seed)
        
        # 计算距离矩阵
        self.distance_matrix = self._compute_distance_matrix()
        
        # 信息素矩阵
        self.pheromone = np.ones((self.n_cities, self.n_cities))
        
        # 启发式信息（距离倒数）
        self.eta = 1 / (self.distance_matrix + 1e-10)
        
        # 结果
        self.best_path = None
        self.best_distance = float('inf')
        self.history = {
            'best_distance': [],
            'mean_distance': [],
            'iteration': []
        }
    
    def _compute_distance_matrix(self):
        """计算距离矩阵"""
        n = self.n_cities
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i, j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return dist
    
    def _construct_path(self):
        """单只蚂蚁构建路径"""
        path = [np.random.randint(self.n_cities)]
        visited = set(path)
        
        while len(path) < self.n_cities:
            current = path[-1]
            unvisited = [i for i in range(self.n_cities) if i not in visited]
            
            # 计算转移概率
            probabilities = []
            for j in unvisited:
                p = (self.pheromone[current, j] ** self.alpha) * \
                    (self.eta[current, j] ** self.beta)
                probabilities.append(p)
            
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()
            
            # 选择下一个城市
            next_city = np.random.choice(unvisited, p=probabilities)
            path.append(next_city)
            visited.add(next_city)
        
        return path
    
    def _calculate_distance(self, path):
        """计算路径总距离"""
        distance = 0
        for i in range(len(path)):
            distance += self.distance_matrix[path[i], path[(i+1) % len(path)]]
        return distance
    
    def _update_pheromone(self, paths, distances):
        """更新信息素"""
        # 信息素蒸发
        self.pheromone *= (1 - self.rho)
        
        # 信息素增加
        for path, dist in zip(paths, distances):
            delta = self.Q / dist
            for i in range(len(path)):
                u, v = path[i], path[(i+1) % len(path)]
                self.pheromone[u, v] += delta
                self.pheromone[v, u] += delta
    
    def optimize(self):
        """执行ACO优化"""
        if self.verbose:
            print("\n" + "="*60)
            print("🐜 蚁群算法 (ACO) 开始...")
            print("="*60)
            print(f"  城市数量: {self.n_cities}")
            print(f"  蚂蚁数量: {self.n_ants}")
            print(f"  最大迭代: {self.max_iter}")
            print(f"  α={self.alpha}, β={self.beta}, ρ={self.rho}")
            print("-"*60)
        
        for iteration in range(self.max_iter):
            paths = []
            distances = []
            
            # 每只蚂蚁构建路径
            for _ in range(self.n_ants):
                path = self._construct_path()
                distance = self._calculate_distance(path)
                paths.append(path)
                distances.append(distance)
                
                # 更新最优
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path.copy()
            
            # 更新信息素
            self._update_pheromone(paths, distances)
            
            # 记录历史
            self.history['iteration'].append(iteration)
            self.history['best_distance'].append(self.best_distance)
            self.history['mean_distance'].append(np.mean(distances))
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"  Iter {iteration+1:4d}: Best = {self.best_distance:.2f}, "
                      f"Mean = {np.mean(distances):.2f}")
        
        if self.verbose:
            self._print_summary()
        
        return self.best_path, self.best_distance
    
    def _print_summary(self):
        """打印结果摘要"""
        print("\n" + "="*60)
        print("📊 ACO 优化完成")
        print("="*60)
        print(f"  最优路径: {[x+1 for x in self.best_path]}")
        print(f"  最短距离: {self.best_distance:.4f}")
        print("="*60)
    
    def get_results(self):
        """获取结果"""
        return {
            'best_path': self.best_path,
            'best_distance': self.best_distance,
            'history': self.history
        }


# ============================================================
# 第九部分：可视化模块 (Visualization)
# ============================================================

class OptimizationVisualizer:
    """优化算法可视化类"""
    
    def __init__(self):
        self.colors = PlotStyleConfig.ALGO_COLORS
    
    def plot_convergence(self, optimizer, title=None, save_path=None):
        """绘制收敛曲线"""
        history = optimizer.history if hasattr(optimizer, 'history') else optimizer.get_results()['history']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        iterations = history['iteration'] if 'iteration' in history else range(len(history['best_fitness']))
        best_key = 'best_fitness' if 'best_fitness' in history else 'best_distance'
        mean_key = 'mean_fitness' if 'mean_fitness' in history else 'mean_distance'
        
        ax.plot(iterations, history[best_key], 
               linewidth=2.5, color='#C73E1D', label='最优值 (Best)')
        ax.plot(iterations, history[mean_key], 
               linewidth=1.5, color='#2E86AB', alpha=0.7, label='平均值 (Mean)')
        ax.fill_between(iterations, history[best_key], 
                       alpha=0.2, color='#C73E1D')
        
        ax.set_xlabel('迭代次数 (Iteration)', fontweight='bold')
        ax.set_ylabel('目标函数值 (Objective)', fontweight='bold')
        ax.set_title(title or '算法收敛曲线 (Convergence Curve)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_comparison(self, results_dict, title="算法对比", save_path=None):
        """
        多算法收敛曲线对比
        
        :param results_dict: {'算法名': optimizer_or_results, ...}
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = list(self.colors.values())
        
        # 收敛曲线对比
        ax1 = axes[0]
        for i, (name, result) in enumerate(results_dict.items()):
            history = result.history if hasattr(result, 'history') else result['history']
            best_key = 'best_fitness' if 'best_fitness' in history else 'best_distance'
            iterations = history['iteration'] if 'iteration' in history else range(len(history[best_key]))
            
            color = self.colors.get(name, colors[i % len(colors)])
            ax1.plot(iterations, history[best_key], 
                    linewidth=2, label=name, color=color)
        
        ax1.set_xlabel('迭代次数', fontweight='bold')
        ax1.set_ylabel('最优值', fontweight='bold')
        ax1.set_title('(a) 收敛曲线对比', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 最终结果柱状图
        ax2 = axes[1]
        names = list(results_dict.keys())
        final_values = []
        for result in results_dict.values():
            if hasattr(result, 'best_fitness'):
                final_values.append(result.best_fitness)
            elif 'best_fitness' in result:
                final_values.append(result['best_fitness'])
            else:
                final_values.append(result.get('best_distance', result['history']['best_distance'][-1]))
        
        bars = ax2.bar(names, final_values, 
                      color=[self.colors.get(n, colors[i % len(colors)]) for i, n in enumerate(names)],
                      edgecolor='white', linewidth=2)
        ax2.set_ylabel('最终最优值', fontweight='bold')
        ax2.set_title('(b) 最终结果对比', fontsize=12, fontweight='bold')
        
        for bar, val in zip(bars, final_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_values)*0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_tsp_solution(self, cities, path, title="TSP最优路径", save_path=None):
        """绘制TSP解"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制城市点
        ax.scatter(cities[:, 0], cities[:, 1], 
                  s=150, c='#2E86AB', edgecolors='white', linewidths=2, zorder=5)
        
        # 标注城市编号
        for i, (x, y) in enumerate(cities):
            ax.annotate(f'{i+1}', (x, y), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
        
        # 绘制路径
        path_cities = cities[path + [path[0]]]
        ax.plot(path_cities[:, 0], path_cities[:, 1], 
               'o-', color='#F18F01', linewidth=2, markersize=0, alpha=0.8)
        
        # 标记起点
        ax.scatter(cities[path[0], 0], cities[path[0], 1],
                  s=300, marker='*', c='#C73E1D', edgecolors='white', 
                  linewidths=2, zorder=10, label='起点')
        
        ax.set_xlabel('X坐标', fontweight='bold')
        ax.set_ylabel('Y坐标', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_function_landscape(self, func, bounds, best_solution=None, 
                                title="函数landscape", save_path=None):
        """绘制2D函数landscape和最优解"""
        x = np.linspace(bounds[0], bounds[1], 100)
        y = np.linspace(bounds[0], bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[func(np.array([xi, yi])) for xi, yi in zip(xrow, yrow)] 
                     for xrow, yrow in zip(X, Y)])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 等高线图
        ax1 = axes[0]
        contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(contour, ax=ax1)
        if best_solution is not None:
            ax1.scatter(best_solution[0], best_solution[1], 
                       s=200, marker='*', c='red', edgecolors='white', 
                       linewidths=2, label='最优解')
            ax1.legend()
        ax1.set_xlabel('x1', fontweight='bold')
        ax1.set_ylabel('x2', fontweight='bold')
        ax1.set_title('(a) 等高线图', fontsize=12, fontweight='bold')
        
        # 3D曲面图
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        if best_solution is not None:
            z_best = func(best_solution)
            ax2.scatter(best_solution[0], best_solution[1], z_best,
                       s=200, marker='*', c='red')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('f(x)')
        ax2.set_title('(b) 3D曲面图', fontsize=12, fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 第十部分：算法对比分析 (Algorithm Comparison)
# ============================================================

class AlgorithmComparator:
    """算法对比分析类"""
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.results = {}
        self.visualizer = OptimizationVisualizer()
    
    def compare_on_function(self, func, bounds, n_dims=2, n_trials=10, max_iter=100):
        """
        在标准测试函数上对比算法
        
        :param func: 测试函数
        :param bounds: 变量范围
        :param n_dims: 维度
        :param n_trials: 运行次数
        :param max_iter: 最大迭代
        """
        algorithms = {
            'PSO': lambda seed: ParticleSwarmOptimization(
                func, bounds, n_dims, max_iter=max_iter, random_seed=seed, verbose=False
            ),
            'GA': lambda seed: GeneticAlgorithm(
                func, bounds, n_dims, max_iter=max_iter, random_seed=seed, verbose=False
            )
        }
        
        results = {name: [] for name in algorithms}
        
        for trial in range(n_trials):
            for name, algo_factory in algorithms.items():
                algo = algo_factory(self.random_seed + trial)
                algo.optimize()
                results[name].append({
                    'best_fitness': algo.best_fitness,
                    'best_solution': algo.best_solution,
                    'history': algo.history
                })
        
        self.results = results
        return self
    
    def statistical_summary(self):
        """生成统计摘要"""
        print("\n" + "="*70)
        print("📊 算法对比统计摘要 (Statistical Summary)")
        print("="*70)
        
        summary = {}
        for name, trials in self.results.items():
            fitness_values = [t['best_fitness'] for t in trials]
            summary[name] = {
                'mean': np.mean(fitness_values),
                'std': np.std(fitness_values),
                'best': np.min(fitness_values),
                'worst': np.max(fitness_values)
            }
            print(f"\n  {name}:")
            print(f"    Mean: {summary[name]['mean']:.6f} ± {summary[name]['std']:.6f}")
            print(f"    Best: {summary[name]['best']:.6f}")
            print(f"    Worst: {summary[name]['worst']:.6f}")
        
        print("="*70)
        return summary
    
    def plot_boxplot(self, save_path=None):
        """绘制箱线图对比"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = []
        labels = []
        colors = []
        
        for name, trials in self.results.items():
            fitness_values = [t['best_fitness'] for t in trials]
            data.append(fitness_values)
            labels.append(name)
            colors.append(PlotStyleConfig.ALGO_COLORS.get(name, '#2E86AB'))
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('最优适应度值', fontweight='bold')
        ax.set_title('算法性能对比箱线图', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 第十一部分：主程序与完整示例 (Main Program)
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("   OPTIMIZATION MODELS FOR MCM/ICM")
    print("   优化类模型 - PSO + GA + ACO")
    print("   Extended Version with Visualization & Comparison")
    print("="*70)
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    📊 优化算法分析流程                            ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║   [PSO] 粒子群优化 ──→ 连续优化问题                              ║
    ║      │                                                           ║
    ║      ├─ 优点：收敛快，参数少，易实现                              ║
    ║      └─ 适用：函数优化、参数调优、神经网络训练                    ║
    ║                                                                  ║
    ║   [SA] 模拟退火 ──→ 全局优化问题                                 ║
    ║      │                                                           ║
    ║      ├─ 优点：可跳出局部最优，参数鲁棒性好                        ║
    ║      └─ 适用：组合优化、路径规划、调度问题                        ║
    ║                                                                  ║
    ║   [GA] 遗传算法 ──→ 连续/离散优化问题                            ║
    ║      │                                                           ║
    ║      ├─ 优点：全局搜索能力强，适应性好                            ║
    ║      └─ 适用：组合优化、调度问题、特征选择                        ║
    ║                                                                  ║
    ║   [ACO] 蚁群算法 ──→ 组合优化问题                                ║
    ║      │                                                           ║
    ║      ├─ 优点：正反馈机制，分布式计算                              ║
    ║      └─ 适用：TSP、VRP、路径规划                                 ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    visualizer = OptimizationVisualizer()
    
    # ================================================================
    # 示例1：PSO求解Rastrigin函数
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 1: PSO求解Rastrigin函数")
    print("="*70)
    
    print("\n目标函数: f(x) = 10n + Σ[x_i² - 10cos(2πx_i)]")
    print("理论最优: f(0,0) = 0\n")
    
    pso = ParticleSwarmOptimization(
        objective_func=BenchmarkFunctions.rastrigin,
        bounds=(-5.12, 5.12),
        n_dims=2,
        pop_size=40,
        max_iter=100,
        w=0.7, c1=2.0, c2=2.0,
        w_decay=True,
        verbose=True
    )
    pso_solution, pso_fitness = pso.optimize()
    
    # 可视化
    visualizer.plot_convergence(pso, title="PSO收敛曲线 - Rastrigin函数")
    visualizer.plot_function_landscape(
        BenchmarkFunctions.rastrigin, (-5.12, 5.12), pso_solution,
        title="Rastrigin函数与PSO最优解"
    )
    
    # ================================================================
    # 示例2：SA求解Ackley函数
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 2: SA求解Ackley函数")
    print("="*70)
    
    print("\n目标函数: Ackley函数（多峰复杂函数）")
    print("理论最优: f(0,0) = 0\n")
    
    sa = SimulatedAnnealing(
        objective_func=BenchmarkFunctions.ackley,
        bounds=(-32.768, 32.768),
        n_dims=2,
        initial_temp=100.0,
        min_temp=1e-8,
        cooling_rate=0.95,
        max_iter=2000,
        max_iter_per_temp=20,
        adaptive_step=True,
        verbose=True
    )
    sa_solution, sa_fitness = sa.optimize()
    
    visualizer.plot_convergence(sa, title="SA收敛曲线 - Ackley函数")
    visualizer.plot_function_landscape(
        BenchmarkFunctions.ackley, (-5, 5), sa_solution,
        title="Ackley函数与SA最优解"
    )
    
    # ================================================================
    # 示例3：GA求解Rosenbrock函数
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 3: GA求解Rosenbrock函数")
    print("="*70)
    
    print("\n目标函数: f(x) = Σ[100(x_{i+1}-x_i²)² + (1-x_i)²]")
    print("理论最优: f(1,1) = 0\n")
    
    ga = GeneticAlgorithm(
        objective_func=BenchmarkFunctions.rosenbrock,
        bounds=(-5, 10),
        n_dims=2,
        pop_size=60,
        max_iter=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        selection_method='tournament',
        crossover_method='uniform',
        elitism=True,
        verbose=True
    )
    ga_solution, ga_fitness = ga.optimize()
    
    visualizer.plot_convergence(ga, title="GA收敛曲线 - Rosenbrock函数")
    
    # ================================================================
    # 示例4：ACO求解TSP问题
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 4: ACO求解TSP问题")
    print("="*70)
    
    # 生成随机城市
    np.random.seed(2026)
    n_cities = 15
    cities = np.random.uniform(0, 100, (n_cities, 2))
    
    print(f"\n城市数量: {n_cities}")
    print("目标: 找到访问所有城市的最短路径\n")
    
    aco = AntColonyOptimization(
        cities=cities,
        n_ants=30,
        max_iter=80,
        alpha=1.0, beta=3.0, rho=0.4, Q=100,
        verbose=True
    )
    aco_path, aco_distance = aco.optimize()
    
    visualizer.plot_tsp_solution(cities, aco_path, 
                                 title=f"ACO-TSP最优路径 (距离: {aco_distance:.2f})")
    visualizer.plot_convergence(aco, title="ACO收敛曲线 - TSP问题")
    
    # ================================================================
    # 示例5：SA-TSP求解TSP问题（与ACO对比）
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 5: SA-TSP求解TSP问题")
    print("="*70)
    
    sa_tsp = SimulatedAnnealingTSP(
        cities=cities,
        initial_temp=1000.0,
        min_temp=1e-6,
        cooling_rate=0.995,
        max_iter_per_temp=50,
        verbose=True
    )
    sa_tsp_path, sa_tsp_distance = sa_tsp.optimize()
    
    visualizer.plot_tsp_solution(cities, sa_tsp_path,
                                 title=f"SA-TSP最优路径 (距离: {sa_tsp_distance:.2f})")
    
    print(f"\n📊 TSP算法对比: ACO={aco_distance:.2f} vs SA={sa_tsp_distance:.2f}")
    
    # ================================================================
    # 示例6：四种算法综合对比
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 6: PSO vs SA vs GA 算法对比")
    print("="*70)
    
    # 可视化对比（连续优化算法）
    visualizer.plot_comparison({'PSO': pso, 'SA': sa, 'GA': ga}, 
                               title="PSO vs SA vs GA 收敛曲线对比")
    
    # ================================================================
    # 使用说明
    # ================================================================
    print("\n" + "="*70)
    print("📖 使用说明 (Usage Guide)")
    print("="*70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                     优化算法使用指南                             │
    └─────────────────────────────────────────────────────────────────┘
    
    【算法选择建议】
    
    1️⃣ PSO（粒子群优化）
       ├─ 适用: 连续变量优化、参数调优
       ├─ 优点: 收敛快、参数少、易实现
       └─ 参数: w=0.7, c1=c2=2.0（默认即可）
    
    2️⃣ SA（模拟退火）
       ├─ 适用: 全局优化、组合优化、跳出局部最优
       ├─ 优点: 理论保证收敛、参数鲁棒性好
       └─ 参数: T0=100, α=0.95, 自适应步长
    
    3️⃣ GA（遗传算法）
       ├─ 适用: 离散/连续优化、组合优化
       ├─ 优点: 全局搜索能力强、鲁棒性好
       └─ 参数: Pc=0.8, Pm=0.1, 锦标赛选择
    
    4️⃣ ACO（蚁群算法）
       ├─ 适用: TSP、VRP等路径问题
       ├─ 优点: 正反馈、分布式、并行性好
       └─ 参数: α=1, β=2-5, ρ=0.1-0.5
    
    5️⃣ SA-TSP（模拟退火TSP版）
       ├─ 适用: 旅行商问题、路径优化
       ├─ 优点: 2-opt邻域、多种扰动策略
       └─ 参数: T0=1000, α=0.995
    
    【自定义目标函数】
    
    def my_objective(x):
        # x是numpy数组
        return x[0]**2 + x[1]**2  # 返回标量
    
    # PSO示例
    optimizer = ParticleSwarmOptimization(
        objective_func=my_objective,
        bounds=(-10, 10),
        n_dims=2
    )
    
    # SA示例
    optimizer = SimulatedAnnealing(
        objective_func=my_objective,
        bounds=(-10, 10),
        n_dims=2,
        initial_temp=100.0,
        cooling_rate=0.95
    )
    
    【论文图表建议】
    
    Figure 1: 问题描述（函数landscape/城市分布）
    Figure 2: 收敛曲线
    Figure 3: 最优解可视化
    Figure 4: 算法对比（箱线图/收敛曲线）
    Figure 5: 参数敏感性分析
    
    Table 1: 算法参数设置
    Table 2: 多次运行统计结果（Mean±Std）
    Table 3: 与其他方法对比（PSO/SA/GA/ACO）
    """)
    
    print("\n" + "="*70)
    print("   ✅ All examples completed successfully!")
    print("   💡 Use the above code templates for your MCM/ICM paper")
    print("="*70)
