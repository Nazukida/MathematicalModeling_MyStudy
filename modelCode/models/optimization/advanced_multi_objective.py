"""
============================================================
高级多目标规划模型 (Advanced Multi-Objective Programming)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：多目标优化、帕累托分析、权重法、ε-约束法、完整可视化
特点：完备的数据预处理 + 模型求解 + 结果可视化三位一体

使用场景：
- 投资组合优化（收益vs风险）
- 供应链设计（成本vs服务）
- 工程设计（性能vs成本vs重量）
- 资源分配（效率vs公平）
- 环境经济分析（经济发展vs环境保护）

作者：MCM/ICM Team
日期：2026年1月
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, differential_evolution
from typing import Callable, List, Dict, Tuple, Optional, Union
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')


# ============================================================
# 第一部分：图表配置
# ============================================================

class MOPPlotConfig:
    """多目标规划可视化配置"""
    
    COLORS = {
        'pareto': '#E94F37',        # 帕累托前沿颜色
        'dominated': '#CCCCCC',     # 被支配解颜色
        'selected': '#27AE60',      # 选中解颜色
        'utopia': '#2E86AB',        # 理想点颜色
        'nadir': '#F18F01',         # 最差点颜色
        'tradeoff': '#6B4C9A',      # 权衡曲线颜色
        'grid': '#E0E0E0'
    }
    
    PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4C9A', '#1B998B']
    
    @staticmethod
    def setup():
        plt.style.use('seaborn-v0_8-whitegrid')
        rcParams['figure.figsize'] = (12, 8)
        rcParams['figure.dpi'] = 100
        rcParams['savefig.dpi'] = 300
        rcParams['font.size'] = 11
        rcParams['axes.titlesize'] = 14
        rcParams['axes.labelsize'] = 12
        rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False

MOPPlotConfig.setup()


# ============================================================
# 第二部分：数据预处理模块
# ============================================================

class MOPDataPreprocessor:
    """
    多目标规划数据预处理器
    
    功能：
    1. 目标函数标准化（统一为最小化）
    2. 数据归一化
    3. 理想点/最差点计算
    4. 权重生成
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.processing_log = []
    
    def _log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        if self.verbose:
            print(log_entry)
    
    def standardize_objectives(self, 
                               objectives: List[Callable],
                               senses: List[str]) -> List[Callable]:
        """
        将所有目标标准化为最小化问题
        
        :param objectives: 目标函数列表
        :param senses: 优化方向列表 ['min', 'max', ...]
        :return: 标准化后的目标函数列表
        """
        self._log("标准化目标函数（统一为最小化）...")
        
        standardized = []
        for i, (obj, sense) in enumerate(zip(objectives, senses)):
            if sense.lower() == 'max':
                standardized.append(lambda x, f=obj: -f(x))
                self._log(f"  目标{i+1}: max → -min（取负）")
            else:
                standardized.append(obj)
                self._log(f"  目标{i+1}: min（保持不变）")
        
        return standardized
    
    def normalize_objectives(self,
                             pareto_front: np.ndarray,
                             method: str = 'minmax') -> Tuple[np.ndarray, Dict]:
        """
        归一化帕累托前沿
        
        :param pareto_front: 帕累托前沿 (n_solutions, n_objectives)
        :param method: 'minmax' 或 'ideal-nadir'
        :return: (归一化前沿, 参数字典)
        """
        self._log("归一化目标值...")
        
        if method == 'minmax':
            min_vals = np.min(pareto_front, axis=0)
            max_vals = np.max(pareto_front, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1
            normalized = (pareto_front - min_vals) / range_vals
            params = {'min': min_vals, 'max': max_vals}
        else:  # ideal-nadir
            ideal = np.min(pareto_front, axis=0)
            nadir = np.max(pareto_front, axis=0)
            range_vals = nadir - ideal
            range_vals[range_vals == 0] = 1
            normalized = (pareto_front - ideal) / range_vals
            params = {'ideal': ideal, 'nadir': nadir}
        
        return normalized, params
    
    def compute_ideal_nadir(self,
                            objectives: List[Callable],
                            bounds: List[Tuple],
                            constraints: Optional[List[Dict]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算理想点和最差点
        
        :return: (ideal_point, nadir_point)
        """
        self._log("计算理想点和最差点...")
        
        n_obj = len(objectives)
        ideal = np.zeros(n_obj)
        nadir = np.zeros(n_obj)
        
        for i, obj in enumerate(objectives):
            # 最小化得到理想点
            result = differential_evolution(obj, bounds, seed=42, maxiter=100)
            ideal[i] = result.fun
            
            # 最大化得到最差点
            neg_obj = lambda x: -obj(x)
            result = differential_evolution(neg_obj, bounds, seed=42, maxiter=100)
            nadir[i] = -result.fun
        
        self._log(f"  理想点: {ideal}")
        self._log(f"  最差点: {nadir}")
        
        return ideal, nadir
    
    def generate_weights(self,
                         n_objectives: int,
                         n_weights: int = 20,
                         method: str = 'uniform') -> np.ndarray:
        """
        生成权重向量集合
        
        :param n_objectives: 目标数量
        :param n_weights: 权重组数
        :param method: 'uniform', 'random', 'das-dennis'
        :return: 权重矩阵 (n_weights, n_objectives)
        """
        self._log(f"生成权重向量 (方法: {method})...")
        
        if method == 'uniform':
            if n_objectives == 2:
                w1 = np.linspace(0, 1, n_weights)
                weights = np.column_stack([w1, 1 - w1])
            else:
                # 简单的均匀采样
                weights = np.random.dirichlet(np.ones(n_objectives), n_weights)
        elif method == 'random':
            weights = np.random.dirichlet(np.ones(n_objectives), n_weights)
        elif method == 'das-dennis':
            # Das-Dennis方法生成参考点
            weights = self._das_dennis(n_objectives, n_weights)
        
        self._log(f"  生成 {len(weights)} 组权重")
        return weights
    
    def _das_dennis(self, n_obj: int, n_points: int) -> np.ndarray:
        """Das-Dennis参考点生成"""
        # 计算分层数
        H = 1
        while self._comb(H + n_obj - 1, n_obj - 1) < n_points:
            H += 1
        
        # 生成参考点
        points = []
        
        def generate(left, depth, current):
            if depth == n_obj - 1:
                current.append(left / H)
                points.append(current[:])
                current.pop()
                return
            for i in range(left + 1):
                current.append(i / H)
                generate(left - i, depth + 1, current)
                current.pop()
        
        generate(H, 0, [])
        return np.array(points[:n_points])
    
    def _comb(self, n, k):
        """组合数"""
        from math import factorial
        return factorial(n) // (factorial(k) * factorial(n - k))


# ============================================================
# 第三部分：多目标规划求解器
# ============================================================

class MultiObjectiveSolver:
    """
    多目标规划求解器
    
    支持方法：
    1. 加权法 (Weighted Sum)
    2. ε-约束法 (ε-Constraint)
    3. 目标规划法 (Goal Programming)
    4. NSGA-II (已在nsga2_multi_objective.py中实现)
    5. 字典序法 (Lexicographic)
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.result = None
        self.pareto_front = None
        self.pareto_solutions = None
    
    def weighted_sum(self,
                     objectives: List[Callable],
                     weights: np.ndarray,
                     bounds: List[Tuple],
                     constraints: Optional[List[Dict]] = None,
                     x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        加权法求解多目标优化
        
        :param objectives: 目标函数列表（均为最小化）
        :param weights: 权重矩阵 (n_weights, n_objectives)
        :param bounds: 变量边界
        :param constraints: 约束条件
        :return: (pareto_solutions, pareto_front)
        """
        if self.verbose:
            print("\n" + "="*60)
            print("   加权法多目标优化")
            print("="*60)
            print(f"  目标数: {len(objectives)}")
            print(f"  权重组数: {len(weights)}")
        
        n_weights = len(weights)
        n_obj = len(objectives)
        n_var = len(bounds)
        
        solutions = []
        front = []
        
        for i, w in enumerate(weights):
            # 加权目标函数
            def weighted_obj(x, w=w):
                return sum(w[j] * objectives[j](x) for j in range(n_obj))
            
            # 初始点
            if x0 is None:
                x0_current = np.array([np.mean(b) for b in bounds])
            else:
                x0_current = x0.copy()
            
            # 求解
            result = minimize(weighted_obj, x0_current, method='SLSQP',
                            bounds=bounds, constraints=constraints or [])
            
            if result.success:
                solutions.append(result.x)
                obj_values = [objectives[j](result.x) for j in range(n_obj)]
                front.append(obj_values)
                
                if self.verbose and (i + 1) % 5 == 0:
                    print(f"  完成 {i+1}/{n_weights} 组权重")
        
        # 筛选非支配解
        solutions = np.array(solutions)
        front = np.array(front)
        
        pareto_mask = self._non_dominated_filter(front)
        
        self.pareto_solutions = solutions[pareto_mask]
        self.pareto_front = front[pareto_mask]
        
        if self.verbose:
            print(f"\n  找到 {len(self.pareto_front)} 个帕累托最优解")
        
        return self.pareto_solutions, self.pareto_front
    
    def epsilon_constraint(self,
                           objectives: List[Callable],
                           primary_idx: int,
                           epsilon_ranges: List[Tuple],
                           n_points: int,
                           bounds: List[Tuple],
                           base_constraints: Optional[List[Dict]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        ε-约束法求解多目标优化
        
        :param objectives: 目标函数列表
        :param primary_idx: 主目标索引（保持为目标函数）
        :param epsilon_ranges: 其他目标的ε范围 [(min, max), ...]
        :param n_points: 每个ε范围的采样点数
        :param bounds: 变量边界
        :param base_constraints: 基础约束
        :return: (pareto_solutions, pareto_front)
        """
        if self.verbose:
            print("\n" + "="*60)
            print("   ε-约束法多目标优化")
            print("="*60)
            print(f"  主目标: 目标{primary_idx + 1}")
        
        n_obj = len(objectives)
        solutions = []
        front = []
        
        # 生成ε值网格
        eps_grids = []
        for j, eps_range in enumerate(epsilon_ranges):
            if j == primary_idx:
                eps_grids.append([None])
            else:
                eps_grids.append(np.linspace(eps_range[0], eps_range[1], n_points))
        
        # 遍历所有ε组合
        from itertools import product
        
        eps_combinations = list(product(*[eps_grids[j] for j in range(n_obj) if j != primary_idx]))
        
        for eps_vals in eps_combinations:
            # 构建约束
            constraints = list(base_constraints) if base_constraints else []
            
            eps_idx = 0
            for j in range(n_obj):
                if j != primary_idx:
                    eps_val = eps_vals[eps_idx]
                    eps_idx += 1
                    # 添加ε约束: f_j(x) <= eps
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, j=j, e=eps_val: e - objectives[j](x)
                    })
            
            # 优化主目标
            x0 = np.array([np.mean(b) for b in bounds])
            
            try:
                result = minimize(objectives[primary_idx], x0, method='SLSQP',
                                bounds=bounds, constraints=constraints)
                
                if result.success:
                    solutions.append(result.x)
                    obj_values = [objectives[j](result.x) for j in range(n_obj)]
                    front.append(obj_values)
            except:
                pass
        
        # 筛选非支配解
        if len(front) > 0:
            solutions = np.array(solutions)
            front = np.array(front)
            pareto_mask = self._non_dominated_filter(front)
            
            self.pareto_solutions = solutions[pareto_mask]
            self.pareto_front = front[pareto_mask]
        else:
            self.pareto_solutions = np.array([])
            self.pareto_front = np.array([])
        
        if self.verbose:
            print(f"\n  找到 {len(self.pareto_front)} 个帕累托最优解")
        
        return self.pareto_solutions, self.pareto_front
    
    def goal_programming(self,
                         objectives: List[Callable],
                         goals: List[float],
                         priorities: Optional[List[int]] = None,
                         bounds: List[Tuple] = None,
                         constraints: Optional[List[Dict]] = None) -> Dict:
        """
        目标规划法
        
        :param objectives: 目标函数列表
        :param goals: 各目标的期望值
        :param priorities: 优先级（1最高）
        :param bounds: 变量边界
        :param constraints: 约束条件
        :return: 求解结果
        """
        if self.verbose:
            print("\n" + "="*60)
            print("   目标规划法")
            print("="*60)
        
        n_obj = len(objectives)
        
        if priorities is None:
            priorities = list(range(1, n_obj + 1))
        
        # 按优先级分组
        priority_levels = sorted(set(priorities))
        
        x0 = np.array([np.mean(b) for b in bounds])
        current_x = x0.copy()
        current_constraints = list(constraints) if constraints else []
        
        for level in priority_levels:
            # 当前优先级的目标
            level_objectives = [(i, objectives[i], goals[i]) 
                               for i, p in enumerate(priorities) if p == level]
            
            if self.verbose:
                print(f"\n  处理优先级 {level} (目标: {[i+1 for i, _, _ in level_objectives]})")
            
            # 最小化偏差
            def deviation_obj(x):
                total = 0
                for i, obj, goal in level_objectives:
                    dev = obj(x) - goal
                    total += dev ** 2
                return total
            
            result = minimize(deviation_obj, current_x, method='SLSQP',
                            bounds=bounds, constraints=current_constraints)
            
            if result.success:
                current_x = result.x
                
                # 将本级目标固定为约束，继续下一级
                for i, obj, goal in level_objectives:
                    achieved = obj(current_x)
                    current_constraints.append({
                        'type': 'eq',
                        'fun': lambda x, f=obj, v=achieved: f(x) - v
                    })
        
        # 计算最终目标值
        final_objectives = [objectives[i](current_x) for i in range(n_obj)]
        deviations = [final_objectives[i] - goals[i] for i in range(n_obj)]
        
        self.result = {
            'success': True,
            'x': current_x,
            'objectives': final_objectives,
            'goals': goals,
            'deviations': deviations,
            'priorities': priorities
        }
        
        if self.verbose:
            print("\n  目标规划结果:")
            for i in range(n_obj):
                status = "✅" if abs(deviations[i]) < 0.01 * abs(goals[i]) else "⚠️"
                print(f"    目标{i+1}: 期望={goals[i]:.4f}, 实际={final_objectives[i]:.4f}, 偏差={deviations[i]:.4f} {status}")
        
        return self.result
    
    def lexicographic(self,
                      objectives: List[Callable],
                      priority_order: List[int],
                      bounds: List[Tuple],
                      tolerances: Optional[List[float]] = None,
                      constraints: Optional[List[Dict]] = None) -> Dict:
        """
        字典序法（优先级法）
        
        :param objectives: 目标函数列表
        :param priority_order: 优化顺序（索引列表）
        :param bounds: 变量边界
        :param tolerances: 允许的目标值恶化容忍度
        :return: 求解结果
        """
        if self.verbose:
            print("\n" + "="*60)
            print("   字典序法多目标优化")
            print("="*60)
            print(f"  优化顺序: {[f'目标{i+1}' for i in priority_order]}")
        
        n_obj = len(objectives)
        
        if tolerances is None:
            tolerances = [0.01] * n_obj  # 默认1%容忍度
        
        x0 = np.array([np.mean(b) for b in bounds])
        current_x = x0.copy()
        current_constraints = list(constraints) if constraints else []
        achieved_values = {}
        
        for step, obj_idx in enumerate(priority_order):
            if self.verbose:
                print(f"\n  步骤 {step+1}: 优化目标{obj_idx+1}")
            
            result = minimize(objectives[obj_idx], current_x, method='SLSQP',
                            bounds=bounds, constraints=current_constraints)
            
            if result.success:
                current_x = result.x
                opt_value = objectives[obj_idx](current_x)
                achieved_values[obj_idx] = opt_value
                
                # 添加容忍约束
                tolerance = tolerances[obj_idx]
                upper_bound = opt_value * (1 + tolerance) if opt_value >= 0 else opt_value * (1 - tolerance)
                
                current_constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, f=objectives[obj_idx], ub=upper_bound: ub - f(x)
                })
                
                if self.verbose:
                    print(f"    最优值: {opt_value:.4f} (容忍上界: {upper_bound:.4f})")
        
        final_objectives = [objectives[i](current_x) for i in range(n_obj)]
        
        self.result = {
            'success': True,
            'x': current_x,
            'objectives': final_objectives,
            'priority_order': priority_order,
            'achieved_at_step': achieved_values
        }
        
        return self.result
    
    def _non_dominated_filter(self, front: np.ndarray) -> np.ndarray:
        """筛选非支配解"""
        n = len(front)
        is_pareto = np.ones(n, dtype=bool)
        
        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j or not is_pareto[j]:
                    continue
                # 检查j是否支配i
                if np.all(front[j] <= front[i]) and np.any(front[j] < front[i]):
                    is_pareto[i] = False
                    break
        
        return is_pareto


# ============================================================
# 第四部分：决策分析模块
# ============================================================

class ParetoAnalyzer:
    """
    帕累托分析器
    
    功能：
    1. 计算帕累托前沿指标
    2. 选择最佳折中解
    3. 权衡分析
    4. 边际分析
    """
    
    def __init__(self, pareto_front: np.ndarray, pareto_solutions: np.ndarray):
        self.front = pareto_front
        self.solutions = pareto_solutions
        self.n_solutions = len(pareto_front)
        self.n_objectives = pareto_front.shape[1] if len(pareto_front) > 0 else 0
    
    def compute_metrics(self) -> Dict:
        """计算帕累托前沿指标"""
        if self.n_solutions == 0:
            return {}
        
        metrics = {
            'n_solutions': self.n_solutions,
            'ideal_point': np.min(self.front, axis=0),
            'nadir_point': np.max(self.front, axis=0),
            'spread': np.max(self.front, axis=0) - np.min(self.front, axis=0),
            'hypervolume': self._compute_hypervolume()
        }
        
        return metrics
    
    def _compute_hypervolume(self, ref_point: Optional[np.ndarray] = None) -> float:
        """计算超体积指标（2D简化版本）"""
        if self.n_objectives != 2:
            return np.nan
        
        if ref_point is None:
            ref_point = np.max(self.front, axis=0) * 1.1
        
        # 按第一个目标排序
        sorted_idx = np.argsort(self.front[:, 0])
        sorted_front = self.front[sorted_idx]
        
        hv = 0
        prev_x = sorted_front[0, 0]
        prev_y = ref_point[1]
        
        for point in sorted_front:
            hv += (point[0] - prev_x) * prev_y
            prev_x = point[0]
            prev_y = point[1]
        
        hv += (ref_point[0] - prev_x) * prev_y
        
        return hv
    
    def find_knee_point(self) -> Tuple[int, np.ndarray]:
        """
        找到膝点（最佳折中解）
        
        使用归一化距离法
        """
        if self.n_solutions == 0:
            return -1, None
        
        # 归一化
        min_vals = np.min(self.front, axis=0)
        max_vals = np.max(self.front, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        normalized = (self.front - min_vals) / range_vals
        
        # 到理想点的欧氏距离
        distances = np.sqrt(np.sum(normalized ** 2, axis=1))
        
        knee_idx = np.argmin(distances)
        
        return knee_idx, self.solutions[knee_idx]
    
    def find_by_weights(self, weights: np.ndarray) -> Tuple[int, np.ndarray]:
        """根据权重偏好选择解"""
        if self.n_solutions == 0:
            return -1, None
        
        # 归一化
        min_vals = np.min(self.front, axis=0)
        max_vals = np.max(self.front, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        normalized = (self.front - min_vals) / range_vals
        
        # 加权和
        weighted_sum = np.dot(normalized, weights)
        best_idx = np.argmin(weighted_sum)
        
        return best_idx, self.solutions[best_idx]
    
    def tradeoff_analysis(self, obj_i: int, obj_j: int) -> pd.DataFrame:
        """
        两目标间的权衡分析
        
        计算边际替代率
        """
        if self.n_solutions < 2:
            return pd.DataFrame()
        
        # 按目标i排序
        sorted_idx = np.argsort(self.front[:, obj_i])
        sorted_front = self.front[sorted_idx]
        sorted_solutions = self.solutions[sorted_idx]
        
        # 计算边际替代率
        mrs = []
        for k in range(len(sorted_front) - 1):
            delta_i = sorted_front[k+1, obj_i] - sorted_front[k, obj_i]
            delta_j = sorted_front[k+1, obj_j] - sorted_front[k, obj_j]
            if abs(delta_i) > 1e-10:
                mrs.append(-delta_j / delta_i)
            else:
                mrs.append(np.nan)
        mrs.append(np.nan)
        
        df = pd.DataFrame({
            f'目标{obj_i+1}': sorted_front[:, obj_i],
            f'目标{obj_j+1}': sorted_front[:, obj_j],
            '边际替代率': mrs
        })
        
        return df


# ============================================================
# 第五部分：可视化模块
# ============================================================

class MOPVisualizer:
    """
    多目标规划可视化器
    
    功能：
    1. 帕累托前沿图
    2. 平行坐标图
    3. 雷达图
    4. 权衡曲线图
    """
    
    def __init__(self, save_dir: str = './figures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_pareto_front_2d(self,
                              front: np.ndarray,
                              obj_names: Optional[List[str]] = None,
                              highlight_idx: Optional[int] = None,
                              ideal_point: Optional[np.ndarray] = None,
                              title: str = '帕累托前沿',
                              save_name: Optional[str] = None):
        """绘制2D帕累托前沿"""
        if front.shape[1] != 2:
            print("此函数仅支持2目标问题")
            return
        
        if obj_names is None:
            obj_names = ['目标1', '目标2']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 散点图
        ax.scatter(front[:, 0], front[:, 1],
                  c=MOPPlotConfig.COLORS['pareto'], s=80,
                  edgecolor='white', linewidth=2, alpha=0.8,
                  label='帕累托最优解')
        
        # 连接线（排序后）
        sorted_idx = np.argsort(front[:, 0])
        ax.plot(front[sorted_idx, 0], front[sorted_idx, 1],
               'k--', alpha=0.3, linewidth=1)
        
        # 高亮特定解
        if highlight_idx is not None:
            ax.scatter(front[highlight_idx, 0], front[highlight_idx, 1],
                      c=MOPPlotConfig.COLORS['selected'], s=200, marker='*',
                      edgecolor='black', linewidth=2, zorder=5,
                      label='选中解')
        
        # 理想点
        if ideal_point is not None:
            ax.scatter(ideal_point[0], ideal_point[1],
                      c=MOPPlotConfig.COLORS['utopia'], s=150, marker='D',
                      edgecolor='black', linewidth=2, zorder=5,
                      label='理想点')
        
        ax.set_xlabel(obj_names[0], fontsize=12, fontweight='bold')
        ax.set_ylabel(obj_names[1], fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pareto_front_3d(self,
                              front: np.ndarray,
                              obj_names: Optional[List[str]] = None,
                              title: str = '3D帕累托前沿',
                              save_name: Optional[str] = None):
        """绘制3D帕累托前沿"""
        if front.shape[1] != 3:
            print("此函数仅支持3目标问题")
            return
        
        if obj_names is None:
            obj_names = ['目标1', '目标2', '目标3']
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(front[:, 0], front[:, 1], front[:, 2],
                  c=MOPPlotConfig.COLORS['pareto'], s=80,
                  edgecolor='white', linewidth=1, alpha=0.8)
        
        ax.set_xlabel(obj_names[0], fontsize=10, fontweight='bold')
        ax.set_ylabel(obj_names[1], fontsize=10, fontweight='bold')
        ax.set_zlabel(obj_names[2], fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_parallel_coordinates(self,
                                   front: np.ndarray,
                                   obj_names: Optional[List[str]] = None,
                                   highlight_idx: Optional[int] = None,
                                   title: str = '平行坐标图',
                                   save_name: Optional[str] = None):
        """绘制平行坐标图"""
        n_solutions, n_obj = front.shape
        
        if obj_names is None:
            obj_names = [f'目标{i+1}' for i in range(n_obj)]
        
        # 归一化
        min_vals = np.min(front, axis=0)
        max_vals = np.max(front, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        normalized = (front - min_vals) / range_vals
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制所有解
        for i in range(n_solutions):
            if highlight_idx is not None and i == highlight_idx:
                continue
            ax.plot(range(n_obj), normalized[i], 'o-',
                   color=MOPPlotConfig.COLORS['pareto'], alpha=0.3, linewidth=1)
        
        # 高亮特定解
        if highlight_idx is not None:
            ax.plot(range(n_obj), normalized[highlight_idx], 'o-',
                   color=MOPPlotConfig.COLORS['selected'], linewidth=3,
                   markersize=10, label='选中解')
            ax.legend()
        
        ax.set_xticks(range(n_obj))
        ax.set_xticklabels(obj_names)
        ax.set_ylabel('归一化目标值', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_radar(self,
                   solution_objectives: np.ndarray,
                   obj_names: Optional[List[str]] = None,
                   title: str = '解的雷达图',
                   save_name: Optional[str] = None):
        """绘制单个解的雷达图"""
        n_obj = len(solution_objectives)
        
        if obj_names is None:
            obj_names = [f'目标{i+1}' for i in range(n_obj)]
        
        # 角度
        angles = np.linspace(0, 2 * np.pi, n_obj, endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        values = solution_objectives.tolist()
        values += values[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        ax.plot(angles, values, 'o-', linewidth=2,
               color=MOPPlotConfig.COLORS['selected'])
        ax.fill(angles, values, alpha=0.25,
               color=MOPPlotConfig.COLORS['selected'])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(obj_names)
        ax.set_title(title, fontsize=14, fontweight='bold', y=1.08)
        
        plt.tight_layout()
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_solution_comparison(self,
                                  front: np.ndarray,
                                  solution_indices: List[int],
                                  obj_names: Optional[List[str]] = None,
                                  solution_labels: Optional[List[str]] = None,
                                  title: str = '方案比较',
                                  save_name: Optional[str] = None):
        """绘制多个方案的对比图"""
        n_obj = front.shape[1]
        n_compare = len(solution_indices)
        
        if obj_names is None:
            obj_names = [f'目标{i+1}' for i in range(n_obj)]
        if solution_labels is None:
            solution_labels = [f'方案{i+1}' for i in range(n_compare)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(n_obj)
        width = 0.8 / n_compare
        
        colors = MOPPlotConfig.PALETTE[:n_compare]
        
        for i, (idx, label, color) in enumerate(zip(solution_indices, solution_labels, colors)):
            offset = (i - n_compare/2 + 0.5) * width
            ax.bar(x + offset, front[idx], width, label=label, color=color, edgecolor='white')
        
        ax.set_xticks(x)
        ax.set_xticklabels(obj_names)
        ax.set_ylabel('目标值', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 第六部分：完整工作流
# ============================================================

class MultiObjectivePipeline:
    """
    多目标规划完整工作流
    
    集成数据预处理、模型求解、帕累托分析、结果可视化
    """
    
    def __init__(self, verbose: bool = True, save_dir: str = './figures'):
        self.preprocessor = MOPDataPreprocessor(verbose)
        self.solver = MultiObjectiveSolver(verbose)
        self.analyzer = None
        self.visualizer = MOPVisualizer(save_dir)
        self.verbose = verbose
    
    def run(self,
            objectives: List[Callable],
            senses: List[str],
            bounds: List[Tuple],
            constraints: Optional[List[Dict]] = None,
            method: str = 'weighted_sum',
            n_weights: int = 50,
            obj_names: Optional[List[str]] = None,
            plot_pareto: bool = True,
            plot_parallel: bool = True,
            find_knee: bool = True) -> Dict:
        """
        执行完整的多目标优化流程
        
        :param objectives: 目标函数列表
        :param senses: 优化方向 ['min', 'max', ...]
        :param bounds: 变量边界
        :param constraints: 约束条件
        :param method: 'weighted_sum', 'epsilon_constraint', 'nsga2'
        :param n_weights: 权重/采样数量
        :param obj_names: 目标名称
        :return: 结果字典
        """
        if self.verbose:
            print("\n" + "="*60)
            print("   多目标规划完整工作流")
            print("="*60)
            print(f"  目标数: {len(objectives)}")
            print(f"  方法: {method}")
        
        n_obj = len(objectives)
        
        if obj_names is None:
            obj_names = [f'目标{i+1}' for i in range(n_obj)]
        
        # 标准化目标函数
        std_objectives = self.preprocessor.standardize_objectives(objectives, senses)
        
        # 生成权重
        if method == 'weighted_sum':
            weights = self.preprocessor.generate_weights(n_obj, n_weights, 'uniform')
            solutions, front = self.solver.weighted_sum(std_objectives, weights, bounds, constraints)
        elif method == 'epsilon_constraint':
            # 先计算各目标的范围
            eps_ranges = []
            for obj in std_objectives:
                result = differential_evolution(obj, bounds, seed=42, maxiter=100)
                min_val = result.fun
                result = differential_evolution(lambda x: -obj(x), bounds, seed=42, maxiter=100)
                max_val = -result.fun
                eps_ranges.append((min_val, max_val))
            
            solutions, front = self.solver.epsilon_constraint(
                std_objectives, 0, eps_ranges, n_weights, bounds, constraints)
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        # 还原最大化目标的符号
        for i, sense in enumerate(senses):
            if sense.lower() == 'max':
                front[:, i] = -front[:, i]
        
        # 帕累托分析
        self.analyzer = ParetoAnalyzer(front, solutions)
        metrics = self.analyzer.compute_metrics()
        
        knee_idx, knee_solution = None, None
        if find_knee:
            knee_idx, knee_solution = self.analyzer.find_knee_point()
        
        # 可视化
        if plot_pareto and n_obj == 2:
            self.visualizer.plot_pareto_front_2d(
                front, obj_names, knee_idx,
                metrics.get('ideal_point'),
                title='帕累托前沿与最佳折中解'
            )
        elif plot_pareto and n_obj == 3:
            self.visualizer.plot_pareto_front_3d(front, obj_names)
        
        if plot_parallel:
            self.visualizer.plot_parallel_coordinates(front, obj_names, knee_idx)
        
        result = {
            'pareto_solutions': solutions,
            'pareto_front': front,
            'n_solutions': len(solutions),
            'metrics': metrics,
            'knee_index': knee_idx,
            'knee_solution': knee_solution,
            'knee_objectives': front[knee_idx] if knee_idx is not None else None,
            'obj_names': obj_names
        }
        
        if self.verbose:
            self._print_summary(result)
        
        return result
    
    def _print_summary(self, result):
        """打印结果摘要"""
        print("\n" + "="*60)
        print("📊 多目标优化结果摘要")
        print("="*60)
        print(f"  帕累托最优解数量: {result['n_solutions']}")
        
        if result['metrics']:
            m = result['metrics']
            print(f"\n  理想点: {m['ideal_point']}")
            print(f"  最差点: {m['nadir_point']}")
            print(f"  目标范围: {m['spread']}")
        
        if result['knee_index'] is not None:
            print(f"\n  最佳折中解 (膝点):")
            print(f"    索引: {result['knee_index']}")
            print(f"    目标值: {result['knee_objectives']}")
        
        print("="*60)


# ============================================================
# 示例：投资组合优化（收益vs风险）
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("   示例：投资组合多目标优化（最大化收益 & 最小化风险）")
    print("="*70)
    
    # 4种资产的预期收益率和协方差矩阵
    expected_returns = np.array([0.12, 0.08, 0.05, 0.06])  # 科技股、消费股、债券、黄金
    cov_matrix = np.array([
        [0.04, 0.01, -0.005, 0.002],
        [0.01, 0.02, 0.003, 0.001],
        [-0.005, 0.003, 0.01, -0.002],
        [0.002, 0.001, -0.002, 0.015]
    ])
    
    # 目标1: 预期收益（最大化）
    def portfolio_return(x):
        return np.dot(expected_returns, x)
    
    # 目标2: 风险（最小化）
    def portfolio_risk(x):
        return np.sqrt(x @ cov_matrix @ x)
    
    # 约束：投资比例和为1
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ]
    
    # 边界：每种资产0-100%
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
    
    # 创建工作流
    pipeline = MultiObjectivePipeline(verbose=True)
    
    # 求解
    result = pipeline.run(
        objectives=[portfolio_return, portfolio_risk],
        senses=['max', 'min'],  # 收益最大化，风险最小化
        bounds=bounds,
        constraints=constraints,
        method='weighted_sum',
        n_weights=50,
        obj_names=['预期收益', '风险(标准差)']
    )
    
    # 展示最佳折中方案
    print("\n" + "="*50)
    print("📈 最佳折中投资方案")
    print("="*50)
    if result['knee_solution'] is not None:
        x = result['knee_solution']
        assets = ['科技股', '消费股', '债券', '黄金']
        print("资产配置比例:")
        for i, asset in enumerate(assets):
            print(f"  {asset}: {x[i]*100:.1f}%")
        print(f"\n预期年收益率: {result['knee_objectives'][0]*100:.2f}%")
        print(f"风险(标准差): {result['knee_objectives'][1]*100:.2f}%")
    print("="*50)
    
    # 权衡分析
    print("\n" + "="*50)
    print("📉 收益-风险权衡分析")
    print("="*50)
    tradeoff = pipeline.analyzer.tradeoff_analysis(0, 1)
    print(tradeoff.head(10).to_string(index=False))
    print("="*50)
