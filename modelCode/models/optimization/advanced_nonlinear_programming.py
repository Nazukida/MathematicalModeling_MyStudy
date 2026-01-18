"""
============================================================
高级非线性规划模型 (Advanced Nonlinear Programming)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：非线性优化、约束处理、灵敏度分析、完整可视化
特点：完备的数据预处理 + 模型求解 + 结果可视化三位一体

使用场景：
- 投资组合优化（二次规划）
- 生产计划优化
- 曲线拟合与回归
- 工程设计优化

作者：MCM/ICM Team
日期：2026年1月
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize, differential_evolution
from scipy.stats import zscore
from typing import Callable, List, Dict, Tuple, Optional, Union
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# ============================================================
# 第一部分：图表配置
# ============================================================

class NLPPlotConfig:
    """非线性规划可视化配置"""
    
    COLORS = {
        'optimal': '#E94F37',       # 最优点颜色
        'feasible': '#2E86AB',      # 可行域颜色
        'constraint': '#F18F01',    # 约束线颜色
        'contour': '#6B4C9A',       # 等高线颜色
        'path': '#27AE60',          # 迭代路径颜色
        'grid': '#E0E0E0'
    }
    
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

NLPPlotConfig.setup()


# ============================================================
# 第二部分：数据预处理模块
# ============================================================

class NLPDataPreprocessor:
    """
    非线性规划数据预处理器
    
    功能：
    1. 数据清洗（缺失值、异常值）
    2. 数据标准化
    3. 参数范围估计
    4. 初始点选择
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
    
    def clean_data(self, data: Union[np.ndarray, pd.DataFrame], 
                   method: str = 'median') -> np.ndarray:
        """
        数据清洗
        
        :param data: 输入数据
        :param method: 缺失值填充方法 ('mean', 'median', 'drop')
        :return: 清洗后的数据
        """
        self._log("开始数据清洗...")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        data = np.array(data, dtype=float)
        
        # 处理缺失值
        nan_count = np.sum(np.isnan(data))
        if nan_count > 0:
            self._log(f"  发现 {nan_count} 个缺失值")
            if method == 'mean':
                col_means = np.nanmean(data, axis=0)
                for i in range(data.shape[1]):
                    data[np.isnan(data[:, i]), i] = col_means[i]
            elif method == 'median':
                col_medians = np.nanmedian(data, axis=0)
                for i in range(data.shape[1]):
                    data[np.isnan(data[:, i]), i] = col_medians[i]
            elif method == 'drop':
                data = data[~np.any(np.isnan(data), axis=1)]
            self._log(f"  使用 {method} 方法处理完成")
        else:
            self._log("  未发现缺失值")
        
        return data
    
    def detect_outliers(self, data: np.ndarray, 
                        method: str = 'zscore', 
                        threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        异常值检测
        
        :param method: 'zscore' 或 'iqr'
        :param threshold: 阈值
        :return: (清洗后数据, 异常值索引)
        """
        self._log(f"异常值检测 (方法: {method}, 阈值: {threshold})...")
        
        if method == 'zscore':
            z_scores = np.abs(zscore(data, axis=0, nan_policy='omit'))
            outlier_mask = np.any(z_scores > threshold, axis=1)
        elif method == 'iqr':
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outlier_mask = np.any((data < lower) | (data > upper), axis=1)
        
        outlier_indices = np.where(outlier_mask)[0]
        clean_data = data[~outlier_mask]
        
        self._log(f"  检测到 {len(outlier_indices)} 个异常样本")
        
        return clean_data, outlier_indices
    
    def normalize(self, data: np.ndarray, 
                  method: str = 'minmax') -> Tuple[np.ndarray, Dict]:
        """
        数据标准化
        
        :param method: 'minmax', 'zscore', 'robust'
        :return: (标准化数据, 参数字典用于反标准化)
        """
        self._log(f"数据标准化 (方法: {method})...")
        
        params = {'method': method}
        
        if method == 'minmax':
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # 避免除零
            normalized = (data - min_vals) / range_vals
            params['min'] = min_vals
            params['max'] = max_vals
        elif method == 'zscore':
            mean_vals = np.mean(data, axis=0)
            std_vals = np.std(data, axis=0)
            std_vals[std_vals == 0] = 1
            normalized = (data - mean_vals) / std_vals
            params['mean'] = mean_vals
            params['std'] = std_vals
        elif method == 'robust':
            median_vals = np.median(data, axis=0)
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            IQR[IQR == 0] = 1
            normalized = (data - median_vals) / IQR
            params['median'] = median_vals
            params['IQR'] = IQR
        
        return normalized, params
    
    def estimate_bounds(self, data: np.ndarray, 
                        expand_ratio: float = 0.2) -> List[Tuple[float, float]]:
        """
        基于数据估计变量边界
        
        :param expand_ratio: 边界扩展比例
        :return: 边界列表 [(min1, max1), (min2, max2), ...]
        """
        self._log("估计变量边界...")
        
        bounds = []
        for i in range(data.shape[1]):
            col = data[:, i]
            min_val, max_val = np.min(col), np.max(col)
            range_val = max_val - min_val
            lower = min_val - expand_ratio * range_val
            upper = max_val + expand_ratio * range_val
            bounds.append((lower, upper))
            self._log(f"  变量 x{i+1}: [{lower:.4f}, {upper:.4f}]")
        
        return bounds
    
    def generate_initial_points(self, bounds: List[Tuple], 
                                n_points: int = 10,
                                method: str = 'random') -> np.ndarray:
        """
        生成多个初始点用于多起点优化
        
        :param method: 'random', 'latin', 'grid'
        :return: 初始点数组 (n_points, n_dim)
        """
        n_dim = len(bounds)
        
        if method == 'random':
            points = np.zeros((n_points, n_dim))
            for i, (lb, ub) in enumerate(bounds):
                points[:, i] = np.random.uniform(lb, ub, n_points)
        elif method == 'latin':
            # 拉丁超立方采样
            points = np.zeros((n_points, n_dim))
            for i, (lb, ub) in enumerate(bounds):
                perm = np.random.permutation(n_points)
                points[:, i] = lb + (perm + np.random.rand(n_points)) * (ub - lb) / n_points
        elif method == 'grid':
            # 网格采样
            n_per_dim = max(2, int(n_points ** (1/n_dim)))
            grids = [np.linspace(lb, ub, n_per_dim) for lb, ub in bounds]
            mesh = np.meshgrid(*grids)
            points = np.column_stack([m.ravel() for m in mesh])[:n_points]
        
        self._log(f"生成 {len(points)} 个初始点 (方法: {method})")
        return points


# ============================================================
# 第三部分：非线性规划求解器
# ============================================================

class NonlinearProgrammingSolver:
    """
    非线性规划求解器
    
    支持：
    1. 无约束优化
    2. 等式约束
    3. 不等式约束
    4. 边界约束
    5. 多起点全局优化
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.history = []  # 记录迭代历史
        self.result = None
        
    def _callback(self, x):
        """迭代回调函数，记录优化路径"""
        self.history.append(x.copy())
    
    def solve(self, 
              objective: Callable,
              x0: np.ndarray,
              bounds: Optional[List[Tuple]] = None,
              constraints: Optional[List[Dict]] = None,
              method: str = 'SLSQP',
              options: Optional[Dict] = None) -> Dict:
        """
        求解非线性规划问题
        
        :param objective: 目标函数 f(x) -> float
        :param x0: 初始点
        :param bounds: 变量边界 [(min, max), ...]
        :param constraints: 约束条件列表
            [{'type': 'ineq', 'fun': g}, {'type': 'eq', 'fun': h}]
            不等式约束: g(x) >= 0
            等式约束: h(x) = 0
        :param method: 'SLSQP', 'trust-constr', 'COBYLA', 'L-BFGS-B'
        :param options: 求解器选项
        :return: 结果字典
        """
        self.history = []
        
        if self.verbose:
            print("\n" + "="*60)
            print("   非线性规划求解器 (NLP Solver)")
            print("="*60)
            print(f"  方法: {method}")
            print(f"  变量维度: {len(x0)}")
            print(f"  初始点: {x0}")
        
        default_options = {
            'maxiter': 1000,
            'ftol': 1e-8,
            'disp': False
        }
        if options:
            default_options.update(options)
        
        # 调用scipy优化器
        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            constraints=constraints or [],
            options=default_options,
            callback=self._callback
        )
        
        self.result = {
            'success': result.success,
            'x': result.x,
            'fun': result.fun,
            'message': result.message,
            'nit': result.nit if hasattr(result, 'nit') else len(self.history),
            'nfev': result.nfev if hasattr(result, 'nfev') else 0,
            'history': np.array(self.history) if self.history else None
        }
        
        if self.verbose:
            self._print_result()
        
        return self.result
    
    def multistart_solve(self,
                         objective: Callable,
                         bounds: List[Tuple],
                         n_starts: int = 10,
                         constraints: Optional[List[Dict]] = None,
                         method: str = 'SLSQP') -> Dict:
        """
        多起点全局优化
        
        :param n_starts: 起始点数量
        :return: 最优结果
        """
        if self.verbose:
            print(f"\n多起点优化: {n_starts} 个起始点")
        
        preprocessor = NLPDataPreprocessor(verbose=False)
        initial_points = preprocessor.generate_initial_points(bounds, n_starts, 'latin')
        
        best_result = None
        all_results = []
        
        for i, x0 in enumerate(initial_points):
            self.history = []
            result = self.solve(objective, x0, bounds, constraints, method)
            all_results.append(result)
            
            if result['success']:
                if best_result is None or result['fun'] < best_result['fun']:
                    best_result = result
        
        if best_result is None and all_results:
            best_result = min(all_results, key=lambda r: r['fun'])
        
        if self.verbose:
            print(f"\n最优解来自第 {all_results.index(best_result)+1} 个起始点")
        
        self.result = best_result
        return best_result
    
    def global_solve(self,
                     objective: Callable,
                     bounds: List[Tuple],
                     constraints: Optional[List[Dict]] = None,
                     maxiter: int = 1000) -> Dict:
        """
        全局优化（差分进化算法）
        
        适用于非凸问题或存在多个局部最优的情况
        """
        if self.verbose:
            print("\n全局优化 (差分进化算法)")
        
        # 差分进化不直接支持约束，使用惩罚函数法
        if constraints:
            penalty_weight = 1e6
            
            def penalized_objective(x):
                val = objective(x)
                for con in constraints:
                    c_val = con['fun'](x)
                    if con['type'] == 'ineq':
                        val += penalty_weight * max(0, -c_val) ** 2
                    elif con['type'] == 'eq':
                        val += penalty_weight * c_val ** 2
                return val
        else:
            penalized_objective = objective
        
        result = differential_evolution(
            penalized_objective,
            bounds,
            maxiter=maxiter,
            seed=42,
            polish=True
        )
        
        self.result = {
            'success': result.success,
            'x': result.x,
            'fun': objective(result.x),  # 返回原始目标值
            'message': result.message,
            'nit': result.nit,
            'nfev': result.nfev,
            'history': None
        }
        
        if self.verbose:
            self._print_result()
        
        return self.result
    
    def _print_result(self):
        """打印求解结果"""
        r = self.result
        print("\n" + "-"*50)
        print("📊 求解结果")
        print("-"*50)
        print(f"  状态: {'✅ 成功' if r['success'] else '❌ 失败'}")
        print(f"  最优解: {r['x']}")
        print(f"  最优目标值: {r['fun']:.6f}")
        print(f"  迭代次数: {r['nit']}")
        print(f"  函数评估次数: {r['nfev']}")
        print(f"  消息: {r['message']}")
        print("-"*50)


# ============================================================
# 第四部分：灵敏度分析
# ============================================================

class NLPSensitivityAnalyzer:
    """
    灵敏度分析器
    
    功能：
    1. 参数灵敏度分析
    2. 约束活跃性分析
    3. 影子价格计算
    """
    
    def __init__(self, solver: NonlinearProgrammingSolver):
        self.solver = solver
        self.results = {}
    
    def parameter_sensitivity(self,
                              objective_builder: Callable,
                              param_name: str,
                              param_values: np.ndarray,
                              base_x0: np.ndarray,
                              bounds: List[Tuple],
                              constraints: Optional[List[Dict]] = None) -> Dict:
        """
        参数灵敏度分析
        
        :param objective_builder: 给定参数返回目标函数的函数
        :param param_name: 参数名称
        :param param_values: 参数取值范围
        :return: 分析结果
        """
        print(f"\n参数灵敏度分析: {param_name}")
        print("-"*40)
        
        optimal_values = []
        optimal_objectives = []
        
        for param in param_values:
            obj_func = objective_builder(param)
            result = self.solver.solve(obj_func, base_x0, bounds, constraints)
            
            if result['success']:
                optimal_values.append(result['x'])
                optimal_objectives.append(result['fun'])
            else:
                optimal_values.append(None)
                optimal_objectives.append(np.nan)
        
        self.results[param_name] = {
            'param_values': param_values,
            'optimal_solutions': optimal_values,
            'optimal_objectives': optimal_objectives
        }
        
        return self.results[param_name]
    
    def plot_sensitivity(self, param_name: str, save_path: Optional[str] = None):
        """绘制灵敏度分析图"""
        if param_name not in self.results:
            print(f"未找到参数 {param_name} 的分析结果")
            return
        
        data = self.results[param_name]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(data['param_values'], data['optimal_objectives'],
                'o-', color=NLPPlotConfig.COLORS['optimal'],
                linewidth=2, markersize=8, label='最优目标值')
        
        ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('最优目标值', fontsize=12, fontweight='bold')
        ax.set_title(f'灵敏度分析: {param_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 第五部分：可视化模块
# ============================================================

class NLPVisualizer:
    """
    非线性规划可视化器
    
    功能：
    1. 目标函数等高线图
    2. 可行域可视化
    3. 优化路径动画
    4. 结果汇总图
    """
    
    def __init__(self, save_dir: str = './figures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_contour_with_constraints(self,
                                      objective: Callable,
                                      bounds: List[Tuple],
                                      constraints: Optional[List[Dict]] = None,
                                      optimal_point: Optional[np.ndarray] = None,
                                      history: Optional[np.ndarray] = None,
                                      title: str = '非线性规划问题',
                                      save_name: Optional[str] = None):
        """
        绘制2D问题的等高线图与约束
        
        :param objective: 目标函数
        :param bounds: 变量边界
        :param constraints: 约束条件
        :param optimal_point: 最优解
        :param history: 迭代历史
        """
        if len(bounds) != 2:
            print("等高线图仅支持2维问题")
            return
        
        x1_range = np.linspace(bounds[0][0], bounds[0][1], 100)
        x2_range = np.linspace(bounds[1][0], bounds[1][1], 100)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        
        Z = np.zeros_like(X1)
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                Z[i, j] = objective(np.array([X1[i, j], X2[i, j]]))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 等高线
        contour = ax.contour(X1, X2, Z, levels=20, colors=NLPPlotConfig.COLORS['contour'], alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)
        contourf = ax.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.3)
        plt.colorbar(contourf, ax=ax, label='目标函数值')
        
        # 约束边界
        if constraints:
            for i, con in enumerate(constraints):
                C = np.zeros_like(X1)
                for ii in range(X1.shape[0]):
                    for jj in range(X1.shape[1]):
                        C[ii, jj] = con['fun'](np.array([X1[ii, jj], X2[ii, jj]]))
                
                if con['type'] == 'ineq':
                    ax.contour(X1, X2, C, levels=[0], colors=NLPPlotConfig.COLORS['constraint'],
                              linewidths=2, linestyles='--')
                    ax.contourf(X1, X2, C, levels=[0, np.inf], colors=[NLPPlotConfig.COLORS['feasible']],
                               alpha=0.1)
                elif con['type'] == 'eq':
                    ax.contour(X1, X2, C, levels=[0], colors='red', linewidths=2)
        
        # 优化路径
        if history is not None and len(history) > 1:
            ax.plot(history[:, 0], history[:, 1], 'o-',
                   color=NLPPlotConfig.COLORS['path'], 
                   linewidth=1.5, markersize=4, alpha=0.7, label='优化路径')
        
        # 最优点
        if optimal_point is not None:
            ax.scatter(optimal_point[0], optimal_point[1],
                      c=NLPPlotConfig.COLORS['optimal'], s=200, marker='*',
                      edgecolor='white', linewidth=2, zorder=5, label='最优解')
        
        ax.set_xlabel('$x_1$', fontsize=12, fontweight='bold')
        ax.set_ylabel('$x_2$', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_convergence(self,
                         objective: Callable,
                         history: np.ndarray,
                         title: str = '收敛曲线',
                         save_name: Optional[str] = None):
        """绘制收敛曲线"""
        if history is None or len(history) == 0:
            print("无迭代历史数据")
            return
        
        objectives = [objective(x) for x in history]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(range(len(objectives)), objectives, 'o-',
               color=NLPPlotConfig.COLORS['optimal'], linewidth=2, markersize=4)
        
        ax.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
        ax.set_ylabel('目标函数值', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标注最优值
        min_idx = np.argmin(objectives)
        ax.axhline(y=objectives[min_idx], color='red', linestyle='--', alpha=0.5)
        ax.annotate(f'最优值: {objectives[min_idx]:.4f}', 
                   xy=(min_idx, objectives[min_idx]),
                   xytext=(min_idx + len(objectives)*0.1, objectives[min_idx]),
                   fontsize=10, color='red')
        
        plt.tight_layout()
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_solution_summary(self,
                              result: Dict,
                              variable_names: Optional[List[str]] = None,
                              save_name: Optional[str] = None):
        """绘制求解结果汇总图"""
        x = result['x']
        n_vars = len(x)
        
        if variable_names is None:
            variable_names = [f'$x_{i+1}$' for i in range(n_vars)]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：变量取值
        colors = NLPPlotConfig.COLORS
        bars = axes[0].bar(variable_names, x, color=colors['feasible'], 
                          edgecolor='white', linewidth=1.5)
        axes[0].set_ylabel('变量值', fontsize=12, fontweight='bold')
        axes[0].set_title('最优解各变量取值', fontsize=14, fontweight='bold')
        
        for bar, val in zip(bars, x):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 右图：求解信息
        info_text = f"""
求解状态: {'成功 ✅' if result['success'] else '失败 ❌'}

最优目标值: {result['fun']:.6f}

迭代次数: {result['nit']}

函数评估次数: {result['nfev']}

消息: {result['message']}
        """
        axes[1].text(0.1, 0.5, info_text, fontsize=12, 
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[1].axis('off')
        axes[1].set_title('求解信息', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 第六部分：完整工作流
# ============================================================

class NonlinearProgrammingPipeline:
    """
    非线性规划完整工作流
    
    集成数据预处理、模型求解、结果可视化
    """
    
    def __init__(self, verbose: bool = True, save_dir: str = './figures'):
        self.preprocessor = NLPDataPreprocessor(verbose)
        self.solver = NonlinearProgrammingSolver(verbose)
        self.visualizer = NLPVisualizer(save_dir)
        self.verbose = verbose
    
    def run(self,
            objective: Callable,
            bounds: List[Tuple],
            constraints: Optional[List[Dict]] = None,
            x0: Optional[np.ndarray] = None,
            method: str = 'SLSQP',
            multistart: bool = False,
            n_starts: int = 10,
            global_optimization: bool = False,
            plot_contour: bool = True,
            plot_convergence: bool = True,
            variable_names: Optional[List[str]] = None) -> Dict:
        """
        执行完整的非线性规划求解流程
        
        :param objective: 目标函数
        :param bounds: 变量边界
        :param constraints: 约束条件
        :param x0: 初始点（可选）
        :param method: 求解方法
        :param multistart: 是否使用多起点优化
        :param n_starts: 多起点数量
        :param global_optimization: 是否使用全局优化
        :param plot_contour: 是否绘制等高线图（仅2D）
        :param plot_convergence: 是否绘制收敛曲线
        :return: 求解结果
        """
        if self.verbose:
            print("\n" + "="*60)
            print("   非线性规划完整工作流")
            print("="*60)
        
        # 生成初始点
        if x0 is None:
            x0 = self.preprocessor.generate_initial_points(bounds, 1, 'random')[0]
        
        # 求解
        if global_optimization:
            result = self.solver.global_solve(objective, bounds, constraints)
        elif multistart:
            result = self.solver.multistart_solve(objective, bounds, n_starts, constraints, method)
        else:
            result = self.solver.solve(objective, x0, bounds, constraints, method)
        
        # 可视化
        if plot_contour and len(bounds) == 2:
            self.visualizer.plot_contour_with_constraints(
                objective, bounds, constraints,
                optimal_point=result['x'],
                history=result.get('history'),
                title='非线性规划求解结果'
            )
        
        if plot_convergence and result.get('history') is not None:
            self.visualizer.plot_convergence(objective, result['history'])
        
        self.visualizer.plot_solution_summary(result, variable_names)
        
        return result


# ============================================================
# 示例：投资组合优化问题
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("   示例：投资组合优化问题（风险最小化）")
    print("="*70)
    
    # 问题描述：4种资产，最小化风险的同时保证收益
    # 变量：x1, x2, x3, x4 为各资产投资比例
    
    # 预期收益率
    expected_returns = np.array([0.12, 0.08, 0.05, 0.06])  # 科技股、消费股、债券、黄金
    
    # 协方差矩阵（风险相关性）
    cov_matrix = np.array([
        [0.04, 0.01, -0.005, 0.002],
        [0.01, 0.02, 0.003, 0.001],
        [-0.005, 0.003, 0.01, -0.002],
        [0.002, 0.001, -0.002, 0.015]
    ])
    
    # 目标函数：最小化投资组合风险（方差）
    def portfolio_risk(x):
        return x @ cov_matrix @ x
    
    # 约束条件
    constraints = [
        # 投资比例和为1
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        # 预期收益至少7%
        {'type': 'ineq', 'fun': lambda x: np.dot(expected_returns, x) - 0.07}
    ]
    
    # 边界：每种资产投资比例在0到1之间
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
    
    # 创建工作流
    pipeline = NonlinearProgrammingPipeline(verbose=True)
    
    # 求解
    result = pipeline.run(
        objective=portfolio_risk,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        multistart=True,
        n_starts=5,
        plot_contour=False,  # 4维问题不绘制等高线
        variable_names=['科技股', '消费股', '债券', '黄金']
    )
    
    # 输出详细结果
    print("\n" + "="*50)
    print("📈 投资组合优化结果")
    print("="*50)
    x = result['x']
    print(f"科技股投资比例: {x[0]*100:.2f}%")
    print(f"消费股投资比例: {x[1]*100:.2f}%")
    print(f"债券投资比例:   {x[2]*100:.2f}%")
    print(f"黄金投资比例:   {x[3]*100:.2f}%")
    print(f"\n预期收益率: {np.dot(expected_returns, x)*100:.2f}%")
    print(f"投资组合风险(标准差): {np.sqrt(result['fun'])*100:.2f}%")
    print("="*50)
