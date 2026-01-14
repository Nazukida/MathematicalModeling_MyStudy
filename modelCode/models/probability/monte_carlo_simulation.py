"""
============================================================
蒙特卡洛模拟 (Monte Carlo Simulation)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：概率模拟、风险分析、数值积分、随机过程模拟
原理：通过大量随机采样近似期望值和概率分布
作者：MCM/ICM Team
日期：2026年1月
============================================================

应用场景：
- 金融风险评估（VaR计算）
- 项目时间/成本估计
- 物理系统模拟
- 排队系统分析
- 复杂积分计算
- 期权定价
- 可靠性分析

核心思想：
E[f(X)] ≈ (1/N) Σ f(Xᵢ)，Xᵢ ~ P(X)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from visualization.plot_config import PlotStyleConfig, FigureSaver

PlotStyleConfig.setup_style()
warnings.filterwarnings('ignore')


class MonteCarloSimulator:
    """
    蒙特卡洛模拟基础类
    
    核心功能：
    1. 随机变量采样
    2. 期望值估计
    3. 置信区间计算
    4. 收敛性分析
    5. 方差缩减技术
    """
    
    def __init__(self, n_simulations=10000, random_seed=42, verbose=True):
        """
        初始化模拟器
        
        :param n_simulations: 模拟次数
        :param random_seed: 随机种子（可重复性）
        :param verbose: 是否打印详细信息
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.verbose = verbose
        
        np.random.seed(random_seed)
        
        self.results = None
        self.mean = None
        self.std = None
        self.ci_lower = None
        self.ci_upper = None
        
    def simulate(self, simulation_func, *args, **kwargs):
        """
        执行蒙特卡洛模拟
        
        :param simulation_func: 单次模拟函数，返回一个数值结果
        :param args, kwargs: 传递给模拟函数的参数
        :return: 模拟结果数组
        """
        self.results = np.array([simulation_func(*args, **kwargs) 
                                 for _ in range(self.n_simulations)])
        
        self._calculate_statistics()
        
        if self.verbose:
            self._print_results()
            
        return self.results
    
    def simulate_vectorized(self, simulation_func, *args, **kwargs):
        """
        向量化蒙特卡洛模拟（更快）
        
        :param simulation_func: 向量化模拟函数，接受n_simulations参数
        """
        self.results = simulation_func(self.n_simulations, *args, **kwargs)
        self._calculate_statistics()
        
        if self.verbose:
            self._print_results()
            
        return self.results
    
    def _calculate_statistics(self):
        """计算统计量"""
        self.mean = np.mean(self.results)
        self.std = np.std(self.results)
        se = self.std / np.sqrt(self.n_simulations)
        self.ci_lower = self.mean - 1.96 * se
        self.ci_upper = self.mean + 1.96 * se
        
    def _print_results(self):
        """打印模拟结果"""
        print("\n" + "="*55)
        print("🎲 蒙特卡洛模拟结果")
        print("="*55)
        print(f"\n  模拟次数: {self.n_simulations:,}")
        print(f"\n  统计摘要:")
        print(f"    均值: {self.mean:.4f}")
        print(f"    标准差: {self.std:.4f}")
        print(f"    最小值: {np.min(self.results):.4f}")
        print(f"    最大值: {np.max(self.results):.4f}")
        print(f"    中位数: {np.median(self.results):.4f}")
        print(f"\n  95% 置信区间: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]")
        print("="*55)
    
    def percentile(self, q):
        """计算分位数"""
        return np.percentile(self.results, q)
    
    def probability_above(self, threshold):
        """P(X > threshold)"""
        prob = np.mean(self.results > threshold)
        if self.verbose:
            print(f"\n  P(X > {threshold}) = {prob:.4f} ({prob*100:.2f}%)")
        return prob
    
    def probability_below(self, threshold):
        """P(X < threshold)"""
        prob = np.mean(self.results < threshold)
        if self.verbose:
            print(f"\n  P(X < {threshold}) = {prob:.4f} ({prob*100:.2f}%)")
        return prob
    
    def probability_between(self, lower, upper):
        """P(lower < X < upper)"""
        prob = np.mean((self.results > lower) & (self.results < upper))
        if self.verbose:
            print(f"\n  P({lower} < X < {upper}) = {prob:.4f} ({prob*100:.2f}%)")
        return prob
    
    def value_at_risk(self, confidence=0.95):
        """
        计算VaR（风险价值）
        
        :param confidence: 置信水平
        :return: VaR值
        """
        var = np.percentile(self.results, (1 - confidence) * 100)
        if self.verbose:
            print(f"\n  VaR ({confidence*100:.0f}%): {var:.4f}")
        return var
    
    def conditional_value_at_risk(self, confidence=0.95):
        """
        计算CVaR（条件风险价值，也叫Expected Shortfall）
        """
        var = self.value_at_risk(confidence)
        cvar = np.mean(self.results[self.results <= var])
        if self.verbose:
            print(f"  CVaR ({confidence*100:.0f}%): {cvar:.4f}")
        return cvar
    
    # ==================== 可视化方法 ====================
    
    def plot_distribution(self, title='蒙特卡洛模拟结果分布', save_path=None):
        """绘制结果分布"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 直方图
        n, bins, patches = ax.hist(self.results, bins=50, density=True, 
                                   color=PlotStyleConfig.COLORS['primary'],
                                   alpha=0.7, edgecolor='white', linewidth=1.2)
        
        # KDE曲线
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(self.results)
        x_range = np.linspace(self.results.min(), self.results.max(), 200)
        ax.plot(x_range, kde(x_range), color=PlotStyleConfig.COLORS['danger'],
               linewidth=2.5, label='KDE估计')
        
        # 标记均值和置信区间
        ax.axvline(self.mean, color=PlotStyleConfig.COLORS['neutral'],
                  linestyle='--', linewidth=2, label=f'均值 = {self.mean:.3f}')
        ax.axvline(self.ci_lower, color=PlotStyleConfig.COLORS['accent'],
                  linestyle=':', linewidth=2)
        ax.axvline(self.ci_upper, color=PlotStyleConfig.COLORS['accent'],
                  linestyle=':', linewidth=2, label='95% CI')
        
        ax.set_xlabel('模拟结果', fontsize=12, fontweight='bold')
        ax.set_ylabel('概率密度', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right')
        
        # 添加统计信息
        textstr = f'n = {self.n_simulations:,}\n'
        textstr += f'μ = {self.mean:.4f}\n'
        textstr += f'σ = {self.std:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax
    
    def plot_convergence(self, save_path=None):
        """绘制收敛性分析图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 累积均值收敛
        cumulative_mean = np.cumsum(self.results) / np.arange(1, len(self.results) + 1)
        axes[0].plot(cumulative_mean, color=PlotStyleConfig.COLORS['primary'],
                    linewidth=1.5, alpha=0.8)
        axes[0].axhline(self.mean, color=PlotStyleConfig.COLORS['danger'],
                       linestyle='--', linewidth=2, label=f'最终均值 = {self.mean:.4f}')
        axes[0].set_xlabel('模拟次数', fontweight='bold')
        axes[0].set_ylabel('累积均值', fontweight='bold')
        axes[0].set_title('均值收敛性', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].set_xscale('log')
        
        # 标准误收敛
        n_points = min(1000, self.n_simulations)
        sample_sizes = np.logspace(1, np.log10(self.n_simulations), n_points).astype(int)
        sample_sizes = np.unique(sample_sizes)
        
        std_errors = []
        for n in sample_sizes:
            se = np.std(self.results[:n]) / np.sqrt(n)
            std_errors.append(se)
        
        axes[1].plot(sample_sizes, std_errors, color=PlotStyleConfig.COLORS['secondary'],
                    linewidth=2)
        axes[1].set_xlabel('模拟次数', fontweight='bold')
        axes[1].set_ylabel('标准误', fontweight='bold')
        axes[1].set_title('标准误收敛性', fontsize=12, fontweight='bold')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.suptitle('蒙特卡洛模拟收敛分析', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, axes


class ProjectRiskSimulator(MonteCarloSimulator):
    """
    项目风险模拟器
    
    适用场景：
    - 项目工期估计
    - 成本预算分析
    - 资源需求评估
    """
    
    def __init__(self, n_simulations=10000, random_seed=42, verbose=True):
        super().__init__(n_simulations, random_seed, verbose)
        self.tasks = []
        
    def add_task(self, name, optimistic, most_likely, pessimistic, 
                 distribution='pert'):
        """
        添加任务（三点估计）
        
        :param name: 任务名称
        :param optimistic: 乐观估计
        :param most_likely: 最可能估计
        :param pessimistic: 悲观估计
        :param distribution: 'pert'(PERT分布) / 'triangular'(三角分布)
        """
        self.tasks.append({
            'name': name,
            'optimistic': optimistic,
            'most_likely': most_likely,
            'pessimistic': pessimistic,
            'distribution': distribution
        })
        
    def _sample_task(self, task):
        """对单个任务进行采样"""
        o, m, p = task['optimistic'], task['most_likely'], task['pessimistic']
        
        if task['distribution'] == 'pert':
            # PERT分布（Beta分布变体）
            mu = (o + 4*m + p) / 6
            sigma = (p - o) / 6
            # 使用正态近似
            return max(o, min(p, np.random.normal(mu, sigma)))
        else:
            # 三角分布
            return np.random.triangular(o, m, p)
    
    def simulate_project(self, method='sequential'):
        """
        模拟项目
        
        :param method: 'sequential'(顺序执行) / 'parallel'(并行执行，取最大)
        """
        def single_simulation():
            task_durations = [self._sample_task(t) for t in self.tasks]
            if method == 'sequential':
                return sum(task_durations)
            else:
                return max(task_durations)
        
        self.simulate(single_simulation)
        return self.results
    
    def plot_task_distributions(self, save_path=None):
        """绘制各任务的分布"""
        n_tasks = len(self.tasks)
        fig, axes = plt.subplots(1, n_tasks, figsize=(4*n_tasks, 4))
        
        if n_tasks == 1:
            axes = [axes]
        
        colors = PlotStyleConfig.get_palette(n_tasks)
        
        for i, (task, color) in enumerate(zip(self.tasks, colors)):
            samples = np.array([self._sample_task(task) for _ in range(5000)])
            
            axes[i].hist(samples, bins=30, density=True, color=color, 
                        alpha=0.7, edgecolor='white')
            axes[i].axvline(task['most_likely'], color='red', linestyle='--',
                           label=f'最可能: {task["most_likely"]}')
            axes[i].set_title(task['name'], fontweight='bold')
            axes[i].set_xlabel('时长')
            axes[i].legend(fontsize=8)
            
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
        
        plt.suptitle('各任务时长分布', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, axes


class FinancialMonteCarlo(MonteCarloSimulator):
    """
    金融蒙特卡洛模拟器
    
    适用场景：
    - 投资组合风险分析
    - 期权定价
    - 退休规划
    """
    
    def __init__(self, n_simulations=10000, random_seed=42, verbose=True):
        super().__init__(n_simulations, random_seed, verbose)
        
    def geometric_brownian_motion(self, S0, mu, sigma, T, n_steps=252):
        """
        几何布朗运动（股票价格模拟）
        
        :param S0: 初始价格
        :param mu: 年化收益率
        :param sigma: 年化波动率
        :param T: 时间（年）
        :param n_steps: 时间步数
        :return: 价格路径矩阵 (n_simulations, n_steps+1)
        """
        dt = T / n_steps
        paths = np.zeros((self.n_simulations, n_steps + 1))
        paths[:, 0] = S0
        
        for t in range(1, n_steps + 1):
            Z = np.random.standard_normal(self.n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
        
        self.results = paths[:, -1]  # 最终价格
        self._calculate_statistics()
        
        if self.verbose:
            print("\n" + "="*55)
            print("📈 几何布朗运动模拟")
            print("="*55)
            print(f"  初始价格: {S0}")
            print(f"  年化收益率: {mu*100:.1f}%")
            print(f"  年化波动率: {sigma*100:.1f}%")
            print(f"  模拟期限: {T} 年")
            print(f"\n  最终价格统计:")
            print(f"    均值: {self.mean:.2f}")
            print(f"    标准差: {self.std:.2f}")
            print(f"    95% CI: [{self.ci_lower:.2f}, {self.ci_upper:.2f}]")
            print("="*55)
        
        return paths
    
    def black_scholes_option(self, S0, K, r, sigma, T, option_type='call'):
        """
        使用蒙特卡洛计算欧式期权价格
        
        :param S0: 标的资产现价
        :param K: 执行价格
        :param r: 无风险利率
        :param sigma: 波动率
        :param T: 到期时间（年）
        :param option_type: 'call' / 'put'
        """
        # 模拟最终价格
        Z = np.random.standard_normal(self.n_simulations)
        ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        
        # 计算收益
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # 折现
        option_price = np.exp(-r * T) * np.mean(payoffs)
        option_std = np.exp(-r * T) * np.std(payoffs) / np.sqrt(self.n_simulations)
        
        self.results = payoffs
        self._calculate_statistics()
        
        if self.verbose:
            print("\n" + "="*55)
            print(f"📊 欧式{option_type.upper()}期权蒙特卡洛定价")
            print("="*55)
            print(f"  标的价格 S₀: {S0}")
            print(f"  执行价格 K: {K}")
            print(f"  无风险利率 r: {r*100:.1f}%")
            print(f"  波动率 σ: {sigma*100:.1f}%")
            print(f"  到期时间 T: {T} 年")
            print(f"\n  期权价格: {option_price:.4f} ± {1.96*option_std:.4f}")
            print("="*55)
        
        return option_price, option_std
    
    def portfolio_simulation(self, initial_value, returns_mean, returns_cov, 
                            weights, years=10):
        """
        投资组合模拟
        
        :param initial_value: 初始投资额
        :param returns_mean: 各资产年化收益率向量
        :param returns_cov: 收益率协方差矩阵
        :param weights: 投资权重
        :param years: 投资年限
        """
        n_assets = len(weights)
        weights = np.array(weights)
        returns_mean = np.array(returns_mean)
        
        # 模拟多年收益
        final_values = np.zeros(self.n_simulations)
        
        for i in range(self.n_simulations):
            value = initial_value
            for _ in range(years):
                # 从多元正态分布采样年收益率
                annual_returns = np.random.multivariate_normal(returns_mean, returns_cov)
                portfolio_return = np.dot(weights, annual_returns)
                value *= (1 + portfolio_return)
            final_values[i] = value
        
        self.results = final_values
        self._calculate_statistics()
        
        if self.verbose:
            print("\n" + "="*55)
            print("💰 投资组合蒙特卡洛模拟")
            print("="*55)
            print(f"  初始投资: {initial_value:,.0f}")
            print(f"  投资年限: {years} 年")
            print(f"\n  最终价值统计:")
            print(f"    均值: {self.mean:,.0f}")
            print(f"    中位数: {np.median(self.results):,.0f}")
            print(f"    5%分位: {self.percentile(5):,.0f}")
            print(f"    95%分位: {self.percentile(95):,.0f}")
            print(f"\n  风险指标:")
            print(f"    VaR(95%): {self.value_at_risk(0.95):,.0f}")
            print("="*55)
        
        return self.results
    
    def plot_price_paths(self, paths, n_paths=100, save_path=None):
        """绘制价格路径"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 绘制部分路径
        for i in range(min(n_paths, len(paths))):
            ax.plot(paths[i], alpha=0.2, linewidth=0.5, 
                   color=PlotStyleConfig.COLORS['primary'])
        
        # 绘制均值路径
        mean_path = np.mean(paths, axis=0)
        ax.plot(mean_path, color=PlotStyleConfig.COLORS['danger'], 
               linewidth=2.5, label='均值路径')
        
        # 绘制分位数带
        q5 = np.percentile(paths, 5, axis=0)
        q95 = np.percentile(paths, 95, axis=0)
        ax.fill_between(range(len(mean_path)), q5, q95, 
                       color=PlotStyleConfig.COLORS['accent'], alpha=0.3, 
                       label='90% 置信带')
        
        ax.set_xlabel('时间步', fontsize=12, fontweight='bold')
        ax.set_ylabel('价格', fontsize=12, fontweight='bold')
        ax.set_title('蒙特卡洛价格路径模拟', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper left')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax


class MonteCarloIntegration:
    """
    蒙特卡洛积分
    
    用于计算高维积分或复杂区域上的积分
    """
    
    def __init__(self, n_samples=100000, random_seed=42, verbose=True):
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.verbose = verbose
        np.random.seed(random_seed)
        
    def integrate(self, func, bounds, importance_sampling=False):
        """
        计算定积分
        
        :param func: 被积函数 f(x) 或 f(x1, x2, ...)
        :param bounds: 积分范围列表 [(a1, b1), (a2, b2), ...]
        :return: 积分估计值，标准误
        """
        bounds = np.array(bounds)
        n_dims = len(bounds)
        
        # 计算积分区域体积
        volume = np.prod(bounds[:, 1] - bounds[:, 0])
        
        # 均匀采样
        samples = np.random.uniform(
            bounds[:, 0], bounds[:, 1], 
            size=(self.n_samples, n_dims)
        )
        
        # 计算函数值
        if n_dims == 1:
            func_values = np.array([func(x[0]) for x in samples])
        else:
            func_values = np.array([func(*x) for x in samples])
        
        # 估计积分
        integral = volume * np.mean(func_values)
        std_error = volume * np.std(func_values) / np.sqrt(self.n_samples)
        
        if self.verbose:
            print("\n" + "="*55)
            print("∫ 蒙特卡洛积分")
            print("="*55)
            print(f"  维度: {n_dims}")
            print(f"  采样点: {self.n_samples:,}")
            print(f"  积分区域体积: {volume:.4f}")
            print(f"\n  积分估计: {integral:.6f}")
            print(f"  标准误: {std_error:.6f}")
            print(f"  95% CI: [{integral-1.96*std_error:.6f}, {integral+1.96*std_error:.6f}]")
            print("="*55)
        
        return integral, std_error
    
    def estimate_pi(self):
        """
        经典案例：蒙特卡洛估计π
        """
        x = np.random.uniform(-1, 1, self.n_samples)
        y = np.random.uniform(-1, 1, self.n_samples)
        
        inside = (x**2 + y**2) <= 1
        pi_estimate = 4 * np.mean(inside)
        std_error = 4 * np.std(inside) / np.sqrt(self.n_samples)
        
        if self.verbose:
            print("\n" + "="*55)
            print("🥧 蒙特卡洛估计 π")
            print("="*55)
            print(f"  采样点: {self.n_samples:,}")
            print(f"  落在圆内: {np.sum(inside):,} ({np.mean(inside)*100:.2f}%)")
            print(f"\n  π 估计值: {pi_estimate:.6f}")
            print(f"  真实值: {np.pi:.6f}")
            print(f"  误差: {abs(pi_estimate - np.pi):.6f}")
            print("="*55)
        
        return pi_estimate


if __name__ == "__main__":
    print("="*60)
    print("🎲 蒙特卡洛模拟演示")
    print("="*60)
    
    # ================== 示例1: 基础模拟 ==================
    print("\n" + "="*60)
    print("示例1: 基础蒙特卡洛模拟")
    print("="*60)
    
    mc = MonteCarloSimulator(n_simulations=50000)
    
    # 模拟：掷两个骰子的和
    def dice_sum():
        return np.random.randint(1, 7) + np.random.randint(1, 7)
    
    results = mc.simulate(dice_sum)
    mc.probability_above(7)
    
    fig1, ax1 = mc.plot_distribution(title='两骰子之和分布')
    plt.show()
    
    # ================== 示例2: 项目风险模拟 ==================
    print("\n" + "="*60)
    print("示例2: 项目工期风险模拟")
    print("="*60)
    
    project_mc = ProjectRiskSimulator(n_simulations=10000)
    project_mc.add_task('需求分析', optimistic=5, most_likely=7, pessimistic=12)
    project_mc.add_task('设计', optimistic=10, most_likely=15, pessimistic=25)
    project_mc.add_task('开发', optimistic=20, most_likely=30, pessimistic=50)
    project_mc.add_task('测试', optimistic=8, most_likely=12, pessimistic=20)
    
    project_mc.simulate_project(method='sequential')
    
    print(f"\n  90%概率能在 {project_mc.percentile(90):.1f} 天内完成")
    
    fig2, ax2 = project_mc.plot_distribution(title='项目总工期分布')
    plt.show()
    
    fig3, axes3 = project_mc.plot_task_distributions()
    plt.show()
    
    # ================== 示例3: 金融模拟 ==================
    print("\n" + "="*60)
    print("示例3: 股票价格模拟 (几何布朗运动)")
    print("="*60)
    
    fin_mc = FinancialMonteCarlo(n_simulations=10000)
    
    # 模拟股票价格（初始100，年化收益8%，波动率20%，1年）
    paths = fin_mc.geometric_brownian_motion(S0=100, mu=0.08, sigma=0.20, T=1)
    
    fig4, ax4 = fin_mc.plot_price_paths(paths)
    plt.show()
    
    fig5, ax5 = fin_mc.plot_distribution(title='一年后股票价格分布')
    plt.show()
    
    # 期权定价
    print("\n" + "="*60)
    print("示例4: 欧式期权蒙特卡洛定价")
    print("="*60)
    
    fin_mc.black_scholes_option(S0=100, K=105, r=0.05, sigma=0.2, T=0.5, option_type='call')
    
    # ================== 示例4: 蒙特卡洛积分 ==================
    print("\n" + "="*60)
    print("示例5: 蒙特卡洛积分")
    print("="*60)
    
    mc_int = MonteCarloIntegration(n_samples=100000)
    
    # 计算 ∫₀¹ x² dx = 1/3
    integral, se = mc_int.integrate(lambda x: x**2, [(0, 1)])
    print(f"  真实值: 0.333333")
    
    # 估计π
    mc_int.estimate_pi()
    
    print("\n✅ 蒙特卡洛模拟演示完成!")
