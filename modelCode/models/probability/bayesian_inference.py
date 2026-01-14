"""
============================================================
贝叶斯推断模型 (Bayesian Inference Model)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：由果推因、参数估计、不确定性量化、后验分布计算
原理：贝叶斯定理 P(θ|D) ∝ P(D|θ) × P(θ)
作者：MCM/ICM Team
日期：2026年1月
============================================================

应用场景：
- 逆问题：由观测结果推断原因/参数
- 参数不确定性量化
- 预测区间估计
- 模型更新（新数据到来时）
- 疾病诊断、设备故障诊断

核心公式：
后验 ∝ 似然 × 先验
P(θ|Data) ∝ P(Data|θ) × P(θ)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from visualization.plot_config import PlotStyleConfig, FigureSaver

PlotStyleConfig.setup_style()
warnings.filterwarnings('ignore')


class BayesianInference:
    """
    贝叶斯推断基础类
    
    核心功能：
    1. 共轭先验分析（正态-正态、Beta-二项等）
    2. 网格近似法
    3. MCMC采样（Metropolis-Hastings）
    4. 后验分布可视化
    5. 贝叶斯因子计算
    """
    
    def __init__(self, verbose=True):
        """
        初始化贝叶斯推断器
        
        :param verbose: 是否打印详细信息
        """
        self.verbose = verbose
        self.prior = None
        self.posterior = None
        self.data = None
        self.samples = None
        
    def _print_header(self, title):
        """打印标题"""
        print("\n" + "="*60)
        print(f"📊 {title}")
        print("="*60)


class NormalNormalBayes(BayesianInference):
    """
    正态-正态共轭模型
    
    场景：已知方差σ²，估计均值μ
    
    先验：μ ~ N(μ₀, τ₀²)
    似然：X|μ ~ N(μ, σ²)
    后验：μ|X ~ N(μₙ, τₙ²)
    
    适用：连续数据的均值估计，如测量值、评分等
    """
    
    def __init__(self, prior_mu=0, prior_tau=10, known_sigma=1, verbose=True):
        """
        初始化正态-正态模型
        
        :param prior_mu: 先验均值 μ₀
        :param prior_tau: 先验标准差 τ₀（反映不确定性）
        :param known_sigma: 已知的数据标准差 σ
        """
        super().__init__(verbose)
        self.prior_mu = prior_mu
        self.prior_tau = prior_tau
        self.known_sigma = known_sigma
        
        # 后验参数
        self.posterior_mu = None
        self.posterior_tau = None
        
    def fit(self, data):
        """
        根据数据更新后验分布
        
        :param data: 观测数据
        :return: self
        """
        self.data = np.array(data).flatten()
        n = len(self.data)
        x_bar = np.mean(self.data)
        
        # 共轭更新公式
        prior_precision = 1 / self.prior_tau**2
        likelihood_precision = n / self.known_sigma**2
        
        posterior_precision = prior_precision + likelihood_precision
        self.posterior_tau = 1 / np.sqrt(posterior_precision)
        
        self.posterior_mu = (prior_precision * self.prior_mu + 
                            likelihood_precision * x_bar) / posterior_precision
        
        if self.verbose:
            self._print_results()
            
        return self
    
    def _print_results(self):
        """打印推断结果"""
        self._print_header("正态-正态 贝叶斯推断")
        print(f"\n  📌 先验分布: N({self.prior_mu:.4f}, {self.prior_tau:.4f}²)")
        print(f"  📌 已知标准差: σ = {self.known_sigma:.4f}")
        print(f"\n  📊 观测数据:")
        print(f"     样本量 n = {len(self.data)}")
        print(f"     样本均值 x̄ = {np.mean(self.data):.4f}")
        print(f"\n  ✨ 后验分布: N({self.posterior_mu:.4f}, {self.posterior_tau:.4f}²)")
        print(f"\n  📈 后验统计:")
        print(f"     后验均值 (点估计): {self.posterior_mu:.4f}")
        print(f"     后验标准差: {self.posterior_tau:.4f}")
        
        ci_lower, ci_upper = self.credible_interval(0.95)
        print(f"     95% 可信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print("="*60)
    
    def credible_interval(self, level=0.95):
        """
        计算后验可信区间
        
        :param level: 可信水平（如0.95表示95%）
        :return: (下界, 上界)
        """
        alpha = 1 - level
        lower = stats.norm.ppf(alpha/2, self.posterior_mu, self.posterior_tau)
        upper = stats.norm.ppf(1-alpha/2, self.posterior_mu, self.posterior_tau)
        return lower, upper
    
    def posterior_pdf(self, theta):
        """后验概率密度"""
        return stats.norm.pdf(theta, self.posterior_mu, self.posterior_tau)
    
    def prior_pdf(self, theta):
        """先验概率密度"""
        return stats.norm.pdf(theta, self.prior_mu, self.prior_tau)
    
    def predict(self, n_samples=1000):
        """
        后验预测分布（预测新观测值）
        
        :param n_samples: 生成样本数
        :return: 预测样本
        """
        # 从后验分布采样μ，然后从N(μ, σ²)采样
        mu_samples = np.random.normal(self.posterior_mu, self.posterior_tau, n_samples)
        predictions = np.random.normal(mu_samples, self.known_sigma)
        return predictions
    
    def update(self, new_data):
        """
        序列贝叶斯更新（新数据到来时）
        
        :param new_data: 新观测数据
        :return: self
        """
        # 将当前后验作为新的先验
        self.prior_mu = self.posterior_mu
        self.prior_tau = self.posterior_tau
        
        # 用新数据更新
        return self.fit(new_data)
    
    def plot_distributions(self, save_path=None):
        """绘制先验、似然、后验分布"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 确定绘图范围
        x_min = min(self.prior_mu - 4*self.prior_tau, 
                   np.mean(self.data) - 4*self.known_sigma/np.sqrt(len(self.data)))
        x_max = max(self.prior_mu + 4*self.prior_tau,
                   np.mean(self.data) + 4*self.known_sigma/np.sqrt(len(self.data)))
        theta = np.linspace(x_min, x_max, 500)
        
        # 1. 先验分布
        prior = self.prior_pdf(theta)
        axes[0].plot(theta, prior, color=PlotStyleConfig.COLORS['primary'], linewidth=2.5)
        axes[0].fill_between(theta, prior, alpha=0.3, color=PlotStyleConfig.COLORS['primary'])
        axes[0].axvline(self.prior_mu, color='gray', linestyle='--', linewidth=1.5)
        axes[0].set_title('先验分布 P(θ)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('θ (参数)', fontweight='bold')
        axes[0].set_ylabel('概率密度', fontweight='bold')
        
        # 2. 似然函数
        n = len(self.data)
        x_bar = np.mean(self.data)
        se = self.known_sigma / np.sqrt(n)
        likelihood = stats.norm.pdf(theta, x_bar, se)
        axes[1].plot(theta, likelihood, color=PlotStyleConfig.COLORS['secondary'], linewidth=2.5)
        axes[1].fill_between(theta, likelihood, alpha=0.3, color=PlotStyleConfig.COLORS['secondary'])
        axes[1].axvline(x_bar, color='gray', linestyle='--', linewidth=1.5)
        axes[1].set_title('似然函数 P(D|θ)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('θ (参数)', fontweight='bold')
        
        # 3. 后验分布
        posterior = self.posterior_pdf(theta)
        axes[2].plot(theta, posterior, color=PlotStyleConfig.COLORS['danger'], linewidth=2.5)
        axes[2].fill_between(theta, posterior, alpha=0.3, color=PlotStyleConfig.COLORS['danger'])
        axes[2].axvline(self.posterior_mu, color='gray', linestyle='--', linewidth=1.5)
        
        # 标记可信区间
        ci_lower, ci_upper = self.credible_interval(0.95)
        mask = (theta >= ci_lower) & (theta <= ci_upper)
        axes[2].fill_between(theta[mask], posterior[mask], alpha=0.5, 
                            color=PlotStyleConfig.COLORS['accent'], label='95% CI')
        
        axes[2].set_title('后验分布 P(θ|D)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('θ (参数)', fontweight='bold')
        axes[2].legend()
        
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.suptitle('贝叶斯推断: 先验 × 似然 → 后验', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, axes
    
    def plot_updating(self, data_sequence, save_path=None):
        """
        可视化序列贝叶斯更新过程
        
        :param data_sequence: 数据序列列表 [[第1批], [第2批], ...]
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 重置到原始先验
        current_mu = self.prior_mu
        current_tau = self.prior_tau
        
        theta = np.linspace(current_mu - 5*current_tau, 
                           current_mu + 5*current_tau, 500)
        
        colors = PlotStyleConfig.get_palette(len(data_sequence) + 1)
        
        # 绘制先验
        prior = stats.norm.pdf(theta, current_mu, current_tau)
        ax.plot(theta, prior, color=colors[0], linewidth=2, 
               linestyle='--', label='先验', alpha=0.7)
        
        # 逐步更新并绘制
        for i, data_batch in enumerate(data_sequence):
            n = len(data_batch)
            x_bar = np.mean(data_batch)
            
            prior_precision = 1 / current_tau**2
            likelihood_precision = n / self.known_sigma**2
            posterior_precision = prior_precision + likelihood_precision
            
            new_tau = 1 / np.sqrt(posterior_precision)
            new_mu = (prior_precision * current_mu + 
                     likelihood_precision * x_bar) / posterior_precision
            
            posterior = stats.norm.pdf(theta, new_mu, new_tau)
            ax.plot(theta, posterior, color=colors[i+1], linewidth=2.5,
                   label=f'批次{i+1}后 (n={n})')
            
            current_mu, current_tau = new_mu, new_tau
        
        ax.set_xlabel('θ (参数)', fontsize=12, fontweight='bold')
        ax.set_ylabel('概率密度', fontsize=12, fontweight='bold')
        ax.set_title('贝叶斯序列更新过程', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax


class BetaBinomialBayes(BayesianInference):
    """
    Beta-二项共轭模型
    
    场景：估计成功概率 p（如点击率、转化率、合格率）
    
    先验：p ~ Beta(α, β)
    似然：X|p ~ Binomial(n, p)
    后验：p|X ~ Beta(α + k, β + n - k)
    
    适用：二值数据的概率估计
    """
    
    def __init__(self, prior_alpha=1, prior_beta=1, verbose=True):
        """
        初始化Beta-二项模型
        
        :param prior_alpha: Beta先验参数 α（可理解为先验成功次数）
        :param prior_beta: Beta先验参数 β（可理解为先验失败次数）
        """
        super().__init__(verbose)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # 后验参数
        self.posterior_alpha = None
        self.posterior_beta = None
        self.n_trials = None
        self.n_success = None
        
    def fit(self, n_success, n_trials):
        """
        根据观测更新后验
        
        :param n_success: 成功次数
        :param n_trials: 总试验次数
        :return: self
        """
        self.n_success = n_success
        self.n_trials = n_trials
        
        # 共轭更新
        self.posterior_alpha = self.prior_alpha + n_success
        self.posterior_beta = self.prior_beta + (n_trials - n_success)
        
        if self.verbose:
            self._print_results()
            
        return self
    
    def _print_results(self):
        """打印结果"""
        self._print_header("Beta-二项 贝叶斯推断")
        print(f"\n  📌 先验分布: Beta({self.prior_alpha}, {self.prior_beta})")
        prior_mean = self.prior_alpha / (self.prior_alpha + self.prior_beta)
        print(f"     先验均值: {prior_mean:.4f}")
        
        print(f"\n  📊 观测数据:")
        print(f"     总试验 n = {self.n_trials}")
        print(f"     成功次数 k = {self.n_success}")
        print(f"     观测比例: {self.n_success/self.n_trials:.4f}")
        
        print(f"\n  ✨ 后验分布: Beta({self.posterior_alpha}, {self.posterior_beta})")
        posterior_mean = self.posterior_alpha / (self.posterior_alpha + self.posterior_beta)
        posterior_mode = (self.posterior_alpha - 1) / (self.posterior_alpha + self.posterior_beta - 2)
        posterior_var = (self.posterior_alpha * self.posterior_beta) / \
                       ((self.posterior_alpha + self.posterior_beta)**2 * 
                        (self.posterior_alpha + self.posterior_beta + 1))
        
        print(f"\n  📈 后验统计:")
        print(f"     后验均值: {posterior_mean:.4f}")
        print(f"     后验众数: {posterior_mode:.4f}")
        print(f"     后验标准差: {np.sqrt(posterior_var):.4f}")
        
        ci_lower, ci_upper = self.credible_interval(0.95)
        print(f"     95% 可信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print("="*60)
    
    def credible_interval(self, level=0.95):
        """计算后验可信区间"""
        alpha_ci = 1 - level
        lower = stats.beta.ppf(alpha_ci/2, self.posterior_alpha, self.posterior_beta)
        upper = stats.beta.ppf(1-alpha_ci/2, self.posterior_alpha, self.posterior_beta)
        return lower, upper
    
    def posterior_pdf(self, p):
        """后验概率密度"""
        return stats.beta.pdf(p, self.posterior_alpha, self.posterior_beta)
    
    def prior_pdf(self, p):
        """先验概率密度"""
        return stats.beta.pdf(p, self.prior_alpha, self.prior_beta)
    
    def probability_greater_than(self, threshold):
        """P(p > threshold | data)"""
        prob = 1 - stats.beta.cdf(threshold, self.posterior_alpha, self.posterior_beta)
        if self.verbose:
            print(f"\n  P(p > {threshold}) = {prob:.4f}")
        return prob
    
    def plot_distributions(self, save_path=None):
        """绘制先验和后验分布"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        p = np.linspace(0.001, 0.999, 500)
        
        # 先验
        prior = self.prior_pdf(p)
        ax.plot(p, prior, color=PlotStyleConfig.COLORS['primary'], 
               linewidth=2.5, linestyle='--', label=f'先验 Beta({self.prior_alpha}, {self.prior_beta})')
        ax.fill_between(p, prior, alpha=0.2, color=PlotStyleConfig.COLORS['primary'])
        
        # 后验
        posterior = self.posterior_pdf(p)
        ax.plot(p, posterior, color=PlotStyleConfig.COLORS['danger'], 
               linewidth=2.5, label=f'后验 Beta({self.posterior_alpha}, {self.posterior_beta})')
        ax.fill_between(p, posterior, alpha=0.3, color=PlotStyleConfig.COLORS['danger'])
        
        # 标记可信区间
        ci_lower, ci_upper = self.credible_interval(0.95)
        ax.axvline(ci_lower, color=PlotStyleConfig.COLORS['accent'], linestyle=':', linewidth=2)
        ax.axvline(ci_upper, color=PlotStyleConfig.COLORS['accent'], linestyle=':', linewidth=2)
        
        # 标记观测比例
        obs_rate = self.n_success / self.n_trials
        ax.axvline(obs_rate, color='gray', linestyle='--', linewidth=1.5,
                  label=f'观测比例 = {obs_rate:.3f}')
        
        ax.set_xlabel('成功概率 p', fontsize=12, fontweight='bold')
        ax.set_ylabel('概率密度', fontsize=12, fontweight='bold')
        ax.set_title('Beta-二项贝叶斯推断', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right')
        ax.set_xlim(0, 1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax


class MCMCBayesian(BayesianInference):
    """
    MCMC贝叶斯推断（Metropolis-Hastings算法）
    
    适用于无共轭先验或复杂后验分布的情况
    """
    
    def __init__(self, log_likelihood_func, log_prior_func, 
                 proposal_std=0.5, n_samples=10000, burn_in=1000, verbose=True):
        """
        初始化MCMC采样器
        
        :param log_likelihood_func: 对数似然函数 f(theta, data) -> log p(data|theta)
        :param log_prior_func: 对数先验函数 f(theta) -> log p(theta)
        :param proposal_std: 提议分布标准差
        :param n_samples: 采样数量
        :param burn_in: 预烧期样本数
        """
        super().__init__(verbose)
        self.log_likelihood = log_likelihood_func
        self.log_prior = log_prior_func
        self.proposal_std = proposal_std
        self.n_samples = n_samples
        self.burn_in = burn_in
        
        self.samples = None
        self.acceptance_rate = None
        
    def log_posterior(self, theta, data):
        """对数后验（非归一化）"""
        return self.log_likelihood(theta, data) + self.log_prior(theta)
    
    def fit(self, data, initial_theta=None):
        """
        使用MCMC采样后验分布
        
        :param data: 观测数据
        :param initial_theta: 初始参数值
        :return: self
        """
        self.data = data
        n_dims = 1 if initial_theta is None or np.isscalar(initial_theta) else len(initial_theta)
        
        if initial_theta is None:
            current = np.zeros(n_dims)
        else:
            current = np.atleast_1d(initial_theta).astype(float)
        
        samples = []
        accepted = 0
        
        current_log_post = self.log_posterior(current, data)
        
        for i in range(self.n_samples + self.burn_in):
            # 提议新状态
            proposal = current + np.random.normal(0, self.proposal_std, n_dims)
            proposal_log_post = self.log_posterior(proposal, data)
            
            # Metropolis-Hastings接受率
            log_alpha = proposal_log_post - current_log_post
            
            if np.log(np.random.random()) < log_alpha:
                current = proposal
                current_log_post = proposal_log_post
                if i >= self.burn_in:
                    accepted += 1
            
            if i >= self.burn_in:
                samples.append(current.copy())
        
        self.samples = np.array(samples)
        self.acceptance_rate = accepted / self.n_samples
        
        if self.verbose:
            self._print_results()
            
        return self
    
    def _print_results(self):
        """打印MCMC结果"""
        self._print_header("MCMC 贝叶斯推断")
        print(f"\n  📌 采样设置:")
        print(f"     总样本数: {self.n_samples}")
        print(f"     预烧期: {self.burn_in}")
        print(f"     接受率: {self.acceptance_rate*100:.1f}%")
        
        print(f"\n  📈 后验统计:")
        if self.samples.ndim == 1 or self.samples.shape[1] == 1:
            samples = self.samples.flatten()
            print(f"     后验均值: {np.mean(samples):.4f}")
            print(f"     后验中位数: {np.median(samples):.4f}")
            print(f"     后验标准差: {np.std(samples):.4f}")
            print(f"     95% CI: [{np.percentile(samples, 2.5):.4f}, {np.percentile(samples, 97.5):.4f}]")
        else:
            for i in range(self.samples.shape[1]):
                print(f"\n     参数 {i+1}:")
                print(f"       均值: {np.mean(self.samples[:, i]):.4f}")
                print(f"       95% CI: [{np.percentile(self.samples[:, i], 2.5):.4f}, "
                      f"{np.percentile(self.samples[:, i], 97.5):.4f}]")
        print("="*60)
    
    def credible_interval(self, param_idx=0, level=0.95):
        """计算可信区间"""
        alpha = 1 - level
        if self.samples.ndim == 1:
            samples = self.samples
        else:
            samples = self.samples[:, param_idx]
        return np.percentile(samples, [alpha/2*100, (1-alpha/2)*100])
    
    def plot_trace(self, param_idx=0, save_path=None):
        """绘制追踪图和后验分布"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if self.samples.ndim == 1:
            samples = self.samples
        else:
            samples = self.samples[:, param_idx]
        
        # 追踪图
        axes[0].plot(samples, color=PlotStyleConfig.COLORS['primary'], 
                    alpha=0.7, linewidth=0.5)
        axes[0].axhline(np.mean(samples), color=PlotStyleConfig.COLORS['danger'],
                       linestyle='--', linewidth=2, label=f'均值 = {np.mean(samples):.3f}')
        axes[0].set_xlabel('迭代次数', fontweight='bold')
        axes[0].set_ylabel('参数值', fontweight='bold')
        axes[0].set_title('MCMC 追踪图', fontsize=12, fontweight='bold')
        axes[0].legend()
        
        # 后验分布
        axes[1].hist(samples, bins=50, density=True, 
                    color=PlotStyleConfig.COLORS['primary'], alpha=0.7, edgecolor='white')
        
        # KDE曲线
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(samples)
        x_range = np.linspace(samples.min(), samples.max(), 200)
        axes[1].plot(x_range, kde(x_range), color=PlotStyleConfig.COLORS['danger'],
                    linewidth=2.5, label='KDE')
        
        # 可信区间
        ci = self.credible_interval(param_idx, 0.95)
        axes[1].axvline(ci[0], color=PlotStyleConfig.COLORS['accent'], 
                       linestyle=':', linewidth=2)
        axes[1].axvline(ci[1], color=PlotStyleConfig.COLORS['accent'], 
                       linestyle=':', linewidth=2, label='95% CI')
        
        axes[1].set_xlabel('参数值', fontweight='bold')
        axes[1].set_ylabel('概率密度', fontweight='bold')
        axes[1].set_title('后验分布', fontsize=12, fontweight='bold')
        axes[1].legend()
        
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, axes


class BayesianParameterEstimation:
    """
    贝叶斯参数反演 - 逆问题求解
    
    场景：给定观测数据和正向模型，反推模型参数
    
    应用：
    - 通过磨损程度推测人流量
    - 通过污染浓度推测污染源强度
    - 通过信号推测系统参数
    """
    
    def __init__(self, forward_model, param_bounds, noise_std=1.0, verbose=True):
        """
        初始化参数反演器
        
        :param forward_model: 正向模型函数 f(params) -> predictions
        :param param_bounds: 参数范围列表 [(low1, high1), (low2, high2), ...]
        :param noise_std: 观测噪声标准差
        """
        self.forward_model = forward_model
        self.param_bounds = param_bounds
        self.noise_std = noise_std
        self.verbose = verbose
        self.n_params = len(param_bounds)
        
        self.samples = None
        self.map_estimate = None
        self.posterior_mean = None
        
    def log_likelihood(self, params, observations):
        """对数似然：假设高斯噪声"""
        try:
            predictions = self.forward_model(params)
            residuals = observations - predictions
            return -0.5 * np.sum((residuals / self.noise_std)**2)
        except:
            return -np.inf
    
    def log_prior(self, params):
        """均匀先验（在边界内）"""
        for i, (low, high) in enumerate(self.param_bounds):
            if params[i] < low or params[i] > high:
                return -np.inf
        return 0.0
    
    def fit(self, observations, n_samples=10000, proposal_stds=None):
        """
        执行贝叶斯参数反演
        
        :param observations: 观测数据
        :param n_samples: MCMC样本数
        :param proposal_stds: 各参数的提议标准差
        """
        if proposal_stds is None:
            proposal_stds = [(b[1]-b[0])/10 for b in self.param_bounds]
        
        # 初始值：参数范围中点
        current = np.array([(b[0]+b[1])/2 for b in self.param_bounds])
        
        samples = []
        burn_in = n_samples // 5
        accepted = 0
        
        current_log_post = self.log_likelihood(current, observations) + self.log_prior(current)
        
        for i in range(n_samples + burn_in):
            # 提议
            proposal = current + np.random.normal(0, proposal_stds)
            proposal_log_post = self.log_likelihood(proposal, observations) + self.log_prior(proposal)
            
            # 接受/拒绝
            log_alpha = proposal_log_post - current_log_post
            if np.log(np.random.random()) < log_alpha:
                current = proposal
                current_log_post = proposal_log_post
                if i >= burn_in:
                    accepted += 1
            
            if i >= burn_in:
                samples.append(current.copy())
        
        self.samples = np.array(samples)
        self.posterior_mean = np.mean(self.samples, axis=0)
        self.map_estimate = self.samples[np.argmax([self.log_likelihood(s, observations) 
                                                     for s in self.samples])]
        
        if self.verbose:
            self._print_results(accepted / n_samples)
            
        return self
    
    def _print_results(self, acceptance_rate):
        """打印反演结果"""
        print("\n" + "="*60)
        print("📊 贝叶斯参数反演结果")
        print("="*60)
        print(f"\n  接受率: {acceptance_rate*100:.1f}%")
        print(f"\n  参数估计:")
        print("  " + "-"*50)
        print(f"  {'参数':^8} {'后验均值':^12} {'MAP估计':^12} {'95% CI':^20}")
        print("  " + "-"*50)
        
        for i in range(self.n_params):
            mean = self.posterior_mean[i]
            map_val = self.map_estimate[i]
            ci = np.percentile(self.samples[:, i], [2.5, 97.5])
            print(f"  θ{i+1:^6} {mean:^12.4f} {map_val:^12.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
        
        print("  " + "-"*50)
        print("="*60)
    
    def plot_corner(self, param_names=None, save_path=None):
        """绘制角图（参数联合分布）"""
        n = self.n_params
        
        if param_names is None:
            param_names = [f'θ{i+1}' for i in range(n)]
        
        fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))
        
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                
                if i == j:
                    # 对角线：边缘分布
                    ax.hist(self.samples[:, i], bins=30, density=True,
                           color=PlotStyleConfig.COLORS['primary'], alpha=0.7)
                    ax.axvline(self.posterior_mean[i], color='red', linestyle='--')
                elif i > j:
                    # 下三角：散点图
                    ax.scatter(self.samples[:, j], self.samples[:, i], 
                              alpha=0.1, s=1, c=PlotStyleConfig.COLORS['primary'])
                    ax.axhline(self.posterior_mean[i], color='red', linestyle='--', alpha=0.5)
                    ax.axvline(self.posterior_mean[j], color='red', linestyle='--', alpha=0.5)
                else:
                    # 上三角：隐藏
                    ax.set_visible(False)
                
                if i == n-1:
                    ax.set_xlabel(param_names[j], fontweight='bold')
                if j == 0 and i != 0:
                    ax.set_ylabel(param_names[i], fontweight='bold')
        
        plt.suptitle('参数后验联合分布', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, axes


if __name__ == "__main__":
    print("="*60)
    print("📊 贝叶斯推断模型演示")
    print("="*60)
    
    # ================== 示例1: 正态-正态推断 ==================
    print("\n" + "="*60)
    print("示例1: 正态-正态贝叶斯推断")
    print("="*60)
    
    np.random.seed(42)
    true_mu = 75
    data = np.random.normal(true_mu, 10, 50)  # 真实均值75
    
    # 先验：认为均值在70左右，但不太确定
    bayes = NormalNormalBayes(prior_mu=70, prior_tau=20, known_sigma=10)
    bayes.fit(data)
    
    fig1, axes1 = bayes.plot_distributions()
    plt.show()
    
    # 序列更新演示
    data_batches = [
        np.random.normal(75, 10, 10),
        np.random.normal(75, 10, 20),
        np.random.normal(75, 10, 30),
    ]
    fig2, ax2 = bayes.plot_updating(data_batches)
    plt.show()
    
    # ================== 示例2: Beta-二项推断 ==================
    print("\n" + "="*60)
    print("示例2: Beta-二项贝叶斯推断（转化率估计）")
    print("="*60)
    
    # 场景：网站A/B测试，100次访问中有23次转化
    beta_bayes = BetaBinomialBayes(prior_alpha=2, prior_beta=8)  # 先验认为转化率约20%
    beta_bayes.fit(n_success=23, n_trials=100)
    
    # 计算概率
    beta_bayes.probability_greater_than(0.20)  # P(转化率>20%)
    
    fig3, ax3 = beta_bayes.plot_distributions()
    plt.show()
    
    # ================== 示例3: 参数反演（逆问题）==================
    print("\n" + "="*60)
    print("示例3: 贝叶斯参数反演（由果推因）")
    print("="*60)
    
    # 场景：磨损模型 wear = k * flow * time
    # 已知时间和磨损量，反推人流量和磨损系数
    
    def wear_model(params):
        """正向模型：磨损量 = k * flow * time"""
        k, flow = params
        time = np.array([1, 2, 3, 4, 5])  # 5年观测
        return k * flow * time
    
    # 生成模拟观测数据（真实参数: k=0.01, flow=1000）
    true_params = [0.01, 1000]
    true_wear = wear_model(true_params)
    observed_wear = true_wear + np.random.normal(0, 0.5, len(true_wear))
    
    print(f"观测磨损量: {observed_wear}")
    print(f"真实参数: k={true_params[0]}, flow={true_params[1]}")
    
    # 贝叶斯反演
    inverter = BayesianParameterEstimation(
        forward_model=wear_model,
        param_bounds=[(0.001, 0.1), (100, 5000)],  # k和flow的范围
        noise_std=0.5
    )
    inverter.fit(observed_wear, n_samples=5000)
    
    fig4, axes4 = inverter.plot_corner(param_names=['磨损系数k', '人流量flow'])
    plt.show()
    
    print("\n✅ 贝叶斯推断模型演示完成!")
