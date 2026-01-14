"""
============================================================
高斯分布（正态分布）模型 (Gaussian Distribution Model)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：概率分布分析、参数估计、假设检验、置信区间
原理：正态分布 N(μ, σ²) 描述连续随机变量
作者：MCM/ICM Team
日期：2026年1月
============================================================

应用场景：
- 测量误差分析
- 质量控制（6σ分析）
- 金融收益率建模
- 自然现象统计分析
- 不确定性量化

数学基础：
概率密度函数: f(x) = (1/(σ√(2π))) * exp(-(x-μ)²/(2σ²))
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
from visualization.plot_config import PlotStyleConfig, FigureSaver, PlotTemplates

PlotStyleConfig.setup_style()
warnings.filterwarnings('ignore')


class GaussianDistribution:
    """
    高斯分布（正态分布）分析类
    
    核心功能：
    1. 参数估计（MLE最大似然估计）
    2. 概率密度函数 (PDF) 与累积分布函数 (CDF)
    3. 置信区间计算
    4. 正态性检验
    5. 分布可视化
    """
    
    def __init__(self, mu=None, sigma=None, verbose=True):
        """
        初始化高斯分布
        
        :param mu: 均值（None则从数据估计）
        :param sigma: 标准差（None则从数据估计）
        :param verbose: 是否打印详细信息
        """
        self.mu = mu
        self.sigma = sigma
        self.verbose = verbose
        self.data = None
        self.fitted = False
        self.normality_test = None
        
    def fit(self, data):
        """
        从数据拟合分布参数（最大似然估计）
        
        :param data: 观测数据（数组或Series）
        :return: self
        """
        self.data = np.array(data).flatten()
        
        # MLE估计
        self.mu = np.mean(self.data)
        self.sigma = np.std(self.data, ddof=1)  # 无偏估计
        
        self.fitted = True
        
        if self.verbose:
            self._print_fit_results()
            
        return self
    
    def _print_fit_results(self):
        """打印拟合结果"""
        print("\n" + "="*55)
        print("📊 高斯分布参数估计 (MLE)")
        print("="*55)
        print(f"\n  样本量: n = {len(self.data)}")
        print(f"\n  估计参数:")
        print(f"    均值 μ = {self.mu:.4f}")
        print(f"    标准差 σ = {self.sigma:.4f}")
        print(f"    方差 σ² = {self.sigma**2:.4f}")
        print(f"\n  样本统计:")
        print(f"    最小值: {self.data.min():.4f}")
        print(f"    最大值: {self.data.max():.4f}")
        print(f"    中位数: {np.median(self.data):.4f}")
        print("="*55)
    
    def pdf(self, x):
        """
        概率密度函数
        
        :param x: 自变量值（标量或数组）
        :return: 概率密度
        """
        return stats.norm.pdf(x, loc=self.mu, scale=self.sigma)
    
    def cdf(self, x):
        """
        累积分布函数 P(X ≤ x)
        
        :param x: 自变量值
        :return: 累积概率
        """
        return stats.norm.cdf(x, loc=self.mu, scale=self.sigma)
    
    def ppf(self, q):
        """
        分位点函数（CDF的逆函数）
        
        :param q: 概率值 (0-1)
        :return: 对应的分位点
        """
        return stats.norm.ppf(q, loc=self.mu, scale=self.sigma)
    
    def probability_range(self, a, b):
        """
        计算 P(a ≤ X ≤ b)
        
        :param a: 下界
        :param b: 上界
        :return: 概率
        """
        prob = self.cdf(b) - self.cdf(a)
        
        if self.verbose:
            print(f"\n  P({a:.2f} ≤ X ≤ {b:.2f}) = {prob:.4f} ({prob*100:.2f}%)")
            
        return prob
    
    def confidence_interval(self, confidence=0.95):
        """
        计算均值的置信区间
        
        :param confidence: 置信水平（默认0.95）
        :return: (下界, 上界)
        """
        if not self.fitted:
            raise ValueError("请先调用 fit() 拟合数据")
            
        n = len(self.data)
        se = self.sigma / np.sqrt(n)  # 标准误
        
        # t分布临界值
        alpha = 1 - confidence
        t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
        
        lower = self.mu - t_crit * se
        upper = self.mu + t_crit * se
        
        if self.verbose:
            print(f"\n  {confidence*100:.0f}% 置信区间: [{lower:.4f}, {upper:.4f}]")
            
        return (lower, upper)
    
    def predict_interval(self, confidence=0.95):
        """
        计算预测区间（单个新观测值的区间）
        
        :param confidence: 置信水平
        :return: (下界, 上界)
        """
        alpha = 1 - confidence
        z = stats.norm.ppf(1 - alpha/2)
        
        lower = self.mu - z * self.sigma
        upper = self.mu + z * self.sigma
        
        if self.verbose:
            print(f"\n  {confidence*100:.0f}% 预测区间: [{lower:.4f}, {upper:.4f}]")
            
        return (lower, upper)
    
    def normality_test(self, method='shapiro'):
        """
        正态性检验
        
        :param method: 'shapiro' / 'ks' / 'anderson' / 'all'
        :return: 检验结果字典
        """
        if not self.fitted:
            raise ValueError("请先调用 fit() 拟合数据")
            
        results = {}
        
        if method in ['shapiro', 'all']:
            stat, p = stats.shapiro(self.data[:5000])  # Shapiro限制5000样本
            results['shapiro'] = {'statistic': stat, 'p_value': p, 
                                 'normal': p > 0.05}
        
        if method in ['ks', 'all']:
            # Kolmogorov-Smirnov检验
            stat, p = stats.kstest(self.data, 'norm', args=(self.mu, self.sigma))
            results['ks'] = {'statistic': stat, 'p_value': p,
                           'normal': p > 0.05}
        
        if method in ['anderson', 'all']:
            # Anderson-Darling检验
            result = stats.anderson(self.data, dist='norm')
            results['anderson'] = {
                'statistic': result.statistic,
                'critical_values': dict(zip(result.significance_level, result.critical_values)),
                'normal': result.statistic < result.critical_values[2]  # 5%显著性
            }
        
        if method in ['dagostino', 'all'] and len(self.data) >= 20:
            # D'Agostino K² 检验
            stat, p = stats.normaltest(self.data)
            results['dagostino'] = {'statistic': stat, 'p_value': p,
                                   'normal': p > 0.05}
        
        self.normality_test_results = results
        
        if self.verbose:
            print("\n" + "="*55)
            print("🔬 正态性检验结果")
            print("="*55)
            for test_name, result in results.items():
                status = "✅ 符合正态" if result.get('normal', False) else "❌ 不符合"
                print(f"\n  {test_name.upper()}检验:")
                print(f"    统计量: {result['statistic']:.4f}")
                if 'p_value' in result:
                    print(f"    p值: {result['p_value']:.4f}")
                print(f"    结论: {status}")
            print("="*55)
            
        return results
    
    def sample(self, n=100):
        """
        生成随机样本
        
        :param n: 样本量
        :return: 随机样本数组
        """
        return np.random.normal(self.mu, self.sigma, n)
    
    def zscore(self, x):
        """
        计算Z分数（标准化）
        
        :param x: 原始值
        :return: Z分数
        """
        return (x - self.mu) / self.sigma
    
    def six_sigma_analysis(self):
        """
        六西格玛质量控制分析
        
        :return: 各sigma范围的概率
        """
        ranges = {}
        for k in range(1, 7):
            prob = self.probability_range(self.mu - k*self.sigma, self.mu + k*self.sigma)
            ranges[f'{k}σ'] = {
                'range': (self.mu - k*self.sigma, self.mu + k*self.sigma),
                'probability': prob,
                'defects_per_million': (1 - prob) * 1e6
            }
        
        if self.verbose:
            print("\n" + "="*55)
            print("📏 六西格玛分析 (6σ Quality Control)")
            print("="*55)
            print(f"\n  μ = {self.mu:.4f}, σ = {self.sigma:.4f}")
            print("\n  范围        概率          百万缺陷数")
            print("  " + "-"*45)
            for name, info in ranges.items():
                print(f"  {name:6s}    {info['probability']*100:7.4f}%    {info['defects_per_million']:12.2f}")
            print("="*55)
            
        return ranges
    
    # ==================== 可视化方法 ====================
    
    def plot_distribution(self, show_data=True, n_std=4, save_path=None):
        """
        绘制概率分布图
        
        :param show_data: 是否显示原始数据直方图
        :param n_std: 显示几个标准差范围
        :param save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x = np.linspace(self.mu - n_std*self.sigma, self.mu + n_std*self.sigma, 500)
        y = self.pdf(x)
        
        # 绘制PDF曲线
        ax.plot(x, y, color=PlotStyleConfig.COLORS['danger'], linewidth=3, 
               label=f'N({self.mu:.2f}, {self.sigma:.2f}²)')
        
        # 填充区域
        ax.fill_between(x, y, alpha=0.3, color=PlotStyleConfig.COLORS['danger'])
        
        # 绘制数据直方图
        if show_data and self.data is not None:
            ax.hist(self.data, bins=30, density=True, alpha=0.5, 
                   color=PlotStyleConfig.COLORS['primary'], edgecolor='white',
                   label='观测数据', linewidth=1.5)
        
        # 标记均值和标准差
        ax.axvline(self.mu, color=PlotStyleConfig.COLORS['neutral'], 
                  linestyle='--', linewidth=2, label=f'μ = {self.mu:.2f}')
        
        for k in [1, 2, 3]:
            ax.axvline(self.mu + k*self.sigma, color=PlotStyleConfig.COLORS['accent'], 
                      linestyle=':', alpha=0.7)
            ax.axvline(self.mu - k*self.sigma, color=PlotStyleConfig.COLORS['accent'], 
                      linestyle=':', alpha=0.7)
        
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('概率密度', fontsize=12, fontweight='bold')
        ax.set_title('高斯分布 (正态分布)', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=10)
        
        # 添加统计信息文本框
        textstr = f'μ = {self.mu:.3f}\nσ = {self.sigma:.3f}\nσ² = {self.sigma**2:.3f}'
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
    
    def plot_cdf(self, save_path=None):
        """绘制累积分布函数"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.linspace(self.mu - 4*self.sigma, self.mu + 4*self.sigma, 500)
        y = self.cdf(x)
        
        ax.plot(x, y, color=PlotStyleConfig.COLORS['primary'], linewidth=3)
        ax.fill_between(x, y, alpha=0.2, color=PlotStyleConfig.COLORS['primary'])
        
        # 标记关键分位点
        for q in [0.025, 0.25, 0.5, 0.75, 0.975]:
            xq = self.ppf(q)
            ax.axhline(q, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(xq, color='gray', linestyle=':', alpha=0.5)
            ax.plot(xq, q, 'o', color=PlotStyleConfig.COLORS['danger'], markersize=8)
            ax.annotate(f'{q*100:.1f}%', (xq, q), textcoords="offset points",
                       xytext=(10, 5), fontsize=9)
        
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('累积概率 P(X ≤ x)', fontsize=12, fontweight='bold')
        ax.set_title('累积分布函数 (CDF)', fontsize=14, fontweight='bold', pad=15)
        
        ax.set_ylim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax
    
    def plot_qq(self, save_path=None):
        """绘制Q-Q图（正态性检验可视化）"""
        if not self.fitted:
            raise ValueError("请先调用 fit() 拟合数据")
            
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 计算理论分位数和样本分位数
        (osm, osr), (slope, intercept, r) = stats.probplot(self.data, dist="norm")
        
        ax.scatter(osm, osr, c=PlotStyleConfig.COLORS['primary'], 
                  alpha=0.6, s=50, edgecolors='white')
        
        # 拟合线
        line_x = np.array([osm.min(), osm.max()])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, 'r--', linewidth=2, 
               label=f'拟合线 (R² = {r**2:.4f})')
        
        ax.set_xlabel('理论分位数', fontsize=12, fontweight='bold')
        ax.set_ylabel('样本分位数', fontsize=12, fontweight='bold')
        ax.set_title('Q-Q 图 (正态性检验)', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper left')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 添加结论
        is_normal = r**2 > 0.95
        conclusion = "数据基本符合正态分布" if is_normal else "数据可能不符合正态分布"
        props = dict(boxstyle='round', facecolor='lightgreen' if is_normal else 'lightyellow', alpha=0.8)
        ax.text(0.98, 0.02, conclusion, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax
    
    def plot_sigma_ranges(self, save_path=None):
        """绘制σ范围概率图"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x = np.linspace(self.mu - 4*self.sigma, self.mu + 4*self.sigma, 500)
        y = self.pdf(x)
        
        # 底层曲线
        ax.plot(x, y, color='black', linewidth=2)
        
        # 填充不同σ区域
        colors = ['#27AE60', '#F18F01', '#E74C3C', '#9B59B6']
        labels = ['±1σ (68.27%)', '±2σ (95.45%)', '±3σ (99.73%)', '±4σ (99.99%)']
        
        for k in range(4, 0, -1):
            mask = (x >= self.mu - k*self.sigma) & (x <= self.mu + k*self.sigma)
            ax.fill_between(x[mask], y[mask], alpha=0.4, color=colors[k-1], label=labels[k-1])
        
        ax.axvline(self.mu, color='black', linestyle='--', linewidth=2)
        
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('概率密度', fontsize=12, fontweight='bold')
        ax.set_title('正态分布 σ 范围概率', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax


class MultiGaussianAnalyzer:
    """
    多组高斯分布比较分析
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.distributions = {}
        
    def add_group(self, name, data):
        """添加数据组"""
        gauss = GaussianDistribution(verbose=False)
        gauss.fit(data)
        self.distributions[name] = gauss
        
    def compare(self):
        """比较各组分布"""
        results = []
        for name, dist in self.distributions.items():
            results.append({
                'name': name,
                'n': len(dist.data),
                'mean': dist.mu,
                'std': dist.sigma,
                'var': dist.sigma**2,
                'min': dist.data.min(),
                'max': dist.data.max()
            })
        
        df = pd.DataFrame(results)
        
        if self.verbose:
            print("\n" + "="*70)
            print("📊 多组高斯分布比较")
            print("="*70)
            print(df.to_string(index=False))
            print("="*70)
            
        return df
    
    def plot_comparison(self, save_path=None):
        """绘制多分布对比图"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = PlotStyleConfig.get_palette(len(self.distributions))
        
        all_data = np.concatenate([d.data for d in self.distributions.values()])
        x_min, x_max = all_data.min(), all_data.max()
        padding = (x_max - x_min) * 0.2
        x = np.linspace(x_min - padding, x_max + padding, 500)
        
        for (name, dist), color in zip(self.distributions.items(), colors):
            y = dist.pdf(x)
            ax.plot(x, y, color=color, linewidth=2.5, label=f'{name} (μ={dist.mu:.2f}, σ={dist.sigma:.2f})')
            ax.fill_between(x, y, alpha=0.2, color=color)
        
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('概率密度', fontsize=12, fontweight='bold')
        ax.set_title('多组高斯分布对比', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax


if __name__ == "__main__":
    print("="*60)
    print("📊 高斯分布模型演示")
    print("="*60)
    
    # 1. 生成模拟数据
    np.random.seed(42)
    data = np.random.normal(loc=100, scale=15, size=500)
    
    # 2. 拟合分布
    gauss = GaussianDistribution()
    gauss.fit(data)
    
    # 3. 概率计算
    print("\n" + "-"*40)
    print("概率计算示例:")
    gauss.probability_range(85, 115)
    gauss.confidence_interval(0.95)
    gauss.predict_interval(0.95)
    
    # 4. 正态性检验
    gauss.normality_test(method='all')
    
    # 5. 六西格玛分析
    gauss.six_sigma_analysis()
    
    # 6. 可视化
    fig1, ax1 = gauss.plot_distribution()
    plt.show()
    
    fig2, ax2 = gauss.plot_cdf()
    plt.show()
    
    fig3, ax3 = gauss.plot_qq()
    plt.show()
    
    fig4, ax4 = gauss.plot_sigma_ranges()
    plt.show()
    
    # 7. 多组比较
    print("\n" + "-"*40)
    print("多组分布比较:")
    analyzer = MultiGaussianAnalyzer()
    analyzer.add_group('组A', np.random.normal(50, 10, 300))
    analyzer.add_group('组B', np.random.normal(60, 8, 300))
    analyzer.add_group('组C', np.random.normal(55, 15, 300))
    analyzer.compare()
    
    fig5, ax5 = analyzer.plot_comparison()
    plt.show()
    
    print("\n✅ 高斯分布模型演示完成!")
