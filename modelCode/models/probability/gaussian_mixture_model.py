"""
============================================================
高斯混合模型 (Gaussian Mixture Model, GMM)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：聚类分析、密度估计、异常检测、软分类
原理：假设数据由多个高斯分布混合生成，使用EM算法估计参数
作者：MCM/ICM Team
日期：2026年1月
============================================================

应用场景：
- 客户分群（软聚类）
- 图像分割
- 语音识别
- 异常检测（低概率区域）
- 数据密度建模

数学模型：
p(x) = Σ π_k * N(x | μ_k, Σ_k)
其中 π_k 是混合权重，满足 Σπ_k = 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from visualization.plot_config import PlotStyleConfig, FigureSaver

PlotStyleConfig.setup_style()
warnings.filterwarnings('ignore')


class GMMClustering:
    """
    高斯混合模型聚类分析类
    
    核心功能：
    1. 自动确定最佳聚类数（BIC/AIC）
    2. EM算法参数估计
    3. 软聚类（概率分配）
    4. 异常检测
    5. 丰富的可视化
    """
    
    def __init__(self, n_components='auto', covariance_type='full', 
                 max_components=10, random_state=42, verbose=True):
        """
        初始化GMM模型
        
        :param n_components: 聚类数（'auto'自动选择）
        :param covariance_type: 协方差类型 
                               'full'(完全), 'tied'(共享), 
                               'diag'(对角), 'spherical'(球形)
        :param max_components: 自动选择时的最大聚类数
        :param random_state: 随机种子
        :param verbose: 是否打印详细信息
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_components = max_components
        self.random_state = random_state
        self.verbose = verbose
        
        self.model = None
        self.scaler = None
        self.X_scaled = None
        self.labels_ = None
        self.probabilities_ = None
        self.bic_scores = None
        self.aic_scores = None
        self.optimal_k = None
        self.feature_names = None
        
    def fit(self, X, scale=True):
        """
        拟合GMM模型
        
        :param X: 特征数据（DataFrame或数组）
        :param scale: 是否标准化数据
        :return: self
        """
        # 处理输入
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = np.array(X)
            self.feature_names = [f'特征{i+1}' for i in range(X_array.shape[1])]
        
        # 标准化
        if scale:
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(X_array)
        else:
            self.X_scaled = X_array
        
        # 自动选择聚类数
        if self.n_components == 'auto':
            self._find_optimal_k()
            self.n_components = self.optimal_k
        
        # 拟合模型
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=5
        )
        self.model.fit(self.X_scaled)
        
        # 预测
        self.labels_ = self.model.predict(self.X_scaled)
        self.probabilities_ = self.model.predict_proba(self.X_scaled)
        
        if self.verbose:
            self._print_results()
        
        return self
    
    def _find_optimal_k(self):
        """使用BIC/AIC准则寻找最优聚类数"""
        self.bic_scores = []
        self.aic_scores = []
        k_range = range(1, self.max_components + 1)
        
        for k in k_range:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                n_init=3
            )
            gmm.fit(self.X_scaled)
            self.bic_scores.append(gmm.bic(self.X_scaled))
            self.aic_scores.append(gmm.aic(self.X_scaled))
        
        # 使用BIC选择（BIC惩罚复杂模型）
        self.optimal_k = k_range[np.argmin(self.bic_scores)]
        
        if self.verbose:
            print(f"\n  🔍 自动选择聚类数: k = {self.optimal_k} (基于BIC)")
    
    def _print_results(self):
        """打印拟合结果"""
        print("\n" + "="*60)
        print("📊 高斯混合模型 (GMM) 聚类结果")
        print("="*60)
        print(f"\n  样本量: {len(self.X_scaled)}")
        print(f"  特征数: {self.X_scaled.shape[1]}")
        print(f"  聚类数: {self.n_components}")
        print(f"  协方差类型: {self.covariance_type}")
        
        print(f"\n  各簇统计:")
        print("  " + "-"*50)
        print(f"  {'簇':^6} {'样本数':^10} {'占比':^10} {'混合权重':^12}")
        print("  " + "-"*50)
        
        for k in range(self.n_components):
            n_k = np.sum(self.labels_ == k)
            pct = n_k / len(self.labels_) * 100
            weight = self.model.weights_[k]
            print(f"  {k:^6} {n_k:^10} {pct:^9.1f}% {weight:^12.4f}")
        
        print("  " + "-"*50)
        print(f"\n  模型评估:")
        print(f"    对数似然: {self.model.score(self.X_scaled):.4f}")
        print(f"    BIC: {self.model.bic(self.X_scaled):.2f}")
        print(f"    AIC: {self.model.aic(self.X_scaled):.2f}")
        print(f"    收敛: {'是' if self.model.converged_ else '否'}")
        print("="*60)
    
    def predict(self, X_new):
        """预测新数据的簇标签"""
        if self.scaler:
            X_new_scaled = self.scaler.transform(X_new)
        else:
            X_new_scaled = X_new
        return self.model.predict(X_new_scaled)
    
    def predict_proba(self, X_new):
        """预测新数据属于各簇的概率"""
        if self.scaler:
            X_new_scaled = self.scaler.transform(X_new)
        else:
            X_new_scaled = X_new
        return self.model.predict_proba(X_new_scaled)
    
    def get_cluster_summary(self):
        """
        获取各簇的统计摘要
        
        :return: DataFrame，包含各簇的均值等统计信息
        """
        summary = []
        
        # 获取原始尺度的均值
        if self.scaler:
            means = self.scaler.inverse_transform(self.model.means_)
        else:
            means = self.model.means_
        
        for k in range(self.n_components):
            cluster_info = {'簇': k, '样本数': np.sum(self.labels_ == k)}
            cluster_info['权重'] = self.model.weights_[k]
            
            for i, name in enumerate(self.feature_names):
                cluster_info[f'{name}_均值'] = means[k, i]
            
            summary.append(cluster_info)
        
        return pd.DataFrame(summary)
    
    def detect_anomalies(self, threshold=0.01):
        """
        基于密度的异常检测
        
        :param threshold: 概率阈值（低于此值视为异常）
        :return: 异常样本的索引
        """
        # 计算每个样本的对数似然
        log_prob = self.model.score_samples(self.X_scaled)
        
        # 转换为概率密度
        prob = np.exp(log_prob)
        
        # 使用分位数确定阈值
        cutoff = np.percentile(prob, threshold * 100)
        anomalies = np.where(prob < cutoff)[0]
        
        if self.verbose:
            print(f"\n  🔍 异常检测: 发现 {len(anomalies)} 个异常点 (阈值: {threshold*100:.1f}%)")
        
        return anomalies, prob
    
    def sample(self, n_samples=100):
        """从拟合的GMM生成新样本"""
        samples, labels = self.model.sample(n_samples)
        if self.scaler:
            samples = self.scaler.inverse_transform(samples)
        return samples, labels
    
    # ==================== 可视化方法 ====================
    
    def plot_bic_aic(self, save_path=None):
        """绘制BIC/AIC曲线（用于选择聚类数）"""
        if self.bic_scores is None:
            print("需要先使用 n_components='auto' 拟合模型")
            return None, None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        k_range = range(1, len(self.bic_scores) + 1)
        
        ax.plot(k_range, self.bic_scores, 'o-', color=PlotStyleConfig.COLORS['primary'],
               linewidth=2.5, markersize=8, label='BIC')
        ax.plot(k_range, self.aic_scores, 's--', color=PlotStyleConfig.COLORS['secondary'],
               linewidth=2.5, markersize=8, label='AIC')
        
        # 标记最优点
        ax.axvline(self.optimal_k, color=PlotStyleConfig.COLORS['accent'],
                  linestyle=':', linewidth=2, label=f'最优 k={self.optimal_k}')
        ax.scatter([self.optimal_k], [self.bic_scores[self.optimal_k-1]], 
                  s=200, color=PlotStyleConfig.COLORS['danger'], zorder=5, marker='*')
        
        ax.set_xlabel('聚类数 k', fontsize=12, fontweight='bold')
        ax.set_ylabel('信息准则值', fontsize=12, fontweight='bold')
        ax.set_title('GMM 聚类数选择 (BIC/AIC)', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right')
        ax.set_xticks(k_range)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax
    
    def plot_clusters_2d(self, feature_indices=(0, 1), show_ellipse=True, save_path=None):
        """
        2D聚类可视化
        
        :param feature_indices: 显示的两个特征索引
        :param show_ellipse: 是否显示协方差椭圆
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        i, j = feature_indices
        X_plot = self.X_scaled[:, [i, j]]
        
        colors = PlotStyleConfig.get_palette(self.n_components)
        
        # 绘制散点
        for k in range(self.n_components):
            mask = self.labels_ == k
            ax.scatter(X_plot[mask, 0], X_plot[mask, 1], 
                      c=colors[k], label=f'簇 {k}', s=50, alpha=0.6, edgecolors='white')
        
        # 绘制协方差椭圆
        if show_ellipse:
            for k in range(self.n_components):
                mean = self.model.means_[k, [i, j]]
                
                if self.covariance_type == 'full':
                    cov = self.model.covariances_[k][[i, j], :][:, [i, j]]
                elif self.covariance_type == 'tied':
                    cov = self.model.covariances_[[i, j], :][:, [i, j]]
                elif self.covariance_type == 'diag':
                    cov = np.diag(self.model.covariances_[k, [i, j]])
                else:  # spherical
                    cov = np.eye(2) * self.model.covariances_[k]
                
                self._draw_ellipse(ax, mean, cov, colors[k])
        
        # 绘制簇中心
        centers = self.model.means_[:, [i, j]]
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, 
                  marker='X', edgecolors='white', linewidth=2, label='簇中心', zorder=5)
        
        ax.set_xlabel(f'{self.feature_names[i]} (标准化)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{self.feature_names[j]} (标准化)', fontsize=12, fontweight='bold')
        ax.set_title('GMM 聚类结果 (2D)', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax
    
    def _draw_ellipse(self, ax, mean, cov, color, n_std=2):
        """绘制协方差椭圆"""
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                         facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(ellipse)
    
    def plot_probability_heatmap(self, save_path=None):
        """绘制样本归属概率热力图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 选取部分样本（太多则采样）
        n_show = min(100, len(self.probabilities_))
        indices = np.random.choice(len(self.probabilities_), n_show, replace=False)
        probs = self.probabilities_[sorted(indices)]
        
        im = ax.imshow(probs, aspect='auto', cmap='YlOrRd')
        
        ax.set_xlabel('簇', fontsize=12, fontweight='bold')
        ax.set_ylabel('样本', fontsize=12, fontweight='bold')
        ax.set_title('样本归属概率分布 (软聚类)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(range(self.n_components))
        ax.set_xticklabels([f'簇{k}' for k in range(self.n_components)])
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('归属概率', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax
    
    def plot_density_contour(self, feature_indices=(0, 1), n_points=100, save_path=None):
        """绘制GMM密度等高线"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        i, j = feature_indices
        X_plot = self.X_scaled[:, [i, j]]
        
        # 创建网格
        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points),
                            np.linspace(y_min, y_max, n_points))
        
        # 构建完整特征空间的网格点
        grid_points = np.zeros((n_points * n_points, self.X_scaled.shape[1]))
        grid_points[:, i] = xx.ravel()
        grid_points[:, j] = yy.ravel()
        # 其他特征用均值填充
        for k in range(self.X_scaled.shape[1]):
            if k not in [i, j]:
                grid_points[:, k] = self.X_scaled[:, k].mean()
        
        # 计算密度
        Z = np.exp(self.model.score_samples(grid_points))
        Z = Z.reshape(xx.shape)
        
        # 绘制等高线
        contour = ax.contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.7)
        ax.contour(xx, yy, Z, levels=10, colors='white', alpha=0.5, linewidths=0.5)
        
        # 绘制数据点
        ax.scatter(X_plot[:, 0], X_plot[:, 1], c='white', s=20, alpha=0.5, edgecolors='black')
        
        # 绘制簇中心
        centers = self.model.means_[:, [i, j]]
        ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, 
                  marker='X', edgecolors='white', linewidth=2, zorder=5)
        
        ax.set_xlabel(f'{self.feature_names[i]}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{self.feature_names[j]}', fontsize=12, fontweight='bold')
        ax.set_title('GMM 密度估计等高线', fontsize=14, fontweight='bold', pad=15)
        
        cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label('概率密度', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax


def generate_gmm_sample_data(n_samples=500, n_clusters=3, n_features=2, random_state=42):
    """
    生成GMM测试数据
    
    :return: X数据, 真实标签
    """
    np.random.seed(random_state)
    
    samples_per_cluster = n_samples // n_clusters
    X = []
    y = []
    
    # 随机生成聚类中心
    centers = np.random.randn(n_clusters, n_features) * 5
    
    for k in range(n_clusters):
        # 随机协方差
        A = np.random.randn(n_features, n_features)
        cov = A @ A.T / n_features + np.eye(n_features) * 0.5
        
        samples = np.random.multivariate_normal(centers[k], cov, samples_per_cluster)
        X.append(samples)
        y.extend([k] * samples_per_cluster)
    
    return np.vstack(X), np.array(y)


if __name__ == "__main__":
    print("="*60)
    print("📊 高斯混合模型 (GMM) 演示")
    print("="*60)
    
    # 1. 生成测试数据
    X, y_true = generate_gmm_sample_data(n_samples=500, n_clusters=3, n_features=4)
    feature_names = ['特征A', '特征B', '特征C', '特征D']
    df = pd.DataFrame(X, columns=feature_names)
    
    print(f"\n数据形状: {X.shape}")
    print(f"真实聚类数: {len(np.unique(y_true))}")
    
    # 2. 自动选择聚类数并拟合
    gmm = GMMClustering(n_components='auto', max_components=8)
    gmm.fit(df)
    
    # 3. 获取聚类摘要
    summary = gmm.get_cluster_summary()
    print("\n簇统计摘要:")
    print(summary)
    
    # 4. 异常检测
    anomalies, probs = gmm.detect_anomalies(threshold=0.02)
    
    # 5. 可视化
    fig1, ax1 = gmm.plot_bic_aic()
    plt.show()
    
    fig2, ax2 = gmm.plot_clusters_2d(feature_indices=(0, 1))
    plt.show()
    
    fig3, ax3 = gmm.plot_probability_heatmap()
    plt.show()
    
    fig4, ax4 = gmm.plot_density_contour(feature_indices=(0, 1))
    plt.show()
    
    # 6. 生成新样本
    new_samples, new_labels = gmm.sample(20)
    print(f"\n生成 {len(new_samples)} 个新样本")
    
    print("\n✅ GMM 演示完成!")
