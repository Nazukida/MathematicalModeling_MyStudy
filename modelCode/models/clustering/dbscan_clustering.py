"""
============================================================
DBSCAN 密度聚类分析 (Density-Based Spatial Clustering)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：基于密度的聚类、自动确定簇数、异常点检测
优势：无需指定簇数、可识别任意形状簇、自动检测噪声
作者：MCM/ICM Team
日期：2026年1月
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from visualization.plot_config import PlotStyleConfig, FigureSaver

PlotStyleConfig.setup_style()


class DBSCANClusterer:
    """
    DBSCAN密度聚类封装类
    
    核心原理：
    1. 定义ε-邻域和最小点数MinPts
    2. 识别核心点、边界点、噪声点
    3. 从核心点出发扩展聚类
    4. 未被访问的点标记为噪声
    
    应用场景：
    - 地理空间聚类（如热点区域识别）
    - 异常检测（噪声点即为异常）
    - 不规则形状簇的发现
    - 客户分群、图像分割
    
    优势：
    - 无需预先指定簇数
    - 可发现任意形状的簇
    - 对噪声具有鲁棒性
    - 自动标记异常点
    """
    
    def __init__(self, eps=None, min_samples=None, auto_tune=True, verbose=True):
        """
        参数配置
        
        :param eps: 邻域半径（None则自动选择）
        :param min_samples: 最小样本数（None则自动选择）
        :param auto_tune: 是否自动调参
        :param verbose: 是否打印过程
        """
        self.eps = eps
        self.min_samples = min_samples
        self.auto_tune = auto_tune
        self.verbose = verbose
        
        self.scaler = StandardScaler()
        self.model = None
        self.labels_ = None
        self.n_clusters_ = None
        self.n_noise_ = None
        self.core_sample_indices_ = None
        self.metrics_ = {}
        
    def _find_optimal_eps(self, X, k=None):
        """
        使用K-距离图法找最优eps
        
        原理：计算每个点到第k个最近邻的距离，
        排序后绘图，肘点位置对应最佳eps
        """
        if k is None:
            k = max(2 * X.shape[1], 5)  # 2*维度或至少5
        
        # 计算k近邻距离
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, _ = nbrs.kneighbors(X)
        k_distances = distances[:, -1]  # 第k个近邻的距离
        
        # 排序
        k_distances_sorted = np.sort(k_distances)
        
        # 使用二阶差分找肘点
        diff1 = np.diff(k_distances_sorted)
        diff2 = np.diff(diff1)
        
        # 找到曲率最大的点
        if len(diff2) > 0:
            elbow_idx = np.argmax(diff2) + 2
            optimal_eps = k_distances_sorted[elbow_idx]
        else:
            # 使用中位数作为备选
            optimal_eps = np.median(k_distances)
        
        return optimal_eps, k_distances_sorted
    
    def _find_optimal_min_samples(self, X):
        """
        根据数据维度和大小选择min_samples
        
        经验规则：
        - min_samples >= 维度 + 1
        - 一般取 2 * 维度
        - 数据量大时可适当增加
        """
        n_samples, n_features = X.shape
        
        # 基础值：2 * 维度
        base_min_samples = 2 * n_features
        
        # 根据数据量调整
        if n_samples > 10000:
            min_samples = max(base_min_samples, int(np.log(n_samples)))
        elif n_samples > 1000:
            min_samples = max(base_min_samples, 5)
        else:
            min_samples = max(base_min_samples, 3)
        
        return min_samples
    
    def fit_predict(self, X, standardize=True):
        """
        拟合并预测聚类标签
        
        :param X: 输入数据（DataFrame或数组）
        :param standardize: 是否标准化
        :return: 聚类标签（-1表示噪声）
        """
        # 转换为数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X.copy()
        
        # 标准化
        if standardize:
            X_scaled = self.scaler.fit_transform(X_array)
        else:
            X_scaled = X_array
        
        # 自动调参
        if self.auto_tune:
            if self.min_samples is None:
                self.min_samples = self._find_optimal_min_samples(X_scaled)
            
            if self.eps is None:
                self.eps, self._k_distances = self._find_optimal_eps(
                    X_scaled, k=self.min_samples
                )
        
        # 默认参数
        if self.eps is None:
            self.eps = 0.5
        if self.min_samples is None:
            self.min_samples = 5
        
        # 执行DBSCAN聚类
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = self.model.fit_predict(X_scaled)
        
        # 统计结果
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.n_noise_ = list(self.labels_).count(-1)
        self.core_sample_indices_ = self.model.core_sample_indices_
        
        # 计算评估指标（排除噪声点）
        if self.n_clusters_ > 1:
            mask = self.labels_ != -1
            if mask.sum() > self.n_clusters_:
                self.metrics_['silhouette'] = silhouette_score(
                    X_scaled[mask], self.labels_[mask]
                )
                self.metrics_['calinski_harabasz'] = calinski_harabasz_score(
                    X_scaled[mask], self.labels_[mask]
                )
                self.metrics_['davies_bouldin'] = davies_bouldin_score(
                    X_scaled[mask], self.labels_[mask]
                )
        
        if self.verbose:
            self._print_results(X_array)
        
        return self.labels_
    
    def _print_results(self, X):
        """打印聚类结果"""
        print("\n" + "=" * 60)
        print("DBSCAN 密度聚类分析结果")
        print("=" * 60)
        
        print(f"\n【参数设置】")
        print(f"  - ε (邻域半径): {self.eps:.4f}")
        print(f"  - MinPts (最小样本数): {self.min_samples}")
        
        print(f"\n【聚类结果】")
        print(f"  - 发现的簇数: {self.n_clusters_}")
        print(f"  - 噪声点数量: {self.n_noise_} ({100*self.n_noise_/len(self.labels_):.1f}%)")
        print(f"  - 核心点数量: {len(self.core_sample_indices_)}")
        
        print(f"\n【各簇样本分布】")
        for cluster_id in sorted(set(self.labels_)):
            count = list(self.labels_).count(cluster_id)
            if cluster_id == -1:
                print(f"  - 噪声点: {count} 个")
            else:
                print(f"  - 簇 {cluster_id}: {count} 个样本")
        
        if self.metrics_:
            print(f"\n【聚类质量评估】")
            print(f"  - 轮廓系数 (Silhouette): {self.metrics_.get('silhouette', 'N/A'):.4f}")
            print(f"    (范围[-1,1]，越大越好)")
            print(f"  - CH指数 (Calinski-Harabasz): {self.metrics_.get('calinski_harabasz', 'N/A'):.2f}")
            print(f"    (越大越好，表示簇间距离/簇内距离)")
            print(f"  - DB指数 (Davies-Bouldin): {self.metrics_.get('davies_bouldin', 'N/A'):.4f}")
            print(f"    (越小越好，表示簇内紧密且簇间分离)")
        
        print("=" * 60)
    
    def plot_k_distance(self, k=None, save_path=None):
        """
        绘制K-距离图用于选择eps
        
        :param k: K近邻的K值
        :param save_path: 保存路径
        """
        if not hasattr(self, '_k_distances'):
            print("请先调用 fit_predict() 方法")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制K-距离曲线
        ax.plot(range(len(self._k_distances)), self._k_distances, 
                'b-', linewidth=2, label='K-距离')
        
        # 标记选定的eps
        ax.axhline(y=self.eps, color='r', linestyle='--', 
                   label=f'选定的 ε = {self.eps:.4f}')
        
        ax.set_xlabel('样本点（按距离排序）', fontsize=12)
        ax.set_ylabel(f'{self.min_samples}-距离', fontsize=12)
        ax.set_title('K-距离图 (用于确定最优 ε)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            FigureSaver.save(fig, save_path)
        plt.show()
        
        return fig
    
    def plot_clusters_2d(self, X, feature_names=None, use_pca=True, 
                         highlight_noise=True, save_path=None):
        """
        绘制2D聚类结果图
        
        :param X: 原始数据
        :param feature_names: 特征名称
        :param use_pca: 高维数据是否使用PCA降维
        :param highlight_noise: 是否突出显示噪声点
        :param save_path: 保存路径
        """
        if self.labels_ is None:
            print("请先调用 fit_predict() 方法")
            return
        
        # 转换为数组
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = X
            if feature_names is None:
                feature_names = [f'特征{i+1}' for i in range(X.shape[1])]
        
        # 标准化
        X_scaled = self.scaler.transform(X_array)
        
        # 降维到2D
        if X_scaled.shape[1] > 2:
            if use_pca:
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X_scaled)
                x_label = f'PC1 ({pca.explained_variance_ratio_[0]:.1%})'
                y_label = f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
            else:
                X_2d = X_scaled[:, :2]
                x_label = feature_names[0]
                y_label = feature_names[1]
        else:
            X_2d = X_scaled
            x_label = feature_names[0]
            y_label = feature_names[1] if len(feature_names) > 1 else '特征2'
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 图1：聚类结果
        ax1 = axes[0]
        
        # 获取唯一的簇标签（排序，噪声在最后）
        unique_labels = sorted(set(self.labels_))
        
        # 定义颜色
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 10)))
        
        for idx, cluster_id in enumerate(unique_labels):
            mask = self.labels_ == cluster_id
            
            if cluster_id == -1:
                # 噪声点
                if highlight_noise:
                    ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                               c='gray', marker='x', s=50, alpha=0.5,
                               label=f'噪声 ({mask.sum()})')
            else:
                # 正常簇
                ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                           c=[colors[idx]], s=50, alpha=0.7,
                           label=f'簇 {cluster_id} ({mask.sum()})')
        
        # 标记核心点
        core_mask = np.zeros(len(self.labels_), dtype=bool)
        core_mask[self.core_sample_indices_] = True
        ax1.scatter(X_2d[core_mask, 0], X_2d[core_mask, 1], 
                   facecolors='none', edgecolors='black', s=100, 
                   linewidths=1, alpha=0.5, label='核心点边界')
        
        ax1.set_xlabel(x_label, fontsize=12)
        ax1.set_ylabel(y_label, fontsize=12)
        ax1.set_title(f'DBSCAN聚类结果 (ε={self.eps:.3f}, MinPts={self.min_samples})', 
                     fontsize=14)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 图2：点类型分布
        ax2 = axes[1]
        
        # 统计三类点
        n_core = len(self.core_sample_indices_)
        n_noise = self.n_noise_
        n_border = len(self.labels_) - n_core - n_noise
        
        categories = ['核心点', '边界点', '噪声点']
        counts = [n_core, n_border, n_noise]
        colors_pie = ['#2ecc71', '#3498db', '#e74c3c']
        
        wedges, texts, autotexts = ax2.pie(counts, labels=categories, 
                                            colors=colors_pie, autopct='%1.1f%%',
                                            startangle=90, explode=(0.05, 0, 0.1))
        ax2.set_title('点类型分布', fontsize=14)
        
        # 添加图例
        legend_labels = [f'{cat}: {cnt}' for cat, cnt in zip(categories, counts)]
        ax2.legend(wedges, legend_labels, loc='lower center', 
                  bbox_to_anchor=(0.5, -0.1))
        
        plt.tight_layout()
        
        if save_path:
            FigureSaver.save(fig, save_path)
        plt.show()
        
        return fig
    
    def get_noise_indices(self):
        """获取噪声点的索引"""
        if self.labels_ is None:
            return None
        return np.where(self.labels_ == -1)[0]
    
    def get_cluster_centers(self, X):
        """
        计算各簇的中心点（均值）
        
        :param X: 原始数据
        :return: 各簇中心的DataFrame
        """
        if self.labels_ is None:
            return None
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            columns = list(X.columns)
        else:
            X_array = X
            columns = [f'特征{i+1}' for i in range(X.shape[1])]
        
        centers = []
        for cluster_id in sorted(set(self.labels_)):
            if cluster_id == -1:
                continue
            mask = self.labels_ == cluster_id
            center = X_array[mask].mean(axis=0)
            centers.append(center)
        
        centers_df = pd.DataFrame(centers, 
                                   columns=columns,
                                   index=[f'簇{i}' for i in range(len(centers))])
        return centers_df


def eps_sensitivity_analysis(X, eps_range, min_samples=5, verbose=True):
    """
    eps参数敏感性分析
    
    :param X: 输入数据
    :param eps_range: eps值范围列表
    :param min_samples: 固定的min_samples值
    :param verbose: 是否打印详情
    :return: 分析结果DataFrame
    """
    results = []
    
    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = n_noise / len(labels)
        
        # 计算轮廓系数（如果有足够的簇）
        silhouette = None
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > n_clusters:
                silhouette = silhouette_score(X[mask], labels[mask])
        
        results.append({
            'eps': eps,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'silhouette': silhouette
        })
    
    results_df = pd.DataFrame(results)
    
    if verbose:
        print("\n【eps敏感性分析结果】")
        print(results_df.to_string(index=False))
    
    return results_df


def plot_eps_sensitivity(results_df, save_path=None):
    """
    绘制eps敏感性分析图
    
    :param results_df: eps_sensitivity_analysis的输出
    :param save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 图1：簇数量 vs eps
    ax1 = axes[0, 0]
    ax1.plot(results_df['eps'], results_df['n_clusters'], 'b-o', linewidth=2)
    ax1.set_xlabel('ε (邻域半径)', fontsize=12)
    ax1.set_ylabel('簇数量', fontsize=12)
    ax1.set_title('簇数量随 ε 变化', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 图2：噪声比例 vs eps
    ax2 = axes[0, 1]
    ax2.plot(results_df['eps'], results_df['noise_ratio']*100, 'r-o', linewidth=2)
    ax2.set_xlabel('ε (邻域半径)', fontsize=12)
    ax2.set_ylabel('噪声比例 (%)', fontsize=12)
    ax2.set_title('噪声比例随 ε 变化', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # 图3：轮廓系数 vs eps
    ax3 = axes[1, 0]
    valid = results_df['silhouette'].notna()
    if valid.any():
        ax3.plot(results_df.loc[valid, 'eps'], 
                results_df.loc[valid, 'silhouette'], 'g-o', linewidth=2)
        ax3.set_xlabel('ε (邻域半径)', fontsize=12)
        ax3.set_ylabel('轮廓系数', fontsize=12)
        ax3.set_title('轮廓系数随 ε 变化', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # 标记最优点
        best_idx = results_df.loc[valid, 'silhouette'].idxmax()
        best_eps = results_df.loc[best_idx, 'eps']
        best_sil = results_df.loc[best_idx, 'silhouette']
        ax3.scatter([best_eps], [best_sil], s=200, c='red', marker='*', 
                   label=f'最优: ε={best_eps:.3f}', zorder=5)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, '无法计算轮廓系数\n(簇数不足)', 
                ha='center', va='center', fontsize=14)
    
    # 图4：综合推荐
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 找最优eps
    if valid.any():
        best_idx = results_df.loc[valid, 'silhouette'].idxmax()
        best_row = results_df.loc[best_idx]
        recommendation = f"""
【参数推荐】

基于轮廓系数最优的参数选择：

  ε (eps) = {best_row['eps']:.4f}
  
  对应结果：
  - 簇数量: {int(best_row['n_clusters'])}
  - 噪声点: {int(best_row['n_noise'])} ({best_row['noise_ratio']*100:.1f}%)
  - 轮廓系数: {best_row['silhouette']:.4f}

【选择建议】
- 若噪声过多，适当增大 ε
- 若簇数过少，适当减小 ε
- 需根据业务需求权衡
        """
    else:
        recommendation = "无法给出推荐，请调整参数范围"
    
    ax4.text(0.1, 0.5, recommendation, fontsize=12, 
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('参数推荐', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        FigureSaver.save(fig, save_path)
    plt.show()
    
    return fig


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("DBSCAN 密度聚类分析 - 使用示例")
    print("=" * 60)
    
    # 生成示例数据（包含三个簇和一些噪声）
    np.random.seed(42)
    
    # 簇1：左上角
    cluster1 = np.random.randn(50, 2) * 0.5 + np.array([-3, 3])
    
    # 簇2：右下角（椭圆形）
    cluster2 = np.random.randn(70, 2) * np.array([1.5, 0.5]) + np.array([3, -2])
    
    # 簇3：中心（月牙形）
    theta = np.linspace(0, np.pi, 60)
    cluster3 = np.column_stack([2*np.cos(theta) + np.random.randn(60)*0.2,
                                 np.sin(theta) + np.random.randn(60)*0.2])
    
    # 噪声点
    noise = np.random.uniform(-6, 6, (20, 2))
    
    # 合并数据
    X = np.vstack([cluster1, cluster2, cluster3, noise])
    
    print(f"\n数据集信息：")
    print(f"  - 样本数: {X.shape[0]}")
    print(f"  - 特征数: {X.shape[1]}")
    print(f"  - 预期簇数: 3")
    print(f"  - 预期噪声点: ~20")
    
    # 创建DBSCAN聚类器
    print("\n" + "-" * 40)
    print("【自动调参模式】")
    dbscan = DBSCANClusterer(auto_tune=True, verbose=True)
    
    # 执行聚类（已标准化的数据，不需要再标准化）
    labels = dbscan.fit_predict(X, standardize=True)
    
    # 绘制K-距离图
    print("\n绘制K-距离图...")
    dbscan.plot_k_distance()
    
    # 绘制聚类结果
    print("\n绘制聚类结果...")
    dbscan.plot_clusters_2d(X, feature_names=['X', 'Y'], use_pca=False)
    
    # 获取簇中心
    centers = dbscan.get_cluster_centers(X)
    print("\n【各簇中心】")
    print(centers)
    
    # 获取噪声点索引
    noise_idx = dbscan.get_noise_indices()
    print(f"\n噪声点索引: {noise_idx[:10]}... (共{len(noise_idx)}个)")
    
    # eps敏感性分析
    print("\n" + "-" * 40)
    print("【eps敏感性分析】")
    
    eps_range = np.linspace(0.2, 1.5, 15)
    sensitivity_results = eps_sensitivity_analysis(
        StandardScaler().fit_transform(X), 
        eps_range, 
        min_samples=5
    )
    
    # 绘制敏感性分析图
    plot_eps_sensitivity(sensitivity_results)
    
    print("\n" + "=" * 60)
    print("DBSCAN聚类分析完成！")
    print("=" * 60)
