"""
============================================================
PCA 主成分分析降维 (Principal Component Analysis)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：高维数据降维、特征提取、数据可视化
原理：寻找方差最大的投影方向
作者：MCM/ICM Team
日期：2026年1月
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.plot_config import PlotStyleConfig, FigureSaver

PlotStyleConfig.setup_style()


class PCAReducer:
    """
    PCA降维封装类
    
    核心原理：
    1. 数据标准化（零均值，单位方差）
    2. 计算协方差矩阵
    3. 求特征值和特征向量
    4. 选取前k个主成分
    
    应用场景：
    - 高维数据可视化（降到2D/3D）
    - 去除特征间的共线性
    - 降低计算复杂度
    - 数据压缩
    """
    
    def __init__(self, n_components=None, variance_threshold=0.85, verbose=True):
        """
        参数配置
        
        :param n_components: 保留的主成分数（None自动选择）
        :param variance_threshold: 自动选择时的方差阈值
        :param verbose: 是否打印过程
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.verbose = verbose
        
        self.scaler = StandardScaler()
        self.pca = None
        self.explained_variance = None
        self.cumulative_variance = None
        self.components = None
        self.feature_names = None
        self.n_selected = None
    
    def fit_transform(self, X):
        """
        拟合并转换数据
        
        :param X: 原始数据（DataFrame或数组）
        :return: 降维后的数据
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"特征{i+1}" for i in range(X.shape[1])]
        
        # 标准化
        X_std = self.scaler.fit_transform(X)
        
        # 拟合PCA
        self.pca = PCA()
        pca_result = self.pca.fit_transform(X_std)
        
        # 计算方差贡献
        self.explained_variance = self.pca.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.explained_variance)
        self.components = self.pca.components_
        
        # 确定保留的主成分数
        if self.n_components is None:
            self.n_selected = np.argmax(self.cumulative_variance >= self.variance_threshold) + 1
        else:
            self.n_selected = self.n_components
        
        if self.verbose:
            self._print_results(X)
        
        return pca_result[:, :self.n_selected]
    
    def _print_results(self, X):
        """打印结果"""
        print("\n" + "="*50)
        print("📊 PCA 主成分分析结果")
        print("="*50)
        print(f"\n  原始维度: {X.shape[1]}")
        print(f"  保留主成分: {self.n_selected}")
        print(f"  方差阈值: {self.variance_threshold*100:.0f}%")
        print(f"\n  各主成分方差贡献:")
        for i, (var, cum) in enumerate(zip(self.explained_variance, self.cumulative_variance)):
            bar = "█" * int(var * 30)
            print(f"    PC{i+1}: {var:.4f} (累计: {cum:.4f}) {bar}")
            if i >= min(self.n_selected + 2, 10) - 1:
                if len(self.explained_variance) > 10:
                    print("    ...")
                break
        print(f"\n  降维后维度: {self.n_selected}")
        print(f"  保留信息量: {self.cumulative_variance[self.n_selected-1]*100:.1f}%")
        print("="*50)
    
    def get_loadings(self):
        """获取主成分载荷（各特征对主成分的贡献）"""
        loadings = pd.DataFrame(
            self.components[:self.n_selected].T,
            index=self.feature_names,
            columns=[f"PC{i+1}" for i in range(self.n_selected)]
        )
        return loadings
    
    def plot_variance(self, save_path=None):
        """可视化方差贡献"""
        if self.explained_variance is None:
            raise ValueError("请先调用fit_transform()")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n = min(len(self.explained_variance), 15)
        x = np.arange(1, n + 1)
        
        # 柱状图
        ax.bar(x, self.explained_variance[:n], color=PlotStyleConfig.COLORS['primary'], 
              edgecolor='white', linewidth=2, alpha=0.8, label='单个方差贡献')
        
        # 累计曲线
        ax.plot(x, self.cumulative_variance[:n], 'o-', color=PlotStyleConfig.COLORS['danger'], 
               linewidth=2.5, markersize=8, label='累计方差')
        
        # 阈值线
        ax.axhline(y=self.variance_threshold, color=PlotStyleConfig.COLORS['success'], 
                  linestyle='--', linewidth=2, label=f'阈值 ({self.variance_threshold*100:.0f}%)')
        
        # 标记选择点
        ax.axvline(x=self.n_selected, color=PlotStyleConfig.COLORS['accent'], 
                  linestyle=':', linewidth=2, label=f'选择 {self.n_selected} 个主成分')
        
        ax.set_xlabel('主成分', fontsize=12, fontweight='bold')
        ax.set_ylabel('方差贡献率', fontsize=12, fontweight='bold')
        ax.set_title('PCA 方差解释率分析', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.legend(loc='center right')
        ax.set_ylim(0, 1.1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax
    
    def plot_loadings_heatmap(self, save_path=None):
        """可视化主成分载荷热力图"""
        loadings = self.get_loadings()
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(self.feature_names) * 0.4)))
        
        im = ax.imshow(loadings.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        ax.set_xticks(np.arange(self.n_selected))
        ax.set_yticks(np.arange(len(self.feature_names)))
        ax.set_xticklabels([f'PC{i+1}' for i in range(self.n_selected)])
        ax.set_yticklabels(self.feature_names)
        
        # 添加数值标注
        for i in range(len(self.feature_names)):
            for j in range(self.n_selected):
                text = ax.text(j, i, f'{loadings.values[i, j]:.2f}',
                              ha='center', va='center', color='black', fontsize=9)
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('载荷系数', fontweight='bold')
        
        ax.set_title('主成分载荷热力图', fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax
    
    def plot_2d_scatter(self, X_pca, labels=None, save_path=None):
        """
        2D散点图可视化
        
        :param X_pca: PCA转换后的数据
        :param labels: 类别标签（可选）
        """
        if X_pca.shape[1] < 2:
            raise ValueError("需要至少2个主成分")
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = PlotStyleConfig.get_palette(len(unique_labels))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=colors[i], label=f'{label}', s=60, alpha=0.7, edgecolors='white')
            ax.legend()
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                      c=PlotStyleConfig.COLORS['primary'], s=60, alpha=0.7, edgecolors='white')
        
        ax.set_xlabel(f'PC1 ({self.explained_variance[0]*100:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({self.explained_variance[1]*100:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_title('PCA 2D 可视化', fontsize=14, fontweight='bold', pad=15)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            saver = FigureSaver(os.path.dirname(save_path))
            saver.save(fig, os.path.basename(save_path).split('.')[0])
        
        return fig, ax


if __name__ == "__main__":
    # 演示
    print("="*60)
    print("📊 PCA 主成分分析演示")
    print("="*60)
    
    # 生成高维测试数据
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    # 生成有结构的数据（前3个特征有信息）
    informative = np.random.randn(n_samples, 3)
    noise = np.random.randn(n_samples, n_features - 3) * 0.5
    
    # 部分特征是信息特征的线性组合
    X = np.hstack([
        informative,
        informative @ np.random.randn(3, 4),  # 冗余特征
        noise[:, :3]  # 噪声特征
    ])
    
    feature_names = [f'特征{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # PCA降维
    reducer = PCAReducer(variance_threshold=0.9)
    X_reduced = reducer.fit_transform(df)
    
    print(f"\n降维结果形状: {X_reduced.shape}")
    
    # 可视化
    fig1, ax1 = reducer.plot_variance()
    plt.show()
    
    fig2, ax2 = reducer.plot_loadings_heatmap()
    plt.show()
    
    # 生成标签进行分类可视化
    labels = np.random.choice(['A类', 'B类', 'C类'], n_samples)
    fig3, ax3 = reducer.plot_2d_scatter(X_reduced, labels)
    plt.show()
    
    print("\n✅ PCA 演示完成!")
