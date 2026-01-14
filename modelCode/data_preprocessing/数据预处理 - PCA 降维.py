"""
============================================================
PCA 主成分分析降维 (Principal Component Analysis)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：高维数据降维、特征提取、数据可视化、去除冗余
原理：寻找方差最大的投影方向
作者：MCM/ICM Team
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 图表美化设置
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


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
            if i >= self.n_selected - 1:
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
        
        n = len(self.explained_variance)
        x = np.arange(1, n + 1)
        
        # 柱状图
        bars = ax.bar(x, self.explained_variance, color='#2E86AB', 
                     edgecolor='white', linewidth=2, alpha=0.8, label='单个方差贡献')
        
        # 累计曲线
        ax.plot(x, self.cumulative_variance, 'o-', color='#E94F37', 
               linewidth=2.5, markersize=8, label='累计方差')
        
        # 阈值线
        ax.axhline(y=self.variance_threshold, color='green', linestyle='--',
                  linewidth=2, label=f'阈值 ({self.variance_threshold*100:.0f}%)')
        
        # 标记选择点
        ax.axvline(x=self.n_selected, color='orange', linestyle=':',
                  linewidth=2, label=f'选择 {self.n_selected} 个主成分')
        
        ax.set_xlabel('主成分', fontsize=12, fontweight='bold')
        ax.set_ylabel('方差贡献比', fontsize=12, fontweight='bold')
        ax.set_title('PCA 方差贡献分析', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.legend(loc='center right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_2d(self, X, y=None, save_path=None):
        """2D散点图可视化"""
        X_pca = self.fit_transform(X)
        
        if X_pca.shape[1] < 2:
            print("需要至少2个主成分才能绘制2D图")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if y is not None:
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                               cmap='viridis', s=60, alpha=0.7, edgecolor='white')
            plt.colorbar(scatter, ax=ax, label='类别')
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], color='#2E86AB',
                      s=60, alpha=0.7, edgecolor='white')
        
        ax.set_xlabel(f'PC1 ({self.explained_variance[0]*100:.1f}%)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({self.explained_variance[1]*100:.1f}%)',
                     fontsize=12, fontweight='bold')
        ax.set_title('PCA 2D 投影', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("   PCA 主成分分析演示")
    print("="*60)
    
    # 1. 生成高维数据（5个特征，含相关性）
    np.random.seed(42)
    n = 200
    feature1 = np.random.normal(0, 1, n)
    feature2 = 0.8*feature1 + np.random.normal(0, 0.5, n)  # 与feature1相关
    feature3 = 0.7*feature1 + 0.2*feature2 + np.random.normal(0, 0.4, n)
    feature4 = np.random.normal(1, 1, n)  # 独立特征
    feature5 = 0.6*feature4 + np.random.normal(0, 0.6, n)
    
    data = pd.DataFrame({
        "f1": feature1, "f2": feature2, "f3": feature3, 
        "f4": feature4, "f5": feature5
    })
    
    print("\n原始数据概览：")
    print(data.describe().round(2))
    
    # 2. PCA降维
    pca = PCAReducer(variance_threshold=0.85, verbose=True)
    data_reduced = pca.fit_transform(data)
    
    print(f"\n降维后数据形状: {data_reduced.shape}")
    
    # 3. 主成分载荷
    loadings = pca.get_loadings()
    print("\n主成分载荷（各特征对主成分的贡献）：")
    print(loadings.round(4))
    
    # 4. 可视化
    pca.plot_variance()
    
    # 5. 带标签的2D可视化
    labels = np.random.choice([0, 1, 2], n)  # 模拟类别标签
    pca.plot_2d(data, y=labels)
