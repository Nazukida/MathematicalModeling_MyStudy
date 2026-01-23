"""
============================================================
因子分析 (Factor Analysis)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：发现潜在因子结构、降维、变量分组解释
原理：假设观测变量由少数公共因子线性组合而成
与PCA区别：FA关注共同方差，具有更强的解释意义
作者：MCM/ICM Team
日期：2026年1月
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from scipy.stats import bartlett, zscore
from scipy.linalg import svd
import warnings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.plot_config import PlotStyleConfig, FigureSaver

PlotStyleConfig.setup_style()


class FactorAnalyzer:
    """
    因子分析封装类
    
    核心原理：
    X = ΛF + ε
    - X: 观测变量 (p维)
    - F: 公共因子 (m维, m << p)  
    - Λ: 因子载荷矩阵 (p × m)
    - ε: 特殊因子（误差项）
    
    应用场景：
    - 问卷/量表分析（发现潜在构念）
    - 变量降维与分组
    - 构建综合评价指标
    - 验证理论假设（变量结构）
    
    与PCA的主要区别：
    | PCA | 因子分析 |
    |-----|---------|
    | 解释总方差 | 解释共同方差 |
    | 无特殊假设 | 假设存在潜在因子 |
    | 结果唯一 | 需要因子旋转 |
    | 数据压缩 | 理论验证/结构发现 |
    """
    
    def __init__(self, n_factors=None, rotation='varimax', 
                 variance_threshold=0.85, verbose=True):
        """
        参数配置
        
        :param n_factors: 提取的因子数（None自动确定）
        :param rotation: 旋转方法 ('varimax', 'promax', 'quartimax', None)
        :param variance_threshold: 自动确定因子数时的累计方差阈值
        :param verbose: 是否打印详细信息
        """
        self.n_factors = n_factors
        self.rotation = rotation
        self.variance_threshold = variance_threshold
        self.verbose = verbose
        
        self.scaler = StandardScaler()
        self.model = None
        self.loadings_ = None
        self.communalities_ = None
        self.eigenvalues_ = None
        self.variance_explained_ = None
        self.factor_scores_ = None
        self.feature_names_ = None
        self.n_selected_ = None
        
    def _check_factorability(self, X):
        """
        检验数据是否适合做因子分析
        
        使用Bartlett球形检验：
        H0: 相关矩阵是单位矩阵（变量间无相关性）
        H1: 相关矩阵不是单位矩阵（变量间存在相关性）
        """
        n, p = X.shape
        corr_matrix = np.corrcoef(X.T)
        
        # Bartlett检验
        chi_square = -(n - 1 - (2*p + 5)/6) * np.log(np.linalg.det(corr_matrix))
        df = p * (p - 1) / 2
        
        # 计算p值（近似）
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(chi_square, df)
        
        # KMO检验（简化版本）
        # 计算偏相关矩阵
        inv_corr = np.linalg.inv(corr_matrix)
        partial_corr = np.zeros_like(corr_matrix)
        for i in range(p):
            for j in range(p):
                partial_corr[i, j] = -inv_corr[i, j] / np.sqrt(inv_corr[i, i] * inv_corr[j, j])
        
        # KMO = Σr² / (Σr² + Σp²)
        r_sum = np.sum(np.triu(corr_matrix, 1)**2)
        p_sum = np.sum(np.triu(partial_corr, 1)**2)
        kmo = r_sum / (r_sum + p_sum) if (r_sum + p_sum) > 0 else 0
        
        return {
            'bartlett_chi_square': chi_square,
            'bartlett_df': df,
            'bartlett_p_value': p_value,
            'kmo': kmo
        }
    
    def _determine_n_factors(self, X):
        """
        自动确定因子数量
        
        使用方法：
        1. Kaiser准则：特征值 > 1
        2. 累计方差贡献率 > 阈值
        """
        # 计算相关矩阵的特征值
        corr_matrix = np.corrcoef(X.T)
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # 降序
        
        self.eigenvalues_ = eigenvalues
        
        # Kaiser准则
        n_kaiser = np.sum(eigenvalues > 1)
        
        # 累计方差准则
        variance_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_variance = np.cumsum(variance_ratio)
        n_variance = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        
        # 取较保守的值
        n_factors = min(n_kaiser, n_variance)
        n_factors = max(n_factors, 2)  # 至少2个因子
        n_factors = min(n_factors, X.shape[1] - 1)  # 不超过变量数-1
        
        return n_factors, eigenvalues, cumulative_variance
    
    def _varimax_rotation(self, loadings, gamma=1.0, max_iter=100, tol=1e-6):
        """
        Varimax正交旋转
        
        目标：使每个因子上的载荷尽量两极化
        （高载荷更高，低载荷更低）
        """
        n_vars, n_factors = loadings.shape
        rotated = loadings.copy()
        
        for _ in range(max_iter):
            old_rotated = rotated.copy()
            
            for i in range(n_factors - 1):
                for j in range(i + 1, n_factors):
                    # 对因子i和j进行旋转
                    u = rotated[:, i]
                    v = rotated[:, j]
                    
                    u2 = u**2
                    v2 = v**2
                    
                    A = np.sum(u2 - v2)
                    B = 2 * np.sum(u * v)
                    C = np.sum((u2 - v2)**2 - 4 * (u * v)**2)
                    D = 4 * np.sum((u2 - v2) * u * v)
                    
                    phi = 0.25 * np.arctan2(D, C)
                    
                    cos_phi = np.cos(phi)
                    sin_phi = np.sin(phi)
                    
                    rotated[:, i] = u * cos_phi + v * sin_phi
                    rotated[:, j] = -u * sin_phi + v * cos_phi
            
            # 检查收敛
            if np.sum((rotated - old_rotated)**2) < tol:
                break
        
        return rotated
    
    def fit_transform(self, X, feature_names=None):
        """
        拟合因子分析模型并计算因子得分
        
        :param X: 输入数据 (n_samples, n_features)
        :param feature_names: 特征名称列表
        :return: 因子得分矩阵
        """
        # 处理DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_array = X.values
        else:
            X_array = X.copy()
            self.feature_names_ = feature_names or [f'X{i+1}' for i in range(X.shape[1])]
        
        # 标准化
        X_std = self.scaler.fit_transform(X_array)
        
        # 检验因子分析适用性
        if self.verbose:
            factorability = self._check_factorability(X_std)
            print("\n" + "=" * 60)
            print("因子分析适用性检验")
            print("=" * 60)
            print(f"\n【Bartlett球形检验】")
            print(f"  - 卡方值: {factorability['bartlett_chi_square']:.2f}")
            print(f"  - 自由度: {int(factorability['bartlett_df'])}")
            print(f"  - p值: {factorability['bartlett_p_value']:.4e}")
            if factorability['bartlett_p_value'] < 0.05:
                print("  ✓ p < 0.05，拒绝原假设，适合做因子分析")
            else:
                print("  ✗ p >= 0.05，变量间相关性不足，可能不适合因子分析")
            
            print(f"\n【KMO检验】")
            print(f"  - KMO值: {factorability['kmo']:.4f}")
            if factorability['kmo'] >= 0.9:
                print("  ✓ 极佳 (KMO >= 0.9)")
            elif factorability['kmo'] >= 0.8:
                print("  ✓ 良好 (0.8 <= KMO < 0.9)")
            elif factorability['kmo'] >= 0.7:
                print("  ○ 中等 (0.7 <= KMO < 0.8)")
            elif factorability['kmo'] >= 0.6:
                print("  △ 一般 (0.6 <= KMO < 0.7)")
            else:
                print("  ✗ 不适合做因子分析 (KMO < 0.6)")
        
        # 确定因子数
        if self.n_factors is None:
            self.n_selected_, self.eigenvalues_, cum_var = self._determine_n_factors(X_std)
        else:
            self.n_selected_ = self.n_factors
            _, self.eigenvalues_, cum_var = self._determine_n_factors(X_std)
        
        # 使用sklearn的FactorAnalysis
        self.model = FactorAnalysis(n_components=self.n_selected_, random_state=42)
        self.factor_scores_ = self.model.fit_transform(X_std)
        
        # 获取因子载荷矩阵
        self.loadings_ = self.model.components_.T
        
        # 因子旋转
        if self.rotation == 'varimax' and self.n_selected_ > 1:
            self.loadings_ = self._varimax_rotation(self.loadings_)
        
        # 计算共同度
        self.communalities_ = np.sum(self.loadings_**2, axis=1)
        
        # 计算方差解释
        self.variance_explained_ = np.sum(self.loadings_**2, axis=0)
        self.variance_ratio_ = self.variance_explained_ / X_std.shape[1]
        
        if self.verbose:
            self._print_results()
        
        return self.factor_scores_
    
    def _print_results(self):
        """打印分析结果"""
        print("\n" + "=" * 60)
        print("因子分析结果")
        print("=" * 60)
        
        print(f"\n【基本信息】")
        print(f"  - 原始变量数: {len(self.feature_names_)}")
        print(f"  - 提取因子数: {self.n_selected_}")
        print(f"  - 旋转方法: {self.rotation or '无旋转'}")
        
        # 方差解释
        print(f"\n【因子方差解释】")
        print(f"  {'因子':<8} {'方差贡献':<12} {'贡献率':<12} {'累计贡献率':<12}")
        print(f"  {'-'*44}")
        cum_var = 0
        for i in range(self.n_selected_):
            var = self.variance_explained_[i]
            ratio = self.variance_ratio_[i]
            cum_var += ratio
            print(f"  {'F'+str(i+1):<8} {var:<12.4f} {ratio*100:<11.2f}% {cum_var*100:<11.2f}%")
        
        # 因子载荷矩阵
        print(f"\n【因子载荷矩阵】(|载荷| > 0.5 标记为 *)")
        header = "  变量" + " " * 10
        for i in range(self.n_selected_):
            header += f"{'F'+str(i+1):<12}"
        header += "共同度"
        print(header)
        print(f"  {'-'*(16 + 12*self.n_selected_ + 8)}")
        
        for j, name in enumerate(self.feature_names_):
            row = f"  {name:<16}"
            for i in range(self.n_selected_):
                loading = self.loadings_[j, i]
                marker = "*" if abs(loading) > 0.5 else " "
                row += f"{loading:>10.4f}{marker} "
            row += f"{self.communalities_[j]:>8.4f}"
            print(row)
        
        # 因子命名建议
        print(f"\n【因子命名建议】")
        for i in range(self.n_selected_):
            # 找到该因子上载荷最高的变量
            loadings_i = self.loadings_[:, i]
            top_indices = np.argsort(np.abs(loadings_i))[::-1][:3]
            top_vars = [self.feature_names_[idx] for idx in top_indices]
            top_loadings = [loadings_i[idx] for idx in top_indices]
            
            print(f"  因子 F{i+1}:")
            for var, load in zip(top_vars, top_loadings):
                direction = "+" if load > 0 else "-"
                print(f"    {direction} {var} ({load:.3f})")
        
        print("=" * 60)
    
    def plot_scree(self, save_path=None):
        """
        绘制碎石图（确定因子数）
        """
        if self.eigenvalues_ is None:
            print("请先调用 fit_transform() 方法")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n_eigenvalues = len(self.eigenvalues_)
        x = range(1, n_eigenvalues + 1)
        
        # 绘制特征值
        ax.plot(x, self.eigenvalues_, 'b-o', linewidth=2, markersize=8, label='特征值')
        ax.axhline(y=1, color='r', linestyle='--', label='Kaiser准则 (λ=1)')
        
        # 标记选定的因子数
        ax.axvline(x=self.n_selected_, color='g', linestyle=':', 
                   label=f'选定因子数: {self.n_selected_}')
        
        ax.set_xlabel('因子序号', fontsize=12)
        ax.set_ylabel('特征值', fontsize=12)
        ax.set_title('碎石图 (Scree Plot)', fontsize=14)
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            FigureSaver.save(fig, save_path)
        plt.show()
        
        return fig
    
    def plot_loadings_heatmap(self, save_path=None):
        """
        绘制因子载荷热力图
        """
        if self.loadings_ is None:
            print("请先调用 fit_transform() 方法")
            return
        
        fig, ax = plt.subplots(figsize=(10, max(8, len(self.feature_names_) * 0.4)))
        
        # 创建热力图数据
        loadings_df = pd.DataFrame(
            self.loadings_,
            index=self.feature_names_,
            columns=[f'F{i+1}' for i in range(self.n_selected_)]
        )
        
        # 绘制热力图
        im = ax.imshow(loadings_df.values, cmap='RdBu_r', aspect='auto',
                       vmin=-1, vmax=1)
        
        # 设置刻度
        ax.set_xticks(range(self.n_selected_))
        ax.set_xticklabels(loadings_df.columns)
        ax.set_yticks(range(len(self.feature_names_)))
        ax.set_yticklabels(loadings_df.index)
        
        # 添加数值标注
        for i in range(len(self.feature_names_)):
            for j in range(self.n_selected_):
                value = loadings_df.iloc[i, j]
                color = 'white' if abs(value) > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color=color, fontsize=10)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('因子载荷', fontsize=12)
        
        ax.set_xlabel('因子', fontsize=12)
        ax.set_ylabel('变量', fontsize=12)
        ax.set_title('因子载荷矩阵热力图', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            FigureSaver.save(fig, save_path)
        plt.show()
        
        return fig
    
    def plot_factor_scores_2d(self, factor1=0, factor2=1, labels=None, save_path=None):
        """
        绘制因子得分散点图（2D）
        
        :param factor1: 第一个因子的索引
        :param factor2: 第二个因子的索引  
        :param labels: 样本标签（用于着色）
        :param save_path: 保存路径
        """
        if self.factor_scores_ is None:
            print("请先调用 fit_transform() 方法")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(self.factor_scores_[mask, factor1], 
                          self.factor_scores_[mask, factor2],
                          c=[colors[i]], s=50, alpha=0.7, label=f'{label}')
            ax.legend()
        else:
            ax.scatter(self.factor_scores_[:, factor1], 
                      self.factor_scores_[:, factor2],
                      c='steelblue', s=50, alpha=0.7)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel(f'因子 F{factor1+1} ({self.variance_ratio_[factor1]*100:.1f}%)', 
                     fontsize=12)
        ax.set_ylabel(f'因子 F{factor2+1} ({self.variance_ratio_[factor2]*100:.1f}%)', 
                     fontsize=12)
        ax.set_title('因子得分散点图', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            FigureSaver.save(fig, save_path)
        plt.show()
        
        return fig
    
    def plot_biplot(self, factor1=0, factor2=1, save_path=None):
        """
        绘制双标图（Biplot）
        
        同时展示样本的因子得分和变量的因子载荷
        """
        if self.factor_scores_ is None or self.loadings_ is None:
            print("请先调用 fit_transform() 方法")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 标准化因子得分以便可视化
        scores_scaled = self.factor_scores_ / np.abs(self.factor_scores_).max(axis=0)
        
        # 绘制样本点
        ax.scatter(scores_scaled[:, factor1], scores_scaled[:, factor2],
                  c='steelblue', s=30, alpha=0.5, label='样本')
        
        # 绘制变量向量
        for i, name in enumerate(self.feature_names_):
            loading1 = self.loadings_[i, factor1]
            loading2 = self.loadings_[i, factor2]
            
            ax.arrow(0, 0, loading1*0.9, loading2*0.9, 
                    head_width=0.03, head_length=0.02, fc='red', ec='red')
            ax.text(loading1*1.05, loading2*1.05, name, 
                   fontsize=10, ha='center', va='center')
        
        # 绘制参考圆
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
        ax.add_patch(circle)
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        
        ax.set_xlabel(f'因子 F{factor1+1} ({self.variance_ratio_[factor1]*100:.1f}%)', 
                     fontsize=12)
        ax.set_ylabel(f'因子 F{factor2+1} ({self.variance_ratio_[factor2]*100:.1f}%)', 
                     fontsize=12)
        ax.set_title('双标图 (Biplot)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            FigureSaver.save(fig, save_path)
        plt.show()
        
        return fig
    
    def get_factor_weights(self, method='variance'):
        """
        获取因子权重（用于综合评价）
        
        :param method: 权重计算方法
                      'variance': 基于方差贡献率
                      'equal': 等权重
        :return: 权重数组
        """
        if self.variance_ratio_ is None:
            return None
        
        if method == 'variance':
            weights = self.variance_ratio_ / self.variance_ratio_.sum()
        elif method == 'equal':
            weights = np.ones(self.n_selected_) / self.n_selected_
        else:
            weights = self.variance_ratio_ / self.variance_ratio_.sum()
        
        return weights
    
    def compute_composite_score(self, X=None, method='variance'):
        """
        计算综合因子得分
        
        :param X: 新数据（None则使用训练数据）
        :param method: 权重方法
        :return: 综合得分数组
        """
        if X is not None:
            if isinstance(X, pd.DataFrame):
                X = X.values
            X_std = self.scaler.transform(X)
            factor_scores = self.model.transform(X_std)
        else:
            factor_scores = self.factor_scores_
        
        weights = self.get_factor_weights(method)
        composite_score = np.dot(factor_scores, weights)
        
        return composite_score


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("因子分析 - 使用示例")
    print("=" * 60)
    
    # 生成示例数据（模拟问卷数据，3个潜在因子）
    np.random.seed(42)
    n_samples = 200
    
    # 三个潜在因子
    F1 = np.random.randn(n_samples)  # 学习能力
    F2 = np.random.randn(n_samples)  # 社交能力
    F3 = np.random.randn(n_samples)  # 创造力
    
    # 观测变量（9个）
    X1 = 0.8*F1 + 0.2*np.random.randn(n_samples)  # 数学成绩
    X2 = 0.9*F1 + 0.1*np.random.randn(n_samples)  # 语文成绩
    X3 = 0.7*F1 + 0.3*np.random.randn(n_samples)  # 英语成绩
    
    X4 = 0.8*F2 + 0.2*np.random.randn(n_samples)  # 沟通能力
    X5 = 0.85*F2 + 0.15*np.random.randn(n_samples)  # 团队协作
    X6 = 0.75*F2 + 0.25*np.random.randn(n_samples)  # 领导力
    
    X7 = 0.9*F3 + 0.1*np.random.randn(n_samples)  # 创新思维
    X8 = 0.8*F3 + 0.2*np.random.randn(n_samples)  # 艺术感知
    X9 = 0.7*F3 + 0.3*np.random.randn(n_samples)  # 问题解决
    
    # 构建数据集
    data = pd.DataFrame({
        '数学成绩': X1,
        '语文成绩': X2,
        '英语成绩': X3,
        '沟通能力': X4,
        '团队协作': X5,
        '领导力': X6,
        '创新思维': X7,
        '艺术感知': X8,
        '问题解决': X9
    })
    
    print(f"\n数据集信息：")
    print(f"  - 样本数: {data.shape[0]}")
    print(f"  - 变量数: {data.shape[1]}")
    print(f"  - 变量: {list(data.columns)}")
    
    # 创建因子分析器
    print("\n" + "-" * 40)
    fa = FactorAnalyzer(n_factors=None, rotation='varimax', verbose=True)
    
    # 执行因子分析
    factor_scores = fa.fit_transform(data)
    
    # 绘制碎石图
    print("\n绘制碎石图...")
    fa.plot_scree()
    
    # 绘制因子载荷热力图
    print("\n绘制因子载荷热力图...")
    fa.plot_loadings_heatmap()
    
    # 绘制因子得分散点图
    print("\n绘制因子得分散点图...")
    fa.plot_factor_scores_2d(factor1=0, factor2=1)
    
    # 绘制双标图
    print("\n绘制双标图...")
    fa.plot_biplot(factor1=0, factor2=1)
    
    # 计算综合得分
    composite_scores = fa.compute_composite_score(method='variance')
    
    print("\n【综合因子得分（前10个样本）】")
    print(f"  权重: {fa.get_factor_weights('variance')}")
    for i in range(10):
        print(f"  样本 {i+1}: {composite_scores[i]:.4f}")
    
    print("\n" + "=" * 60)
    print("因子分析完成！")
    print("=" * 60)
