"""
============================================================
数据建模完整教程 (Data Modeling Complete Tutorial)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
本教程展示如何串联：
1. 数据预处理 (Data Preprocessing)
2. 模型分析 (Model Analysis)  
3. 结果可视化 (Visualization)

包含三大核心技术：
- PCA/因子分析 (降维/结构发现)
- DBSCAN聚类 (无监督分类)
- LDA判别分析 (有监督分类)

作者：MCM/ICM Team
日期：2026年1月
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from data_preprocessing.pca_reduction import PCAReducer
from data_preprocessing.factor_analysis import FactorAnalyzer
from data_preprocessing.preprocessing_toolkit import DataPreprocessor
from models.clustering.dbscan_clustering import DBSCANClusterer
from models.classification.lda_classification import LDAClassifier, compare_pca_lda


class DataModelingPipeline:
    """
    数据建模完整流水线
    
    核心流程：
    ┌─────────────────────────────────────────────────────────┐
    │   原始数据                                               │
    │      ↓                                                  │
    │   Step 1: 数据预处理                                     │
    │      ├── 缺失值处理                                      │
    │      ├── 异常值检测                                      │
    │      └── 标准化                                         │
    │      ↓                                                  │
    │   Step 2: 探索性分析                                     │
    │      ├── 描述统计                                        │
    │      ├── 相关性分析                                      │
    │      └── 分布可视化                                      │
    │      ↓                                                  │
    │   Step 3: 特征工程/降维                                  │
    │      ├── PCA（数据压缩）                                 │
    │      ├── 因子分析（结构发现）                            │
    │      └── LDA（有监督降维）                               │
    │      ↓                                                  │
    │   Step 4: 模型分析                                       │
    │      ├── 聚类分析（无标签）                              │
    │      └── 分类分析（有标签）                              │
    │      ↓                                                  │
    │   Step 5: 结果可视化与评估                               │
    │      ├── 聚类/分类结果图                                 │
    │      ├── 评估指标                                        │
    │      └── 敏感性分析                                      │
    └─────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, verbose=True):
        """
        初始化流水线
        
        :param verbose: 是否打印详细信息
        """
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.preprocessor = None
        self.pca = None
        self.fa = None
        self.lda = None
        self.dbscan = None
        
        self.X_raw = None
        self.X_clean = None
        self.X_scaled = None
        self.y = None
        self.feature_names = None
        
        self.results = {}
    
    def load_data(self, X, y=None, feature_names=None):
        """
        加载数据
        
        :param X: 特征矩阵（DataFrame或数组）
        :param y: 标签向量（可选）
        :param feature_names: 特征名称（可选）
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            self.X_raw = X.values
        else:
            self.X_raw = X.copy()
            self.feature_names = feature_names or [f'X{i+1}' for i in range(X.shape[1])]
        
        if y is not None:
            if isinstance(y, pd.Series):
                self.y = y.values
            else:
                self.y = y.copy()
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("数据加载完成")
            print("=" * 60)
            print(f"  - 样本数: {self.X_raw.shape[0]}")
            print(f"  - 特征数: {self.X_raw.shape[1]}")
            print(f"  - 特征名: {self.feature_names}")
            if self.y is not None:
                print(f"  - 标签类别: {np.unique(self.y)}")
        
        return self
    
    def preprocess(self, handle_missing='mean', detect_outliers=True, 
                   outlier_method='iqr', standardize=True):
        """
        Step 1: 数据预处理
        
        :param handle_missing: 缺失值处理方法 ('mean', 'median', 'drop')
        :param detect_outliers: 是否检测异常值
        :param outlier_method: 异常值检测方法 ('iqr', '3sigma')
        :param standardize: 是否标准化
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("Step 1: 数据预处理")
            print("=" * 60)
        
        X = self.X_raw.copy()
        
        # 1.1 缺失值处理
        missing_count = np.isnan(X).sum()
        if missing_count > 0:
            if self.verbose:
                print(f"\n【缺失值处理】")
                print(f"  - 发现 {missing_count} 个缺失值")
            
            if handle_missing == 'mean':
                col_means = np.nanmean(X, axis=0)
                for j in range(X.shape[1]):
                    mask = np.isnan(X[:, j])
                    X[mask, j] = col_means[j]
            elif handle_missing == 'median':
                col_medians = np.nanmedian(X, axis=0)
                for j in range(X.shape[1]):
                    mask = np.isnan(X[:, j])
                    X[mask, j] = col_medians[j]
            elif handle_missing == 'drop':
                mask = ~np.any(np.isnan(X), axis=1)
                X = X[mask]
                if self.y is not None:
                    self.y = self.y[mask]
            
            if self.verbose:
                print(f"  - 处理方法: {handle_missing}")
                print(f"  - 处理后缺失值: {np.isnan(X).sum()}")
        else:
            if self.verbose:
                print(f"\n【缺失值检查】无缺失值")
        
        # 1.2 异常值检测
        if detect_outliers:
            if self.verbose:
                print(f"\n【异常值检测】")
                print(f"  - 检测方法: {outlier_method}")
            
            outlier_mask = np.zeros(X.shape[0], dtype=bool)
            
            for j in range(X.shape[1]):
                col = X[:, j]
                if outlier_method == 'iqr':
                    Q1, Q3 = np.percentile(col, [25, 75])
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    outlier_mask |= (col < lower) | (col > upper)
                elif outlier_method == '3sigma':
                    mean = np.mean(col)
                    std = np.std(col)
                    outlier_mask |= (np.abs(col - mean) > 3 * std)
            
            n_outliers = outlier_mask.sum()
            if self.verbose:
                print(f"  - 发现 {n_outliers} 个异常样本 ({100*n_outliers/len(X):.1f}%)")
                if n_outliers > 0:
                    print(f"  - 异常点索引: {np.where(outlier_mask)[0][:10]}...")
            
            self.results['outlier_mask'] = outlier_mask
            self.results['n_outliers'] = n_outliers
        
        self.X_clean = X
        
        # 1.3 标准化
        if standardize:
            self.X_scaled = self.scaler.fit_transform(X)
            if self.verbose:
                print(f"\n【数据标准化】")
                print(f"  - 方法: Z-score标准化")
                print(f"  - 标准化后均值: {np.mean(self.X_scaled, axis=0).round(4)}")
                print(f"  - 标准化后标准差: {np.std(self.X_scaled, axis=0).round(4)}")
        else:
            self.X_scaled = X
        
        return self
    
    def explore(self, plot_correlation=True, plot_distribution=True):
        """
        Step 2: 探索性数据分析
        
        :param plot_correlation: 是否绘制相关性热力图
        :param plot_distribution: 是否绘制分布图
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("Step 2: 探索性数据分析")
            print("=" * 60)
        
        X = self.X_clean
        
        # 2.1 描述统计
        if self.verbose:
            print(f"\n【描述统计】")
            stats_df = pd.DataFrame({
                '均值': np.mean(X, axis=0),
                '标准差': np.std(X, axis=0),
                '最小值': np.min(X, axis=0),
                '25%': np.percentile(X, 25, axis=0),
                '中位数': np.median(X, axis=0),
                '75%': np.percentile(X, 75, axis=0),
                '最大值': np.max(X, axis=0)
            }, index=self.feature_names)
            print(stats_df.round(4).to_string())
        
        # 2.2 相关性分析
        if plot_correlation:
            corr_matrix = np.corrcoef(X.T)
            self.results['correlation_matrix'] = corr_matrix
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            
            ax.set_xticks(range(len(self.feature_names)))
            ax.set_yticks(range(len(self.feature_names)))
            ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
            ax.set_yticklabels(self.feature_names)
            
            for i in range(len(self.feature_names)):
                for j in range(len(self.feature_names)):
                    color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                    ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                           ha='center', va='center', color=color)
            
            plt.colorbar(im, ax=ax, label='相关系数')
            ax.set_title('特征相关性热力图', fontsize=14)
            plt.tight_layout()
            plt.show()
        
        # 2.3 分布可视化
        if plot_distribution:
            n_features = len(self.feature_names)
            n_cols = min(4, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
            axes = np.array(axes).flatten() if n_features > 1 else [axes]
            
            for i, (name, ax) in enumerate(zip(self.feature_names, axes)):
                ax.hist(X[:, i], bins=30, color='steelblue', alpha=0.7, edgecolor='white')
                ax.set_xlabel(name)
                ax.set_ylabel('频数')
                ax.set_title(f'{name} 分布')
            
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            
            plt.suptitle('特征分布直方图', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.show()
        
        return self
    
    def reduce_dimensions(self, method='pca', n_components=None, **kwargs):
        """
        Step 3: 降维/特征提取
        
        :param method: 降维方法 ('pca', 'fa', 'lda')
        :param n_components: 保留的维度数
        :param kwargs: 其他参数
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"Step 3: 降维分析 - {method.upper()}")
            print("=" * 60)
        
        if method == 'pca':
            self.pca = PCAReducer(n_components=n_components, 
                                  variance_threshold=kwargs.get('variance_threshold', 0.85),
                                  verbose=self.verbose)
            X_reduced = self.pca.fit_transform(self.X_scaled)
            
            # 可视化
            self.pca.plot_variance()
            if X_reduced.shape[1] >= 2:
                self.pca.plot_2d(self.X_scaled, labels=self.y)
            
            self.results['pca'] = {
                'X_reduced': X_reduced,
                'explained_variance': self.pca.explained_variance,
                'n_components': self.pca.n_selected
            }
            
        elif method == 'fa':
            self.fa = FactorAnalyzer(n_factors=n_components,
                                     rotation=kwargs.get('rotation', 'varimax'),
                                     verbose=self.verbose)
            X_reduced = self.fa.fit_transform(pd.DataFrame(self.X_scaled, 
                                                           columns=self.feature_names))
            
            # 可视化
            self.fa.plot_scree()
            self.fa.plot_loadings_heatmap()
            if X_reduced.shape[1] >= 2:
                self.fa.plot_factor_scores_2d(labels=self.y)
            
            self.results['fa'] = {
                'X_reduced': X_reduced,
                'loadings': self.fa.loadings_,
                'n_factors': self.fa.n_selected_
            }
            
        elif method == 'lda':
            if self.y is None:
                raise ValueError("LDA需要标签信息，请提供y参数")
            
            self.lda = LDAClassifier(n_components=n_components, verbose=self.verbose)
            self.lda.fit(pd.DataFrame(self.X_scaled, columns=self.feature_names), self.y)
            X_reduced = self.lda.transform(self.X_scaled)
            
            # 可视化
            self.lda.plot_projection_2d(
                pd.DataFrame(self.X_scaled, columns=self.feature_names), 
                pd.Series(self.y)
            )
            
            self.results['lda'] = {
                'X_reduced': X_reduced,
                'explained_variance': self.lda.explained_variance_ratio_,
                'n_components': self.lda.n_components
            }
        
        return self
    
    def cluster(self, method='dbscan', use_reduced=True, **kwargs):
        """
        Step 4A: 聚类分析（无监督）
        
        :param method: 聚类方法 ('dbscan', 'kmeans')
        :param use_reduced: 是否使用降维后的数据
        :param kwargs: 其他参数
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"Step 4: 聚类分析 - {method.upper()}")
            print("=" * 60)
        
        # 选择数据
        if use_reduced and 'pca' in self.results:
            X_cluster = self.results['pca']['X_reduced']
        elif use_reduced and 'fa' in self.results:
            X_cluster = self.results['fa']['X_reduced']
        else:
            X_cluster = self.X_scaled
        
        if method == 'dbscan':
            self.dbscan = DBSCANClusterer(
                eps=kwargs.get('eps', None),
                min_samples=kwargs.get('min_samples', None),
                auto_tune=kwargs.get('auto_tune', True),
                verbose=self.verbose
            )
            labels = self.dbscan.fit_predict(X_cluster, standardize=False)
            
            # 可视化
            self.dbscan.plot_k_distance()
            self.dbscan.plot_clusters_2d(X_cluster, use_pca=(X_cluster.shape[1] > 2))
            
            self.results['clustering'] = {
                'labels': labels,
                'n_clusters': self.dbscan.n_clusters_,
                'n_noise': self.dbscan.n_noise_,
                'metrics': self.dbscan.metrics_
            }
        
        return self
    
    def classify(self, test_size=0.3, cv=5, **kwargs):
        """
        Step 4B: 分类分析（有监督）
        
        :param test_size: 测试集比例
        :param cv: 交叉验证折数
        """
        if self.y is None:
            raise ValueError("分类需要标签信息，请提供y参数")
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Step 4: 分类分析 - LDA")
            print("=" * 60)
        
        X_df = pd.DataFrame(self.X_scaled, columns=self.feature_names)
        y_series = pd.Series(self.y)
        
        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=test_size, random_state=42, stratify=y_series
        )
        
        # 训练模型
        self.lda = LDAClassifier(verbose=self.verbose)
        self.lda.fit(X_train, y_train)
        
        # 评估
        metrics = self.lda.evaluate(X_test, y_test)
        cv_scores = self.lda.cross_validate(X_df, y_series, cv=cv)
        
        # 可视化
        self.lda.plot_projection_2d(X_df, y_series)
        self.lda.plot_confusion_matrix(X_test, y_test)
        self.lda.plot_roc_curve(X_test, y_test)
        self.lda.plot_feature_importance()
        
        self.results['classification'] = {
            'accuracy': metrics['accuracy'],
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        return self
    
    def summary(self):
        """
        生成分析总结报告
        """
        print("\n" + "=" * 60)
        print("数据建模分析总结报告")
        print("=" * 60)
        
        print(f"\n【数据概览】")
        print(f"  - 原始样本数: {self.X_raw.shape[0]}")
        print(f"  - 特征数: {self.X_raw.shape[1]}")
        
        if 'n_outliers' in self.results:
            print(f"  - 异常值数量: {self.results['n_outliers']}")
        
        if 'pca' in self.results:
            print(f"\n【PCA降维】")
            print(f"  - 保留主成分数: {self.results['pca']['n_components']}")
            cum_var = np.sum(self.results['pca']['explained_variance'][:self.results['pca']['n_components']])
            print(f"  - 累计方差解释: {cum_var*100:.2f}%")
        
        if 'fa' in self.results:
            print(f"\n【因子分析】")
            print(f"  - 提取因子数: {self.results['fa']['n_factors']}")
        
        if 'clustering' in self.results:
            print(f"\n【聚类分析】")
            print(f"  - 发现簇数: {self.results['clustering']['n_clusters']}")
            print(f"  - 噪声点数: {self.results['clustering']['n_noise']}")
            if self.results['clustering']['metrics']:
                print(f"  - 轮廓系数: {self.results['clustering']['metrics'].get('silhouette', 'N/A'):.4f}")
        
        if 'classification' in self.results:
            print(f"\n【分类分析】")
            print(f"  - 测试集准确率: {self.results['classification']['accuracy']*100:.2f}%")
            print(f"  - 交叉验证: {self.results['classification']['cv_mean']*100:.2f}% ± {self.results['classification']['cv_std']*100:.2f}%")
        
        print("\n" + "=" * 60)
        
        return self.results


# ==================== 完整使用示例 ====================

def example_unsupervised():
    """
    无监督学习完整流程示例
    
    场景：对未知数据进行聚类分析
    流程：预处理 → PCA降维 → DBSCAN聚类
    """
    print("\n" + "=" * 60)
    print("示例1：无监督学习流程 (PCA + DBSCAN)")
    print("=" * 60)
    
    # 生成示例数据
    np.random.seed(42)
    
    # 三个簇
    cluster1 = np.random.randn(80, 5) * 0.8 + np.array([2, 2, 0, -1, 1])
    cluster2 = np.random.randn(100, 5) * 1.0 + np.array([-2, -2, 3, 2, -1])
    cluster3 = np.random.randn(60, 5) * 0.6 + np.array([0, 4, -2, 0, 3])
    noise = np.random.randn(20, 5) * 3  # 噪声
    
    X = np.vstack([cluster1, cluster2, cluster3, noise])
    
    # 添加一些缺失值
    X[10, 2] = np.nan
    X[50, 0] = np.nan
    
    feature_names = ['销售额', '利润率', '客户数', '投诉率', '满意度']
    
    # 创建流水线
    pipeline = DataModelingPipeline(verbose=True)
    
    # 执行完整流程
    (pipeline
     .load_data(X, feature_names=feature_names)
     .preprocess(handle_missing='mean', detect_outliers=True)
     .explore(plot_correlation=True, plot_distribution=True)
     .reduce_dimensions(method='pca', variance_threshold=0.85)
     .cluster(method='dbscan', use_reduced=True)
     .summary())
    
    return pipeline


def example_supervised():
    """
    有监督学习完整流程示例
    
    场景：使用LDA进行分类
    流程：预处理 → LDA降维+分类 → 评估
    """
    print("\n" + "=" * 60)
    print("示例2：有监督学习流程 (LDA分类)")
    print("=" * 60)
    
    # 加载鸢尾花数据集
    from sklearn.datasets import load_iris
    iris = load_iris()
    
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target_names[iris.target])
    
    # 创建流水线
    pipeline = DataModelingPipeline(verbose=True)
    
    # 执行完整流程
    (pipeline
     .load_data(X, y=y)
     .preprocess(detect_outliers=True)
     .explore(plot_correlation=True, plot_distribution=False)
     .reduce_dimensions(method='lda')
     .classify(test_size=0.3, cv=5)
     .summary())
    
    return pipeline


def example_factor_analysis():
    """
    因子分析示例
    
    场景：发现问卷数据的潜在结构
    流程：预处理 → 因子分析 → 综合评分
    """
    print("\n" + "=" * 60)
    print("示例3：因子分析流程")
    print("=" * 60)
    
    # 模拟问卷数据（9个题目，3个潜在因子）
    np.random.seed(42)
    n_samples = 150
    
    # 三个潜在因子
    F1 = np.random.randn(n_samples)  # 学习能力
    F2 = np.random.randn(n_samples)  # 社交能力
    F3 = np.random.randn(n_samples)  # 创造力
    
    # 观测变量
    data = pd.DataFrame({
        '数学': 0.85*F1 + 0.15*np.random.randn(n_samples),
        '语文': 0.80*F1 + 0.20*np.random.randn(n_samples),
        '英语': 0.75*F1 + 0.25*np.random.randn(n_samples),
        '沟通': 0.85*F2 + 0.15*np.random.randn(n_samples),
        '合作': 0.80*F2 + 0.20*np.random.randn(n_samples),
        '领导': 0.70*F2 + 0.30*np.random.randn(n_samples),
        '创新': 0.90*F3 + 0.10*np.random.randn(n_samples),
        '艺术': 0.75*F3 + 0.25*np.random.randn(n_samples),
        '思维': 0.80*F3 + 0.20*np.random.randn(n_samples),
    })
    
    # 创建流水线
    pipeline = DataModelingPipeline(verbose=True)
    
    # 执行因子分析流程
    (pipeline
     .load_data(data)
     .preprocess(detect_outliers=False)
     .reduce_dimensions(method='fa', rotation='varimax')
     .summary())
    
    # 计算综合得分
    if pipeline.fa is not None:
        composite_scores = pipeline.fa.compute_composite_score(method='variance')
        print("\n【综合因子得分（前10个样本）】")
        for i in range(10):
            print(f"  样本 {i+1}: {composite_scores[i]:.4f}")
    
    return pipeline


def quick_start_guide():
    """
    快速入门指南
    """
    guide = """
╔══════════════════════════════════════════════════════════════╗
║              数据建模快速入门指南                              ║
╚══════════════════════════════════════════════════════════════╝

【一、基本使用流程】

from data_modeling_tutorial import DataModelingPipeline

# 1. 创建流水线
pipeline = DataModelingPipeline(verbose=True)

# 2. 加载数据
pipeline.load_data(X, y=labels)  # y可选

# 3. 数据预处理
pipeline.preprocess(handle_missing='mean', detect_outliers=True)

# 4. 探索性分析
pipeline.explore(plot_correlation=True)

# 5. 降维分析
pipeline.reduce_dimensions(method='pca')  # 或 'fa', 'lda'

# 6. 聚类/分类
pipeline.cluster(method='dbscan')  # 无监督
# 或
pipeline.classify(test_size=0.3)   # 有监督

# 7. 生成报告
pipeline.summary()

--------------------------------------------------------------

【二、单独使用各模块】

# PCA降维
from data_preprocessing.pca_reduction import PCAReducer
pca = PCAReducer(variance_threshold=0.85)
X_pca = pca.fit_transform(X)
pca.plot_variance()
pca.plot_2d(X)

# 因子分析
from data_preprocessing.factor_analysis import FactorAnalyzer
fa = FactorAnalyzer(n_factors=3, rotation='varimax')
scores = fa.fit_transform(data)
fa.plot_loadings_heatmap()
fa.plot_biplot()

# DBSCAN聚类
from models.clustering.dbscan_clustering import DBSCANClusterer
dbscan = DBSCANClusterer(auto_tune=True)
labels = dbscan.fit_predict(X)
dbscan.plot_clusters_2d(X)

# LDA判别分析
from models.classification.lda_classification import LDAClassifier
lda = LDAClassifier()
lda.fit(X_train, y_train)
lda.evaluate(X_test, y_test)
lda.plot_projection_2d(X, y)

--------------------------------------------------------------

【三、场景选择指南】

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  有没有标签(y)?                                              │
│    ├─ 没有 → 无监督学习                                      │
│    │    ├─ 目标是降维? → PCA                                 │
│    │    ├─ 目标是发现结构? → 因子分析                        │
│    │    └─ 目标是分组? → DBSCAN/K-means                      │
│    │                                                        │
│    └─ 有 → 有监督学习                                        │
│         ├─ 目标是分类? → LDA                                 │
│         └─ 目标是降维+分类? → LDA                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘

--------------------------------------------------------------

【四、美赛常用组合】

1. 数据探索 + 降维:
   预处理 → 相关性分析 → PCA → 2D可视化

2. 聚类分析:
   预处理 → PCA → DBSCAN → 聚类结果图

3. 分类问题:
   预处理 → LDA → 分类评估 → ROC曲线

4. 综合评价:
   预处理 → 因子分析 → 综合得分 → 排名

--------------------------------------------------------------

【五、输出物清单】

可视化图表:
  - 相关性热力图
  - 特征分布图
  - PCA碎石图/方差解释图
  - 因子载荷热力图
  - 聚类结果图
  - LDA投影图
  - 混淆矩阵
  - ROC曲线
  - 特征重要性图

评估指标:
  - PCA: 方差解释率
  - DBSCAN: 轮廓系数、CH指数、DB指数
  - LDA: 准确率、交叉验证分数、AUC

╚══════════════════════════════════════════════════════════════╝
    """
    print(guide)


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 显示快速入门指南
    quick_start_guide()
    
    # 运行示例
    print("\n" + "=" * 60)
    print("运行完整示例...")
    print("=" * 60)
    
    # 示例1：无监督学习
    # pipeline1 = example_unsupervised()
    
    # 示例2：有监督学习
    pipeline2 = example_supervised()
    
    # 示例3：因子分析
    # pipeline3 = example_factor_analysis()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
