"""
============================================================
数据预处理与特征工程工具集 (Data Preprocessing & Feature Engineering)
包含：PCA降维 + IQR异常值检测 + 标准化 + 缺失值处理 + 特征选择
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：数据清洗、异常值处理、降维、特征工程
特点：完整的参数设置、可视化与美化、结果解释
作者：MCM/ICM Team
日期：2026年1月
============================================================

使用场景：
- 数据清洗与预处理
- 异常值检测与处理
- 降维与特征提取
- 数据质量评估
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

warnings.filterwarnings('ignore')


# ============================================================
# 第一部分：全局配置与美化设置 (Global Configuration)
# ============================================================

class PlotStyleConfig:
    """图表美化配置类"""
    
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'success': '#27AE60',
        'danger': '#C73E1D',
        'neutral': '#3B3B3B'
    }
    
    PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4C9A', '#1B998B']
    
    @staticmethod
    def setup_style():
        plt.style.use('seaborn-v0_8-whitegrid')
        rcParams['figure.figsize'] = (12, 8)
        rcParams['figure.dpi'] = 100
        rcParams['savefig.dpi'] = 300
        rcParams['font.size'] = 11
        rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False

PlotStyleConfig.setup_style()


# ============================================================
# 第二部分：样本数据生成器 (Sample Data Generator)
# ============================================================

class SampleDataGenerator:
    """样本数据生成器 - 用于测试和演示"""
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_with_missing_and_outliers(self, n_samples=200, 
                                            n_features=6, 
                                            missing_rate=0.1,
                                            outlier_rate=0.05):
        """
        生成带有缺失值和异常值的数据
        
        :param n_samples: 样本数量
        :param n_features: 特征数量
        :param missing_rate: 缺失率
        :param outlier_rate: 异常值比例
        """
        # 生成基础数据
        feature_names = [f'特征{i+1}' for i in range(n_features)]
        data = pd.DataFrame(
            np.random.randn(n_samples, n_features) * 10 + 50,
            columns=feature_names
        )
        
        # 添加缺失值
        n_missing = int(n_samples * n_features * missing_rate)
        for _ in range(n_missing):
            row = np.random.randint(0, n_samples)
            col = np.random.randint(0, n_features)
            data.iloc[row, col] = np.nan
        
        # 添加异常值
        n_outliers = int(n_samples * n_features * outlier_rate)
        for _ in range(n_outliers):
            row = np.random.randint(0, n_samples)
            col = np.random.randint(0, n_features)
            # 异常值为正常值的3-5倍
            data.iloc[row, col] = data.iloc[row, col] * np.random.choice([-1, 1]) * np.random.uniform(3, 5)
        
        return {
            'data': data,
            'feature_names': feature_names,
            'n_samples': n_samples,
            'n_features': n_features,
            'missing_rate': missing_rate,
            'outlier_rate': outlier_rate
        }
    
    def generate_high_dimensional(self, n_samples=300, n_features=20,
                                   n_informative=5, n_redundant=5):
        """
        生成高维数据（适合PCA降维）
        
        :param n_samples: 样本数
        :param n_features: 总特征数
        :param n_informative: 信息特征数
        :param n_redundant: 冗余特征数
        """
        # 生成信息特征
        informative = np.random.randn(n_samples, n_informative)
        
        # 生成冗余特征（信息特征的线性组合）
        redundant = np.zeros((n_samples, n_redundant))
        for i in range(n_redundant):
            weights = np.random.randn(n_informative)
            redundant[:, i] = informative @ weights + np.random.randn(n_samples) * 0.1
        
        # 生成噪声特征
        n_noise = n_features - n_informative - n_redundant
        noise = np.random.randn(n_samples, n_noise)
        
        # 合并
        data = np.hstack([informative, redundant, noise])
        
        feature_names = [f'特征{i+1}' for i in range(n_features)]
        df = pd.DataFrame(data, columns=feature_names)
        
        return {
            'data': df,
            'feature_names': feature_names,
            'n_informative': n_informative,
            'n_redundant': n_redundant,
            'n_noise': n_noise
        }


# ============================================================
# 第三部分：缺失值处理器 (Missing Value Handler)
# ============================================================

class MissingValueHandler:
    """
    缺失值处理工具
    
    方法：
    - 删除法：删除含缺失值的行/列
    - 填充法：均值/中位数/众数/常数填充
    - 插值法：KNN插补
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.original_missing_info = None
        self.handled_data = None
    
    def analyze_missing(self, data):
        """分析缺失值情况"""
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        missing_count = data.isnull().sum()
        missing_rate = (data.isnull().sum() / len(data) * 100).round(2)
        
        self.original_missing_info = pd.DataFrame({
            '缺失数量': missing_count,
            '缺失率(%)': missing_rate
        })
        
        total_missing = data.isnull().sum().sum()
        total_cells = data.shape[0] * data.shape[1]
        
        if self.verbose:
            print("\n" + "="*60)
            print("📊 缺失值分析报告")
            print("="*60)
            print(f"  数据形状: {data.shape}")
            print(f"  总缺失值: {total_missing} / {total_cells} ({total_missing/total_cells*100:.2f}%)")
            print(f"\n  各列缺失情况:")
            print(self.original_missing_info[self.original_missing_info['缺失数量'] > 0])
            print("="*60)
        
        return self.original_missing_info
    
    def fill_missing(self, data, method='mean', constant=0, n_neighbors=5):
        """
        填充缺失值
        
        :param data: 数据
        :param method: 填充方法
            - 'mean': 均值填充
            - 'median': 中位数填充
            - 'mode': 众数填充
            - 'constant': 常数填充
            - 'knn': KNN插补
        :param constant: 常数填充的值
        :param n_neighbors: KNN的近邻数
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        data = data.copy()
        
        if method == 'mean':
            imputer = SimpleImputer(strategy='mean')
            filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        
        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
            filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        
        elif method == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
            filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        
        elif method == 'constant':
            imputer = SimpleImputer(strategy='constant', fill_value=constant)
            filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=n_neighbors)
            filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        
        else:
            raise ValueError(f"不支持的填充方法: {method}")
        
        self.handled_data = filled
        
        if self.verbose:
            print(f"\n✅ 缺失值填充完成 (方法: {method})")
            print(f"  填充前缺失: {data.isnull().sum().sum()}")
            print(f"  填充后缺失: {filled.isnull().sum().sum()}")
        
        return filled
    
    def drop_missing(self, data, axis=0, thresh=None):
        """
        删除含缺失值的行/列
        
        :param axis: 0删除行，1删除列
        :param thresh: 非缺失值的最小数量
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        original_shape = data.shape
        
        if thresh is not None:
            cleaned = data.dropna(axis=axis, thresh=thresh)
        else:
            cleaned = data.dropna(axis=axis)
        
        self.handled_data = cleaned
        
        if self.verbose:
            if axis == 0:
                print(f"\n✅ 删除含缺失值的行")
                print(f"  删除前: {original_shape[0]} 行")
                print(f"  删除后: {cleaned.shape[0]} 行")
            else:
                print(f"\n✅ 删除含缺失值的列")
                print(f"  删除前: {original_shape[1]} 列")
                print(f"  删除后: {cleaned.shape[1]} 列")
        
        return cleaned


# ============================================================
# 第四部分：异常值检测器 (Outlier Detector)
# ============================================================

class OutlierDetector:
    """
    异常值检测与处理工具
    
    方法：
    - IQR方法：四分位数间距
    - Z-score方法：标准化得分
    - 箱线图法：可视化检测
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.outlier_info = {}
        self.outlier_mask = None
    
    def detect_iqr(self, data, column=None, factor=1.5):
        """
        IQR方法检测异常值
        
        IQR = Q3 - Q1
        异常值: < Q1 - factor*IQR 或 > Q3 + factor*IQR
        
        :param data: 数据
        :param column: 指定列（None则检测所有列）
        :param factor: IQR倍数（通常1.5为轻度异常，3为极度异常）
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        if column is not None:
            columns = [column]
        else:
            columns = data.select_dtypes(include=[np.number]).columns
        
        outlier_mask = pd.DataFrame(False, index=data.index, columns=columns)
        self.outlier_info = {}
        
        for col in columns:
            values = data[col].dropna()
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            is_outlier = (data[col] < lower_bound) | (data[col] > upper_bound)
            outlier_mask[col] = is_outlier
            
            n_outliers = is_outlier.sum()
            
            self.outlier_info[col] = {
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'n_outliers': n_outliers,
                'outlier_rate': n_outliers / len(data) * 100
            }
        
        self.outlier_mask = outlier_mask
        
        if self.verbose:
            self._print_iqr_results()
        
        return outlier_mask
    
    def _print_iqr_results(self):
        """打印IQR检测结果"""
        print("\n" + "="*70)
        print("🔍 IQR异常值检测结果")
        print("="*70)
        
        for col, info in self.outlier_info.items():
            if info['n_outliers'] > 0:
                print(f"\n  📌 {col}:")
                print(f"      Q1={info['Q1']:.2f}, Q3={info['Q3']:.2f}, IQR={info['IQR']:.2f}")
                print(f"      正常范围: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
                print(f"      异常值: {info['n_outliers']} 个 ({info['outlier_rate']:.2f}%)")
        
        total_outliers = self.outlier_mask.sum().sum()
        print(f"\n  总异常值数量: {total_outliers}")
        print("="*70)
    
    def detect_zscore(self, data, column=None, threshold=3.0):
        """
        Z-score方法检测异常值
        
        Z = (x - mean) / std
        异常值: |Z| > threshold
        
        :param threshold: Z分数阈值（通常2或3）
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        if column is not None:
            columns = [column]
        else:
            columns = data.select_dtypes(include=[np.number]).columns
        
        outlier_mask = pd.DataFrame(False, index=data.index, columns=columns)
        
        for col in columns:
            values = data[col].dropna()
            mean = values.mean()
            std = values.std()
            
            if std == 0:
                continue
            
            z_scores = (data[col] - mean) / std
            is_outlier = np.abs(z_scores) > threshold
            outlier_mask[col] = is_outlier
        
        self.outlier_mask = outlier_mask
        
        if self.verbose:
            total = outlier_mask.sum().sum()
            print(f"\n✅ Z-score检测完成 (阈值={threshold})")
            print(f"  总异常值: {total}")
        
        return outlier_mask
    
    def handle_outliers(self, data, method='clip', outlier_mask=None):
        """
        处理异常值
        
        :param method:
            - 'remove': 删除异常值所在行
            - 'clip': 截断到边界值
            - 'replace_mean': 替换为均值
            - 'replace_median': 替换为中位数
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        if outlier_mask is None:
            outlier_mask = self.outlier_mask
        
        data = data.copy()
        
        if method == 'remove':
            # 删除任意列有异常值的行
            mask = outlier_mask.any(axis=1)
            handled = data[~mask]
            
        elif method == 'clip':
            handled = data.copy()
            for col in outlier_mask.columns:
                if col in self.outlier_info:
                    lower = self.outlier_info[col]['lower_bound']
                    upper = self.outlier_info[col]['upper_bound']
                    handled[col] = handled[col].clip(lower, upper)
        
        elif method == 'replace_mean':
            handled = data.copy()
            for col in outlier_mask.columns:
                mean_val = data.loc[~outlier_mask[col], col].mean()
                handled.loc[outlier_mask[col], col] = mean_val
        
        elif method == 'replace_median':
            handled = data.copy()
            for col in outlier_mask.columns:
                median_val = data.loc[~outlier_mask[col], col].median()
                handled.loc[outlier_mask[col], col] = median_val
        
        else:
            raise ValueError(f"不支持的处理方法: {method}")
        
        if self.verbose:
            print(f"\n✅ 异常值处理完成 (方法: {method})")
            print(f"  处理前行数: {len(data)}")
            print(f"  处理后行数: {len(handled)}")
        
        return handled


# ============================================================
# 第五部分：数据标准化器 (Data Scaler)
# ============================================================

class DataScaler:
    """
    数据标准化工具
    
    方法：
    - Z-score标准化
    - Min-Max标准化
    - Robust标准化
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.scaler = None
        self.original_stats = None
    
    def fit_transform(self, data, method='standard'):
        """
        标准化数据
        
        :param method:
            - 'standard': Z-score标准化 (x-mean)/std
            - 'minmax': Min-Max标准化 到[0,1]
            - 'robust': 鲁棒标准化（对异常值不敏感）
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        # 记录原始统计信息
        self.original_stats = {
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max()
        }
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
        
        scaled = pd.DataFrame(
            self.scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
        
        if self.verbose:
            print(f"\n✅ 数据标准化完成 (方法: {method})")
            print(f"  标准化前均值: {data.mean().mean():.4f}")
            print(f"  标准化后均值: {scaled.mean().mean():.4f}")
        
        return scaled
    
    def inverse_transform(self, data):
        """逆变换"""
        if self.scaler is None:
            raise ValueError("请先调用fit_transform")
        
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        original = pd.DataFrame(
            self.scaler.inverse_transform(data),
            columns=data.columns,
            index=data.index
        )
        
        return original


# ============================================================
# 第六部分：PCA降维工具 (PCA Dimensionality Reduction)
# ============================================================

class PCAReducer:
    """
    PCA主成分分析降维
    
    原理：
    通过正交变换将相关变量转换为线性不相关的主成分
    按方差大小排序，保留主要信息
    
    应用：
    - 高维数据可视化
    - 特征压缩
    - 去除噪声
    """
    
    def __init__(self, n_components=None, verbose=True):
        """
        参数配置
        
        :param n_components:
            - int: 保留的主成分数量
            - float (0-1): 保留的方差比例
            - None: 保留所有主成分
        """
        self.n_components = n_components
        self.verbose = verbose
        self.pca = None
        self.scaler = None
        self.explained_variance_ratio = None
        self.components = None
        self.feature_names = None
    
    def fit_transform(self, data, scale=True):
        """
        执行PCA降维
        
        :param data: 原始数据
        :param scale: 是否先标准化
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        self.feature_names = list(data.columns)
        
        # 标准化
        if scale:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(data)
        else:
            scaled_data = data.values
        
        # PCA
        self.pca = PCA(n_components=self.n_components)
        transformed = self.pca.fit_transform(scaled_data)
        
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.components = self.pca.components_
        
        # 创建结果DataFrame
        n_pcs = transformed.shape[1]
        pc_names = [f'PC{i+1}' for i in range(n_pcs)]
        result = pd.DataFrame(transformed, columns=pc_names, index=data.index)
        
        if self.verbose:
            self._print_results()
        
        return result
    
    def _print_results(self):
        """打印PCA结果"""
        print("\n" + "="*60)
        print("📊 PCA降维分析结果")
        print("="*60)
        print(f"  原始特征数: {len(self.feature_names)}")
        print(f"  保留主成分数: {len(self.explained_variance_ratio)}")
        print(f"\n  各主成分方差解释率:")
        
        cumulative = 0
        for i, ratio in enumerate(self.explained_variance_ratio):
            cumulative += ratio
            print(f"    PC{i+1}: {ratio*100:6.2f}% (累计: {cumulative*100:6.2f}%)")
        
        print(f"\n  总方差解释率: {self.explained_variance_ratio.sum()*100:.2f}%")
        print("="*60)
    
    def get_loadings(self):
        """获取载荷矩阵（主成分与原始变量的相关系数）"""
        if self.pca is None:
            raise ValueError("请先调用fit_transform")
        
        loadings = pd.DataFrame(
            self.components.T,
            index=self.feature_names,
            columns=[f'PC{i+1}' for i in range(len(self.explained_variance_ratio))]
        )
        
        return loadings
    
    def select_n_components(self, data, target_variance=0.95, scale=True):
        """
        自动选择主成分数量
        
        :param target_variance: 目标方差解释率
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        if scale:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)
        else:
            scaled = data.values
        
        pca_full = PCA()
        pca_full.fit(scaled)
        
        cumulative = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumulative >= target_variance) + 1
        
        if self.verbose:
            print(f"\n✅ 自动选择主成分数量")
            print(f"  目标方差解释率: {target_variance*100}%")
            print(f"  推荐主成分数: {n_components}")
            print(f"  实际方差解释率: {cumulative[n_components-1]*100:.2f}%")
        
        return n_components, cumulative


# ============================================================
# 第七部分：特征选择器 (Feature Selector)
# ============================================================

class FeatureSelector:
    """
    特征选择工具
    
    方法：
    - 方差阈值
    - 相关性过滤
    - 统计检验（F检验、互信息）
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.selected_features = None
        self.feature_scores = None
    
    def select_by_variance(self, data, threshold=0.01):
        """
        方差阈值选择
        低方差特征通常信息量小
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        variances = data.var()
        selected = variances[variances > threshold].index.tolist()
        dropped = variances[variances <= threshold].index.tolist()
        
        self.selected_features = selected
        
        if self.verbose:
            print(f"\n✅ 方差阈值特征选择 (阈值={threshold})")
            print(f"  保留特征: {len(selected)}")
            print(f"  删除特征: {len(dropped)}")
            if dropped:
                print(f"  删除的低方差特征: {dropped}")
        
        return data[selected]
    
    def select_by_correlation(self, data, threshold=0.9):
        """
        相关性过滤
        删除高度相关的冗余特征
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        corr_matrix = data.corr().abs()
        
        # 上三角矩阵
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # 找出高相关特征
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        selected = [col for col in data.columns if col not in to_drop]
        self.selected_features = selected
        
        if self.verbose:
            print(f"\n✅ 相关性过滤 (阈值={threshold})")
            print(f"  保留特征: {len(selected)}")
            print(f"  删除特征: {len(to_drop)}")
            if to_drop:
                print(f"  删除的冗余特征: {to_drop}")
        
        return data[selected]
    
    def select_k_best(self, X, y, k=5, method='f_classif'):
        """
        统计检验选择Top K特征
        
        :param method:
            - 'f_classif': F检验（分类问题）
            - 'mutual_info': 互信息（分类问题）
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        if method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        selector.fit(X, y)
        
        scores = pd.Series(selector.scores_, index=X.columns)
        self.feature_scores = scores.sort_values(ascending=False)
        
        mask = selector.get_support()
        selected = X.columns[mask].tolist()
        self.selected_features = selected
        
        if self.verbose:
            print(f"\n✅ SelectKBest特征选择 (k={k}, 方法={method})")
            print(f"  选中特征: {selected}")
            print(f"\n  特征评分:")
            for name, score in self.feature_scores.head(10).items():
                print(f"    {name}: {score:.4f}")
        
        return X[selected], self.feature_scores


# ============================================================
# 第八部分：可视化模块 (Visualization)
# ============================================================

class PreprocessingVisualizer:
    """数据预处理可视化类"""
    
    def __init__(self):
        self.colors = PlotStyleConfig.PALETTE
    
    def plot_missing_heatmap(self, data, title="缺失值分布热力图", save_path=None):
        """绘制缺失值热力图"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        missing = data.isnull().astype(int)
        im = ax.imshow(missing.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        ax.set_yticks(range(len(data.columns)))
        ax.set_yticklabels(data.columns)
        ax.set_xlabel('样本索引', fontweight='bold')
        ax.set_ylabel('特征', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['非缺失', '缺失'])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_boxplot(self, data, title="箱线图 - 异常值可视化", save_path=None):
        """绘制箱线图"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data.boxplot(ax=ax, patch_artist=True,
                    boxprops=dict(facecolor=self.colors[0], alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=5))
        
        ax.set_xlabel('特征', fontweight='bold')
        ax.set_ylabel('值', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pca_variance(self, explained_variance_ratio, 
                          title="PCA方差解释率", save_path=None):
        """绘制PCA方差解释图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        n_components = len(explained_variance_ratio)
        x = range(1, n_components + 1)
        cumulative = np.cumsum(explained_variance_ratio)
        
        # 单独方差
        ax1 = axes[0]
        bars = ax1.bar(x, explained_variance_ratio * 100, 
                      color=self.colors[0], edgecolor='white', linewidth=2)
        ax1.set_xlabel('主成分', fontweight='bold')
        ax1.set_ylabel('方差解释率 (%)', fontweight='bold')
        ax1.set_title('(a) 各主成分方差解释率', fontweight='bold')
        ax1.set_xticks(x)
        
        for bar, val in zip(bars, explained_variance_ratio):
            if val > 0.02:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val*100:.1f}%', ha='center', fontsize=9)
        
        # 累计方差
        ax2 = axes[1]
        ax2.plot(x, cumulative * 100, 'o-', color=self.colors[1], 
                linewidth=2.5, markersize=8)
        ax2.axhline(y=95, color='red', linestyle='--', label='95%阈值')
        ax2.axhline(y=90, color='orange', linestyle='--', label='90%阈值')
        ax2.fill_between(x, cumulative * 100, alpha=0.3, color=self.colors[1])
        ax2.set_xlabel('主成分数量', fontweight='bold')
        ax2.set_ylabel('累计方差解释率 (%)', fontweight='bold')
        ax2.set_title('(b) 累计方差解释率', fontweight='bold')
        ax2.set_xticks(x)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pca_scatter(self, pca_result, labels=None, 
                         title="PCA降维散点图", save_path=None):
        """绘制PCA降维后的散点图（2D或3D）"""
        n_components = min(pca_result.shape[1], 3)
        
        if n_components == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if labels is not None:
                unique_labels = np.unique(labels)
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(pca_result.iloc[mask, 0], pca_result.iloc[mask, 1],
                              s=50, alpha=0.7, c=self.colors[i % len(self.colors)],
                              label=f'类别 {label}', edgecolors='white')
                ax.legend()
            else:
                ax.scatter(pca_result.iloc[:, 0], pca_result.iloc[:, 1],
                          s=50, alpha=0.7, c=self.colors[0], edgecolors='white')
            
            ax.set_xlabel('PC1', fontweight='bold')
            ax.set_ylabel('PC2', fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        elif n_components >= 3:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            if labels is not None:
                unique_labels = np.unique(labels)
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(pca_result.iloc[mask, 0], 
                              pca_result.iloc[mask, 1],
                              pca_result.iloc[mask, 2],
                              s=50, alpha=0.7, c=self.colors[i % len(self.colors)],
                              label=f'类别 {label}')
                ax.legend()
            else:
                ax.scatter(pca_result.iloc[:, 0], 
                          pca_result.iloc[:, 1],
                          pca_result.iloc[:, 2],
                          s=50, alpha=0.7, c=self.colors[0])
            
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, data, title="特征相关性热力图", save_path=None):
        """绘制相关性热力图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr = data.corr()
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)
        
        # 添加数值
        for i in range(len(corr)):
            for j in range(len(corr)):
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                              ha='center', va='center',
                              color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black',
                              fontsize=8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='相关系数')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 第九部分：主程序与完整示例 (Main Program)
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("   DATA PREPROCESSING & FEATURE ENGINEERING FOR MCM/ICM")
    print("   数据预处理与特征工程工具集")
    print("   Extended Version with Visualization")
    print("="*70)
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    🔧 数据预处理流程                              ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║   [缺失值处理]                                                    ║
    ║      ├─ 均值/中位数/众数填充                                      ║
    ║      ├─ KNN插补                                                  ║
    ║      └─ 删除法                                                   ║
    ║                                                                  ║
    ║   [异常值检测]                                                    ║
    ║      ├─ IQR方法: 四分位距                                        ║
    ║      ├─ Z-score方法: 标准化得分                                  ║
    ║      └─ 处理: 删除/截断/替换                                      ║
    ║                                                                  ║
    ║   [标准化]                                                        ║
    ║      ├─ Z-score: (x-mean)/std                                    ║
    ║      ├─ Min-Max: 缩放到[0,1]                                     ║
    ║      └─ Robust: 鲁棒标准化                                       ║
    ║                                                                  ║
    ║   [降维]                                                          ║
    ║      └─ PCA: 主成分分析                                          ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    generator = SampleDataGenerator(random_seed=2026)
    visualizer = PreprocessingVisualizer()
    
    # ================================================================
    # 示例1：缺失值处理
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 1: 缺失值检测与处理")
    print("="*70)
    
    data_info = generator.generate_with_missing_and_outliers(
        n_samples=200, n_features=6, missing_rate=0.1, outlier_rate=0.05
    )
    data = data_info['data']
    
    print(f"\n生成数据:")
    print(f"  样本数: {data_info['n_samples']}")
    print(f"  特征数: {data_info['n_features']}")
    
    handler = MissingValueHandler(verbose=True)
    handler.analyze_missing(data)
    
    visualizer.plot_missing_heatmap(data, title="缺失值分布热力图")
    
    # 填充缺失值
    filled_data = handler.fill_missing(data, method='knn', n_neighbors=5)
    
    # ================================================================
    # 示例2：异常值检测
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 2: 异常值检测与处理")
    print("="*70)
    
    detector = OutlierDetector(verbose=True)
    outlier_mask = detector.detect_iqr(filled_data, factor=1.5)
    
    visualizer.plot_boxplot(filled_data, title="异常值检测箱线图")
    
    # 处理异常值
    cleaned_data = detector.handle_outliers(filled_data, method='clip')
    
    visualizer.plot_boxplot(cleaned_data, title="处理后的数据箱线图")
    
    # ================================================================
    # 示例3：数据标准化
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 3: 数据标准化")
    print("="*70)
    
    scaler = DataScaler(verbose=True)
    
    # Z-score标准化
    scaled_standard = scaler.fit_transform(cleaned_data, method='standard')
    print(f"\nZ-score标准化后统计:")
    print(f"  均值范围: [{scaled_standard.mean().min():.4f}, {scaled_standard.mean().max():.4f}]")
    print(f"  标准差范围: [{scaled_standard.std().min():.4f}, {scaled_standard.std().max():.4f}]")
    
    # Min-Max标准化
    scaled_minmax = scaler.fit_transform(cleaned_data, method='minmax')
    print(f"\nMin-Max标准化后范围:")
    print(f"  最小值: {scaled_minmax.min().min():.4f}")
    print(f"  最大值: {scaled_minmax.max().max():.4f}")
    
    # ================================================================
    # 示例4：PCA降维
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 4: PCA主成分分析")
    print("="*70)
    
    # 生成高维数据
    high_dim_data = generator.generate_high_dimensional(
        n_samples=300, n_features=20, n_informative=5, n_redundant=5
    )
    
    print(f"\n高维数据:")
    print(f"  样本数: 300")
    print(f"  特征数: 20 (信息特征5, 冗余特征5, 噪声10)")
    
    # 自动选择主成分数
    pca = PCAReducer(verbose=True)
    n_optimal, cumulative = pca.select_n_components(
        high_dim_data['data'], target_variance=0.95
    )
    
    # 执行PCA
    pca = PCAReducer(n_components=n_optimal, verbose=True)
    pca_result = pca.fit_transform(high_dim_data['data'])
    
    visualizer.plot_pca_variance(
        pca.explained_variance_ratio,
        title="PCA方差解释率分析"
    )
    
    visualizer.plot_pca_scatter(
        pca_result,
        title="PCA降维结果 (2D)"
    )
    
    # 载荷矩阵
    loadings = pca.get_loadings()
    print("\n主成分载荷矩阵 (前5特征):")
    print(loadings.head())
    
    # ================================================================
    # 示例5：特征选择
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 5: 特征选择")
    print("="*70)
    
    # 相关性热力图
    visualizer.plot_correlation_heatmap(
        cleaned_data,
        title="特征相关性热力图"
    )
    
    selector = FeatureSelector(verbose=True)
    
    # 方差阈值
    selected_var = selector.select_by_variance(cleaned_data, threshold=1.0)
    
    # 相关性过滤
    selected_corr = selector.select_by_correlation(cleaned_data, threshold=0.8)
    
    # ================================================================
    # 使用说明
    # ================================================================
    print("\n" + "="*70)
    print("📖 使用说明 (Usage Guide)")
    print("="*70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                   数据预处理工具使用指南                         │
    └─────────────────────────────────────────────────────────────────┘
    
    【缺失值处理】
    
    handler = MissingValueHandler()
    handler.analyze_missing(data)               # 分析
    filled = handler.fill_missing(data, 'knn')  # KNN填充
    cleaned = handler.drop_missing(data)        # 删除
    
    【异常值处理】
    
    detector = OutlierDetector()
    mask = detector.detect_iqr(data, factor=1.5)   # IQR检测
    mask = detector.detect_zscore(data, threshold=3)  # Z-score检测
    
    cleaned = detector.handle_outliers(data, 'clip')  # 截断处理
    cleaned = detector.handle_outliers(data, 'remove')  # 删除处理
    
    【数据标准化】
    
    scaler = DataScaler()
    scaled = scaler.fit_transform(data, 'standard')  # Z-score
    scaled = scaler.fit_transform(data, 'minmax')    # Min-Max
    scaled = scaler.fit_transform(data, 'robust')    # Robust
    
    【PCA降维】
    
    pca = PCAReducer(n_components=0.95)  # 保留95%方差
    pca = PCAReducer(n_components=3)     # 保留3个主成分
    result = pca.fit_transform(data)
    loadings = pca.get_loadings()        # 获取载荷矩阵
    
    【特征选择】
    
    selector = FeatureSelector()
    selected = selector.select_by_variance(data, threshold=0.01)
    selected = selector.select_by_correlation(data, threshold=0.9)
    selected, scores = selector.select_k_best(X, y, k=5)
    
    【论文图表建议】
    
    Figure 1: 缺失值分布热力图
    Figure 2: 箱线图（异常值）
    Figure 3: PCA方差解释率图
    Figure 4: PCA降维散点图
    Figure 5: 特征相关性热力图
    
    Table 1: 数据基本信息
    Table 2: 缺失值统计
    Table 3: PCA主成分贡献率
    """)
    
    print("\n" + "="*70)
    print("   ✅ All examples completed successfully!")
    print("   💡 Use the above code templates for your MCM/ICM paper")
    print("="*70)
