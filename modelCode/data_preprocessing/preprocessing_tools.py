"""
============================================================
数据预处理工具包 (Data Preprocessing Toolkit)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：数据清洗、标准化、异常值处理、缺失值填充
作者：MCM/ICM Team
日期：2026年1月
============================================================
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

warnings.filterwarnings('ignore')


class DataCleaner:
    """
    数据清洗工具类
    
    功能：
    - 缺失值检测与填充
    - 重复值处理
    - 数据类型转换
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.cleaning_report = {}
        
    def check_quality(self, df):
        """
        检查数据质量
        
        :param df: DataFrame
        :return: 质量报告字典
        """
        report = {
            'shape': df.shape,
            'missing_count': df.isnull().sum().to_dict(),
            'missing_percent': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'dtypes': df.dtypes.to_dict()
        }
        
        if self.verbose:
            print("\n" + "="*50)
            print("📋 数据质量检查报告")
            print("="*50)
            print(f"\n  数据维度: {report['shape'][0]} 行 × {report['shape'][1]} 列")
            print(f"  重复行数: {report['duplicates']}")
            print("\n  缺失值统计:")
            for col, cnt in report['missing_count'].items():
                if cnt > 0:
                    pct = report['missing_percent'][col]
                    print(f"    {col}: {cnt} ({pct:.1f}%)")
            print("="*50)
            
        self.cleaning_report = report
        return report
    
    def fill_missing(self, df, method='auto', columns=None):
        """
        填充缺失值
        
        :param df: DataFrame
        :param method: 'auto'/'mean'/'median'/'mode'/'ffill'/'bfill'/'knn'
        :param columns: 指定列，None表示所有列
        :return: 填充后的DataFrame
        """
        df_filled = df.copy()
        
        if columns is None:
            columns = df.columns[df.isnull().any()].tolist()
            
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'auto':
                # 数值型用中位数，分类型用众数
                if pd.api.types.is_numeric_dtype(df[col]):
                    df_filled[col].fillna(df[col].median(), inplace=True)
                else:
                    df_filled[col].fillna(df[col].mode()[0], inplace=True)
            elif method == 'mean':
                df_filled[col].fillna(df[col].mean(), inplace=True)
            elif method == 'median':
                df_filled[col].fillna(df[col].median(), inplace=True)
            elif method == 'mode':
                df_filled[col].fillna(df[col].mode()[0], inplace=True)
            elif method == 'ffill':
                df_filled[col].fillna(method='ffill', inplace=True)
            elif method == 'bfill':
                df_filled[col].fillna(method='bfill', inplace=True)
            elif method == 'knn':
                # KNN填充（仅数值列）
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if col in numeric_cols:
                    imputer = KNNImputer(n_neighbors=5)
                    df_filled[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    
        if self.verbose:
            remaining = df_filled.isnull().sum().sum()
            print(f"\n  ✅ 缺失值填充完成，剩余缺失: {remaining}")
            
        return df_filled
    
    def remove_duplicates(self, df, keep='first'):
        """移除重复行"""
        n_before = len(df)
        df_clean = df.drop_duplicates(keep=keep)
        n_removed = n_before - len(df_clean)
        
        if self.verbose:
            print(f"\n  ✅ 移除重复行: {n_removed} 行")
            
        return df_clean


class DataScaler:
    """
    数据标准化/归一化工具类
    
    支持方法：
    - standard: Z-score标准化 (x - mean) / std
    - minmax: 最小最大归一化 [0, 1]
    - robust: 鲁棒标准化（对异常值不敏感）
    """
    
    def __init__(self, method='standard', verbose=True):
        """
        :param method: 'standard' / 'minmax' / 'robust'
        """
        self.method = method
        self.verbose = verbose
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的方法: {method}")
            
        self.is_fitted = False
        self.feature_names = None
        
    def fit_transform(self, X):
        """拟合并转换"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        else:
            X_scaled = self.scaler.fit_transform(X)
            
        self.is_fitted = True
        
        if self.verbose:
            print(f"\n  ✅ 数据标准化完成 (方法: {self.method})")
            
        return X_scaled
    
    def transform(self, X):
        """转换新数据"""
        if not self.is_fitted:
            raise ValueError("请先调用 fit_transform()")
            
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler.transform(X)
            return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        return self.scaler.transform(X)
    
    def inverse_transform(self, X):
        """逆转换"""
        if isinstance(X, pd.DataFrame):
            X_inv = self.scaler.inverse_transform(X)
            return pd.DataFrame(X_inv, columns=self.feature_names, index=X.index)
        return self.scaler.inverse_transform(X)


class OutlierDetector:
    """
    异常值检测与处理类
    
    支持方法：
    - IQR: 四分位距法
    - zscore: Z分数法
    - isolation: 孤立森林（需要sklearn）
    """
    
    def __init__(self, method='iqr', threshold=1.5, verbose=True):
        """
        :param method: 'iqr' / 'zscore' / 'isolation'
        :param threshold: IQR的k值 或 zscore阈值
        """
        self.method = method
        self.threshold = threshold
        self.verbose = verbose
        self.outlier_info = {}
        
    def detect(self, data, column=None):
        """
        检测异常值
        
        :param data: DataFrame, Series 或 array
        :param column: 列名（DataFrame时使用）
        :return: 布尔掩码（True=异常值）
        """
        # 处理输入
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = data.columns[0]
            values = data[column].values
        elif isinstance(data, pd.Series):
            values = data.values
            column = data.name or 'value'
        else:
            values = np.array(data)
            column = 'value'
            
        # 检测
        if self.method == 'iqr':
            Q1, Q3 = np.percentile(values, [25, 75])
            IQR = Q3 - Q1
            lower = Q1 - self.threshold * IQR
            upper = Q3 + self.threshold * IQR
            mask = (values < lower) | (values > upper)
            self.outlier_info = {'Q1': Q1, 'Q3': Q3, 'IQR': IQR, 
                                'lower': lower, 'upper': upper}
                                
        elif self.method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(values))
            mask = z_scores > self.threshold
            self.outlier_info = {'threshold': self.threshold}
            
        elif self.method == 'isolation':
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso.fit_predict(values.reshape(-1, 1))
            mask = predictions == -1
            
        else:
            raise ValueError(f"不支持的方法: {self.method}")
            
        if self.verbose:
            n_outliers = mask.sum()
            pct = n_outliers / len(values) * 100
            print(f"\n  🔍 异常值检测 ({self.method}): 发现 {n_outliers} 个 ({pct:.1f}%)")
            
        return mask
    
    def remove(self, data, column=None):
        """移除异常值"""
        mask = self.detect(data, column)
        
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data[~mask]
        return data[~mask]
    
    def replace(self, data, column=None, method='median'):
        """
        替换异常值
        
        :param method: 'median' / 'mean' / 'clip'
        """
        mask = self.detect(data, column)
        
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            col = column or data.columns[0]
            values = df[col].values.copy()
        else:
            values = np.array(data).copy()
            
        if method == 'median':
            replacement = np.median(values[~mask])
            values[mask] = replacement
        elif method == 'mean':
            replacement = np.mean(values[~mask])
            values[mask] = replacement
        elif method == 'clip':
            if self.method == 'iqr':
                values = np.clip(values, self.outlier_info['lower'], self.outlier_info['upper'])
            else:
                values[mask] = np.median(values[~mask])
                
        if isinstance(data, pd.DataFrame):
            df[col] = values
            return df
        return values


class FeatureSelector:
    """
    特征选择工具类
    
    支持方法：
    - variance: 方差过滤
    - correlation: 相关性过滤
    - mutual_info: 互信息
    """
    
    def __init__(self, method='correlation', threshold=0.9, verbose=True):
        """
        :param method: 'variance' / 'correlation' / 'mutual_info'
        :param threshold: 过滤阈值
        """
        self.method = method
        self.threshold = threshold
        self.verbose = verbose
        self.selected_features = None
        self.dropped_features = None
        
    def fit_transform(self, X, y=None):
        """
        选择特征
        
        :param X: 特征DataFrame
        :param y: 标签（互信息时使用）
        :return: 筛选后的特征
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
            
        if self.method == 'variance':
            # 移除低方差特征
            variances = X.var()
            mask = variances > self.threshold
            self.selected_features = list(X.columns[mask])
            self.dropped_features = list(X.columns[~mask])
            
        elif self.method == 'correlation':
            # 移除高相关特征
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > self.threshold)]
            self.dropped_features = to_drop
            self.selected_features = [col for col in X.columns if col not in to_drop]
            
        elif self.method == 'mutual_info':
            if y is None:
                raise ValueError("互信息方法需要提供 y")
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            
            # 自动判断分类/回归
            if len(np.unique(y)) < 10:
                mi = mutual_info_classif(X, y)
            else:
                mi = mutual_info_regression(X, y)
                
            mask = mi > self.threshold
            self.selected_features = list(X.columns[mask])
            self.dropped_features = list(X.columns[~mask])
            
        if self.verbose:
            print(f"\n  ✅ 特征选择 ({self.method})")
            print(f"     保留特征: {len(self.selected_features)}")
            print(f"     移除特征: {len(self.dropped_features)}")
            if self.dropped_features:
                print(f"     移除列: {self.dropped_features[:5]}{'...' if len(self.dropped_features) > 5 else ''}")
                
        return X[self.selected_features]


# 便捷函数
def quick_preprocess(df, fill_missing='auto', scale='standard', remove_outliers=True):
    """
    快速数据预处理
    
    :param df: 原始DataFrame
    :param fill_missing: 缺失值填充方法
    :param scale: 标准化方法 ('standard'/'minmax'/None)
    :param remove_outliers: 是否移除异常值
    :return: 预处理后的DataFrame
    """
    print("\n" + "="*50)
    print("🔧 快速数据预处理")
    print("="*50)
    
    # 1. 数据清洗
    cleaner = DataCleaner(verbose=True)
    cleaner.check_quality(df)
    df = cleaner.fill_missing(df, method=fill_missing)
    df = cleaner.remove_duplicates(df)
    
    # 2. 异常值处理
    if remove_outliers:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        detector = OutlierDetector(method='iqr', verbose=False)
        for col in numeric_cols:
            df = detector.replace(df, column=col, method='clip')
        print(f"\n  ✅ 异常值处理完成")
    
    # 3. 标准化
    if scale:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaler = DataScaler(method=scale, verbose=True)
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
    print("\n" + "="*50)
    print("✅ 预处理完成!")
    print("="*50)
    
    return df


if __name__ == "__main__":
    # 演示
    print("="*60)
    print("📊 数据预处理工具演示")
    print("="*60)
    
    # 生成测试数据
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        '特征1': np.random.randn(n) * 10 + 50,
        '特征2': np.random.randn(n) * 5 + 30,
        '特征3': np.random.randn(n) * 15 + 100,
    })
    
    # 添加缺失值和异常值
    df.loc[5:10, '特征1'] = np.nan
    df.loc[0, '特征2'] = 200  # 异常值
    
    # 快速预处理
    df_clean = quick_preprocess(df)
    print("\n处理后数据预览:")
    print(df_clean.head())
