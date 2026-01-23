"""
数据预处理模块 (Data Preprocessing Module)
==========================================

包含数据清洗、标准化、降维等预处理工具。

模块结构：
- preprocessing_tools.py: 数据清洗、标准化、异常值处理
- pca_reduction.py: PCA主成分分析降维
- factor_analysis.py: 因子分析（新增）

使用方法：
    from data_preprocessing.preprocessing_tools import DataCleaner, DataScaler, OutlierDetector
    from data_preprocessing.pca_reduction import PCAReducer
    from data_preprocessing.factor_analysis import FactorAnalyzer
"""

from .preprocessing_tools import (
    DataCleaner, 
    DataScaler, 
    OutlierDetector, 
    FeatureSelector,
    quick_preprocess
)
from .pca_reduction import PCAReducer
from .factor_analysis import FactorAnalyzer

__all__ = [
    'DataCleaner',
    'DataScaler', 
    'OutlierDetector',
    'FeatureSelector',
    'quick_preprocess',
    'PCAReducer',
    'FactorAnalyzer'
]
