"""
分类模型模块 (Classification Module)
=====================================

包含多种分类算法的实现。

模块结构：
- decision_tree_classification.py: 决策树分类
- knn_classification.py: K近邻分类
- naive_bayes_classification.py: 朴素贝叶斯分类
- lda_classification.py: LDA线性判别分析（新增）

使用方法：
    from models.classification.lda_classification import LDAClassifier, compare_pca_lda
"""

from .lda_classification import LDAClassifier, compare_pca_lda

__all__ = [
    'LDAClassifier',
    'compare_pca_lda'
]
