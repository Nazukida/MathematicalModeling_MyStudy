"""
============================================================
综合评价模型模块 (Evaluation Models Module)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================

包含模型：
1. evaluation_toolkit.py - 熵权法、CRITIC法、TOPSIS法
   - EntropyWeightMethod: 熵权法（客观赋权）
   - CRITICMethod: CRITIC法（考虑相关性的客观赋权）
   - TOPSIS: 逼近理想解排序法
   - EvaluationDataPreprocessor: 评价数据预处理器
   - EvaluationVisualizer: 评价可视化器
   - SensitivityAnalyzer: 敏感性分析器
   - CombinedEvaluation: 组合评价模型

2. evaluation_methods.py - 其他评价方法
   - AHP: 层次分析法（主观赋权）
   - DEA: 数据包络分析
   - FuzzyComprehensiveEvaluation: 模糊综合评价
   - GreyRelationalAnalysis: 灰色关联分析

3. comprehensive_evaluation_tutorial.py - 完整教程
   - DataPreprocessor: 数据预处理类
   - EntropyWeight: 熵权法（简化版）
   - CRITIC: CRITIC法（简化版）
   - CombinedWeight: 组合赋权法
   - TOPSIS: TOPSIS法（简化版）
   - GreyRelationalAnalysis: 灰色关联分析
   - EvaluationVisualizer: 可视化器
   - run_complete_example: 完整案例演示

使用方法：
    from models.evaluation import (
        EntropyWeightMethod, CRITICMethod, TOPSIS,
        AHP, GreyRelationalAnalysis, EvaluationVisualizer
    )
    
    # 或者直接导入教程模块运行案例
    from models.evaluation.comprehensive_evaluation_tutorial import run_complete_example
    results = run_complete_example()

作者：MCM/ICM Team
日期：2026年1月20日
"""

# 从 evaluation_toolkit 导入
from .evaluation_toolkit import (
    EntropyWeightMethod,
    CRITICMethod,
    TOPSIS,
    EvaluationDataPreprocessor,
    EvaluationVisualizer,
    SensitivityAnalyzer,
    CombinedEvaluation,
    PlotStyleConfig
)

# 从 evaluation_methods 导入
from .evaluation_methods import (
    AHP,
    DEA,
    FuzzyComprehensiveEvaluation,
    GreyRelationalAnalysis
)

__all__ = [
    # evaluation_toolkit
    'EntropyWeightMethod',
    'CRITICMethod',
    'TOPSIS',
    'EvaluationDataPreprocessor',
    'EvaluationVisualizer',
    'SensitivityAnalyzer',
    'CombinedEvaluation',
    'PlotStyleConfig',
    # evaluation_methods
    'AHP',
    'DEA',
    'FuzzyComprehensiveEvaluation',
    'GreyRelationalAnalysis',
]
