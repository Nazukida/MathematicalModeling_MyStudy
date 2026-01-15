# 模型工作流模块
"""
用于串联数据预处理、模型和可视化的工作流模块

可用模型适配器：
- DynamicProgrammingAdapter: 动态规划（背包问题等）
- OptimizationAdapter: 优化算法（PSO/GA等）
- LinearProgrammingAdapter: 线性规划
- GreyPredictionAdapter: 灰色预测 GM(1,1)
- KMeansAdapter: K-Means 聚类
- TOPSISAdapter: TOPSIS 综合评价
- RegressionAdapter: 回归分析（线性/Ridge/Lasso/多项式）
"""

from .model_validation_pipeline import (
    ModelValidationPipeline,
    PipelineData,
    # 预处理步骤
    PreprocessingStep,
    MissingValueStep,
    OutlierRemovalStep,
    NormalizationStep,
    # 模型适配器（多种模型可选）
    ModelAdapter,
    DynamicProgrammingAdapter,
    OptimizationAdapter,
    LinearProgrammingAdapter,
    GreyPredictionAdapter,
    KMeansAdapter,
    TOPSISAdapter,
    RegressionAdapter,
    # 可视化步骤
    VisualizationStep,
    DPTableVisualization,
    ConvergenceVisualization,
    DataComparisonVisualization,
    # 快速验证函数
    quick_dp_validation,
    quick_optimization_validation
)

__all__ = [
    # 核心类
    'ModelValidationPipeline',
    'PipelineData',
    # 预处理
    'PreprocessingStep',
    'MissingValueStep',
    'OutlierRemovalStep', 
    'NormalizationStep',
    # 模型适配器（可替换）
    'ModelAdapter',
    'DynamicProgrammingAdapter',
    'OptimizationAdapter',
    'LinearProgrammingAdapter',
    'GreyPredictionAdapter',
    'KMeansAdapter',
    'TOPSISAdapter',
    'RegressionAdapter',
    # 可视化
    'VisualizationStep',
    'DPTableVisualization',
    'ConvergenceVisualization',
    'DataComparisonVisualization',
    # 快速验证
    'quick_dp_validation',
    'quick_optimization_validation'
]
