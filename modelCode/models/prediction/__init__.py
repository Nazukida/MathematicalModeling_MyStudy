"""
============================================================
预测模型模块 (Prediction Models Module)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================

包含模型：
1. prediction_toolkit.py - 时间序列分析、回归预测、可视化
   - TimeSeriesAnalyzer: 时间序列分析器
   - MovingAveragePredictor: 移动平均预测
   - RegressionPredictor: 回归预测（随机森林/梯度提升）
   - TimeSeriesGenerator: 数据生成器

2. arma_prediction.py - ARIMA时间序列预测
   - 平稳性检验（ADF检验）
   - ACF/PACF分析
   - ARIMA模型拟合与预测

3. grey_prediction.py - 灰色预测模型
   - GM(1,1)灰色预测
   - 模型检验（后验差比C、小误差概率P）

4. quadratic_exponential_smoothing.py - 指数平滑
   - 一次指数平滑
   - 二次指数平滑（Holt方法）

5. xgboost_regression.py - XGBoost回归预测

6. comprehensive_prediction_tutorial.py - 完整教程（推荐入口）
   - PredictionDataPreprocessor: 数据预处理
   - MovingAverageModel: 移动平均模型
   - ExponentialSmoothingModel: 指数平滑模型
   - GreyPredictionModel: 灰色预测模型
   - RegressionPredictionModel: 回归预测模型
   - PredictionVisualizer: 可视化器
   - run_complete_example: 完整案例演示

使用方法：
    # 方式1：导入具体类
    from models.prediction.comprehensive_prediction_tutorial import (
        PredictionDataPreprocessor,
        ExponentialSmoothingModel,
        GreyPredictionModel,
        PredictionVisualizer
    )
    
    # 方式2：直接运行教程
    from models.prediction.comprehensive_prediction_tutorial import run_complete_example
    results = run_complete_example()

作者：MCM/ICM Team
日期：2026年1月22日
"""

# 从 comprehensive_prediction_tutorial 导入主要类
try:
    from .comprehensive_prediction_tutorial import (
        PredictionDataPreprocessor,
        MovingAverageModel,
        ExponentialSmoothingModel,
        GreyPredictionModel,
        RegressionPredictionModel,
        PredictionVisualizer,
        run_complete_example,
        print_workflow,
        print_usage_guide
    )
except ImportError:
    pass

# 从 prediction_toolkit 导入
try:
    from .prediction_toolkit import (
        TimeSeriesGenerator,
        TimeSeriesAnalyzer,
        MovingAveragePredictor,
        RegressionPredictor,
        PlotStyleConfig
    )
except ImportError:
    pass

__all__ = [
    # comprehensive_prediction_tutorial
    'PredictionDataPreprocessor',
    'MovingAverageModel',
    'ExponentialSmoothingModel',
    'GreyPredictionModel',
    'RegressionPredictionModel',
    'PredictionVisualizer',
    'run_complete_example',
    'print_workflow',
    'print_usage_guide',
    # prediction_toolkit
    'TimeSeriesGenerator',
    'TimeSeriesAnalyzer',
    'MovingAveragePredictor',
    'RegressionPredictor',
    'PlotStyleConfig',
]
