"""
优化算法模块 (Optimization Algorithms)
=====================================

包含粒子群优化、遗传算法、差分进化、参数反演、高级规划模型等算法。

模块结构：
- optimization_algorithms.py: PSO、GA、DE、参数反演
- advanced_nonlinear_programming.py: 非线性规划完整工作流
- advanced_integer_programming.py: 整数规划/0-1规划完整工作流
- advanced_multi_objective.py: 多目标规划完整工作流
- nsga2_multi_objective.py: NSGA-II多目标进化算法

使用方法：
    # 基础优化算法
    from models.optimization import PSO, GeneticAlgorithm, DifferentialEvolution
    from models.optimization import ParameterInversion, BenchmarkFunctions
    
    # 高级规划模型（推荐使用Pipeline工作流）
    from models.optimization.advanced_nonlinear_programming import (
        NonlinearProgrammingPipeline,  # 非线性规划工作流
        NonlinearProgrammingSolver,     # 非线性规划求解器
        NLPDataPreprocessor,            # 数据预处理
        NLPVisualizer                   # 可视化
    )
    
    from models.optimization.advanced_integer_programming import (
        IntegerProgrammingPipeline,     # 整数规划工作流
        IntegerProgrammingSolver,       # 整数规划求解器
        IPDataPreprocessor,             # 数据预处理
        IPVisualizer                    # 可视化
    )
    
    from models.optimization.advanced_multi_objective import (
        MultiObjectivePipeline,         # 多目标规划工作流
        MultiObjectiveSolver,           # 多目标求解器
        ParetoAnalyzer,                 # 帕累托分析
        MOPVisualizer                   # 可视化
    )
    
    from models.optimization.nsga2_multi_objective import NSGAII
"""

from .optimization_algorithms import (
    PSO,
    GeneticAlgorithm,
    DifferentialEvolution,
    ParameterInversion,
    BenchmarkFunctions,
    compare_optimizers
)

# 高级规划模型导入
try:
    from .advanced_nonlinear_programming import (
        NonlinearProgrammingPipeline,
        NonlinearProgrammingSolver,
        NLPDataPreprocessor,
        NLPVisualizer,
        NLPSensitivityAnalyzer
    )
except ImportError:
    pass

try:
    from .advanced_integer_programming import (
        IntegerProgrammingPipeline,
        IntegerProgrammingSolver,
        IPDataPreprocessor,
        IPVisualizer
    )
except ImportError:
    pass

try:
    from .advanced_multi_objective import (
        MultiObjectivePipeline,
        MultiObjectiveSolver,
        ParetoAnalyzer,
        MOPDataPreprocessor,
        MOPVisualizer
    )
except ImportError:
    pass

try:
    from .nsga2_multi_objective import NSGAII
except ImportError:
    pass

__all__ = [
    # 基础优化算法
    'PSO',
    'GeneticAlgorithm', 
    'DifferentialEvolution',
    'ParameterInversion',
    'BenchmarkFunctions',
    'compare_optimizers',
    
    # 非线性规划
    'NonlinearProgrammingPipeline',
    'NonlinearProgrammingSolver',
    'NLPDataPreprocessor',
    'NLPVisualizer',
    'NLPSensitivityAnalyzer',
    
    # 整数规划
    'IntegerProgrammingPipeline',
    'IntegerProgrammingSolver',
    'IPDataPreprocessor',
    'IPVisualizer',
    
    # 多目标规划
    'MultiObjectivePipeline',
    'MultiObjectiveSolver',
    'ParetoAnalyzer',
    'MOPDataPreprocessor',
    'MOPVisualizer',
    'NSGAII'
]
