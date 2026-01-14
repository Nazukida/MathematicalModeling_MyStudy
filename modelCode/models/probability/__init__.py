"""
概率统计模型 (Probability & Statistics Models)
=============================================

包含高斯分布、GMM聚类、贝叶斯推断、蒙特卡洛模拟等模型。

模块结构：
- gaussian_distribution.py: 高斯分布分析、参数估计、正态性检验
- gaussian_mixture_model.py: GMM聚类、软分类、异常检测
- bayesian_inference.py: 贝叶斯推断、由果推因、参数反演
- monte_carlo_simulation.py: 蒙特卡洛模拟、风险分析、数值积分

使用方法：
    from models.probability import GaussianDistribution, GMMClustering
    from models.probability import NormalNormalBayes, BetaBinomialBayes, BayesianParameterEstimation
    from models.probability import MonteCarloSimulator, FinancialMonteCarlo
"""

from .gaussian_distribution import GaussianDistribution, MultiGaussianAnalyzer
from .gaussian_mixture_model import GMMClustering, generate_gmm_sample_data
from .bayesian_inference import (
    NormalNormalBayes, 
    BetaBinomialBayes, 
    MCMCBayesian,
    BayesianParameterEstimation
)
from .monte_carlo_simulation import (
    MonteCarloSimulator,
    ProjectRiskSimulator,
    FinancialMonteCarlo,
    MonteCarloIntegration
)

__all__ = [
    # 高斯分布
    'GaussianDistribution',
    'MultiGaussianAnalyzer',
    
    # GMM
    'GMMClustering',
    'generate_gmm_sample_data',
    
    # 贝叶斯
    'NormalNormalBayes',
    'BetaBinomialBayes',
    'MCMCBayesian',
    'BayesianParameterEstimation',
    
    # 蒙特卡洛
    'MonteCarloSimulator',
    'ProjectRiskSimulator',
    'FinancialMonteCarlo',
    'MonteCarloIntegration'
]
