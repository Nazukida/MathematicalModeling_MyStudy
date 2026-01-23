"""
动力学模型 (Dynamics Models)
============================

包含各类动力学模型和系统仿真工具。

模块：
- glv_ecosystem_model: 广义 Lotka-Volterra 生态系统模型
- war_model: 战争模型
- risk_transfer_matrix: 风险转移矩阵模型（执法干预后风险迁移）

使用方法：
    from models.dynamics import RiskTransferMatrix, ScenarioConfig, SCENARIOS
    from models.dynamics import create_grid_network, create_random_network
"""

from .risk_transfer_matrix import (
    RiskTransferMatrix,
    ScenarioConfig,
    SCENARIOS,
    create_grid_network,
    create_random_network,
    demo_risk_transfer
)

__all__ = [
    'RiskTransferMatrix',
    'ScenarioConfig', 
    'SCENARIOS',
    'create_grid_network',
    'create_random_network',
    'demo_risk_transfer'
]
