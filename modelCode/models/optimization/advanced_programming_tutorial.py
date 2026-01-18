"""
============================================================
高级规划模型使用教程
Advanced Programming Models Tutorial
============================================================

本教程介绍如何使用三种高级规划模型：
1. 非线性规划 (Nonlinear Programming)
2. 整数规划 / 0-1规划 (Integer/Binary Programming)
3. 多目标规划 (Multi-Objective Programming)

核心设计理念：
    数据预处理 → 模型求解 → 结果可视化
    
每个模型都提供：
- DataPreprocessor: 数据清洗、验证、标准化
- Solver: 核心求解器
- Visualizer: 结果可视化
- Pipeline: 一站式完整工作流（推荐使用）

作者：MCM/ICM Team
日期：2026年1月
============================================================
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


# ============================================================
# 快速开始：使用Pipeline（推荐方式）
# ============================================================
"""
Pipeline是最简单的使用方式，它自动完成：
✓ 数据预处理
✓ 模型求解
✓ 结果可视化

你只需要提供：问题定义 + 调用run()
"""


# ============================================================
# 示例1：非线性规划 - 投资组合优化
# ============================================================

def demo_nonlinear_programming():
    """
    非线性规划示例：投资组合风险最小化
    
    问题：选择4种资产的投资比例，最小化风险的同时保证收益
    """
    print("\n" + "="*70)
    print("   非线性规划示例：投资组合优化")
    print("="*70)
    
    from models.optimization.advanced_nonlinear_programming import (
        NonlinearProgrammingPipeline
    )
    
    # --- 第1步：定义问题数据 ---
    # 预期收益率
    expected_returns = np.array([0.12, 0.08, 0.05, 0.06])
    
    # 协方差矩阵（衡量风险和资产相关性）
    cov_matrix = np.array([
        [0.04, 0.01, -0.005, 0.002],
        [0.01, 0.02, 0.003, 0.001],
        [-0.005, 0.003, 0.01, -0.002],
        [0.002, 0.001, -0.002, 0.015]
    ])
    
    # --- 第2步：定义目标函数（最小化风险=方差） ---
    def portfolio_risk(x):
        """投资组合风险（方差）"""
        return x @ cov_matrix @ x
    
    # --- 第3步：定义约束条件 ---
    constraints = [
        # 投资比例和为1
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        # 预期收益至少7%
        {'type': 'ineq', 'fun': lambda x: np.dot(expected_returns, x) - 0.07}
    ]
    
    # --- 第4步：定义变量边界 ---
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
    
    # --- 第5步：创建Pipeline并求解 ---
    pipeline = NonlinearProgrammingPipeline(verbose=True)
    
    result = pipeline.run(
        objective=portfolio_risk,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',           # 求解方法
        multistart=True,          # 多起点优化
        n_starts=5,               # 起始点数量
        plot_contour=False,       # 4维不绘制等高线
        variable_names=['科技股', '消费股', '债券', '黄金']
    )
    
    # --- 第6步：解读结果 ---
    print("\n最优投资组合:")
    for i, name in enumerate(['科技股', '消费股', '债券', '黄金']):
        print(f"  {name}: {result['x'][i]*100:.1f}%")
    print(f"最小风险(标准差): {np.sqrt(result['fun'])*100:.2f}%")
    
    return result


# ============================================================
# 示例2：整数规划 - 项目选择问题
# ============================================================

def demo_integer_programming():
    """
    整数规划示例：投资项目选择（0-1背包问题）
    
    问题：在有限预算下，选择哪些项目投资以最大化收益
    """
    print("\n" + "="*70)
    print("   整数规划示例：投资项目选择")
    print("="*70)
    
    from models.optimization.advanced_integer_programming import (
        IntegerProgrammingPipeline
    )
    
    # --- 第1步：准备数据（可以用DataFrame或字典） ---
    projects = pd.DataFrame({
        '项目名称': ['AI研发', '市场拓展', '设备升级', '人才培训', '品牌建设', '供应链优化'],
        '投资成本(万元)': [150, 80, 120, 50, 90, 110],
        '预期收益(万元)': [200, 100, 160, 70, 130, 140]
    })
    
    budget = 300  # 总预算300万元
    
    # --- 第2步：创建Pipeline并求解 ---
    pipeline = IntegerProgrammingPipeline(verbose=True)
    
    result = pipeline.run_knapsack(
        data=projects,
        budget=budget,
        value_col='预期收益(万元)',
        cost_col='投资成本(万元)',
        name_col='项目名称',
        plot_selection=True,     # 绘制选择结果
        plot_usage=True,         # 绘制资源使用
        plot_efficiency=True,    # 绘制效率分析
        plot_summary=True        # 绘制汇总图
    )
    
    # --- 第3步：解读结果 ---
    print("\n决策结果:")
    print(f"  选中项目: {', '.join(result['selected_items'])}")
    print(f"  总投资: {result['total_cost']:.0f}万元")
    print(f"  总收益: {result['total_value']:.0f}万元")
    print(f"  投资回报率: {(result['total_value']/result['total_cost']-1)*100:.1f}%")
    
    return result


# ============================================================
# 示例3：多目标规划 - 收益与风险权衡
# ============================================================

def demo_multi_objective():
    """
    多目标规划示例：投资组合的收益-风险权衡
    
    问题：同时优化两个冲突的目标（最大化收益 & 最小化风险）
    """
    print("\n" + "="*70)
    print("   多目标规划示例：收益与风险权衡")
    print("="*70)
    
    from models.optimization.advanced_multi_objective import (
        MultiObjectivePipeline
    )
    
    # --- 第1步：定义问题数据 ---
    expected_returns = np.array([0.12, 0.08, 0.05, 0.06])
    cov_matrix = np.array([
        [0.04, 0.01, -0.005, 0.002],
        [0.01, 0.02, 0.003, 0.001],
        [-0.005, 0.003, 0.01, -0.002],
        [0.002, 0.001, -0.002, 0.015]
    ])
    
    # --- 第2步：定义多个目标函数 ---
    def portfolio_return(x):
        """预期收益（希望最大化）"""
        return np.dot(expected_returns, x)
    
    def portfolio_risk(x):
        """风险标准差（希望最小化）"""
        return np.sqrt(x @ cov_matrix @ x)
    
    # --- 第3步：定义约束和边界 ---
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ]
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
    
    # --- 第4步：创建Pipeline并求解 ---
    pipeline = MultiObjectivePipeline(verbose=True)
    
    result = pipeline.run(
        objectives=[portfolio_return, portfolio_risk],
        senses=['max', 'min'],    # 收益最大化，风险最小化
        bounds=bounds,
        constraints=constraints,
        method='weighted_sum',     # 使用加权法
        n_weights=50,              # 生成50组权重
        obj_names=['预期收益', '风险(标准差)'],
        plot_pareto=True,          # 绘制帕累托前沿
        plot_parallel=True,        # 绘制平行坐标图
        find_knee=True             # 找最佳折中解
    )
    
    # --- 第5步：解读结果 ---
    print("\n帕累托前沿包含", result['n_solutions'], "个非支配解")
    
    if result['knee_solution'] is not None:
        print("\n最佳折中方案（膝点）:")
        assets = ['科技股', '消费股', '债券', '黄金']
        for i, asset in enumerate(assets):
            print(f"  {asset}: {result['knee_solution'][i]*100:.1f}%")
        print(f"  预期收益: {result['knee_objectives'][0]*100:.2f}%")
        print(f"  风险: {result['knee_objectives'][1]*100:.2f}%")
    
    return result


# ============================================================
# 进阶用法：分步调用各组件
# ============================================================

def demo_step_by_step():
    """
    进阶用法：分步调用预处理器、求解器、可视化器
    
    适用于需要更精细控制的场景
    """
    print("\n" + "="*70)
    print("   进阶用法：分步调用各组件")
    print("="*70)
    
    from models.optimization.advanced_nonlinear_programming import (
        NLPDataPreprocessor,
        NonlinearProgrammingSolver,
        NLPVisualizer,
        NLPSensitivityAnalyzer
    )
    
    # ========== 第1步：数据预处理 ==========
    print("\n【步骤1：数据预处理】")
    preprocessor = NLPDataPreprocessor(verbose=True)
    
    # 假设我们有一些原始数据需要清洗
    raw_data = np.array([
        [1.2, 3.5],
        [2.1, np.nan],  # 包含缺失值
        [1.8, 4.2],
        [100, 200],     # 可能是异常值
        [2.5, 3.8]
    ])
    
    # 清洗数据
    clean_data = preprocessor.clean_data(raw_data, method='median')
    
    # 异常值检测
    clean_data, outliers = preprocessor.detect_outliers(clean_data, method='zscore', threshold=2.5)
    
    # 估计变量边界
    bounds = preprocessor.estimate_bounds(clean_data)
    
    # 生成初始点
    initial_points = preprocessor.generate_initial_points(bounds, n_points=5, method='latin')
    
    
    # ========== 第2步：定义问题并求解 ==========
    print("\n【步骤2：模型求解】")
    
    # 定义一个简单的优化问题
    def objective(x):
        return (x[0] - 2)**2 + (x[1] - 3)**2
    
    bounds = [(0, 5), (0, 5)]
    
    solver = NonlinearProgrammingSolver(verbose=True)
    
    # 单次求解
    result = solver.solve(
        objective=objective,
        x0=np.array([0.5, 0.5]),
        bounds=bounds,
        method='SLSQP'
    )
    
    # 或者使用多起点优化
    result_multistart = solver.multistart_solve(
        objective=objective,
        bounds=bounds,
        n_starts=10
    )
    
    # 或者使用全局优化
    result_global = solver.global_solve(
        objective=objective,
        bounds=bounds
    )
    
    
    # ========== 第3步：灵敏度分析 ==========
    print("\n【步骤3：灵敏度分析】")
    
    analyzer = NLPSensitivityAnalyzer(solver)
    
    # 分析参数变化对结果的影响
    def build_objective(c):
        """构建参数化目标函数"""
        def obj(x):
            return (x[0] - c)**2 + (x[1] - 3)**2
        return obj
    
    sensitivity_result = analyzer.parameter_sensitivity(
        objective_builder=build_objective,
        param_name='目标中心x坐标',
        param_values=np.linspace(1, 4, 10),
        base_x0=np.array([2, 3]),
        bounds=bounds
    )
    
    # 绘制灵敏度图
    analyzer.plot_sensitivity('目标中心x坐标')
    
    
    # ========== 第4步：可视化 ==========
    print("\n【步骤4：可视化】")
    
    visualizer = NLPVisualizer(save_dir='./figures')
    
    # 绘制等高线图
    visualizer.plot_contour_with_constraints(
        objective=objective,
        bounds=bounds,
        optimal_point=result['x'],
        history=result.get('history'),
        title='非线性规划求解过程'
    )
    
    # 绘制收敛曲线
    if result.get('history') is not None:
        visualizer.plot_convergence(objective, result['history'])
    
    # 绘制结果汇总
    visualizer.plot_solution_summary(result, variable_names=['x₁', 'x₂'])
    
    return result


# ============================================================
# 完整工作流程串联示意
# ============================================================
"""
┌─────────────────────────────────────────────────────────────┐
│                     完整工作流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐     │
│  │ 原始数据      │   │ 问题定义      │   │ 参数设置      │     │
│  │ (CSV/Excel)  │   │ (目标函数)    │   │ (边界/约束)   │     │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘     │
│         │                  │                  │              │
│         ▼                  ▼                  ▼              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              DataPreprocessor (数据预处理)             │   │
│  │  • 缺失值处理  • 异常值检测  • 数据标准化              │   │
│  │  • 边界估计    • 初始点生成                           │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                                │
│                             ▼                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Solver (求解器)                      │   │
│  │  • 单次优化    • 多起点优化   • 全局优化               │   │
│  │  • 约束处理    • 迭代记录                              │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                                │
│                             ▼                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            SensitivityAnalyzer (灵敏度分析)            │   │
│  │  • 参数灵敏度  • 约束活跃性  • 影子价格                │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                                │
│                             ▼                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │               Visualizer (可视化)                      │   │
│  │  • 等高线图    • 收敛曲线    • 帕累托前沿               │   │
│  │  • 结果汇总    • 对比分析                              │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                                │
│                             ▼                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  最终输出                              │   │
│  │  • 最优解      • 可视化图表   • 分析报告               │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘

【简化使用方式】
使用 Pipeline 可以一行代码完成上述所有步骤：

    pipeline = XXXPipeline(verbose=True)
    result = pipeline.run(...)

Pipeline 内部自动调用各组件，完成：
1. 数据预处理 → 2. 模型求解 → 3. 结果可视化
"""


# ============================================================
# 三种模型的选择指南
# ============================================================
"""
┌──────────────────────────────────────────────────────────────┐
│                    如何选择合适的模型？                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Q1: 你的决策变量可以取任意实数吗？                            │
│      │                                                        │
│      ├─ 是 → Q2: 目标函数或约束是否包含非线性项？              │
│      │         │                                              │
│      │         ├─ 是 → 使用【非线性规划】                      │
│      │         │       NonlinearProgrammingPipeline           │
│      │         │                                              │
│      │         └─ 否 → 使用线性规划 (scipy.optimize.linprog)  │
│      │                                                        │
│      └─ 否 → Q3: 变量只能取0或1吗？                           │
│               │                                               │
│               ├─ 是 → 使用【0-1规划/背包问题】                 │
│               │       IntegerProgrammingPipeline.run_knapsack │
│               │                                               │
│               └─ 否 → 使用【一般整数规划】                     │
│                       IntegerProgrammingSolver.solve_custom   │
│                                                               │
│  Q4: 你有多个需要同时优化的目标吗？                            │
│      │                                                        │
│      └─ 是 → 使用【多目标规划】                                │
│              MultiObjectivePipeline                           │
│              • 加权法 (method='weighted_sum')                 │
│              • ε-约束法 (method='epsilon_constraint')         │
│                                                               │
└──────────────────────────────────────────────────────────────┘
"""


# ============================================================
# 常见问题与解决方案
# ============================================================
"""
【Q1: pulp模块找不到】
整数规划需要安装pulp库：
    pip install pulp

【Q2: 求解失败或结果不合理】
1. 检查约束是否存在冲突（无可行解）
2. 尝试调整初始点
3. 使用多起点优化 (multistart=True)
4. 使用全局优化 (solver.global_solve)

【Q3: 多目标优化帕累托解太少】
1. 增加权重数量 (n_weights)
2. 尝试不同的方法 (weighted_sum → epsilon_constraint)
3. 检查目标函数定义是否正确

【Q4: 如何保存图表？】
Pipeline和Visualizer都支持save_dir参数：
    pipeline = XXXPipeline(save_dir='./my_figures')
    
或者在可视化时指定文件名：
    visualizer.plot_xxx(..., save_name='my_plot.png')

【Q5: 如何获取中间计算结果？】
不使用Pipeline，而是分步调用各组件（见demo_step_by_step）
"""


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("   高级规划模型使用教程 - 完整演示")
    print("="*70)
    
    # 运行示例（取消注释以运行）
    
    # 示例1：非线性规划
    demo_nonlinear_programming()
    
    # 示例2：整数规划
    # demo_integer_programming()
    
    # 示例3：多目标规划
    # demo_multi_objective()
    
    # 进阶用法：分步调用
    # demo_step_by_step()
    
    print("\n教程运行完毕！")
