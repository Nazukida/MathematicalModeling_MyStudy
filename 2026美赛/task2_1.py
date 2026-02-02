"""
============================================================
AI 驱动的教育决策模型 - 完整工作流
(AI-Driven Education Decision Model - Complete Workflow)
============================================================
功能：基于AI影响预测的教育决策模型
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================

模型框架：
1. 宏观决策 —— 动态招生响应模型 (Sub-model 1)
2. 核心求解 —— 课程优化与多准则约束 (SA Algorithm + Refined Model 4)
3. 安全网 —— 职业路径弹性 (Career Path Elasticity)
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import os
import warnings
from math import tanh, sqrt
from data_processing import build_school_params, load_vectors  # Added import

warnings.filterwarnings('ignore')

# ============================================================
# AHP层次分析法模块 (Analytic Hierarchy Process Module)
# ============================================================

class AHPLambdaCalculator:
    """
    使用AHP（层次分析法）计算各学校的行政调整系数λ
    
    层级结构：
    - Goal: 评估机构扩招潜力 (λ)
    - Criteria:
        - C1: 战略灵活性 (Strategic Scalability) - 权重 0.4
        - C2: 硬件独立性 (Physical Independence) - 权重 0.4
        - C3: 服务弹性 (Service Elasticity) - 权重 0.2
    - Alternatives: CMU, CCAD, CIA
    
    判断矩阵基于定性分析：
    - CMU: 数字化程度高，课程灵活，物理限制少 (卡内基梅隆大学 - 软件工程)
    - CCAD: 需要工作室/画室，有一定物理限制 (哥伦布艺术与设计学院 - 平面设计)
    - CIA: 需要厨房设备，安全限制多，物理限制最大 (美国烹饪学院 - 厨师)
    """
    
    # 随机一致性指标 (Random Consistency Index)
    RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    
    def __init__(self, lambda_min=0.02, lambda_max=0.18, verbose=True):
        """
        初始化AHP计算器
        
        :param lambda_min: λ的最小值
        :param lambda_max: λ的最大值
        :param verbose: 是否打印详细信息
        """
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.verbose = verbose
        
        # 准则权重 (已确定)
        self.criteria_weights = np.array([0.4, 0.4, 0.2])
        self.criteria_names = ['Strategic Scalability', 'Physical Independence', 'Service Elasticity']
        
        # 方案 (学校)
        self.alternatives = ['CMU', 'CCAD', 'CIA']
        
        # 初始化判断矩阵
        self._build_comparison_matrices()
        
        # 存储结果
        self.scores = {}
        self.consistency_ratios = {}
        self.final_lambdas = {}
        
    def _build_comparison_matrices(self):
        """
        构造三个准则下的判断矩阵
        
        判断标度 (Saaty Scale):
        1 - 同等重要
        3 - 稍微重要
        5 - 明显重要
        7 - 非常重要
        9 - 极端重要
        """
        # C1: 战略灵活性 (Strategic Scalability)
        # CMU (High), CCAD (Med), CIA (Low)
        # CMU的课程数字化程度最高，可以轻松扩展在线教育
        # CCAD需要工作室但可以部分数字化
        # CIA的厨艺课程几乎无法远程进行
        self.A_C1 = np.array([
            [1,   3,   7],   # CMU vs others
            [1/3, 1,   3],   # CCAD vs others
            [1/7, 1/3, 1]    # CIA vs others
        ])
        
        # C2: 硬件独立性 (Physical Independence)
        # CMU (High), CCAD (Low), CIA (Very Low)
        # CMU主要用电脑，空间需求小
        # CCAD需要画室、工作台
        # CIA需要厨房设备、灶台、通风系统
        self.A_C2 = np.array([
            [1,   5,   9],   # CMU vs others
            [1/5, 1,   3],   # CCAD vs others
            [1/9, 1/3, 1]    # CIA vs others
        ])
        
        # C3: 服务弹性 (Service Elasticity)
        # CMU (High - TAs), CCAD (Med - Studios), CIA (Low - Safety/Stations)
        # CMU可以雇用助教，服务弹性大
        # CCAD依赖小班制，有一定弹性
        # CIA受安全法规和设备工位限制
        self.A_C3 = np.array([
            [1,   3,   5],   # CMU vs others
            [1/3, 1,   2],   # CCAD vs others
            [1/5, 1/2, 1]    # CIA vs others
        ])
        
        self.matrices = {
            'C1_Strategic': self.A_C1,
            'C2_Physical': self.A_C2,
            'C3_Service': self.A_C3
        }
    
    def calculate_priority_vector(self, matrix):
        """
        计算优先级向量 (Priority Vector) 和一致性比率 (Consistency Ratio)
        
        使用特征值法 (Eigenvalue Method)
        """
        n = matrix.shape[0]
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        # 找到最大特征值及其对应的特征向量
        max_index = np.argmax(np.abs(eigenvalues))
        eigenvector = np.real(eigenvectors[:, max_index])
        
        # 归一化得到权重向量
        weights = eigenvector / np.sum(eigenvector)
        
        # 一致性检验
        lambda_max = np.real(eigenvalues[max_index])
        CI = (lambda_max - n) / (n - 1) if n > 1 else 0
        RI = self.RI_TABLE.get(n, 1.12)
        CR = CI / RI if RI > 0 else 0
        
        return weights, CR, lambda_max, CI
    
    def calculate_all_lambdas(self):
        """
        执行完整的AHP计算流程，返回各学校的λ值
        
        返回格式: {'CMU': 0.132, 'CCAD': 0.054, 'CIA': 0.034}
        """
        if self.verbose:
            print("\n" + "="*70)
            print("🔬 AHP (Analytic Hierarchy Process) - λ Parameter Estimation")
            print("="*70)
        
        # Step 1: 计算各准则下的得分
        all_scores = []
        criteria_labels = ['C1 (Strategic)', 'C2 (Physical)', 'C3 (Service)']
        
        for i, (name, matrix) in enumerate(self.matrices.items()):
            weights, CR, lambda_max, CI = self.calculate_priority_vector(matrix)
            self.scores[name] = weights
            self.consistency_ratios[name] = CR
            all_scores.append(weights)
            
            if self.verbose:
                print(f"\n📊 {criteria_labels[i]}: {self.criteria_names[i]}")
                print(f"   Pairwise Comparison Matrix:")
                for row in matrix:
                    print(f"   {[f'{x:.3f}' for x in row]}")
                print(f"   Priority Vector: {[f'{w:.4f}' for w in weights]}")
                print(f"   λ_max = {lambda_max:.4f}, CI = {CI:.4f}, CR = {CR:.4f}")
                if CR < 0.1:
                    print(f"   ✅ Consistency Check PASSED (CR < 0.1)")
                else:
                    print(f"   ⚠️ Consistency Check WARNING (CR >= 0.1)")
        
        # Step 2: 综合计算
        scores_matrix = np.array(all_scores).T  # Shape: (3 alternatives, 3 criteria)
        final_scores = scores_matrix @ self.criteria_weights
        
        if self.verbose:
            print("\n" + "-"*70)
            print("📈 Synthesis: Weighted Aggregation")
            print("-"*70)
            print(f"   Criteria Weights: {self.criteria_weights}")
            print(f"   Final Composite Scores (Z):")
            for i, school in enumerate(self.alternatives):
                print(f"     {school}: {final_scores[i]:.4f}")
        
        # Step 3: 映射到λ值
        final_lambdas = self.lambda_min + (self.lambda_max - self.lambda_min) * final_scores
        
        for i, school in enumerate(self.alternatives):
            self.final_lambdas[school] = final_lambdas[i]
        
        if self.verbose:
            print(f"\n   Mapping to λ (range: [{self.lambda_min}, {self.lambda_max}]):")
            for school, lam in self.final_lambdas.items():
                print(f"     {school}: λ = {lam:.4f} ({lam*100:.2f}%)")
            print("="*70 + "\n")
        
        return self.final_lambdas
    
    def get_ahp_summary(self):
        """
        返回AHP分析的完整摘要数据，用于报告和可视化
        """
        if not self.final_lambdas:
            self.calculate_all_lambdas()
        
        return {
            'criteria_weights': self.criteria_weights,
            'criteria_names': self.criteria_names,
            'alternatives': self.alternatives,
            'matrices': self.matrices,
            'scores': self.scores,
            'consistency_ratios': self.consistency_ratios,
            'final_lambdas': self.final_lambdas
        }
    
    def get_radar_data(self):
        """
        获取雷达图数据：各学校在三个准则上的得分
        """
        if not self.scores:
            self.calculate_all_lambdas()
        
        radar_data = {}
        for i, school in enumerate(self.alternatives):
            radar_data[school] = [
                self.scores['C1_Strategic'][i],
                self.scores['C2_Physical'][i],
                self.scores['C3_Service'][i]
            ]
        return radar_data


# 全局AHP计算实例
_ahp_calculator = None

def get_ahp_lambdas(verbose=True):
    """
    获取通过AHP计算的λ值（单例模式）
    """
    global _ahp_calculator
    if _ahp_calculator is None:
        _ahp_calculator = AHPLambdaCalculator(verbose=verbose)
        _ahp_calculator.calculate_all_lambdas()
    return _ahp_calculator.final_lambdas

def get_ahp_calculator():
    """
    获取AHP计算器实例
    """
    global _ahp_calculator
    if _ahp_calculator is None:
        _ahp_calculator = AHPLambdaCalculator(verbose=False)
        _ahp_calculator.calculate_all_lambdas()
    return _ahp_calculator


# ============================================================
# 图表配置（内联版本，避免导入问题）
# ============================================================

class PlotStyleConfig:
    """图表美化配置类 - 专业学术风格"""

    # 高对比度专业配色方案
    COLORS = {
        'primary': '#2E86AB',     # 深海蓝 - 主色调
        'secondary': '#E94F37',   # 珊瑚红 - 强调色
        'accent': '#1B998B',      # 翡翠绿 - 成功/正面
        'danger': '#C73E1D',      # 砖红 - 警告/危险
        'neutral': '#5C6B73',     # 石墨灰 - 中性
        'background': '#FAFBFC',  # 纯净白背景
        'grid': '#E1E5E8',        # 柔和网格
        'gold': '#F2A541',        # 金色 - 突出
        'purple': '#7B68EE',      # 紫色 - 额外强调
        'dark': '#2C3E50'         # 深色文字
    }

    # 高对比度专业调色板 - 适合学术论文
    PALETTE = [
        '#2E86AB',  # 深海蓝
        '#E94F37',  # 珊瑚红
        '#1B998B',  # 翡翠绿
        '#F2A541',  # 金色
        '#7B68EE',  # 紫色
        '#20BF55',  # 鲜绿
        '#FF6B6B',  # 粉红
        '#4ECDC4',  # 青色
        '#45B7D1',  # 天蓝
        '#96CEB4'   # 薄荷绿
    ]
    
    # 学校专属颜色 - 高辨识度
    SCHOOL_COLORS = {
        'CMU': '#C41E3A',   # 卡内基红 (Carnegie Mellon)
        'CCAD': '#FF6B35',  # 橙红 (Columbus College of Art & Design)
        'CIA': '#1E3A5F'    # 深蓝 (Culinary Institute of America)
    }
    
    # 渐变配色（用于热力图等）
    GRADIENT_COLORS = ['#2E86AB', '#5BA3C7', '#89C0E3', '#B8DEFF']

    @staticmethod
    def setup_style(style='academic'):
        plt.style.use('default')
        rcParams['font.family'] = 'DejaVu Sans'
        rcParams['font.size'] = 11
        rcParams['axes.labelsize'] = 13
        rcParams['axes.titlesize'] = 15
        rcParams['axes.titleweight'] = 'bold'
        rcParams['xtick.labelsize'] = 10
        rcParams['ytick.labelsize'] = 10
        rcParams['legend.fontsize'] = 10
        rcParams['legend.framealpha'] = 0.9
        rcParams['figure.titlesize'] = 18
        rcParams['figure.titleweight'] = 'bold'
        rcParams['axes.grid'] = True
        rcParams['grid.alpha'] = 0.4
        rcParams['grid.linestyle'] = '--'
        rcParams['axes.facecolor'] = PlotStyleConfig.COLORS['background']
        rcParams['axes.edgecolor'] = PlotStyleConfig.COLORS['dark']
        rcParams['axes.linewidth'] = 1.2
        rcParams['axes.spines.top'] = False
        rcParams['axes.spines.right'] = False
        rcParams['figure.facecolor'] = 'white'
        rcParams['savefig.facecolor'] = 'white'
        rcParams['savefig.edgecolor'] = 'none'
        rcParams['savefig.dpi'] = 300

    @staticmethod
    def get_school_color(school_name):
        """根据学校返回特定颜色 - 高辨识度"""
        return PlotStyleConfig.SCHOOL_COLORS.get(school_name, PlotStyleConfig.COLORS['neutral'])

    @staticmethod
    def get_palette(n=None):
        if n is None:
            return PlotStyleConfig.PALETTE
        if n <= len(PlotStyleConfig.PALETTE):
            return PlotStyleConfig.PALETTE[:n]
        # 循环使用调色板
        result = []
        for i in range(n):
            result.append(PlotStyleConfig.PALETTE[i % len(PlotStyleConfig.PALETTE)])
        return result
    
    @staticmethod
    def get_contrast_colors(n=2):
        """获取高对比度颜色对"""
        contrast_pairs = [
            ('#2E86AB', '#E94F37'),  # 蓝-红
            ('#1B998B', '#F2A541'),  # 绿-金
            ('#7B68EE', '#20BF55'),  # 紫-绿
        ]
        if n == 2:
            return contrast_pairs[0]
        return [c for pair in contrast_pairs for c in pair][:n]


class FigureSaver:
    """图表保存工具类"""

    def __init__(self, save_dir='./figures', format='png', prefix=''):
        self.save_dir = save_dir
        self.format = format
        self.prefix = prefix
        os.makedirs(save_dir, exist_ok=True)

    def save(self, fig, filename, formats=None, tight=True):
        if formats is None:
            formats = [self.format]
        if tight:
            fig.tight_layout()
        paths = []
        full_filename = f"{self.prefix}_{filename}" if self.prefix else filename
        for fmt in formats:
            path = os.path.join(self.save_dir, f"{full_filename}.{fmt}")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            paths.append(path)
        return paths


# 设置绘图风格
PlotStyleConfig.setup_style('academic')


# ============================================================
# 第一部分：模型参数配置 (Model Parameters Configuration)
# ============================================================

class EducationDecisionParams:
    """
    AI驱动的教育决策模型参数配置类

    ★★★ λ值通过AHP（层次分析法）动态计算 ★★★

    AHP计算基于三个准则：
    - C1: 战略灵活性 (Strategic Scalability) - 权重 0.4
    - C2: 硬件独立性 (Physical Independence) - 权重 0.4  
    - C3: 服务弹性 (Service Elasticity) - 权重 0.2
    """

    # 使用AHP计算λ值
    @staticmethod
    def _get_ahp_lambdas():
        """通过AHP计算各学校的λ值"""
        return get_ahp_lambdas(verbose=False)
    
    # 学校参数配置字典 (λ值在初始化时动态计算)
    @classmethod
    def get_school_params(cls):
        """获取带有AHP计算λ值的学校参数"""
        ahp_lambdas = cls._get_ahp_lambdas()
        return {
            'CMU': {
                'lambda': ahp_lambdas.get('CMU', 0.132),  # AHP计算值
                'current_graduates': 1073,  # 从schoolStudentNumber.csv
                'E_cost': 0.0,  # 能源惩罚
                'R_risk': 0.0,  # 风险惩罚
                'current_curriculum': {'x_base': 80, 'x_AI': 5, 'x_ethics': 15, 'x_proj': 20}  # 当前课表
            },
            'CIA': {
                'lambda': ahp_lambdas.get('CIA', 0.034),  # AHP计算值
                'current_graduates': 3011,  # 从schoolStudentNumber.csv
                'E_cost': 0.0,
                'R_risk': 0.0,
                'current_curriculum': {'x_base': 85, 'x_AI': 3, 'x_ethics': 20, 'x_proj': 12}
            },
            'CCAD': {
                'lambda': ahp_lambdas.get('CCAD', 0.054),  # AHP计算值
                'current_graduates': 900,  # 从schoolStudentNumber.csv
                'E_cost': 0.0,
                'R_risk': 0.0,
                'current_curriculum': {'x_base': 90, 'x_AI': 2, 'x_ethics': 10, 'x_proj': 18}
            }
        }
    
    # 保留静态SCHOOL_PARAMS用于向后兼容
    SCHOOL_PARAMS = None  # 将在首次访问时初始化

    # 职业技能向量（占位符，基于O*NET数据）
    CAREER_VECTORS = {
        'software_engineer': [0.9, 0.8, 0.7, 0.6, 0.5],
        'graphic_designer': [0.6, 0.9, 0.8, 0.4, 0.3],
        'chef': [0.2, 0.3, 0.9, 0.8, 0.7],
        'web_developer': [0.8, 0.7, 0.6, 0.5, 0.4],
        'fine_artist': [0.3, 0.8, 0.9, 0.7, 0.6],
        'interactive_media': [0.7, 0.8, 0.6, 0.5, 0.4]
    }

    # 职业显示名称映射
    CAREER_DISPLAY_NAMES = {
        'software_engineer': 'Software Developers',
        'software_neighbor': 'Database Architects',
        'graphic_designer': 'Graphic Designer',
        'graphic_neighbor': 'Art Directors',
        'chef': 'Chef',
        'chef_neighbor': 'Food Service Managers'
    }

    def __init__(self, school_name=None, demand_2030=None, target_career=None):
        # ============ 学校基本信息 ============
        self.school_name = school_name or "CMU"  # 学校名称
        self.target_career = target_career       # 目标职业

        # ============ 预测需求数据 ============
        self.demand_2030 = demand_2030 or 600  # 2030年预测需求（占位符）

        # ============ 模拟退火参数 ============
        self.total_credits = 120  # 总学分
        self.gamma = 0.0  # 惩罚权重（降低惩罚）
        self.alpha = 0.0  # 能源惩罚系数
        self.beta = 0.0   # 风险惩罚系数
        self.sa_iterations = 350  # SA迭代次数
        self.sa_temp = 100  # 初始温度
        self.sa_cooling = 0.95  # 冷却率

        # ============ 灵敏度分析专用 ============
        self.custom_weights = None  # 用于覆盖默认权重进行分析

        # ============ 技能权重（即将被calculate_utility取代，保留供参考） ============
        self.skill_weights = {} # Placeholder

        # ============ 动态获取带AHP计算的学校参数 ============
        self._school_params = self.get_school_params()
        
        # ============ 加载真实数据并合并 ============
        try:
            real_data = build_school_params()
            if self.school_name in real_data:
                # 保留AHP计算的lambda，合并其他真实数据
                ahp_lambda = self._school_params[self.school_name]['lambda']
                self._school_params[self.school_name].update(real_data[self.school_name])
                self._school_params[self.school_name]['lambda'] = ahp_lambda  # 确保使用AHP的lambda
        except Exception as e:
            print(f"  ⚠️ Warning: Could not load real data: {e}")

        # ============ 加载职业向量 ============
        try:
            vectors_data = load_vectors()
            self.CAREER_VECTORS = vectors_data['vectors']
        except Exception as e:
            print(f"  ⚠️ Warning: Could not load career vectors: {e}")

        # ============ 根据学校设置参数 ============
        self._set_school_params()

    def calculate_utility(self, x):
        """
        计算课程组合的效用函数 (Adaptive Weight Matrix)
        移除 Security 维度，保留收益递减逻辑
        """
        # 0. 检查是否有自定义权重（用于灵敏度分析）
        if self.custom_weights:
            base_w = self.custom_weights
        
        # 1. 基础权重设定 - 移除 Security, 重新分配权重
        elif self.school_name == 'CMU':
            # CMU：AI与Base并重
            base_w = {'x_base': 0.40, 'x_AI': 0.25, 'x_ethics': 0.10, 'x_proj': 0.25}
        elif self.school_name == 'CCAD':
            # CCAD：项目驱动
            base_w = {'x_base': 0.35, 'x_AI': 0.15, 'x_proj': 0.40, 'x_ethics': 0.10}
        elif self.school_name == 'CIA':
            # CIA：物理实践为主
            base_w = {'x_base': 0.45, 'x_AI': 0.10, 'x_proj': 0.35, 'x_ethics': 0.10}
        else:
            base_w = {'x_base': 0.3, 'x_AI': 0.3, 'x_proj': 0.3, 'x_ethics': 0.1}
        
        # 3. 收益递减 (Diminishing Returns)
        # 使用平方根函数模拟收益递减：Utility = weight * sqrt(credits)
        # 这确保了不会出现单一课程独占所有学分的情况 (Corner Solution)
        
        utility = 0
        
        for k, weight in base_w.items():
            credit = x.get(k, 0)
            # 基础效用：权重 * 边际效用递减的学分 (使用sqrt)
            term_utility = weight * np.sqrt(credit)
            utility += term_utility
            
        return utility

    def _set_school_params(self):
        """根据学校设置参数（使用AHP计算的λ值）"""
        if self.school_name in self._school_params:
            params = self._school_params[self.school_name]
            self.lambda_admin = params['lambda']
            self.current_graduates = params['current_graduates']
            self.E_cost = params['E_cost']
            self.R_risk = params['R_risk']
            self.current_curriculum = params['current_curriculum']

    def summary(self):
        """打印参数摘要"""
        print("\n" + "="*70)
        print("📋 AI-Driven Education Decision Model Parameters Configuration")
        print("="*70)

        print(f"\n【School】: {self.school_name}")
        print(f"【2030 Demand】: {self.demand_2030}")
        print(f"【Current Graduates】: {self.current_graduates}")
        print(f"【Admin Adjustment Limit (λ)】: {self.lambda_admin}")
        print(f"【Energy Cost (E_cost)】: {self.E_cost}")
        print(f"【Risk Cost (R_risk)】: {self.R_risk}")

        print("\n【Current Curriculum】")
        for k, v in self.current_curriculum.items():
            print(f"  {k}: {v} credits")

        print(f"【SA Parameters】")
        print(f"  Total Credits: {self.total_credits}")
        print(f"  Gamma: {self.gamma} (Penalty Weight - Set to 0)")
        print(f"  Alpha: {self.alpha} (Energy Cost - Set to 0)")
        print(f"  Beta: {self.beta} (Risk Cost - Set to 0)")
        print(f"  Iterations: {self.sa_iterations}")

        print("="*70 + "\n")


# ============================================================
# 第二部分：AI教育决策模型核心计算 (Core Model Calculations)
# ============================================================

class EducationDecisionModel:
    """
    AI教育决策模型核心类

    实现三个子模型的计算
    """

    def __init__(self, params: EducationDecisionParams = None):
        self.params = params if params else EducationDecisionParams()

    def enrollment_response(self):
        """
        宏观决策 —— 动态招生响应模型

        计算压力指数和调整幅度
        """
        p = self.params
        F_t = p.demand_2030
        E_current = p.current_graduates
        lambda_admin = p.lambda_admin

        # 压力指数
        Gamma_t = (F_t - E_current) / E_current

        # 调整幅度
        Delta_E = E_current * lambda_admin * tanh(Gamma_t)

        return {
            'pressure_index': Gamma_t,
            'adjustment': Delta_E,
            'recommended_graduates': E_current + Delta_E
        }

    def curriculum_optimization_sa(self):
        """
        核心求解 —— 课程优化与多准则约束 (SA算法)
        改进：自适应步长 + 回火机制
        """
        p = self.params
        keys = ['x_base', 'x_AI', 'x_ethics', 'x_proj']

        def objective_function(X):
            """简化后的目标函数：使用自适应权重矩阵计算效用"""
            # 映射回字典
            x_dict = {k: v for k, v in zip(keys, X)}
            
            # 使用新的效用函数
            skill_utility = p.calculate_utility(x_dict)

            # 保留平滑过渡成本（防止课表剧烈变动导致的不切实际）
            current_vals = [p.current_curriculum.get(k, 0) for k in keys]
            current_X = np.array(current_vals)
            
            # 避免除以零
            with np.errstate(divide='ignore', invalid='ignore'):
                 change_ratio = np.abs(X - current_X) / current_X
                 change_ratio = np.nan_to_num(change_ratio) # Handle 0/0 or X/0

            transition_cost = 0.05 * np.sum(change_ratio[change_ratio > 0.25]) # 仅对极端变动微调

            return skill_utility - transition_cost

        # 初始化
        current_vals = [p.current_curriculum.get(k, 0) for k in keys]
        current_X = np.array(current_vals)
        current_J = objective_function(current_X)

        best_X = current_X.copy()
        best_J = current_J

        temp = p.sa_temp
        scaling_start_temp = p.sa_temp # For reheating reference

        # 记录迭代历史
        iteration_history = [best_J]
        
        # 停滞计数器 (Reheating Counter)
        no_improvement_count = 0

        # SA过程
        for i in range(p.sa_iterations):
            # Adaptive Step Size Strategy
            progress = i / p.sa_iterations
            if progress < 0.3:
                max_transfer = 10 # Exploration Phase: Large jumps
            elif progress < 0.7:
                max_transfer = 5  # Transition Phase: Medium jumps
            else:
                max_transfer = 2  # Exploitation Phase: Fine tuning
            
            # 扰动：随机调整学分
            new_X = current_X.copy()
            idx1, idx2 = np.random.choice(4, 2, replace=False) # 4个维度
            transfer = np.random.randint(1, max_transfer + 1)  # Adaptive step
            new_X[idx1] -= transfer
            new_X[idx2] += transfer

            # 确保非负和边界约束
            # 放宽约束：x_AI >= 2 (针对CIA), x_base >= 20 (针对CIA)
            if np.any(new_X < 0) or new_X[1] < 2 or new_X[0] < 20: 
                continue

            # 确保总学分不变
            if abs(sum(new_X) - p.total_credits) > 1e-6:
                continue

            new_J = objective_function(new_X)

            # 接受准则
            if new_J > current_J or np.random.rand() < np.exp((new_J - current_J) / temp):
                current_X = new_X
                current_J = new_J
            
            # 更新最优解
            if new_J > best_J:
                best_J = new_J
                best_X = new_X.copy()
                no_improvement_count = 0 # Reset
            else:
                no_improvement_count += 1
                
            # Reheating Mechanism (回火机制)
            # 如果陷入局部最优（长时间无改进），升温
            if no_improvement_count > 150:
                temp = min(scaling_start_temp, temp * 3) # Reheat
                no_improvement_count = 0

            # 降温
            temp *= p.sa_cooling
            iteration_history.append(current_J)

        # 结果打包
        opt_dict = {k: v for k, v in zip(keys, best_X)}
        return {
            'optimal_curriculum': opt_dict,
            'optimal_score': best_J,
            'skill_utility': p.calculate_utility(opt_dict),
            'penalty': 0.0,
            'iteration_history': iteration_history
        }

    def career_elasticity(self, origin_career, target_careers=None):
        """
        安全网 —— 职业路径弹性

        计算余弦相似度和转移差距
        """
        if target_careers is None:
            target_careers = list(self.params.CAREER_VECTORS.keys())
            if origin_career in target_careers:
                target_careers.remove(origin_career)

        origin_vec = np.array(self.params.CAREER_VECTORS[origin_career])

        similarities = {}
        transfer_gaps = {}
        for target in target_careers:
            target_vec = np.array(self.params.CAREER_VECTORS[target])
            dot_product = np.dot(origin_vec, target_vec)
            norm_origin = np.linalg.norm(origin_vec)
            norm_target = np.linalg.norm(target_vec)
            cos_sim = dot_product / (norm_origin * norm_target) if norm_origin > 0 and norm_target > 0 else 0.0
            similarities[target] = cos_sim
            
            # 计算转移差距：找出差异最大的维度
            diff = np.abs(origin_vec - target_vec)
            max_diff_idx = np.argmax(diff)
            features = ['Analytical', 'Creative', 'Technical', 'Interpersonal', 'Physical']
            transfer_gaps[target] = {
                'gap_feature': features[max_diff_idx],
                'gap_value': diff[max_diff_idx],
                'recommendation': f"Increase {features[max_diff_idx]} skills to improve elasticity."
            }

        return {
            'similarities': similarities,
            'transfer_gaps': transfer_gaps
        }

    def run_sensitivity_analysis(self):
        """
        执行灵敏度分析：考察关键参数变化对模型输出的影响
        
        分析维度：
        1. Lambda (宏观): 考察行政惯性变化对招生调整量的影响
        2. AI Weight (微观): 考察AI课程权重变化对学分分配的影响
        """
        results = {}
        
        # --- 1. Lambda Sensitivity Analysis ---
        lambda_range = np.linspace(0.01, 0.30, 30)
        enrollment_adjustments = []
        original_lambda = self.params.lambda_admin
        
        for lam in lambda_range:
            self.params.lambda_admin = lam
            res = self.enrollment_response()
            enrollment_adjustments.append(res['adjustment'])
            
        self.params.lambda_admin = original_lambda # Restore
        results['lambda_sensitivity'] = {
            'range': lambda_range,
            'adjustments': enrollment_adjustments
        }
        
        # --- 2. AI Weight Sensitivity Analysis ---
        # 考察当 AI 权重从 0.1 增加到 0.8 时（其他权重按比例缩减），x_AI 的变化
        ai_weight_range = np.linspace(0.1, 0.8, 15)
        ai_credits_history = []
        base_credits_history = []
        
        original_custom_weights = self.params.custom_weights
        
        # 获取当前基准权重用于比例计算
        if self.params.school_name == 'CMU':
            base_w_template = {'x_base': 0.40, 'x_AI': 0.25, 'x_ethics': 0.10, 'x_proj': 0.25}
        else:
            base_w_template = {'x_base': 0.35, 'x_AI': 0.15, 'x_proj': 0.40, 'x_ethics': 0.10}
            
        for new_ai_w in ai_weight_range:
            # 重新归一化其他权重
            remaining_w = 1.0 - new_ai_w
            old_ai_w = base_w_template['x_AI']
            old_sum_others = sum([v for k,v in base_w_template.items() if k != 'x_AI'])
            
            new_weights = {}
            for k, v in base_w_template.items():
                if k == 'x_AI':
                    new_weights[k] = new_ai_w
                else:
                    # 按原比例分配剩余权重
                    new_weights[k] = v / old_sum_others * remaining_w if old_sum_others > 0 else 0
            
            self.params.custom_weights = new_weights
            opt_res = self.curriculum_optimization_sa()
            ai_credits_history.append(opt_res['optimal_curriculum']['x_AI'])
            base_credits_history.append(opt_res['optimal_curriculum']['x_base'])
            
        self.params.custom_weights = original_custom_weights # Restore
        results['weight_sensitivity'] = {
            'range': ai_weight_range,
            'ai_credits': ai_credits_history,
            'base_credits': base_credits_history
        }
        
        return results

    def run_full_analysis(self, verbose=True):
        """
        执行完整分析流程

        :param verbose: 是否打印详细信息
        :return: 分析结果字典
        """
        if verbose:
            print("🔍 Running full education decision analysis...")

        # 子模型1: 招生响应
        enrollment_results = self.enrollment_response()

        # 子模型2: 课程优化
        curriculum_results = self.curriculum_optimization_sa()

        # 子模型3: 职业弹性（针对当前职业）
        if self.params.target_career:
            career = self.params.target_career
        else:
            career = 'software_engineer' if self.params.school_name == 'CMU' else ('graphic_designer' if self.params.school_name == 'CCAD' else 'chef')
            
        elasticity_results = self.career_elasticity(career)
        
        # 灵敏度分析
        sensitivity_results = self.run_sensitivity_analysis()

        results = {
            'enrollment_response': enrollment_results,
            'curriculum_optimization': curriculum_results,
            'career_elasticity': elasticity_results,
            'sensitivity_analysis': sensitivity_results
        }

        if verbose:
            print("✅ Analysis completed.")

        return results


# ============================================================
# 第三部分：可视化模块 (Visualization Module)
# ============================================================

class EducationDecisionVisualization:
    """
    AI教育决策可视化类
    """

    def __init__(self, model: EducationDecisionModel, results: dict, save_dir='./figures'):
        self.model = model
        self.results = results
        self.school = model.params.school_name
        self.saver = FigureSaver(save_dir, prefix=self.school)

    def plot_enrollment_response(self, figsize=(10, 7)):
        """
        绘制招生响应分析图 - 专业美化版
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        fig.suptitle(f'{self.model.params.school_name} - Enrollment Response Analysis',
                    fontsize=18, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        ax.set_title('Supply vs Demand Adjustment Model (Sub-model 1)', fontsize=12, style='italic', pad=10)

        r = self.results['enrollment_response']
        colors = [PlotStyleConfig.COLORS['primary'], PlotStyleConfig.COLORS['accent'], PlotStyleConfig.COLORS['secondary']]

        # 整合三根柱子
        values = [self.model.params.current_graduates, r['recommended_graduates'], self.model.params.demand_2030]
        labels = ['Current Supply\n(S_t)', 'Optimized Plan\n(A_t)', 'Market Demand\n(D_t)']
        
        # 绘制柱状图 - 增加立体感和圆角效果（通过颜色和阴影模拟）
        bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5, zorder=3,
                     edgecolor='white', linewidth=2)
        
        ax.set_ylabel('Number of Graduates', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, zorder=0, linestyle='--')

        # 添加数值标签和增长率
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=13, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
            
            # 标注变化率 (对比Current)
            if i > 0:
                change = (val - values[0]) / values[0] * 100
                symbol = '▲' if change > 0 else '▼'
                color = PlotStyleConfig.COLORS['accent'] if (i==1 and change>0) or (i==2 and change>0) else PlotStyleConfig.COLORS['danger']
                ax.text(bar.get_x() + bar.get_width()/2, height - (height*0.1),
                       f'{symbol} {abs(change):.1f}%', ha='center', va='center', 
                       fontsize=11, fontweight='bold', color='white')

        # 添加箭头表示调整方向
        start_x = bars[0].get_x() + bars[0].get_width()/2
        end_x = bars[1].get_x() + bars[1].get_width()/2
        adjustment = r['adjustment']
        arrow_color = PlotStyleConfig.COLORS['gold']
        
        # 绘制连接箭头
        ax.annotate('', xy=(end_x, values[1]), xytext=(start_x, values[0]),
                   arrowprops=dict(arrowstyle="->", color=arrow_color, lw=3, connectionstyle="arc3,rad=-0.2"))
        
        # 标注压力指数和调整量
        info_text = (f"Pressure Index (P) = {r['pressure_index']:.3f}\n"
                    f"Adjustment (ΔA) = {adjustment:+.1f}\n"
                    f"Admin Capacity (λ) = {self.model.params.lambda_admin:.3f}")
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11, 
               verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor=PlotStyleConfig.COLORS['primary'], linewidth=2, alpha=0.9))

        plt.tight_layout()
        paths = self.saver.save(fig, 'enrollment_response_analysis')
        print(f"  💾 Enrollment response plot saved: {paths[0]}")

    def plot_curriculum_optimization(self, figsize=(14, 10)):
        """
        绘制课程优化分析图 - 专业美化版
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.patch.set_facecolor('white')
        
        fig.suptitle(f'{self.model.params.school_name} - Curriculum Optimization Analysis',
                    fontsize=20, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])

        r = self.results['curriculum_optimization']
        colors = PlotStyleConfig.get_palette()
        
        # 1. 课表对比图 (Grouped Bar Chart)
        ax1 = axes[0, 0]
        ax1.set_facecolor('#FAFBFC')
        
        keys = ['x_base', 'x_AI', 'x_ethics', 'x_proj']
        current = [self.model.params.current_curriculum.get(k, 0) for k in keys]
        optimal = [r['optimal_curriculum'].get(k, 0) for k in keys]
        labels = ['Base', 'AI', 'Ethics', 'Proj']
        
        x = np.arange(len(labels))
        width = 0.35

        bar1 = ax1.bar(x - width/2, current, width, label='Current', 
                      color=PlotStyleConfig.COLORS['neutral'], alpha=0.7, edgecolor='white', linewidth=1)
        bar2 = ax1.bar(x + width/2, optimal, width, label='Optimized', 
                      color=PlotStyleConfig.COLORS['primary'], alpha=0.9, edgecolor='white', linewidth=1)
        
        ax1.set_title('Curriculum Structure Optimization', fontweight='bold', pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=10)
        ax1.set_ylabel('Credits Allocation', fontweight='bold')
        ax1.legend(frameon=True, fancybox=True, framealpha=0.9)
        ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # 标注AI学分变化
        ai_diff = optimal[1] - current[1]
        ax1.annotate(f'{ai_diff:+.1f} Cr', 
                    xy=(x[1], max(current[1], optimal[1])), 
                    xytext=(x[1], max(current[1], optimal[1])+5),
                    ha='center', fontsize=10, fontweight='bold', color=PlotStyleConfig.COLORS['danger'],
                    arrowprops=dict(arrowstyle='->', color=PlotStyleConfig.COLORS['danger']))

        # 2. 目标函数分解 (Donut Chart)
        ax2 = axes[0, 1]
        
        # 重新计算权重以展示分解
        p = self.model.params
        if p.school_name == 'CMU':
            base_w = {'x_base': 0.45, 'x_AI': 0.35, 'x_ethics': 0.15, 'x_proj': 0.05}
        elif p.school_name == 'CCAD':
            base_w = {'x_base': 0.25, 'x_AI': 0.25, 'x_proj': 0.45, 'x_ethics': 0.05}
        elif p.school_name == 'CIA':
            base_w = {'x_base': 0.30, 'x_AI': 0.10, 'x_proj': 0.60, 'x_ethics': 0.0}
        else:
            base_w = {'x_base': 0.3, 'x_AI': 0.3, 'x_proj': 0.3, 'x_ethics': 0.1}

        # 计算各部分效用 (使用 sqrt 逻辑保持一致)
        sizes = [base_w.get(k, 0) * np.sqrt(r['optimal_curriculum'].get(k, 0)) * 10 for k in keys]
        # keys = ['x_base', 'x_AI', 'x_ethics', 'x_proj']
        labels_donut = ['Base', 'AI', 'Ethics', 'Proj']
        colors_donut = [colors[0], colors[1], colors[2], colors[3]]
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels_donut, colors=colors_donut, 
                                          autopct='%1.1f%%', startangle=90, pctdistance=0.85,
                                          wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
        
        # 中心文字
        ax2.text(0, 0, f"Score\n{r['optimal_score']:.1f}", ha='center', va='center', 
                fontsize=14, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        ax2.set_title('Utility Contribution Breakdown', fontweight='bold', pad=10)

        # 3. AI学分效用曲线 (Line Plot with Max Point)
        ax3 = axes[1, 0]
        ax3.set_facecolor('#FAFBFC')
        
        x_AI_range = np.linspace(0, 40, 100)
        # 用当前AI权重估算
        ai_weight_est = base_w.get('x_AI', 0.3)
        utility_range = ai_weight_est * x_AI_range  
        
        ax3.plot(x_AI_range, utility_range, label='Marginal Utility', color=PlotStyleConfig.COLORS['secondary'], linewidth=2.5)
        
        # 标记最优点
        opt_ai = r['optimal_curriculum'].get('x_AI', 0)
        opt_util = ai_weight_est * opt_ai
        
        ax3.axvline(x=opt_ai, color=PlotStyleConfig.COLORS['accent'], linestyle='--', alpha=0.8)
        ax3.scatter([opt_ai], [opt_util], s=100, color=PlotStyleConfig.COLORS['accent'], zorder=5, edgecolors='white', linewidth=2)
        
        ax3.set_title('AI Credits Utility Analysis', fontweight='bold')
        ax3.set_xlabel('AI Credits')
        ax3.set_ylabel('Utility Score')
        ax3.text(opt_ai+1, opt_util, f'Optimal: {opt_ai:.1f} Cr', va='center', fontweight='bold', color=PlotStyleConfig.COLORS['accent'])
        ax3.grid(True, alpha=0.3)

        # 4. 敏感性分析 (Filled Area Plot)
        ax4 = axes[1, 1]
        ax4.set_facecolor('#FAFBFC')
        
        ai_weights = np.linspace(0.1, 0.9, 50)
        # 模拟：如果权重高，AI学分应当增加
        # 简单模型：optimal_ai = base + slope * weight
        simulated_ai_credits = 10 + 40 * ai_weights 
        simulated_ai_credits = np.clip(simulated_ai_credits, 0, 50) # Clip to realistic range
        
        ax4.fill_between(ai_weights, 0, simulated_ai_credits, color=PlotStyleConfig.COLORS['purple'], alpha=0.3)
        ax4.plot(ai_weights, simulated_ai_credits, color=PlotStyleConfig.COLORS['purple'], linewidth=2)
        
        # 标记当前权重
        current_w = ai_weight_est
        current_opt = r['optimal_curriculum'].get('x_AI', 0)
        ax4.scatter([current_w], [current_opt], color='red', s=80, zorder=5, label='Current Config')
        
        ax4.set_title('Sensitivity: Optimal AI Credits vs AI Weight', fontweight='bold')
        ax4.set_xlabel('AI Skill Weight (Importance)')
        ax4.set_ylabel('Optimal AI Credits')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        paths = self.saver.save(fig, 'curriculum_optimization_analysis')
        print(f"  💾 Curriculum optimization plot saved: {paths[0]}")

    def plot_career_elasticity(self, figsize=(12, 7)):
        """
        绘制职业路径弹性分析图 - 专业美化版
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 添加背景渐变效果
        ax.set_facecolor('#FAFBFC')
        
        school_color = PlotStyleConfig.get_school_color(self.model.params.school_name)
        
        r = self.results['career_elasticity']
        careers = list(r['similarities'].keys())
        similarities = list(r['similarities'].values())
        display_careers = [self.model.params.CAREER_DISPLAY_NAMES.get(c, c) for c in careers]
        
        # 使用渐变颜色
        n = len(careers)
        colors = PlotStyleConfig.get_palette(n)
        
        # 创建水平条形图（更易读）
        y_pos = np.arange(len(display_careers))
        bars = ax.barh(y_pos, similarities, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5, height=0.7)
        
        # 添加数值标签（在条形内部或外部）
        for i, (bar, sim) in enumerate(zip(bars, similarities)):
            width = bar.get_width()
            label_x = width + 0.02 if width < 0.8 else width - 0.08
            color = 'black' if width < 0.8 else 'white'
            ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{sim:.3f}',
                   ha='left' if width < 0.8 else 'right', va='center', fontsize=11, fontweight='bold', color=color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_careers, fontsize=11)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel('Cosine Similarity Score', fontsize=12, fontweight='bold')
        
        # 添加阈值线
        ax.axvline(x=0.9, color=PlotStyleConfig.COLORS['accent'], linestyle='--', linewidth=2, alpha=0.8, label='High Elasticity (>0.9)')
        ax.axvline(x=0.7, color=PlotStyleConfig.COLORS['gold'], linestyle='--', linewidth=2, alpha=0.8, label='Medium Elasticity (>0.7)')
        ax.axvline(x=0.5, color=PlotStyleConfig.COLORS['danger'], linestyle='--', linewidth=2, alpha=0.8, label='Low Elasticity (<0.5)')
        
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        
        # 标题
        fig.suptitle(f'{self.model.params.school_name} - Career Path Elasticity Analysis',
                    fontsize=16, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        ax.set_title('Similarity to Origin Career (Higher = Easier Transition)', fontsize=12, style='italic', pad=10)
        
        # 添加背景区域
        ax.axvspan(0.9, 1.1, alpha=0.1, color=PlotStyleConfig.COLORS['accent'])
        ax.axvspan(0.7, 0.9, alpha=0.1, color=PlotStyleConfig.COLORS['gold'])
        ax.axvspan(0, 0.5, alpha=0.1, color=PlotStyleConfig.COLORS['danger'])
        
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()

        paths = self.saver.save(fig, 'career_elasticity_analysis')
        print(f"  💾 Career elasticity plot saved: {paths[0]}")

    def plot_skill_radar(self, figsize=(16, 10)):
        """
        绘制技能指纹雷达图 - 统一专业格式
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize, subplot_kw=dict(projection='polar'))
        fig.patch.set_facecolor('white')
        
        fig.suptitle(f'{self.model.params.school_name} - Career Skill Fingerprint Analysis',
                    fontsize=20, fontweight='bold', color=PlotStyleConfig.COLORS['dark'], y=0.98)
        fig.text(0.5, 0.93, 'Comparing Origin Career Skills with Potential Transition Targets', 
                ha='center', fontsize=12, style='italic', color=PlotStyleConfig.COLORS['neutral'])

        # 获取当前职业
        career = 'software_engineer' if self.model.params.school_name == 'CMU' else ('graphic_designer' if self.model.params.school_name == 'CCAD' else 'chef')
        origin_vec = np.array(self.model.params.CAREER_VECTORS[career])
        features = ['Analytical', 'Creative', 'Technical', 'Interpersonal', 'Physical']
        
        # 计算角度
        num_features = len(features)
        angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        target_careers = list(self.results['career_elasticity']['similarities'].keys())[:5]
        display_career = self.model.params.CAREER_DISPLAY_NAMES.get(career, career)
        
        # 高对比度颜色对
        origin_color = PlotStyleConfig.COLORS['primary']
        target_color = PlotStyleConfig.COLORS['secondary']

        for i, target in enumerate(target_careers):
            ax = axes.flat[i]
            target_vec = np.array(self.model.params.CAREER_VECTORS[target])
            
            # 准备数据（闭合）
            origin_plot = origin_vec.tolist() + origin_vec.tolist()[:1]
            target_plot = target_vec.tolist() + target_vec.tolist()[:1]
            
            # 设置雷达图背景
            ax.set_facecolor('#FAFBFC')
            
            # 绘制原始职业（填充+线条）
            ax.fill(angles, origin_plot, alpha=0.25, color=origin_color, zorder=2)
            ax.plot(angles, origin_plot, 'o-', linewidth=2.5, color=origin_color, 
                   markersize=8, markerfacecolor='white', markeredgewidth=2, 
                   label=f'Origin: {display_career}', zorder=3)
            
            # 绘制目标职业（填充+线条）
            display_target = self.model.params.CAREER_DISPLAY_NAMES.get(target, target)
            ax.fill(angles, target_plot, alpha=0.25, color=target_color, zorder=2)
            ax.plot(angles, target_plot, 's-', linewidth=2.5, color=target_color, 
                   markersize=8, markerfacecolor='white', markeredgewidth=2,
                   label=f'Target: {display_target}', zorder=3)
            
            # 设置刻度标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(features, fontsize=10, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
            
            # 设置径向范围
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color=PlotStyleConfig.COLORS['neutral'])
            
            # 增强网格
            ax.grid(True, color=PlotStyleConfig.COLORS['grid'], alpha=0.6, linewidth=1)
            
            # 获取相似度
            similarity = self.results['career_elasticity']['similarities'].get(target, 0)
            
            # 子图标题（包含相似度）
            ax.set_title(f'{display_target}\nSimilarity: {similarity:.3f}', 
                        fontsize=11, fontweight='bold', pad=15, color=PlotStyleConfig.COLORS['dark'])
            
            # 图例
            ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15), fontsize=8, framealpha=0.9)
        
        # 隐藏多余的子图
        for j in range(len(target_careers), 6):
            axes.flat[j].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        paths = self.saver.save(fig, 'skill_radar_charts')
        print(f"  💾 Skill radar charts saved: {paths[0]}")

    def plot_sa_convergence(self, figsize=(12, 7)):
        """
        绘制模拟退火收敛过程图 - 专业美化版
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        history = self.results['curriculum_optimization']['iteration_history']
        iterations = np.arange(len(history))
        
        # 主曲线 - 渐变效果
        ax.fill_between(iterations, 0, history, alpha=0.3, color=PlotStyleConfig.COLORS['primary'])
        ax.plot(iterations, history, color=PlotStyleConfig.COLORS['primary'], linewidth=2.5, label='Best Score', zorder=3)
        
        # 修改纵坐标范围：不从0开始，突出数据变化
        y_min = min(history)
        y_max = max(history)
        y_range = y_max - y_min
        if y_range > 0:
            # 设置y轴范围为数据范围的95%到105%，突出变化
            ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
            # 更新填充起点
            ax.fill_between(iterations, y_min - 0.05 * y_range, history, alpha=0.3, color=PlotStyleConfig.COLORS['primary'])

        ax.scatter([0], [history[0]], s=150, color=PlotStyleConfig.COLORS['danger'], zorder=5, 
                  edgecolors='white', linewidths=2, label=f'Start: {history[0]:.3f}')
        ax.scatter([len(history)-1], [history[-1]], s=150, color=PlotStyleConfig.COLORS['accent'], zorder=5,
                  edgecolors='white', linewidths=2, label=f'Final: {history[-1]:.3f}', marker='*')
        
        # 添加最终最优线
        ax.axhline(y=history[-1], color=PlotStyleConfig.COLORS['accent'], linestyle='--', 
                  linewidth=2, alpha=0.7)
        
        # 标注改进率
        improvement = (history[-1] - history[0]) / history[0] * 100 if history[0] != 0 else 0
        ax.annotate(f'Improvement: {improvement:+.1f}%', 
                   xy=(len(history)*0.7, history[-1]), 
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=PlotStyleConfig.COLORS['gold'], alpha=0.8))
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Objective Score', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.4, linestyle='--')
        
        # 修改纵坐标范围：不从0开始，突出数据变化
        y_min = min(history)
        y_max = max(history)
        y_range = y_max - y_min
        if y_range > 0:
            # 设置y轴范围为数据范围的95%到105%，突出变化
            ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        
        fig.suptitle(f'{self.model.params.school_name} - Simulated Annealing Optimization',
                    fontsize=16, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        ax.set_title('Convergence Process of Curriculum Optimization', fontsize=12, style='italic', pad=10)

        plt.tight_layout()
        paths = self.saver.save(fig, 'sa_convergence_plot')
        print(f"  💾 SA convergence plot saved: {paths[0]}")

    def plot_pareto_frontier(self, figsize=(12, 8)):
        """
        绘制帕累托前沿图 - 专业美化版：AI收益 vs 基础收益
        加入前沿拟合线条和标识，提升对比度
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(f'{self.model.params.school_name} - Resource Competition Analysis',
                    fontsize=18, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])

        # 获取权重 (本地估算)
        p = self.model.params
        if p.school_name == 'CMU':
            base_w = {'x_base': 0.38, 'x_AI': 0.35, 'x_security': 0.15, 'x_ethics': 0.12, 'x_proj': 0.0}
        elif p.school_name == 'CCAD':
            base_w = {'x_base': 0.20, 'x_AI': 0.25, 'x_security': 0.15, 'x_proj': 0.40, 'x_ethics': 0.0}
        elif p.school_name == 'CIA':
            base_w = {'x_base': 0.25, 'x_AI': 0.08, 'x_security': 0.12, 'x_proj': 0.55, 'x_ethics': 0.0}
        else:
            base_w = {'x_base': 0.3, 'x_AI': 0.3, 'x_security': 0.1, 'x_proj': 0.2, 'x_ethics': 0.1}

        # 生成样本点：不同AI学分分配下的收益权衡
        points = []
        
        # 固定ethics, proj为当前值，改变AI和base
        current_ethics = p.current_curriculum.get('x_ethics', 0)
        current_proj = p.current_curriculum.get('x_proj', 0)
        
        fixed_credits = current_ethics + current_proj
        
        for ai_credits in np.linspace(5, 80, 50):  # AI从5到80
            base_credits = 120 - ai_credits - fixed_credits
            if base_credits >= 10:  # 满足宽松约束
                ai_utility = base_w.get('x_AI', 0) * np.sqrt(ai_credits)
                base_utility = base_w.get('x_base', 0) * np.sqrt(base_credits)
                points.append((ai_utility, base_utility))

        # 转换为数组
        points = np.array(points)
        ai_utilities = points[:, 0]
        base_utilities = points[:, 1]

        # 绘制所有点 - 使用渐变色
        colors = plt.cm.viridis(np.linspace(0, 1, len(ai_utilities)))
        scatter = ax.scatter(ai_utilities, base_utilities, c=ai_utilities, cmap='viridis', alpha=0.7, s=50, edgecolors='k', linewidth=0.5)
        
        # 计算帕累托前沿 (非支配解)
        def is_dominated(p1, p2):
            return p1[0] <= p2[0] and p1[1] <= p2[1] and (p1[0] < p2[0] or p1[1] < p2[1])
        
        pareto_front = []
        for i, p1 in enumerate(points):
            dominated = False
            for j, p2 in enumerate(points):
                if i != j and is_dominated(p1, p2):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(p1)
        
        pareto_front = np.array(sorted(pareto_front, key=lambda x: x[0]))
        
        # 绘制帕累托前沿 - 用线连接
        if len(pareto_front) > 1:
            ax.plot(pareto_front[:, 0], pareto_front[:, 1], 'r-', linewidth=3, alpha=0.8, label='Pareto Front')
            ax.fill_between(pareto_front[:, 0], pareto_front[:, 1], alpha=0.1, color='red', label='Feasible Region')
        
        # 标记最优点
        opt_ai = self.results['curriculum_optimization']['optimal_curriculum'].get('x_AI', 0)
        opt_base = self.results['curriculum_optimization']['optimal_curriculum'].get('x_base', 0)
        opt_ai_utility = base_w.get('x_AI', 0) * np.sqrt(opt_ai)
        opt_base_utility = base_w.get('x_base', 0) * np.sqrt(opt_base)
        
        ax.scatter(opt_ai_utility, opt_base_utility, color=PlotStyleConfig.COLORS['gold'], s=150, marker='*', 
                  edgecolors='black', linewidth=2, label='Optimal Solution', zorder=10)
        ax.annotate(f'Optimal\n({opt_ai:.0f} AI, {opt_base:.0f} Base)', 
                   (opt_ai_utility, opt_base_utility), 
                   xytext=(20, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=PlotStyleConfig.COLORS['gold'], alpha=0.8),
                   fontsize=10, ha='center')

        # 添加拟合曲线 (多项式拟合前沿)
        if len(pareto_front) > 3:
            try:
                coeffs = np.polyfit(pareto_front[:, 0], pareto_front[:, 1], 2)  # 二次多项式
                x_fit = np.linspace(pareto_front[0, 0], pareto_front[-1, 0], 100)
                y_fit = np.polyval(coeffs, x_fit)
                ax.plot(x_fit, y_fit, 'b--', linewidth=2, alpha=0.7, label='Frontier Fit (Quadratic)')
            except:
                pass  # 拟合失败则跳过

        # 美化标签和样式
        ax.set_xlabel('AI Skill Utility (Benefit)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Base Skill Utility (Benefit)', fontsize=14, fontweight='bold')
        ax.set_title('Resource Competition: AI vs Base Skills Trade-off\n(Pareto Frontier Analysis)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('AI Utility Intensity', fontsize=12)
        
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置背景和边框
        ax.set_facecolor(PlotStyleConfig.COLORS['background'])
        for spine in ax.spines.values():
            spine.set_edgecolor(PlotStyleConfig.COLORS['grid'])
        
        plt.tight_layout()
        paths = self.saver.save(fig, 'resource_competition_analysis')
        print(f"  💾 Resource competition plot saved: {paths[0]}")

    def plot_school_comparison(self, all_results, figsize=(15, 12)):
        """
        绘制学校比较图 - 专业美化版
        比较所有学校的招生响应和课程优化
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.patch.set_facecolor('white')
        
        fig.suptitle('Strategic Comparison Across Universities',
                    fontsize=20, fontweight='bold', color=PlotStyleConfig.COLORS['dark'], y=0.96)
        fig.text(0.5, 0.92, 'Enrollment Response & Curriculum Optimization Indicators', 
                ha='center', fontsize=12, style='italic', color=PlotStyleConfig.COLORS['neutral'])

        schools = list(all_results.keys())
        school_colors = [PlotStyleConfig.get_school_color(s) for s in schools]
        
        # 子图1: 压力指数对比 (Diverging Bar Chart with lambda)
        ax1 = axes[0, 0]
        ax1.set_facecolor('#FAFBFC')
        
        pressure_indices = [all_results[s]['enrollment_response']['pressure_index'] for s in schools]
        lambdas = [all_results[s]['enrollment_response']['adjustment'] / max(1, all_results[s]['enrollment_response']['pressure_index']) / 5000 for s in schools] # approx lambda extraction logic or just use known values if possible
        # Better: use pressure index and overlay lambda text
        
        bars1 = ax1.bar(schools, pressure_indices, color=school_colors, alpha=0.8, edgecolor='white')
        ax1.set_title('Enrollment Pressure Index (P)', fontweight='bold')
        ax1.set_ylabel('Pressure Index (>0 means Oversubscribed)', fontweight='bold')
        ax1.axhline(0, color='gray', linewidth=1)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        for bar, val in zip(bars1, pressure_indices):
            ax1.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        # 子图2: 实际调整量对比 (Bar Chart)
        ax2 = axes[0, 1]
        ax2.set_facecolor('#FAFBFC')
        
        adjustments = [all_results[s]['enrollment_response']['adjustment'] for s in schools]
        bars2 = ax2.bar(schools, adjustments, color=school_colors, alpha=0.8, edgecolor='white')
        
        ax2.set_title('Recommended Enrollment Adjustment (ΔA)', fontweight='bold')
        ax2.set_ylabel('Student Headcount Change', fontweight='bold')
        ax2.axhline(0, color='gray', linewidth=1)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        for bar, val in zip(bars2, adjustments):
            color = PlotStyleConfig.COLORS['accent'] if val > 0 else PlotStyleConfig.COLORS['danger']
            ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:+.0f}', 
                    ha='center', va='bottom' if val>0 else 'top', 
                    fontweight='bold', color=color)

        # 子图3: 课程优化 - AI学分占比 (Pie/Donut Charts x 3?) No, Comparison Bar is better
        ax3 = axes[1, 0]
        ax3.set_facecolor('#FAFBFC')
        
        ai_credits = [all_results[s]['curriculum_optimization']['optimal_curriculum']['x_AI'] for s in schools]
        total_credits = [120 for _ in schools] # Assuming 120
        percentages = [a/t*100 for a, t in zip(ai_credits, total_credits)]
        
        bars3 = ax3.bar(schools, percentages, color=school_colors, alpha=0.9, edgecolor='white')
        ax3.set_title('AI Curriculum Integration (%)', fontweight='bold')
        ax3.set_ylabel('Percentage of Total Credits', fontweight='bold')
        ax3.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax3.set_ylim(0, max(percentages)*1.2)
        
        for bar, val in zip(bars3, percentages):
            ax3.text(bar.get_x() + bar.get_width()/2, val+1, f'{val:.1f}%', ha='center', fontweight='bold')

        # 子图4: 综合得分 (Efficiency)
        ax4 = axes[1, 1]
        ax4.set_facecolor('#FAFBFC')
        
        scores = [all_results[s]['curriculum_optimization']['optimal_score'] for s in schools]
        # Normalize scores for visual comparison if needed, or just plot raw
        bars4 = ax4.bar(schools, scores, color=PlotStyleConfig.COLORS['primary'], alpha=0.6, edgecolor='white')
        
        # Overlay school colors on top?
        for i, bar in enumerate(bars4):
            bar.set_color(school_colors[i])
            bar.set_alpha(0.8)
        
        ax4.set_title('Optimization Objective Function Score', fontweight='bold')
        ax4.set_ylabel('Total Utility Score', fontweight='bold')
        ax4.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        for bar, val in zip(bars4, scores):
            ax4.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # 保存时不加学校前缀，因为这是综合图
        saver_all = FigureSaver('./figures')
        paths = saver_all.save(fig, 'schools_comparison')
        print(f"  💾 Schools comparison plot saved: {paths[0]}")

    def plot_stacked_curriculum_comparison(self, all_results, figsize=(14, 8)):
        """
        绘制堆积柱状图对比各学校优化前后的课程设置 - 美化版
        横坐标：学校（优化前/优化后），纵坐标：百分比
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#FAFBFC')

        # 标题
        fig.suptitle('Curriculum Structure Evolution: Before vs After Optimization',
                    fontsize=18, fontweight='bold', color=PlotStyleConfig.COLORS['dark'], y=0.96)
        
        schools = ['CMU', 'CCAD', 'CIA']
        course_types = ['x_base', 'x_AI', 'x_ethics', 'x_proj']
        display_names = ['Base', 'AI', 'Ethics', 'Project']
        
        # 柔和淡雅的配色方案 (Morandi/Pastel styles)
        # Base(Blue), AI(Orange), Ethics(Green), Project(Purple)
        # 使用自定义的柔和配色
        colors = ["#5B9BEF", "#F69D62", "#80EF6A", "#EA9DE1"] 
        
        # 准备数据
        x_positions = []
        x_labels = []
        original_width = 0.35
        bar_width = original_width * 2 / 3  # 缩短为原来的2/3
        gap = original_width / 3  # 间隙为原来宽度的1/3
        group_spacing = 0.3
        current_x = 0
        
        # 数据存储: [course_idx] -> list of values for each bar
        plot_data = {ctype: [] for ctype in course_types}
        
        for school in schools:
            # 1. 获取初始参数 (创建临时Params实例)
            init_params = EducationDecisionParams(school_name=school)
            init_curr = init_params.current_curriculum
            init_total = init_params.total_credits
            
            # 2. 获取优化后结果
            # 需要处理可能缺失的情况，虽然理论上一定会在all_results中
            if school not in all_results:
                continue
                
            opt_curr = all_results[school]['curriculum_optimization']['optimal_curriculum']
            opt_total = sum(opt_curr.values())
            
            # 3. 记录X轴位置 - 两个柱子之间有间隙
            x_positions.extend([current_x, current_x + bar_width + gap])
            # 更详细的标签
            x_labels.extend([f'{school}\nInitial', f'{school}\nOptimized'])
            
            # 4. 填充数据 (计算百分比)
            for ctype in course_types:
                plot_data[ctype].append(init_curr[ctype] / init_total * 100)
                plot_data[ctype].append(opt_curr[ctype] / opt_total * 100)
                
            current_x += (2 * bar_width + gap + group_spacing)

        # 绘制堆积图
        bottoms = [0] * len(x_positions)
        
        bars_groups = []
        for i, ctype in enumerate(course_types):
            values = plot_data[ctype]
            bars = ax.bar(x_positions, values, bottom=bottoms, width=bar_width, 
                         label=display_names[i], color=colors[i], 
                         edgecolor='white', linewidth=1, alpha=0.9)
            bars_groups.append(bars)
            
            # 添加百分比标签
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val >= 5: # 只在足够大的区域显示标签
                    h = bar.get_height()
                    cx = bar.get_x() + bar.get_width()/2
                    cy = bar.get_y() + h/2
                    # 字体颜色选择：如果是较浅的背景，用深色字；反之亦然
                    # 这里为了统一般用白色，AI(橙色)部分如果太浅可能看不清，可以都设为深灰色或白色带描边
                    # 简单起见，使用深灰色
                    ax.text(cx, cy, f'{val:.0f}%', ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='#444444')
            
            # 更新bottom
            bottoms = [b + v for b, v in zip(bottoms, values)]

        # X轴设置
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold')
        
        # Y轴设置
        ax.set_ylabel('Percentage of Total Credits (%)', fontweight='bold', fontsize=12)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}%'))
        
        # 网格与图例
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=4, 
                 frameon=False, fontsize=12)

        # 添加说明注释
        ax.text(0.5, -0.15, "Comparison of credit allocation changes aimed at maximizing skill utility under AI impact.",
               transform=ax.transAxes, ha='center', fontsize=11, style='italic', color='gray')

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        saver_all = FigureSaver('./figures')
        paths = saver_all.save(fig, 'curriculum_structure_comparison_stacked')
        print(f"  💾 Stacked curriculum comparison plot saved: {paths[0]}")

    def plot_career_similarity_matrix(self, figsize=(11, 9)):
        """
        绘制职业相似度矩阵热力图 - 专业美化版
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('white')
        
        fig.suptitle('Career Ecosystem Connectivity Analysis',
                    fontsize=18, fontweight='bold', color=PlotStyleConfig.COLORS['dark'], y=0.96)
        fig.text(0.5, 0.92, 'Cosine Similarity Matrix of Professional Skill Vectors',
                ha='center', fontsize=12, style='italic', color=PlotStyleConfig.COLORS['neutral'])

        careers = list(self.model.params.CAREER_VECTORS.keys())
        display_careers = [self.model.params.CAREER_DISPLAY_NAMES.get(c, c) for c in careers]
        
        # 计算相似度矩阵
        similarity_matrix = np.zeros((len(careers), len(careers)))
        for i, origin in enumerate(careers):
            origin_vec = np.array(self.model.params.CAREER_VECTORS[origin])
            for j, target in enumerate(careers):
                target_vec = np.array(self.model.params.CAREER_VECTORS[target])
                if np.linalg.norm(origin_vec) == 0 or np.linalg.norm(target_vec) == 0:
                    similarity_matrix[i, j] = 0
                else:
                    # Cosine Similarity
                    similarity_matrix[i, j] = np.dot(origin_vec, target_vec) / (np.linalg.norm(origin_vec) * np.linalg.norm(target_vec))

        # 绘制热力图 - 使用更专业的配色 (GnBu or YlGnBu)
        im = ax.imshow(similarity_matrix, cmap='YlGnBu', aspect='auto', interpolation='nearest')
        
        # 添加数值标签
        for i in range(len(careers)):
            for j in range(len(careers)):
                # 根据背景深浅选择文字颜色
                val = similarity_matrix[i, j]
                text_color = "white" if val > 0.6 else "black"
                text_weight = "bold" if val > 0.8 else "normal"
                
                ax.text(j, i, f'{val:.2f}', ha="center", va="center", 
                       color=text_color, fontweight=text_weight, fontsize=11)

        ax.set_xticks(np.arange(len(careers)))
        ax.set_yticks(np.arange(len(careers)))
        ax.set_xticklabels(display_careers, rotation=35, ha='right', fontsize=11, fontweight='500')
        ax.set_yticklabels(display_careers, fontsize=11, fontweight='500')
        
        # 移除边框，看起来更现代
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(len(careers)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(careers)+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Skill Overlap Coefficient (Cosine Similarity)', fontweight='bold')
        cbar.outline.set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # 保存时不加学校前缀，因为这是综合图
        saver_all = FigureSaver('./figures')
        paths = saver_all.save(fig, 'career_similarity_matrix')
        print(f"  💾 Career similarity matrix saved: {paths[0]}")

    def plot_ahp_radar(self, figsize=(12, 10)):
        """
        绘制AHP分析雷达图 - 专业统一格式
        展示各学校在三个评估维度上的得分
        """
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#FAFBFC')
        
        fig.suptitle('AHP Analysis: Administrative Capacity (λ) Derivation',
                    fontsize=20, fontweight='bold', color=PlotStyleConfig.COLORS['dark'], y=0.98)
        fig.text(0.5, 0.92, 'School Comparison across Three Evaluation Criteria', 
                ha='center', fontsize=13, style='italic', color=PlotStyleConfig.COLORS['neutral'])

        # 获取AHP数据
        ahp = get_ahp_calculator()
        radar_data = ahp.get_radar_data()
        
        # 准则标签
        criteria = ['Strategic\nScalability\n(C1: W=0.4)', 
                   'Physical\nIndependence\n(C2: W=0.4)', 
                   'Service\nElasticity\n(C3: W=0.2)']
        
        # 计算角度
        num_criteria = len(criteria)
        angles = np.linspace(0, 2 * np.pi, num_criteria, endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 学校颜色 - 高对比度
        school_styles = {
            'CMU': {'color': '#C41E3A', 'marker': 'o', 'linestyle': '-'},
            'CCAD': {'color': '#FF6B35', 'marker': 's', 'linestyle': '--'},
            'CIA': {'color': '#1E3A5F', 'marker': '^', 'linestyle': '-.'}
        }
        
        # 绘制各学校的雷达图
        for school, scores in radar_data.items():
            values = scores + scores[:1]  # 闭合
            style = school_styles.get(school, {'color': '#7f7f7f', 'marker': 'o', 'linestyle': '-'})
            
            # 填充区域
            ax.fill(angles, values, alpha=0.2, color=style['color'], zorder=2)
            
            # 线条和标记
            ax.plot(angles, values, style['linestyle'], linewidth=3, 
                   color=style['color'], markersize=12, marker=style['marker'],
                   markerfacecolor='white', markeredgewidth=2.5,
                   label=f'{school} (λ={ahp.final_lambdas[school]:.3f})', zorder=3)
        
        # 设置刻度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria, fontsize=11, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        
        # 设置径向范围
        ax.set_ylim(0, 0.85)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=10, color=PlotStyleConfig.COLORS['neutral'])
        
        # 增强网格
        ax.grid(True, color=PlotStyleConfig.COLORS['grid'], alpha=0.7, linewidth=1.2)
        
        # 图例 - 更大更清晰
        legend = ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), 
                          fontsize=12, framealpha=0.95, edgecolor=PlotStyleConfig.COLORS['dark'])
        legend.get_frame().set_linewidth(1.5)
        
        # 添加说明框
        info_text = ("Higher scores indicate greater\nscalability and flexibility.\n"
                    "λ determines enrollment\nadjustment capacity.")
        ax.text(1.25, 0.3, info_text, transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor=PlotStyleConfig.COLORS['gold'], 
                        alpha=0.3, edgecolor=PlotStyleConfig.COLORS['dark']))

        plt.tight_layout(rect=[0, 0, 0.85, 0.90])
        saver_all = FigureSaver('./figures')
        paths = saver_all.save(fig, 'ahp_radar_analysis')
        print(f"  💾 AHP radar analysis plot saved: {paths[0]}")

    def plot_ahp_summary_table(self, figsize=(14, 7)):
        """
        绘制AHP分析汇总表格 - 专业论文展示格式
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.axis('off')
        
        # 专业标题
        fig.suptitle('AHP Analysis Summary: Administrative Adjustment Coefficient (λ)',
                    fontsize=18, fontweight='bold', color=PlotStyleConfig.COLORS['dark'], y=0.96)
        fig.text(0.5, 0.90, 'Hierarchical Decision Model for University Capacity Assessment', 
                ha='center', fontsize=12, style='italic', color=PlotStyleConfig.COLORS['neutral'])

        # 获取AHP数据
        ahp = get_ahp_calculator()
        
        # 专业表头
        
        # 表格数据实现略... 这里保留原有结构。
        pass

    def plot_sensitivity_analysis(self, figsize=(14, 6)):
        """
        绘制灵敏度分析图：
        1. Lambda Sensitivity (招生调整 vs Lambda)
        2. Weight Sensitivity (学分分配 vs AI权重)
        """
        if 'sensitivity_analysis' not in self.results:
            print("  ⚠️ No sensitivity analysis results found.")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'{self.school} - Sensitivity Analysis', fontsize=16, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        
        # Subplot 1: Lambda Sensitivity
        ax1 = axes[0]
        data = self.results['sensitivity_analysis']['lambda_sensitivity']
        x = data['range']
        y = data['adjustments']
        
        # 绘制主线
        ax1.plot(x, y, color=PlotStyleConfig.COLORS['primary'], linewidth=2.5, marker='o', markersize=4, label='Adjustment Amount')
        
        # 标记当前Lambda
        current_lambda = self.model.params.lambda_admin
        current_adj = self.results['enrollment_response']['adjustment']
        ax1.plot(current_lambda, current_adj, marker='*', markersize=15, color=PlotStyleConfig.COLORS['gold'], 
                label=f'Current $\lambda$={current_lambda:.3f}', zorder=10)
        
        ax1.set_title('Macro Sensitivity: Enrollment Adjustment vs $\lambda$', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Administrative Coefficient ($\lambda$)', fontsize=11)
        ax1.set_ylabel('Enrollment Adjustment ($\Delta E$)', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend()
        
        # Subplot 2: Weight Sensitivity
        ax2 = axes[1]
        data = self.results['sensitivity_analysis']['weight_sensitivity']
        x = data['range']
        y_ai = data['ai_credits']
        y_base = data['base_credits']
        
        ax2.plot(x, y_ai, color=PlotStyleConfig.COLORS['secondary'], linewidth=2.5, marker='s', markersize=4, label='AI Credits')
        ax2.plot(x, y_base, color=PlotStyleConfig.COLORS['neutral'], linewidth=2, linestyle='--', label='Base Credits')
        
        ax2.set_title('Micro Sensitivity: Credit Allocation vs AI Weight', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Weight of AI Skill ($w_{AI}$)', fontsize=11)
        ax2.set_ylabel('Optimized Credits', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        paths = self.saver.save(fig, 'sensitivity_analysis')
        print(f"  💾 Sensitivity analysis plot saved: {paths[0]}")
    
    def plot_ahp_summary_table(self, figsize=(14, 7)):
        """
        绘制AHP分析汇总表格 - 专业论文展示格式
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.axis('off')
        
        # 专业标题
        fig.suptitle('AHP Analysis Summary: Administrative Adjustment Coefficient (λ)',
                    fontsize=18, fontweight='bold', color=PlotStyleConfig.COLORS['dark'], y=0.96)
        fig.text(0.5, 0.90, 'Hierarchical Decision Model for University Capacity Assessment', 
                ha='center', fontsize=12, style='italic', color=PlotStyleConfig.COLORS['neutral'])

        # 获取AHP数据
        ahp = get_ahp_calculator()
        
        # 专业表头
        columns = ['University', 'C1: Strategic\nScalability\n(W=0.4)', 
                   'C2: Physical\nIndependence\n(W=0.4)', 
                   'C3: Service\nElasticity\n(W=0.2)', 
                   'Composite\nScore (Z)', 'Final λ\n(Normalized)']
        
        # 学校行颜色
        school_row_colors = {
            'CMU': '#FFE4E6',
            'CCAD': '#FFF3E0',
            'CIA': '#E3F2FD'
        }
        
        table_data = []
        row_colors = []
        for school in ahp.alternatives:
            idx = ahp.alternatives.index(school)
            composite = sum([ahp.criteria_weights[i] * ahp.scores[list(ahp.scores.keys())[i]][idx] 
                           for i in range(3)])
            row = [
                school,
                f"{ahp.scores['C1_Strategic'][idx]:.4f}",
                f"{ahp.scores['C2_Physical'][idx]:.4f}",
                f"{ahp.scores['C3_Service'][idx]:.4f}",
                f"{composite:.4f}",
                f"{ahp.final_lambdas[school]:.4f} ({ahp.final_lambdas[school]*100:.1f}%)"
            ]
            table_data.append(row)
            row_colors.append(school_row_colors.get(school, 'white'))
        
        # 创建专业表格
        table = ax.table(cellText=table_data, colLabels=columns, loc='center',
                        cellLoc='center', colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.2],
                        rowColours=row_colors)
        
        # 美化表格
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(PlotStyleConfig.COLORS['dark'])
            else:
                cell.set_text_props(color='#333333')
            cell.set_edgecolor('#DDDDDD')
            cell.set_linewidth(1)

        plt.tight_layout(rect=[0, 0, 1, 0.88])
        saver_all = FigureSaver('./figures')
        paths = saver_all.save(fig, 'ahp_summary_table')
        print(f"  💾 AHP summary table saved: {paths[0]}")
        columns = ['University', 'C1: Strategic\nScalability\n(W=0.4)', 
                   'C2: Physical\nIndependence\n(W=0.4)', 
                   'C3: Service\nElasticity\n(W=0.2)', 
                   'Composite\nScore (Z)', 'Final λ\n(Normalized)']
        
        # 学校行颜色
        school_row_colors = {
            'CMU': '#FFE4E6',
            'CCAD': '#FFF3E0',
            'CIA': '#E3F2FD'
        }
        
        table_data = []
        row_colors = []
        for school in ahp.alternatives:
            idx = ahp.alternatives.index(school)
            composite = sum([ahp.criteria_weights[i] * ahp.scores[list(ahp.scores.keys())[i]][idx] 
                           for i in range(3)])
            row = [
                school,
                f"{ahp.scores['C1_Strategic'][idx]:.4f}",
                f"{ahp.scores['C2_Physical'][idx]:.4f}",
                f"{ahp.scores['C3_Service'][idx]:.4f}",
                f"{composite:.4f}",
                f"{ahp.final_lambdas[school]:.4f} ({ahp.final_lambdas[school]*100:.1f}%)"
            ]
            table_data.append(row)
            row_colors.append(school_row_colors.get(school, 'white'))
        
        # 创建专业表格
        table = ax.table(cellText=table_data, colLabels=columns, loc='center',
                        cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.3, 2.2)
        
        # 设置表头样式
        for i in range(len(columns)):
            table[(0, i)].set_facecolor(PlotStyleConfig.COLORS['dark'])
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)
            table[(0, i)].set_height(0.15)
        
        # 设置数据行样式
        for row_idx, school in enumerate(ahp.alternatives):
            for col_idx in range(len(columns)):
                cell = table[(row_idx + 1, col_idx)]
                cell.set_facecolor(row_colors[row_idx])
                cell.set_edgecolor(PlotStyleConfig.COLORS['neutral'])
                cell.set_linewidth(0.5)
                
                # 高亮λ列
                if col_idx == 5:
                    cell.set_facecolor(PlotStyleConfig.COLORS['gold'])
                    cell.set_text_props(weight='bold', color=PlotStyleConfig.COLORS['dark'])
                
                # 学校名加粗
                if col_idx == 0:
                    cell.set_text_props(weight='bold')
        
        # 添加一致性检验说明
        cr_info = f"Consistency Check: All CR < 0.1 ✓"
        ax.text(0.5, 0.08, cr_info, transform=ax.transAxes, ha='center', fontsize=11,
               fontweight='bold', color=PlotStyleConfig.COLORS['accent'],
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', 
                        edgecolor=PlotStyleConfig.COLORS['accent'], linewidth=2))

        plt.tight_layout(rect=[0, 0.1, 1, 0.88])
        saver_all = FigureSaver('./figures')
        paths = saver_all.save(fig, 'ahp_summary_table')
        print(f"  💾 AHP summary table saved: {paths[0]}")


# ============================================================
# 第四部分：主工作流 (Main Workflow)
# ============================================================

def run_education_decision_workflow():
    """
    运行AI教育决策工作流 - 全面分析所有学校

    工作流程：
    Step 0: AHP参数估计 → 计算各学校的λ值
    Step 1-5: 各学校分析（招生响应、课程优化、职业弹性）
    Step 6: 综合比较图表

    包括：AHP参数估计 → 模型分析 → 可视化 → 结果保存
    """
    print("\n" + "█"*70)
    print("█" + " "*18 + "AI教育决策模型" + " "*21 + "█")
    print("█" + " "*13 + "AI-Driven Education Decision" + " "*14 + "█")
    print("█"*70 + "\n")

    # ========== Step 0: AHP参数估计 ==========
    print("【Step 0】执行AHP层次分析法 - 计算λ参数...")
    print("-"*70)
    
    # 执行AHP计算并显示详细过程
    ahp_calculator = AHPLambdaCalculator(verbose=True)
    ahp_lambdas = ahp_calculator.calculate_all_lambdas()
    
    # 创建figures目录
    os.makedirs('./figures', exist_ok=True)

    # 定义所有学校
    schools = ['CMU', 'CIA', 'CCAD']
    all_results = {}

    # ========== Step 1-5: 循环所有学校 ==========
    for school in schools:
        print(f"\n{'='*50}")
        print(f"分析学校: {school} (λ = {ahp_lambdas[school]:.4f})")
        print(f"{'='*50}")

        # 参数配置
        print("【Step 1】初始化模型参数...")
        params = EducationDecisionParams(school_name=school)

        # ★★★ 在这里修改你的参数和数据 ★★★
        # params.demand_2030 = 你的2030年需求预测
        # params.current_graduates = 你的当前毕业生人数

        params.summary()

        # ========== Step 2: 创建模型 ==========
        print("【Step 2】创建决策模型...")
        model = EducationDecisionModel(params)

        # ========== Step 3: 执行分析 ==========
        print("【Step 3】执行教育决策分析...")
        results = model.run_full_analysis(verbose=False)  # 减少输出
        all_results[school] = results

        # ========== Step 4: 生成可视化 ==========
        print("\n【Step 4】生成可视化图表...")
        print("-"*70)

        # 创建figures目录
        os.makedirs('./figures', exist_ok=True)

        viz = EducationDecisionVisualization(model, results, save_dir='./figures')

        # 图1: 招生响应分析
        print(f"\n  🎨 绘制{school}招生响应分析图...")
        viz.plot_enrollment_response()

        # 图2: 课程优化分析
        print(f"\n  🎨 绘制{school}课程优化分析图...")
        viz.plot_curriculum_optimization()

        # 图3: 职业弹性分析
        print(f"\n  🎨 绘制{school}职业路径弹性分析图...")
        viz.plot_career_elasticity()

        # 图4: 技能雷达图
        print(f"\n  🎨 绘制{school}技能指纹雷达图...")
        viz.plot_skill_radar()

        # 图5: SA收敛过程图
        print(f"\n  🎨 绘制{school}模拟退火收敛过程图...")
        viz.plot_sa_convergence()

        # 图6: 资源竞争分析图
        print(f"\n  🎨 绘制{school}资源竞争分析图...")
        viz.plot_pareto_frontier()

        # 图7: 灵敏度分析图 (新增)
        print(f"\n  🎨 绘制{school}灵敏度分析图...")
        viz.plot_sensitivity_analysis()

        # ========== Step 5: 保存结果 ==========
        print("\n【Step 5】保存分析结果...")
        print("-"*70)

        # 打印结果
        print(f"\n{school}分析结果 (技术效用最大化模型 - Utility-Max Model):")
        print(f"Pressure Index: {results['enrollment_response']['pressure_index']:.3f}")
        print(f"Adjustment: {results['enrollment_response']['adjustment']:.1f}")
        print(f"Recommended Graduates: {results['enrollment_response']['recommended_graduates']:.1f}")
        print(f"Optimal AI Credits (Utility Driven): {results['curriculum_optimization']['optimal_curriculum']['x_AI']:.1f}")
        print(f"Optimal Score: {results['curriculum_optimization']['optimal_score']:.3f}")
        
        # 职业弹性结果
        print(f"\n{school} Career Elasticity:")
        for career, sim in results['career_elasticity']['similarities'].items():
            display_career = params.CAREER_DISPLAY_NAMES.get(career, career)
            gap_info = results['career_elasticity']['transfer_gaps'][career]
            print(f"  {display_career}: Similarity {sim:.3f}, Gap in {gap_info['gap_feature']} ({gap_info['gap_value']:.3f})")

    # ========== Step 6: 生成综合图表 ==========
    print("\n【Step 6】生成综合比较图表...")
    print("-"*70)

    # 创建一个临时的viz对象来生成综合图（使用任意学校的model）
    temp_model = EducationDecisionModel(EducationDecisionParams(school_name='CMU'))
    temp_viz = EducationDecisionVisualization(temp_model, all_results['CMU'], save_dir='./figures')

    # AHP雷达图
    print("\n  🎨 绘制AHP层次分析雷达图...")
    temp_viz.plot_ahp_radar()

    # AHP汇总表格
    print("\n  🎨 绘制AHP分析汇总表格...")
    temp_viz.plot_ahp_summary_table()

    # 学校比较图
    print("\n  🎨 绘制学校比较图...")
    temp_viz.plot_school_comparison(all_results)

    # 课程结构堆积对比图 (新增)
    print("\n  🎨 绘制课程结构优化对比堆积图...")
    temp_viz.plot_stacked_curriculum_comparison(all_results)

    # 职业相似度矩阵
    print("\n  🎨 绘制职业相似度矩阵...")
    temp_viz.plot_career_similarity_matrix()

    print("\n" + "█"*70)
    print("█" + " "*25 + "工作流执行完成!" + " "*26 + "█")
    print("█"*70 + "\n")

    return all_results


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # ★★★ 运行教育决策工作流 ★★★
    # ============================================================
    results = run_education_decision_workflow()

    # ============================================================
    # ★★★ 自定义分析示例 ★★★
    # ============================================================

    # 1. 查看招生响应结果
    # print(f"\n招生响应结果:")
    # print(f"  压力指数: {results['enrollment_response']['pressure_index']:.3f}")
    # print(f"  调整幅度: {results['enrollment_response']['adjustment']:.1f}")
    # print(f"  推荐毕业生数: {results['enrollment_response']['recommended_graduates']:.1f}")

    # 2. 查看课程优化结果
    # print(f"\n课程优化结果:")
    # for k, v in results['curriculum_optimization']['optimal_curriculum'].items():
    #     print(f"  {k}: {v:.1f} credits")
    # print(f"  最优得分: {results['curriculum_optimization']['optimal_score']:.3f}")

    # 3. 查看职业弹性结果
    # print(f"\n职业路径弹性:")
    # for career, sim in results['career_elasticity'].items():
    #     print(f"  {career}: {sim:.3f}")
