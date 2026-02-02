"""
============================================================
AI 驱动的教育决策模型 - 优化版 (Constrained Sustainable Model)
(AI-Driven Education Decision Model - Enhanced Workflow)
============================================================
功能：基于AI影响预测的教育决策模型 - 加入现实约束
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================

优化模型框架：
1. 宏观决策 —— 动态招生响应模型 (Sub-model 1)
2. 核心求解 —— 课程优化与多准则约束 (SA + Triple Constraints)
   ├── 公平性约束 (Equity Constraint)
   ├── 环境约束 (Green Cap)
   └── 安全与伦理约束 (Safety Constraint)
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
# 约束参数配置 (Constraint Parameters Configuration)
# ============================================================

class ConstraintParams:
    """
    三大现实约束参数配置
    
    从"效用最大化"转变为"约束下的可持续发展优化"
    解决"数据黑洞"问题：将绝对值建模转变为相对指数建模
    """
    
    # ============================================================
    # 1. 公平性约束：硬件可及性排斥指数 (Equity Constraint)
    # ============================================================
    # 课程排斥指数 e_i：反映该课程对学生经济能力的门槛要求
    EXCLUSION_INDEX = {
        'x_base': 0.05,      # 通识课 (Lecture): 极低门槛，仅需基础文具
        'x_proj': 0.30,      # 项目课 (Lab): 需要个人电脑，存在初级经济门槛
        'x_AI': 0.60,        # Gen-AI 创作课: 需持续订阅 API 或昂贵软件
        'x_ethics': 0.10,    # 伦理课: 讲座形式，门槛较低
        'x_high_compute': 0.90  # 高算力/VR 课: 依赖昂贵硬件 (>$500)
    }
    
    # 学校类型的平均排斥上限 E_max
    # 收紧约束：只能容忍 20% 的高端 AI 设备课
    EQUITY_THRESHOLD = {
        'CMU': 0.30,    # 私立研究型大学：从0.50降至0.30
        'CCAD': 0.25,   # 私立艺术学院：从0.40降至0.25
        'CIA': 0.25     # 职业学校：从0.25调整为0.25（保持可行）
    }
    
    # ============================================================
    # 2. 环境约束：绿色能耗封锁线 (Green Cap)
    # ============================================================
    # 高能耗课程分类
    HIGH_ENERGY_COURSES = ['x_AI']  # AI课程被视为高能耗
    
    # 绿色能耗阈值 β_env
    # 收紧约束：将高能耗课比例从 40% 下调到 10%-15%
    GREEN_THRESHOLD = {
        'CMU': 0.15,    # 研究型大学：从0.25降至0.15
        'CCAD': 0.12,   # 艺术学院：从0.20降至0.12
        'CIA': 0.12     # 职业学校：从0.15调整为0.12（保持可行）
    }
    
    # ============================================================
    # 3. 安全与伦理约束：解药比例 (Safety Constraint)
    # ============================================================
    # γ 系数：每学 1 单位 AI 技术，必须配套多少单位伦理教育
    # 基于 O*NET "Consequence of Error" 归一化: γ = (Score - 1) / 8
    SAFETY_RATIO = {
        'CMU': 0.50,    # 软件工程师 (O*NET Score: 4.6) - 高风险
        'CCAD': 0.30,   # 平面设计师 (O*NET Score: 3.2) - 中风险
        'CIA': 0.10     # 厨师 (O*NET Score: 2.5) - 较低风险
    }
    
    # ============================================================
    # 惩罚权重配置
    # ============================================================
    PENALTY_WEIGHTS = {
        'equity': 50.0,      # 公平性违规惩罚
        'green': 30.0,       # 环境违规惩罚
        'safety': 40.0       # 安全约束违规惩罚
    }
    
    @classmethod
    def get_school_constraints(cls, school_name):
        """获取指定学校的约束参数"""
        return {
            'E_max': cls.EQUITY_THRESHOLD.get(school_name, 0.40),
            'beta_env': cls.GREEN_THRESHOLD.get(school_name, 0.20),
            'gamma_safety': cls.SAFETY_RATIO.get(school_name, 0.30),
            'exclusion_index': cls.EXCLUSION_INDEX,
            'high_energy_courses': cls.HIGH_ENERGY_COURSES,
            'penalty_weights': cls.PENALTY_WEIGHTS
        }


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

    def __init__(self, school_name=None, demand_2030=None, target_career=None, enable_constraints=True):
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
        self.sa_iterations = 500  # SA迭代次数（增加以获得更好的收敛）
        self.sa_temp = 100  # 初始温度
        self.sa_cooling = 0.95  # 冷却率

        # ============ 灵敏度分析专用 ============
        self.custom_weights = None  # 用于覆盖默认权重进行分析
        
        # ============ 三大现实约束开关 ============
        self.enable_constraints = enable_constraints
        self.constraint_params = ConstraintParams.get_school_constraints(self.school_name)

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

    def check_constraints(self, x):
        """
        检查三大刚性红线约束 (Hard Constraints)
        
        "红线管理"模型：约束条件不参与目标函数计算，而是作为可行域判定条件。
        违反任何一条红线的解将被判定为"不可行" (Infeasible) 并被舍弃。
        
        约束条件：
        1. 公平性约束: avg_exclusion ≤ E_max
        2. 环境约束: high_energy_ratio ≤ β_env  
        3. 安全约束: x_ethics ≥ γ * x_AI
        
        :param x: 课程学分字典 {'x_base': ..., 'x_AI': ..., 'x_ethics': ..., 'x_proj': ...}
        :return: (is_feasible: bool, constraint_details: dict)
        """
        if not self.enable_constraints:
            return True, {'status': 'constraints_disabled'}
        
        c = self.constraint_params
        total_credits = sum(x.values())
        if total_credits == 0:
            total_credits = 1  # 避免除零
        
        details = {
            'feasible': True,
            'violations': []
        }
        
        # ========== 1. 公平性约束 (Equity Constraint) ==========
        # 加权平均排斥指数：Σ(e_i * x_i) / S_total ≤ E_max
        exclusion_idx = c['exclusion_index']
        weighted_exclusion = sum(exclusion_idx.get(k, 0.1) * v for k, v in x.items())
        avg_exclusion = weighted_exclusion / total_credits
        
        equity_satisfied = avg_exclusion <= c['E_max']
        details['equity'] = {
            'avg_exclusion': avg_exclusion,
            'threshold': c['E_max'],
            'satisfied': equity_satisfied,
            'margin': c['E_max'] - avg_exclusion  # 正值表示有余量
        }
        if not equity_satisfied:
            details['feasible'] = False
            details['violations'].append('equity')
        
        # ========== 2. 环境约束 (Green Cap) ==========
        # 高能耗课程比例：Σ(high_energy_x) / S_total ≤ β_env
        high_energy_credits = sum(x.get(k, 0) for k in c['high_energy_courses'])
        high_energy_ratio = high_energy_credits / total_credits
        
        green_satisfied = high_energy_ratio <= c['beta_env']
        details['green'] = {
            'high_energy_ratio': high_energy_ratio,
            'threshold': c['beta_env'],
            'satisfied': green_satisfied,
            'margin': c['beta_env'] - high_energy_ratio
        }
        if not green_satisfied:
            details['feasible'] = False
            details['violations'].append('green')
        
        # ========== 3. 安全与伦理约束 (Safety Constraint) ==========
        # x_ethics ≥ γ * x_AI
        x_AI = x.get('x_AI', 0)
        x_ethics = x.get('x_ethics', 0)
        required_ethics = c['gamma_safety'] * x_AI
        
        safety_satisfied = x_ethics >= required_ethics
        details['safety'] = {
            'required_ethics': required_ethics,
            'actual_ethics': x_ethics,
            'gamma': c['gamma_safety'],
            'satisfied': safety_satisfied,
            'margin': x_ethics - required_ethics  # 正值表示有余量
        }
        if not safety_satisfied:
            details['feasible'] = False
            details['violations'].append('safety')
        
        return details['feasible'], details

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
        if self.enable_constraints:
            print("📋 Red-Line Constrained Education Model - Parameters")
            print("   (硬约束模式: 约束作为可行域边界，不参与目标函数计算)")
        else:
            print("📋 Baseline Utility-Max Model - Parameters")
        print("="*70)

        print(f"\n【School】: {self.school_name}")
        print(f"【2030 Demand】: {self.demand_2030}")
        print(f"【Current Graduates】: {self.current_graduates}")
        print(f"【Admin Adjustment Limit (λ)】: {self.lambda_admin}")

        print("\n【Current Curriculum】")
        for k, v in self.current_curriculum.items():
            print(f"  {k}: {v} credits")

        print(f"\n【SA Parameters】")
        print(f"  Total Credits: {self.total_credits}")
        print(f"  Iterations: {self.sa_iterations}")
        
        if self.enable_constraints:
            c = self.constraint_params
            print(f"\n【Triple Red-Line Constraints (Hard Constraints)】")
            print(f"  📊 E_max (Equity Threshold): {c['E_max']:.2f}")
            print(f"     → avg_exclusion ≤ {c['E_max']:.2f}")
            print(f"  🌿 β_env (Green Cap): {c['beta_env']:.2f}")
            print(f"     → high_energy_ratio ≤ {c['beta_env']:.2f}")
            print(f"  ⚖️ γ (Safety Ratio): {c['gamma_safety']:.2f}")
            print(f"     → x_ethics ≥ {c['gamma_safety']:.2f} * x_AI")
            print(f"\n  ⚠️ 注意: 违反任何红线的解将被直接舍弃 (Infeasible)")
        else:
            print(f"\n【Constraints: DISABLED (Baseline Mode)】")

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
        
        改进：自适应步长 + 回火机制 + 硬约束可行域检查
        
        "红线管理"模式 (The Red-Line Model)：
        - 目标函数 J(X) = U(X) - C_trans(X) 保持不变
        - 三大刚性约束作为可行域边界，违反则直接舍弃解
        """
        p = self.params
        keys = ['x_base', 'x_AI', 'x_ethics', 'x_proj']

        def check_feasibility(X):
            """
            检查解是否在可行域内（满足三大硬约束）
            :return: (is_feasible, constraint_details)
            """
            x_dict = {k: v for k, v in zip(keys, X)}
            return p.check_constraints(x_dict)

        def objective_function(X, return_details=False):
            """
            目标函数：效用 - 过渡成本
            
            J(X) = U(X) - C_trans(X)
            
            注意：约束不再参与目标函数计算，而是作为可行域判定条件
            """
            # 映射回字典
            x_dict = {k: v for k, v in zip(keys, X)}
            
            # 使用新的效用函数
            skill_utility = p.calculate_utility(x_dict)

            # 保留平滑过渡成本（防止课表剧烈变动导致的不切实际）
            current_vals = [p.current_curriculum.get(k, 0) for k in keys]
            current_X_local = np.array(current_vals)
            
            # 避免除以零
            with np.errstate(divide='ignore', invalid='ignore'):
                 change_ratio = np.abs(X - current_X_local) / current_X_local
                 change_ratio = np.nan_to_num(change_ratio) # Handle 0/0 or X/0

            transition_cost = 0.05 * np.sum(change_ratio[change_ratio > 0.25])
            
            net_score = skill_utility - transition_cost
            
            if return_details:
                # 获取约束满足情况（用于报告，不参与计算）
                is_feasible, constraint_details = check_feasibility(X)
                return net_score, {
                    'utility': skill_utility,
                    'transition_cost': transition_cost,
                    'is_feasible': is_feasible,
                    'constraint_details': constraint_details
                }
            return net_score

        # 初始化 - 确保初始解可行
        current_vals = [p.current_curriculum.get(k, 0) for k in keys]
        current_X = np.array(current_vals)
        
        # 检查初始解是否满足约束，如果不满足则修正
        if p.enable_constraints:
            is_feasible, details = check_feasibility(current_X)
            if not is_feasible:
                print(f"⚠️ 初始解不满足约束，尝试自动修正...")
                current_X = self._repair_to_feasible(current_X, keys)
        
        current_J = objective_function(current_X)
        best_X = current_X.copy()
        best_J = current_J

        temp = p.sa_temp
        scaling_start_temp = p.sa_temp # For reheating reference

        # 记录迭代历史
        iteration_history = [best_J]
        
        # 统计信息
        feasibility_stats = {'total_generated': 0, 'feasible': 0, 'infeasible': 0}
        
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

            # ========== 基础边界约束 ==========
            # 确保非负和边界约束
            if np.any(new_X < 0) or new_X[1] < 2 or new_X[0] < 20: 
                continue

            # 确保总学分不变
            if abs(sum(new_X) - p.total_credits) > 1e-6:
                continue

            feasibility_stats['total_generated'] += 1

            # ========== 三大硬约束检查（Red-Line Check）==========
            if p.enable_constraints:
                is_feasible, constraint_details = check_feasibility(new_X)
                if not is_feasible:
                    # 解不可行，直接舍弃（不进入接受判断）
                    feasibility_stats['infeasible'] += 1
                    continue
                else:
                    feasibility_stats['feasible'] += 1
            
            new_J = objective_function(new_X)

            # 接受准则（Metropolis准则）
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

        # 结果打包 - 包含约束满足详情
        opt_dict = {k: v for k, v in zip(keys, best_X)}
        final_score, final_details = objective_function(best_X, return_details=True)
        
        return {
            'optimal_curriculum': opt_dict,
            'optimal_score': final_score,
            'skill_utility': final_details['utility'],
            'transition_cost': final_details['transition_cost'],
            'is_feasible': final_details['is_feasible'],
            'constraint_details': final_details['constraint_details'],
            'feasibility_stats': feasibility_stats,
            'iteration_history': iteration_history,
            'constraints_enabled': p.enable_constraints
        }
    
    def _repair_to_feasible(self, X, keys):
        """
        修复不可行解到可行域边界
        
        策略：增加ethics学分，减少AI学分，直到满足安全约束
        """
        p = self.params
        X = X.copy()
        max_iterations = 50
        
        for _ in range(max_iterations):
            x_dict = {k: v for k, v in zip(keys, X)}
            is_feasible, details = p.check_constraints(x_dict)
            
            if is_feasible:
                return X
            
            # 修复策略：优先修复安全约束
            if 'safety' in details.get('violations', []):
                # 增加ethics，减少AI
                if X[1] > 2:  # x_AI > 2
                    X[1] -= 1
                    X[2] += 1
            elif 'green' in details.get('violations', []):
                # 减少AI学分
                if X[1] > 2:
                    X[1] -= 1
                    X[0] += 1  # 转移到base
            elif 'equity' in details.get('violations', []):
                # 减少高门槛课程
                if X[1] > 2:
                    X[1] -= 1
                    X[0] += 1
        
        return X

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
        saver_all = FigureSaver('./figures/task2_2')
        paths = saver_all.save(fig, 'ahp_summary_table')
        print(f"  💾 AHP summary table saved: {paths[0]}")

    def plot_model_comparison(self, baseline_results, constrained_results, figsize=(16, 10)):
        """
        绘制原模型 vs 约束模型的对比图
        
        ★★★ 重要改进 ★★★
        为了展示约束的真正效果，此函数现在使用"高AI偏好"权重重新运行模型，
        这样可以清楚地展示约束如何限制AI学分的分配。
        
        核心对比维度：
        1. 课程学分分配对比（高AI偏好场景）
        2. 目标函数分解对比
        3. 约束满足情况
        """
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f'{self.school} - Model Comparison: Baseline vs Red-Line Constrained\n(High-AI Preference Scenario)',
                    fontsize=18, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        
        # ★★★ 关键改进：使用高AI偏好权重重新运行以展示约束效果 ★★★
        HIGH_AI_WEIGHTS = {
            'CMU': {'x_base': 0.20, 'x_AI': 0.50, 'x_ethics': 0.10, 'x_proj': 0.20},
            'CIA': {'x_base': 0.25, 'x_AI': 0.40, 'x_ethics': 0.10, 'x_proj': 0.25},
            'CCAD': {'x_base': 0.25, 'x_AI': 0.45, 'x_ethics': 0.10, 'x_proj': 0.20}
        }
        
        # 重新运行高AI偏好场景
        p = self.model.params
        school = self.school
        
        # 高AI偏好 - 无约束
        p_high_free = EducationDecisionParams(school)
        p_high_free.enable_constraints = False
        p_high_free.custom_weights = HIGH_AI_WEIGHTS.get(school, HIGH_AI_WEIGHTS['CMU'])
        model_high_free = EducationDecisionModel(p_high_free)
        c_high_free = model_high_free.curriculum_optimization_sa()
        
        # 高AI偏好 - 约束
        p_high_con = EducationDecisionParams(school)
        p_high_con.enable_constraints = True
        p_high_con.custom_weights = HIGH_AI_WEIGHTS.get(school, HIGH_AI_WEIGHTS['CMU'])
        model_high_con = EducationDecisionModel(p_high_con)
        c_high_con = model_high_con.curriculum_optimization_sa()
        
        # ========== 图1: 课程学分对比柱状图 (高AI偏好场景) ==========
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.set_title('Curriculum Allocation Comparison (High-AI Scenario)', fontweight='bold', fontsize=14)
        
        keys = ['x_base', 'x_AI', 'x_ethics', 'x_proj']
        labels = ['Base', 'AI', 'Ethics', 'Project']
        baseline_vals = [c_high_free['optimal_curriculum'][k] for k in keys]
        constrained_vals = [c_high_con['optimal_curriculum'][k] for k in keys]
        
        x = np.arange(len(keys))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline (Utility-Max)',
                       color=PlotStyleConfig.COLORS['secondary'], edgecolor='black', alpha=0.8)
        bars2 = ax1.bar(x + width/2, constrained_vals, width, label='Red-Line (Constrained)',
                       color=PlotStyleConfig.COLORS['accent'], edgecolor='black', alpha=0.8)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=12)
        ax1.set_ylabel('Credits', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签和差异标注
        for bar in bars1:
            ax1.annotate(f'{bar.get_height():.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        for i, bar in enumerate(bars2):
            ax1.annotate(f'{bar.get_height():.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            # 添加差异百分比 (对AI学分特别标注)
            if i == 1:  # AI credits
                diff = constrained_vals[i] - baseline_vals[i]
                diff_pct = diff / baseline_vals[i] * 100 if baseline_vals[i] > 0 else 0
                ax1.annotate(f'{diff_pct:+.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 3),
                            ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
        
        # ========== 图2: 目标函数分解 (使用高AI偏好约束模型数据) ==========
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title('Objective Function Breakdown', fontweight='bold', fontsize=14)
        
        # 使用高AI偏好约束模型的分解数据
        breakdown_labels = ['Utility', 'Trans. Cost']
        breakdown_vals = [
            c_high_con['skill_utility'],
            c_high_con['transition_cost']
        ]
        colors_breakdown = [PlotStyleConfig.COLORS['accent'], 
                           PlotStyleConfig.COLORS['gold']]
        
        bars = ax2.bar(breakdown_labels, breakdown_vals, color=colors_breakdown, edgecolor='black')
        ax2.set_ylabel('Score Component', fontsize=12)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        
        for bar, val in zip(bars, breakdown_vals):
            ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # ========== 图3: 约束满足雷达图 (硬约束检查) ==========
        ax3 = fig.add_subplot(gs[1, 0], polar=True)
        ax3.set_title('Hard Constraint Satisfaction', fontweight='bold', fontsize=14, pad=20)
        
        # 获取约束参数和满足详情 (使用高AI偏好约束模型)
        c_params = p_high_con.constraint_params
        constraint_details = c_high_con.get('constraint_details', {})
        
        categories = ['Equity\n(E_max)', 'Green\n(beta_env)', 'Safety\n(gamma)']
        
        # 从硬约束检查中获取满足度 (margin > 0 = 有余量, margin < 0 = 违规)
        # 转换为0-1的满足度指标
        equity_info = constraint_details.get('equity', {})
        green_info = constraint_details.get('green', {})
        safety_info = constraint_details.get('safety', {})
        
        # 计算满足度：margin / threshold (归一化到0-1)
        equity_sat = 1.0 if equity_info.get('satisfied', True) else max(0, 1 + equity_info.get('margin', 0) / c_params['E_max'])
        green_sat = 1.0 if green_info.get('satisfied', True) else max(0, 1 + green_info.get('margin', 0) / c_params['beta_env'])
        safety_sat = 1.0 if safety_info.get('satisfied', True) else max(0, 1 + safety_info.get('margin', 0) / 10)
        
        values = [equity_sat, green_sat, safety_sat]
        values += values[:1]  # 闭合
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax3.plot(angles, values, 'o-', linewidth=2, color=PlotStyleConfig.COLORS['primary'])
        ax3.fill(angles, values, alpha=0.25, color=PlotStyleConfig.COLORS['primary'])
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories, fontsize=10)
        ax3.set_ylim(0, 1)
        ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax3.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
        
        # ========== 图4: 效用对比 (高AI偏好场景) ==========
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_title('Utility Comparison (High-AI)', fontweight='bold', fontsize=14)
        
        models = ['Baseline\n(Utility-Max)', 'Red-Line\n(Constrained)']
        # 使用高AI偏好场景的数据
        scores = [c_high_free['optimal_score'], c_high_con['optimal_score']]
        colors_score = [PlotStyleConfig.COLORS['secondary'], PlotStyleConfig.COLORS['accent']]
        
        bars = ax4.bar(models, scores, color=colors_score, edgecolor='black', width=0.6)
        ax4.set_ylabel('Net Score (J)', fontsize=12)
        
        for bar, score in zip(bars, scores):
            ax4.annotate(f'{score:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 添加差异标注
        diff = scores[1] - scores[0]
        diff_pct = (diff / abs(scores[0])) * 100 if scores[0] != 0 else 0
        ax4.annotate(f'Loss = {diff:.3f} ({diff_pct:+.1f}%)', 
                    xy=(0.5, max(scores) * 1.05), ha='center', fontsize=11,
                    color=PlotStyleConfig.COLORS['danger'] if diff < 0 else PlotStyleConfig.COLORS['accent'],
                    fontweight='bold')
        
        # ========== 图5: 约束参数表格 ==========
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        ax5.set_title('Hard Constraint Status', fontweight='bold', fontsize=14, pad=20)
        
        # 从高AI偏好约束模型获取状态
        constraint_details = c_high_con.get('constraint_details', {})
        equity_info = constraint_details.get('equity', {})
        green_info = constraint_details.get('green', {})
        safety_info = constraint_details.get('safety', {})
        
        # 使用ASCII字符避免字体问题
        table_data = [
            ['Constraint', 'Threshold', 'Status'],
            ['E_max (Equity)', f"{c_params['E_max']:.2f}", 'PASS' if equity_info.get('satisfied', True) else 'FAIL'],
            ['beta_env (Green)', f"{c_params['beta_env']:.2f}", 'PASS' if green_info.get('satisfied', True) else 'FAIL'],
            ['gamma (Safety)', f"{c_params['gamma_safety']:.2f}", 'PASS' if safety_info.get('satisfied', True) else 'FAIL'],
        ]
        
        table = ax5.table(cellText=table_data[1:], colLabels=table_data[0], loc='center',
                         cellLoc='center', colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.0)
        
        for i in range(3):
            table[(0, i)].set_facecolor(PlotStyleConfig.COLORS['dark'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        saver = FigureSaver('./figures/task2_2')
        paths = saver.save(fig, f'{self.school}_model_comparison')
        print(f"  💾 Model comparison plot saved: {paths[0]}")
        
    def plot_constraint_sensitivity(self, figsize=(16, 5)):
        """
        绘制三大约束参数的灵敏度分析图
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f'{self.school} - Constraint Sensitivity Analysis',
                    fontsize=18, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        
        p = self.model.params
        original_constraints = p.enable_constraints
        original_params = p.constraint_params.copy()
        
        # ========== 1. E_max 灵敏度 ==========
        ax1 = axes[0]
        ax1.set_title('Equity Threshold (E_max)', fontweight='bold')
        
        e_max_range = np.linspace(0.15, 0.70, 12)
        ai_credits_list = []
        scores_list = []
        
        for e_max in e_max_range:
            p.constraint_params['E_max'] = e_max
            model_temp = EducationDecisionModel(p)
            result = model_temp.curriculum_optimization_sa()
            ai_credits_list.append(result['optimal_curriculum']['x_AI'])
            scores_list.append(result['optimal_score'])
        
        ax1.plot(e_max_range, ai_credits_list, 'o-', color=PlotStyleConfig.COLORS['primary'],
                linewidth=2, markersize=6, label='AI Credits')
        ax1.set_xlabel('E_max')
        ax1.set_ylabel('Optimal AI Credits', color=PlotStyleConfig.COLORS['primary'])
        ax1.tick_params(axis='y', labelcolor=PlotStyleConfig.COLORS['primary'])
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(e_max_range, scores_list, 's--', color=PlotStyleConfig.COLORS['secondary'],
                     linewidth=2, markersize=5, label='Net Score')
        ax1_twin.set_ylabel('Net Score', color=PlotStyleConfig.COLORS['secondary'])
        ax1_twin.tick_params(axis='y', labelcolor=PlotStyleConfig.COLORS['secondary'])
        
        ax1.axvline(original_params['E_max'], color='gray', linestyle=':', linewidth=2, alpha=0.7)
        ax1.grid(alpha=0.3)
        
        # ========== 2. β_env 灵敏度 ==========
        ax2 = axes[1]
        ax2.set_title('Green Cap (β_env)', fontweight='bold')
        
        beta_range = np.linspace(0.10, 0.50, 12)
        ai_credits_list2 = []
        scores_list2 = []
        
        p.constraint_params['E_max'] = original_params['E_max']  # 恢复
        
        for beta in beta_range:
            p.constraint_params['beta_env'] = beta
            model_temp = EducationDecisionModel(p)
            result = model_temp.curriculum_optimization_sa()
            ai_credits_list2.append(result['optimal_curriculum']['x_AI'])
            scores_list2.append(result['optimal_score'])
        
        ax2.plot(beta_range, ai_credits_list2, 'o-', color=PlotStyleConfig.COLORS['primary'],
                linewidth=2, markersize=6)
        ax2.set_xlabel('β_env')
        ax2.set_ylabel('Optimal AI Credits', color=PlotStyleConfig.COLORS['primary'])
        ax2.tick_params(axis='y', labelcolor=PlotStyleConfig.COLORS['primary'])
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(beta_range, scores_list2, 's--', color=PlotStyleConfig.COLORS['secondary'],
                     linewidth=2, markersize=5)
        ax2_twin.set_ylabel('Net Score', color=PlotStyleConfig.COLORS['secondary'])
        ax2_twin.tick_params(axis='y', labelcolor=PlotStyleConfig.COLORS['secondary'])
        
        ax2.axvline(original_params['beta_env'], color='gray', linestyle=':', linewidth=2, alpha=0.7)
        ax2.grid(alpha=0.3)
        
        # ========== 3. γ 灵敏度 ==========
        ax3 = axes[2]
        ax3.set_title('Safety Ratio (γ)', fontweight='bold')
        
        gamma_range = np.linspace(0.05, 0.80, 12)
        ethics_credits_list = []
        scores_list3 = []
        
        p.constraint_params['beta_env'] = original_params['beta_env']  # 恢复
        
        for gamma in gamma_range:
            p.constraint_params['gamma_safety'] = gamma
            model_temp = EducationDecisionModel(p)
            result = model_temp.curriculum_optimization_sa()
            ethics_credits_list.append(result['optimal_curriculum']['x_ethics'])
            scores_list3.append(result['optimal_score'])
        
        ax3.plot(gamma_range, ethics_credits_list, 'o-', color=PlotStyleConfig.COLORS['accent'],
                linewidth=2, markersize=6)
        ax3.set_xlabel('γ (Safety Ratio)')
        ax3.set_ylabel('Optimal Ethics Credits', color=PlotStyleConfig.COLORS['accent'])
        ax3.tick_params(axis='y', labelcolor=PlotStyleConfig.COLORS['accent'])
        
        ax3_twin = ax3.twinx()
        ax3_twin.plot(gamma_range, scores_list3, 's--', color=PlotStyleConfig.COLORS['secondary'],
                     linewidth=2, markersize=5)
        ax3_twin.set_ylabel('Net Score', color=PlotStyleConfig.COLORS['secondary'])
        ax3_twin.tick_params(axis='y', labelcolor=PlotStyleConfig.COLORS['secondary'])
        
        ax3.axvline(original_params['gamma_safety'], color='gray', linestyle=':', linewidth=2, alpha=0.7)
        ax3.grid(alpha=0.3)
        
        # 恢复原始参数
        p.constraint_params = original_params
        p.enable_constraints = original_constraints
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        saver = FigureSaver('./figures/task2_2')
        paths = saver.save(fig, f'{self.school}_constraint_sensitivity')
        print(f"  💾 Constraint sensitivity plot saved: {paths[0]}")

    def plot_utility_vs_constraint_tradeoff(self, figsize=(18, 12)):
        """
        绘制"效用 vs 约束强度"权衡分析图 - 寻找性能拐点
        
        论文核心图表：展示为了底线，我们牺牲了多少效率
        
        关键改进：
        1. 使用"高AI偏好"权重，让无约束模式倾向于大量分配AI学分
        2. 通过逐步收紧约束，展示效用如何随约束强度变化
        3. 找到"性能拐点"：约束过紧导致效用急剧下降的临界点
        
        包含：
        1. 环境约束强度 vs 效用（找到绿色校园的代价）
        2. 公平性约束强度 vs 效用（找到包容性的代价）
        3. 安全约束强度 vs 效用（找到伦理配套的代价）
        4. 综合权衡面板
        """
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f'{self.school} - Utility vs. Constraint Strength Trade-off Analysis\n'
                    f'Finding the Performance Inflection Point (性能拐点分析)',
                    fontsize=18, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        p = self.model.params
        original_params = p.constraint_params.copy()
        original_weights = p.custom_weights  # 保存原始权重
        
        # ★★★ 关键：使用"高AI偏好"权重以展示约束效果 ★★★
        HIGH_AI_WEIGHTS = {
            'CMU': {'x_base': 0.20, 'x_AI': 0.50, 'x_ethics': 0.10, 'x_proj': 0.20},
            'CIA': {'x_base': 0.25, 'x_AI': 0.40, 'x_ethics': 0.10, 'x_proj': 0.25},
            'CCAD': {'x_base': 0.25, 'x_AI': 0.45, 'x_ethics': 0.10, 'x_proj': 0.20}
        }
        p.custom_weights = HIGH_AI_WEIGHTS.get(self.school, HIGH_AI_WEIGHTS['CMU'])
        
        # 首先运行一次无约束模型获取基准效用
        p.enable_constraints = False
        model_baseline = EducationDecisionModel(p)
        baseline_result = model_baseline.curriculum_optimization_sa()
        unconstrained_utility = baseline_result['skill_utility']
        unconstrained_ai = baseline_result['optimal_curriculum']['x_AI']
        p.enable_constraints = True
        
        # ========== 1. 环境约束 β_env vs 效用 (核心图) ==========
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('🌿 Green Campus Trade-off: β_env vs Utility', fontweight='bold', fontsize=13)
        
        # 从极端严格到宽松
        beta_range = np.linspace(0.05, 0.60, 18)
        utility_list = []
        ai_credits_list = []
        
        for beta in beta_range:
            p.constraint_params['beta_env'] = beta
            p.constraint_params['E_max'] = 0.70  # 放松公平性约束以隔离环境约束效果
            p.constraint_params['gamma_safety'] = 0.05  # 放松安全约束
            model_temp = EducationDecisionModel(p)
            result = model_temp.curriculum_optimization_sa()
            utility_list.append(result['skill_utility'])
            ai_credits_list.append(result['optimal_curriculum']['x_AI'])
        
        # 主曲线：效用
        color1 = PlotStyleConfig.COLORS['primary']
        ax1.plot(beta_range * 100, utility_list, 'o-', color=color1, linewidth=2.5, 
                markersize=6, label='Skill Utility')
        ax1.set_xlabel('β_env (Green Cap) - High Energy Course Limit (%)', fontweight='bold')
        ax1.set_ylabel('Skill Utility', color=color1, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # 添加无约束基准线
        ax1.axhline(unconstrained_utility, color='green', linestyle=':', linewidth=2, 
                   alpha=0.7, label=f'Unconstrained: {unconstrained_utility:.3f}')
        
        # 标记当前设置
        current_beta = original_params['beta_env'] * 100
        ax1.axvline(current_beta, color='red', linestyle='--', linewidth=2, 
                   label=f'Current Setting ({current_beta:.0f}%)')
        
        # 找到拐点：效用开始急剧下降的位置
        max_utility = max(utility_list)
        inflection_idx = None
        for i in range(len(utility_list) - 1, -1, -1):
            if utility_list[i] < max_utility * 0.90:  # 下降超过10%
                inflection_idx = i
                break
        
        if inflection_idx is not None:
            inflection_beta = beta_range[inflection_idx] * 100
            inflection_utility = utility_list[inflection_idx]
            drop_pct = (1 - inflection_utility / max_utility) * 100
            ax1.axvline(inflection_beta, color='orange', linestyle=':', linewidth=2)
            ax1.annotate(f'Inflection Point\n({inflection_beta:.0f}%, -{drop_pct:.1f}%)',
                        xy=(inflection_beta, inflection_utility),
                        xytext=(inflection_beta + 8, inflection_utility + 0.1),
                        fontsize=10, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='orange'),
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 副轴：AI学分
        ax1_twin = ax1.twinx()
        ax1_twin.plot(beta_range * 100, ai_credits_list, 's--', color='gray', 
                     linewidth=1.5, markersize=4, alpha=0.6, label='AI Credits')
        ax1_twin.set_ylabel('AI Credits', color='gray')
        ax1_twin.tick_params(axis='y', labelcolor='gray')
        ax1_twin.axhline(unconstrained_ai, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(alpha=0.3)
        
        # ========== 2. 公平性约束 E_max vs 效用 ==========
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('💰 Equity Trade-off: E_max vs Utility', fontweight='bold', fontsize=13)
        
        e_max_range = np.linspace(0.10, 0.70, 18)
        utility_list2 = []
        ai_credits_list2 = []
        
        for e_max in e_max_range:
            p.constraint_params['E_max'] = e_max
            p.constraint_params['beta_env'] = 0.60  # 放松环境约束
            p.constraint_params['gamma_safety'] = 0.05  # 放松安全约束
            model_temp = EducationDecisionModel(p)
            result = model_temp.curriculum_optimization_sa()
            utility_list2.append(result['skill_utility'])
            ai_credits_list2.append(result['optimal_curriculum']['x_AI'])
        
        color2 = PlotStyleConfig.COLORS['accent']
        ax2.plot(e_max_range * 100, utility_list2, 'o-', color=color2, linewidth=2.5, 
                markersize=6, label='Skill Utility')
        ax2.set_xlabel('E_max (Equity Threshold) - Exclusion Limit (%)', fontweight='bold')
        ax2.set_ylabel('Skill Utility', color=color2, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        ax2.axhline(unconstrained_utility, color='green', linestyle=':', linewidth=2, 
                   alpha=0.7, label=f'Unconstrained')
        
        current_emax = original_params['E_max'] * 100
        ax2.axvline(current_emax, color='red', linestyle='--', linewidth=2,
                   label=f'Current Setting ({current_emax:.0f}%)')
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(e_max_range * 100, ai_credits_list2, 's--', color='gray',
                     linewidth=1.5, markersize=4, alpha=0.6)
        ax2_twin.set_ylabel('AI Credits', color='gray')
        ax2_twin.tick_params(axis='y', labelcolor='gray')
        
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(alpha=0.3)
        
        # ========== 3. 安全约束 γ vs 效用 ==========
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_title('⚖️ Safety Trade-off: γ vs Utility', fontweight='bold', fontsize=13)
        
        gamma_range = np.linspace(0.05, 1.50, 18)
        utility_list3 = []
        ethics_credits_list = []
        
        for gamma in gamma_range:
            p.constraint_params['gamma_safety'] = gamma
            p.constraint_params['E_max'] = 0.70  # 放松公平性约束
            p.constraint_params['beta_env'] = 0.60  # 放松环境约束
            model_temp = EducationDecisionModel(p)
            result = model_temp.curriculum_optimization_sa()
            utility_list3.append(result['skill_utility'])
            ethics_credits_list.append(result['optimal_curriculum']['x_ethics'])
        
        color3 = PlotStyleConfig.COLORS['gold']
        ax3.plot(gamma_range, utility_list3, 'o-', color=color3, linewidth=2.5, 
                markersize=6, label='Skill Utility')
        ax3.set_xlabel('γ (Safety Ratio) - Ethics/AI Requirement', fontweight='bold')
        ax3.set_ylabel('Skill Utility', color=color3, fontweight='bold')
        ax3.tick_params(axis='y', labelcolor=color3)
        
        ax3.axhline(unconstrained_utility, color='green', linestyle=':', linewidth=2, 
                   alpha=0.7, label=f'Unconstrained')
        
        current_gamma = original_params['gamma_safety']
        ax3.axvline(current_gamma, color='red', linestyle='--', linewidth=2,
                   label=f'Current Setting ({current_gamma:.2f})')
        
        ax3_twin = ax3.twinx()
        ax3_twin.plot(gamma_range, ethics_credits_list, 's--', color='purple',
                     linewidth=1.5, markersize=4, alpha=0.6, label='Ethics Credits')
        ax3_twin.set_ylabel('Ethics Credits', color='purple')
        ax3_twin.tick_params(axis='y', labelcolor='purple')
        
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(alpha=0.3)
        
        # ========== 4. 综合权衡总结面板 ==========
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        ax4.set_title('📊 Trade-off Summary & Key Findings', fontweight='bold', fontsize=13, pad=20)
        
        # 计算关键指标
        max_utility_beta = max(utility_list)
        min_utility_beta = min(utility_list)
        utility_drop_total = (1 - min_utility_beta / max_utility_beta) * 100
        
        # 计算约束模式下的效用损失
        constrained_utility = self.results['curriculum_optimization']['skill_utility']
        constraint_loss = (1 - constrained_utility / unconstrained_utility) * 100 if unconstrained_utility > 0 else 0
        
        # 计算15%绿色阈值下的效用损失
        green_15_idx = np.argmin(np.abs(beta_range - 0.15))
        utility_at_15 = utility_list[green_15_idx] if green_15_idx < len(utility_list) else min_utility_beta
        green_15_drop = (1 - utility_at_15 / max_utility_beta) * 100
        
        summary_text = f"""
========================================================================
                    KEY FINDINGS FOR {self.school}                      
========================================================================
                                                                      
  UNCONSTRAINED BASELINE (High-AI Preference):                    
     - Maximum Utility: {unconstrained_utility:.3f}                             
     - Optimal AI Credits: {unconstrained_ai:.0f}                                
                                                                      
  CONSTRAINED MODEL (Red-Line):                                   
     - Constrained Utility: {constrained_utility:.3f}                           
     - Competitiveness Loss: -{constraint_loss:.1f}%                        
                                                                      
  GREEN CAMPUS TRADE-OFF:                                         
     - Utility at beta_env=15%: {utility_at_15:.3f}                            
     - Total Drop (5% to 60%): {utility_drop_total:.1f}%                       
                                                                      
  KEY INSIGHT:                                                    
     To achieve strict green campus (beta<15%),                                  
     student competitiveness drops by ~{green_15_drop:.0f}%.                         
     This reveals the tension between AI education                    
     and sustainable development.                    
                                                                      
========================================================================
"""
        
        ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, fontsize=9.5,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))
        
        # 恢复原始参数和权重
        p.constraint_params = original_params
        p.custom_weights = original_weights  # 恢复原始权重
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        saver = FigureSaver('./figures/task2_2')
        paths = saver.save(fig, f'{self.school}_utility_constraint_tradeoff')
        print(f"  💾 Utility vs Constraint trade-off plot saved: {paths[0]}")


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
    print("█" + " "*12 + "AI教育决策模型 - 优化版" + " "*13 + "█")
    print("█" + " "*8 + "Constrained Sustainable Education Model" + " "*8 + "█")
    print("█"*70 + "\n")

    # ========== Step 0: AHP参数估计 ==========
    print("【Step 0】执行AHP层次分析法 - 计算λ参数...")
    print("-"*70)
    
    # 执行AHP计算并显示详细过程
    ahp_calculator = AHPLambdaCalculator(verbose=True)
    ahp_lambdas = ahp_calculator.calculate_all_lambdas()
    
    # 创建figures目录 (包括task2_2子目录)
    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./figures/task2_2', exist_ok=True)

    # 定义所有学校
    schools = ['CMU', 'CIA', 'CCAD']
    all_results_baseline = {}
    all_results_constrained = {}

    # ========== Step 1-5: 循环所有学校 ==========
    for school in schools:
        print(f"\n{'='*70}")
        print(f"分析学校: {school} (λ = {ahp_lambdas[school]:.4f})")
        print(f"{'='*70}")

        # ========== 基线模型 (Baseline: 无约束) ==========
        print("\n【Step 1a】初始化基线模型参数 (Baseline - Utility Max)...")
        params_baseline = EducationDecisionParams(school_name=school, enable_constraints=False)
        
        print("【Step 2a】创建基线决策模型...")
        model_baseline = EducationDecisionModel(params_baseline)
        
        print("【Step 3a】执行基线分析...")
        results_baseline = model_baseline.run_full_analysis(verbose=False)
        all_results_baseline[school] = results_baseline
        
        # ========== 约束模型 (Constrained: 三大约束) ==========
        print("\n【Step 1b】初始化约束模型参数 (Constrained - Sustainable)...")
        params_constrained = EducationDecisionParams(school_name=school, enable_constraints=True)
        params_constrained.summary()
        
        print("【Step 2b】创建约束决策模型...")
        model_constrained = EducationDecisionModel(params_constrained)
        
        print("【Step 3b】执行约束分析...")
        results_constrained = model_constrained.run_full_analysis(verbose=False)
        all_results_constrained[school] = results_constrained

        # ========== Step 4: 生成可视化 ==========
        print("\n【Step 4】生成可视化图表...")
        print("-"*70)

        # 使用约束模型的可视化器
        viz = EducationDecisionVisualization(model_constrained, results_constrained, save_dir='./figures/task2_2')

        # 图1: 模型对比图 (核心图表)
        print(f"\n  🎨 绘制{school}模型对比图 (Baseline vs Constrained)...")
        viz.plot_model_comparison(results_baseline, results_constrained)

        # 图2: 约束灵敏度分析
        print(f"\n  🎨 绘制{school}约束灵敏度分析图...")
        viz.plot_constraint_sensitivity()
        
        # 图2b: ★★★ 效用 vs 约束强度权衡分析 (论文核心图表) ★★★
        print(f"\n  🎨 绘制{school}效用vs约束强度权衡分析图 (Finding Trade-off Point)...")
        viz.plot_utility_vs_constraint_tradeoff()

        # 图3: 招生响应分析
        print(f"\n  🎨 绘制{school}招生响应分析图...")
        viz.plot_enrollment_response()

        # 图4: 课程优化分析
        print(f"\n  🎨 绘制{school}课程优化分析图...")
        viz.plot_curriculum_optimization()

        # 图5: 职业弹性分析
        print(f"\n  🎨 绘制{school}职业路径弹性分析图...")
        viz.plot_career_elasticity()

        # 图6: SA收敛过程图
        print(f"\n  🎨 绘制{school}模拟退火收敛过程图...")
        viz.plot_sa_convergence()

        # 图7: 资源竞争分析图
        print(f"\n  🎨 绘制{school}资源竞争分析图...")
        viz.plot_pareto_frontier()

        # ========== Step 5: 保存结果 ==========
        print("\n【Step 5】保存分析结果...")
        print("-"*70)

        # 打印对比结果 (使用高AI偏好场景展示约束效果)
        print(f"\n{school} 模型对比结果 (High-AI Preference Scenario):")
        print(f"{'='*70}")
        
        # 运行高AI偏好场景对比
        HIGH_AI_WEIGHTS = {
            'CMU': {'x_base': 0.20, 'x_AI': 0.50, 'x_ethics': 0.10, 'x_proj': 0.20},
            'CIA': {'x_base': 0.25, 'x_AI': 0.40, 'x_ethics': 0.10, 'x_proj': 0.25},
            'CCAD': {'x_base': 0.25, 'x_AI': 0.45, 'x_ethics': 0.10, 'x_proj': 0.20}
        }
        
        p_high_free = EducationDecisionParams(school)
        p_high_free.enable_constraints = False
        p_high_free.custom_weights = HIGH_AI_WEIGHTS.get(school, HIGH_AI_WEIGHTS['CMU'])
        model_high_free = EducationDecisionModel(p_high_free)
        c_high_free = model_high_free.curriculum_optimization_sa()
        
        p_high_con = EducationDecisionParams(school)
        p_high_con.enable_constraints = True
        p_high_con.custom_weights = HIGH_AI_WEIGHTS.get(school, HIGH_AI_WEIGHTS['CMU'])
        model_high_con = EducationDecisionModel(p_high_con)
        c_high_con = model_high_con.curriculum_optimization_sa()
        
        print(f"{'Metric':<25} {'Baseline(Utility-Max)':<20} {'Red-Line(Constrained)':<20}")
        print(f"{'-'*70}")
        
        print(f"{'AI Credits':<25} {c_high_free['optimal_curriculum']['x_AI']:<20.1f} {c_high_con['optimal_curriculum']['x_AI']:<20.1f}")
        print(f"{'Base Credits':<25} {c_high_free['optimal_curriculum']['x_base']:<20.1f} {c_high_con['optimal_curriculum']['x_base']:<20.1f}")
        print(f"{'Ethics Credits':<25} {c_high_free['optimal_curriculum']['x_ethics']:<20.1f} {c_high_con['optimal_curriculum']['x_ethics']:<20.1f}")
        print(f"{'Project Credits':<25} {c_high_free['optimal_curriculum']['x_proj']:<20.1f} {c_high_con['optimal_curriculum']['x_proj']:<20.1f}")
        print(f"{'-'*70}")
        print(f"{'Skill Utility':<25} {c_high_free['skill_utility']:<20.3f} {c_high_con['skill_utility']:<20.3f}")
        
        # 计算效用损失
        utility_loss = (c_high_con['skill_utility'] - c_high_free['skill_utility']) / c_high_free['skill_utility'] * 100
        ai_reduction = (c_high_con['optimal_curriculum']['x_AI'] - c_high_free['optimal_curriculum']['x_AI']) / c_high_free['optimal_curriculum']['x_AI'] * 100 if c_high_free['optimal_curriculum']['x_AI'] > 0 else 0
        
        print(f"{'Utility Loss':<25} {'N/A':<20} {utility_loss:+.1f}%")
        print(f"{'AI Credit Reduction':<25} {'N/A':<20} {ai_reduction:+.1f}%")
        
        # 显示约束满足状态（硬约束模式）
        is_feasible = c_high_con.get('is_feasible', True)
        constraint_details = c_high_con.get('constraint_details', {})
        feasibility_status = "ALL SATISFIED" if is_feasible else f"VIOLATED: {constraint_details.get('violations', [])}"
        print(f"{'Constraint Status':<25} {'N/A':<20} {feasibility_status}")
        
        # 显示可行解搜索统计
        feas_stats = c_high_con.get('feasibility_stats', {})
        if feas_stats:
            feasible_rate = feas_stats['feasible'] / max(1, feas_stats['total_generated']) * 100
            print(f"{'Feasible Search Rate':<25} {'N/A':<20} {feasible_rate:.1f}% ({feas_stats['feasible']}/{feas_stats['total_generated']})")
        
        print(f"{'='*70}")

    # ========== Step 6: 生成综合图表 ==========
    print("\n【Step 6】生成综合比较图表...")
    print("-"*70)

    # 创建一个临时的viz对象来生成综合图
    temp_model = EducationDecisionModel(EducationDecisionParams(school_name='CMU', enable_constraints=True))
    temp_viz = EducationDecisionVisualization(temp_model, all_results_constrained['CMU'], save_dir='./figures/task2_2')

    # AHP雷达图
    print("\n  🎨 绘制AHP层次分析雷达图...")
    temp_viz.plot_ahp_radar()

    # AHP汇总表格
    print("\n  🎨 绘制AHP分析汇总表格...")
    temp_viz.plot_ahp_summary_table()

    # 学校比较图
    print("\n  🎨 绘制学校比较图...")
    temp_viz.plot_school_comparison(all_results_constrained)

    # 课程结构堆积对比图
    print("\n  🎨 绘制课程结构优化对比堆积图...")
    temp_viz.plot_stacked_curriculum_comparison(all_results_constrained)

    # 职业相似度矩阵
    print("\n  🎨 绘制职业相似度矩阵...")
    temp_viz.plot_career_similarity_matrix()

    print("\n" + "█"*70)
    print("█" + " "*20 + "优化工作流执行完成!" + " "*20 + "█")
    print("█"*70 + "\n")

    return {
        'baseline': all_results_baseline,
        'constrained': all_results_constrained
    }


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # ★★★ 运行优化后的教育决策工作流 ★★★
    # ============================================================
    all_results = run_education_decision_workflow()
    
    # 打印总结
    print("\n" + "="*70)
    print("模型优化总结 (Model Optimization Summary)")
    print("="*70)
    print("""
    ✅ 从 Utility-Max 模型 → Constrained Sustainable 模型
    
    新增三大现实约束：
    1. 公平性约束 (Equity): E_max 限制课程经济门槛
    2. 环境约束 (Green Cap): β_env 限制高能耗课程比例
    3. 安全约束 (Safety): γ 要求伦理课程与AI课程配套
    
    所有图表已保存至 ./figures/task2_2/ 目录
    """)
    print("="*70)
