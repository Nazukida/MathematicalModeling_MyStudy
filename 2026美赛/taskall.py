"""
============================================================
AI 职业演化综合模型 - 完整工作流
(AI Career Evolution Comprehensive Models - Complete Workflow)
============================================================
功能：实现四个高级模型用于AI对职业的影响分析
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================

模型框架：
1. 模型 I：职业演化动态模型（SD + 贝叶斯网络）
2. 模型 II：教育决策优化模型 (AHP + MOEA/D)
3. 模型 III：综合成功评价模型（模糊评价 + 相关性分析）
4. 模型 IV：泛化与推广模型 (CBR + GWR)
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# 图表配置（内联版本，避免导入问题）
# ============================================================

class PlotStyleConfig:
    """图表美化配置类"""

    COLORS = {
        'primary': '#1f77b4',  # 深蓝 - 历史/基准
        'secondary': '#ff7f0e',  # 橙色 - AI影响/预测
        'accent': '#2ca02c',  # 绿色 - 成功/突出
        'danger': '#d62728',  # 红色 - 危险/起始点
        'neutral': '#7f7f7f',  # 灰色 - 中性
        'background': '#f8f9fa',  # 极浅灰背景
        'grid': '#e9ecef'  # 浅灰网格
    }

    PALETTE = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896']

    @staticmethod
    def setup_style(style='academic'):
        """设置学术风格"""
        plt.style.use('default')  # 使用默认风格作为基础
        rcParams['font.family'] = 'DejaVu Sans'  # 或者 'SimHei' 如果支持中文
        rcParams['font.size'] = 12
        rcParams['axes.labelsize'] = 14
        rcParams['axes.titlesize'] = 16
        rcParams['xtick.labelsize'] = 12
        rcParams['ytick.labelsize'] = 12
        rcParams['legend.fontsize'] = 12
        rcParams['figure.titlesize'] = 18

        # 网格和背景
        rcParams['axes.grid'] = True
        rcParams['grid.alpha'] = 0.3
        rcParams['axes.facecolor'] = PlotStyleConfig.COLORS['background']

    @staticmethod
    def get_palette(n=None):
        """获取调色板"""
        if n is None:
            return PlotStyleConfig.PALETTE
        return PlotStyleConfig.PALETTE[:n] if n <= len(PlotStyleConfig.PALETTE) else PlotStyleConfig.PALETTE * (n // len(PlotStyleConfig.PALETTE)) + PlotStyleConfig.PALETTE[:n % len(PlotStyleConfig.PALETTE)]


class FigureSaver:
    """图表保存工具类"""

    def __init__(self, save_dir='./figures', format='png'):
        self.save_dir = save_dir
        self.format = format
        os.makedirs(save_dir, exist_ok=True)

    def save(self, fig, filename, formats=None, tight=True):
        if formats is None:
            formats = [self.format]
        if tight:
            plt.tight_layout()
        paths = []
        for fmt in formats:
            path = os.path.join(self.save_dir, f"{filename}.{fmt}")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            paths.append(path)
        plt.close(fig)  # 关闭图表以释放内存
        return paths


# 设置绘图风格
PlotStyleConfig.setup_style('academic')


# ============================================================
# 第一部分：模型 I - 职业演化动态模型（SD + 贝叶斯网络）
# ============================================================

class SDParams:
    """
    系统动力学模型参数配置类
    """

    def __init__(self, occupation_name='chef'):
        self.occupation_name = occupation_name

        # 职业特定参数
        self.params = {
            'chef': {
                'alpha': 0.1,  # 创造效应系数
                'beta': 0.15,  # 替代效应系数 (厨师较低，因为物理操作)
                'gamma': 0.2,  # 权重系数
                'initial_L': 100,  # 初始劳动力需求
                'initial_T': 0.1,  # 初始技术成熟度
                'initial_S': 0.8,  # 初始技能匹配度 (厨师较高)
                'initial_H': 0.9   # 初始人类核心素养 (厨师较高)
            },
            'software_engineer': {
                'alpha': 0.2,
                'beta': 0.4,
                'gamma': 0.3,
                'initial_L': 150,
                'initial_T': 0.1,
                'initial_S': 0.9,
                'initial_H': 0.85
            },
            'graphic_designer': {
                'alpha': 0.15,
                'beta': 0.25,
                'gamma': 0.25,
                'initial_L': 80,
                'initial_T': 0.1,
                'initial_S': 0.85,
                'initial_H': 0.8
            }
        }

        # 设置当前职业参数
        if self.occupation_name in self.params:
            p = self.params[self.occupation_name]
            self.alpha = p['alpha']
            self.beta = p['beta']
            self.gamma = p['gamma']
            self.initial_L = p['initial_L']
            self.initial_T = p['initial_T']
            self.initial_S = p['initial_S']
            self.initial_H = p['initial_H']
        else:
            self.alpha = 0.1
            self.beta = 0.15
            self.gamma = 0.2
            self.initial_L = 100
            self.initial_T = 0.1
            self.initial_S = 0.8
            self.initial_H = 0.9


class SDModel:
    """
    系统动力学模型类
    """

    def __init__(self, params: SDParams):
        self.params = params

    def system_dynamics(self, y, t):
        """
        系统动力学微分方程 (增强版，针对 Gen-AI)

        dy/dt = f(y, t)

        y = [L, T, S, H]  # L: 劳动力需求, T: 技术成熟度, S: 技能匹配度, H: 人类核心素养
        """
        L, T, S, H = y
        p = self.params

        # 劳动力需求：引入逻辑斯谛增长模拟非线性冲击
        K = 200  # 承载容量 (最大劳动力需求)
        dL_dt = p.alpha * T * S * (1 - L/K) - p.beta * T * L

        # 技术成熟度：逻辑斯谛增长
        dT_dt = p.gamma * T * (1 - T)

        # 技能匹配度：随时间提升，但随技术进步下降
        dS_dt = 0.05 * (1 - S) - 0.1 * dT_dt

        # 人类核心素养：创造力溢价 vs 技能萎缩
        creativity_premium = 0.1 * T * H  # 创造力溢价
        skill_atrophy = 0.05 * T * (1 - H)  # 技能萎缩
        dH_dt = creativity_premium - skill_atrophy

        return [dL_dt, dT_dt, dS_dt, dH_dt]

    def simulate(self, t_span):
        """
        模拟系统动力学

        :param t_span: 时间跨度
        :return: 模拟结果
        """
        y0 = [self.params.initial_L, self.params.initial_T, self.params.initial_S, self.params.initial_H]
        t = np.linspace(0, t_span, 100)
        sol = odeint(self.system_dynamics, y0, t)

        return {
            'time': t,
            'labor_demand': sol[:, 0],
            'tech_maturity': sol[:, 1],
            'skill_matching': sol[:, 2],
            'human_core_competence': sol[:, 3]
        }


class BNParams:
    """
    贝叶斯网络参数配置类
    """

    def __init__(self):
        # 条件概率表 (CPT)
        # P(High_Impact | Tech_Breakthrough, Policy)
        self.cpt_high_impact = {
            ('True', 'Supportive'): 0.8,
            ('True', 'Neutral'): 0.6,
            ('True', 'Restrictive'): 0.3,
            ('False', 'Supportive'): 0.4,
            ('False', 'Neutral'): 0.2,
            ('False', 'Restrictive'): 0.1
        }

        # 先验概率
        self.prior_tech_breakthrough = 0.3
        self.prior_policy = {'Supportive': 0.4, 'Neutral': 0.4, 'Restrictive': 0.2}


class BNModel:
    """
    贝叶斯网络模型类
    """

    def __init__(self, params: BNParams):
        self.params = params

    def compute_probability(self, tech_breakthrough, policy):
        """
        计算高冲击概率

        :param tech_breakthrough: 技术突破 (True/False)
        :param policy: 政策 ('Supportive'/'Neutral'/'Restrictive')
        :return: 高冲击概率
        """
        key = (str(tech_breakthrough), policy)
        return self.params.cpt_high_impact.get(key, 0.5)

    def predict_impact(self, scenarios):
        """
        预测不同情景下的冲击

        :param scenarios: 情景列表 [(tech, policy), ...]
        :return: 预测结果
        """
        results = []
        for tech, policy in scenarios:
            prob = self.compute_probability(tech, policy)
            results.append({
                'tech_breakthrough': tech,
                'policy': policy,
                'high_impact_prob': prob
            })
        return results


class SD_BN_Model:
    """
    综合SD + BN模型
    """

    def __init__(self, sd_params=None, bn_params=None):
        self.sd_params = sd_params or SDParams()
        self.bn_params = bn_params or BNParams()
        self.sd_model = SDModel(self.sd_params)
        self.bn_model = BNModel(self.bn_params)

    def run_simulation(self, t_span=10):
        """
        运行完整模拟

        :param t_span: 时间跨度
        :return: 结果字典
        """
        # SD模拟
        sd_results = self.sd_model.simulate(t_span)

        # BN情景分析
        scenarios = [
            (True, 'Supportive'),
            (True, 'Neutral'),
            (True, 'Restrictive'),
            (False, 'Supportive'),
            (False, 'Neutral'),
            (False, 'Restrictive')
        ]
        bn_results = self.bn_model.predict_impact(scenarios)

        return {
            'sd_results': sd_results,
            'bn_results': bn_results,
            'occupation': self.sd_params.occupation_name
        }


# ============================================================
# 第二部分：模型 II - 教育决策优化模型 (AHP + MOEA/D)
# ============================================================

class AHPParams:
    """
    AHP参数配置类
    """

    def __init__(self):
        # 判断矩阵 (针对四个准则: 就业竞争力, 技艺独特性, 教学成本, 伦理合规性)
        # 准则: C1=就业竞争力, C2=技艺独特性, C3=教学成本, C4=伦理合规性
        self.judgment_matrix = np.array([
            [1, 3, 1/2, 2],    # C1 vs others
            [1/3, 1, 1/4, 1/2], # C2 vs others
            [2, 4, 1, 3],      # C3 vs others
            [1/2, 2, 1/3, 1]   # C4 vs others
        ])

        self.criteria = ['Employment Competitiveness', 'Artistic Uniqueness', 'Teaching Cost', 'Ethical Compliance']


class AHPModel:
    """
    AHP模型类
    """

    def __init__(self, params: AHPParams):
        self.params = params

    def calculate_weights(self):
        """
        计算准则权重

        :return: 权重向量和一致性比率
        """
        A = self.params.judgment_matrix
        n = A.shape[0]

        # 计算特征值和特征向量
        eigenvals, eigenvecs = np.linalg.eig(A)
        max_eigenval = np.max(eigenvals.real)
        weights = eigenvecs[:, np.argmax(eigenvals.real)].real
        weights = weights / np.sum(weights)

        # 计算一致性指标
        CI = (max_eigenval - n) / (n - 1)
        RI = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45][n-1]  # RI表
        CR = CI / RI if RI > 0 else 0

        return weights, CR

    def sensitivity_analysis(self, perturbation=0.1):
        """
        AHP权重灵敏度分析

        :param perturbation: 权重波动幅度 (±10%)
        :return: 灵敏度分析结果
        """
        original_weights, original_CR = self.calculate_weights()
        results = []

        for i in range(len(original_weights)):
            # 增加权重
            perturbed_matrix = self.params.judgment_matrix.copy()
            perturbed_matrix[i, :] *= (1 + perturbation)
            perturbed_matrix[:, i] /= (1 + perturbation)

            # 重新计算权重
            A = perturbed_matrix
            n = A.shape[0]
            eigenvals, eigenvecs = np.linalg.eig(A)
            max_eigenval = np.max(eigenvals.real)
            new_weights = eigenvecs[:, np.argmax(eigenvals.real)].real
            new_weights = new_weights / np.sum(new_weights)
            CI = (max_eigenval - n) / (n - 1)
            RI = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45][n-1]
            new_CR = CI / RI if RI > 0 else 0

            weight_change = np.abs(new_weights - original_weights)
            max_change = np.max(weight_change)

            results.append({
                'criterion': self.params.criteria[i],
                'original_weight': original_weights[i],
                'new_weight': new_weights.tolist(),
                'max_weight_change': max_change,
                'new_CR': new_CR
            })

        return results


class MOEADParams:
    """
    MOEA/D参数配置类
    """

    def __init__(self):
        self.population_size = 50
        self.max_generations = 10  # 减少代数以便展示
        self.neighborhood_size = 10
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1

        # 目标函数权重
        self.weights = np.random.rand(self.population_size, 3)  # 三个目标
        self.weights = self.weights / np.sum(self.weights, axis=1, keepdims=True)


class MOEADModel:
    """
    MOEA/D多目标优化模型
    """

    def __init__(self, params: MOEADParams):
        self.params = params

    def objective_functions(self, x):
        """
        目标函数

        :param x: 决策变量 [basic_enrollment, ai_enrollment, course_reform_rate]
        :return: 目标值 [f1, f2, f3]
        """
        basic_enrollment, ai_enrollment, course_reform_rate = x

        # f1: 就业率目标 (最大化)
        f1 = - (0.5 * basic_enrollment + 0.7 * ai_enrollment + 0.3 * course_reform_rate)

        # f2: 转型成本目标 (最小化)
        f2 = 0.4 * basic_enrollment + 0.6 * ai_enrollment + 0.5 * course_reform_rate

        # f3: 碳足迹/环境影响目标 (最小化，AI可降低食材浪费)
        f3 = 0.3 * basic_enrollment - 0.2 * ai_enrollment - 0.1 * course_reform_rate

        return [f1, f2, f3]

    def optimize(self):
        """
        执行MOEA/D优化

        :return: 帕累托前沿和演化历史
        """
        # 初始化种群
        population = []
        for i in range(self.params.population_size):
            x = np.random.rand(3)  # [basic_enrollment, ai_enrollment, course_reform_rate]
            f = self.objective_functions(x)
            population.append({'x': x, 'f': f})

        evolution_history = [population.copy()]

        # 进化过程
        for gen in range(self.params.max_generations):
            new_population = []
            for i, ind in enumerate(population):
                # 选择邻域
                neighbors = self._get_neighbors(i, population)

                # 交叉
                if np.random.rand() < self.params.crossover_rate:
                    parent1 = ind
                    parent2 = np.random.choice(neighbors)
                    offspring_x = self._crossover(parent1['x'], parent2['x'])
                else:
                    offspring_x = ind['x'].copy()

                # 变异
                offspring_x = self._mutate(offspring_x)

                # 评估
                offspring_f = self.objective_functions(offspring_x)
                offspring = {'x': offspring_x, 'f': offspring_f}

                # 更新邻域
                for neighbor in neighbors:
                    if self._dominates(offspring, neighbor):
                        neighbor.update(offspring)

                new_population.append(offspring)

            population = new_population
            evolution_history.append(population.copy())

        # 提取帕累托前沿
        pareto_front = []
        for ind in population:
            dominated = False
            for other in population:
                if self._dominates(other, ind):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(ind)

        return pareto_front, evolution_history

    def _get_neighbors(self, index, population):
        """获取邻域个体"""
        start = max(0, index - self.params.neighborhood_size // 2)
        end = min(len(population), index + self.params.neighborhood_size // 2 + 1)
        neighbors = population[start:end]
        if len(neighbors) < self.params.neighborhood_size:
            neighbors.extend(population[:self.params.neighborhood_size - len(neighbors)])
        return neighbors

    def _crossover(self, x1, x2):
        """简单交叉"""
        alpha = np.random.rand()
        return alpha * x1 + (1 - alpha) * x2

    def _mutate(self, x):
        """简单变异"""
        for i in range(len(x)):
            if np.random.rand() < self.params.mutation_rate:
                x[i] += np.random.normal(0, 0.1)
                x[i] = np.clip(x[i], 0, 1)  # 保持在[0,1]范围内
        return x

    def _dominates(self, ind1, ind2):
        """检查ind1是否支配ind2"""
        better_or_equal = all(f1 <= f2 for f1, f2 in zip(ind1['f'], ind2['f']))
        strictly_better = any(f1 < f2 for f1, f2 in zip(ind1['f'], ind2['f']))
        return better_or_equal and strictly_better


class AHP_MOEAD_Model:
    """
    综合AHP + MOEA/D模型
    """

    def __init__(self, ahp_params=None, moead_params=None):
        self.ahp_params = ahp_params or AHPParams()
        self.moead_params = moead_params or MOEADParams()
        self.ahp_model = AHPModel(self.ahp_params)
        self.moead_model = MOEADModel(self.moead_params)

    def run_optimization(self):
        """
        运行完整优化

        :return: 结果字典
        """
        # AHP权重计算
        weights, CR = self.ahp_model.calculate_weights()

        # AHP灵敏度分析
        sensitivity_results = self.ahp_model.sensitivity_analysis()

        # MOEA/D优化
        pareto_front, evolution_history = self.moead_model.optimize()

        # 膝盖点分析
        knee_point = self._find_knee_point(pareto_front)

        return {
            'ahp_weights': weights,
            'ahp_CR': CR,
            'sensitivity_results': sensitivity_results,
            'pareto_front': pareto_front,
            'evolution_history': evolution_history,
            'knee_point': knee_point,
            'criteria': self.ahp_params.criteria
        }

    def _find_knee_point(self, pareto_front):
        """
        寻找帕累托前沿的膝盖点

        :param pareto_front: 帕累托前沿
        :return: 膝盖点索引和具体建议
        """
        if len(pareto_front) < 3:
            return None

        # 计算每个点的"膝盖度" (与理想点的距离 + 与邻点的角度)
        points = np.array([ind['f'] for ind in pareto_front])
        ideal_point = np.min(points, axis=0)

        max_distances = np.max(points - ideal_point, axis=0)
        normalized_points = (points - ideal_point) / max_distances

        knee_scores = []
        for i, point in enumerate(normalized_points):
            # 计算到理想点的距离
            distance = np.linalg.norm(point)

            # 计算与前后点的角度变化
            if i == 0:
                prev_vector = point - normalized_points[i+1]
            elif i == len(normalized_points) - 1:
                prev_vector = normalized_points[i-1] - point
            else:
                prev_vector = normalized_points[i-1] - point

            if i == 0:
                next_vector = normalized_points[i+1] - point
            elif i == len(normalized_points) - 1:
                next_vector = point - normalized_points[i-1]
            else:
                next_vector = point - normalized_points[i+1]

            # 计算角度
            cos_angle = np.dot(prev_vector, next_vector) / (np.linalg.norm(prev_vector) * np.linalg.norm(next_vector))
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            # 膝盖度 = 距离 + 角度权重
            knee_score = distance + 0.5 * angle
            knee_scores.append(knee_score)

        knee_index = np.argmin(knee_scores)
        knee_solution = pareto_front[knee_index]

        # 生成政策建议
        basic_enrollment, ai_enrollment, course_reform_rate = knee_solution['x']
        policy_recommendation = f"""
        建议某厨师学校投入:
        - 传统技艺教学预算: {basic_enrollment*100:.0f}%
        - AI相关教学预算: {ai_enrollment*100:.0f}%
        - 课程改革投入: {course_reform_rate*100:.0f}%

        这将实现就业率提升 {(1-knee_solution['f'][0])*100:.1f}%,
        转型成本控制在 {knee_solution['f'][1]:.2f},
        环境影响最小化至 {knee_solution['f'][2]:.2f}
        """

        return {
            'index': knee_index,
            'solution': knee_solution,
            'policy_recommendation': policy_recommendation.strip()
        }


# ============================================================
# 第三部分：模型 III - 综合成功评价模型（模糊评价 + 相关性分析）
# ============================================================

class FCEParams:
    """
    模糊综合评价参数配置类
    """

    def __init__(self):
        # 评价集 U = {优秀, 良好, 中等, 较差}
        self.evaluation_set = ['优秀', '良好', '中等', '较差']

        # 权重向量 (针对准则: 伦理素养, 审美能力, 技术技能, 就业前景, 人类自主性, 文化传承)
        self.weights = np.array([0.2, 0.15, 0.2, 0.15, 0.15, 0.15])

        # 隶属度矩阵示例 (针对不同政策)
        self.membership_matrix = {
            '禁用AI': np.array([
                [0.8, 0.15, 0.05, 0.0],   # 伦理素养
                [0.7, 0.2, 0.1, 0.0],     # 审美能力
                [0.1, 0.3, 0.4, 0.2],     # 技术技能
                [0.2, 0.4, 0.3, 0.1],     # 就业前景
                [0.9, 0.08, 0.02, 0.0],   # 人类自主性
                [0.8, 0.15, 0.05, 0.0]    # 文化传承
            ]),
            '全员拥抱AI': np.array([
                [0.1, 0.2, 0.4, 0.3],    # 伦理素养
                [0.2, 0.3, 0.3, 0.2],    # 审美能力
                [0.9, 0.08, 0.02, 0.0],  # 技术技能
                [0.8, 0.15, 0.05, 0.0],  # 就业前景
                [0.3, 0.3, 0.3, 0.1],    # 人类自主性
                [0.2, 0.3, 0.3, 0.2]     # 文化传承
            ])
        }


class FCEModel:
    """
    模糊综合评价模型
    """

    def __init__(self, params: FCEParams):
        self.params = params

    def evaluate_policy(self, policy_name):
        """
        评价特定政策

        :param policy_name: 政策名称
        :return: 综合评价向量
        """
        if policy_name not in self.params.membership_matrix:
            raise ValueError(f"Policy {policy_name} not found")

        R = self.params.membership_matrix[policy_name]
        W = self.params.weights

        # 综合评价: B = W * R
        B = np.dot(W, R)

        return B


class CorrelationParams:
    """
    相关性分析参数配置类
    """

    def __init__(self):
        # 模拟数据: AI融入度 vs 核心竞争力
        np.random.seed(42)
        self.ai_integration = np.random.rand(100) * 100  # 0-100%
        self.core_competence = 50 + 0.5 * self.ai_integration + np.random.normal(0, 10, 100)


class CorrelationModel:
    """
    相关性分析模型
    """

    def __init__(self, params: CorrelationParams):
        self.params = params

    def analyze_correlation(self):
        """
        分析AI融入度与核心竞争力的相关性

        :return: 相关系数和p值
        """
        from scipy.stats import pearsonr

        corr, p_value = pearsonr(self.params.ai_integration, self.params.core_competence)

        return {
            'correlation_coefficient': corr,
            'p_value': p_value,
            'ai_integration': self.params.ai_integration,
            'core_competence': self.params.core_competence
        }


class FCE_Correlation_Model:
    """
    综合模糊评价 + 相关性分析模型
    """

    def __init__(self, fce_params=None, corr_params=None, ahp_weights=None):
        self.fce_params = fce_params or FCEParams()
        self.corr_params = corr_params or CorrelationParams()
        self.fce_model = FCEModel(self.fce_params)
        self.corr_model = CorrelationModel(self.corr_params)
        # 从AHP获取权重，形成闭环 (注释掉，因为FCE现在有6个准则，而AHP有4个)
        # if ahp_weights is not None:
        #     self.fce_params.weights = ahp_weights[:4]  # 取前四个权重对应FCE的四个准则

    def run_evaluation(self):
        """
        运行完整评价

        :return: 结果字典
        """
        # 模糊评价
        policies = list(self.fce_params.membership_matrix.keys())
        fce_results = {}
        for policy in policies:
            fce_results[policy] = self.fce_model.evaluate_policy(policy)

        # 相关性分析
        corr_results = self.corr_model.analyze_correlation()

        return {
            'fce_results': fce_results,
            'corr_results': corr_results,
            'evaluation_set': self.fce_params.evaluation_set,
            'weights': self.fce_params.weights
        }


# ============================================================
# 第四部分：模型 IV - 泛化与推广模型 (CBR + GWR)
# ============================================================

class CBRParams:
    """
    案例推理参数配置类
    """

    def __init__(self):
        # 案例库
        self.case_base = [
            {
                'id': 1,
                'name': 'CIA Culinary Institute',
                'digital_level': 0.7,
                'budget_per_student': 50000,
                'ai_integration': 0.3,
                'outcome': 'success'
            },
            {
                'id': 2,
                'name': 'Cloud Kitchen Startup',
                'digital_level': 0.9,
                'budget_per_student': 20000,
                'ai_integration': 0.8,
                'outcome': 'success'
            },
            {
                'id': 3,
                'name': 'Michelin Traditional Restaurant',
                'digital_level': 0.4,
                'budget_per_student': 10000,
                'ai_integration': 0.1,
                'outcome': 'moderate'
            },
            {
                'id': 4,
                'name': 'Silicon Valley Tech-Focused Culinary School',
                'digital_level': 0.95,
                'budget_per_student': 60000,
                'ai_integration': 0.9,
                'outcome': 'success'
            }
        ]


class CBRModel:
    """
    案例推理模型
    """

    def __init__(self, params: CBRParams):
        self.params = params

    def calculate_similarity(self, query_case, base_case):
        """
        计算相似度

        :param query_case: 查询案例
        :param base_case: 基准案例
        :return: 相似度分数
        """
        # 简单欧几里得距离
        features = ['digital_level', 'budget_per_student', 'ai_integration']
        distance = 0
        for feature in features:
            distance += (query_case[feature] - base_case[feature]) ** 2
        similarity = 1 / (1 + np.sqrt(distance))
        return similarity

    def retrieve_similar_cases(self, query_case, top_k=2):
        """
        检索相似案例

        :param query_case: 查询案例
        :param top_k: 返回最相似案例数量
        :return: 相似案例列表
        """
        similarities = []
        for case in self.params.case_base:
            sim = self.calculate_similarity(query_case, case)
            similarities.append((case, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class GWRParams:
    """
    地理加权回归参数配置类
    """

    def __init__(self):
        # 真实城市经纬度数据 (示例)
        self.cities = {
            'New York': {'lat': 40.7128, 'lon': -74.0060},
            'Paris': {'lat': 48.8566, 'lon': 2.3522},
            'Tokyo': {'lat': 35.6762, 'lon': 139.6503},
            'London': {'lat': 51.5074, 'lon': -0.1278},
            'Beijing': {'lat': 39.9042, 'lon': 116.4074},
            'San Francisco': {'lat': 37.7749, 'lon': -122.4194},
            'Berlin': {'lat': 52.5200, 'lon': 13.4050},
            'Sydney': {'lat': -33.8688, 'lon': 151.2093}
        }

        # 转换为坐标数组
        self.coordinates = np.array([[city['lat'], city['lon']] for city in self.cities.values()])
        self.city_names = list(self.cities.keys())

        # 模拟AI融入度和本地参数
        np.random.seed(42)
        self.ai_integration = np.random.rand(len(self.cities)) * 100
        self.local_parameters = np.random.rand(len(self.cities), 3)


class GWRModel:
    """
    地理加权回归模型
    """

    def __init__(self, params: GWRParams):
        self.params = params

    def local_regression(self, target_point, bandwidth=10):
        """
        局部回归

        :param target_point: 目标点坐标
        :param bandwidth: 带宽
        :return: 局部参数
        """
        # 计算权重 (高斯核)
        distances = np.linalg.norm(self.params.coordinates - target_point, axis=1)
        weights = np.exp(-distances ** 2 / (2 * bandwidth ** 2))
        weights = weights / np.sum(weights)

        # 加权回归 (简化)
        X = np.column_stack([np.ones(len(weights)), self.params.ai_integration])
        y = self.params.local_parameters[:, 0]  # 示例目标变量

        # 加权最小二乘
        W = np.diag(weights)
        beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y

        return beta

    def predict_local_parameters(self, new_points):
        """
        预测新点的局部参数

        :param new_points: 新点坐标
        :return: 预测参数
        """
        predictions = []
        for point in new_points:
            beta = self.local_regression(point)
            predictions.append(beta)
        return np.array(predictions)


class CBR_GWR_Model:
    """
    综合CBR + GWR模型
    """

    def __init__(self, cbr_params=None, gwr_params=None):
        self.cbr_params = cbr_params or CBRParams()
        self.gwr_params = gwr_params or GWRParams()
        self.cbr_model = CBRModel(self.cbr_params)
        self.gwr_model = GWRModel(self.gwr_params)

    def generalize_solution(self, query_case, new_locations):
        """
        泛化解决方案

        :param query_case: 查询案例
        :param new_locations: 新位置坐标
        :return: 泛化结果
        """
        # CBR检索
        similar_cases = self.cbr_model.retrieve_similar_cases(query_case)

        # GWR预测
        local_params = self.gwr_model.predict_local_parameters(new_locations)

        return {
            'similar_cases': similar_cases,
            'local_parameters': local_params,
            'query_case': query_case,
            'new_locations': new_locations
        }


# ============================================================
# 可视化模块
# ============================================================

class ComprehensiveVisualization:
    """
    综合模型可视化类
    """

    def __init__(self, save_dir='./figures'):
        self.saver = FigureSaver(save_dir)

    def plot_sd_results(self, results):
        """
        绘制SD模型结果
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))

        sd_res = results['sd_results']
        axes[0,0].plot(sd_res['time'], sd_res['labor_demand'], 'b-', linewidth=2)
        axes[0,0].set_title('Labor Demand Evolution')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Labor Demand')
        axes[0,0].grid(True)

        axes[0,1].plot(sd_res['time'], sd_res['tech_maturity'], 'r-', linewidth=2)
        axes[0,1].set_title('Technology Maturity Evolution')
        axes[0,1].set_xlabel('Time')
        axes[0,1].set_ylabel('Technology Maturity')
        axes[0,1].grid(True)

        axes[1,0].plot(sd_res['time'], sd_res['skill_matching'], 'g-', linewidth=2)
        axes[1,0].set_title('Skill Matching Degree')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Skill Matching')
        axes[1,0].grid(True)

        axes[1,1].plot(sd_res['time'], sd_res['human_core_competence'], 'm-', linewidth=2)
        axes[1,1].set_title('Human Core Competence Evolution')
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('Human Core Competence')
        axes[1,1].grid(True)

        plt.suptitle(f'System Dynamics Model - {results["occupation"]}')
        paths = self.saver.save(fig, 'sd_model_results')
        print(f"SD model visualization saved: {paths[0]}")

    def plot_bn_results(self, results):
        """
        绘制BN模型结果
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        bn_res = results['bn_results']
        scenarios = [f"{r['tech_breakthrough']}_{r['policy']}" for r in bn_res]
        probs = [r['high_impact_prob'] for r in bn_res]

        ax.bar(scenarios, probs, color='skyblue')
        ax.set_title('Bayesian Network - Impact Probability by Scenario')
        ax.set_ylabel('High Impact Probability')
        ax.tick_params(axis='x', rotation=45)

        paths = self.saver.save(fig, 'bn_model_results')
        print(f"BN model visualization saved: {paths[0]}")

    def plot_ahp_weights(self, results):
        """
        绘制AHP权重
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        weights = results['ahp_weights']
        criteria = results['criteria']

        ax.bar(criteria, weights, color='lightgreen')
        ax.set_title(f'AHP Weights (CR = {results["ahp_CR"]:.3f})')
        ax.set_ylabel('Weight')
        ax.tick_params(axis='x', rotation=45)

        paths = self.saver.save(fig, 'ahp_weights')
        print(f"AHP weights visualization saved: {paths[0]}")

    def plot_pareto_front(self, results):
        """
        绘制帕累托前沿
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        front = results['pareto_front']
        f1_vals = [ind['f'][0] for ind in front]
        f2_vals = [ind['f'][1] for ind in front]
        f3_vals = [ind['f'][2] for ind in front]

        ax.scatter(f1_vals, f2_vals, f3_vals, c='red', s=50, alpha=0.7)
        ax.set_title('MOEA/D Pareto Front (3D)')
        ax.set_xlabel('Objective 1 (Employment Rate)')
        ax.set_ylabel('Objective 2 (Transition Cost)')
        ax.set_zlabel('Objective 3 (Carbon Footprint)')

        paths = self.saver.save(fig, 'pareto_front')
        print(f"Pareto front visualization saved: {paths[0]}")

    def plot_evolution_process(self, results):
        """
        绘制演化过程
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        history = results['evolution_history']

        for gen in [0, len(history)//4, len(history)//2, 3*len(history)//4, -1]:
            pop = history[gen]
            f1 = [ind['f'][0] for ind in pop]
            f2 = [ind['f'][1] for ind in pop]
            f3 = [ind['f'][2] for ind in pop]

            row = gen // (len(history)//2)
            col = gen % 3

            axes[row, col].scatter(f1, f2, alpha=0.6)
            axes[row, col].set_title(f'Generation {gen}')
            axes[row, col].set_xlabel('Obj1')
            axes[row, col].set_ylabel('Obj2')
            axes[row, col].grid(True)

        plt.suptitle('MOEA/D Evolution Process')
        paths = self.saver.save(fig, 'evolution_process')
        print(f"Evolution process visualization saved: {paths[0]}")

    def plot_fce_results(self, results):
        """
        绘制模糊评价结果
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        fce_res = results['fce_results']
        eval_set = results['evaluation_set']

        policies = list(fce_res.keys())
        for i, policy in enumerate(policies):
            ax.bar(np.arange(len(eval_set)) + i*0.2, fce_res[policy],
                   width=0.2, label=policy, alpha=0.7)

        ax.set_xticks(np.arange(len(eval_set)) + 0.2)
        ax.set_xticklabels(eval_set)
        ax.set_title('Fuzzy Comprehensive Evaluation')
        ax.set_ylabel('Membership Degree')
        ax.legend()

        paths = self.saver.save(fig, 'fce_results')
        print(f"FCE results visualization saved: {paths[0]}")

    def plot_correlation_results(self, results):
        """
        绘制相关性分析结果
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        corr_res = results['corr_results']
        ax.scatter(corr_res['ai_integration'], corr_res['core_competence'], alpha=0.6)
        ax.set_title(f'Correlation Analysis (r = {corr_res["correlation_coefficient"]:.3f})')
        ax.set_xlabel('AI Integration Degree (%)')
        ax.set_ylabel('Core Competence')
        ax.grid(True)

        paths = self.saver.save(fig, 'correlation_analysis')
        print(f"Correlation analysis visualization saved: {paths[0]}")

    def plot_gwr_spatial_sensitivity(self, results):
        """
        绘制GWR空间敏感度分布
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        locations = results['new_locations']
        sensitivities = results['local_parameters'][:, 1]  # 假设第二列是AI敏感度

        sc = ax.scatter(locations[:, 1], locations[:, 0], c=sensitivities,
                        cmap='viridis', s=100, edgecolor='black')
        plt.colorbar(sc, label='AI Sensitivity Coefficient')
        ax.set_title('Spatial Distribution of AI Integration Impact')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True)

        paths = self.saver.save(fig, 'gwr_spatial_sensitivity')
        print(f"GWR spatial sensitivity visualization saved: {paths[0]}")


def sensitivity_analysis(sd_model, param_name, param_range, t_span=10):
    """
    灵敏度分析

    :param sd_model: SD模型实例
    :param param_name: 参数名称
    :param param_range: 参数范围
    :param t_span: 时间跨度
    :return: 分析结果
    """
    results = []
    original_value = getattr(sd_model.params, param_name)

    for value in param_range:
        setattr(sd_model.params, param_name, value)
        sim_result = sd_model.simulate(t_span)
        results.append({
            'param_value': value,
            'final_labor_demand': sim_result['labor_demand'][-1],
            'final_tech_maturity': sim_result['tech_maturity'][-1],
            'final_skill_matching': sim_result['skill_matching'][-1]
        })

    # 恢复原始值
    setattr(sd_model.params, param_name, original_value)

    return results

def run_comprehensive_models():
    """
    运行所有四个模型的完整工作流
    """
    print("\n" + "█"*70)
    print("█" + " "*15 + "AI职业演化综合模型" + " "*16 + "█")
    print("█" + " "*10 + "Comprehensive AI Career Evolution Models" + " "*11 + "█")
    print("█"*70 + "\n")

    # 创建figures目录
    figures_dir = './figures'
    os.makedirs(figures_dir, exist_ok=True)

    viz = ComprehensiveVisualization(save_dir=figures_dir)

    # ========== 模型 I: SD + BN ==========
    print("【Model I】职业演化动态模型 (SD + BN)...")
    sd_bn_model = SD_BN_Model()
    sd_bn_results = sd_bn_model.run_simulation()

    print("  📊 绘制SD模型结果...")
    viz.plot_sd_results(sd_bn_results)

    print("  📊 绘制BN模型结果...")
    viz.plot_bn_results(sd_bn_results)

    # ========== 模型 II: AHP + MOEA/D ==========
    print("\n【Model II】教育决策优化模型 (AHP + MOEA/D)...")
    ahp_moead_model = AHP_MOEAD_Model()
    ahp_moead_results = ahp_moead_model.run_optimization()

    print("  📊 绘制AHP权重...")
    viz.plot_ahp_weights(ahp_moead_results)

    print("  📊 绘制帕累托前沿...")
    viz.plot_pareto_front(ahp_moead_results)

    print("  📊 绘制演化过程...")
    viz.plot_evolution_process(ahp_moead_results)

    # ========== 模型 III: 模糊评价 + 相关性分析 ==========
    print("\n【Model III】综合成功评价模型 (FCE + Correlation)...")
    fce_corr_model = FCE_Correlation_Model(ahp_weights=ahp_moead_results['ahp_weights'])
    fce_corr_results = fce_corr_model.run_evaluation()

    print("  📊 绘制模糊评价结果...")
    viz.plot_fce_results(fce_corr_results)

    print("  📊 绘制相关性分析...")
    viz.plot_correlation_results(fce_corr_results)

    # ========== 模型 IV: CBR + GWR ==========
    print("\n【Model IV】泛化与推广模型 (CBR + GWR)...")
    cbr_gwr_model = CBR_GWR_Model()

    # 示例查询
    query_case = {
        'digital_level': 0.6,
        'budget_per_student': 40000,
        'ai_integration': 0.4
    }
    new_locations = np.array([[50, 50], [70, 30]])

    cbr_gwr_results = cbr_gwr_model.generalize_solution(query_case, new_locations)

    print(f"  📊 找到 {len(cbr_gwr_results['similar_cases'])} 个相似案例")
    print(f"  📊 为 {len(new_locations)} 个新位置预测了局部参数")

    print("  📊 绘制GWR空间敏感度...")
    viz.plot_gwr_spatial_sensitivity(cbr_gwr_results)

    # 灵敏度分析
    print("\n【Sensitivity Analysis】参数灵敏度分析...")
    sd_model = SDModel(SDParams('chef'))
    beta_range = np.linspace(0.05, 0.25, 10)
    sensitivity_results = sensitivity_analysis(sd_model, 'beta', beta_range)

    # 绘制灵敏度分析结果
    fig, ax = plt.subplots(figsize=(10, 6))
    param_values = [r['param_value'] for r in sensitivity_results]
    labor_demands = [r['final_labor_demand'] for r in sensitivity_results]
    ax.plot(param_values, labor_demands, 'bo-', linewidth=2)
    ax.set_title('Sensitivity Analysis: Beta Parameter vs Final Labor Demand')
    ax.set_xlabel('Beta (Substitution Rate)')
    ax.set_ylabel('Final Labor Demand')
    ax.grid(True)
    paths = viz.saver.save(fig, 'sensitivity_analysis')
    print(f"Sensitivity analysis visualization saved: {paths[0]}")

    print("\n" + "█"*70)
    print("█" + " "*23 + "综合模型执行完成!" + " "*24 + "█")
    print("█"*70 + "\n")

    return {
        'model_I': sd_bn_results,
        'model_II': ahp_moead_results,
        'model_III': fce_corr_results,
        'model_IV': cbr_gwr_results,
        'sensitivity': sensitivity_results
    }


if __name__ == "__main__":
    # 运行所有模型
    all_results = run_comprehensive_models()

    # 示例：访问特定模型结果
    print("\n示例结果访问:")
    print(f"SD模型最终劳动力需求: {all_results['model_I']['sd_results']['labor_demand'][-1]:.2f}")
    print(f"人类核心素养最终值: {all_results['model_I']['sd_results']['human_core_competence'][-1]:.2f}")
    print(f"AHP一致性比率: {all_results['model_II']['ahp_CR']:.3f}")
    print(f"权重灵敏度最大变化: {max(r['max_weight_change'] for r in all_results['model_II']['sensitivity_results']):.3f}")
    print(f"膝盖点政策建议: {all_results['model_II']['knee_point']['policy_recommendation'][:100]}...")
    print(f"相关系数: {all_results['model_III']['corr_results']['correlation_coefficient']:.3f}")
