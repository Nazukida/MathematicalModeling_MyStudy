"""
============================================================
Juneau旅游可持续性模型 - 完整工作流
(Juneau Tourism Sustainability Model - Complete Workflow)
============================================================
基于论文: "Economy, Ecology, and Social Welfare: A Win-Win Approach for Sustainable Tourism in Juneau"
(Team #2501687)

功能：多目标优化模型（经济、环境、社会福利）
最大化总社会效用 U = P + E + S
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================

模型框架：
1. 游客需求函数 N(t) = 调整后的季节性需求
2. 经济利润 P = 游客利润 + 政策收入/成本
3. 环境水平 E = -环境成本 + 生态恢复力 + 投资回报
4. 社会福利 S = 就业收益 - 社会影响 + 投资回报
5. 目标函数 U = P + E + S (通过CVM转换为单目标)
"""

import sys
import os

# 添加模型库路径
MODEL_CODE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelCode')
sys.path.insert(0, MODEL_CODE_PATH)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize, differential_evolution
import warnings
from itertools import product

warnings.filterwarnings('ignore')

# ============================================================
# 图表配置（内联版本，避免导入问题）
# ============================================================

class PlotStyleConfig:
    """图表美化配置类"""

    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'success': '#27AE60',
        'danger': '#C73E1D',
        'neutral': '#3B3B3B',
        'background': '#FAFAFA',
        'grid': '#E0E0E0'
    }

    PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4C9A', '#1B998B', '#E94F37', '#44AF69']

    @staticmethod
    def setup_style(style='academic'):
        """设置学术风格图表"""
        if style == 'academic':
            plt.style.use('default')

            # 设置中文字体支持 - Windows系统
            import platform
            if platform.system() == 'Windows':
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            else:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']

            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            plt.rcParams['font.family'] = 'sans-serif'

            rcParams['font.size'] = 10
            rcParams['axes.labelsize'] = 11
            rcParams['axes.titlesize'] = 12
            rcParams['xtick.labelsize'] = 9
            rcParams['ytick.labelsize'] = 9
            rcParams['legend.fontsize'] = 9
            rcParams['figure.titlesize'] = 14

            # 网格和边框
            rcParams['axes.grid'] = True
            rcParams['grid.alpha'] = 0.3
            rcParams['grid.color'] = PlotStyleConfig.COLORS['grid']
            rcParams['axes.edgecolor'] = PlotStyleConfig.COLORS['neutral']
            rcParams['axes.facecolor'] = PlotStyleConfig.COLORS['background']

    @staticmethod
    def get_palette(n=None):
        """获取调色板"""
        if n is None:
            return PlotStyleConfig.PALETTE
        return PlotStyleConfig.PALETTE[:n]

class FigureSaver:
    """图表保存工具类"""

    def __init__(self, save_dir='./figures', format='png'):
        self.save_dir = save_dir
        self.format = format
        os.makedirs(save_dir, exist_ok=True)

    def save(self, fig, filename, formats=None, tight=True):
        """保存图表"""
        if formats is None:
            formats = [self.format]

        if tight:
            fig.tight_layout()

        paths = []
        for fmt in formats:
            path = os.path.join(self.save_dir, f"{filename}.{fmt}")
            fig.savefig(path, dpi=300, bbox_inches='tight',
                       facecolor=fig.get_facecolor(), edgecolor='none')
            paths.append(path)
            print(f"  📊 图表已保存: {path}")
        return paths

warnings.filterwarnings('ignore')

# 设置绘图风格
PlotStyleConfig.setup_style('academic')

# ============================================================
# 第一部分：模型参数配置 (Model Parameters Configuration)
# ============================================================

class ParameterRange:
    """
    参数范围类 - 支持固定值或区间

    使用方法：
        # 固定值
        param = ParameterRange(120)

        # 范围（均匀分布）
        param = ParameterRange(100, 140)

        # 范围（正态分布）
        param = ParameterRange(120, std=10, distribution='normal')
    """

    def __init__(self, value, upper=None, std=None, distribution='uniform'):
        """
        :param value: 固定值，或范围下界
        :param upper: 范围上界（若为None则为固定值）
        :param std: 标准差（用于正态分布）
        :param distribution: 'uniform' 或 'normal'
        """
        self.is_range = (upper is not None) or (std is not None)

        if upper is not None:
            self.lower = value
            self.upper = upper
            self.mean = (value + upper) / 2
            self.std = (upper - value) / 4  # 95%置信区间
        elif std is not None:
            self.mean = value
            self.std = std
            self.lower = value - 2 * std
            self.upper = value + 2 * std
        else:
            self.value = value
            self.lower = value
            self.upper = value
            self.mean = value
            self.std = 0

        self.distribution = distribution

    def get_value(self):
        """获取固定值或均值"""
        if self.is_range:
            return self.mean
        return self.value

    def sample(self, n=1):
        """采样n个值"""
        if not self.is_range:
            return np.full(n, self.value)

        if self.distribution == 'uniform':
            return np.random.uniform(self.lower, self.upper, n)
        else:  # normal
            samples = np.random.normal(self.mean, self.std, n)
            return np.clip(samples, self.lower, self.upper)

    def __repr__(self):
        if self.is_range:
            return f"[{self.lower:.4g}, {self.upper:.4g}]"
        return f"{self.value}"


class JuneauModelParams:
    """
    Juneau旅游模型参数配置类

    ★★★ 需要调整的参数在这里修改 ★★★

    支持两种方式定义参数：
    1. 固定值: self.A = 16822
    2. 范围:   self.A = ParameterRange(16000, 18000)  # 均匀分布
    """

    def __init__(self):
        # ============ 游客需求模型参数 ============
        self.A = 16822          # 游客流量振幅
        self.B = 5514           # 游客流量基线

        # ============ 经济模型参数 ============
        self.p = 200            # 每位游客利润（$）
        self.base_revenue = self.B * self.p  # 基线收入

        # ============ 环境模型参数 ============
        self.e = 66.13          # 每人排放量（kg CO2）
        self.SCC = 190          # 碳社会成本（$/吨）
        self.carbon_cost_per_tourist = self.e / 1000 * self.SCC  # 12.56 $/人
        self.ERI_max = 2e5      # 最大生态恢复力
        self.beta = 1e-4        # 恢复力衰减系数
        self.alpha1 = 1e-4      # 环境投资效率系数
        self.Gamma1m = 1e8      # 环境投资最大回报
        self.Gamma10 = 1e4      # 环境投资基线回报
        self.I10 = 0            # 环境投资初始偏移

        # ============ 社会福利模型参数 ============
        self.pop = 32000        # 当地人口
        self.eta = 0.1          # 就业影响系数
        self.Med = 50000        # 年均收入（$）
        self.pi_inf = 0.02      # 通胀率
        self.S1_yearly = 7774865  # 年就业收益常数
        self.beta2 = 1e-4       # 社会影响系数
        self.alpha2 = 1e-4      # 社会投资效率系数
        self.Gamma2m = 1e8      # 社会投资最大回报
        self.Gamma20 = 1e4      # 社会投资基线回报
        self.I20 = 0            # 社会投资初始偏移

        # ============ 决策变量范围 ============
        self.c1_range = (5000, 20000)    # 峰季游客上限（人/日）
        self.c2_range = (1000, 10000)    # 非峰季游客目标（人/日）
        self.I_range = (0, 500000)       # 投资范围（$）- 放宽限制
        self.gamma1_range = (0, 1)       # 环境投资比例
        self.x1_range = (-50000, 50000)  # 峰季税收调整参数
        self.x2_range = (-50000, 50000)  # 非峰季补贴调整参数

        # ============ 约束阈值 ============
        self.E_min = 0          # 环境最小可接受水平
        self.S_min = 0          # 社会福利最小可接受水平

        # ============ 网格搜索分辨率 ============
        self.c1_steps = 16       # c1网格数量
        self.c2_steps = 16       # c2网格数量
        self.I_steps = 11        # I网格数量
        self.gamma1_steps = 9    # gamma1网格数量
        self.x1_steps = 11       # x1网格数量
        self.x2_steps = 11       # x2网格数量

    def _get_param_value(self, param):
        """获取参数值（支持固定值或ParameterRange）"""
        if isinstance(param, ParameterRange):
            return param.get_value()
        return param

    def _get_param_display(self, param):
        """获取参数显示字符串"""
        if isinstance(param, ParameterRange):
            return str(param)
        return f"{param}"

    def has_uncertainty(self):
        """检查是否有不确定性参数"""
        params_to_check = ['A', 'B', 'p', 'e', 'SCC', 'ERI_max', 'beta',
                          'alpha1', 'Gamma1m', 'Gamma10', 'I10',
                          'pop', 'eta', 'Med', 'pi_inf', 'S1_yearly', 'beta2',
                          'alpha2', 'Gamma2m', 'Gamma20', 'I20']
        for name in params_to_check:
            if isinstance(getattr(self, name), ParameterRange):
                return True
        return False

    def get_uncertain_params(self):
        """获取所有不确定性参数的名称和范围"""
        params_to_check = ['A', 'B', 'p', 'e', 'SCC', 'ERI_max', 'beta',
                          'alpha1', 'Gamma1m', 'Gamma10', 'I10',
                          'pop', 'eta', 'Med', 'pi_inf', 'S1_yearly', 'beta2',
                          'alpha2', 'Gamma2m', 'Gamma20', 'I20']
        uncertain = {}
        for name in params_to_check:
            param = getattr(self, name)
            if isinstance(param, ParameterRange):
                uncertain[name] = param
        return uncertain

    def sample_params(self):
        """采样一组参数值，返回新的参数对象"""
        sampled = JuneauModelParams()
        params_to_sample = ['A', 'B', 'p', 'e', 'SCC', 'ERI_max', 'beta',
                           'alpha1', 'Gamma1m', 'Gamma10', 'I10',
                           'pop', 'eta', 'Med', 'pi_inf', 'S1_yearly', 'beta2',
                           'alpha2', 'Gamma2m', 'Gamma20', 'I20']
        for name in params_to_sample:
            param = getattr(self, name)
            if isinstance(param, ParameterRange):
                setattr(sampled, name, param.sample(1)[0])
        return sampled

    def summary(self):
        """打印参数摘要"""
        print("\n" + "="*70)
        print("📋 Juneau旅游模型参数配置 (Juneau Tourism Model Parameters)")
        print("="*70)

        if self.has_uncertainty():
            print("⚠️  检测到不确定性参数，将进行范围分析")

        print("\n【游客需求模型】 N0(t) = max(-A*cos(2πt/365) + B, 0)")
        print(f"  A = {self._get_param_display(self.A)} (流量振幅)")
        print(f"  B = {self._get_param_display(self.B)} (流量基线)")

        print("\n【经济模型】 P = Σ[N(t)*p + f(t) - I]")
        print(f"  p = ${self._get_param_display(self.p)}/人 (每人利润)")
        print(f"  基线收入 = ${self.base_revenue:,.0f}")

        print("\n【环境模型】 E = Σ[-E_cost(t) + ERI(t) + Γ1(I)]")
        print(f"  e = {self._get_param_display(self.e)} kg/人 (排放量)")
        print(f"  SCC = ${self._get_param_display(self.SCC)}/吨 (碳成本)")
        print(f"  碳成本/人 = ${self.carbon_cost_per_tourist:.2f}")
        print(f"  ERI_max = {self._get_param_display(self.ERI_max)}")
        print(f"  β = {self._get_param_display(self.beta)}")
        print(f"  α1 = {self._get_param_display(self.alpha1)}")
        print(f"  Γ1m = {self._get_param_display(self.Gamma1m)}")
        print(f"  Γ10 = {self._get_param_display(self.Gamma10)}")

        print("\n【社会福利模型】 S = Σ[S1 + S2 + Γ2(I)]")
        print(f"  pop = {self._get_param_display(self.pop)} (人口)")
        print(f"  η = {self._get_param_display(self.eta)}")
        print(f"  Med = ${self._get_param_display(self.Med)} (年收入)")
        print(f"  π_inf = {self._get_param_display(self.pi_inf)}")
        print(f"  S1_yearly = ${self._get_param_value(self.S1_yearly):,.0f}")
        print(f"  β2 = {self._get_param_display(self.beta2)}")
        print(f"  α2 = {self._get_param_display(self.alpha2)}")
        print(f"  Γ2m = {self._get_param_display(self.Gamma2m)}")
        print(f"  Γ20 = {self._get_param_display(self.Gamma20)}")

        print("\n【决策变量范围】")
        print(f"  c1 ∈ [{self.c1_range[0]:,}, {self.c1_range[1]:,}] 人/日")
        print(f"  c2 ∈ [{self.c2_range[0]:,}, {self.c2_range[1]:,}] 人/日")
        print(f"  I ∈ [{self.I_range[0]:,}, {self.I_range[1]:,.0f}] $")
        print(f"  γ1 ∈ [{self.gamma1_range[0]}, {self.gamma1_range[1]}]")
        print(f"  x1 ∈ [{self.x1_range[0]:,}, {self.x1_range[1]:,}]")
        print(f"  x2 ∈ [{self.x2_range[0]:,}, {self.x2_range[1]:,}]")

        print("\n【约束阈值】")
        print(f"  E_min = {self.E_min}")
        print(f"  S_min = {self.S_min}")
        print("="*70 + "\n")


# ============================================================
# 第二部分：Juneau模型核心计算 (Core Model Calculations)
# ============================================================

class JuneauModel:
    """
    Juneau旅游可持续性模型核心类

    实现所有模型方程的计算
    """

    def __init__(self, params: JuneauModelParams = None):
        """
        初始化模型

        :param params: 参数配置对象，若为None则使用默认参数
        """
        self.params = params if params else JuneauModelParams()

    def _get_param(self, name):
        """获取参数值（支持ParameterRange）"""
        param = getattr(self.params, name)
        if isinstance(param, ParameterRange):
            return param.get_value()
        return param

    def natural_demand(self, t):
        """
        计算自然游客需求 N0(t)

        :param t: 年中的天数（1-365）
        :return: 自然需求量（人/日）
        """
        A = self._get_param('A')
        B = self._get_param('B')
        N0 = -A * np.cos(2 * np.pi * t / 365) + B
        return max(N0, 0)

    def actual_demand(self, t, c1, c2):
        """
        计算实际游客数 N(t)

        :param t: 年中的天数
        :param c1: 峰季游客上限
        :param c2: 非峰季游客目标
        :return: 实际游客数
        """
        N0 = self.natural_demand(t)
        if 121 <= t <= 270:  # 峰季 (约5-9月)
            return min(N0, c1)
        else:  # 非峰季
            return max(N0, c2)

    def policy_revenue_cost(self, t, x1, x2):
        """
        计算政策收入/成本函数 f(t)

        :param t: 年中的天数
        :param x1: 峰季税收调整参数
        :param x2: 非峰季补贴调整参数
        :return: 每日政策收入/成本（$）
        """
        # f(t) = (x1 - x2)/2 * cos(2πt/365 + π) + (x1 + x2)/2
        # 这是从峰季x1到非峰季x2的余弦波
        phase_shift = np.pi  # 相移π使峰值在峰季
        f_t = ((x1 - x2) / 2) * np.cos(2 * np.pi * t / 365 + phase_shift) + ((x1 + x2) / 2)
        return f_t

    def economic_profit(self, c1, c2, I, x1, x2):
        """
        计算经济利润 P

        :param c1: 峰季游客上限
        :param c2: 非峰季游客目标
        :param I: 每日投资
        :param x1: 峰季税收调整参数
        :param x2: 非峰季补贴调整参数
        :return: 年经济利润（$）
        """
        p = self._get_param('p')
        P = 0
        for t in range(1, 366):  # 1-365天
            N_t = self.actual_demand(t, c1, c2)
            f_t = self.policy_revenue_cost(t, x1, x2)
            P += N_t * p + f_t - I
        return P

    def environmental_cost(self, t, N_t):
        """
        计算环境成本 E_cost(t)

        :param t: 年中的天数
        :param N_t: 当日游客数
        :return: 当日环境成本（$）
        """
        carbon_cost = self._get_param('carbon_cost_per_tourist')
        return N_t * carbon_cost

    def ecosystem_resilience(self, t, N_t):
        """
        计算生态恢复力 ERI(t)

        :param t: 年中的天数
        :param N_t: 当日游客数
        :return: 当日生态恢复力
        """
        ERI_max = self._get_param('ERI_max')
        beta = self._get_param('beta')
        return ERI_max / (1 + beta * N_t)

    def environmental_investment_return(self, I, gamma1):
        """
        计算环境投资回报 Γ1(I)

        :param I: 每日投资
        :param gamma1: 环境投资比例
        :return: 每日环境投资回报
        """
        Gamma1m = self._get_param('Gamma1m')
        Gamma10 = self._get_param('Gamma10')
        alpha1 = self._get_param('alpha1')
        I10 = self._get_param('I10')

        I_env = gamma1 * I
        if I_env <= I10:
            return Gamma10
        else:
            ratio = Gamma1m / Gamma10 - 1
            exp_term = np.exp(-alpha1 * (I_env - I10))
            return Gamma1m / (1 + ratio * exp_term)

    def environmental_level(self, c1, c2, I, gamma1):
        """
        计算环境水平 E

        :param c1: 峰季游客上限
        :param c2: 非峰季游客目标
        :param I: 每日投资
        :param gamma1: 环境投资比例
        :return: 年环境水平
        """
        E = 0
        Gamma1 = self.environmental_investment_return(I, gamma1)

        for t in range(1, 366):
            N_t = self.actual_demand(t, c1, c2)
            E_cost = self.environmental_cost(t, N_t)
            ERI = self.ecosystem_resilience(t, N_t)
            E += -E_cost + ERI + Gamma1

        return E

    def social_employment_benefit(self):
        """
        计算就业收益 S1（常数）

        :return: 年就业收益（$）
        """
        return self._get_param('S1_yearly')

    def social_negative_impact(self, c1, c2):
        """
        计算社会负面影响 S2

        :param c1: 峰季游客上限
        :param c2: 非峰季游客目标
        :return: 年社会负面影响
        """
        beta2 = self._get_param('beta2')
        S2 = 0
        for t in range(1, 366):
            N_t = self.actual_demand(t, c1, c2)
            # 简化为线性关系，可根据需要调整
            negative_score = beta2 * N_t
            S2 += negative_score
        return S2

    def social_investment_return(self, I, gamma1):
        """
        计算社会投资回报 Γ2(I)

        :param I: 每日投资
        :param gamma1: 环境投资比例
        :return: 每日社会投资回报
        """
        gamma2 = 1 - gamma1
        Gamma2m = self._get_param('Gamma2m')
        Gamma20 = self._get_param('Gamma20')
        alpha2 = self._get_param('alpha2')
        I20 = self._get_param('I20')

        I_social = gamma2 * I
        if I_social <= I20:
            return Gamma20
        else:
            ratio = Gamma2m / Gamma20 - 1
            exp_term = np.exp(-alpha2 * (I_social - I20))
            return Gamma2m / (1 + ratio * exp_term)

    def social_welfare(self, c1, c2, I, gamma1):
        """
        计算社会福利 S

        :param c1: 峰季游客上限
        :param c2: 非峰季游客目标
        :param I: 每日投资
        :param gamma1: 环境投资比例
        :return: 年社会福利
        """
        S1 = self.social_employment_benefit()
        S2 = self.social_negative_impact(c1, c2)
        Gamma2 = self.social_investment_return(I, gamma1)

        S = S1 - S2 + 365 * Gamma2  # Gamma2是每日回报
        return S

    def total_utility(self, c1, c2, I, gamma1, x1, x2):
        """
        计算总社会效用 U = P + E + S

        :param c1: 峰季游客上限
        :param c2: 非峰季游客目标
        :param I: 每日投资
        :param gamma1: 环境投资比例
        :param x1: 峰季税收调整参数
        :param x2: 非峰季补贴调整参数
        :return: 总效用值
        """
        P = self.economic_profit(c1, c2, I, x1, x2)
        E = self.environmental_level(c1, c2, I, gamma1)
        S = self.social_welfare(c1, c2, I, gamma1)
        U = P + E + S
        return U

    def evaluate_policy(self, c1, c2, I, gamma1, x1, x2):
        """
        评估单个政策点的所有指标

        :param c1: 峰季游客上限
        :param c2: 非峰季游客目标
        :param I: 每日投资
        :param gamma1: 环境投资比例
        :param x1: 峰季税收调整参数
        :param x2: 非峰季补贴调整参数
        :return: dict，包含所有中间变量和评价指标
        """
        P = self.economic_profit(c1, c2, I, x1, x2)
        E = self.environmental_level(c1, c2, I, gamma1)
        S = self.social_welfare(c1, c2, I, gamma1)
        U = P + E + S

        return {
            'c1': c1, 'c2': c2, 'I': I, 'gamma1': gamma1, 'x1': x1, 'x2': x2,
            'P': P, 'E': E, 'S': S, 'U': U
        }

    def check_constraints(self, c1, c2, I, gamma1, x1, x2):
        """
        检查约束条件

        :return: 是否满足所有约束
        """
        E = self.environmental_level(c1, c2, I, gamma1)
        S = self.social_welfare(c1, c2, I, gamma1)

        return (E >= self.params.E_min and
                S >= self.params.S_min and
                c1 >= c2 and
                x1 >= x2 and
                0 <= gamma1 <= 1 and
                self.params.I_range[0] <= I <= self.params.I_range[1])


# ============================================================
# 第三部分：政策优化搜索 (Policy Optimization Search)
# ============================================================

class JuneauPolicyOptimizer:
    """
    Juneau政策优化类

    使用scipy.optimize进行非线性优化
    """

    def __init__(self, model: JuneauModel):
        """
        初始化优化器

        :param model: JuneauModel实例
        """
        self.model = model
        self.params = model.params
        self.best_solution = None
        self.optimization_result = None

    def objective_function(self, x):
        """
        目标函数：最大化 U = P + E + S

        :param x: 决策变量 [c1, c2, I, gamma1, x1, x2]
        :return: -U (因为scipy.optimize是最小化)
        """
        c1, c2, I, gamma1, x1, x2 = x
        U = self.model.total_utility(c1, c2, I, gamma1, x1, x2)
        return -U  # 最小化负效用 = 最大化效用

    def constraint_function(self, x):
        """
        约束函数 - 简化约束以确保可行性

        :param x: 决策变量
        :return: 约束值列表
        """
        c1, c2, I, gamma1, x1, x2 = x

        constraints = [
            c1 - c2,                # c1 >= c2
            gamma1,                 # gamma1 >= 0
            0.99 - gamma1,          # gamma1 <= 0.99 (避免边界问题)
            I - self.params.I_range[0],  # I >= I_min
            self.params.I_range[1] - I   # I <= I_max
        ]

        return constraints

    def optimize(self, method='COBYLA', max_iter=5000):
        """
        执行优化

        :param method: 优化方法
        :param max_iter: 最大迭代次数
        :return: 优化结果
        """
        # 变量边界
        bounds = [
            self.params.c1_range,      # c1
            self.params.c2_range,      # c2
            self.params.I_range,       # I
            self.params.gamma1_range,  # gamma1
            self.params.x1_range,      # x1
            self.params.x2_range       # x2
        ]

        # 初始猜测 - 使用更保守的值
        x0 = [
            12000,  # c1 - 峰季游客上限
            3000,   # c2 - 非峰季游客目标
            50000,  # I - 每日投资
            0.5,    # gamma1 - 环境投资比例
            5000,   # x1 - 峰季税收
            -2000   # x2 - 非峰季补贴
        ]

        # 约束
        constraints = {
            'type': 'ineq',
            'fun': self.constraint_function
        }

        # 执行优化
        result = minimize(
            self.objective_function,
            x0,
            method='COBYLA',  # 改用COBYLA方法
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'disp': True}
        )

        self.optimization_result = result

        # 即使没有完全收敛，也接受结果如果它在可行域内
        x_opt = result.x
        if self.check_feasibility(x_opt):
            self.best_solution = {
                'c1': x_opt[0],
                'c2': x_opt[1],
                'I': x_opt[2],
                'gamma1': x_opt[3],
                'x1': x_opt[4],
                'x2': x_opt[5],
                'U': -result.fun,  # 恢复正值
                'success': result.success
            }
            # 计算其他指标
            details = self.model.evaluate_policy(**{k: v for k, v in self.best_solution.items() if k not in ['U', 'success']})
            self.best_solution.update(details)
            print(f"✓ 找到可行解 (U = ${self.best_solution['U']:,.0f})")
        else:
            print("❌ 未找到可行解")
            self.best_solution = None

        return self.best_solution

    def check_feasibility(self, x):
        """
        检查解的可行性

        :param x: 决策变量
        :return: 是否可行
        """
        c1, c2, I, gamma1, x1, x2 = x

        # 检查边界
        if not (self.params.c1_range[0] <= c1 <= self.params.c1_range[1]):
            return False
        if not (self.params.c2_range[0] <= c2 <= self.params.c2_range[1]):
            return False
        if not (self.params.I_range[0] <= I <= self.params.I_range[1]):
            return False
        if not (self.params.gamma1_range[0] <= gamma1 <= self.params.gamma1_range[1]):
            return False
        if not (self.params.x1_range[0] <= x1 <= self.params.x1_range[1]):
            return False
        if not (self.params.x2_range[0] <= x2 <= self.params.x2_range[1]):
            return False

        # 检查约束
        if c1 < c2:
            return False
        if gamma1 < 0 or gamma1 > 0.99:
            return False

        return True

    def get_optimal_policy(self):
        """获取最优政策"""
        return self.best_solution


# ============================================================
# 第四部分：可视化模块 (Visualization Module)
# ============================================================

class JuneauVisualization:
    """
    Juneau模型分析可视化类
    """

    def __init__(self, model: JuneauModel, optimizer: JuneauPolicyOptimizer, save_dir='./figures'):
        self.model = model
        self.optimizer = optimizer
        self.params = model.params
        self.saver = FigureSaver(save_dir)

    def plot_seasonal_demand(self, c1=None, c2=None, figsize=(12, 6)):
        """
        绘制季节性游客需求曲线
        """
        fig, ax = plt.subplots(figsize=figsize)

        t_vals = np.arange(1, 366)
        N0_vals = [self.model.natural_demand(t) for t in t_vals]

        ax.plot(t_vals, N0_vals, 'b-', label='自然需求 N0(t)', linewidth=2)

        if c1 is not None and c2 is not None:
            N_vals = [self.model.actual_demand(t, c1, c2) for t in t_vals]
            ax.plot(t_vals, N_vals, 'r--', label=f'政策调整 N(t) (c1={c1:.0f}, c2={c2:.0f})', linewidth=2)

        ax.axvspan(121, 270, alpha=0.2, color='yellow', label='峰季 (5-9月)')

        ax.set_xlabel('天数 (Day of Year)')
        ax.set_ylabel('游客数 (Tourists per Day)')
        ax.set_title('Juneau游客季节性需求曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.saver.save(fig, 'seasonal_demand')
        plt.show()

    def plot_policy_revenue_cost(self, x1, x2, figsize=(12, 6)):
        """
        绘制政策收入/成本函数
        """
        fig, ax = plt.subplots(figsize=figsize)

        t_vals = np.arange(1, 366)
        f_vals = [self.model.policy_revenue_cost(t, x1, x2) for t in t_vals]

        ax.plot(t_vals, f_vals, 'g-', linewidth=2, label=f'政策函数 f(t) (x1={x1:.0f}, x2={x2:.0f})')

        ax.axvspan(121, 270, alpha=0.2, color='yellow', label='峰季 (5-9月)')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        ax.set_xlabel('天数 (Day of Year)')
        ax.set_ylabel('政策收入/成本 ($ per Day)')
        ax.set_title('政策收入/成本函数')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.saver.save(fig, 'policy_revenue_cost')
        plt.show()

    def plot_investment_returns(self, I_max=100000, figsize=(12, 6)):
        """
        绘制投资回报函数
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        I_vals = np.linspace(0, I_max, 100)
        gamma1 = 0.5  # 假设50%分配给环境

        # 环境投资回报
        Gamma1_vals = [self.model.environmental_investment_return(I, gamma1) for I in I_vals]
        ax1.plot(I_vals, Gamma1_vals, 'b-', linewidth=2, label='环境投资回报 Γ1(I)')
        ax1.set_xlabel('每日投资 ($)')
        ax1.set_ylabel('每日回报 ($)')
        ax1.set_title('环境投资回报函数')
        ax1.grid(True, alpha=0.3)

        # 社会投资回报
        Gamma2_vals = [self.model.social_investment_return(I, gamma1) for I in I_vals]
        ax2.plot(I_vals, Gamma2_vals, 'r-', linewidth=2, label='社会投资回报 Γ2(I)')
        ax2.set_xlabel('每日投资 ($)')
        ax2.set_ylabel('每日回报 ($)')
        ax2.set_title('社会投资回报函数')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self.saver.save(fig, 'investment_returns')
        plt.show()

    def plot_optimal_policy_summary(self, figsize=(14, 8)):
        """
        绘制最优政策摘要
        """
        if self.optimizer.best_solution is None:
            print("没有最优解可显示")
            return

        sol = self.optimizer.best_solution

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 决策变量
        variables = ['c1', 'c2', 'I', 'gamma1']
        values = [sol['c1'], sol['c2'], sol['I'], sol['gamma1']]
        labels = ['峰季游客上限\n(人/日)', '非峰季游客目标\n(人/日)', '每日投资\n($)', '环境投资比例']

        axes[0,0].bar(labels, values, color=PlotStyleConfig.get_palette(4))
        axes[0,0].set_title('最优决策变量', fontsize=12, fontweight='bold')
        axes[0,0].tick_params(axis='x', rotation=45)

        # 政策参数
        policy_vars = ['x1', 'x2']
        policy_vals = [sol['x1'], sol['x2']]
        policy_labels = ['峰季税收调整\n($)', '非峰季补贴调整\n($)']

        axes[0,1].bar(policy_labels, policy_vals, color=['green', 'red'])
        axes[0,1].set_title('政策调整参数', fontsize=12, fontweight='bold')
        axes[0,1].tick_params(axis='x', rotation=45)

        # 目标函数分量
        objectives = ['P', 'E', 'S', 'U']
        obj_values = [sol['P'], sol['E'], sol['S'], sol['U']]
        obj_labels = ['经济利润\n($)', '环境水平\n($)', '社会福利\n($)', '总效用\n($)']

        axes[1,0].bar(obj_labels, obj_values, color=PlotStyleConfig.get_palette(4))
        axes[1,0].set_title('目标函数分量', fontsize=12, fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)

        # 约束检查
        constraints = ['环境约束满足', '社会约束满足', '游客容量约束', '税收补贴约束', '投资比例约束', '投资范围约束']
        status = [
            sol['E'] >= self.params.E_min,
            sol['S'] >= self.params.S_min,
            sol['c1'] >= sol['c2'],
            sol['x1'] >= sol['x2'],
            0 <= sol['gamma1'] <= 1,
            self.params.I_range[0] <= sol['I'] <= self.params.I_range[1]
        ]

        colors = ['green' if s else 'red' for s in status]
        axes[1,1].bar(constraints, [1]*len(status), color=colors)
        axes[1,1].set_title('约束满足情况', fontsize=12, fontweight='bold')
        axes[1,1].set_ylim(0, 1.5)
        axes[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        self.saver.save(fig, 'optimal_policy_summary')
        plt.show()

    def test_chinese_support(self, figsize=(10, 6)):
        """
        测试中文字体支持
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 创建一些测试数据
        x = ['经济', '环境', '社会', '总效用']
        y = [100, 200, 50, 350]

        bars = ax.bar(x, y, color=PlotStyleConfig.get_palette(4))
        ax.set_title('中文测试图表 - Chinese Font Test', fontsize=14, fontweight='bold')
        ax.set_xlabel('维度 (Dimensions)')
        ax.set_ylabel('数值 (Values)')

        # 添加数值标签
        for bar, value in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{value}', ha='center', va='bottom', fontsize=10)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.saver.save(fig, 'chinese_test')
        plt.show()


# ============================================================
# 第五部分：主工作流 (Main Workflow)
# ============================================================

def run_juneau_workflow():
    """
    运行完整的Juneau旅游可持续性模型工作流

    包括：参数配置 → 模型创建 → 优化求解 → 可视化
    """
    print("\n" + "█"*70)
    print("█" + " "*20 + "Juneau旅游可持续性模型" + " "*21 + "█")
    print("█" + " "*15 + "Juneau Tourism Sustainability Model" + " "*16 + "█")
    print("█"*70 + "\n")

    # ========== Step 1: 参数配置 ==========
    print("【Step 1】初始化模型参数...")
    params = JuneauModelParams()

    # ★★★ 在这里修改你的参数 ★★★
    # params.A = 18000        # 调整流量振幅
    # params.p = 250          # 调整每人利润
    # params.SCC = 200        # 调整碳成本

    params.summary()

    # ========== Step 2: 创建模型 ==========
    print("【Step 2】创建Juneau模型...")
    model = JuneauModel(params)

    # ========== Step 3: 优化求解 ==========
    print("【Step 3】执行政策优化...")
    optimizer = JuneauPolicyOptimizer(model)
    optimal_policy = optimizer.optimize(method='COBYLA', max_iter=5000)

    if optimal_policy:
        print("\n【最优政策结果】")
        print("-"*70)
        print(f"峰季游客上限 c1* = {optimal_policy['c1']:,.0f} 人/日")
        print(f"非峰季游客目标 c2* = {optimal_policy['c2']:,.0f} 人/日")
        print(f"每日投资 I* = ${optimal_policy['I']:,.0f}")
        print(f"环境投资比例 γ1* = {optimal_policy['gamma1']:.3f}")
        print(f"峰季税收调整 x1* = ${optimal_policy['x1']:,.0f}")
        print(f"非峰季补贴调整 x2* = ${optimal_policy['x2']:,.0f}")
        print(f"经济利润 P* = ${optimal_policy['P']:,.0f}")
        print(f"环境水平 E* = ${optimal_policy['E']:,.0f}")
        print(f"社会福利 S* = ${optimal_policy['S']:,.0f}")
        print(f"总效用 U* = ${optimal_policy['U']:,.0f}")
    else:
        print("❌ 优化失败，无法找到可行解")
        return

    # ========== Step 4: 可视化 ==========
    print("\n【Step 4】生成可视化图表...")
    print("-"*70)

    # 创建figures目录
    os.makedirs('./figures', exist_ok=True)

    viz = JuneauVisualization(model, optimizer, save_dir='./figures')

    # 图1: 季节性需求
    print("\n  🎨 绘制季节性游客需求曲线...")
    viz.plot_seasonal_demand(optimal_policy['c1'], optimal_policy['c2'])

    # 图2: 政策收入/成本函数
    print("\n  🎨 绘制政策收入/成本函数...")
    viz.plot_policy_revenue_cost(optimal_policy['x1'], optimal_policy['x2'])

    # 图3: 投资回报函数
    print("\n  🎨 绘制投资回报函数...")
    viz.plot_investment_returns()

    # 图4: 最优政策摘要
    print("\n  🎨 绘制最优政策摘要...")
    viz.plot_optimal_policy_summary()

    # 图5: 中文测试
    print("\n  🎨 测试中文字体支持...")
    viz.test_chinese_support()

    # ========== Step 5: 保存结果 ==========
    print("\n【Step 5】保存结果...")
    print("-"*70)

    # 保存最优政策结果
    result_df = pd.DataFrame([optimal_policy])
    result_df.to_csv('./figures/juneau_optimal_policy.csv', index=False)
    print("  📁 最优政策结果已保存: ./figures/juneau_optimal_policy.csv")

    print("\n" + "█"*70)
    print("█" + " "*25 + "工作流执行完成!" + " "*26 + "█")
    print("█"*70 + "\n")

    return params, model, optimizer, viz


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # ★★★ 使用示例: 运行完整工作流 ★★★
    # ============================================================
    params, model, optimizer, viz = run_juneau_workflow()

    # ============================================================
    # ★★★ 其他自定义分析 ★★★
    # ============================================================

    # 1. 查看特定政策的详细评估结果
    # result = model.evaluate_policy(c1=15000, c2=5000, I=50000, gamma1=0.6, x1=10000, x2=-5000)
    # print(result)

    # 2. 比较不同投资分配比例的效果
    # for gamma1 in [0.2, 0.4, 0.6, 0.8]:
    #     result = model.evaluate_policy(c1=15000, c2=5000, I=50000, gamma1=gamma1, x1=10000, x2=-5000)
    #     print(f"γ1={gamma1}: U=${result['U']:,.0f}")

    # 3. 分析季节性需求模式
    # t_vals = np.arange(1, 366)
    # demands = [model.natural_demand(t) for t in t_vals]
    # plt.plot(t_vals, demands)
    # plt.show()