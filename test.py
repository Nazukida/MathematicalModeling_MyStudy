"""
============================================================
旅游政策优化模型 - 完整工作流
(Tourism Policy Optimization Model - Complete Workflow)
============================================================
功能：多目标旅游政策优化（经济、环境、居民满意度）
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================

模型框架：
1. 游客需求函数 D(f,t) = D0 - af - bt
2. 收入模型 R = fV + (t/100)(θcV)
3. 经济维度 g1: Π = cV + R - Cost(V)
4. 环境维度 g2: E = αV - βI
5. 居民维度 g3: S = S0 - γmax(0, V/cap - 1) + δI
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
        """设置全局绘图风格"""
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('seaborn-whitegrid')
        rcParams['figure.figsize'] = (12, 8)
        rcParams['figure.dpi'] = 100
        rcParams['savefig.dpi'] = 300
        rcParams['font.size'] = 11
        rcParams['axes.titlesize'] = 14
        rcParams['axes.labelsize'] = 12
        rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        rcParams['axes.unicode_minus'] = False
    
    @staticmethod
    def get_palette(n=None):
        """获取配色板"""
        palette = PlotStyleConfig.PALETTE
        if n is not None:
            if n <= len(palette):
                return palette[:n]
            else:
                return [palette[i % len(palette)] for i in range(n)]
        return palette


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
            fig.tight_layout()
        paths = []
        for fmt in formats:
            path = os.path.join(self.save_dir, f"{filename}.{fmt}")
            fig.savefig(path, format=fmt, bbox_inches='tight', 
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


class TourismPolicyParams:
    """
    旅游政策模型参数配置类
    
    ★★★ 需要调整的参数在这里修改 ★★★
    
    支持两种方式定义参数：
    1. 固定值: self.D0 = 20000
    2. 范围:   self.D0 = ParameterRange(18000, 22000)  # 均匀分布
              self.D0 = ParameterRange(20000, std=1000)  # 正态分布
    """
    
    def __init__(self):
        # ============ 游客需求模型参数 ============
        # 可以用 ParameterRange(下界, 上界) 定义范围
        self.D0 = 20000       # 零收费零税下的潜在需求（人/日）
        self.a = 120          # 游客费敏感度：每+$1，需求下降120人/日
        self.b = 350          # 酒店税敏感度：每+1%，需求下降350人/日
        
        # ============ 收入模型参数 ============
        self.c = 250          # 人均总消费（$/人/日，不含游客费）
        self.theta = 0.35     # 住宿/应税部分占比（可调至0.15-0.25若游轮一日游）
        
        # ============ 成本模型参数 ============
        self.a0 = 40          # 基础公共服务边际成本（$/人）
        self.a1 = 0.003       # 拥挤导致的凸成本系数（$/人²）
        
        # ============ 环境模型参数 ============
        self.alpha = 1.0        # 每个游客贡献1单位环境压力
        self.beta = 0.00005   # 每$1治理投入抵消5e-5单位压力
        
        # ============ 居民满意度模型参数 ============
        self.cap = 12000      # 居民可接受阈值（人/日）
        self.S0 = 0.85        # 基准满意度（0-1）
        self.gamma = 0.30     # 超载惩罚系数
        self.delta = 2e-8     # 投入补偿效应系数
        
        # ============ 政策变量范围 ============
        self.N_range = (5000, 20000)   # 游客容量限制（人/日）
        self.f_range = (0, 50)          # 游客费（$）
        self.t_range = (0, 15)          # 酒店税率（%）
        self.x_range = (0, 0.8)         # 再投资比例
        
        # ============ 约束阈值 ============
        self.g2_threshold = 0.60    # 环境维度及格线
        self.g3_threshold = 0.65    # 居民满意度及格线
        
        # ============ 网格搜索分辨率 ============
        self.N_steps = 16      # N的网格数量
        self.f_steps = 11      # f的网格数量
        self.t_steps = 16      # t的网格数量
        self.x_steps = 9       # x的网格数量
    
    def _get_param_value(self, param):
        """获取参数值（支持固定值或ParameterRange）"""
        if isinstance(param, ParameterRange):
            return param.get_value()
        return param
    
    def _get_param_display(self, param):
        """获取参数显示字符串"""
        if isinstance(param, ParameterRange):
            return f"{param}"
        return f"{param}"
    
    def has_uncertainty(self):
        """检查是否有不确定性参数"""
        params_to_check = ['D0', 'a', 'b', 'c', 'theta', 'a0', 'a1', 
                          'alpha', 'beta', 'cap', 'S0', 'gamma', 'delta']
        for name in params_to_check:
            param = getattr(self, name)
            if isinstance(param, ParameterRange) and param.is_range:
                return True
        return False
    
    def get_uncertain_params(self):
        """获取所有不确定性参数的名称和范围"""
        params_to_check = ['D0', 'a', 'b', 'c', 'theta', 'a0', 'a1', 
                          'alpha', 'beta', 'cap', 'S0', 'gamma', 'delta']
        uncertain = {}
        for name in params_to_check:
            param = getattr(self, name)
            if isinstance(param, ParameterRange) and param.is_range:
                uncertain[name] = param
        return uncertain
    
    def sample_params(self):
        """采样一组参数值，返回新的参数对象"""
        sampled = TourismPolicyParams()
        params_to_sample = ['D0', 'a', 'b', 'c', 'theta', 'a0', 'a1', 
                           'alpha', 'beta', 'cap', 'S0', 'gamma', 'delta']
        for name in params_to_sample:
            param = getattr(self, name)
            if isinstance(param, ParameterRange):
                setattr(sampled, name, param.sample(1)[0])
            else:
                setattr(sampled, name, param)
        # 复制其他参数
        sampled.N_range = self.N_range
        sampled.f_range = self.f_range
        sampled.t_range = self.t_range
        sampled.x_range = self.x_range
        sampled.g2_threshold = self.g2_threshold
        sampled.g3_threshold = self.g3_threshold
        sampled.N_steps = self.N_steps
        sampled.f_steps = self.f_steps
        sampled.t_steps = self.t_steps
        sampled.x_steps = self.x_steps
        return sampled
    
    def summary(self):
        """打印参数摘要"""
        print("\n" + "="*70)
        print("📋 旅游政策模型参数配置 (Tourism Policy Model Parameters)")
        print("="*70)
        
        # 检查是否有不确定性参数
        if self.has_uncertainty():
            print("\n⚠️  检测到不确定性参数（范围定义），将进行蒙特卡洛分析")
            uncertain = self.get_uncertain_params()
            print(f"    不确定参数: {list(uncertain.keys())}")
        
        print("\n【游客需求模型】 D(f,t) = D0 - a*f - b*t")
        print(f"  D0 = {self._get_param_display(self.D0)} 人/日 (潜在需求)")
        print(f"  a  = {self._get_param_display(self.a)} (游客费敏感度)")
        print(f"  b  = {self._get_param_display(self.b)} (酒店税敏感度)")
        
        print("\n【收入模型】 R = f*V + (t/100)*θ*c*V")
        print(f"  c     = ${self._get_param_display(self.c)}/人/日 (人均消费)")
        print(f"  θ     = {self._get_param_display(self.theta)} (住宿应税比例)")
        
        print("\n【成本模型】 Cost(V) = a0*V + a1*V²")
        print(f"  a0 = ${self._get_param_display(self.a0)}/人 (边际成本)")
        print(f"  a1 = ${self._get_param_display(self.a1)}/人² (拥挤成本)")
        
        print("\n【环境模型】 E = α*V - β*I")
        print(f"  α = {self._get_param_display(self.alpha)} (环境压力系数)")
        print(f"  β = {self._get_param_display(self.beta)} (治理效果系数)")
        
        print("\n【居民满意度】 S = S0 - γ*max(0, V/cap-1) + δ*I")
        print(f"  cap   = {self._get_param_display(self.cap)} 人/日 (承载阈值)")
        print(f"  S0    = {self._get_param_display(self.S0)} (基准满意度)")
        print(f"  γ     = {self._get_param_display(self.gamma)} (超载惩罚)")
        print(f"  δ     = {self._get_param_display(self.delta)} (投入补偿)")
        
        print("\n【政策变量范围】")
        print(f"  N ∈ [{self.N_range[0]:,}, {self.N_range[1]:,}] 人/日")
        print(f"  f ∈ [{self.f_range[0]}, {self.f_range[1]}] $")
        print(f"  t ∈ [{self.t_range[0]}, {self.t_range[1]}] %")
        print(f"  x ∈ [{self.x_range[0]}, {self.x_range[1]}]")
        
        print("\n【约束阈值】")
        print(f"  g2_bar = {self.g2_threshold} (环境及格线)")
        print(f"  g3_bar = {self.g3_threshold} (居民满意度及格线)")
        print("="*70 + "\n")


# ============================================================
# 第二部分：旅游政策模型核心计算 (Core Model Calculations)
# ============================================================

class TourismPolicyModel:
    """
    旅游政策优化模型核心类
    
    实现所有模型方程的计算
    支持参数范围（ParameterRange）
    """
    
    def __init__(self, params: TourismPolicyParams = None):
        """
        初始化模型
        
        :param params: 参数配置对象，若为None则使用默认参数
        """
        self.params = params if params else TourismPolicyParams()
        self.E_min = None
        self.E_max = None
        self.S_min = 0
        self.S_max = 1
    
    def _get_param(self, name):
        """获取参数值（支持ParameterRange）"""
        param = getattr(self.params, name)
        if isinstance(param, ParameterRange):
            return param.get_value()
        return param
        
    def demand(self, f, t):
        """
        计算游客需求 D(f,t)
        
        :param f: 游客费（$）
        :param t: 酒店税率（%）
        :return: 需求量（人/日）
        """
        D0 = self._get_param('D0')
        a = self._get_param('a')
        b = self._get_param('b')
        D = D0 - a * f - b * t
        return max(0, D)  # 需求不能为负
    
    def actual_visitors(self, N, f, t):
        """
        计算实际到访游客数 V = min(N, D(f,t))
        
        :param N: 游客容量限制（人/日）
        :param f: 游客费（$）
        :param t: 酒店税率（%）
        :return: 实际游客数
        """
        D = self.demand(f, t)
        return min(N, D)
    
    def revenue(self, V, f, t):
        """
        计算政府收入 R = f*V + (t/100)*θ*c*V
        
        :param V: 实际游客数
        :param f: 游客费（$）
        :param t: 酒店税率（%）
        :return: 政府收入（$）
        """
        theta = self._get_param('theta')
        c = self._get_param('c')
        R = f * V + (t / 100) * theta * c * V
        return R
    
    def reinvestment(self, R, x):
        """
        计算再投资金额 I = x * R
        
        :param R: 政府收入（$）
        :param x: 再投资比例
        :return: 再投资金额（$）
        """
        return x * R
    
    def cost(self, V):
        """
        计算公共服务成本 Cost(V) = a0*V + a1*V²
        
        :param V: 实际游客数
        :return: 成本（$）
        """
        a0 = self._get_param('a0')
        a1 = self._get_param('a1')
        return a0 * V + a1 * V**2
    
    def economic_score(self, V, R):
        """
        计算经济维度得分 g1 = Π = c*V + R - Cost(V)
        
        注意：这里返回的是净经济效益（$），不是归一化得分
        
        :param V: 实际游客数
        :param R: 政府收入（$）
        :return: 净经济效益Π（$）
        """
        c = self._get_param('c')
        Pi = c * V + R - self.cost(V)
        return Pi
    
    def environmental_pressure(self, V, I):
        """
        计算环境压力 E = α*V - β*I
        
        :param V: 实际游客数
        :param I: 再投资金额（$）
        :return: 环境压力指数
        """
        alpha = self._get_param('alpha')
        beta = self._get_param('beta')
        E = alpha * V - beta * I
        return E
    
    def environmental_score(self, E):
        """
        计算环境维度得分 g2 = 1 - (E - E_min) / (E_max - E_min)
        
        需要先调用 compute_bounds() 来计算 E_min, E_max
        
        :param E: 环境压力
        :return: 环境得分 g2 ∈ [0,1]
        """
        if self.E_min is None or self.E_max is None:
            raise ValueError("请先调用 compute_bounds() 计算边界值")
        
        if self.E_max == self.E_min:
            return 0.5
        
        g2 = 1 - (E - self.E_min) / (self.E_max - self.E_min)
        return np.clip(g2, 0, 1)
    
    def resident_satisfaction(self, V, I):
        """
        计算居民满意度 S = S0 - γ*max(0, V/cap - 1) + δ*I
        
        :param V: 实际游客数
        :param I: 再投资金额（$）
        :return: 居民满意度 S
        """
        cap = self._get_param('cap')
        S0 = self._get_param('S0')
        gamma = self._get_param('gamma')
        delta = self._get_param('delta')
        overload = max(0, V / cap - 1)
        S = S0 - gamma * overload + delta * I
        return S
    
    def resident_score(self, S):
        """
        计算居民维度得分 g3 = (S - S_min) / (S_max - S_min)
        
        :param S: 居民满意度
        :return: 居民得分 g3 ∈ [0,1]
        """
        if self.S_max == self.S_min:
            return 0.5
        
        g3 = (S - self.S_min) / (self.S_max - self.S_min)
        return np.clip(g3, 0, 1)
    
    def evaluate_policy(self, N, f, t, x):
        """
        评估单个政策点的所有指标
        
        :param N: 游客容量限制（人/日）
        :param f: 游客费（$）
        :param t: 酒店税率（%）
        :param x: 再投资比例
        :return: dict，包含所有中间变量和评价指标
        """
        # 计算中间变量
        D = self.demand(f, t)
        V = self.actual_visitors(N, f, t)
        R = self.revenue(V, f, t)
        I = self.reinvestment(R, x)
        Cost = self.cost(V)
        
        # 计算各维度指标
        Pi = self.economic_score(V, R)  # 经济效益（原始值，单位$）
        E = self.environmental_pressure(V, I)
        S = self.resident_satisfaction(V, I)
        
        return {
            'N': N, 'f': f, 't': t, 'x': x,  # 政策变量
            'D': D, 'V': V, 'R': R, 'I': I, 'Cost': Cost,  # 中间变量
            'Pi': Pi, 'E': E, 'S': S  # 原始维度值
        }
    
    def compute_bounds(self, N_vals, f_vals, t_vals, x_vals):
        """
        通过网格扫描计算 E_min, E_max 边界
        
        :param N_vals: N取值列表
        :param f_vals: f取值列表
        :param t_vals: t取值列表
        :param x_vals: x取值列表
        """
        print("  🔍 正在计算边界值 (Computing bounds)...")
        
        E_list = []
        S_list = []
        
        for N in N_vals:
            for f in f_vals:
                for t in t_vals:
                    for x in x_vals:
                        result = self.evaluate_policy(N, f, t, x)
                        E_list.append(result['E'])
                        S_list.append(result['S'])
        
        self.E_min = min(E_list)
        self.E_max = max(E_list)
        self.S_min = min(S_list)
        self.S_max = max(S_list)
        
        print(f"    E_min = {self.E_min:.2f}, E_max = {self.E_max:.2f}")
        print(f"    S_min = {self.S_min:.4f}, S_max = {self.S_max:.4f}")
        # print(f"\n  ✅ 完成 {len(MonteCarloAnalysis.result_df)} 次有效模拟")


# ============================================================
# 第三部分：政策网格搜索 (Policy Grid Search)
# ============================================================

class PolicyGridSearch:
    """
    四维政策网格搜索类
    
    搜索最优政策组合
    """
    
    def __init__(self, model: TourismPolicyModel):
        """
        初始化网格搜索
        
        :param model: TourismPolicyModel实例
        """
        self.model = model
        self.params = model.params
        self.results_df = None
        self.feasible_df = None
        
    def create_grid(self):
        """创建四维搜索网格"""
        p = self.params
        
        N_vals = np.linspace(p.N_range[0], p.N_range[1], p.N_steps)
        f_vals = np.linspace(p.f_range[0], p.f_range[1], p.f_steps)
        t_vals = np.linspace(p.t_range[0], p.t_range[1], p.t_steps)
        x_vals = np.linspace(p.x_range[0], p.x_range[1], p.x_steps)
        
        return N_vals, f_vals, t_vals, x_vals
    
    def run_search(self, verbose=True):
        """
        执行网格搜索
        
        :param verbose: 是否打印详细信息
        :return: DataFrame，所有政策点的评价结果
        """
        if verbose:
            print("\n" + "="*70)
            print("🔎 开始政策网格搜索 (Policy Grid Search)")
            print("="*70)
        
        N_vals, f_vals, t_vals, x_vals = self.create_grid()
        
        total_points = len(N_vals) * len(f_vals) * len(t_vals) * len(x_vals)
        if verbose:
            print(f"  网格规模: {len(N_vals)}×{len(f_vals)}×{len(t_vals)}×{len(x_vals)} = {total_points:,} 个政策点")
        
        # 首先计算边界值
        self.model.compute_bounds(N_vals, f_vals, t_vals, x_vals)
        
        if verbose:
            print("  📊 正在评估所有政策点...")
        
        results = []
        count = 0
        for N in N_vals:
            for f in f_vals:
                for t in t_vals:
                    for x in x_vals:
                        result = self.model.evaluate_policy(N, f, t, x)
                        
                        # 计算归一化得分
                        g2 = self.model.environmental_score(result['E'])
                        g3 = self.model.resident_score(result['S'])
                        
                        result['g1'] = result['Pi']  # g1就是经济效益Π
                        result['g2'] = g2
                        result['g3'] = g3
                        
                        results.append(result)
                        count += 1
        
        self.results_df = pd.DataFrame(results)
        
        if verbose:
            print(f"  ✅ 评估完成，共 {len(self.results_df):,} 个政策点")
        
        return self.results_df
    
    def filter_feasible(self, verbose=True):
        """
        筛选满足约束的可行解
        
        :param verbose: 是否打印详细信息
        :return: DataFrame，可行的政策点
        """
        if self.results_df is None:
            raise ValueError("请先调用 run_search() 进行网格搜索")
        
        p = self.params
        
        # 筛选条件: g2 >= g2_bar AND g3 >= g3_bar
        mask = (self.results_df['g2'] >= p.g2_threshold) & \
               (self.results_df['g3'] >= p.g3_threshold)
        
        self.feasible_df = self.results_df[mask].copy()
        
        if verbose:
            total = len(self.results_df)
            feasible = len(self.feasible_df)
            print(f"\n  📋 可行解筛选结果:")
            print(f"    约束条件: g2 ≥ {p.g2_threshold}, g3 ≥ {p.g3_threshold}")
            print(f"    可行解数量: {feasible:,} / {total:,} ({100*feasible/total:.1f}%)")
        
        return self.feasible_df
    
    def get_top_policies(self, n=5, sort_by='g1', ascending=False):
        """
        获取Top N政策
        
        :param n: 返回数量
        :param sort_by: 排序依据（'g1', 'g2', 'g3'）
        :param ascending: 是否升序
        :return: DataFrame
        """
        if self.feasible_df is None or len(self.feasible_df) == 0:
            print("  ⚠️ 无可行解!")
            return None
        
        top_df = self.feasible_df.sort_values(by=sort_by, ascending=ascending).head(n)
        return top_df.reset_index(drop=True)
    
    def get_optimal_policy(self):
        """
        获取最优政策（g1最大的可行解）
        
        :return: dict，最优政策
        """
        if self.feasible_df is None or len(self.feasible_df) == 0:
            print("  ⚠️ 无可行解!")
            return None
        
        best_idx = self.feasible_df['g1'].idxmax()
        return self.feasible_df.loc[best_idx].to_dict()
    
    def summary(self):
        """打印搜索结果摘要"""
        print("\n" + "="*70)
        print("📊 政策搜索结果摘要 (Policy Search Results Summary)")
        print("="*70)
        
        if self.results_df is None:
            print("  ⚠️ 尚未进行搜索，请先调用 run_search()")
            return
        
        # 全局统计
        print("\n【全局统计】")
        print(f"  总政策点数: {len(self.results_df):,}")
        print(f"  g1 (经济效益Π): [{self.results_df['g1'].min():,.0f}, {self.results_df['g1'].max():,.0f}] $")
        print(f"  g2 (环境得分): [{self.results_df['g2'].min():.3f}, {self.results_df['g2'].max():.3f}]")
        print(f"  g3 (居民得分): [{self.results_df['g3'].min():.3f}, {self.results_df['g3'].max():.3f}]")
        
        if self.feasible_df is not None:
            print(f"\n【可行解统计】")
            print(f"  可行解数量: {len(self.feasible_df):,}")
            
            if len(self.feasible_df) > 0:
                best = self.get_optimal_policy()
                print(f"\n【最优政策（g1最大）】")
                print(f"  N = {best['N']:,.0f} 人/日")
                print(f"  f = ${best['f']:.1f}")
                print(f"  t = {best['t']:.1f}%")
                print(f"  x = {best['x']:.2f}")
                print(f"  ────────────────")
                print(f"  V = {best['V']:,.0f} 人/日")
                print(f"  R = ${best['R']:,.0f}")
                print(f"  I = ${best['I']:,.0f}")
                print(f"  ────────────────")
                print(f"  g1 (Π) = ${best['g1']:,.0f}")
                print(f"  g2 = {best['g2']:.3f}")
                print(f"  g3 = {best['g3']:.3f}")
        
        print("="*70 + "\n")


# ============================================================
# 第四部分：可视化模块 (Visualization Module)
# ============================================================

class PolicyVisualization:
    """
    政策分析可视化类
    """
    
    def __init__(self, search: PolicyGridSearch, save_dir='./figures'):
        """
        初始化可视化
        
        :param search: PolicyGridSearch实例
        :param save_dir: 图表保存目录
        """
        self.search = search
        self.model = search.model
        self.params = search.params
        self.saver = FigureSaver(save_dir)
        
    def plot_feasible_region(self, figsize=(14, 5)):
        """
        绘制可行域散点图
        
        图1: g1 vs g2
        图2: g1 vs g3
        图3: g2 vs g3
        """
        if self.search.results_df is None:
            print("请先运行搜索!")
            return
        
        df = self.search.results_df.copy()
        feasible = self.search.feasible_df
        
        # 标记可行解
        df['feasible'] = (df['g2'] >= self.params.g2_threshold) & \
                         (df['g3'] >= self.params.g3_threshold)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 采样绘制（避免点太多）
        sample_size = min(5000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        # 图1: g1 vs g2
        ax1 = axes[0]
        colors1 = np.where(df_sample['feasible'], PlotStyleConfig.COLORS['success'], '#CCCCCC')
        ax1.scatter(df_sample['g1']/1e6, df_sample['g2'], c=colors1, alpha=0.4, s=10)
        ax1.axhline(y=self.params.g2_threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'g2 threshold = {self.params.g2_threshold}')
        ax1.set_xlabel('Economic Performance g1 (Million $)', fontweight='bold')
        ax1.set_ylabel('Environmental Score g2', fontweight='bold')
        ax1.set_title('g1 vs g2 (Feasible Region)', fontweight='bold')
        ax1.legend()
        
        # 图2: g1 vs g3
        ax2 = axes[1]
        colors2 = np.where(df_sample['feasible'], PlotStyleConfig.COLORS['success'], '#CCCCCC')
        ax2.scatter(df_sample['g1']/1e6, df_sample['g3'], c=colors2, alpha=0.4, s=10)
        ax2.axhline(y=self.params.g3_threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'g3 threshold = {self.params.g3_threshold}')
        ax2.set_xlabel('Economic Performance g1 (Million $)', fontweight='bold')
        ax2.set_ylabel('Resident Satisfaction g3', fontweight='bold')
        ax2.set_title('g1 vs g3 (Feasible Region)', fontweight='bold')
        ax2.legend()
        
        # 图3: g2 vs g3
        ax3 = axes[2]
        colors3 = np.where(df_sample['feasible'], PlotStyleConfig.COLORS['success'], '#CCCCCC')
        scatter = ax3.scatter(df_sample['g2'], df_sample['g3'], 
                             c=df_sample['g1']/1e6, alpha=0.6, s=15, cmap='viridis')
        ax3.axvline(x=self.params.g2_threshold, color='red', linestyle='--', linewidth=2)
        ax3.axhline(y=self.params.g3_threshold, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Environmental Score g2', fontweight='bold')
        ax3.set_ylabel('Resident Satisfaction g3', fontweight='bold')
        ax3.set_title('g2 vs g3 (Color by g1)', fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('g1 (Million $)')
        
        plt.tight_layout()
        self.saver.save(fig, 'feasible_region', formats=['png', 'pdf'])
        plt.show()
        
        return fig
    
    def plot_pareto_frontier(self, figsize=(12, 5)):
        """
        绘制Pareto前沿
        
        展示g1与g2、g1与g3的权衡关系
        """
        if self.search.feasible_df is None or len(self.search.feasible_df) == 0:
            print("无可行解，无法绘制Pareto前沿!")
            return
        
        feasible = self.search.feasible_df.copy()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 图1: g1 vs g2 Pareto
        ax1 = axes[0]
        ax1.scatter(feasible['g1']/1e6, feasible['g2'], 
                   c=PlotStyleConfig.COLORS['primary'], alpha=0.5, s=20, label='Feasible solutions')
        
        # 找Pareto最优点（g1和g2都尽量大）
        pareto_mask_1 = self._find_pareto_optimal(feasible, ['g1', 'g2'])
        pareto_1 = feasible[pareto_mask_1].sort_values('g1')
        ax1.plot(pareto_1['g1']/1e6, pareto_1['g2'], 'r-o', linewidth=2, markersize=6, 
                label='Pareto frontier')
        
        ax1.set_xlabel('Economic Performance g1 (Million $)', fontweight='bold')
        ax1.set_ylabel('Environmental Score g2', fontweight='bold')
        ax1.set_title('Pareto Frontier: g1 vs g2', fontweight='bold')
        ax1.legend()
        
        # 图2: g1 vs g3 Pareto
        ax2 = axes[1]
        ax2.scatter(feasible['g1']/1e6, feasible['g3'], 
                   c=PlotStyleConfig.COLORS['secondary'], alpha=0.5, s=20, label='Feasible solutions')
        
        pareto_mask_2 = self._find_pareto_optimal(feasible, ['g1', 'g3'])
        pareto_2 = feasible[pareto_mask_2].sort_values('g1')
        ax2.plot(pareto_2['g1']/1e6, pareto_2['g3'], 'r-o', linewidth=2, markersize=6, 
                label='Pareto frontier')
        
        ax2.set_xlabel('Economic Performance g1 (Million $)', fontweight='bold')
        ax2.set_ylabel('Resident Satisfaction g3', fontweight='bold')
        ax2.set_title('Pareto Frontier: g1 vs g3', fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        self.saver.save(fig, 'pareto_frontier', formats=['png', 'pdf'])
        plt.show()
        
        return fig
    
    def _find_pareto_optimal(self, df, objectives):
        """
        找Pareto最优解（假设都是最大化）
        """
        values = df[objectives].values
        n = len(values)
        is_pareto = np.ones(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # 如果j在所有目标上都不劣于i，且至少一个目标严格优于i
                    if all(values[j] >= values[i]) and any(values[j] > values[i]):
                        is_pareto[i] = False
                        break
        
        return is_pareto
    
    def plot_policy_comparison(self, top_n=5, baseline=None, figsize=(12, 6)):
        """
        绘制政策对比条形图
        
        :param top_n: 对比的Top N政策
        :param baseline: 基准政策（dict）
        """
        top_policies = self.search.get_top_policies(n=top_n)
        if top_policies is None:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 准备数据
        labels = [f"Policy {i+1}\n(N={int(row['N']):,}, f=${row['f']:.0f})" 
                  for i, row in top_policies.iterrows()]
        
        if baseline is not None:
            labels.append("Baseline")
            g1_values = list(top_policies['g1']/1e6) + [baseline.get('g1', 0)/1e6]
            g2_values = list(top_policies['g2']) + [baseline.get('g2', 0)]
            g3_values = list(top_policies['g3']) + [baseline.get('g3', 0)]
        else:
            g1_values = list(top_policies['g1']/1e6)
            g2_values = list(top_policies['g2'])
            g3_values = list(top_policies['g3'])
        
        colors = PlotStyleConfig.get_palette(len(labels))
        
        # g1 对比
        ax1 = axes[0]
        bars1 = ax1.bar(range(len(labels)), g1_values, color=colors)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('g1 (Million $)', fontweight='bold')
        ax1.set_title('Economic Performance', fontweight='bold')
        for bar, val in zip(bars1, g1_values):
            ax1.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        
        # g2 对比
        ax2 = axes[1]
        bars2 = ax2.bar(range(len(labels)), g2_values, color=colors)
        ax2.axhline(y=self.params.g2_threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('g2', fontweight='bold')
        ax2.set_title('Environmental Score', fontweight='bold')
        ax2.legend()
        for bar, val in zip(bars2, g2_values):
            ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        
        # g3 对比
        ax3 = axes[2]
        bars3 = ax3.bar(range(len(labels)), g3_values, color=colors)
        ax3.axhline(y=self.params.g3_threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('g3', fontweight='bold')
        ax3.set_title('Resident Satisfaction', fontweight='bold')
        ax3.legend()
        for bar, val in zip(bars3, g3_values):
            ax3.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        
        plt.tight_layout()
        self.saver.save(fig, 'policy_comparison', formats=['png', 'pdf'])
        plt.show()
        
        return fig
    
    def plot_sensitivity_analysis(self, variable='f', n_points=20, figsize=(12, 5)):
        """
        单变量敏感性分析（政策变量 N, f, t, x）
        
        :param variable: 分析的变量（'N', 'f', 't', 'x'）
        :param n_points: 分析点数
        """
        best = self.search.get_optimal_policy()
        if best is None:
            return
        
        # 设置变量范围
        ranges = {
            'N': self.params.N_range,
            'f': self.params.f_range,
            't': self.params.t_range,
            'x': self.params.x_range
        }
        
        var_range = ranges[variable]
        var_values = np.linspace(var_range[0], var_range[1], n_points)
        
        g1_list, g2_list, g3_list = [], [], []
        
        for val in var_values:
            # 基于最优解变动单一变量
            N = val if variable == 'N' else best['N']
            f = val if variable == 'f' else best['f']
            t = val if variable == 't' else best['t']
            x = val if variable == 'x' else best['x']
            
            result = self.model.evaluate_policy(N, f, t, x)
            g2 = self.model.environmental_score(result['E'])
            g3 = self.model.resident_score(result['S'])
            
            g1_list.append(result['Pi'])
            g2_list.append(g2)
            g3_list.append(g3)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        var_labels = {'N': 'Visitor Cap N (people/day)', 'f': 'Visitor Fee f ($)', 
                     't': 'Hotel Tax t (%)', 'x': 'Reinvestment Ratio x'}
        
        # g1 敏感性
        ax1 = axes[0]
        ax1.plot(var_values, np.array(g1_list)/1e6, 
                color=PlotStyleConfig.COLORS['primary'], linewidth=2.5, marker='o', markersize=4)
        ax1.axvline(x=best[variable], color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimal')
        ax1.set_xlabel(var_labels[variable], fontweight='bold')
        ax1.set_ylabel('g1 (Million $)', fontweight='bold')
        ax1.set_title(f'Sensitivity of g1 to {variable}', fontweight='bold')
        ax1.legend()
        
        # g2 敏感性
        ax2 = axes[1]
        ax2.plot(var_values, g2_list, 
                color=PlotStyleConfig.COLORS['secondary'], linewidth=2.5, marker='s', markersize=4)
        ax2.axhline(y=self.params.g2_threshold, color='gray', linestyle='--', linewidth=1.5, label='Threshold')
        ax2.axvline(x=best[variable], color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimal')
        ax2.set_xlabel(var_labels[variable], fontweight='bold')
        ax2.set_ylabel('g2', fontweight='bold')
        ax2.set_title(f'Sensitivity of g2 to {variable}', fontweight='bold')
        ax2.legend()
        
        # g3 敏感性
        ax3 = axes[2]
        ax3.plot(var_values, g3_list, 
                color=PlotStyleConfig.COLORS['accent'], linewidth=2.5, marker='^', markersize=4)
        ax3.axhline(y=self.params.g3_threshold, color='gray', linestyle='--', linewidth=1.5, label='Threshold')
        ax3.axvline(x=best[variable], color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimal')
        ax3.set_xlabel(var_labels[variable], fontweight='bold')
        ax3.set_ylabel('g3', fontweight='bold')
        ax3.set_title(f'Sensitivity of g3 to {variable}', fontweight='bold')
        ax3.legend()
        
        plt.tight_layout()
        self.saver.save(fig, f'sensitivity_{variable}', formats=['png', 'pdf'])
        plt.show()
        
        return fig
    
    def plot_parameter_sensitivity(self, param_name='D0', variation=0.2, n_points=20, figsize=(12, 5)):
        """
        模型参数敏感性分析（D0, a, b, c, θ 等）
        
        :param param_name: 参数名称
        :param variation: 变化幅度（如0.2表示±20%）
        :param n_points: 分析点数
        """
        best = self.search.get_optimal_policy()
        if best is None:
            return
        
        # 获取参数基准值
        base_value = self.model._get_param(param_name)
        
        # 参数变化范围 (±variation)
        param_min = base_value * (1 - variation)
        param_max = base_value * (1 + variation)
        param_values = np.linspace(param_min, param_max, n_points)
        
        g1_list, g2_list, g3_list = [], [], []
        
        # 保存原始参数
        original_value = getattr(self.params, param_name)
        
        for val in param_values:
            # 临时修改参数
            setattr(self.params, param_name, val)
            
            # 用最优政策评估（不需要重新计算边界，直接用原始值）
            result = self.model.evaluate_policy(best['N'], best['f'], best['t'], best['x'])
            
            # 计算得分（使用原始边界，保持可比性）
            g1_list.append(result['Pi'])
            g2_list.append(result['E'])  # 原始环境压力
            g3_list.append(result['S'])  # 原始满意度
        
        # 恢复原始参数
        setattr(self.params, param_name, original_value)
        
        # 归一化 g2, g3 用于显示
        g2_arr = np.array(g2_list)
        g3_arr = np.array(g3_list)
        g2_norm = 1 - (g2_arr - g2_arr.min()) / (g2_arr.max() - g2_arr.min() + 1e-10)
        g3_norm = (g3_arr - g3_arr.min()) / (g3_arr.max() - g3_arr.min() + 1e-10)
        
        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        param_labels = {
            'D0': 'Base Demand D0', 'a': 'Fee Sensitivity a', 'b': 'Tax Sensitivity b',
            'c': 'Per-capita Spending c ($)', 'theta': 'Taxable Ratio θ',
            'a0': 'Marginal Cost a0', 'a1': 'Congestion Cost a1',
            'alpha': 'Env. Pressure α', 'beta': 'Treatment Effect β',
            'cap': 'Capacity Threshold cap', 'S0': 'Base Satisfaction S0',
            'gamma': 'Overload Penalty γ', 'delta': 'Investment Effect δ'
        }
        
        x_label = param_labels.get(param_name, param_name)
        
        # g1
        ax1 = axes[0]
        ax1.plot(param_values, np.array(g1_list)/1e6, 
                color=PlotStyleConfig.COLORS['primary'], linewidth=2.5, marker='o', markersize=4)
        ax1.axvline(x=base_value, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
        ax1.set_xlabel(x_label, fontweight='bold')
        ax1.set_ylabel('g1 (Million $)', fontweight='bold')
        ax1.set_title(f'Sensitivity of g1 to {param_name}', fontweight='bold')
        ax1.legend()
        
        # g2 (原始E值，越低越好)
        ax2 = axes[1]
        ax2.plot(param_values, g2_arr, 
                color=PlotStyleConfig.COLORS['secondary'], linewidth=2.5, marker='s', markersize=4)
        ax2.axvline(x=base_value, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
        ax2.set_xlabel(x_label, fontweight='bold')
        ax2.set_ylabel('Environmental Pressure E', fontweight='bold')
        ax2.set_title(f'Sensitivity of E to {param_name}', fontweight='bold')
        ax2.legend()
        
        # g3 (原始S值，越高越好)
        ax3 = axes[2]
        ax3.plot(param_values, g3_arr, 
                color=PlotStyleConfig.COLORS['accent'], linewidth=2.5, marker='^', markersize=4)
        ax3.axvline(x=base_value, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
        ax3.set_xlabel(x_label, fontweight='bold')
        ax3.set_ylabel('Resident Satisfaction S', fontweight='bold')
        ax3.set_title(f'Sensitivity of S to {param_name}', fontweight='bold')
        ax3.legend()
        
        plt.tight_layout()
        self.saver.save(fig, f'param_sensitivity_{param_name}', formats=['png', 'pdf'])
        plt.show()
        
        return fig


# ============================================================
# 第五部分：敏感性分析模块 (Sensitivity Analysis)
# ============================================================

class MonteCarloAnalysis:
    """
    蒙特卡洛参数不确定性分析
    
    当模型参数定义为范围时，通过多次采样分析结果的稳健性
    """
    
    def __init__(self, base_params: TourismPolicyParams, n_simulations=100):
        """
        :param base_params: 包含范围参数的参数配置
        :param n_simulations: 蒙特卡洛模拟次数
        """
        self.base_params = base_params
        self.n_simulations = n_simulations
        self.results = []
        self.optimal_policies = []
        
    def run(self, policy_N=None, policy_f=None, policy_t=None, policy_x=None, verbose=True):
        """
        运行蒙特卡洛分析
        
        :param policy_N/f/t/x: 固定的政策值（若为None则使用搜索最优）
        :return: 分析结果DataFrame
        """
        if verbose:
            print("\n" + "="*70)
            print("🎲 蒙特卡洛参数不确定性分析 (Monte Carlo Uncertainty Analysis)")
            print("="*70)
            uncertain = self.base_params.get_uncertain_params()
            print(f"  不确定参数: {list(uncertain.keys())}")
            print(f"  模拟次数: {self.n_simulations}")
        
        self.results = []
        self.optimal_policies = []
        
        for i in range(self.n_simulations):
            if verbose and (i + 1) % 20 == 0:
                print(f"    进度: {i+1}/{self.n_simulations}")
            
            # 采样一组参数
            sampled_params = self.base_params.sample_params()
            
            # 创建模型
            model = TourismPolicyModel(sampled_params)
            
            # 网格搜索
            search = PolicyGridSearch(model)
            search.run_search(verbose=False)
            search.filter_feasible(verbose=False)
            
            # 获取最优政策
            best = search.get_optimal_policy()
            
            if best is not None:
                self.optimal_policies.append(best)
                self.results.append({
                    'simulation': i + 1,
                    'opt_N': best['N'],
                    'opt_f': best['f'],
                    'opt_t': best['t'],
                    'opt_x': best['x'],
                    'opt_g1': best['g1'],
                    'opt_g2': best['g2'],
                    'opt_g3': best['g3'],
                    'n_feasible': len(search.feasible_df)
                })
        
        self.results_df = pd.DataFrame(self.results)
        
        if verbose:
            self._print_summary()
        
        return self.results_df
    
    def _print_summary(self):
        """打印蒙特卡洛分析摘要"""
        if len(self.results_df) == 0:
            print("  ⚠️ 无有效结果!")
            return
        
        print(f"\n  ✅ 完成 {len(self.results_df)} 次有效模拟")
        print("\n【最优政策统计（均值±标准差）】")
        print(f"  N* = {self.results_df['opt_N'].mean():,.0f} ± {self.results_df['opt_N'].std():,.0f}")
        print(f"  f* = ${self.results_df['opt_f'].mean():.1f} ± {self.results_df['opt_f'].std():.1f}")
        print(f"  t* = {self.results_df['opt_t'].mean():.1f}% ± {self.results_df['opt_t'].std():.1f}%")
        print(f"  x* = {self.results_df['opt_x'].mean():.2f} ± {self.results_df['opt_x'].std():.2f}")
        print("\n【目标值统计】")
        print(f"  g1* = ${self.results_df['opt_g1'].mean()/1e6:.2f}M ± ${self.results_df['opt_g1'].std()/1e6:.2f}M")
        print(f"  g2* = {self.results_df['opt_g2'].mean():.3f} ± {self.results_df['opt_g2'].std():.3f}")
        print(f"  g3* = {self.results_df['opt_g3'].mean():.3f} ± {self.results_df['opt_g3'].std():.3f}")
        print("="*70)
    
    def plot_results(self, figsize=(14, 10)):
        """
        绘制蒙特卡洛分析结果图
        """
        if len(self.results_df) == 0:
            print("无结果可绘制!")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 第一行：最优政策变量分布
        ax1 = axes[0, 0]
        ax1.hist(self.results_df['opt_N'], bins=20, color=PlotStyleConfig.COLORS['primary'], 
                alpha=0.7, edgecolor='white')
        ax1.axvline(self.results_df['opt_N'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.set_xlabel('Optimal N (people/day)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Distribution of Optimal N', fontweight='bold')
        ax1.legend()
        
        ax2 = axes[0, 1]
        ax2.hist(self.results_df['opt_f'], bins=20, color=PlotStyleConfig.COLORS['secondary'], 
                alpha=0.7, edgecolor='white')
        ax2.axvline(self.results_df['opt_f'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.set_xlabel('Optimal f ($)', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Distribution of Optimal f', fontweight='bold')
        ax2.legend()
        
        ax3 = axes[0, 2]
        ax3.hist(self.results_df['opt_t'], bins=20, color=PlotStyleConfig.COLORS['accent'], 
                alpha=0.7, edgecolor='white')
        ax3.axvline(self.results_df['opt_t'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax3.set_xlabel('Optimal t (%)', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Distribution of Optimal t', fontweight='bold')
        ax3.legend()
        
        # 第二行：目标值分布
        ax4 = axes[1, 0]
        ax4.hist(self.results_df['opt_g1']/1e6, bins=20, color=PlotStyleConfig.COLORS['primary'], 
                alpha=0.7, edgecolor='white')
        ax4.axvline(self.results_df['opt_g1'].mean()/1e6, color='red', linestyle='--', linewidth=2, label='Mean')
        ax4.set_xlabel('Optimal g1 (Million $)', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Distribution of Optimal Economic Performance', fontweight='bold')
        ax4.legend()
        
        ax5 = axes[1, 1]
        ax5.hist(self.results_df['opt_g2'], bins=20, color=PlotStyleConfig.COLORS['secondary'], 
                alpha=0.7, edgecolor='white')
        ax5.axvline(self.results_df['opt_g2'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax5.set_xlabel('Optimal g2', fontweight='bold')
        ax5.set_ylabel('Frequency', fontweight='bold')
        ax5.set_title('Distribution of Environmental Score', fontweight='bold')
        ax5.legend()
        
        ax6 = axes[1, 2]
        ax6.hist(self.results_df['opt_g3'], bins=20, color=PlotStyleConfig.COLORS['accent'], 
                alpha=0.7, edgecolor='white')
        ax6.axvline(self.results_df['opt_g3'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax6.set_xlabel('Optimal g3', fontweight='bold')
        ax6.set_ylabel('Frequency', fontweight='bold')
        ax6.set_title('Distribution of Resident Satisfaction', fontweight='bold')
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig('./figures/monte_carlo_results.png', dpi=300, bbox_inches='tight')
        plt.savefig('./figures/monte_carlo_results.pdf', bbox_inches='tight')
        print("  📊 图表已保存: ./figures/monte_carlo_results.png/pdf")
        plt.show()
        
        return fig
    
    def get_robust_policy(self):
        """
        获取鲁棒最优政策（基于蒙特卡洛均值）
        """
        if len(self.results_df) == 0:
            return None
        
        return {
            'N': self.results_df['opt_N'].mean(),
            'f': self.results_df['opt_f'].mean(),
            't': self.results_df['opt_t'].mean(),
            'x': self.results_df['opt_x'].mean(),
            'g1_mean': self.results_df['opt_g1'].mean(),
            'g1_std': self.results_df['opt_g1'].std(),
            'g2_mean': self.results_df['opt_g2'].mean(),
            'g3_mean': self.results_df['opt_g3'].mean()
        }


class ThresholdSensitivityAnalysis:
    """
    阈值敏感性分析
    
    分析约束阈值变化对可行解的影响
    """
    
    def __init__(self, search: PolicyGridSearch):
        self.search = search
        self.results_df = search.results_df
        
    def analyze_threshold_impact(self, g2_thresholds=None, g3_thresholds=None):
        """
        分析不同阈值下的可行解数量和最优解变化
        """
        if g2_thresholds is None:
            g2_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        if g3_thresholds is None:
            g3_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        
        results = []
        
        for g2_bar in g2_thresholds:
            for g3_bar in g3_thresholds:
                mask = (self.results_df['g2'] >= g2_bar) & (self.results_df['g3'] >= g3_bar)
                feasible = self.results_df[mask]
                
                if len(feasible) > 0:
                    best_g1 = feasible['g1'].max()
                    best_idx = feasible['g1'].idxmax()
                    best_policy = feasible.loc[best_idx]
                else:
                    best_g1 = np.nan
                    best_policy = None
                
                results.append({
                    'g2_threshold': g2_bar,
                    'g3_threshold': g3_bar,
                    'n_feasible': len(feasible),
                    'best_g1': best_g1
                })
        
        return pd.DataFrame(results)
    
    def plot_threshold_heatmap(self, figsize=(10, 8)):
        """
        绘制阈值对可行解数量的影响热力图
        """
        analysis = self.analyze_threshold_impact()
        
        pivot = analysis.pivot(index='g3_threshold', columns='g2_threshold', values='n_feasible')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(pivot.values, cmap='YlOrRd_r', aspect='auto')
        
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f'{x:.2f}' for x in pivot.columns])
        ax.set_yticklabels([f'{y:.2f}' for y in pivot.index])
        
        # 添加数值标注
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                text = ax.text(j, i, f'{int(val):,}', ha='center', va='center', 
                              color='white' if val < pivot.values.max()/2 else 'black', fontsize=10)
        
        ax.set_xlabel('Environmental Threshold (g2_bar)', fontweight='bold')
        ax.set_ylabel('Resident Threshold (g3_bar)', fontweight='bold')
        ax.set_title('Number of Feasible Solutions by Threshold Settings', fontweight='bold', pad=15)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Feasible Solutions Count')
        
        plt.tight_layout()
        plt.savefig('./figures/threshold_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.savefig('./figures/threshold_sensitivity.pdf', bbox_inches='tight')
        print("  📊 图表已保存: ./figures/threshold_sensitivity.png/pdf")
        plt.show()
        
        return fig, analysis


# ============================================================
# 第六部分：主工作流 (Main Workflow)
# ============================================================

def run_complete_workflow():
    """
    运行完整的旅游政策优化工作流
    
    包括：参数配置 → 网格搜索 → 可行解筛选 → 可视化 → 敏感性分析
    """
    print("\n" + "█"*70)
    print("█" + " "*25 + "旅游政策优化模型" + " "*25 + "█")
    print("█" + " "*20 + "Tourism Policy Optimization" + " "*21 + "█")
    print("█"*70 + "\n")
    
    # ========== Step 1: 参数配置 ==========
    print("【Step 1】初始化模型参数...")
    params = TourismPolicyParams()
    
    # ★★★ 在这里修改你的参数 ★★★
    # params.D0 = 25000        # 调整潜在需求
    # params.a = 100           # 调整游客费敏感度
    # params.b = 400           # 调整税率敏感度
    # params.theta = 0.25      # 若游轮一日游，降低住宿占比
    # params.g2_threshold = 0.7  # 提高环境要求
    
    params.summary()
    
    # ========== Step 2: 创建模型 ==========
    print("【Step 2】创建优化模型...")
    model = TourismPolicyModel(params)
    
    # ========== Step 3: 网格搜索 ==========
    print("【Step 3】执行政策网格搜索...")
    search = PolicyGridSearch(model)
    search.run_search(verbose=True)
    
    # ========== Step 4: 筛选可行解 ==========
    print("【Step 4】筛选可行解...")
    search.filter_feasible(verbose=True)
    search.summary()
    
    # ========== Step 5: 输出Top政策 ==========
    print("\n【Step 5】Top 5 最优政策:")
    print("-"*70)
    top5 = search.get_top_policies(n=5)
    if top5 is not None:
        display_cols = ['N', 'f', 't', 'x', 'V', 'R', 'I', 'g1', 'g2', 'g3']
        print(top5[display_cols].to_string(index=False))
    
    # ========== Step 6: 可视化 ==========
    print("\n【Step 6】生成可视化图表...")
    print("-"*70)
    
    # 创建figures目录
    os.makedirs('./figures', exist_ok=True)
    
    viz = PolicyVisualization(search, save_dir='./figures')
    
    # 图1: 可行域
    print("\n  🎨 绘制可行域散点图...")
    viz.plot_feasible_region()
    
    # 图2: Pareto前沿
    print("\n  🎨 绘制Pareto前沿...")
    viz.plot_pareto_frontier()
    
    # 图3: 政策对比
    print("\n  🎨 绘制政策对比条形图...")
    # 定义基准政策（无干预）
    baseline = model.evaluate_policy(N=20000, f=0, t=0, x=0)
    baseline['g1'] = baseline['Pi']
    baseline['g2'] = model.environmental_score(baseline['E'])
    baseline['g3'] = model.resident_score(baseline['S'])
    viz.plot_policy_comparison(top_n=5, baseline=baseline)
    
    # 图4-7: 政策变量敏感性分析
    print("\n  🎨 绘制政策变量敏感性分析图...")
    for var in ['N', 'f', 't', 'x']:
        viz.plot_sensitivity_analysis(variable=var)
    
    # 图8-12: 模型参数敏感性分析（鲁棒性检验）
    print("\n  🔬 绘制模型参数敏感性分析图（鲁棒性检验）...")
    key_params = ['D0', 'a', 'c', 'alpha', 'gamma']  # 可修改为你关心的参数
    for param in key_params:
        viz.plot_parameter_sensitivity(param_name=param, variation=0.2)
    
    # ========== Step 7: 阈值敏感性分析 ==========
    print("\n【Step 7】阈值敏感性分析...")
    print("-"*70)
    threshold_analysis = ThresholdSensitivityAnalysis(search)
    fig, threshold_df = threshold_analysis.plot_threshold_heatmap()
    
    # ========== Step 8: 保存结果 ==========
    print("\n【Step 8】保存结果...")
    print("-"*70)
    
    # 保存所有结果
    search.results_df.to_csv('./figures/all_policies.csv', index=False)
    print("  📁 全部政策结果已保存: ./figures/all_policies.csv")
    
    if search.feasible_df is not None and len(search.feasible_df) > 0:
        search.feasible_df.to_csv('./figures/feasible_policies.csv', index=False)
        print("  📁 可行解已保存: ./figures/feasible_policies.csv")
    
    threshold_df.to_csv('./figures/threshold_sensitivity.csv', index=False)
    print("  📁 阈值敏感性分析已保存: ./figures/threshold_sensitivity.csv")
    
    print("\n" + "█"*70)
    print("█" + " "*25 + "工作流执行完成!" + " "*26 + "█")
    print("█"*70 + "\n")
    
    return params, model, search, viz


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":
    
    # ============================================================
    # ★★★ 使用示例1: 固定参数运行（默认） ★★★
    # ============================================================
    params, model, search, viz = run_complete_workflow()
    
    
    # ============================================================
    # ★★★ 使用示例2: 定义参数范围进行蒙特卡洛分析 ★★★
    # ============================================================
    # 如果你希望参数是一个范围，取消下面的注释：
    
    print("\n" + "="*70)
    print("🎲 开始蒙特卡洛参数不确定性分析")
    print("="*70)
    
    # 创建带范围的参数
    params_mc = TourismPolicyParams()
    
    # 设置参数范围（可以只设置你认为不确定的参数）
    params_mc.D0 = ParameterRange(18000, 22000)       # D0 在 18000-22000 之间
    params_mc.a = ParameterRange(100, 140)            # a 在 100-140 之间
    params_mc.b = ParameterRange(300, 400)            # b 在 300-400 之间
    params_mc.c = ParameterRange(200, 300)            # 人均消费在 200-300 之间
    params_mc.theta = ParameterRange(0.25, 0.45)      # θ 在 0.25-0.45 之间
    params_mc.alpha = ParameterRange(0.8, 1.2)
    # 查看参数配置
    params_mc.summary()
    
    # 运行蒙特卡洛分析（100次模拟）
    mc_analysis = MonteCarloAnalysis(params_mc, n_simulations=10000)
    mc_results = mc_analysis.run(verbose=True)
    
    # 绘制蒙特卡洛结果
    mc_analysis.plot_results()
    
    # 获取鲁棒最优政策
    robust_policy = mc_analysis.get_robust_policy()
    print("\n【鲁棒最优政策】（参数不确定性下的推荐）")
    print(f"  N* = {robust_policy['N']:,.0f} 人/日")
    print(f"  f* = ${robust_policy['f']:.1f}")
    print(f"  t* = {robust_policy['t']:.1f}%")
    print(f"  x* = {robust_policy['x']:.2f}")
    print(f"  g1* = ${robust_policy['g1_mean']/1e6:.2f}M ± ${robust_policy['g1_std']/1e6:.2f}M")
    
    # 保存蒙特卡洛结果
    mc_results.to_csv('./figures/monte_carlo_results.csv', index=False)
    print("  📁 蒙特卡洛结果已保存: ./figures/monte_carlo_results.csv")


    # ============================================================
    # ★★★ 其他自定义分析 ★★★
    # ============================================================
    # 
    # 1. 查看特定政策的详细评估结果
    # result = model.evaluate_policy(N=12000, f=25, t=8, x=0.5)
    # print(result)
    # 
    # 2. 获取最优解
    # best = search.get_optimal_policy()
    # print(f"最优政策: N={best['N']}, f={best['f']}, t={best['t']}, x={best['x']}")
    # 
    # 3. 调整阈值重新筛选
    # params.g2_threshold = 0.7
    # params.g3_threshold = 0.7
    # search.filter_feasible()
