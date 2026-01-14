"""
============================================================
微分方程与系统动力学模型 (Differential Equations & System Dynamics)
包含：ODE求解 + 流行病学模型 + 人口动力学 + 系统动力学
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：常微分方程数值求解、SIR/SEIR模型、Logistic增长、SD建模
特点：完整的参数设置、可视化与美化、参数敏感性分析
作者：MCM/ICM Team
日期：2026年1月
============================================================

使用场景：
- 传染病传播预测 (SIR/SEIR)
- 人口增长预测 (Logistic/Malthus)
- 捕食者-被捕食者模型 (Lotka-Volterra)
- 系统动力学分析 (Stock-Flow)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')


# ============================================================
# 第一部分：全局配置与美化设置 (Global Configuration)
# ============================================================

class PlotStyleConfig:
    """图表美化配置类"""
    
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'success': '#27AE60',
        'danger': '#C73E1D',
        'neutral': '#3B3B3B'
    }
    
    PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4C9A', '#1B998B']
    
    # 传染病模型专用颜色
    EPIDEMIC_COLORS = {
        'S': '#2E86AB',  # 易感者 - 蓝色
        'E': '#F18F01',  # 暴露者 - 橙色
        'I': '#C73E1D',  # 感染者 - 红色
        'R': '#27AE60',  # 康复者 - 绿色
        'D': '#3B3B3B'   # 死亡者 - 黑色
    }
    
    @staticmethod
    def setup_style():
        plt.style.use('seaborn-v0_8-whitegrid')
        rcParams['figure.figsize'] = (12, 8)
        rcParams['figure.dpi'] = 100
        rcParams['savefig.dpi'] = 300
        rcParams['font.size'] = 11
        rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False

PlotStyleConfig.setup_style()


# ============================================================
# 第二部分：基础ODE求解器 (ODE Solver Base)
# ============================================================

class ODESolver:
    """
    常微分方程求解器基类
    
    支持的求解方法：
    - odeint: scipy传统方法，Fortran LSODA
    - solve_ivp: 新接口，支持多种方法
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.solution = None
        self.t = None
        self.history = {}
    
    def solve(self, func, y0, t_span, method='RK45', t_eval=None, **kwargs):
        """
        求解ODE
        
        :param func: 微分方程函数 dy/dt = f(t, y)
        :param y0: 初始条件
        :param t_span: 时间区间 [t0, tf]
        :param method: 求解方法
            - 'RK45': Runge-Kutta 4(5) (默认)
            - 'RK23': Runge-Kutta 2(3)
            - 'DOP853': Dormand-Prince 8阶
            - 'Radau': Radau IIA 5阶 (刚性问题)
            - 'BDF': BDF (刚性问题)
            - 'LSODA': 自动刚性检测
        :param t_eval: 输出时间点
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 500)
        
        sol = solve_ivp(func, t_span, y0, method=method, 
                       t_eval=t_eval, **kwargs)
        
        self.t = sol.t
        self.solution = sol.y
        
        return self.t, self.solution


# ============================================================
# 第三部分：人口动力学模型 (Population Dynamics)
# ============================================================

class PopulationDynamics:
    """
    人口动力学模型
    
    模型：
    1. Malthus指数增长
    2. Logistic增长
    3. 带延迟的Logistic增长
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.params = {}
        self.solution = None
        self.t = None
    
    def malthus_growth(self, P0, r, t):
        """
        马尔萨斯指数增长模型
        
        dP/dt = r * P
        P(t) = P0 * exp(r*t)
        
        :param P0: 初始人口
        :param r: 增长率
        :param t: 时间点数组
        """
        self.params = {'model': 'Malthus', 'P0': P0, 'r': r}
        self.t = np.array(t)
        self.solution = P0 * np.exp(r * self.t)
        
        if self.verbose:
            print(f"\n马尔萨斯模型: P(t) = {P0} × e^({r}t)")
        
        return self.t, self.solution
    
    def logistic_growth(self, P0, r, K, t_span, n_points=500):
        """
        Logistic增长模型
        
        dP/dt = r * P * (1 - P/K)
        
        :param P0: 初始人口
        :param r: 内禀增长率
        :param K: 环境容纳量（最大人口）
        :param t_span: 时间范围 [t0, tf]
        """
        self.params = {'model': 'Logistic', 'P0': P0, 'r': r, 'K': K}
        
        def logistic_ode(t, P):
            return r * P * (1 - P / K)
        
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(logistic_ode, t_span, [P0], t_eval=t_eval)
        
        self.t = sol.t
        self.solution = sol.y[0]
        
        if self.verbose:
            print(f"\nLogistic模型: dP/dt = {r}P(1 - P/{K})")
            print(f"  环境容纳量 K = {K}")
            print(f"  最终人口 P(∞) → {K}")
        
        return self.t, self.solution
    
    def fit_logistic(self, t_data, P_data, P0_guess=None, r_guess=None, K_guess=None):
        """
        从数据拟合Logistic参数
        
        :param t_data: 时间数据
        :param P_data: 人口数据
        """
        def logistic_func(t, r, K):
            P0 = P_data[0]
            return K / (1 + (K/P0 - 1) * np.exp(-r * t))
        
        if r_guess is None:
            r_guess = 0.1
        if K_guess is None:
            K_guess = max(P_data) * 2
        
        try:
            popt, pcov = curve_fit(logistic_func, t_data, P_data, 
                                   p0=[r_guess, K_guess], maxfev=5000)
            r_fit, K_fit = popt
            
            if self.verbose:
                print(f"\nLogistic拟合结果:")
                print(f"  增长率 r = {r_fit:.4f}")
                print(f"  容纳量 K = {K_fit:.2f}")
            
            return {'r': r_fit, 'K': K_fit, 'P0': P_data[0]}
        except:
            print("拟合失败，请检查数据或初始猜测值")
            return None


class LotkaVolterra:
    """
    Lotka-Volterra捕食者-被捕食者模型
    
    dx/dt = αx - βxy  (被捕食者)
    dy/dt = δxy - γy  (捕食者)
    
    参数说明：
    - α: 被捕食者自然增长率
    - β: 捕食效率
    - γ: 捕食者自然死亡率
    - δ: 捕食转化效率
    """
    
    def __init__(self, alpha=1.0, beta=0.1, gamma=1.5, delta=0.075, verbose=True):
        """
        参数配置
        
        :param alpha: 被捕食者增长率 (兔子繁殖率)
        :param beta: 捕食效率 (狼吃兔子的效率)
        :param gamma: 捕食者死亡率 (狼的死亡率)
        :param delta: 转化效率 (吃兔子转化为狼的效率)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.verbose = verbose
        
        self.t = None
        self.prey = None  # 被捕食者
        self.predator = None  # 捕食者
    
    def _system(self, t, y):
        """ODE系统"""
        x, y_pred = y
        dxdt = self.alpha * x - self.beta * x * y_pred
        dydt = self.delta * x * y_pred - self.gamma * y_pred
        return [dxdt, dydt]
    
    def solve(self, x0, y0, t_span, n_points=1000):
        """
        求解方程组
        
        :param x0: 被捕食者初始数量
        :param y0: 捕食者初始数量
        :param t_span: 时间范围
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        sol = solve_ivp(self._system, t_span, [x0, y0], 
                       t_eval=t_eval, method='RK45')
        
        self.t = sol.t
        self.prey = sol.y[0]
        self.predator = sol.y[1]
        
        if self.verbose:
            self._print_results()
        
        return self.t, self.prey, self.predator
    
    def _print_results(self):
        """打印结果"""
        print("\n" + "="*60)
        print("🐺🐰 Lotka-Volterra 捕食模型")
        print("="*60)
        print(f"  参数:")
        print(f"    α (prey growth) = {self.alpha}")
        print(f"    β (predation) = {self.beta}")
        print(f"    γ (predator death) = {self.gamma}")
        print(f"    δ (conversion) = {self.delta}")
        print(f"\n  结果:")
        print(f"    被捕食者: min={self.prey.min():.2f}, max={self.prey.max():.2f}")
        print(f"    捕食者: min={self.predator.min():.2f}, max={self.predator.max():.2f}")
        print("="*60)


# ============================================================
# 第四部分：流行病学模型 (Epidemiological Models)
# ============================================================

class SIRModel:
    """
    SIR传染病模型
    
    S → I → R
    易感者 → 感染者 → 康复者
    
    dS/dt = -β*S*I/N
    dI/dt = β*S*I/N - γ*I
    dR/dt = γ*I
    
    关键指标：
    - R0 = β/γ: 基本再生数
    - 若R0 > 1，疫情爆发
    - 若R0 < 1，疫情消退
    """
    
    def __init__(self, beta=0.3, gamma=0.1, verbose=True):
        """
        参数配置
        
        :param beta: 传染率
            - 表示每个感染者每天能有效接触的人数
            - 建议：0.1-0.5
            
        :param gamma: 康复率
            - 1/gamma = 平均感染周期（天）
            - 建议：0.05-0.2
        """
        self.beta = beta
        self.gamma = gamma
        self.verbose = verbose
        
        self.R0 = beta / gamma
        self.t = None
        self.S = None
        self.I = None
        self.R = None
        self.history = {}
    
    def _sir_system(self, t, y, N):
        """SIR方程组"""
        S, I, R = y
        dSdt = -self.beta * S * I / N
        dIdt = self.beta * S * I / N - self.gamma * I
        dRdt = self.gamma * I
        return [dSdt, dIdt, dRdt]
    
    def simulate(self, N, I0, R0=0, t_span=(0, 160), n_points=500):
        """
        运行模拟
        
        :param N: 总人口
        :param I0: 初始感染者
        :param R0: 初始康复者
        :param t_span: 模拟时间范围（天）
        """
        S0 = N - I0 - R0
        y0 = [S0, I0, R0]
        
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        sol = solve_ivp(lambda t, y: self._sir_system(t, y, N),
                       t_span, y0, t_eval=t_eval, method='RK45')
        
        self.t = sol.t
        self.S = sol.y[0]
        self.I = sol.y[1]
        self.R = sol.y[2]
        
        # 计算关键指标
        self.peak_infection = self.I.max()
        self.peak_day = self.t[np.argmax(self.I)]
        self.total_infected = N - self.S[-1]
        
        self.history = {
            't': self.t,
            'S': self.S,
            'I': self.I,
            'R': self.R
        }
        
        if self.verbose:
            self._print_results(N)
        
        return self.t, self.S, self.I, self.R
    
    def _print_results(self, N):
        """打印结果"""
        print("\n" + "="*60)
        print("🦠 SIR传染病模型模拟结果")
        print("="*60)
        print(f"  模型参数:")
        print(f"    传染率 β = {self.beta}")
        print(f"    康复率 γ = {self.gamma}")
        print(f"    基本再生数 R0 = {self.R0:.2f}")
        print(f"\n  模拟结果:")
        print(f"    总人口 N = {N}")
        print(f"    感染峰值: {self.peak_infection:.0f} 人")
        print(f"    峰值时间: 第 {self.peak_day:.1f} 天")
        print(f"    最终感染: {self.total_infected:.0f} 人 ({self.total_infected/N*100:.1f}%)")
        print(f"    最终易感: {self.S[-1]:.0f} 人")
        print("="*60)


class SEIRModel:
    """
    SEIR传染病模型
    
    S → E → I → R
    易感者 → 暴露者 → 感染者 → 康复者
    
    新增：潜伏期 E (Exposed)
    
    dS/dt = -β*S*I/N
    dE/dt = β*S*I/N - σ*E
    dI/dt = σ*E - γ*I
    dR/dt = γ*I
    
    参数：
    - σ: 潜伏期转化率 (1/σ = 平均潜伏期)
    """
    
    def __init__(self, beta=0.5, sigma=0.2, gamma=0.1, verbose=True):
        """
        参数配置
        
        :param beta: 传染率
        :param sigma: 潜伏期转化率
            - 1/sigma = 平均潜伏期（天）
            - COVID-19: σ ≈ 1/5.2
        :param gamma: 康复率
        """
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.verbose = verbose
        
        self.R0 = beta / gamma
        self.t = None
        self.S = None
        self.E = None
        self.I = None
        self.R = None
    
    def _seir_system(self, t, y, N):
        """SEIR方程组"""
        S, E, I, R = y
        dSdt = -self.beta * S * I / N
        dEdt = self.beta * S * I / N - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    def simulate(self, N, I0, E0=0, R0=0, t_span=(0, 160), n_points=500):
        """运行模拟"""
        S0 = N - I0 - E0 - R0
        y0 = [S0, E0, I0, R0]
        
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        sol = solve_ivp(lambda t, y: self._seir_system(t, y, N),
                       t_span, y0, t_eval=t_eval, method='RK45')
        
        self.t = sol.t
        self.S = sol.y[0]
        self.E = sol.y[1]
        self.I = sol.y[2]
        self.R = sol.y[3]
        
        if self.verbose:
            self._print_results(N)
        
        return self.t, self.S, self.E, self.I, self.R
    
    def _print_results(self, N):
        """打印结果"""
        print("\n" + "="*60)
        print("🦠 SEIR传染病模型模拟结果")
        print("="*60)
        print(f"  模型参数:")
        print(f"    传染率 β = {self.beta}")
        print(f"    潜伏期 1/σ = {1/self.sigma:.1f} 天")
        print(f"    康复率 γ = {self.gamma}")
        print(f"    基本再生数 R0 = {self.R0:.2f}")
        print(f"\n  结果:")
        print(f"    感染峰值: {self.I.max():.0f} 人")
        print(f"    峰值时间: 第 {self.t[np.argmax(self.I)]:.1f} 天")
        print(f"    暴露峰值: {self.E.max():.0f} 人")
        print("="*60)


# ============================================================
# 第五部分：系统动力学 (System Dynamics)
# ============================================================

class SystemDynamics:
    """
    系统动力学建模基类
    
    核心概念：
    - Stock (存量): 系统中累积的量
    - Flow (流量): 改变存量的速率
    - Feedback (反馈): 正反馈/负反馈
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.stocks = {}
        self.flows = {}
        self.auxiliaries = {}
        self.t = None
        self.solution = None
    
    def add_stock(self, name, initial_value):
        """添加存量"""
        self.stocks[name] = initial_value
    
    def add_flow(self, name, func):
        """添加流量"""
        self.flows[name] = func
    
    def build_water_reservoir_model(self, initial_level=50, 
                                     inflow_rate=10, 
                                     outflow_rate_factor=0.1,
                                     rainfall_rate=2):
        """
        水库蓄水模型示例
        
        存量: 水位
        流入: 降雨 + 河流来水
        流出: 放水（与水位相关）
        """
        self.stocks['水位'] = initial_level
        
        def system_ode(t, y):
            level = y[0]
            inflow = inflow_rate + rainfall_rate * np.sin(2 * np.pi * t / 365)  # 季节性降雨
            outflow = outflow_rate_factor * level  # 水位越高放水越多
            return [inflow - outflow]
        
        return system_ode
    
    def build_population_economy_model(self, P0=1000, K0=10000,
                                         birth_rate=0.02,
                                         death_rate=0.01,
                                         growth_rate=0.03):
        """
        人口-经济耦合模型
        
        存量: 人口P, 资本K
        反馈: 人口增长→劳动力→经济增长→生活水平→出生率
        """
        def system_ode(t, y):
            P, K = y
            # 人口动力学
            birth = birth_rate * P * (1 + K / (K + 10000))  # 经济越好出生率越高
            death = death_rate * P * (1 + 0.5 * P / 50000)  # 人口过多死亡率上升
            dPdt = birth - death
            
            # 经济动力学
            production = growth_rate * K * np.sqrt(P / 1000)  # 生产函数
            depreciation = 0.05 * K  # 资本折旧
            dKdt = production - depreciation
            
            return [dPdt, dKdt]
        
        return system_ode, [P0, K0]
    
    def solve(self, ode_func, y0, t_span, n_points=500):
        """求解系统"""
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        sol = solve_ivp(ode_func, t_span, y0, t_eval=t_eval, method='RK45')
        
        self.t = sol.t
        self.solution = sol.y
        
        return self.t, self.solution


# ============================================================
# 第六部分：参数敏感性分析 (Sensitivity Analysis)
# ============================================================

class SensitivityAnalyzer:
    """敏感性分析模块"""
    
    def __init__(self):
        self.results = {}
    
    def sir_sensitivity(self, N, I0, base_beta=0.3, base_gamma=0.1,
                       beta_range=(0.1, 0.5), gamma_range=(0.05, 0.2),
                       n_values=5, t_span=(0, 160)):
        """
        SIR模型参数敏感性分析
        """
        betas = np.linspace(*beta_range, n_values)
        gammas = np.linspace(*gamma_range, n_values)
        
        self.results['beta_sensitivity'] = []
        self.results['gamma_sensitivity'] = []
        
        # β敏感性
        for beta in betas:
            model = SIRModel(beta=beta, gamma=base_gamma, verbose=False)
            t, S, I, R = model.simulate(N, I0, t_span=t_span)
            self.results['beta_sensitivity'].append({
                'beta': beta,
                'R0': beta / base_gamma,
                'peak_infection': I.max(),
                'peak_day': t[np.argmax(I)],
                'total_infected': N - S[-1],
                't': t, 'I': I
            })
        
        # γ敏感性
        for gamma in gammas:
            model = SIRModel(beta=base_beta, gamma=gamma, verbose=False)
            t, S, I, R = model.simulate(N, I0, t_span=t_span)
            self.results['gamma_sensitivity'].append({
                'gamma': gamma,
                'R0': base_beta / gamma,
                'peak_infection': I.max(),
                'peak_day': t[np.argmax(I)],
                'total_infected': N - S[-1],
                't': t, 'I': I
            })
        
        return self.results


# ============================================================
# 第七部分：可视化模块 (Visualization)
# ============================================================

class ODEVisualizer:
    """微分方程可视化类"""
    
    def __init__(self):
        self.colors = PlotStyleConfig.PALETTE
        self.epidemic_colors = PlotStyleConfig.EPIDEMIC_COLORS
    
    def plot_sir(self, t, S, I, R, title="SIR模型模拟", save_path=None):
        """绘制SIR曲线"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(t, S, color=self.epidemic_colors['S'], 
               linewidth=2.5, label='S (易感者)')
        ax.plot(t, I, color=self.epidemic_colors['I'], 
               linewidth=2.5, label='I (感染者)')
        ax.plot(t, R, color=self.epidemic_colors['R'], 
               linewidth=2.5, label='R (康复者)')
        
        ax.fill_between(t, I, alpha=0.3, color=self.epidemic_colors['I'])
        
        # 标注峰值
        peak_idx = np.argmax(I)
        ax.annotate(f'峰值: {I[peak_idx]:.0f}\n第{t[peak_idx]:.0f}天',
                   xy=(t[peak_idx], I[peak_idx]),
                   xytext=(t[peak_idx] + 10, I[peak_idx] * 1.1),
                   fontsize=10, ha='left',
                   arrowprops=dict(arrowstyle='->', color='gray'))
        
        ax.set_xlabel('时间 (天)', fontweight='bold')
        ax.set_ylabel('人数', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_seir(self, t, S, E, I, R, title="SEIR模型模拟", save_path=None):
        """绘制SEIR曲线"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(t, S, color=self.epidemic_colors['S'], 
               linewidth=2.5, label='S (易感者)')
        ax.plot(t, E, color=self.epidemic_colors['E'], 
               linewidth=2.5, label='E (暴露者)')
        ax.plot(t, I, color=self.epidemic_colors['I'], 
               linewidth=2.5, label='I (感染者)')
        ax.plot(t, R, color=self.epidemic_colors['R'], 
               linewidth=2.5, label='R (康复者)')
        
        ax.fill_between(t, I, alpha=0.3, color=self.epidemic_colors['I'])
        
        ax.set_xlabel('时间 (天)', fontweight='bold')
        ax.set_ylabel('人数', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predator_prey(self, t, prey, predator, 
                           prey_name="被捕食者", predator_name="捕食者",
                           title="Lotka-Volterra捕食模型", save_path=None):
        """绘制捕食者-被捕食者模型"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 时间序列
        ax1 = axes[0]
        ax1.plot(t, prey, color=self.colors[0], linewidth=2.5, label=prey_name)
        ax1.plot(t, predator, color=self.colors[1], linewidth=2.5, label=predator_name)
        ax1.set_xlabel('时间', fontweight='bold')
        ax1.set_ylabel('种群数量', fontweight='bold')
        ax1.set_title('(a) 种群随时间变化', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 相空间图
        ax2 = axes[1]
        ax2.plot(prey, predator, color=self.colors[2], linewidth=1.5)
        ax2.scatter(prey[0], predator[0], color=self.colors[0], s=100, 
                   zorder=5, label='起点', edgecolor='white')
        ax2.set_xlabel(prey_name, fontweight='bold')
        ax2.set_ylabel(predator_name, fontweight='bold')
        ax2.set_title('(b) 相空间轨迹', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_logistic(self, t, P, K=None, title="Logistic增长模型", save_path=None):
        """绘制Logistic增长曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(t, P, color=self.colors[0], linewidth=2.5, label='人口/种群')
        
        if K is not None:
            ax.axhline(y=K, color=self.colors[1], linestyle='--', 
                      linewidth=2, label=f'容纳量 K={K}')
        
        ax.set_xlabel('时间', fontweight='bold')
        ax.set_ylabel('数量', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sensitivity(self, sensitivity_results, param_name='beta',
                        title="参数敏感性分析", save_path=None):
        """绘制敏感性分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        data = sensitivity_results[f'{param_name}_sensitivity']
        
        # 不同参数下的感染曲线
        ax1 = axes[0, 0]
        for i, d in enumerate(data):
            label = f"{param_name}={d[param_name]:.2f}"
            ax1.plot(d['t'], d['I'], linewidth=2, label=label,
                    color=plt.cm.RdYlBu_r(i / len(data)))
        ax1.set_xlabel('时间 (天)')
        ax1.set_ylabel('感染人数')
        ax1.set_title('(a) 感染曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 感染峰值
        ax2 = axes[0, 1]
        params = [d[param_name] for d in data]
        peaks = [d['peak_infection'] for d in data]
        ax2.bar(range(len(params)), peaks, color=self.colors[:len(params)])
        ax2.set_xticks(range(len(params)))
        ax2.set_xticklabels([f'{p:.2f}' for p in params])
        ax2.set_xlabel(param_name)
        ax2.set_ylabel('感染峰值')
        ax2.set_title('(b) 感染峰值')
        
        # 峰值时间
        ax3 = axes[1, 0]
        peak_days = [d['peak_day'] for d in data]
        ax3.plot(params, peak_days, 'o-', color=self.colors[0], 
                linewidth=2, markersize=10)
        ax3.set_xlabel(param_name)
        ax3.set_ylabel('峰值时间 (天)')
        ax3.set_title('(c) 峰值出现时间')
        ax3.grid(True, alpha=0.3)
        
        # R0与总感染人数
        ax4 = axes[1, 1]
        R0s = [d['R0'] for d in data]
        totals = [d['total_infected'] for d in data]
        ax4.plot(R0s, totals, 's-', color=self.colors[1], 
                linewidth=2, markersize=10)
        ax4.axvline(x=1, color='red', linestyle='--', label='R0=1 临界点')
        ax4.set_xlabel('基本再生数 R0')
        ax4.set_ylabel('最终感染总数')
        ax4.set_title('(d) R0与总感染人数')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 第八部分：主程序与完整示例 (Main Program)
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("   DIFFERENTIAL EQUATIONS & SYSTEM DYNAMICS FOR MCM/ICM")
    print("   微分方程与系统动力学模型")
    print("   Extended Version with Visualization")
    print("="*70)
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    📐 微分方程模型分析                           ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║   [人口动力学]                                                    ║
    ║      ├─ Malthus模型: 指数增长                                    ║
    ║      ├─ Logistic模型: S形增长曲线                               ║
    ║      └─ Lotka-Volterra: 捕食者-被捕食者                          ║
    ║                                                                  ║
    ║   [流行病学]                                                      ║
    ║      ├─ SIR模型: S→I→R                                          ║
    ║      ├─ SEIR模型: 含潜伏期                                       ║
    ║      └─ R0: 基本再生数                                           ║
    ║                                                                  ║
    ║   [系统动力学]                                                    ║
    ║      ├─ Stock (存量)                                             ║
    ║      ├─ Flow (流量)                                              ║
    ║      └─ Feedback (反馈)                                          ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    visualizer = ODEVisualizer()
    
    # ================================================================
    # 示例1：Logistic人口增长
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 1: Logistic人口增长模型")
    print("="*70)
    
    pop = PopulationDynamics(verbose=True)
    t, P = pop.logistic_growth(P0=100, r=0.05, K=10000, t_span=(0, 200))
    
    visualizer.plot_logistic(t, P, K=10000, title="Logistic种群增长模型")
    
    # ================================================================
    # 示例2：Lotka-Volterra捕食模型
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 2: Lotka-Volterra捕食者-被捕食者模型")
    print("="*70)
    
    lv = LotkaVolterra(alpha=1.0, beta=0.1, gamma=1.5, delta=0.075, verbose=True)
    t, prey, predator = lv.solve(x0=40, y0=9, t_span=(0, 30))
    
    visualizer.plot_predator_prey(t, prey, predator, 
                                   prey_name="兔子", predator_name="狼",
                                   title="Lotka-Volterra捕食模型")
    
    # ================================================================
    # 示例3：SIR传染病模型
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 3: SIR传染病传播模型")
    print("="*70)
    
    sir = SIRModel(beta=0.3, gamma=0.1, verbose=True)
    t, S, I, R = sir.simulate(N=10000, I0=10, t_span=(0, 160))
    
    visualizer.plot_sir(t, S, I, R, title="SIR传染病模型 (R0=3.0)")
    
    # ================================================================
    # 示例4：SEIR传染病模型
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 4: SEIR传染病传播模型（含潜伏期）")
    print("="*70)
    
    seir = SEIRModel(beta=0.5, sigma=0.2, gamma=0.1, verbose=True)
    t, S, E, I, R = seir.simulate(N=10000, I0=10, E0=20, t_span=(0, 200))
    
    visualizer.plot_seir(t, S, E, I, R, title="SEIR传染病模型 (5天潜伏期)")
    
    # ================================================================
    # 示例5：参数敏感性分析
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 5: SIR模型参数敏感性分析")
    print("="*70)
    
    analyzer = SensitivityAnalyzer()
    sensitivity = analyzer.sir_sensitivity(
        N=10000, I0=10,
        base_beta=0.3, base_gamma=0.1,
        beta_range=(0.15, 0.45), n_values=5
    )
    
    visualizer.plot_sensitivity(sensitivity, param_name='beta',
                                title="传染率β敏感性分析")
    
    # ================================================================
    # 示例6：系统动力学 - 人口经济模型
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 6: 系统动力学 - 人口经济耦合模型")
    print("="*70)
    
    sd = SystemDynamics(verbose=True)
    ode_func, y0 = sd.build_population_economy_model(P0=1000, K0=10000)
    t, solution = sd.solve(ode_func, y0, t_span=(0, 100))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.plot(t, solution[0], color=PlotStyleConfig.PALETTE[0], linewidth=2.5)
    ax1.set_xlabel('时间', fontweight='bold')
    ax1.set_ylabel('人口', fontweight='bold')
    ax1.set_title('(a) 人口增长', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(t, solution[1], color=PlotStyleConfig.PALETTE[1], linewidth=2.5)
    ax2.set_xlabel('时间', fontweight='bold')
    ax2.set_ylabel('资本', fontweight='bold')
    ax2.set_title('(b) 资本积累', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('人口-经济耦合模型', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # ================================================================
    # 使用说明
    # ================================================================
    print("\n" + "="*70)
    print("📖 使用说明 (Usage Guide)")
    print("="*70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                   微分方程模型使用指南                           │
    └─────────────────────────────────────────────────────────────────┘
    
    【人口动力学】
    
    1️⃣ Logistic增长
       pop = PopulationDynamics()
       t, P = pop.logistic_growth(P0=100, r=0.05, K=10000, t_span=(0,200))
    
    2️⃣ Lotka-Volterra
       lv = LotkaVolterra(alpha=1.0, beta=0.1, gamma=1.5, delta=0.075)
       t, prey, predator = lv.solve(x0=40, y0=9, t_span=(0, 30))
    
    【传染病模型】
    
    3️⃣ SIR模型
       sir = SIRModel(beta=0.3, gamma=0.1)
       t, S, I, R = sir.simulate(N=10000, I0=10)
    
    4️⃣ SEIR模型
       seir = SEIRModel(beta=0.5, sigma=0.2, gamma=0.1)
       t, S, E, I, R = seir.simulate(N=10000, I0=10, E0=20)
    
    【关键参数说明】
    
    SIR/SEIR参数:
    - β (beta): 传染率，每个感染者每天有效接触人数
    - γ (gamma): 康复率，1/γ = 平均感染周期
    - σ (sigma): 潜伏期转化率，1/σ = 平均潜伏期
    - R0 = β/γ: 基本再生数
      * R0 > 1: 疫情爆发
      * R0 < 1: 疫情消退
    
    Lotka-Volterra参数:
    - α: 被捕食者自然增长率
    - β: 捕食效率
    - γ: 捕食者死亡率
    - δ: 捕食转化效率
    
    【论文图表建议】
    
    Figure 1: SIR/SEIR时间序列曲线
    Figure 2: 相空间图（Lotka-Volterra）
    Figure 3: 参数敏感性分析
    Figure 4: 不同R0下的疫情对比
    
    Table 1: 模型参数及其含义
    Table 2: 数值结果（峰值、总感染等）
    """)
    
    print("\n" + "="*70)
    print("   ✅ All examples completed successfully!")
    print("   💡 Use the above code templates for your MCM/ICM paper")
    print("="*70)
