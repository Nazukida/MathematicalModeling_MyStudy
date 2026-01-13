"""
============================================================
系统动力学模型 - SIR传染病模型
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：传染病传播模拟、政策干预效果分析、峰值预测
模型：SIR / SEIR / SIRS 变体
作者：MCM/ICM Team
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.integrate import odeint
from scipy.optimize import minimize

# 图表美化设置
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class SIRModel:
    """
    SIR传染病模型封装类
    
    核心方程：
    - dS/dt = -β*S*I/N    (易感者减少)
    - dI/dt = β*S*I/N - γ*I  (感染者变化)
    - dR/dt = γ*I           (康复者增加)
    
    关键参数：
    - β (beta): 传染率，一个感染者每天有效接触人数
    - γ (gamma): 康复率，1/γ 为平均感染周期
    - R0 = β/γ: 基本再生数，R0>1疫情爆发
    """
    
    def __init__(self, beta=0.3, gamma=0.1, verbose=True):
        """
        参数配置
        
        :param beta: 传染率（0.1-0.5常见）
        :param gamma: 康复率（0.05-0.2常见）
        :param verbose: 是否打印过程
        """
        self.beta = beta
        self.gamma = gamma
        self.R0 = beta / gamma  # 基本再生数
        self.verbose = verbose
        self.solution = None
        self.time = None
        self.N = None  # 总人口
    
    def _sir_ode(self, y, t, beta, gamma):
        """SIR微分方程"""
        S, I, R = y
        N = S + I + R
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]
    
    def simulate(self, S0, I0, R0, t_span):
        """
        模拟疫情传播
        
        :param S0: 初始易感者人数
        :param I0: 初始感染者人数
        :param R0: 初始康复者人数
        :param t_span: (t_start, t_end, n_points)
        """
        self.N = S0 + I0 + R0
        self.time = np.linspace(t_span[0], t_span[1], t_span[2])
        y0 = [S0, I0, R0]
        
        self.solution = odeint(self._sir_ode, y0, self.time, 
                               args=(self.beta, self.gamma))
        
        if self.verbose:
            self._print_results(S0, I0, R0)
        
        return self.solution
    
    def _print_results(self, S0, I0, R0):
        """打印结果"""
        S, I, R = self.solution.T
        peak_idx = np.argmax(I)
        
        print("\n" + "="*50)
        print("🦠 SIR 传染病模型模拟结果")
        print("="*50)
        print(f"\n  模型参数:")
        print(f"    传染率 β = {self.beta}")
        print(f"    康复率 γ = {self.gamma}")
        print(f"    基本再生数 R0 = {self.R0:.2f}")
        print(f"\n  初始条件:")
        print(f"    易感者 S0 = {S0}")
        print(f"    感染者 I0 = {I0}")
        print(f"    康复者 R0 = {R0}")
        print(f"\n  疫情峰值:")
        print(f"    峰值时间: 第 {self.time[peak_idx]:.0f} 天")
        print(f"    峰值感染人数: {I[peak_idx]:.0f}")
        print(f"\n  最终状态:")
        print(f"    最终易感者: {S[-1]:.0f}")
        print(f"    最终康复者: {R[-1]:.0f}")
        print(f"    总感染率: {(R[-1]/self.N)*100:.1f}%")
        print("="*50)
    
    def get_peak_info(self):
        """获取峰值信息"""
        I = self.solution[:, 1]
        peak_idx = np.argmax(I)
        return {
            'peak_time': self.time[peak_idx],
            'peak_infected': I[peak_idx],
            'peak_ratio': I[peak_idx] / self.N
        }
    
    def plot_simulation(self, save_path=None):
        """可视化模拟结果"""
        if self.solution is None:
            raise ValueError("请先调用simulate()模拟")
        
        S, I, R = self.solution.T
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：时间序列
        ax1 = axes[0]
        ax1.plot(self.time, S, color='#2E86AB', linewidth=2.5, label='易感者 S')
        ax1.plot(self.time, I, color='#E94F37', linewidth=2.5, label='感染者 I')
        ax1.plot(self.time, R, color='#A8D5BA', linewidth=2.5, label='康复者 R')
        
        # 标记峰值
        peak_idx = np.argmax(I)
        ax1.scatter(self.time[peak_idx], I[peak_idx], color='#E94F37', 
                   s=100, zorder=5, edgecolor='white', linewidth=2)
        ax1.annotate(f'峰值: {I[peak_idx]:.0f}\n第{self.time[peak_idx]:.0f}天',
                    xy=(self.time[peak_idx], I[peak_idx]),
                    xytext=(self.time[peak_idx]+10, I[peak_idx]+50),
                    fontsize=10, ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray'))
        
        ax1.set_xlabel('时间（天）', fontsize=12, fontweight='bold')
        ax1.set_ylabel('人数', fontsize=12, fontweight='bold')
        ax1.set_title(f'SIR模型传染病模拟 (R0={self.R0:.2f})', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 右图：堆叠面积图
        ax2 = axes[1]
        ax2.stackplot(self.time, S, I, R, 
                     labels=['易感者 S', '感染者 I', '康复者 R'],
                     colors=['#2E86AB', '#E94F37', '#A8D5BA'], alpha=0.8)
        ax2.set_xlabel('时间（天）', fontsize=12, fontweight='bold')
        ax2.set_ylabel('人数', fontsize=12, fontweight='bold')
        ax2.set_title('人群状态分布', fontsize=14, fontweight='bold')
        ax2.legend(loc='right', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def sensitivity_analysis(self, S0, I0, R0, t_span, 
                            beta_range=(0.1, 0.5, 5), 
                            gamma_range=(0.05, 0.2, 5)):
        """参数敏感性分析"""
        betas = np.linspace(*beta_range)
        gammas = np.linspace(*gamma_range)
        
        results = []
        for b in betas:
            for g in gammas:
                model = SIRModel(beta=b, gamma=g, verbose=False)
                model.simulate(S0, I0, R0, t_span)
                peak = model.get_peak_info()
                results.append({
                    'beta': b,
                    'gamma': g,
                    'R0': b/g,
                    'peak_time': peak['peak_time'],
                    'peak_infected': peak['peak_infected']
                })
        
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n敏感性分析结果:")
        print(df.to_string(index=False))
        return df


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("   SIR传染病模型演示")
    print("="*60)
    
    # 1. 基本模拟
    model = SIRModel(beta=0.3, gamma=0.1, verbose=True)
    model.simulate(S0=999, I0=1, R0=0, t_span=(0, 100, 100))
    model.plot_simulation()
    
    # 2. 干预措施对比（降低传染率）
    print("\n【干预措施对比】")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for beta, label in [(0.3, '无干预 β=0.3'), 
                         (0.2, '中度干预 β=0.2'),
                         (0.1, '强力干预 β=0.1')]:
        m = SIRModel(beta=beta, gamma=0.1, verbose=False)
        m.simulate(999, 1, 0, (0, 200, 200))
        ax.plot(m.time, m.solution[:, 1], linewidth=2.5, label=label)
    
    ax.set_xlabel('时间（天）', fontsize=12, fontweight='bold')
    ax.set_ylabel('感染人数', fontsize=12, fontweight='bold')
    ax.set_title('不同干预强度下的疫情曲线', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
