"""
============================================================
种群增长微分方程模型 (Population Growth ODE)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：生态系统建模、资源承载力分析、增长趋势预测
方法：Logistic增长、Malthus增长、Lotka-Volterra捕食模型
作者：MCM/ICM Team
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.integrate import odeint

# 图表美化设置
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class PopulationModel:
    """
    种群增长模型封装类
    
    支持模型：
    - Malthus: dN/dt = rN（指数增长）
    - Logistic: dN/dt = rN(1 - N/K)（有限资源）
    - Lotka-Volterra: 捕食者-猎物模型
    
    核心公式（Logistic）：
    - dN/dt = r * N * (1 - N/K)
    - N: 种群数量
    - r: 内禀增长率
    - K: 环境容纳量（最大承载力）
    """
    
    def __init__(self, model_type='logistic', verbose=True):
        """
        参数配置
        
        :param model_type: 'malthus'/'logistic'/'lotka_volterra'
        :param verbose: 是否打印过程
        """
        self.model_type = model_type.lower()
        self.verbose = verbose
        self.solution = None
        self.time = None
        self.params = None
    
    def _logistic(self, N, t, r, K):
        """Logistic增长模型"""
        return r * N * (1 - N / K)
    
    def _malthus(self, N, t, r):
        """Malthus指数增长模型"""
        return r * N
    
    def _lotka_volterra(self, y, t, alpha, beta, gamma, delta):
        """Lotka-Volterra捕食者-猎物模型"""
        prey, predator = y
        dprey_dt = alpha * prey - beta * prey * predator
        dpredator_dt = delta * prey * predator - gamma * predator
        return [dprey_dt, dpredator_dt]
    
    def solve(self, y0, t_span, **params):
        """
        求解微分方程
        
        :param y0: 初始条件（标量或数组）
        :param t_span: 时间范围 (t_start, t_end, n_points)
        :param params: 模型参数
        """
        self.time = np.linspace(t_span[0], t_span[1], t_span[2])
        self.params = params
        
        if self.model_type == 'logistic':
            r = params.get('r', 0.5)
            K = params.get('K', 1000)
            self.solution = odeint(self._logistic, y0, self.time, args=(r, K))
        
        elif self.model_type == 'malthus':
            r = params.get('r', 0.5)
            self.solution = odeint(self._malthus, y0, self.time, args=(r,))
        
        elif self.model_type == 'lotka_volterra':
            alpha = params.get('alpha', 1.1)  # 猎物增长率
            beta = params.get('beta', 0.4)    # 捕食率
            gamma = params.get('gamma', 0.4)  # 捕食者死亡率
            delta = params.get('delta', 0.1)  # 捕食者转化率
            self.solution = odeint(self._lotka_volterra, y0, self.time, 
                                   args=(alpha, beta, gamma, delta))
        
        if self.verbose:
            self._print_results(y0)
        
        return self.solution
    
    def _print_results(self, y0):
        """打印结果"""
        print("\n" + "="*50)
        print(f"🦌 {self.model_type.upper()} 种群模型求解结果")
        print("="*50)
        print(f"\n  初始条件: {y0}")
        print(f"  模型参数: {self.params}")
        
        if self.model_type in ['logistic', 'malthus']:
            print(f"\n  时间演化（采样）:")
            indices = [0, len(self.time)//4, len(self.time)//2, -1]
            for i in indices:
                print(f"    t={self.time[i]:.1f}: N={self.solution[i][0]:.1f}")
            K = self.params.get('K', None)
            if K:
                print(f"\n  稳态值: {self.solution[-1][0]:.1f} (K={K})")
        
        elif self.model_type == 'lotka_volterra':
            print(f"\n  捕食者-猎物动态（末期）:")
            print(f"    猎物: {self.solution[-1][0]:.1f}")
            print(f"    捕食者: {self.solution[-1][1]:.1f}")
        
        print("="*50)
    
    def plot_solution(self, save_path=None):
        """可视化求解结果"""
        if self.solution is None:
            raise ValueError("请先调用solve()求解")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if self.model_type in ['logistic', 'malthus']:
            ax.plot(self.time, self.solution, color='#2E86AB', linewidth=2.5,
                   label='种群数量 N(t)')
            
            if self.model_type == 'logistic':
                K = self.params.get('K', 1000)
                ax.axhline(y=K, color='#E94F37', linestyle='--', linewidth=2,
                          label=f'环境容纳量 K={K}')
            
            ax.set_ylabel('种群数量', fontsize=12, fontweight='bold')
        
        elif self.model_type == 'lotka_volterra':
            ax.plot(self.time, self.solution[:, 0], color='#2E86AB', 
                   linewidth=2.5, label='猎物')
            ax.plot(self.time, self.solution[:, 1], color='#E94F37',
                   linewidth=2.5, label='捕食者')
            ax.set_ylabel('种群数量', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('时间', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.model_type.upper()} 种群增长模型', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_phase_portrait(self, save_path=None):
        """相图（仅Lotka-Volterra）"""
        if self.model_type != 'lotka_volterra':
            print("相图仅适用于Lotka-Volterra模型")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(self.solution[:, 0], self.solution[:, 1], color='#2E86AB',
               linewidth=2)
        ax.scatter(self.solution[0, 0], self.solution[0, 1], color='green',
                  s=100, zorder=5, label='起点')
        ax.scatter(self.solution[-1, 0], self.solution[-1, 1], color='red',
                  s=100, zorder=5, label='终点')
        
        ax.set_xlabel('猎物数量', fontsize=12, fontweight='bold')
        ax.set_ylabel('捕食者数量', fontsize=12, fontweight='bold')
        ax.set_title('Lotka-Volterra 相图', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("   种群增长微分方程模型演示")
    print("="*60)
    
    # 1. Logistic增长模型
    print("\n【Logistic 增长模型】")
    logistic = PopulationModel(model_type='logistic')
    logistic.solve(y0=100, t_span=(0, 20, 100), r=0.5, K=1000)
    logistic.plot_solution()
    
    # 2. Malthus指数增长（对比）
    print("\n【Malthus 指数增长模型】")
    malthus = PopulationModel(model_type='malthus')
    malthus.solve(y0=100, t_span=(0, 10, 100), r=0.3)
    malthus.plot_solution()
    
    # 3. Lotka-Volterra捕食者-猎物模型
    print("\n【Lotka-Volterra 捕食模型】")
    lv = PopulationModel(model_type='lotka_volterra')
    lv.solve(y0=[40, 9], t_span=(0, 50, 500), 
             alpha=1.1, beta=0.4, gamma=0.4, delta=0.1)
    lv.plot_solution()
    lv.plot_phase_portrait()
