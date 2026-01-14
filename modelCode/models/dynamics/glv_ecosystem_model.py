# -*- coding: utf-8 -*-
"""
广义 Lotka-Volterra (GLV) 生态系统模型 - 完整版
================================================
基于建模思路实现：
1. 基础模型：广义 Lotka-Volterra 的改进版（含水资源依赖的相互作用系数）
2. 引入均匀度（Evenness）作为系统变量
3. 模拟环境波动（季节性变化 + 随机干旱）

数据来源：CDR LTER e120 实验数据
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import seaborn as sns
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ==================== 第一部分：数据加载与预处理 ====================

def load_and_preprocess_data(data_path):
    """
    加载并预处理CDR LTER e120实验数据
    
    Parameters:
    -----------
    data_path : str
        数据文件夹路径
    
    Returns:
    --------
    plot_treatments : DataFrame
        样地处理数据
    planted_species : DataFrame  
        种植物种数据
    species_summary : DataFrame
        物种汇总统计
    """
    print("=" * 60)
    print("数据加载与预处理")
    print("=" * 60)
    
    # 加载数据
    plot_treatments = pd.read_csv(f"{data_path}/CDRLTERe120PlotTreatments.csv")
    planted_species = pd.read_csv(f"{data_path}/CDRLTERe120PlantedSpecies.csv")
    all_treatments = pd.read_csv(f"{data_path}/CDRLTERAllBigBioPlotTreatments.csv")
    all_species = pd.read_csv(f"{data_path}/CDRLTERAllBigBioPlantedSpecies.csv")
    
    print(f"\n[1] e120 样地处理数据: {plot_treatments.shape[0]} 行, {plot_treatments.shape[1]} 列")
    print(f"[2] e120 物种数据: {planted_species.shape[0]} 行")
    print(f"[3] BigBio 样地处理数据: {all_treatments.shape[0]} 行")
    print(f"[4] BigBio 物种数据: {all_species.shape[0]} 行")
    
    # 统计物种丰富度分布
    species_richness = plot_treatments.groupby('NumSp').size()
    print(f"\n物种丰富度分布:")
    print(species_richness)
    
    # 功能群分析
    functional_groups = ['C3', 'C4', 'Forb', 'Legume', 'Woody']
    fg_summary = plot_treatments[functional_groups].sum()
    print(f"\n功能群分布:")
    for fg, count in fg_summary.items():
        print(f"  {fg}: {count} 样地")
    
    # 物种统计
    species_counts = planted_species['Species'].value_counts()
    print(f"\n物种总数: {species_counts.shape[0]}")
    print(f"出现频率最高的5个物种:")
    print(species_counts.head())
    
    return plot_treatments, planted_species, all_treatments, all_species


# ==================== 第二部分：数据可视化 ====================

def visualize_data(plot_treatments, planted_species, save_path=None):
    """
    数据探索性可视化
    
    生成图表：
    1. 物种丰富度分布直方图
    2. 功能群分布热力图
    3. 物种组合网络图（简化版）
    """
    print("\n" + "=" * 60)
    print("数据可视化")
    print("=" * 60)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 物种丰富度分布
    ax1 = fig.add_subplot(2, 2, 1)
    richness_counts = plot_treatments['NumSp'].value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(richness_counts)))
    bars = ax1.bar(richness_counts.index, richness_counts.values, color=colors, edgecolor='black')
    ax1.set_xlabel('物种丰富度 (Number of Species)', fontsize=12)
    ax1.set_ylabel('样地数量', fontsize=12)
    ax1.set_title('(a) 物种丰富度分布', fontsize=14, fontweight='bold')
    ax1.set_xticks(richness_counts.index)
    
    # 添加数值标签
    for bar, val in zip(bars, richness_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(val), ha='center', va='bottom', fontsize=10)
    
    # 2. 功能群分布
    ax2 = fig.add_subplot(2, 2, 2)
    functional_groups = ['C3', 'C4', 'Forb', 'Legume', 'Woody']
    fg_names = ['C3草本', 'C4草本', '禾本科', '豆科', '木本']
    fg_sums = plot_treatments[functional_groups].sum()
    colors2 = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
    wedges, texts, autotexts = ax2.pie(fg_sums, labels=fg_names, autopct='%1.1f%%',
                                        colors=colors2, explode=[0.02]*5)
    ax2.set_title('(b) 功能群分布比例', fontsize=14, fontweight='bold')
    
    # 3. 功能群组合热力图
    ax3 = fig.add_subplot(2, 2, 3)
    fg_data = plot_treatments[functional_groups]
    fg_corr = fg_data.T.dot(fg_data)  # 共现矩阵
    sns.heatmap(fg_corr, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=fg_names, yticklabels=fg_names, ax=ax3)
    ax3.set_title('(c) 功能群共现矩阵', fontsize=14, fontweight='bold')
    
    # 4. 物种丰富度与功能群数量关系
    ax4 = fig.add_subplot(2, 2, 4)
    plot_treatments['TotalFG'] = plot_treatments[functional_groups].sum(axis=1)
    scatter = ax4.scatter(plot_treatments['NumSp'], plot_treatments['TotalFG'],
                         c=plot_treatments['FgNum'], cmap='plasma', 
                         s=80, alpha=0.7, edgecolors='black')
    ax4.set_xlabel('物种丰富度', fontsize=12)
    ax4.set_ylabel('功能群数量', fontsize=12)
    ax4.set_title('(d) 物种丰富度 vs 功能群数量', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('功能群编号', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/data_visualization.png", dpi=300, bbox_inches='tight')
        print(f"\n图表已保存: {save_path}/data_visualization.png")
    
    plt.show()
    
    return fig


# ==================== 第三部分：GLV 模型核心实现 ====================

class GLVEcosystemModel:
    """
    广义 Lotka-Volterra 生态系统模型
    
    模型方程：
    dB_i/dt = r_i * B_i * (1 - (B_i + Σ α_ij(W) * B_j) / K_i)
    
    其中 α_ij(W) 是水资源依赖的相互作用系数
    """
    
    def __init__(self, n_species=5, seed=42):
        """
        初始化模型参数
        
        Parameters:
        -----------
        n_species : int
            物种数量
        seed : int
            随机种子
        """
        np.random.seed(seed)
        self.n_species = n_species
        self.species_names = [f'Species_{i+1}' for i in range(n_species)]
        
        # 生态参数初始化
        self.r = np.random.uniform(0.1, 0.5, n_species)  # 固有增长率
        self.K = np.random.uniform(80, 150, n_species)    # 环境容纳量
        
        # 基础相互作用矩阵（竞争为正，互利为负）
        self.alpha_base = np.random.uniform(0.1, 0.5, (n_species, n_species))
        np.fill_diagonal(self.alpha_base, 1.0)  # 对角线为1（自我限制）
        
        # 水资源阈值参数
        self.W_threshold = 0.5  # 水资源阈值
        self.alpha_sensitivity = 2.0  # 相互作用对水资源的敏感度
        
        print(f"\n初始化 GLV 模型: {n_species} 个物种")
        print(f"固有增长率 r: {self.r}")
        print(f"环境容纳量 K: {self.K}")
    
    def alpha_ij(self, W, i, j):
        """
        计算水资源依赖的相互作用系数
        
        公式: α_ij(W) = a * tanh(W - W_threshold) + b
        
        当 W > W_threshold 时，α > 0（竞争）
        当 W < W_threshold 时，α < 0 或接近0（互利/促进）
        
        Parameters:
        -----------
        W : float
            当前水资源量（0-1标准化）
        i, j : int
            物种索引
        
        Returns:
        --------
        float : 相互作用系数
        """
        if i == j:
            return 1.0  # 自我限制始终为1
        
        base = self.alpha_base[i, j]
        # 使用 tanh 函数实现竞争-互利转换
        # a 控制变化范围，b 控制基准值
        a = base
        b = base * 0.2  # 基准偏移
        
        alpha = a * np.tanh(self.alpha_sensitivity * (W - self.W_threshold)) + b
        return max(-0.5, min(1.5, alpha))  # 限制范围
    
    def get_alpha_matrix(self, W):
        """
        获取当前水资源条件下的完整相互作用矩阵
        """
        alpha = np.zeros((self.n_species, self.n_species))
        for i in range(self.n_species):
            for j in range(self.n_species):
                alpha[i, j] = self.alpha_ij(W, i, j)
        return alpha
    
    def water_dynamics(self, t, W_base=0.6, A=0.2, omega=0.5, sigma=0.1, drought_events=None):
        """
        水资源动态方程
        
        W(t) = W_base + A * sin(ωt) + σ * ξ(t)
        
        可以添加突发干旱事件
        """
        # 季节性变化
        W_seasonal = W_base + A * np.sin(omega * t)
        
        # 随机噪声（每个时间点独立）
        np.random.seed(int(t * 1000) % 10000)
        noise = sigma * np.random.randn()
        
        # 突发干旱事件
        drought_effect = 0
        if drought_events is not None:
            for start, end, intensity in drought_events:
                if start <= t <= end:
                    drought_effect = -intensity
                    break
        
        W = W_seasonal + noise + drought_effect
        return max(0.1, min(1.0, W))  # 限制在 [0.1, 1.0]
    
    def glv_equations(self, B, t, W_func, drought_events=None):
        """
        GLV 微分方程组
        
        dB_i/dt = r_i * B_i * (1 - (B_i + Σ α_ij(W) * B_j) / K_i)
        """
        W = W_func(t, drought_events=drought_events)
        alpha = self.get_alpha_matrix(W)
        
        dBdt = np.zeros(self.n_species)
        for i in range(self.n_species):
            # 计算种间竞争/互利效应
            competition_sum = sum(alpha[i, j] * B[j] for j in range(self.n_species) if j != i)
            
            # GLV 方程
            dBdt[i] = self.r[i] * B[i] * (1 - (B[i] + competition_sum) / self.K[i])
        
        return dBdt
    
    def simulate(self, t_span, B0=None, drought_events=None):
        """
        运行模型模拟
        
        Parameters:
        -----------
        t_span : array
            时间点数组
        B0 : array
            初始生物量
        drought_events : list
            干旱事件列表 [(start, end, intensity), ...]
        
        Returns:
        --------
        B : array
            各物种生物量时间序列
        W : array
            水资源时间序列
        """
        if B0 is None:
            B0 = np.random.uniform(10, 30, self.n_species)
        
        # 定义水资源函数
        W_func = lambda t, drought_events=drought_events: self.water_dynamics(
            t, drought_events=drought_events
        )
        
        # 求解 ODE
        B = odeint(self.glv_equations, B0, t_span, args=(W_func, drought_events))
        
        # 计算水资源时间序列
        W = np.array([W_func(t) for t in t_span])
        
        return B, W
    
    def calculate_evenness(self, B):
        """
        计算均匀度 (Evenness)
        
        E(t) = -Σ p_i * ln(p_i) / ln(S)
        
        其中 p_i = B_i / Σ B_k
        """
        total_biomass = B.sum(axis=1)
        total_biomass[total_biomass == 0] = 1e-10  # 避免除零
        
        p = B / total_biomass[:, np.newaxis]
        p[p <= 0] = 1e-10  # 避免 log(0)
        
        H = -np.sum(p * np.log(p), axis=1)  # Shannon 指数
        H_max = np.log(self.n_species)
        
        E = H / H_max
        return E
    
    def calculate_total_yield(self, B, E, facilitation_threshold=0.5):
        """
        计算总产出，考虑均匀度效应
        
        Y_total = Σ B_i * f(E)
        
        当均匀度低时，互利效应减弱
        """
        total_biomass = B.sum(axis=1)
        
        # 均匀度效应系数
        f_E = np.where(E > facilitation_threshold, 
                       1 + 0.2 * (E - facilitation_threshold),  # 高均匀度有增益
                       0.8 + 0.4 * E)  # 低均匀度时效应降低
        
        Y_total = total_biomass * f_E
        return Y_total


# ==================== 第四部分：模型分析与图表生成 ====================

def analyze_and_visualize_model(model, t_span, drought_events=None, save_path=None):
    """
    模型分析与可视化
    
    生成图表：
    1. 生物量动态变化图
    2. 水资源与相互作用系数变化
    3. 均匀度时间序列
    4. 总产出分析
    5. 相位空间图
    """
    print("\n" + "=" * 60)
    print("模型分析与可视化")
    print("=" * 60)
    
    # 运行模拟
    B0 = np.array([20, 25, 15, 30, 22])  # 初始生物量
    B, W = model.simulate(t_span, B0, drought_events)
    E = model.calculate_evenness(B)
    Y = model.calculate_total_yield(B, E)
    
    print(f"\n模拟时长: {t_span[-1]} 时间单位")
    print(f"最终生物量: {B[-1]}")
    print(f"最终均匀度: {E[-1]:.4f}")
    print(f"平均总产出: {Y.mean():.2f}")
    
    # 创建综合图表
    fig = plt.figure(figsize=(18, 14))
    
    # 1. 生物量动态
    ax1 = fig.add_subplot(2, 3, 1)
    colors = plt.cm.Set2(np.linspace(0, 1, model.n_species))
    for i in range(model.n_species):
        ax1.plot(t_span, B[:, i], label=f'物种 {i+1}', color=colors[i], linewidth=2)
    ax1.set_xlabel('时间', fontsize=12)
    ax1.set_ylabel('生物量', fontsize=12)
    ax1.set_title('(a) 物种生物量动态', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 标记干旱期
    if drought_events:
        for start, end, intensity in drought_events:
            ax1.axvspan(start, end, alpha=0.2, color='red', label='干旱期' if start == drought_events[0][0] else '')
    
    # 2. 水资源动态
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t_span, W, 'b-', linewidth=2, label='水资源 W(t)')
    ax2.axhline(y=model.W_threshold, color='r', linestyle='--', label=f'阈值 W_th={model.W_threshold}')
    ax2.fill_between(t_span, 0, W, where=W < model.W_threshold, 
                     alpha=0.3, color='orange', label='水分胁迫区')
    ax2.fill_between(t_span, 0, W, where=W >= model.W_threshold,
                     alpha=0.3, color='blue', label='水分充足区')
    ax2.set_xlabel('时间', fontsize=12)
    ax2.set_ylabel('水资源量 W', fontsize=12)
    ax2.set_title('(b) 水资源动态与阈值', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # 3. 相互作用系数变化
    ax3 = fig.add_subplot(2, 3, 3)
    # 选择典型物种对的相互作用系数
    alpha_12 = [model.alpha_ij(w, 0, 1) for w in W]
    alpha_23 = [model.alpha_ij(w, 1, 2) for w in W]
    alpha_34 = [model.alpha_ij(w, 2, 3) for w in W]
    
    ax3.plot(t_span, alpha_12, label='α₁₂', linewidth=2)
    ax3.plot(t_span, alpha_23, label='α₂₃', linewidth=2)
    ax3.plot(t_span, alpha_34, label='α₃₄', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax3.fill_between(t_span, 0, max(alpha_12), alpha=0.1, color='red', label='竞争区 (α>0)')
    ax3.fill_between(t_span, min(alpha_12), 0, alpha=0.1, color='green', label='互利区 (α<0)')
    ax3.set_xlabel('时间', fontsize=12)
    ax3.set_ylabel('相互作用系数 α', fontsize=12)
    ax3.set_title('(c) 相互作用系数动态变化', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. 均匀度时间序列
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t_span, E, 'g-', linewidth=2, label='均匀度 E(t)')
    ax4.axhline(y=0.5, color='orange', linestyle='--', label='临界均匀度')
    ax4.fill_between(t_span, 0, E, alpha=0.3, color='green')
    ax4.set_xlabel('时间', fontsize=12)
    ax4.set_ylabel('均匀度 E', fontsize=12)
    ax4.set_title('(d) 群落均匀度动态', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3)
    
    # 5. 总产出与水资源关系
    ax5 = fig.add_subplot(2, 3, 5)
    scatter = ax5.scatter(W, Y, c=t_span, cmap='viridis', s=30, alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('时间', fontsize=10)
    
    # 拟合趋势线
    z = np.polyfit(W, Y, 2)
    p = np.poly1d(z)
    W_sorted = np.sort(W)
    ax5.plot(W_sorted, p(W_sorted), 'r--', linewidth=2, label='二次拟合')
    
    ax5.set_xlabel('水资源量 W', fontsize=12)
    ax5.set_ylabel('总产出 Y', fontsize=12)
    ax5.set_title('(e) 水资源与总产出关系', fontsize=14, fontweight='bold')
    ax5.legend(loc='upper left', fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. 相位空间图（选取两个物种）
    ax6 = fig.add_subplot(2, 3, 6)
    sc = ax6.scatter(B[:, 0], B[:, 1], c=W, cmap='coolwarm', s=20, alpha=0.7)
    ax6.plot(B[:, 0], B[:, 1], 'k-', alpha=0.2, linewidth=0.5)
    ax6.scatter(B[0, 0], B[0, 1], c='green', s=100, marker='o', label='起点', zorder=5)
    ax6.scatter(B[-1, 0], B[-1, 1], c='red', s=100, marker='s', label='终点', zorder=5)
    cbar2 = plt.colorbar(sc, ax=ax6)
    cbar2.set_label('水资源 W', fontsize=10)
    ax6.set_xlabel('物种1 生物量', fontsize=12)
    ax6.set_ylabel('物种2 生物量', fontsize=12)
    ax6.set_title('(f) 相位空间轨迹', fontsize=14, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/model_analysis.png", dpi=300, bbox_inches='tight')
        print(f"\n图表已保存: {save_path}/model_analysis.png")
    
    plt.show()
    
    return B, W, E, Y, fig


def visualize_alpha_function(model, save_path=None):
    """
    可视化相互作用系数函数 α(W)
    
    展示竞争-互利转换机制
    """
    print("\n" + "=" * 60)
    print("相互作用系数函数分析")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. α(W) 函数曲线
    ax1 = axes[0]
    W_range = np.linspace(0, 1, 200)
    
    # 不同物种对的 α(W) 曲线
    for i in range(min(3, model.n_species)):
        for j in range(i+1, min(4, model.n_species)):
            alpha_curve = [model.alpha_ij(w, i, j) for w in W_range]
            ax1.plot(W_range, alpha_curve, linewidth=2, 
                    label=f'α_{{{i+1}{j+1}}}(W)')
    
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=1.5)
    ax1.axvline(x=model.W_threshold, color='r', linestyle='--', 
                linewidth=1.5, label=f'$W_{{threshold}}$={model.W_threshold}')
    
    # 填充区域
    ax1.fill_between([0, model.W_threshold], -0.6, 0, alpha=0.2, color='green', 
                     label='互利区域 (Facilitation)')
    ax1.fill_between([model.W_threshold, 1], 0, 1.5, alpha=0.2, color='red',
                     label='竞争区域 (Competition)')
    
    ax1.set_xlabel('水资源量 W', fontsize=12)
    ax1.set_ylabel('相互作用系数 α', fontsize=12)
    ax1.set_title('相互作用系数函数 α(W) = a·tanh(W - W_th) + b', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.6, 1.5)
    ax1.grid(True, alpha=0.3)
    
    # 添加注释
    ax1.annotate('干旱条件下\n物种间互利', xy=(0.2, -0.3), fontsize=10,
                ha='center', color='green', fontweight='bold')
    ax1.annotate('水分充足时\n物种间竞争', xy=(0.8, 0.8), fontsize=10,
                ha='center', color='red', fontweight='bold')
    
    # 2. 热力图：不同水资源条件下的相互作用矩阵
    ax2 = axes[1]
    
    W_values = [0.2, 0.5, 0.8]
    W_labels = ['干旱 (W=0.2)', '临界 (W=0.5)', '湿润 (W=0.8)']
    
    matrices = []
    for w in W_values:
        matrices.append(model.get_alpha_matrix(w))
    
    # 创建子图
    for idx, (matrix, w_label) in enumerate(zip(matrices, W_labels)):
        ax_sub = fig.add_axes([0.55 + idx*0.15, 0.15, 0.12, 0.7])
        im = ax_sub.imshow(matrix, cmap='RdBu_r', vmin=-0.5, vmax=1.5)
        ax_sub.set_title(w_label, fontsize=10)
        ax_sub.set_xticks(range(model.n_species))
        ax_sub.set_yticks(range(model.n_species))
        ax_sub.set_xticklabels([f'S{i+1}' for i in range(model.n_species)], fontsize=8)
        ax_sub.set_yticklabels([f'S{i+1}' for i in range(model.n_species)], fontsize=8)
        
        if idx == 2:
            cbar = plt.colorbar(im, ax=ax_sub, fraction=0.046, pad=0.04)
            cbar.set_label('α', fontsize=10)
    
    # 隐藏原始 ax2
    ax2.axis('off')
    ax2.set_title('不同水资源条件下的相互作用矩阵', fontsize=14, 
                  fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/alpha_function.png", dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}/alpha_function.png")
    
    plt.show()
    
    return fig


def compare_scenarios(model, t_span, save_path=None):
    """
    比较不同情景下的模型结果
    
    情景1：正常季节变化（无干旱）
    情景2：单次干旱事件
    情景3：多次干旱事件（不规则天气循环）
    """
    print("\n" + "=" * 60)
    print("情景比较分析")
    print("=" * 60)
    
    B0 = np.array([20, 25, 15, 30, 22])
    
    # 情景定义
    scenarios = {
        '正常季节': None,
        '单次干旱': [(20, 30, 0.4)],
        '多次干旱': [(15, 22, 0.3), (40, 48, 0.5), (70, 78, 0.35)]
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors_scenario = ['#3498db', '#e74c3c', '#2ecc71']
    
    results = {}
    
    for idx, (scenario_name, drought_events) in enumerate(scenarios.items()):
        print(f"\n运行情景: {scenario_name}")
        
        B, W = model.simulate(t_span, B0, drought_events)
        E = model.calculate_evenness(B)
        Y = model.calculate_total_yield(B, E)
        
        results[scenario_name] = {'B': B, 'W': W, 'E': E, 'Y': Y}
        
        # 上行：生物量动态
        ax_top = axes[0, idx]
        for i in range(model.n_species):
            ax_top.plot(t_span, B[:, i], linewidth=1.5, alpha=0.8)
        ax_top.plot(t_span, B.sum(axis=1), 'k--', linewidth=2, label='总生物量')
        ax_top.set_title(f'{scenario_name}', fontsize=13, fontweight='bold')
        ax_top.set_xlabel('时间', fontsize=11)
        ax_top.set_ylabel('生物量', fontsize=11)
        ax_top.legend(loc='upper right', fontsize=8)
        ax_top.grid(True, alpha=0.3)
        
        # 标记干旱期
        if drought_events:
            for start, end, intensity in drought_events:
                ax_top.axvspan(start, end, alpha=0.3, color='red')
        
        # 下行：水资源和均匀度
        ax_bottom = axes[1, idx]
        ax_bottom.plot(t_span, W, 'b-', linewidth=2, label='水资源 W')
        ax_bottom.axhline(y=model.W_threshold, color='r', linestyle='--', alpha=0.5)
        
        ax_twin = ax_bottom.twinx()
        ax_twin.plot(t_span, E, 'g-', linewidth=2, label='均匀度 E')
        ax_twin.set_ylabel('均匀度 E', fontsize=11, color='green')
        ax_twin.tick_params(axis='y', labelcolor='green')
        
        ax_bottom.set_xlabel('时间', fontsize=11)
        ax_bottom.set_ylabel('水资源 W', fontsize=11, color='blue')
        ax_bottom.tick_params(axis='y', labelcolor='blue')
        ax_bottom.grid(True, alpha=0.3)
        
        # 统计信息
        print(f"  平均总产出: {Y.mean():.2f}")
        print(f"  最终均匀度: {E[-1]:.4f}")
        print(f"  产出变异系数: {Y.std()/Y.mean():.4f}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/scenario_comparison.png", dpi=300, bbox_inches='tight')
        print(f"\n图表已保存: {save_path}/scenario_comparison.png")
    
    plt.show()
    
    # 绘制对比汇总图
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    scenario_names = list(scenarios.keys())
    
    # 1. 总产出对比
    ax = axes2[0]
    for idx, name in enumerate(scenario_names):
        ax.plot(t_span, results[name]['Y'], linewidth=2, 
               color=colors_scenario[idx], label=name)
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('总产出 Y', fontsize=12)
    ax.set_title('总产出对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. 均匀度对比
    ax = axes2[1]
    for idx, name in enumerate(scenario_names):
        ax.plot(t_span, results[name]['E'], linewidth=2,
               color=colors_scenario[idx], label=name)
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('均匀度 E', fontsize=12)
    ax.set_title('均匀度对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. 箱线图统计对比
    ax = axes2[2]
    data_for_boxplot = [results[name]['Y'] for name in scenario_names]
    bp = ax.boxplot(data_for_boxplot, labels=scenario_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_scenario):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('总产出 Y', fontsize=12)
    ax.set_title('总产出分布对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/scenario_summary.png", dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}/scenario_summary.png")
    
    plt.show()
    
    return results, fig, fig2


def plot_model_framework(save_path=None):
    """
    绘制模型框架图，明确展示建模关系
    """
    print("\n" + "=" * 60)
    print("模型框架图")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 定义框的样式
    box_style = dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                     edgecolor='navy', linewidth=2)
    box_style_green = dict(boxstyle='round,pad=0.5', facecolor='lightgreen',
                           edgecolor='darkgreen', linewidth=2)
    box_style_orange = dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                            edgecolor='orange', linewidth=2)
    box_style_red = dict(boxstyle='round,pad=0.5', facecolor='mistyrose',
                         edgecolor='red', linewidth=2)
    
    # 标题
    ax.text(7, 9.5, '广义 Lotka-Volterra 生态系统模型框架', 
            fontsize=16, fontweight='bold', ha='center', va='center')
    
    # 核心方程框
    ax.text(7, 8, 
            r'$\frac{dB_i}{dt} = r_i B_i \left(1 - \frac{B_i + \sum_{j \neq i} \alpha_{ij}(W) B_j}{K_i}\right)$',
            fontsize=14, ha='center', va='center', bbox=box_style)
    
    # 水资源动态
    ax.text(2.5, 6, 
            '水资源动态\n$W(t) = W_{base} + A\\sin(\\omega t) + \\sigma\\xi(t)$',
            fontsize=11, ha='center', va='center', bbox=box_style_green,
            multialignment='center')
    
    # 相互作用系数
    ax.text(7, 6,
            '相互作用系数\n$\\alpha_{ij}(W) = a \\cdot \\tanh(W - W_{th}) + b$\n\n水多→竞争 (α>0)\n水少→互利 (α<0)',
            fontsize=10, ha='center', va='center', bbox=box_style_orange,
            multialignment='center')
    
    # 均匀度
    ax.text(11.5, 6,
            '均匀度\n$E(t) = -\\frac{\\sum p_i \\ln p_i}{\\ln S}$',
            fontsize=11, ha='center', va='center', bbox=box_style_green,
            multialignment='center')
    
    # 总产出
    ax.text(7, 3.5,
            '总产出\n$Y_{total} = \\sum B_i \\cdot f(E)$',
            fontsize=12, ha='center', va='center', bbox=box_style_red,
            multialignment='center')
    
    # 干旱事件
    ax.text(2.5, 3.5,
            '干旱事件\n突发性干扰',
            fontsize=11, ha='center', va='center', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightsalmon', 
                     edgecolor='red', linewidth=2, linestyle='--'),
            multialignment='center')
    
    # 模型输出
    ax.text(11.5, 3.5,
            '模型输出\n• 生物量动态\n• 物种竞争/互利\n• 生态系统稳定性',
            fontsize=10, ha='center', va='center', bbox=box_style,
            multialignment='center')
    
    # 绘制箭头连接
    arrow_style = dict(arrowstyle='->', color='navy', lw=2)
    
    # 水资源 → 相互作用系数
    ax.annotate('', xy=(5.2, 6), xytext=(4, 6), arrowprops=arrow_style)
    
    # 相互作用系数 → 核心方程
    ax.annotate('', xy=(7, 7.3), xytext=(7, 6.8), arrowprops=arrow_style)
    
    # 核心方程 → 均匀度
    ax.annotate('', xy=(10, 6.5), xytext=(8.5, 7.5), arrowprops=arrow_style)
    
    # 均匀度 → 总产出
    ax.annotate('', xy=(9.5, 4), xytext=(10.5, 5.3), arrowprops=arrow_style)
    
    # 核心方程 → 总产出
    ax.annotate('', xy=(7, 4.2), xytext=(7, 7.3), arrowprops=arrow_style)
    
    # 干旱事件 → 水资源
    ax.annotate('', xy=(2.5, 5.2), xytext=(2.5, 4.2), 
                arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))
    
    # 总产出 → 模型输出
    ax.annotate('', xy=(10, 3.5), xytext=(8.5, 3.5), arrowprops=arrow_style)
    
    # 添加参数说明框
    param_text = '''模型参数说明:
• $r_i$: 物种 i 的固有增长率
• $K_i$: 物种 i 的环境容纳量  
• $W_{th}$: 水资源阈值 (竞争-互利转换点)
• $S$: 物种丰富度
• $p_i$: 物种 i 的相对生物量'''
    
    ax.text(0.5, 1.2, param_text, fontsize=10, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='gray', linewidth=1),
            family='monospace')
    
    # 论文结论框
    conclusion_text = '''关键论文结论:
1. 水资源充足时: 物种间主要为竞争关系
2. 干旱胁迫下: 竞争转为互利 (促进作用)
3. 低均匀度削弱多样性的抗旱效应
4. 干旱改变物种相互作用的方向'''
    
    ax.text(10, 1.2, conclusion_text, fontsize=10, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                     edgecolor='orange', linewidth=1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/model_framework.png", dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}/model_framework.png")
    
    plt.show()
    
    return fig


# ==================== 第五部分：模型改良与敏感性分析 ====================

def sensitivity_analysis(model, t_span, save_path=None):
    """
    模型敏感性分析
    
    分析关键参数对模型结果的影响：
    1. 水资源阈值 W_threshold
    2. 相互作用敏感度 α_sensitivity
    3. 初始生物量配置
    """
    print("\n" + "=" * 60)
    print("敏感性分析")
    print("=" * 60)
    
    B0 = np.array([20, 25, 15, 30, 22])
    drought_events = [(30, 45, 0.4)]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # 1. 水资源阈值敏感性
    ax1 = axes[0, 0]
    W_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(W_thresholds)))
    
    for wth, color in zip(W_thresholds, colors):
        model.W_threshold = wth
        B, W = model.simulate(t_span, B0, drought_events)
        Y = model.calculate_total_yield(B, model.calculate_evenness(B))
        ax1.plot(t_span, Y, linewidth=2, color=color, label=f'$W_{{th}}$={wth}')
    
    model.W_threshold = 0.5  # 恢复默认值
    ax1.set_xlabel('时间', fontsize=12)
    ax1.set_ylabel('总产出 Y', fontsize=12)
    ax1.set_title('(a) 水资源阈值敏感性分析', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. 相互作用敏感度分析
    ax2 = axes[0, 1]
    sensitivities = [0.5, 1.0, 2.0, 3.0, 4.0]
    
    for sens, color in zip(sensitivities, colors):
        model.alpha_sensitivity = sens
        B, W = model.simulate(t_span, B0, drought_events)
        Y = model.calculate_total_yield(B, model.calculate_evenness(B))
        ax2.plot(t_span, Y, linewidth=2, color=color, label=f'敏感度={sens}')
    
    model.alpha_sensitivity = 2.0  # 恢复默认值
    ax2.set_xlabel('时间', fontsize=12)
    ax2.set_ylabel('总产出 Y', fontsize=12)
    ax2.set_title('(b) 相互作用敏感度分析', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. 初始均匀度影响
    ax3 = axes[1, 0]
    initial_configs = {
        '高均匀 (20,20,20,20,20)': np.array([20, 20, 20, 20, 20]),
        '中均匀 (30,25,20,15,10)': np.array([30, 25, 20, 15, 10]),
        '低均匀 (60,20,10,5,5)': np.array([60, 20, 10, 5, 5]),
        '单一优势 (80,5,5,5,5)': np.array([80, 5, 5, 5, 5]),
    }
    
    colors3 = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    for (config_name, B0_config), color in zip(initial_configs.items(), colors3):
        B, W = model.simulate(t_span, B0_config, drought_events)
        E = model.calculate_evenness(B)
        ax3.plot(t_span, E, linewidth=2, color=color, label=config_name)
    
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('时间', fontsize=12)
    ax3.set_ylabel('均匀度 E', fontsize=12)
    ax3.set_title('(c) 初始配置对均匀度的影响', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. 参数空间热力图
    ax4 = axes[1, 1]
    W_th_range = np.linspace(0.3, 0.7, 15)
    sens_range = np.linspace(0.5, 4.0, 15)
    
    mean_yields = np.zeros((len(W_th_range), len(sens_range)))
    
    for i, wth in enumerate(W_th_range):
        for j, sens in enumerate(sens_range):
            model.W_threshold = wth
            model.alpha_sensitivity = sens
            B, W = model.simulate(t_span, B0, drought_events)
            Y = model.calculate_total_yield(B, model.calculate_evenness(B))
            mean_yields[i, j] = Y.mean()
    
    model.W_threshold = 0.5
    model.alpha_sensitivity = 2.0
    
    im = ax4.imshow(mean_yields, extent=[0.5, 4.0, 0.3, 0.7], 
                    aspect='auto', origin='lower', cmap='RdYlGn')
    ax4.set_xlabel('相互作用敏感度', fontsize=12)
    ax4.set_ylabel('水资源阈值 $W_{th}$', fontsize=12)
    ax4.set_title('(d) 参数空间：平均总产出', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('平均总产出', fontsize=10)
    
    # 标记最优点
    max_idx = np.unravel_index(mean_yields.argmax(), mean_yields.shape)
    opt_wth = W_th_range[max_idx[0]]
    opt_sens = sens_range[max_idx[1]]
    ax4.scatter(opt_sens, opt_wth, c='blue', s=100, marker='*', 
               edgecolors='white', linewidths=2, zorder=5)
    ax4.annotate(f'最优: ({opt_sens:.1f}, {opt_wth:.2f})', 
                xy=(opt_sens, opt_wth), xytext=(opt_sens+0.5, opt_wth+0.05),
                fontsize=10, color='blue',
                arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/sensitivity_analysis.png", dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}/sensitivity_analysis.png")
    
    plt.show()
    
    return fig


# ==================== 第六部分：主程序 ====================

def main():
    """
    主程序：运行完整的建模分析流程
    """
    print("\n" + "=" * 80)
    print(" 广义 Lotka-Volterra 生态系统模型 - 完整分析 ")
    print("=" * 80)
    
    # 设置路径
    data_path = r"d:\competition\美国大学生数学建模大赛\CDRLTERe120Treatments"
    save_path = r"d:\competition\美国大学生数学建模大赛\modelCode\figures"
    
    # 创建保存目录
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # ========== 1. 数据加载与预处理 ==========
    plot_treatments, planted_species, all_treatments, all_species = \
        load_and_preprocess_data(data_path)
    
    # ========== 2. 数据可视化 ==========
    visualize_data(plot_treatments, planted_species, save_path)
    
    # ========== 3. 初始化 GLV 模型 ==========
    model = GLVEcosystemModel(n_species=5, seed=42)
    
    # 时间设置
    t_span = np.linspace(0, 100, 1000)
    
    # ========== 4. 可视化相互作用函数 ==========
    visualize_alpha_function(model, save_path)
    
    # ========== 5. 模型框架图 ==========
    plot_model_framework(save_path)
    
    # ========== 6. 模型分析（含干旱事件）==========
    drought_events = [(25, 40, 0.4), (65, 75, 0.35)]
    B, W, E, Y, fig = analyze_and_visualize_model(
        model, t_span, drought_events, save_path
    )
    
    # ========== 7. 情景比较 ==========
    results, fig_comp, fig_sum = compare_scenarios(model, t_span, save_path)
    
    # ========== 8. 敏感性分析 ==========
    sensitivity_analysis(model, t_span, save_path)
    
    # ========== 9. 结果汇总 ==========
    print("\n" + "=" * 80)
    print(" 模型分析结果汇总 ")
    print("=" * 80)
    
    print("\n【1】模型核心特点：")
    print("   • 相互作用系数 α(W) 随水资源变化，实现竞争-互利转换")
    print("   • 干旱条件下（W < W_th），物种间从竞争转为互利")
    print("   • 均匀度 E 影响总产出，低均匀度削弱多样性效应")
    
    print("\n【2】情景分析结论：")
    for scenario, data in results.items():
        print(f"   {scenario}:")
        print(f"      - 平均总产出: {data['Y'].mean():.2f}")
        print(f"      - 产出稳定性 (CV): {data['Y'].std()/data['Y'].mean():.4f}")
        print(f"      - 最终均匀度: {data['E'][-1]:.4f}")
    
    print("\n【3】模型改良建议：")
    print("   • 可加入物种特异性的水资源响应曲线")
    print("   • 考虑空间异质性（斑块动态）")
    print("   • 引入时滞效应描述响应延迟")
    print("   • 加入物种功能性状差异")
    
    print("\n" + "=" * 80)
    print(" 所有图表已保存至: " + save_path)
    print("=" * 80)
    
    return model, results


if __name__ == "__main__":
    model, results = main()
