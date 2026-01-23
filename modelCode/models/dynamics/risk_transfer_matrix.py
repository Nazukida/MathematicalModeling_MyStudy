# -*- coding: utf-8 -*-
"""
风险转移矩阵模型 (Risk Transfer Matrix Model)
==============================================
用于模拟执法干预后风险在地区网络中的转移效应。

核心功能：
1. 构建地区网络（邻接矩阵）
2. 计算转移概率矩阵
3. 模拟执法干预后的风险重分布
4. 识别替代热点
5. 情景分析（低/中/高位移）

数学模型：
---------
1. 执法冲击强度: E_u = η * x_u
2. 转移概率: P_{u→v} = A_{uv} * exp(-θ*E_v) / Σ_k A_{uk} * exp(-θ*E_k)
3. 风险更新: Risk_v^{after} = (1-ρ)*Risk_v^{before} + ρ*Σ_u Risk_u^{before}*P_{u→v} - κ*E_v

参数说明：
---------
- η (eta): 执法强度尺度参数
- θ (theta): 对执法强度的敏感程度
- ρ (rho): 迁移比例 [0,1]
- κ (kappa): 执法削减效应（可选）

作者: 美赛建模团队
版本: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class ScenarioConfig:
    """情景配置类"""
    name: str
    rho: float      # 迁移比例
    theta: float    # 执法敏感度
    eta: float = 0.5    # 执法强度尺度
    kappa: float = 0.0  # 削减效应
    
    def __repr__(self):
        return f"Scenario({self.name}: ρ={self.rho}, θ={self.theta}, η={self.eta}, κ={self.kappa})"


# 预定义情景
SCENARIOS = {
    'low': ScenarioConfig('Low Displacement', rho=0.1, theta=1.0, eta=0.3),
    'medium': ScenarioConfig('Medium Displacement', rho=0.3, theta=2.0, eta=0.5),
    'high': ScenarioConfig('High Displacement', rho=0.5, theta=3.0, eta=0.7),
}


class RiskTransferMatrix:
    """
    风险转移矩阵模型
    
    用于模拟执法干预后风险在地区网络中的转移效应。
    
    Parameters:
    -----------
    adjacency_matrix : np.ndarray
        邻接矩阵 A，shape=(n, n)，A[i,j]=1 表示地区i和j相邻
    region_names : List[str], optional
        地区名称列表
    
    Attributes:
    -----------
    n_regions : int
        地区数量
    G : networkx.Graph
        地区网络图
    
    Examples:
    ---------
    >>> # 创建简单的5地区网络
    >>> adj = np.array([
    ...     [0, 1, 1, 0, 0],
    ...     [1, 0, 1, 1, 0],
    ...     [1, 1, 0, 1, 1],
    ...     [0, 1, 1, 0, 1],
    ...     [0, 0, 1, 1, 0]
    ... ])
    >>> regions = ['A', 'B', 'C', 'D', 'E']
    >>> model = RiskTransferMatrix(adj, regions)
    >>> 
    >>> # 初始风险和执法决策
    >>> risk_before = np.array([0.8, 0.6, 0.4, 0.3, 0.2])
    >>> enforcement = np.array([1, 1, 0, 0, 0])  # 对A、B执法
    >>> 
    >>> # 模拟风险转移
    >>> result = model.simulate_transfer(risk_before, enforcement, scenario='medium')
    """
    
    def __init__(self, 
                 adjacency_matrix: np.ndarray,
                 region_names: Optional[List[str]] = None):
        """初始化风险转移矩阵模型"""
        
        self.A = np.array(adjacency_matrix, dtype=float)
        self.n_regions = self.A.shape[0]
        
        # 验证邻接矩阵
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("邻接矩阵必须是方阵")
        
        # 设置地区名称
        if region_names is None:
            self.region_names = [f"Region_{i+1}" for i in range(self.n_regions)]
        else:
            if len(region_names) != self.n_regions:
                raise ValueError(f"地区名称数量({len(region_names)})必须与矩阵维度({self.n_regions})一致")
            self.region_names = list(region_names)
        
        # 构建网络图
        self.G = self._build_network_graph()
        
        # 存储历史记录
        self.history = []
        
        print(f"风险转移矩阵模型初始化完成")
        print(f"  地区数量: {self.n_regions}")
        print(f"  边（相邻关系）数量: {int(np.sum(self.A) / 2)}")
    
    def _build_network_graph(self) -> nx.Graph:
        """从邻接矩阵构建NetworkX图"""
        G = nx.Graph()
        G.add_nodes_from(range(self.n_regions))
        
        for i in range(self.n_regions):
            for j in range(i+1, self.n_regions):
                if self.A[i, j] > 0:
                    G.add_edge(i, j, weight=self.A[i, j])
        
        return G
    
    def compute_enforcement_intensity(self, 
                                       enforcement: np.ndarray, 
                                       eta: float = 0.5) -> np.ndarray:
        """
        计算执法冲击强度
        
        公式: E_u = η * x_u
        
        Parameters:
        -----------
        enforcement : np.ndarray
            执法决策向量 x，shape=(n,)，x[u]=1 表示对地区u执法
        eta : float
            执法强度尺度参数
            
        Returns:
        --------
        E : np.ndarray
            执法冲击强度向量
        """
        return eta * np.array(enforcement, dtype=float)
    
    def compute_transfer_probability(self, 
                                      E: np.ndarray, 
                                      theta: float = 2.0) -> np.ndarray:
        """
        计算转移概率矩阵
        
        公式: P_{u→v} = A_{uv} * exp(-θ*E_v) / Σ_k A_{uk} * exp(-θ*E_k)
        
        直觉：走私者更倾向去"邻近且执法弱"的地方
        
        Parameters:
        -----------
        E : np.ndarray
            执法冲击强度向量
        theta : float
            对执法强度的敏感程度参数
            
        Returns:
        --------
        P : np.ndarray
            转移概率矩阵 shape=(n, n)，P[u,v] 表示从u转移到v的概率
        """
        n = self.n_regions
        P = np.zeros((n, n))
        
        # exp(-θ*E_v)：执法越强，吸引力越低
        attraction = np.exp(-theta * E)
        
        for u in range(n):
            # 对每个源地区u，计算转移到各邻居v的概率
            neighbors_mask = self.A[u] > 0
            
            if np.any(neighbors_mask):
                # 加权吸引力 = 邻接权重 * exp(-θ*E_v)
                weighted_attraction = self.A[u] * attraction
                
                # 归一化得到概率
                total = np.sum(weighted_attraction)
                if total > 0:
                    P[u] = weighted_attraction / total
        
        return P
    
    def simulate_transfer(self,
                          risk_before: np.ndarray,
                          enforcement: np.ndarray,
                          scenario: Union[str, ScenarioConfig] = 'medium',
                          conserve_risk: bool = True) -> Dict:
        """
        模拟执法干预后的风险转移
        
        核心公式（风险守恒版）：
        Risk_v^{after} = (1-ρ)*Risk_v^{before} + ρ*Σ_u Risk_u^{before}*P_{u→v}
        
        核心公式（带削减版）：
        Risk_v^{after} = (1-ρ)*Risk_v^{before} + ρ*Σ_u Risk_u^{before}*P_{u→v} - κ*E_v
        
        Parameters:
        -----------
        risk_before : np.ndarray
            干预前风险分布向量
        enforcement : np.ndarray
            执法决策向量（0/1或连续值）
        scenario : str or ScenarioConfig
            情景配置，可以是 'low', 'medium', 'high' 或自定义配置
        conserve_risk : bool
            是否完全守恒风险（True则忽略kappa参数）
            
        Returns:
        --------
        result : dict
            包含：
            - risk_after: 干预后风险分布
            - risk_change: 风险变化量
            - risk_change_pct: 风险变化百分比
            - transfer_matrix: 转移概率矩阵
            - enforcement_intensity: 执法强度
            - substitution_hotspots: 替代热点（风险增长>30%的地区）
            - scenario: 使用的情景配置
        """
        # 解析情景配置
        if isinstance(scenario, str):
            if scenario not in SCENARIOS:
                raise ValueError(f"未知情景: {scenario}. 可选: {list(SCENARIOS.keys())}")
            config = SCENARIOS[scenario]
        else:
            config = scenario
        
        risk_before = np.array(risk_before, dtype=float)
        enforcement = np.array(enforcement, dtype=float)
        
        # 1. 计算执法冲击强度
        E = self.compute_enforcement_intensity(enforcement, config.eta)
        
        # 2. 计算转移概率矩阵
        P = self.compute_transfer_probability(E, config.theta)
        
        # 3. 计算风险转移
        rho = config.rho
        kappa = 0.0 if conserve_risk else config.kappa
        
        # 风险更新公式
        # 保留部分 + 转入部分 - 削减部分
        stay_component = (1 - rho) * risk_before
        transfer_component = rho * (risk_before @ P)  # 矩阵乘法：Σ_u Risk_u * P_{u→v}
        reduction_component = kappa * E
        
        risk_after = stay_component + transfer_component - reduction_component
        
        # 确保风险非负
        risk_after = np.maximum(risk_after, 0)
        
        # 4. 计算变化
        risk_change = risk_after - risk_before
        
        # 避免除零
        with np.errstate(divide='ignore', invalid='ignore'):
            risk_change_pct = np.where(
                risk_before > 1e-10,
                (risk_after - risk_before) / risk_before,
                0
            )
        
        # 5. 识别替代热点（风险增长超过30%）
        threshold = 0.3
        substitution_hotspots = np.where(risk_change_pct > threshold)[0]
        
        result = {
            'risk_before': risk_before,
            'risk_after': risk_after,
            'risk_change': risk_change,
            'risk_change_pct': risk_change_pct,
            'transfer_matrix': P,
            'enforcement_intensity': E,
            'substitution_hotspots': substitution_hotspots,
            'scenario': config,
            'total_risk_before': np.sum(risk_before),
            'total_risk_after': np.sum(risk_after),
        }
        
        # 记录历史
        self.history.append(result.copy())
        
        return result
    
    def multi_scenario_analysis(self,
                                 risk_before: np.ndarray,
                                 enforcement: np.ndarray,
                                 scenarios: Optional[List[str]] = None) -> pd.DataFrame:
        """
        多情景对比分析
        
        Parameters:
        -----------
        risk_before : np.ndarray
            初始风险分布
        enforcement : np.ndarray
            执法决策
        scenarios : List[str], optional
            要分析的情景列表，默认为 ['low', 'medium', 'high']
            
        Returns:
        --------
        df : pd.DataFrame
            各情景下的风险变化对比表
        """
        if scenarios is None:
            scenarios = ['low', 'medium', 'high']
        
        results = []
        for scenario_name in scenarios:
            result = self.simulate_transfer(risk_before, enforcement, scenario_name)
            
            for i, region in enumerate(self.region_names):
                results.append({
                    'Scenario': scenario_name,
                    'Region': region,
                    'Risk_Before': result['risk_before'][i],
                    'Risk_After': result['risk_after'][i],
                    'Risk_Change': result['risk_change'][i],
                    'Risk_Change_Pct': result['risk_change_pct'][i] * 100,
                    'Is_Hotspot': i in result['substitution_hotspots'],
                    'Enforcement': enforcement[i]
                })
        
        df = pd.DataFrame(results)
        return df
    
    def adaptive_enforcement_cycle(self,
                                    initial_risk: np.ndarray,
                                    initial_enforcement: np.ndarray,
                                    n_cycles: int = 6,
                                    scenario: Union[str, ScenarioConfig] = 'medium',
                                    top_k: int = 3) -> Dict:
        """
        自适应执法更新循环
        
        模拟多轮执法-转移-调整的动态过程
        
        Parameters:
        -----------
        initial_risk : np.ndarray
            初始风险分布
        initial_enforcement : np.ndarray
            初始执法配置
        n_cycles : int
            模拟轮数
        scenario : str or ScenarioConfig
            情景配置
        top_k : int
            每轮重点执法的地区数量
            
        Returns:
        --------
        history : dict
            包含每轮的风险分布和执法配置
        """
        risk = np.array(initial_risk, dtype=float)
        enforcement = np.array(initial_enforcement, dtype=float)
        
        cycle_history = {
            'risks': [risk.copy()],
            'enforcements': [enforcement.copy()],
            'hotspots': [],
            'total_risks': [np.sum(risk)]
        }
        
        print(f"\n{'='*60}")
        print(f"自适应执法循环模拟 (共 {n_cycles} 轮)")
        print(f"{'='*60}")
        
        for cycle in range(n_cycles):
            print(f"\n--- 第 {cycle + 1} 轮 ---")
            
            # 1. 模拟风险转移
            result = self.simulate_transfer(risk, enforcement, scenario, conserve_risk=False)
            
            # 2. 更新风险分布
            risk = result['risk_after']
            
            # 3. 识别新热点，调整执法配置
            hotspots = result['substitution_hotspots']
            print(f"  替代热点: {[self.region_names[h] for h in hotspots]}")
            
            # 4. 自适应更新：对风险最高的top_k地区执法
            top_risk_regions = np.argsort(risk)[-top_k:]
            new_enforcement = np.zeros(self.n_regions)
            new_enforcement[top_risk_regions] = 1
            
            print(f"  新执法重点: {[self.region_names[r] for r in top_risk_regions]}")
            print(f"  总风险: {np.sum(risk):.4f} (变化: {np.sum(result['risk_change']):.4f})")
            
            enforcement = new_enforcement
            
            # 5. 记录历史
            cycle_history['risks'].append(risk.copy())
            cycle_history['enforcements'].append(enforcement.copy())
            cycle_history['hotspots'].append(hotspots)
            cycle_history['total_risks'].append(np.sum(risk))
        
        return cycle_history
    
    def get_cross_regional_recommendations(self,
                                            result: Dict,
                                            threshold: float = 0.3) -> List[Dict]:
        """
        生成跨区域协作建议
        
        Parameters:
        -----------
        result : dict
            simulate_transfer 的返回结果
        threshold : float
            风险增长阈值
            
        Returns:
        --------
        recommendations : List[dict]
            协作建议列表
        """
        recommendations = []
        P = result['transfer_matrix']
        risk_change_pct = result['risk_change_pct']
        
        for v in range(self.n_regions):
            if risk_change_pct[v] > threshold:
                # 找出主要风险来源地区
                sources = []
                for u in range(self.n_regions):
                    if P[u, v] > 0.1:  # 转移概率>10%的来源
                        sources.append({
                            'region': self.region_names[u],
                            'transfer_prob': P[u, v]
                        })
                
                if sources:
                    recommendations.append({
                        'target_region': self.region_names[v],
                        'risk_increase': f"{risk_change_pct[v]*100:.1f}%",
                        'main_sources': sources,
                        'recommendation': f"建议与{', '.join([s['region'] for s in sources])}建立联合执法机制"
                    })
        
        return recommendations
    
    # ==================== 可视化方法 ====================
    
    def plot_network(self, 
                     risk: Optional[np.ndarray] = None,
                     enforcement: Optional[np.ndarray] = None,
                     title: str = "地区网络图",
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        绘制地区网络图
        
        Parameters:
        -----------
        risk : np.ndarray, optional
            风险分布（用于节点大小和颜色）
        enforcement : np.ndarray, optional
            执法配置（用于节点边框）
        title : str
            图标题
        save_path : str, optional
            保存路径
        figsize : tuple
            图尺寸
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 布局
        pos = nx.spring_layout(self.G, k=2, iterations=50, seed=42)
        
        # 节点大小和颜色
        if risk is not None:
            node_sizes = 300 + 700 * (risk / np.max(risk) if np.max(risk) > 0 else risk)
            node_colors = risk
        else:
            node_sizes = 500
            node_colors = 'lightblue'
        
        # 节点边框（执法地区用红色粗边框）
        if enforcement is not None:
            edgecolors = ['red' if e > 0 else 'black' for e in enforcement]
            linewidths = [3 if e > 0 else 1 for e in enforcement]
        else:
            edgecolors = 'black'
            linewidths = 1
        
        # 绘制边
        nx.draw_networkx_edges(self.G, pos, ax=ax, alpha=0.5, width=1.5)
        
        # 绘制节点
        nodes = nx.draw_networkx_nodes(self.G, pos, ax=ax,
                                        node_size=node_sizes,
                                        node_color=node_colors,
                                        cmap=plt.cm.YlOrRd,
                                        edgecolors=edgecolors,
                                        linewidths=linewidths)
        
        if risk is not None:
            plt.colorbar(nodes, ax=ax, label='风险水平')
        
        # 绘制标签
        labels = {i: name for i, name in enumerate(self.region_names)}
        nx.draw_networkx_labels(self.G, pos, labels, ax=ax, font_size=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存至: {save_path}")
        
        return fig
    
    def plot_transfer_heatmap(self,
                               P: np.ndarray,
                               title: str = "转移概率矩阵",
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        绘制转移概率矩阵热力图
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        import seaborn as sns
        sns.heatmap(P, 
                    annot=True, 
                    fmt='.2f',
                    cmap='YlOrRd',
                    xticklabels=self.region_names,
                    yticklabels=self.region_names,
                    ax=ax,
                    cbar_kws={'label': '转移概率'})
        
        ax.set_xlabel('目标地区 (v)', fontsize=12)
        ax.set_ylabel('源地区 (u)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_risk_comparison(self,
                              result: Dict,
                              title: str = "风险分布对比",
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        绘制干预前后风险对比图
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        x = np.arange(self.n_regions)
        width = 0.35
        
        # 子图1：条形图对比
        ax1 = axes[0]
        bars1 = ax1.bar(x - width/2, result['risk_before'], width, 
                        label='干预前', color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, result['risk_after'], width, 
                        label='干预后', color='coral', alpha=0.8)
        
        # 标记替代热点
        for h in result['substitution_hotspots']:
            ax1.annotate('⚠', (h + width/2, result['risk_after'][h]), 
                        ha='center', fontsize=14, color='red')
        
        ax1.set_xlabel('地区')
        ax1.set_ylabel('风险水平')
        ax1.set_title('风险分布对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.region_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 子图2：变化百分比
        ax2 = axes[1]
        colors = ['red' if pct > 0 else 'green' for pct in result['risk_change_pct']]
        bars3 = ax2.bar(x, result['risk_change_pct'] * 100, color=colors, alpha=0.8)
        
        ax2.axhline(y=30, color='red', linestyle='--', linewidth=1, label='热点阈值(30%)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax2.set_xlabel('地区')
        ax2.set_ylabel('风险变化 (%)')
        ax2.set_title('风险变化百分比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.region_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_scenario_comparison(self,
                                  df: pd.DataFrame,
                                  title: str = "多情景风险变化对比",
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """
        绘制多情景对比图
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        import seaborn as sns
        
        # 子图1：风险变化对比
        ax1 = axes[0]
        pivot_change = df.pivot(index='Region', columns='Scenario', values='Risk_Change_Pct')
        pivot_change.plot(kind='bar', ax=ax1, alpha=0.8)
        ax1.axhline(y=30, color='red', linestyle='--', linewidth=1, label='热点阈值')
        ax1.set_xlabel('地区')
        ax1.set_ylabel('风险变化 (%)')
        ax1.set_title('各情景下的风险变化百分比')
        ax1.legend(title='情景')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # 子图2：热力图
        ax2 = axes[1]
        pivot_after = df.pivot(index='Region', columns='Scenario', values='Risk_After')
        sns.heatmap(pivot_after, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2,
                   cbar_kws={'label': '风险水平'})
        ax2.set_title('各情景下的风险分布')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_adaptive_cycle(self,
                             cycle_history: Dict,
                             title: str = "自适应执法循环",
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        绘制自适应循环过程图
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        risks = np.array(cycle_history['risks'])
        n_cycles = len(risks) - 1
        
        # 子图1：总风险变化趋势
        ax1 = axes[0, 0]
        ax1.plot(range(n_cycles + 1), cycle_history['total_risks'], 
                'o-', linewidth=2, markersize=8, color='steelblue')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('总风险')
        ax1.set_title('总风险变化趋势')
        ax1.grid(alpha=0.3)
        
        # 子图2：各地区风险演化
        ax2 = axes[0, 1]
        for i, name in enumerate(self.region_names):
            ax2.plot(range(n_cycles + 1), risks[:, i], 'o-', label=name, markersize=4)
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('风险水平')
        ax2.set_title('各地区风险演化')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(alpha=0.3)
        
        # 子图3：风险分布热力图
        ax3 = axes[1, 0]
        import seaborn as sns
        sns.heatmap(risks.T, 
                    annot=True if self.n_regions <= 8 else False,
                    fmt='.2f',
                    cmap='YlOrRd',
                    xticklabels=[f'T{i}' for i in range(n_cycles + 1)],
                    yticklabels=self.region_names,
                    ax=ax3,
                    cbar_kws={'label': '风险'})
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('地区')
        ax3.set_title('风险时空演化热力图')
        
        # 子图4：执法配置变化
        ax4 = axes[1, 1]
        enforcements = np.array(cycle_history['enforcements'])
        sns.heatmap(enforcements.T,
                    cmap='Reds',
                    xticklabels=[f'T{i}' for i in range(n_cycles + 1)],
                    yticklabels=self.region_names,
                    ax=ax4,
                    cbar_kws={'label': '执法强度'})
        ax4.set_xlabel('时间步')
        ax4.set_ylabel('地区')
        ax4.set_title('执法配置变化')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# ==================== 便捷函数 ====================

def create_grid_network(rows: int, cols: int, 
                        region_prefix: str = "R") -> Tuple[np.ndarray, List[str]]:
    """
    创建网格状地区网络
    
    Parameters:
    -----------
    rows : int
        行数
    cols : int
        列数
    region_prefix : str
        地区名称前缀
        
    Returns:
    --------
    adj : np.ndarray
        邻接矩阵
    names : List[str]
        地区名称列表
    """
    n = rows * cols
    adj = np.zeros((n, n))
    names = []
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            names.append(f"{region_prefix}{i+1}{j+1}")
            
            # 上邻居
            if i > 0:
                adj[idx, (i-1)*cols + j] = 1
                adj[(i-1)*cols + j, idx] = 1
            # 左邻居
            if j > 0:
                adj[idx, i*cols + (j-1)] = 1
                adj[i*cols + (j-1), idx] = 1
    
    return adj, names


def create_random_network(n_regions: int, 
                          edge_probability: float = 0.3,
                          seed: int = 42) -> np.ndarray:
    """
    创建随机地区网络
    
    Parameters:
    -----------
    n_regions : int
        地区数量
    edge_probability : float
        边的概率
    seed : int
        随机种子
        
    Returns:
    --------
    adj : np.ndarray
        邻接矩阵
    """
    np.random.seed(seed)
    adj = np.zeros((n_regions, n_regions))
    
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            if np.random.random() < edge_probability:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # 确保图是连通的
    G = nx.from_numpy_array(adj)
    if not nx.is_connected(G):
        # 添加边使图连通
        components = list(nx.connected_components(G))
        for k in range(len(components) - 1):
            node1 = list(components[k])[0]
            node2 = list(components[k+1])[0]
            adj[node1, node2] = 1
            adj[node2, node1] = 1
    
    return adj


# ==================== 示例演示 ====================

def demo_risk_transfer():
    """
    风险转移矩阵模型演示
    """
    print("=" * 60)
    print("风险转移矩阵模型 - 完整演示")
    print("=" * 60)
    
    # 1. 创建地区网络（模拟边境地区）
    print("\n[1] 创建地区网络")
    regions = ['北部边境', '东北地区', '东部港口', '中部枢纽', 
               '西部山区', '南部边境', '东南沿海']
    n = len(regions)
    
    # 手动定义邻接关系
    adj = np.array([
        [0, 1, 0, 1, 1, 0, 0],  # 北部边境
        [1, 0, 1, 1, 0, 0, 0],  # 东北地区
        [0, 1, 0, 1, 0, 0, 1],  # 东部港口
        [1, 1, 1, 0, 1, 1, 1],  # 中部枢纽（与所有区域相邻）
        [1, 0, 0, 1, 0, 1, 0],  # 西部山区
        [0, 0, 0, 1, 1, 0, 1],  # 南部边境
        [0, 0, 1, 1, 0, 1, 0],  # 东南沿海
    ])
    
    # 初始化模型
    model = RiskTransferMatrix(adj, regions)
    
    # 2. 设置初始风险和执法配置
    print("\n[2] 设置初始条件")
    risk_before = np.array([0.8, 0.6, 0.5, 0.4, 0.3, 0.7, 0.5])
    print(f"初始风险分布: {dict(zip(regions, risk_before))}")
    
    # 对北部边境和南部边境执法
    enforcement = np.array([1, 0, 0, 0, 0, 1, 0])
    print(f"执法配置: 对 {[regions[i] for i in np.where(enforcement>0)[0]]} 加强执法")
    
    # 3. 单情景模拟
    print("\n[3] 单情景模拟 (Medium Displacement)")
    result = model.simulate_transfer(risk_before, enforcement, 'medium')
    
    print(f"\n转移概率矩阵 P:")
    print(pd.DataFrame(result['transfer_matrix'], 
                       index=regions, columns=regions).round(3))
    
    print(f"\n风险变化:")
    for i, region in enumerate(regions):
        change_pct = result['risk_change_pct'][i] * 100
        symbol = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "→"
        hotspot = " ⚠️热点" if i in result['substitution_hotspots'] else ""
        print(f"  {region}: {result['risk_before'][i]:.3f} → {result['risk_after'][i]:.3f} "
              f"({symbol}{abs(change_pct):.1f}%){hotspot}")
    
    print(f"\n总风险: {result['total_risk_before']:.3f} → {result['total_risk_after']:.3f}")
    
    # 4. 多情景对比
    print("\n[4] 多情景对比分析")
    df = model.multi_scenario_analysis(risk_before, enforcement)
    
    print("\n各情景下的替代热点:")
    for scenario in ['low', 'medium', 'high']:
        hotspots = df[(df['Scenario']==scenario) & (df['Is_Hotspot'])]['Region'].tolist()
        print(f"  {scenario}: {hotspots if hotspots else '无'}")
    
    # 5. 跨区域协作建议
    print("\n[5] 跨区域协作建议")
    recommendations = model.get_cross_regional_recommendations(result)
    for rec in recommendations:
        print(f"  目标地区: {rec['target_region']}")
        print(f"    风险增长: {rec['risk_increase']}")
        print(f"    建议: {rec['recommendation']}")
    
    # 6. 自适应执法循环
    print("\n[6] 自适应执法循环模拟")
    cycle_history = model.adaptive_enforcement_cycle(
        risk_before, enforcement, 
        n_cycles=4, 
        scenario='medium',
        top_k=2
    )
    
    # 7. 可视化（如果可能）
    print("\n[7] 生成可视化图表")
    try:
        fig1 = model.plot_network(risk_before, enforcement, "初始风险分布")
        fig2 = model.plot_transfer_heatmap(result['transfer_matrix'], "转移概率矩阵")
        fig3 = model.plot_risk_comparison(result, "干预前后风险对比")
        fig4 = model.plot_scenario_comparison(df, "多情景对比分析")
        fig5 = model.plot_adaptive_cycle(cycle_history, "自适应执法循环")
        plt.show()
    except Exception as e:
        print(f"  可视化跳过: {e}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    
    return model, result, df, cycle_history


if __name__ == "__main__":
    model, result, df, cycle_history = demo_risk_transfer()
