"""
============================================================
TOPSIS法 (Technique for Order Preference by Similarity to Ideal Solution)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：多属性决策，对方案进行综合排序
原理：选择距离正理想解最近、距离负理想解最远的方案
作者：MCM/ICM Team
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 图表美化设置
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class TOPSIS:
    """
    TOPSIS法类
    
    原理：
    1. 构建标准化决策矩阵
    2. 确定正理想解（最优）和负理想解（最劣）
    3. 计算各方案到正/负理想解的距离
    4. 计算相对贴近度进行排序
    
    贴近度 C = D- / (D+ + D-)
    C越接近1，方案越优
    """
    
    def __init__(self, negative_indices=None, weights=None, verbose=True):
        """
        :param negative_indices: 负向指标索引列表
        :param weights: 指标权重（可选，默认等权）
        :param verbose: 是否打印详细信息
        """
        self.negative_indices = negative_indices or []
        self.weights = weights
        self.verbose = verbose
        self.closeness = None
        self.ranking = None
        self.dist_positive = None
        self.dist_negative = None
    
    def fit(self, data):
        """
        执行TOPSIS分析
        
        :param data: DataFrame，行为方案，列为指标
        :return: 贴近度Series
        """
        # 1. 极差标准化
        data_std = (data - data.min()) / (data.max() - data.min() + 1e-10)
        
        # 2. 负向指标转正向
        for idx in self.negative_indices:
            data_std.iloc[:, idx] = 1 - data_std.iloc[:, idx]
        
        # 3. 加权（如有权重）
        if self.weights is not None:
            data_weighted = data_std * self.weights
        else:
            data_weighted = data_std
        
        # 4. 确定理想解
        ideal_positive = data_weighted.max()  # 正理想解（最优）
        ideal_negative = data_weighted.min()  # 负理想解（最劣）
        
        # 5. 计算欧氏距离
        self.dist_positive = np.sqrt(((data_weighted - ideal_positive) ** 2).sum(axis=1))
        self.dist_negative = np.sqrt(((data_weighted - ideal_negative) ** 2).sum(axis=1))
        
        # 6. 计算相对贴近度
        self.closeness = self.dist_negative / (self.dist_positive + self.dist_negative + 1e-10)
        self.ranking = self.closeness.sort_values(ascending=False)
        
        if self.verbose:
            self._print_results(data)
        
        return self.closeness
    
    def _print_results(self, data):
        """打印结果"""
        print("\n" + "="*60)
        print("📊 TOPSIS分析结果")
        print("="*60)
        print(f"\n指标数量: {data.shape[1]}")
        print(f"方案数量: {data.shape[0]}")
        print(f"负向指标: 第{[i+1 for i in self.negative_indices]}列")
        
        print(f"\n各方案距离与贴近度:")
        print(f"{'方案':<10} {'D+':>10} {'D-':>10} {'贴近度':>10}")
        print("-" * 45)
        for name in self.ranking.index:
            print(f"{name:<10} {self.dist_positive[name]:>10.4f} "
                  f"{self.dist_negative[name]:>10.4f} {self.closeness[name]:>10.4f}")
        
        print(f"\n📌 最优方案: {self.ranking.index[0]} (贴近度={self.ranking.iloc[0]:.4f})")
        print("="*60)
    
    def plot_ranking(self, save_path=None):
        """可视化排名"""
        if self.ranking is None:
            raise ValueError("请先调用fit()进行分析")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 贴近度排名
        ax1 = axes[0]
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(self.ranking)))[::-1]
        bars = ax1.barh(self.ranking.index[::-1], self.ranking.values[::-1],
                       color=colors, edgecolor='white', linewidth=2)
        ax1.set_xlabel('贴近度', fontsize=12, fontweight='bold')
        ax1.set_title('(a) 方案排名（贴近度越高越优）', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 1)
        
        for bar, val in zip(bars, self.ranking.values[::-1]):
            ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=10)
        
        # 距离对比
        ax2 = axes[1]
        x = np.arange(len(self.ranking))
        width = 0.35
        ax2.bar(x - width/2, self.dist_positive[self.ranking.index], width,
               label='D+ (距正理想解)', color='#E74C3C', alpha=0.8)
        ax2.bar(x + width/2, self.dist_negative[self.ranking.index], width,
               label='D- (距负理想解)', color='#27AE60', alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.ranking.index)
        ax2.set_ylabel('距离', fontsize=12, fontweight='bold')
        ax2.set_title('(b) 各方案到理想解的距离', fontsize=12, fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    
    # 1. 模拟数据（5方案4指标：前3正向，第4负向）
    data = pd.DataFrame({
        "方案1": [85, 92, 88, 25],
        "方案2": [90, 88, 95, 22],
        "方案3": [78, 90, 92, 28],
        "方案4": [92, 85, 86, 20],
        "方案5": [88, 95, 90, 24]
    }, index=["收益", "效率", "质量", "成本"]).T
    
    print("原始数据：")
    print(data)
    
    # 2. TOPSIS分析（等权重）
    topsis = TOPSIS(negative_indices=[3], verbose=True)
    closeness = topsis.fit(data)
    
    # 3. 可视化
    topsis.plot_ranking()
    
    # 4. 带权重的TOPSIS
    print("\n" + "="*60)
    print("📊 带权重的TOPSIS分析")
    print("="*60)
    weights = np.array([0.3, 0.25, 0.25, 0.2])  # 自定义权重
    topsis_weighted = TOPSIS(negative_indices=[3], weights=weights, verbose=True)
    closeness_weighted = topsis_weighted.fit(data)
