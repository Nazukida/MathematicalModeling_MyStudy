"""
============================================================
熵权法 (Entropy Weight Method)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：客观确定指标权重，避免主观偏差
原理：信息熵越小，指标差异越大，权重越高
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


class EntropyWeightMethod:
    """
    熵权法类
    
    原理：
    1. 指标数据标准化
    2. 计算各指标的信息熵
    3. 熵值越小 → 差异越大 → 权重越高
    
    参数说明：
    - negative_indices: 负向指标的列索引（如成本，越小越好）
    """
    
    def __init__(self, negative_indices=None, verbose=True):
        """
        :param negative_indices: 负向指标索引列表，如[3]表示第4列是负向指标
        :param verbose: 是否打印详细信息
        """
        self.negative_indices = negative_indices or []
        self.verbose = verbose
        self.weights = None
        self.entropy = None
        self.data_normalized = None
    
    def fit(self, data):
        """
        计算权重
        
        :param data: DataFrame，行为方案，列为指标
        :return: 权重Series
        """
        # 1. 极差标准化到[0,1]
        data_std = (data - data.min()) / (data.max() - data.min() + 1e-10)
        
        # 2. 负向指标转正向（成本类：值越小越好）
        for idx in self.negative_indices:
            data_std.iloc[:, idx] = 1 - data_std.iloc[:, idx]
        
        self.data_normalized = data_std
        
        # 3. 计算熵值
        n, m = data_std.shape  # n=方案数, m=指标数
        p = data_std / (data_std.sum(axis=0) + 1e-10)  # 比重矩阵
        p = np.where(p == 0, 1e-10, p)  # 避免log(0)
        
        # 熵值公式: E = -1/ln(n) * Σ(p*ln(p))
        self.entropy = -(1 / np.log(n)) * (p * np.log(p)).sum(axis=0)
        
        # 4. 计算权重（差异系数法）
        diff_coef = 1 - self.entropy  # 差异系数
        self.weights = diff_coef / diff_coef.sum()
        
        if self.verbose:
            self._print_results(data)
        
        return self.weights
    
    def _print_results(self, data):
        """打印结果"""
        print("\n" + "="*50)
        print("📊 熵权法计算结果")
        print("="*50)
        print(f"\n指标名称: {list(data.columns)}")
        print(f"负向指标: 第{[i+1 for i in self.negative_indices]}列")
        print(f"\n各指标熵值:")
        for i, (col, e) in enumerate(zip(data.columns, self.entropy)):
            print(f"  {col}: {e:.4f}")
        print(f"\n各指标权重:")
        for col, w in zip(data.columns, self.weights):
            print(f"  {col}: {w:.4f}")
        print(f"\n权重总和: {self.weights.sum():.4f}")
        print("="*50)
    
    def plot_weights(self, save_path=None):
        """可视化权重分布"""
        if self.weights is None:
            raise ValueError("请先调用fit()计算权重")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(self.weights)))
        
        bars = ax.bar(self.weights.index, self.weights.values, 
                     color=colors, edgecolor='white', linewidth=2)
        
        # 添加数值标签
        for bar, w in zip(bars, self.weights.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{w:.4f}', ha='center', va='bottom', fontsize=11)
        
        ax.set_xlabel('指标', fontsize=12, fontweight='bold')
        ax.set_ylabel('权重', fontsize=12, fontweight='bold')
        ax.set_title('熵权法指标权重分布', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(self.weights.values) * 1.2)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    
    # 1. 模拟数据（5方案4指标：收益、效率、质量为正向，成本为负向）
    data = pd.DataFrame({
        "方案1": [85, 92, 88, 25],
        "方案2": [90, 88, 95, 22],
        "方案3": [78, 90, 92, 28],
        "方案4": [92, 85, 86, 20],
        "方案5": [88, 95, 90, 24]
    }, index=["收益", "效率", "质量", "成本"]).T
    
    print("原始数据：")
    print(data)
    
    # 2. 熵权法计算
    ewm = EntropyWeightMethod(negative_indices=[3], verbose=True)
    weights = ewm.fit(data)
    
    # 3. 可视化
    ewm.plot_weights()
    
    # 4. 综合得分计算
    scores = (ewm.data_normalized * weights).sum(axis=1)
    print("\n📊 各方案综合得分（加权求和）：")
    print(scores.sort_values(ascending=False).round(4))
