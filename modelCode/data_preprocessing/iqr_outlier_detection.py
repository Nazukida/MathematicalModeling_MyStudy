"""
============================================================
IQR 异常值检测 (Interquartile Range Outlier Detection)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：数据清洗、异常值识别、数据质量评估
原理：基于四分位距判断异常点
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


class OutlierDetector:
    """
    IQR异常值检测器
    
    核心公式：
    - IQR = Q3 - Q1
    - 下界 = Q1 - k * IQR
    - 上界 = Q3 + k * IQR
    - k=1.5 检测温和异常值
    - k=3.0 检测极端异常值
    
    应用场景：
    - 数据预处理
    - 传感器数据清洗
    - 金融欺诈检测
    """
    
    def __init__(self, k=1.5, verbose=True):
        """
        参数配置
        
        :param k: IQR倍数（1.5温和/3.0极端）
        :param verbose: 是否打印过程
        """
        self.k = k
        self.verbose = verbose
        self.Q1 = None
        self.Q3 = None
        self.IQR = None
        self.lower_bound = None
        self.upper_bound = None
        self.outliers = None
        self.normal_data = None
    
    def detect(self, data, column=None):
        """
        检测异常值
        
        :param data: DataFrame、Series或数组
        :param column: 列名（DataFrame时使用）
        :return: 异常值DataFrame
        """
        # 处理输入数据
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = data.columns[0]
            values = data[column].values
            df = data.copy()
        elif isinstance(data, pd.Series):
            values = data.values
            column = data.name or 'value'
            df = pd.DataFrame({column: values})
        else:
            values = np.array(data)
            column = 'value'
            df = pd.DataFrame({column: values})
        
        # 计算IQR
        self.Q1 = np.percentile(values, 25)
        self.Q3 = np.percentile(values, 75)
        self.IQR = self.Q3 - self.Q1
        self.lower_bound = self.Q1 - self.k * self.IQR
        self.upper_bound = self.Q3 + self.k * self.IQR
        
        # 标记异常值
        outlier_mask = (values < self.lower_bound) | (values > self.upper_bound)
        df['is_outlier'] = outlier_mask
        
        self.outliers = df[outlier_mask]
        self.normal_data = df[~outlier_mask]
        
        if self.verbose:
            self._print_results(column, len(values))
        
        return self.outliers
    
    def _print_results(self, column, total):
        """打印结果"""
        print("\n" + "="*50)
        print("🔍 IQR 异常值检测结果")
        print("="*50)
        print(f"\n  检测列: {column}")
        print(f"  IQR倍数: k = {self.k}")
        print(f"\n  统计信息:")
        print(f"    Q1 (25%): {self.Q1:.2f}")
        print(f"    Q3 (75%): {self.Q3:.2f}")
        print(f"    IQR: {self.IQR:.2f}")
        print(f"\n  正常范围: [{self.lower_bound:.2f}, {self.upper_bound:.2f}]")
        print(f"\n  检测结果:")
        print(f"    总样本数: {total}")
        print(f"    异常值数: {len(self.outliers)}")
        print(f"    异常比例: {len(self.outliers)/total*100:.1f}%")
        
        if len(self.outliers) > 0:
            print(f"\n  异常值详情:")
            for idx, row in self.outliers.iterrows():
                val = row.iloc[0]
                direction = "过高" if val > self.upper_bound else "过低"
                print(f"    索引{idx}: {val:.2f} ({direction})")
        print("="*50)
    
    def remove_outliers(self, data, column=None):
        """移除异常值并返回清洗后数据"""
        self.detect(data, column)
        return self.normal_data.drop(columns=['is_outlier'])
    
    def replace_outliers(self, data, column=None, method='median'):
        """
        替换异常值
        
        :param method: 'median'/'mean'/'clip'
        """
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = data.columns[0]
            df = data.copy()
        else:
            column = 'value'
            df = pd.DataFrame({column: data})
        
        self.detect(df, column)
        
        if method == 'median':
            replacement = df[column].median()
        elif method == 'mean':
            replacement = df[~df['is_outlier']][column].mean()
        elif method == 'clip':
            df[column] = df[column].clip(self.lower_bound, self.upper_bound)
            return df.drop(columns=['is_outlier'], errors='ignore')
        
        df.loc[df['is_outlier'], column] = replacement
        return df.drop(columns=['is_outlier'], errors='ignore')
    
    def plot_boxplot(self, data, column=None, save_path=None):
        """箱线图可视化"""
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = data.columns[0]
            values = data[column].values
        else:
            values = np.array(data)
            column = 'value'
        
        self.detect(data, column)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：箱线图
        ax1 = axes[0]
        bp = ax1.boxplot(values, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('#2E86AB')
        bp['boxes'][0].set_alpha(0.7)
        
        # 标记异常值
        if len(self.outliers) > 0:
            outlier_vals = self.outliers.iloc[:, 0].values
            ax1.scatter([1]*len(outlier_vals), outlier_vals, color='#E94F37',
                       s=100, zorder=5, marker='x', linewidth=2, label='异常值')
        
        ax1.set_ylabel(column, fontsize=12, fontweight='bold')
        ax1.set_title('箱线图 (Box Plot)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：分布图
        ax2 = axes[1]
        ax2.hist(values, bins=30, color='#2E86AB', alpha=0.7, 
                edgecolor='white', linewidth=1, label='数据分布')
        ax2.axvline(self.lower_bound, color='#E94F37', linestyle='--',
                   linewidth=2, label=f'下界={self.lower_bound:.1f}')
        ax2.axvline(self.upper_bound, color='#E94F37', linestyle='--',
                   linewidth=2, label=f'上界={self.upper_bound:.1f}')
        ax2.axvline(self.Q1, color='green', linestyle=':',
                   linewidth=2, label=f'Q1={self.Q1:.1f}')
        ax2.axvline(self.Q3, color='green', linestyle=':',
                   linewidth=2, label=f'Q3={self.Q3:.1f}')
        
        ax2.set_xlabel(column, fontsize=12, fontweight='bold')
        ax2.set_ylabel('频数', fontsize=12, fontweight='bold')
        ax2.set_title('数据分布与异常边界', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("   IQR 异常值检测演示")
    print("="*60)
    
    # 1. 模拟数据（含异常值的销量数据）
    np.random.seed(42)
    normal_data = np.random.normal(500, 50, 95)  # 正常数据
    outliers = np.array([1200, 80, 1100, 50, 1300])  # 异常值
    sales = np.concatenate([normal_data, outliers])
    data = pd.DataFrame({"销量": sales})
    
    print("\n原始数据概览：")
    print(data.describe().round(2))
    
    # 2. 异常值检测
    detector = OutlierDetector(k=1.5, verbose=True)
    outliers_detected = detector.detect(data, column="销量")
    
    # 3. 可视化
    detector.plot_boxplot(data, column="销量")
    
    # 4. 数据清洗选项
    print("\n【数据清洗选项】")
    
    # 方法1：移除异常值
    clean_data = detector.remove_outliers(data, column="销量")
    print(f"移除后样本数: {len(clean_data)}")
    
    # 方法2：用中位数替换
    replaced_data = detector.replace_outliers(data, column="销量", method='median')
    print(f"替换后最大值: {replaced_data['销量'].max():.2f}")
    
    # 方法3：截断到边界
    clipped_data = detector.replace_outliers(data, column="销量", method='clip')
    print(f"截断后范围: [{clipped_data['销量'].min():.2f}, {clipped_data['销量'].max():.2f}]")
