"""
============================================================
综合评价模型完整教程 (Comprehensive Evaluation Tutorial)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
本教程展示如何将数据预处理、赋权模型、综合评价、可视化完整串联起来

包含内容：
1. 数据预处理模块 (Data Preprocessing)
2. 赋权方法 (Weighting Methods)
   - 熵权法 (Entropy Weight)
   - CRITIC法 (CRITIC Method)
   - 组合赋权法 (Combined Weighting)
3. 综合评价方法 (Evaluation Methods)
   - TOPSIS法 (TOPSIS)
   - 灰色关联分析 (Grey Relational Analysis)
4. 可视化模块 (Visualization)
5. 灵敏度分析 (Sensitivity Analysis)
6. 完整案例演示

作者：MCM/ICM Team
日期：2026年1月20日
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


# ============================================================
# 第一部分：完整工作流程概览
# ============================================================

def print_workflow():
    """打印完整工作流程"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              综合评价模型完整工作流程 (Complete Workflow)                  ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                          ║
    ║   ┌─────────────────────────────────────────────────────────────────┐    ║
    ║   │  Step 1: 数据准备 (Data Preparation)                            │    ║
    ║   │  ├─ 从CSV/DataFrame/数组加载数据                                 │    ║
    ║   │  ├─ 确定评价对象和评价指标                                       │    ║
    ║   │  └─ 确定指标类型（正向/负向）                                    │    ║
    ║   └─────────────────────────────────────────────────────────────────┘    ║
    ║                            ↓                                             ║
    ║   ┌─────────────────────────────────────────────────────────────────┐    ║
    ║   │  Step 2: 数据预处理 (Data Preprocessing)                        │    ║
    ║   │  ├─ 缺失值处理（均值/中值填充）                                  │    ║
    ║   │  ├─ 异常值检测与处理                                             │    ║
    ║   │  └─ 数据标准化（极差法/Z-score）                                 │    ║
    ║   └─────────────────────────────────────────────────────────────────┘    ║
    ║                            ↓                                             ║
    ║   ┌─────────────────────────────────────────────────────────────────┐    ║
    ║   │  Step 3: 确定权重 (Weight Determination)                        │    ║
    ║   │  ├─ 熵权法 (适用于指标独立的情况)                                │    ║
    ║   │  ├─ CRITIC法 (适用于指标相关的情况)                              │    ║
    ║   │  └─ 组合赋权 (主观+客观结合)                                     │    ║
    ║   └─────────────────────────────────────────────────────────────────┘    ║
    ║                            ↓                                             ║
    ║   ┌─────────────────────────────────────────────────────────────────┐    ║
    ║   │  Step 4: 综合评价 (Comprehensive Evaluation)                    │    ║
    ║   │  ├─ TOPSIS法 (逼近理想解排序)                                    │    ║
    ║   │  └─ 灰色关联分析 (形状相似度评价)                                │    ║
    ║   └─────────────────────────────────────────────────────────────────┘    ║
    ║                            ↓                                             ║
    ║   ┌─────────────────────────────────────────────────────────────────┐    ║
    ║   │  Step 5: 可视化分析 (Visualization)                             │    ║
    ║   │  ├─ 权重分布图（条形图/饼图）                                    │    ║
    ║   │  ├─ 评价结果排序图                                               │    ║
    ║   │  ├─ 雷达图（多维度对比）                                         │    ║
    ║   │  └─ 热力图（评价矩阵）                                           │    ║
    ║   └─────────────────────────────────────────────────────────────────┘    ║
    ║                            ↓                                             ║
    ║   ┌─────────────────────────────────────────────────────────────────┐    ║
    ║   │  Step 6: 灵敏度分析 (Sensitivity Analysis)                      │    ║
    ║   │  └─ 权重扰动对结果的影响分析                                     │    ║
    ║   └─────────────────────────────────────────────────────────────────┘    ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)


# ============================================================
# 第二部分：数据预处理类
# ============================================================

class DataPreprocessor:
    """
    综合评价数据预处理器
    功能：数据加载、缺失值处理、异常值检测、数据标准化
    """
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.indicator_names = None
        self.object_names = None
        self.indicator_types = None
        self.preprocessing_log = []
    
    def load_data(self, data, indicator_names=None, object_names=None, indicator_types=None):
        """
        加载数据
        
        :param data: numpy数组、DataFrame或CSV文件路径
        :param indicator_names: 指标名称列表
        :param object_names: 评价对象名称列表
        :param indicator_types: 指标类型 ['positive', 'negative', ...]
        :return: self
        """
        # 如果是文件路径
        if isinstance(data, str):
            df = pd.read_csv(data, index_col=0)
            self.raw_data = df.values.astype(float)
            self.indicator_names = list(df.columns)
            self.object_names = list(df.index)
        # 如果是DataFrame
        elif isinstance(data, pd.DataFrame):
            self.raw_data = data.values.astype(float)
            self.indicator_names = indicator_names or list(data.columns)
            self.object_names = object_names or list(data.index)
        # 如果是numpy数组
        else:
            self.raw_data = np.array(data).astype(float)
            n_objects, n_indicators = self.raw_data.shape
            self.indicator_names = indicator_names or [f"指标{i+1}" for i in range(n_indicators)]
            self.object_names = object_names or [f"方案{i+1}" for i in range(n_objects)]
        
        self.indicator_types = indicator_types
        self.processed_data = self.raw_data.copy()
        self.preprocessing_log.append("数据加载完成")
        
        print(f"✅ 数据加载成功：{len(self.object_names)}个评价对象，{len(self.indicator_names)}个评价指标")
        return self
    
    def check_missing_values(self):
        """检查缺失值"""
        missing_count = np.isnan(self.processed_data).sum()
        if missing_count > 0:
            print(f"⚠️  发现 {missing_count} 个缺失值")
            return True
        else:
            print("✅ 无缺失值")
            return False
    
    def handle_missing_values(self, method='mean'):
        """
        处理缺失值
        
        :param method: 'mean'(均值填充) / 'median'(中值填充) / 'drop'(删除)
        """
        for j in range(self.processed_data.shape[1]):
            col = self.processed_data[:, j]
            mask = np.isnan(col)
            if mask.any():
                if method == 'mean':
                    fill_value = np.nanmean(col)
                elif method == 'median':
                    fill_value = np.nanmedian(col)
                else:
                    continue
                col[mask] = fill_value
        
        self.preprocessing_log.append(f"缺失值处理：{method}")
        print(f"✅ 缺失值已使用 {method} 方法填充")
        return self
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """
        检测异常值
        
        :param method: 'iqr'(四分位距法) / 'zscore'(Z分数法)
        :param threshold: 阈值（IQR法默认1.5，Z分数法默认3）
        """
        outliers = {}
        
        for j, name in enumerate(self.indicator_names):
            col = self.processed_data[:, j]
            
            if method == 'iqr':
                Q1, Q3 = np.percentile(col, [25, 75])
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outlier_mask = (col < lower) | (col > upper)
            elif method == 'zscore':
                z_scores = np.abs((col - np.mean(col)) / np.std(col))
                outlier_mask = z_scores > threshold
            
            if outlier_mask.any():
                outliers[name] = np.where(outlier_mask)[0].tolist()
        
        if outliers:
            print(f"⚠️  检测到异常值：")
            for indicator, indices in outliers.items():
                print(f"    {indicator}: 第 {indices} 行")
        else:
            print("✅ 未检测到异常值")
        
        return outliers
    
    def normalize(self, method='minmax'):
        """
        数据标准化
        
        :param method: 'minmax'(极差法) / 'zscore'(Z分数法)
        """
        if method == 'minmax':
            data_min = self.processed_data.min(axis=0)
            data_max = self.processed_data.max(axis=0)
            self.processed_data = (self.processed_data - data_min) / (data_max - data_min + 1e-10)
        elif method == 'zscore':
            mean = self.processed_data.mean(axis=0)
            std = self.processed_data.std(axis=0)
            self.processed_data = (self.processed_data - mean) / (std + 1e-10)
        
        self.preprocessing_log.append(f"数据标准化：{method}")
        print(f"✅ 数据已使用 {method} 方法标准化")
        return self
    
    def transform_negative_indicators(self):
        """
        将负向指标转为正向
        """
        if self.indicator_types is None:
            print("⚠️  未设置指标类型，跳过负向指标转换")
            return self
        
        for j, ind_type in enumerate(self.indicator_types):
            if ind_type == 'negative':
                self.processed_data[:, j] = 1 - self.processed_data[:, j]
        
        self.preprocessing_log.append("负向指标已转为正向")
        print("✅ 负向指标已转为正向")
        return self
    
    def get_dataframe(self):
        """返回处理后的DataFrame"""
        return pd.DataFrame(
            self.processed_data,
            index=self.object_names,
            columns=self.indicator_names
        )
    
    def summary(self):
        """打印数据摘要"""
        print("\n" + "="*60)
        print("📊 数据预处理摘要")
        print("="*60)
        print(f"评价对象: {self.object_names}")
        print(f"评价指标: {self.indicator_names}")
        print(f"指标类型: {self.indicator_types}")
        print(f"预处理步骤: {self.preprocessing_log}")
        print("\n处理后数据:")
        print(self.get_dataframe().round(4))
        print("="*60)


# ============================================================
# 第三部分：赋权方法
# ============================================================

class EntropyWeight:
    """熵权法"""
    
    def __init__(self):
        self.weights = None
        self.entropy = None
        self.indicator_names = None
    
    def fit(self, data):
        """
        计算熵权法权重
        
        :param data: 标准化后的数据（DataFrame或数组）
        """
        if isinstance(data, pd.DataFrame):
            self.indicator_names = list(data.columns)
            data = data.values
        else:
            self.indicator_names = [f"指标{i+1}" for i in range(data.shape[1])]
        
        n, m = data.shape
        
        # 计算比例矩阵
        data = np.clip(data, 1e-10, None)  # 避免0值
        p = data / data.sum(axis=0)
        p = np.where(p == 0, 1e-10, p)
        
        # 计算熵值
        k = 1 / np.log(n)
        self.entropy = -k * (p * np.log(p)).sum(axis=0)
        
        # 计算权重
        d = 1 - self.entropy
        self.weights = d / d.sum()
        
        return self
    
    def get_weights(self):
        """返回权重Series"""
        return pd.Series(self.weights, index=self.indicator_names)


class CRITIC:
    """CRITIC法"""
    
    def __init__(self):
        self.weights = None
        self.std = None
        self.conflict = None
        self.correlation_matrix = None
        self.indicator_names = None
    
    def fit(self, data):
        """
        计算CRITIC法权重
        
        :param data: 标准化后的数据（DataFrame或数组）
        """
        if isinstance(data, pd.DataFrame):
            self.indicator_names = list(data.columns)
            data = data.values
        else:
            self.indicator_names = [f"指标{i+1}" for i in range(data.shape[1])]
        
        # 计算标准差
        self.std = np.std(data, axis=0, ddof=1)
        
        # 计算相关系数矩阵
        self.correlation_matrix = np.corrcoef(data.T)
        self.correlation_matrix = np.nan_to_num(self.correlation_matrix, nan=1.0)
        
        # 计算冲突性
        self.conflict = np.sum(1 - self.correlation_matrix, axis=1)
        
        # 计算信息量
        information = self.std * self.conflict
        
        # 计算权重
        self.weights = information / information.sum()
        
        return self
    
    def get_weights(self):
        """返回权重Series"""
        return pd.Series(self.weights, index=self.indicator_names)


class CombinedWeight:
    """组合赋权法"""
    
    def __init__(self, alpha=0.5):
        """
        :param alpha: 熵权法的权重系数，CRITIC法系数为 1-alpha
        """
        self.alpha = alpha
        self.weights = None
        self.entropy_weights = None
        self.critic_weights = None
    
    def fit(self, data):
        """计算组合权重"""
        # 熵权法
        entropy = EntropyWeight()
        entropy.fit(data)
        self.entropy_weights = entropy.get_weights()
        
        # CRITIC法
        critic = CRITIC()
        critic.fit(data)
        self.critic_weights = critic.get_weights()
        
        # 组合
        self.weights = self.alpha * self.entropy_weights + (1 - self.alpha) * self.critic_weights
        
        return self
    
    def get_weights(self):
        """返回权重Series"""
        return self.weights


# ============================================================
# 第四部分：综合评价方法
# ============================================================

class TOPSIS:
    """TOPSIS法"""
    
    def __init__(self):
        self.closeness = None
        self.rankings = None
        self.distances_positive = None
        self.distances_negative = None
        self.object_names = None
    
    def fit(self, data, weights):
        """
        执行TOPSIS评价
        
        :param data: 标准化后的数据
        :param weights: 权重向量
        """
        if isinstance(data, pd.DataFrame):
            self.object_names = list(data.index)
            data = data.values
        else:
            self.object_names = [f"方案{i+1}" for i in range(data.shape[0])]
        
        if isinstance(weights, pd.Series):
            weights = weights.values
        
        # 加权标准化
        data_weighted = data * weights
        
        # 理想解和负理想解
        ideal_positive = data_weighted.max(axis=0)
        ideal_negative = data_weighted.min(axis=0)
        
        # 计算距离
        self.distances_positive = np.sqrt(((data_weighted - ideal_positive) ** 2).sum(axis=1))
        self.distances_negative = np.sqrt(((data_weighted - ideal_negative) ** 2).sum(axis=1))
        
        # 计算贴近度
        self.closeness = self.distances_negative / (self.distances_positive + self.distances_negative + 1e-10)
        
        # 排名
        self.rankings = np.argsort(-self.closeness) + 1
        
        return self
    
    def get_results(self):
        """返回结果DataFrame"""
        return pd.DataFrame({
            '评价对象': self.object_names,
            'D+': self.distances_positive.round(4),
            'D-': self.distances_negative.round(4),
            '贴近度': self.closeness.round(4),
            '排名': [np.where(np.argsort(-self.closeness) == i)[0][0] + 1 for i in range(len(self.object_names))]
        }).sort_values('排名')


class GreyRelationalAnalysis:
    """灰色关联分析"""
    
    def __init__(self, rho=0.5):
        self.rho = rho
        self.relational_degrees = None
        self.rankings = None
        self.object_names = None
    
    def fit(self, data, weights=None):
        """
        执行灰色关联分析
        
        :param data: 标准化后的数据
        :param weights: 可选的权重向量
        """
        if isinstance(data, pd.DataFrame):
            self.object_names = list(data.index)
            data = data.values
        else:
            self.object_names = [f"方案{i+1}" for i in range(data.shape[0])]
        
        n, m = data.shape
        
        # 参考序列（最优值）
        reference = data.max(axis=0)
        
        # 差序列
        delta = np.abs(data - reference)
        delta_min = delta.min()
        delta_max = delta.max()
        
        # 关联系数
        xi = (delta_min + self.rho * delta_max) / (delta + self.rho * delta_max)
        
        # 关联度
        if weights is not None:
            if isinstance(weights, pd.Series):
                weights = weights.values
            self.relational_degrees = (xi * weights).sum(axis=1)
        else:
            self.relational_degrees = xi.mean(axis=1)
        
        # 排名
        self.rankings = np.argsort(-self.relational_degrees) + 1
        
        return self
    
    def get_results(self):
        """返回结果DataFrame"""
        return pd.DataFrame({
            '评价对象': self.object_names,
            '灰色关联度': self.relational_degrees.round(4),
            '排名': [np.where(np.argsort(-self.relational_degrees) == i)[0][0] + 1 for i in range(len(self.object_names))]
        }).sort_values('排名')


# ============================================================
# 第五部分：可视化模块
# ============================================================

class EvaluationVisualizer:
    """综合评价可视化器"""
    
    COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4C9A', '#1B998B']
    
    @staticmethod
    def plot_weights_comparison(entropy_weights, critic_weights, combined_weights=None, 
                                 save_path=None):
        """
        对比不同赋权方法的权重
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(entropy_weights))
        width = 0.25
        
        bars1 = ax.bar(x - width, entropy_weights.values, width, label='熵权法', 
                       color='#2E86AB', edgecolor='white')
        bars2 = ax.bar(x, critic_weights.values, width, label='CRITIC法', 
                       color='#A23B72', edgecolor='white')
        
        if combined_weights is not None:
            bars3 = ax.bar(x + width, combined_weights.values, width, label='组合赋权', 
                           color='#F18F01', edgecolor='white')
        
        ax.set_xlabel('评价指标', fontsize=12, fontweight='bold')
        ax.set_ylabel('权重', fontsize=12, fontweight='bold')
        ax.set_title('不同赋权方法权重对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(entropy_weights.index, rotation=15)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_topsis_results(results, title="TOPSIS评价结果", save_path=None):
        """绘制TOPSIS结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 贴近度排序
        ax1 = axes[0]
        sorted_results = results.sort_values('贴近度', ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_results)))
        ax1.barh(sorted_results['评价对象'], sorted_results['贴近度'],
                color=colors, edgecolor='white', linewidth=2)
        ax1.set_xlabel('贴近度', fontweight='bold')
        ax1.set_title('(a) 贴近度排序', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 1)
        
        # 添加排名标注
        for i, (_, row) in enumerate(sorted_results.iterrows()):
            ax1.text(row['贴近度'] + 0.02, i, f"#{int(row['排名'])}", 
                    va='center', fontweight='bold')
        
        # 距离对比
        ax2 = axes[1]
        x = np.arange(len(results))
        width = 0.35
        ax2.bar(x - width/2, results['D+'], width, label='D+ (到理想解)', 
               color='#2E86AB', edgecolor='white')
        ax2.bar(x + width/2, results['D-'], width, label='D- (到负理想解)', 
               color='#A23B72', edgecolor='white')
        ax2.set_xlabel('评价对象', fontweight='bold')
        ax2.set_ylabel('距离', fontweight='bold')
        ax2.set_title('(b) 距离对比', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(results['评价对象'], rotation=15)
        ax2.legend()
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_radar(data, title="多维度雷达图", save_path=None):
        """绘制雷达图"""
        if isinstance(data, pd.DataFrame):
            indicators = list(data.columns)
            object_names = list(data.index)
            values = data.values
        else:
            raise ValueError("请传入DataFrame格式的数据")
        
        n_indicators = len(indicators)
        angles = np.linspace(0, 2 * np.pi, n_indicators, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        colors = EvaluationVisualizer.COLORS
        for i, (name, row) in enumerate(zip(object_names, values)):
            row_values = row.tolist()
            row_values += row_values[:1]
            ax.plot(angles, row_values, 'o-', linewidth=2, label=name,
                   color=colors[i % len(colors)])
            ax.fill(angles, row_values, alpha=0.1, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(indicators, fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_full_report(raw_data, weights, topsis_results, gra_results=None, save_path=None):
        """生成完整评价报告"""
        fig = plt.figure(figsize=(16, 12))
        
        # 子图1: 原始数据热力图
        ax1 = fig.add_subplot(2, 2, 1)
        data_norm = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min())
        im1 = ax1.imshow(data_norm.values, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(np.arange(len(raw_data.columns)))
        ax1.set_yticks(np.arange(len(raw_data.index)))
        ax1.set_xticklabels(raw_data.columns, fontsize=9, rotation=15)
        ax1.set_yticklabels(raw_data.index, fontsize=9)
        ax1.set_title('(a) 评价矩阵热力图', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 子图2: 权重分布
        ax2 = fig.add_subplot(2, 2, 2)
        colors = EvaluationVisualizer.COLORS[:len(weights)]
        bars = ax2.bar(weights.index, weights.values, color=colors, edgecolor='white')
        ax2.set_ylabel('权重', fontweight='bold')
        ax2.set_title('(b) 指标权重分布', fontsize=12, fontweight='bold')
        for bar, val in zip(bars, weights.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax2.tick_params(axis='x', rotation=15)
        
        # 子图3: TOPSIS排序
        ax3 = fig.add_subplot(2, 2, 3)
        sorted_results = topsis_results.sort_values('贴近度', ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_results)))
        ax3.barh(sorted_results['评价对象'], sorted_results['贴近度'],
                color=colors, edgecolor='white')
        ax3.set_xlabel('贴近度', fontweight='bold')
        ax3.set_title('(c) TOPSIS评价排序', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 1)
        for i, (_, row) in enumerate(sorted_results.iterrows()):
            ax3.text(row['贴近度'] + 0.02, i, f"#{int(row['排名'])}", 
                    va='center', fontweight='bold', fontsize=10)
        
        # 子图4: 雷达图
        ax4 = fig.add_subplot(2, 2, 4, polar=True)
        indicators = list(raw_data.columns)
        n_indicators = len(indicators)
        angles = np.linspace(0, 2 * np.pi, n_indicators, endpoint=False).tolist()
        angles += angles[:1]
        
        colors_radar = EvaluationVisualizer.COLORS
        for i, (name, row) in enumerate(data_norm.iterrows()):
            values = row.tolist()
            values += values[:1]
            ax4.plot(angles, values, 'o-', linewidth=2, label=name,
                    color=colors_radar[i % len(colors_radar)])
            ax4.fill(angles, values, alpha=0.1)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(indicators, fontsize=9)
        ax4.set_title('(d) 多维度对比', fontsize=12, fontweight='bold', y=1.08)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
        
        plt.suptitle('综合评价分析报告 (Comprehensive Evaluation Report)', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 第六部分：完整案例演示
# ============================================================

def run_complete_example():
    """运行完整的综合评价案例"""
    
    print_workflow()
    
    print("\n" + "="*70)
    print("🎯 综合评价完整案例：供应商选择问题")
    print("="*70)
    
    # ========================================
    # Step 1: 数据准备
    # ========================================
    print("\n" + "-"*50)
    print("📊 Step 1: 数据准备")
    print("-"*50)
    
    # 创建示例数据：5个供应商，4个评价指标
    np.random.seed(2026)
    data = pd.DataFrame({
        '质量评分': [85, 92, 78, 88, 95],
        '价格(万元)': [120, 95, 150, 110, 85],    # 负向指标
        '交货期(天)': [15, 20, 10, 25, 12],       # 负向指标
        '服务评分': [90, 85, 88, 92, 80]
    }, index=['供应商A', '供应商B', '供应商C', '供应商D', '供应商E'])
    
    indicator_types = ['positive', 'negative', 'negative', 'positive']
    
    print("\n原始数据：")
    print(data)
    print(f"\n指标类型：{indicator_types}")
    
    # ========================================
    # Step 2: 数据预处理
    # ========================================
    print("\n" + "-"*50)
    print("📊 Step 2: 数据预处理")
    print("-"*50)
    
    preprocessor = DataPreprocessor()
    preprocessor.load_data(data, indicator_types=indicator_types)
    preprocessor.check_missing_values()
    preprocessor.detect_outliers(method='iqr')
    preprocessor.normalize(method='minmax')
    preprocessor.transform_negative_indicators()
    
    processed_data = preprocessor.get_dataframe()
    print("\n预处理后的数据（标准化+负向转正向）：")
    print(processed_data.round(4))
    
    # ========================================
    # Step 3: 确定权重
    # ========================================
    print("\n" + "-"*50)
    print("📊 Step 3: 确定权重")
    print("-"*50)
    
    # 3.1 熵权法
    entropy = EntropyWeight()
    entropy.fit(processed_data)
    entropy_weights = entropy.get_weights()
    print("\n熵权法权重：")
    print(entropy_weights.round(4))
    
    # 3.2 CRITIC法
    critic = CRITIC()
    critic.fit(processed_data)
    critic_weights = critic.get_weights()
    print("\nCRITIC法权重：")
    print(critic_weights.round(4))
    
    # 3.3 组合赋权
    combined = CombinedWeight(alpha=0.5)
    combined.fit(processed_data)
    combined_weights = combined.get_weights()
    print("\n组合赋权权重（α=0.5）：")
    print(combined_weights.round(4))
    
    # ========================================
    # Step 4: 综合评价
    # ========================================
    print("\n" + "-"*50)
    print("📊 Step 4: 综合评价")
    print("-"*50)
    
    # 4.1 TOPSIS法
    topsis = TOPSIS()
    topsis.fit(processed_data, combined_weights)
    topsis_results = topsis.get_results()
    print("\nTOPSIS评价结果：")
    print(topsis_results)
    
    # 4.2 灰色关联分析
    gra = GreyRelationalAnalysis(rho=0.5)
    gra.fit(processed_data, combined_weights)
    gra_results = gra.get_results()
    print("\n灰色关联分析结果：")
    print(gra_results)
    
    # ========================================
    # Step 5: 可视化分析
    # ========================================
    print("\n" + "-"*50)
    print("📊 Step 5: 可视化分析")
    print("-"*50)
    
    visualizer = EvaluationVisualizer()
    
    # 权重对比图
    visualizer.plot_weights_comparison(entropy_weights, critic_weights, combined_weights)
    
    # TOPSIS结果图
    visualizer.plot_topsis_results(topsis_results, title="供应商TOPSIS评价结果")
    
    # 雷达图
    visualizer.plot_radar(processed_data, title="供应商多维度对比雷达图")
    
    # 完整报告
    visualizer.plot_full_report(data, combined_weights, topsis_results)
    
    # ========================================
    # 结论
    # ========================================
    print("\n" + "="*70)
    print("🏆 评价结论")
    print("="*70)
    
    best_topsis = topsis_results[topsis_results['排名'] == 1]['评价对象'].values[0]
    best_gra = gra_results[gra_results['排名'] == 1]['评价对象'].values[0]
    
    print(f"\nTOPSIS法最优方案: {best_topsis}")
    print(f"灰色关联分析最优方案: {best_gra}")
    
    if best_topsis == best_gra:
        print(f"\n✅ 两种方法评价结果一致，最终推荐: {best_topsis}")
    else:
        print(f"\n⚠️  两种方法评价结果不一致，建议进行敏感性分析")
    
    print("\n" + "="*70)
    print("   ✅ 综合评价完成！")
    print("="*70)
    
    return {
        'raw_data': data,
        'processed_data': processed_data,
        'entropy_weights': entropy_weights,
        'critic_weights': critic_weights,
        'combined_weights': combined_weights,
        'topsis_results': topsis_results,
        'gra_results': gra_results
    }


# ============================================================
# 第七部分：使用指南
# ============================================================

def print_usage_guide():
    """打印使用指南"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                        综合评价模型使用指南                               ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                          ║
    ║  【快速开始】                                                            ║
    ║                                                                          ║
    ║  from comprehensive_evaluation_tutorial import *                         ║
    ║                                                                          ║
    ║  # 1. 准备数据                                                           ║
    ║  data = pd.DataFrame({                                                   ║
    ║      '指标1': [85, 92, 78],                                              ║
    ║      '指标2': [120, 95, 150],  # 负向                                    ║
    ║      '指标3': [90, 85, 88]                                               ║
    ║  }, index=['方案A', '方案B', '方案C'])                                   ║
    ║  indicator_types = ['positive', 'negative', 'positive']                  ║
    ║                                                                          ║
    ║  # 2. 预处理                                                             ║
    ║  preprocessor = DataPreprocessor()                                       ║
    ║  preprocessor.load_data(data, indicator_types=indicator_types)           ║
    ║  preprocessor.normalize('minmax')                                        ║
    ║  preprocessor.transform_negative_indicators()                            ║
    ║  processed_data = preprocessor.get_dataframe()                           ║
    ║                                                                          ║
    ║  # 3. 计算权重                                                           ║
    ║  entropy = EntropyWeight()                                               ║
    ║  entropy.fit(processed_data)                                             ║
    ║  weights = entropy.get_weights()                                         ║
    ║                                                                          ║
    ║  # 4. TOPSIS评价                                                         ║
    ║  topsis = TOPSIS()                                                       ║
    ║  topsis.fit(processed_data, weights)                                     ║
    ║  results = topsis.get_results()                                          ║
    ║                                                                          ║
    ║  # 5. 可视化                                                             ║
    ║  visualizer = EvaluationVisualizer()                                     ║
    ║  visualizer.plot_topsis_results(results)                                 ║
    ║                                                                          ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  【指标类型说明】                                                        ║
    ║  - 'positive': 正向指标（越大越好）如：收益、质量、效率                  ║
    ║  - 'negative': 负向指标（越小越好）如：成本、风险、时间                  ║
    ║                                                                          ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  【赋权方法选择】                                                        ║
    ║  - 熵权法：适用于指标间相关性较低的情况                                  ║
    ║  - CRITIC法：适用于指标间存在较强相关性的情况                            ║
    ║  - 组合赋权：综合考虑两种方法，更加稳健                                  ║
    ║                                                                          ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  【论文图表建议】                                                        ║
    ║  Figure 1: 评价指标体系（树形图或表格）                                  ║
    ║  Figure 2: 不同赋权方法权重对比                                          ║
    ║  Figure 3: TOPSIS评价结果（贴近度排序）                                  ║
    ║  Figure 4: 多维度雷达图对比                                              ║
    ║  Figure 5: 综合评价报告                                                  ║
    ║                                                                          ║
    ║  Table 1: 原始数据矩阵                                                   ║
    ║  Table 2: 标准化矩阵                                                     ║
    ║  Table 3: 权重计算过程                                                   ║
    ║  Table 4: TOPSIS评价结果                                                 ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    # 运行完整案例
    results = run_complete_example()
    
    # 打印使用指南
    print_usage_guide()
