"""
============================================================
评价类模型 (Evaluation Models)
包含：熵权法 (Entropy Weight) + TOPSIS法
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：多指标综合评价、方案排序与选择
特点：完整的参数设置、数据预处理、可视化与美化
作者：MCM/ICM Team
日期：2026年1月
============================================================

使用场景：
- 多方案综合评价与排序
- 指标权重客观确定
- 供应商选择、项目评估
- 区域发展水平评价

输入数据格式：
- 行：评价对象（方案）
- 列：评价指标
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
from scipy import stats
from datetime import datetime

warnings.filterwarnings('ignore')


# ============================================================
# 第一部分：全局配置与美化设置 (Global Configuration)
# ============================================================

class PlotStyleConfig:
    """图表美化配置类 - 符合学术论文标准"""
    
    # MCM/ICM 推荐配色方案
    COLORS = {
        'primary': '#2E86AB',      # 主色调-深蓝
        'secondary': '#A23B72',    # 辅助色-玫红
        'accent': '#F18F01',       # 强调色-橙色
        'success': '#C73E1D',      # 成功/最优-红色
        'neutral': '#3B3B3B',      # 中性色-深灰
        'background': '#FAFAFA',   # 背景色
        'grid': '#E0E0E0'          # 网格色
    }
    
    # 学术配色板
    PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4C9A', '#1B998B']
    
    @staticmethod
    def setup_style():
        """设置全局绘图风格"""
        plt.style.use('seaborn-v0_8-whitegrid')
        rcParams['figure.figsize'] = (12, 8)
        rcParams['figure.dpi'] = 100
        rcParams['savefig.dpi'] = 300
        rcParams['font.size'] = 11
        rcParams['axes.titlesize'] = 14
        rcParams['axes.labelsize'] = 12
        rcParams['xtick.labelsize'] = 10
        rcParams['ytick.labelsize'] = 10
        rcParams['legend.fontsize'] = 10
        rcParams['figure.facecolor'] = 'white'
        rcParams['axes.facecolor'] = 'white'
        rcParams['axes.edgecolor'] = '#333333'
        rcParams['grid.alpha'] = 0.3
        # 支持中文
        rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False

# 初始化样式
PlotStyleConfig.setup_style()


# ============================================================
# 第二部分：数据预处理模块 (Data Preprocessing)
# ============================================================

class EvaluationDataPreprocessor:
    """评价数据预处理类"""
    
    def __init__(self, random_seed=42):
        """
        初始化预处理器
        :param random_seed: 随机种子
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.raw_data = None
        self.processed_data = None
        self.indicator_names = None
        self.object_names = None
        self.indicator_types = None  # 'positive' or 'negative'
    
    def load_from_csv(self, filepath, index_col=0):
        """从CSV文件加载数据"""
        df = pd.read_csv(filepath, index_col=index_col)
        self.raw_data = df
        self.indicator_names = list(df.columns)
        self.object_names = list(df.index)
        return self
    
    def load_from_dataframe(self, df):
        """从DataFrame加载数据"""
        self.raw_data = df.copy()
        self.indicator_names = list(df.columns)
        self.object_names = list(df.index)
        return self
    
    def load_from_array(self, data, indicator_names=None, object_names=None):
        """从numpy数组加载数据"""
        n_objects, n_indicators = data.shape
        if indicator_names is None:
            indicator_names = [f"指标{i+1}" for i in range(n_indicators)]
        if object_names is None:
            object_names = [f"方案{i+1}" for i in range(n_objects)]
        
        self.raw_data = pd.DataFrame(data, index=object_names, columns=indicator_names)
        self.indicator_names = indicator_names
        self.object_names = object_names
        return self
    
    def generate_demo_data(self, n_objects=5, n_indicators=4, scenario='random'):
        """
        生成演示数据
        
        :param n_objects: 方案数量
        :param n_indicators: 指标数量
        :param scenario: 数据场景
            - 'random': 随机数据
            - 'supplier': 供应商选择场景
            - 'project': 项目评估场景
        """
        if scenario == 'random':
            data = np.random.uniform(60, 100, (n_objects, n_indicators))
            indicator_names = [f"指标{i+1}" for i in range(n_indicators)]
            object_names = [f"方案{i+1}" for i in range(n_objects)]
            self.indicator_types = ['positive'] * n_indicators
            
        elif scenario == 'supplier':
            # 供应商选择：质量、价格、交货期、服务
            data = pd.DataFrame({
                "质量评分": np.random.uniform(70, 95, n_objects),
                "价格(万元)": np.random.uniform(80, 150, n_objects),  # 负向
                "交货期(天)": np.random.randint(5, 30, n_objects),    # 负向
                "服务评分": np.random.uniform(60, 90, n_objects)
            })
            indicator_names = list(data.columns)
            object_names = [f"供应商{i+1}" for i in range(n_objects)]
            self.indicator_types = ['positive', 'negative', 'negative', 'positive']
            data = data.values
            
        elif scenario == 'project':
            # 项目评估：收益、成本、风险、周期
            data = pd.DataFrame({
                "预期收益(万)": np.random.uniform(100, 500, n_objects),
                "投资成本(万)": np.random.uniform(50, 200, n_objects),   # 负向
                "风险等级": np.random.uniform(1, 5, n_objects),          # 负向
                "回报周期(月)": np.random.randint(6, 36, n_objects)      # 负向
            })
            indicator_names = list(data.columns)
            object_names = [f"项目{chr(65+i)}" for i in range(n_objects)]
            self.indicator_types = ['positive', 'negative', 'negative', 'negative']
            data = data.values
        
        self.raw_data = pd.DataFrame(data, index=object_names, columns=indicator_names)
        self.indicator_names = indicator_names
        self.object_names = object_names
        return self
    
    def set_indicator_types(self, types):
        """
        设置指标类型
        :param types: 列表，每个元素为 'positive' 或 'negative'
        """
        self.indicator_types = types
        return self
    
    def check_data_quality(self):
        """检查数据质量并返回报告"""
        report = {
            'shape': self.raw_data.shape,
            'missing_values': self.raw_data.isnull().sum().to_dict(),
            'has_negative': (self.raw_data < 0).any().any(),
            'statistics': self.raw_data.describe().to_dict()
        }
        return report
    
    def handle_missing_values(self, method='mean'):
        """处理缺失值"""
        if method == 'mean':
            self.raw_data = self.raw_data.fillna(self.raw_data.mean())
        elif method == 'median':
            self.raw_data = self.raw_data.fillna(self.raw_data.median())
        elif method == 'drop':
            self.raw_data = self.raw_data.dropna()
        return self
    
    def get_data(self):
        """获取数据"""
        return self.raw_data
    
    def summary(self):
        """打印数据摘要"""
        print("\n" + "="*60)
        print("📊 评价数据摘要 (Evaluation Data Summary)")
        print("="*60)
        print(f"  评价对象数量: {len(self.object_names)}")
        print(f"  评价指标数量: {len(self.indicator_names)}")
        print(f"  指标名称: {self.indicator_names}")
        print(f"  对象名称: {self.object_names}")
        if self.indicator_types:
            print(f"  指标类型: {self.indicator_types}")
        print("\n原始数据:")
        print(self.raw_data.round(2))
        print("="*60 + "\n")


# ============================================================
# 第三部分：熵权法核心算法 (Entropy Weight Method)
# ============================================================

class EntropyWeightMethod:
    """
    熵权法 - 客观赋权方法
    
    原理：
    信息熵反映数据的离散程度，离散程度越大（熵值越小），
    该指标对评价结果的影响越大，权重也应越高。
    
    步骤：
    1. 数据标准化（极差法）
    2. 计算各指标熵值
    3. 计算权重（熵值越小，权重越大）
    """
    
    def __init__(self, verbose=True):
        """
        初始化熵权法
        :param verbose: 是否打印详细信息
        """
        self.verbose = verbose
        self.data = None
        self.data_normalized = None
        self.entropy = None
        self.weights = None
        self.indicator_names = None
        
    def fit(self, data, negative_indices=None, indicator_types=None):
        """
        计算指标权重
        
        :param data: DataFrame或numpy数组，行为对象，列为指标
        :param negative_indices: 负向指标的列索引列表（从0开始）
        :param indicator_types: 指标类型列表 ['positive', 'negative', ...]
        :return: self
        """
        # 数据转换
        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.indicator_names = list(data.columns)
        else:
            self.data = data
            self.indicator_names = [f"指标{i+1}" for i in range(data.shape[1])]
        
        # 确定负向指标
        if indicator_types is not None:
            negative_indices = [i for i, t in enumerate(indicator_types) if t == 'negative']
        elif negative_indices is None:
            negative_indices = []
        
        n, m = self.data.shape  # n=对象数，m=指标数
        
        # Step 1: 极差标准化
        data_min = self.data.min(axis=0)
        data_max = self.data.max(axis=0)
        self.data_normalized = (self.data - data_min) / (data_max - data_min + 1e-10)
        
        # Step 2: 负向指标转正向
        for idx in negative_indices:
            self.data_normalized[:, idx] = 1 - self.data_normalized[:, idx]
        
        # Step 3: 计算比例矩阵
        p = self.data_normalized / (self.data_normalized.sum(axis=0) + 1e-10)
        p = np.where(p == 0, 1e-10, p)  # 避免log(0)
        
        # Step 4: 计算熵值
        k = 1 / np.log(n)  # 系数
        self.entropy = -k * (p * np.log(p)).sum(axis=0)
        
        # Step 5: 计算权重
        d = 1 - self.entropy  # 差异系数
        self.weights = d / d.sum()
        
        if self.verbose:
            self._print_results()
        
        return self
    
    def _print_results(self):
        """打印结果"""
        print("\n" + "="*60)
        print("📊 熵权法计算结果 (Entropy Weight Results)")
        print("="*60)
        
        results = pd.DataFrame({
            '指标': self.indicator_names,
            '熵值': self.entropy,
            '差异系数': 1 - self.entropy,
            '权重': self.weights
        })
        print(results.round(4).to_string(index=False))
        print(f"\n权重总和验证: {self.weights.sum():.4f}")
        print("="*60)
    
    def get_weights(self):
        """返回权重"""
        return pd.Series(self.weights, index=self.indicator_names)
    
    def transform(self, data=None):
        """
        使用熵权法权重计算综合得分
        """
        if data is None:
            data = self.data_normalized
        elif isinstance(data, pd.DataFrame):
            data = data.values
        
        scores = (data * self.weights).sum(axis=1)
        return scores


# ============================================================
# 第三部分B：CRITIC法核心算法 (CRITIC Method)
# ============================================================

class CRITICMethod:
    """
    CRITIC法 - 客观赋权方法
    (Criteria Importance Through Intercriteria Correlation)
    
    原理：
    综合考虑两个维度确定权重：
    1. 对比强度（Contrast Intensity）：用标准差衡量，标准差越大，变异程度越高
    2. 冲突性（Conflicting）：用相关系数衡量，与其他指标相关性越低，冲突性越大
    
    优势：
    - 同时考虑数据变异性和指标间相关性
    - 对冗余指标的权重会自动降低
    - 适用于指标间存在较强相关性的情况
    
    步骤：
    1. 数据标准化
    2. 计算各指标标准差（对比强度）
    3. 计算指标间相关系数矩阵
    4. 计算信息量 = 标准差 × Σ(1-相关系数)
    5. 归一化得到权重
    """
    
    def __init__(self, verbose=True):
        """
        初始化CRITIC法
        :param verbose: 是否打印详细信息
        """
        self.verbose = verbose
        self.data = None
        self.data_normalized = None
        self.std = None  # 标准差
        self.correlation_matrix = None  # 相关系数矩阵
        self.conflict = None  # 冲突性
        self.information = None  # 信息量
        self.weights = None
        self.indicator_names = None
        
    def fit(self, data, negative_indices=None, indicator_types=None):
        """
        计算指标权重
        
        :param data: DataFrame或numpy数组，行为对象，列为指标
        :param negative_indices: 负向指标的列索引列表（从0开始）
        :param indicator_types: 指标类型列表 ['positive', 'negative', ...]
        :return: self
        """
        # 数据转换
        if isinstance(data, pd.DataFrame):
            self.data = data.values.astype(float)
            self.indicator_names = list(data.columns)
        else:
            self.data = data.astype(float)
            self.indicator_names = [f"指标{i+1}" for i in range(data.shape[1])]
        
        # 确定负向指标
        if indicator_types is not None:
            negative_indices = [i for i, t in enumerate(indicator_types) if t == 'negative']
        elif negative_indices is None:
            negative_indices = []
        
        n, m = self.data.shape  # n=对象数，m=指标数
        
        # Step 1: 极差标准化
        data_min = self.data.min(axis=0)
        data_max = self.data.max(axis=0)
        self.data_normalized = (self.data - data_min) / (data_max - data_min + 1e-10)
        
        # Step 2: 负向指标转正向
        for idx in negative_indices:
            self.data_normalized[:, idx] = 1 - self.data_normalized[:, idx]
        
        # Step 3: 计算标准差（对比强度）
        self.std = np.std(self.data_normalized, axis=0, ddof=1)
        
        # Step 4: 计算相关系数矩阵
        self.correlation_matrix = np.corrcoef(self.data_normalized.T)
        # 处理可能的NaN值（当某列全为相同值时）
        self.correlation_matrix = np.nan_to_num(self.correlation_matrix, nan=1.0)
        
        # Step 5: 计算冲突性（与其他指标的相关性越低，冲突性越大）
        self.conflict = np.sum(1 - self.correlation_matrix, axis=1)
        
        # Step 6: 计算信息量
        self.information = self.std * self.conflict
        
        # Step 7: 计算权重
        self.weights = self.information / self.information.sum()
        
        if self.verbose:
            self._print_results()
        
        return self
    
    def _print_results(self):
        """打印结果"""
        print("\n" + "="*70)
        print("📊 CRITIC法计算结果 (CRITIC Method Results)")
        print("="*70)
        
        results = pd.DataFrame({
            '指标': self.indicator_names,
            '标准差(对比强度)': self.std,
            '冲突性': self.conflict,
            '信息量': self.information,
            '权重': self.weights
        })
        print(results.round(4).to_string(index=False))
        print(f"\n权重总和验证: {self.weights.sum():.4f}")
        
        # 打印相关系数矩阵
        print("\n相关系数矩阵:")
        corr_df = pd.DataFrame(
            self.correlation_matrix, 
            index=self.indicator_names, 
            columns=self.indicator_names
        )
        print(corr_df.round(3))
        print("="*70)
    
    def get_weights(self):
        """返回权重"""
        return pd.Series(self.weights, index=self.indicator_names)
    
    def get_correlation_matrix(self):
        """返回相关系数矩阵"""
        return pd.DataFrame(
            self.correlation_matrix,
            index=self.indicator_names,
            columns=self.indicator_names
        )
    
    def transform(self, data=None):
        """
        使用CRITIC法权重计算综合得分
        """
        if data is None:
            data = self.data_normalized
        elif isinstance(data, pd.DataFrame):
            data = data.values
        
        scores = (data * self.weights).sum(axis=1)
        return scores
    
    def plot_analysis(self, figsize=(16, 5), save_path=None):
        """
        可视化CRITIC法分析结果
        
        :param figsize: 图形大小
        :param save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4C9A', '#1B998B']
        
        # 子图1: 权重分布
        ax1 = axes[0]
        bars = ax1.bar(self.indicator_names, self.weights, 
                       color=colors[:len(self.indicator_names)], 
                       edgecolor='white', linewidth=2)
        ax1.set_xlabel('指标 (Indicator)', fontweight='bold')
        ax1.set_ylabel('权重 (Weight)', fontweight='bold')
        ax1.set_title('(a) CRITIC法权重分布', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, max(self.weights) * 1.3)
        for bar, w in zip(bars, self.weights):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{w:.3f}', ha='center', va='bottom', fontsize=10)
        ax1.tick_params(axis='x', rotation=15)
        
        # 子图2: 对比强度与冲突性
        ax2 = axes[1]
        x = np.arange(len(self.indicator_names))
        width = 0.35
        bars1 = ax2.bar(x - width/2, self.std / self.std.max(), width, 
                       label='对比强度(标准化)', color='#2E86AB', edgecolor='white')
        bars2 = ax2.bar(x + width/2, self.conflict / self.conflict.max(), width,
                       label='冲突性(标准化)', color='#A23B72', edgecolor='white')
        ax2.set_xlabel('指标 (Indicator)', fontweight='bold')
        ax2.set_ylabel('标准化值', fontweight='bold')
        ax2.set_title('(b) 对比强度与冲突性分析', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.indicator_names, rotation=15)
        ax2.legend()
        
        # 子图3: 相关系数热力图
        ax3 = axes[2]
        im = ax3.imshow(self.correlation_matrix, cmap='coolwarm', 
                       aspect='auto', vmin=-1, vmax=1)
        ax3.set_xticks(np.arange(len(self.indicator_names)))
        ax3.set_yticks(np.arange(len(self.indicator_names)))
        ax3.set_xticklabels(self.indicator_names, rotation=45, ha='right')
        ax3.set_yticklabels(self.indicator_names)
        ax3.set_title('(c) 指标相关系数矩阵', fontsize=12, fontweight='bold')
        
        # 添加数值标注
        for i in range(len(self.indicator_names)):
            for j in range(len(self.indicator_names)):
                text = ax3.text(j, i, f'{self.correlation_matrix[i, j]:.2f}',
                               ha='center', va='center', fontsize=9,
                               color='white' if abs(self.correlation_matrix[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im, ax=ax3, shrink=0.8)
        
        plt.suptitle('CRITIC法分析报告', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


# ============================================================
# 第四部分：TOPSIS法核心算法 (TOPSIS Method)
# ============================================================

class TOPSIS:
    """
    TOPSIS法 - 逼近理想解排序法
    (Technique for Order Preference by Similarity to Ideal Solution)
    
    原理：
    通过计算各方案与理想解（最优方案）和负理想解（最劣方案）
    的距离，获得各方案的相对贴近度，进行排序。
    
    步骤：
    1. 数据标准化
    2. 加权标准化（可选，与熵权法结合）
    3. 确定正/负理想解
    4. 计算各方案到正/负理想解的距离
    5. 计算相对贴近度
    """
    
    def __init__(self, verbose=True):
        """
        初始化TOPSIS
        :param verbose: 是否打印详细信息
        """
        self.verbose = verbose
        self.data = None
        self.data_normalized = None
        self.weights = None
        self.ideal_positive = None
        self.ideal_negative = None
        self.distances_positive = None
        self.distances_negative = None
        self.closeness = None
        self.rankings = None
        self.object_names = None
        self.indicator_names = None
    
    def fit(self, data, weights=None, negative_indices=None, indicator_types=None):
        """
        执行TOPSIS评价
        
        :param data: DataFrame或numpy数组
        :param weights: 权重向量（可由熵权法得出）
        :param negative_indices: 负向指标索引
        :param indicator_types: 指标类型列表
        :return: self
        """
        # 数据转换
        if isinstance(data, pd.DataFrame):
            self.data = data.values.astype(float)
            self.object_names = list(data.index)
            self.indicator_names = list(data.columns)
        else:
            self.data = data.astype(float)
            self.object_names = [f"方案{i+1}" for i in range(data.shape[0])]
            self.indicator_names = [f"指标{i+1}" for i in range(data.shape[1])]
        
        n, m = self.data.shape
        
        # 确定负向指标
        if indicator_types is not None:
            negative_indices = [i for i, t in enumerate(indicator_types) if t == 'negative']
        elif negative_indices is None:
            negative_indices = []
        
        # 默认等权重
        if weights is None:
            self.weights = np.ones(m) / m
        elif isinstance(weights, pd.Series):
            self.weights = weights.values
        else:
            self.weights = np.array(weights)
        
        # Step 1: 极差标准化
        data_min = self.data.min(axis=0)
        data_max = self.data.max(axis=0)
        self.data_normalized = (self.data - data_min) / (data_max - data_min + 1e-10)
        
        # Step 2: 负向指标转正向
        for idx in negative_indices:
            self.data_normalized[:, idx] = 1 - self.data_normalized[:, idx]
        
        # Step 3: 加权标准化
        data_weighted = self.data_normalized * self.weights
        
        # Step 4: 确定正/负理想解
        self.ideal_positive = data_weighted.max(axis=0)
        self.ideal_negative = data_weighted.min(axis=0)
        
        # Step 5: 计算距离
        self.distances_positive = np.sqrt(((data_weighted - self.ideal_positive) ** 2).sum(axis=1))
        self.distances_negative = np.sqrt(((data_weighted - self.ideal_negative) ** 2).sum(axis=1))
        
        # Step 6: 计算相对贴近度
        self.closeness = self.distances_negative / (self.distances_positive + self.distances_negative + 1e-10)
        
        # Step 7: 排序
        self.rankings = np.argsort(-self.closeness) + 1  # 从1开始
        
        if self.verbose:
            self._print_results()
        
        return self
    
    def _print_results(self):
        """打印结果"""
        print("\n" + "="*60)
        print("📊 TOPSIS评价结果 (TOPSIS Results)")
        print("="*60)
        
        # 构建结果表
        results = pd.DataFrame({
            '评价对象': self.object_names,
            'D+': self.distances_positive,
            'D-': self.distances_negative,
            '贴近度': self.closeness,
            '排名': np.argsort(-self.closeness) + 1
        })
        results = results.sort_values('排名')
        print(results.round(4).to_string(index=False))
        print("\n理想解 (Ideal Positive):", self.ideal_positive.round(4))
        print("负理想解 (Ideal Negative):", self.ideal_negative.round(4))
        print("="*60)
    
    def get_results(self):
        """返回评价结果DataFrame"""
        results = pd.DataFrame({
            '评价对象': self.object_names,
            'D+': self.distances_positive,
            'D-': self.distances_negative,
            '贴近度': self.closeness,
            '排名': np.argsort(-self.closeness) + 1
        })
        return results.sort_values('排名')
    
    def get_best(self):
        """返回最优方案"""
        best_idx = np.argmax(self.closeness)
        return self.object_names[best_idx], self.closeness[best_idx]


# ============================================================
# 第五部分：可视化模块 (Visualization Module)
# ============================================================

class EvaluationVisualizer:
    """评价模型可视化类"""
    
    def __init__(self):
        self.colors = PlotStyleConfig.PALETTE
    
    def plot_weights(self, weights, title="指标权重分布", save_path=None):
        """
        绘制权重分布图
        
        :param weights: 权重Series或dict
        :param title: 图标题
        :param save_path: 保存路径
        """
        if isinstance(weights, dict):
            weights = pd.Series(weights)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 条形图
        ax1 = axes[0]
        bars = ax1.bar(weights.index, weights.values, color=self.colors[:len(weights)], 
                       edgecolor='white', linewidth=2)
        ax1.set_xlabel('指标 (Indicator)', fontweight='bold')
        ax1.set_ylabel('权重 (Weight)', fontweight='bold')
        ax1.set_title('(a) 指标权重条形图', fontsize=13, fontweight='bold')
        ax1.set_ylim(0, max(weights.values) * 1.2)
        # 标注数值
        for bar, val in zip(bars, weights.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        ax1.tick_params(axis='x', rotation=15)
        
        # 饼图
        ax2 = axes[1]
        wedges, texts, autotexts = ax2.pie(weights.values, labels=weights.index, 
                                           autopct='%1.1f%%', colors=self.colors[:len(weights)],
                                           wedgeprops=dict(edgecolor='white', linewidth=2))
        ax2.set_title('(b) 指标权重饼图', fontsize=13, fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_topsis_results(self, results, title="TOPSIS评价结果", save_path=None):
        """
        绘制TOPSIS结果图
        
        :param results: TOPSIS结果DataFrame
        :param title: 图标题
        :param save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 贴近度排序条形图
        ax1 = axes[0]
        sorted_results = results.sort_values('贴近度', ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_results)))
        bars = ax1.barh(sorted_results['评价对象'], sorted_results['贴近度'],
                       color=colors, edgecolor='white', linewidth=2)
        ax1.set_xlabel('贴近度 (Closeness)', fontweight='bold')
        ax1.set_title('(a) 方案贴近度排序', fontsize=13, fontweight='bold')
        ax1.set_xlim(0, 1)
        # 标注排名
        for i, (_, row) in enumerate(sorted_results.iterrows()):
            ax1.text(row['贴近度'] + 0.02, i, f"#{int(row['排名'])}", 
                    va='center', fontweight='bold', fontsize=10)
        
        # 距离对比图
        ax2 = axes[1]
        x = np.arange(len(results))
        width = 0.35
        bars1 = ax2.bar(x - width/2, results['D+'], width, label='D+ (到理想解距离)',
                       color=self.colors[0], edgecolor='white')
        bars2 = ax2.bar(x + width/2, results['D-'], width, label='D- (到负理想解距离)',
                       color=self.colors[1], edgecolor='white')
        ax2.set_xlabel('评价对象', fontweight='bold')
        ax2.set_ylabel('距离', fontweight='bold')
        ax2.set_title('(b) 各方案到理想解的距离', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(results['评价对象'])
        ax2.legend()
        ax2.tick_params(axis='x', rotation=15)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_radar(self, data, object_names=None, title="多维度雷达图", save_path=None):
        """
        绘制雷达图比较各方案
        
        :param data: 标准化后的数据DataFrame
        :param object_names: 对象名称
        :param title: 图标题
        :param save_path: 保存路径
        """
        if isinstance(data, pd.DataFrame):
            indicators = list(data.columns)
            if object_names is None:
                object_names = list(data.index)
            data = data.values
        else:
            indicators = [f"指标{i+1}" for i in range(data.shape[1])]
            if object_names is None:
                object_names = [f"方案{i+1}" for i in range(data.shape[0])]
        
        n_indicators = len(indicators)
        angles = np.linspace(0, 2 * np.pi, n_indicators, endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        for i, (name, row) in enumerate(zip(object_names, data)):
            values = row.tolist()
            values += values[:1]  # 闭合
            ax.plot(angles, values, 'o-', linewidth=2, label=name, 
                   color=self.colors[i % len(self.colors)])
            ax.fill(angles, values, alpha=0.1, color=self.colors[i % len(self.colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(indicators, fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_heatmap(self, data, title="评价矩阵热力图", save_path=None):
        """
        绘制评价矩阵热力图
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 标准化用于显示
        data_norm = (data - data.min()) / (data.max() - data.min())
        
        im = ax.imshow(data_norm.values, cmap='RdYlGn', aspect='auto')
        
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_xticklabels(data.columns)
        ax.set_yticklabels(data.index)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值标注
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                text = ax.text(j, i, f"{data.iloc[i, j]:.1f}",
                              ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_full_report(self, data, weights, topsis_results, save_path=None):
        """
        生成完整评价报告图
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 子图1: 原始数据热力图
        ax1 = fig.add_subplot(2, 2, 1)
        data_norm = (data - data.min()) / (data.max() - data.min())
        im1 = ax1.imshow(data_norm.values, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(np.arange(len(data.columns)))
        ax1.set_yticks(np.arange(len(data.index)))
        ax1.set_xticklabels(data.columns, fontsize=9)
        ax1.set_yticklabels(data.index, fontsize=9)
        ax1.set_title('(a) 评价矩阵', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 子图2: 权重分布
        ax2 = fig.add_subplot(2, 2, 2)
        bars = ax2.bar(weights.index, weights.values, color=self.colors[:len(weights)],
                      edgecolor='white', linewidth=2)
        ax2.set_ylabel('权重', fontweight='bold')
        ax2.set_title('(b) 熵权法权重', fontsize=12, fontweight='bold')
        for bar, val in zip(bars, weights.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax2.tick_params(axis='x', rotation=15)
        
        # 子图3: TOPSIS排序
        ax3 = fig.add_subplot(2, 2, 3)
        sorted_results = topsis_results.sort_values('贴近度', ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_results)))
        ax3.barh(sorted_results['评价对象'], sorted_results['贴近度'],
                color=colors, edgecolor='white', linewidth=2)
        ax3.set_xlabel('贴近度', fontweight='bold')
        ax3.set_title('(c) TOPSIS排序结果', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 1)
        for i, (_, row) in enumerate(sorted_results.iterrows()):
            ax3.text(row['贴近度'] + 0.02, i, f"#{int(row['排名'])}", 
                    va='center', fontweight='bold', fontsize=10)
        
        # 子图4: 雷达图
        ax4 = fig.add_subplot(2, 2, 4, polar=True)
        indicators = list(data.columns)
        n_indicators = len(indicators)
        angles = np.linspace(0, 2 * np.pi, n_indicators, endpoint=False).tolist()
        angles += angles[:1]
        
        for i, (name, row) in enumerate(data_norm.iterrows()):
            values = row.tolist()
            values += values[:1]
            ax4.plot(angles, values, 'o-', linewidth=2, label=name,
                    color=self.colors[i % len(self.colors)])
            ax4.fill(angles, values, alpha=0.1)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(indicators, fontsize=9)
        ax4.set_title('(d) 多维度雷达图', fontsize=12, fontweight='bold', y=1.08)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
        
        plt.suptitle('综合评价分析报告 (Comprehensive Evaluation Report)', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 第六部分：敏感性分析模块 (Sensitivity Analysis)
# ============================================================

class SensitivityAnalyzer:
    """敏感性分析类 - 分析权重变化对结果的影响"""
    
    def __init__(self, data, indicator_types=None):
        """
        初始化
        :param data: 评价数据
        :param indicator_types: 指标类型
        """
        self.data = data
        self.indicator_types = indicator_types
        self.results = {}
    
    def weight_sensitivity(self, base_weights, perturbation_range=0.1, n_samples=100):
        """
        权重扰动敏感性分析
        
        :param base_weights: 基准权重
        :param perturbation_range: 扰动范围 (±%)
        :param n_samples: 采样次数
        """
        results = []
        
        for _ in range(n_samples):
            # 扰动权重
            perturbed = base_weights * (1 + np.random.uniform(-perturbation_range, 
                                                               perturbation_range, 
                                                               len(base_weights)))
            perturbed = perturbed / perturbed.sum()  # 归一化
            
            # 重新计算TOPSIS
            topsis = TOPSIS(verbose=False)
            topsis.fit(self.data, weights=perturbed, indicator_types=self.indicator_types)
            
            results.append({
                'weights': perturbed,
                'rankings': np.argsort(-topsis.closeness) + 1,
                'closeness': topsis.closeness
            })
        
        self.results['weight_sensitivity'] = results
        return self
    
    def plot_sensitivity(self, save_path=None):
        """绘制敏感性分析结果"""
        if 'weight_sensitivity' not in self.results:
            print("请先运行 weight_sensitivity() 方法")
            return
        
        results = self.results['weight_sensitivity']
        n_objects = len(results[0]['closeness'])
        
        # 收集所有贴近度
        all_closeness = np.array([r['closeness'] for r in results])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 箱线图
        ax1 = axes[0]
        bp = ax1.boxplot(all_closeness, labels=[f"方案{i+1}" for i in range(n_objects)],
                        patch_artist=True)
        colors = PlotStyleConfig.PALETTE[:n_objects]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax1.set_xlabel('评价对象', fontweight='bold')
        ax1.set_ylabel('贴近度', fontweight='bold')
        ax1.set_title('(a) 权重扰动下的贴近度分布', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 排名稳定性
        ax2 = axes[1]
        all_rankings = np.array([r['rankings'] for r in results])
        ranking_std = all_rankings.std(axis=0)
        bars = ax2.bar(range(n_objects), ranking_std, color=colors, edgecolor='white')
        ax2.set_xlabel('评价对象', fontweight='bold')
        ax2.set_ylabel('排名标准差', fontweight='bold')
        ax2.set_title('(b) 排名稳定性分析', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(n_objects))
        ax2.set_xticklabels([f"方案{i+1}" for i in range(n_objects)])
        
        plt.suptitle('权重敏感性分析 (Weight Sensitivity Analysis)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 第七部分：组合评价模型 (Combined Evaluation Model)
# ============================================================

class CombinedEvaluation:
    """
    组合评价模型
    结合熵权法和TOPSIS法进行综合评价
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.entropy_method = None
        self.topsis = None
        self.weights = None
        self.results = None
        self.visualizer = EvaluationVisualizer()
    
    def fit(self, data, indicator_types=None):
        """
        执行组合评价
        
        :param data: 评价数据 (DataFrame)
        :param indicator_types: 指标类型列表
        :return: self
        """
        # Step 1: 熵权法计算权重
        self.entropy_method = EntropyWeightMethod(verbose=self.verbose)
        self.entropy_method.fit(data, indicator_types=indicator_types)
        self.weights = self.entropy_method.get_weights()
        
        # Step 2: TOPSIS评价
        self.topsis = TOPSIS(verbose=self.verbose)
        self.topsis.fit(data, weights=self.weights, indicator_types=indicator_types)
        self.results = self.topsis.get_results()
        
        return self
    
    def get_results(self):
        """获取评价结果"""
        return self.results
    
    def get_weights(self):
        """获取权重"""
        return self.weights
    
    def get_best(self):
        """获取最优方案"""
        return self.topsis.get_best()
    
    def plot_report(self, data, save_path=None):
        """生成完整报告"""
        self.visualizer.plot_full_report(data, self.weights, self.results, save_path)


# ============================================================
# 第八部分：主程序与完整示例 (Main Program)
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("   EVALUATION MODELS FOR MCM/ICM")
    print("   评价类模型 - 熵权法 + TOPSIS法")
    print("   Extended Version with Visualization & Analysis")
    print("="*70)
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    📊 评价模型分析流程                            ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║   [第1步] 数据准备 ──→ 收集评价指标数据                           ║
    ║      │                                                           ║
    ║      ├─ 确定评价对象（方案/项目/供应商）                          ║
    ║      └─ 确定评价指标（正向/负向）                                 ║
    ║                                                                  ║
    ║   [第2步] 熵权法 ──→ 客观确定指标权重                             ║
    ║      │                                                           ║
    ║      ├─ 数据标准化                                               ║
    ║      ├─ 计算信息熵                                               ║
    ║      └─ 计算权重（熵值越小，权重越大）                            ║
    ║                                                                  ║
    ║   [第3步] TOPSIS法 ──→ 方案排序                                  ║
    ║      │                                                           ║
    ║      ├─ 加权标准化矩阵                                           ║
    ║      ├─ 确定正/负理想解                                          ║
    ║      ├─ 计算各方案到理想解的距离                                  ║
    ║      └─ 计算相对贴近度，排序                                      ║
    ║                                                                  ║
    ║   [第4步] 敏感性分析 ──→ 验证结果稳定性                           ║
    ║                                                                  ║
    ║   [第5步] 可视化输出 ──→ 生成论文级图表                           ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # ================================================================
    # 示例1：供应商选择问题
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 1: 供应商选择问题 (Supplier Selection)")
    print("="*70)
    
    # 1. 数据准备
    print("\n📊 Step 1: 数据准备")
    print("-" * 50)
    
    preprocessor = EvaluationDataPreprocessor(random_seed=2026)
    preprocessor.generate_demo_data(n_objects=6, scenario='supplier')
    preprocessor.summary()
    
    data = preprocessor.get_data()
    indicator_types = preprocessor.indicator_types
    
    # 2. 组合评价
    print("\n📊 Step 2: 组合评价（熵权法 + TOPSIS）")
    print("-" * 50)
    
    evaluator = CombinedEvaluation(verbose=True)
    evaluator.fit(data, indicator_types=indicator_types)
    
    best_name, best_score = evaluator.get_best()
    print(f"\n🏆 最优方案: {best_name} (贴近度: {best_score:.4f})")
    
    # 3. 可视化
    print("\n📊 Step 3: 可视化分析")
    print("-" * 50)
    
    visualizer = EvaluationVisualizer()
    
    # 权重分布图
    visualizer.plot_weights(evaluator.get_weights(), title="供应商评价指标权重分布")
    
    # TOPSIS结果图
    visualizer.plot_topsis_results(evaluator.get_results(), title="供应商综合评价结果")
    
    # 雷达图
    visualizer.plot_radar(data, title="供应商多维度对比雷达图")
    
    # 完整报告
    evaluator.plot_report(data)
    
    # 4. 敏感性分析
    print("\n📊 Step 4: 敏感性分析")
    print("-" * 50)
    
    sensitivity = SensitivityAnalyzer(data, indicator_types=indicator_types)
    sensitivity.weight_sensitivity(evaluator.get_weights().values, perturbation_range=0.2, n_samples=200)
    sensitivity.plot_sensitivity()
    
    # ================================================================
    # 示例2：项目评估问题
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 2: 项目评估问题 (Project Evaluation)")
    print("="*70)
    
    preprocessor2 = EvaluationDataPreprocessor(random_seed=2026)
    preprocessor2.generate_demo_data(n_objects=5, scenario='project')
    preprocessor2.summary()
    
    data2 = preprocessor2.get_data()
    indicator_types2 = preprocessor2.indicator_types
    
    evaluator2 = CombinedEvaluation(verbose=True)
    evaluator2.fit(data2, indicator_types=indicator_types2)
    evaluator2.plot_report(data2)
    
    # ================================================================
    # 使用说明
    # ================================================================
    print("\n" + "="*70)
    print("📖 使用说明 (Usage Guide)")
    print("="*70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                     评价模型使用指南                             │
    └─────────────────────────────────────────────────────────────────┘
    
    【如何使用自己的数据】
    
    1️⃣ 从CSV文件加载：
       preprocessor = EvaluationDataPreprocessor()
       preprocessor.load_from_csv("your_data.csv")
       preprocessor.set_indicator_types(['positive', 'negative', ...])
    
    2️⃣ 从DataFrame加载：
       preprocessor.load_from_dataframe(your_df)
    
    3️⃣ 从数组加载：
       preprocessor.load_from_array(your_array, indicator_names, object_names)
    
    【指标类型说明】
    
    - 'positive': 正向指标（越大越好）如：收益、质量、效率
    - 'negative': 负向指标（越小越好）如：成本、风险、时间
    
    【论文图表建议】
    
    Figure 1: 评价指标体系（树形图或表格）
    Figure 2: 熵权法权重分布（条形图+饼图）
    Figure 3: TOPSIS评价结果（贴近度排序）
    Figure 4: 多维度雷达图对比
    Figure 5: 敏感性分析（箱线图）
    
    Table 1: 原始数据矩阵
    Table 2: 标准化矩阵
    Table 3: 熵权法权重计算过程
    Table 4: TOPSIS评价结果
    """)
    
    print("\n" + "="*70)
    print("   ✅ All examples completed successfully!")
    print("   💡 Use the above code templates for your MCM/ICM paper")
    print("="*70)
