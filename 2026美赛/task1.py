"""
============================================================
AI 职业演化预测模型 (AI Career Evolution Prediction Model)
============================================================
功能：预测AI影响下不同职业的长期劳动力需求变化
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================

模型框架：
1. 灰色预测基准模型（GM(1,1)）- 预测无AI干预的自然增长趋势
2. 技术渗透速度模型（Logistic S-Curve）- 模拟Gen-AI技术扩散
3. 价值重构叠加模型（Task-Based Recomposition）- 计算替代与创造后的真实需求
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
import os
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

# ============================================================
# 图表配置
# ============================================================

class PlotStyleConfig:
    """图表美化配置类"""
    
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'success': '#27AE60',
        'danger': '#C73E1D',
        'neutral': '#3B3B3B',
        'background': '#FAFAFA',
        'grid': '#E0E0E0'
    }
    
    PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4C9A', '#1B998B', '#E94F37', '#44AF69']
    
    @staticmethod
    def setup_style(style='academic'):
        """设置全局绘图风格"""
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                plt.style.use('default')
        rcParams['figure.figsize'] = (12, 8)
        rcParams['figure.dpi'] = 100
        rcParams['savefig.dpi'] = 300
        rcParams['font.size'] = 11
        rcParams['axes.titlesize'] = 14
        rcParams['axes.labelsize'] = 12
        rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        rcParams['axes.unicode_minus'] = False
    
    @staticmethod
    def get_palette(n=None):
        """获取配色板"""
        palette = PlotStyleConfig.PALETTE
        if n is not None:
            if n <= len(palette):
                return palette[:n]
            else:
                return [palette[i % len(palette)] for i in range(n)]
        return palette


class FigureSaver:
    """图表保存工具类"""
    
    def __init__(self, save_dir='./figures', format='png'):
        self.save_dir = save_dir
        self.format = format
        os.makedirs(save_dir, exist_ok=True)
        
    def save(self, fig, filename, formats=None, tight=True):
        if formats is None:
            formats = [self.format]
        if tight:
            fig.tight_layout()
        paths = []
        for fmt in formats:
            path = os.path.join(self.save_dir, f"{filename}.{fmt}")
            fig.savefig(path, format=fmt, bbox_inches='tight', 
                       facecolor=fig.get_facecolor(), edgecolor='none')
            paths.append(path)
            print(f"  📊 图表已保存: {path}")
        return paths

# 设置绘图风格
PlotStyleConfig.setup_style('academic')

# ============================================================
# 第一部分：数据占位符 (Data Placeholders)
# ============================================================

def load_historical_data(file_path='就业人数.csv'):
    """
    加载历史就业数据
    
    :param file_path: CSV文件路径，格式：career, 2016, 2017, ..., 2023
    :return: DataFrame，包含职业名称和历史数据
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"⚠️  文件 {file_path} 未找到，使用示例数据")
        # 示例数据占位符
        data = {
            'career': ['software_engineer'],
            '2016': [125.62],
            '2017': [131.00],
            '2018': [136.55],
            '2019': [146.92],
            '2020': [184.79],
            '2021': [162.22],
            '2022': [179.53],
            '2023': [189.71]
        }
        return pd.DataFrame(data)


def extract_career_data(df, career_name):
    """
    从DataFrame中提取指定职业的历史数据
    
    :param df: 包含所有职业数据的DataFrame
    :param career_name: 职业名称
    :return: tuple (historical_data, start_year)
        - historical_data: 历史数据数组
        - start_year: 起始年份
    """
    # 清理列名（去除前后空格）
    df.columns = df.columns.str.strip()
    
    if career_name not in df['career'].values:
        available_careers = df['career'].tolist()
        raise ValueError(f"职业 '{career_name}' 不在数据文件中。可用职业: {available_careers}")
    
    career_row = df[df['career'] == career_name].iloc[0]
    
    # 获取所有年份列（排除'career'列）
    year_cols = [col for col in df.columns if col != 'career' and col.strip().isdigit()]
    year_cols = sorted(year_cols, key=lambda x: int(x.strip()))
    
    if len(year_cols) == 0:
        raise ValueError(f"未找到有效的年份列")
    
    historical_data = [career_row[col] for col in year_cols]
    start_year = int(year_cols[0].strip())
    
    # 检查数据有效性
    historical_data = [float(x) if pd.notna(x) else 0.0 for x in historical_data]
    
    return np.array(historical_data), start_year


def get_all_careers(df):
    """
    获取数据文件中所有职业名称列表
    
    :param df: 包含所有职业数据的DataFrame
    :return: 职业名称列表
    """
    return df['career'].tolist()


def get_career_dimensions(career_name):
    """
    获取职业的四个关键维度参数
    
    ★★★ 数据占位符：请根据实际职业特征填写以下参数 ★★★
    
    :param career_name: 职业名称
    :return: dict，包含D1, D2, D3, D4四个维度
    """
    # 为三个职业分别设置维度参数
    dimensions = {
        'software_engineer': {
            'D1': 0.85,      # 任务自动化潜力（0-1），决定S曲线天花板L
                            # 软件工程师：AI自动化潜力高，代码生成、测试等可被AI辅助
            'D2': 0.15,      # 技能演进需求（0-1），决定S曲线斜率k
                            # 软件工程师：需要快速学习新AI工具，技能演进较快
            'D3': 0.25,      # 市场需求弹性（0-1），决定新岗位增量
                            # 软件工程师：AI工具提升效率，可能创造新需求
            'D4': 0.15       # 人本/物理约束（0-1），决定核心防御区
                            # 软件工程师：需要创造性思维、架构设计等，约束较低
        },
        'chef': {
            'D1': 0.25,      # 任务自动化潜力（0-1）
                            # 厨师：烹饪需要人工操作，自动化潜力较低
            'D2': 0.10,      # 技能演进需求（0-1）
                            # 厨师：传统技能为主，AI工具应用较慢
            'D3': 0.15,      # 市场需求弹性（0-1）
                            # 厨师：餐饮需求相对稳定，弹性中等
            'D4': 0.60       # 人本/物理约束（0-1）
                            # 厨师：需要人工操作、创意、服务，约束很高
        },
        'graphic_designer': {
            'D1': 0.55,      # 任务自动化潜力（0-1）
                            # 平面设计师：AI可以辅助设计，但创意部分仍需人工
            'D2': 0.12,      # 技能演进需求（0-1）
                            # 平面设计师：需要学习AI设计工具，演进速度中等
            'D3': 0.20,      # 市场需求弹性（0-1）
                            # 平面设计师：AI工具可能创造新需求，但替代也明显
            'D4': 0.40       # 人本/物理约束（0-1）
                            # 平面设计师：创意、审美、客户沟通等需要人工，约束中等
        },
    }
    
    # 如果职业不在字典中，返回默认值
    if career_name not in dimensions:
        print(f"⚠️  职业 '{career_name}' 未找到，使用默认参数")
        return {
            'D1': 0.50,
            'D2': 0.12,
            'D3': 0.20,
            'D4': 0.30
        }
    
    return dimensions[career_name]


# ============================================================
# 第二部分：模型1 - 灰色预测基准模型 (GM(1,1))
# ============================================================

class GreyModel:
    """
    灰色预测模型 GM(1,1)
    
    用于预测无AI干预下的自然增长趋势
    
    改进后的预测公式：
    - 增长率：r = e^(-a) - 1
    - 自然趋势：Y_t = x^(0)(n) × (1+r)^(t-n)
    其中 x^(0)(n) 是最后一个历史数据点，t 是预测年份，n 是最后一个历史数据点的索引
    """
    
    def __init__(self, data):
        """
        :param data: 一维数组，历史数据序列
        """
        self.data = np.array(data, dtype=float)
        self.n = len(data)
        self.a = None  # 发展系数
        self.b = None  # 灰作用量
        self.r = None  # 增长率 r = e^(-a) - 1
        self.fitted = False
    
    def fit(self):
        """拟合GM(1,1)模型"""
        if self.n < 4:
            raise ValueError("数据点数量不足，至少需要4个点")
        
        # 1. 一次累加生成（AGO）
        x1 = np.cumsum(self.data)
        
        # 2. 构造数据矩阵B和Y
        B = np.zeros((self.n - 1, 2))
        Y = np.zeros(self.n - 1)
        
        for i in range(self.n - 1):
            B[i, 0] = -(x1[i] + x1[i + 1]) / 2
            B[i, 1] = 1
            Y[i] = self.data[i + 1]
        
        # 3. 最小二乘估计
        try:
            params = np.linalg.lstsq(B, Y, rcond=None)[0]
            self.a = params[0]
            self.b = params[1]
        except:
            # 如果求解失败，使用备用方法
            self.a = -0.01
            self.b = np.mean(self.data)
        
        # 4. 计算增长率 r = e^(-a) - 1
        self.r = np.exp(-self.a) - 1
        
        self.fitted = True
    
    def predict(self, steps=10):
        """
        预测未来值
        
        使用改进的自然趋势公式：
        - 增长率：r = e^(-a) - 1（已在fit()中计算）
        - 预测公式：Y_t = x^(0)(n) × (1+r)^(t-n)
        
        其中：
        - x^(0)(n) 是最后一个历史数据点
        - t = n + k (k = 1, 2, ..., steps)
        - 所以 t - n = k
        
        :param steps: 预测步数
        :return: 预测值数组
        """
        if not self.fitted:
            self.fit()
        
        # 获取最后一个历史数据点 x^(0)(n)
        x_last = self.data[-1]
        
        predictions = []
        for k in range(1, steps + 1):
            # t = n + k，所以 t - n = k
            # 使用自然趋势公式：Y_t = x^(0)(n) × (1+r)^(t-n)
            Y_t = x_last * ((1 + self.r) ** k)
            predictions.append(max(0, Y_t))  # 确保非负
        
        return np.array(predictions)
    
    def get_growth_rate(self):
        """
        获取增长率 r
        
        :return: 增长率 r = e^(-a) - 1
        """
        if not self.fitted:
            self.fit()
        return self.r
    
    def get_trend(self):
        """
        获取趋势方向（基于增长率 r）
        
        :return: 趋势描述字符串
        """
        if not self.fitted:
            self.fit()
        
        r = self.r
        
        if r < -0.2:
            return "快速下降"
        elif r < -0.05:
            return "缓慢下降"
        elif r < 0.05:
            return "平稳"
        elif r < 0.2:
            return "缓慢增长"
        else:
            return "快速增长"


# ============================================================
# 第三部分：模型2 - 技术渗透速度模型 (Logistic S-Curve)
# ============================================================

class TechnologyPenetrationModel:
    """
    技术渗透速度模型（Logistic S-Curve）
    
    模拟Gen-AI技术在该职业任务中的扩散广度与速度
    """
    
    def __init__(self, D1, D2, t0=2024):
        """
        :param D1: 任务自动化潜力（0-1），决定饱和上限L
        :param D2: 技能演进需求（0-1），决定增长斜率k
        :param t0: 起始年份（默认2024）
        """
        self.D1 = D1
        self.D2 = D2
        self.t0 = t0
        
        # 参数映射
        self.L = D1  # 饱和上限（渗透率上限）
        self.k = 0.1 + D2 * 0.3  # 增长斜率（0.1-0.4）
    
    def penetration_rate(self, t):
        """
        计算时间点t的AI渗透率
        
        :param t: 时间（年份）
        :return: 渗透率 P(t) ∈ [0, L]
        """
        if t < self.t0:
            return 0.0
        
        # Logistic公式: P(t) = L / (1 + exp(-k*(t-t0)))
        dt = t - self.t0
        P = self.L / (1 + np.exp(-self.k * dt))
        return min(P, self.L)  # 确保不超过上限
    
    def predict(self, years):
        """
        预测多个时间点的渗透率
        
        :param years: 年份数组
        :return: 渗透率数组
        """
        return np.array([self.penetration_rate(t) for t in years])


# ============================================================
# 第四部分：模型3 - 价值重构叠加模型 (Task-Based Recomposition)
# ============================================================

class ValueRecompositionModel:
    """
    价值重构叠加模型
    
    计算AI"替代"与"创造"后的真实劳动力需求
    """
    
    def __init__(self, D3, D4, A=1.5):
        """
        :param D3: 市场需求弹性（0-1），决定新岗位增量
        :param D4: 人本/物理约束（0-1），决定核心防御区
        :param A: AI增强系数（默认1.5，表示使用AI后效率提升50%）
        """
        self.D3 = D3
        self.D4 = D4
        self.A = A
    
    def new_market_increment(self, t, t0=2024):
        """
        计算新市场增量 N(D3, t)
        
        :param t: 时间（年份）
        :param t0: 起始年份
        :return: 增量因子
        """
        dt = max(0, t - t0)
        # 增量随时间逐渐增加，受D3影响
        # 使用Sigmoid函数模拟渐进式增长
        N = self.D3 * (1 / (1 + np.exp(-0.2 * (dt - 5))))
        return N
    
    def compute_final_demand(self, Y_t, P_t, t):
        """
        计算修正后的最终劳动力需求
        
        :param Y_t: 基准预测值（来自GM模型）
        :param P_t: AI渗透率（来自Logistic模型）
        :param t: 时间（年份）
        :return: 修正后的需求 F(t)
        """
        # 1. 人类核心防御区
        human_core = (1 - P_t) * (1 - self.D4)
        
        # 2. AI增强产出
        ai_enhanced = P_t * self.A
        
        # 3. 新市场增量
        N_t = self.new_market_increment(t)
        
        # 4. 最终需求
        F_t = Y_t * (human_core + ai_enhanced + N_t)
        
        return max(0, F_t)  # 确保非负


# ============================================================
# 第五部分：完整预测模型整合
# ============================================================

class AICareerEvolutionModel:
    """
    AI职业演化预测模型（完整整合）
    
    整合三个子模型，提供完整的预测功能
    """
    
    def __init__(self, career_name, historical_data, dimensions, start_year=2016):
        """
        :param career_name: 职业名称
        :param historical_data: 历史数据数组
        :param dimensions: 四个维度参数字典 {'D1': ..., 'D2': ..., 'D3': ..., 'D4': ...}
        :param start_year: 历史数据起始年份
        """
        self.career_name = career_name
        self.historical_data = np.array(historical_data)
        self.dimensions = dimensions
        self.start_year = start_year
        
        # 初始化三个子模型
        self.grey_model = GreyModel(self.historical_data)
        self.penetration_model = TechnologyPenetrationModel(
            D1=dimensions['D1'],
            D2=dimensions['D2'],
            t0=2024
        )
        self.recomposition_model = ValueRecompositionModel(
            D3=dimensions['D3'],
            D4=dimensions['D4']
        )
        # 拟合灰色模型
        self.grey_model.fit()
    
    def predict(self, end_year=2035):
        """
        预测到指定年份
        
        :param end_year: 预测结束年份
        :return: DataFrame，包含所有中间变量和最终预测
        """
        # 历史年份
        hist_years = np.arange(self.start_year, 2024)
        hist_data = self.historical_data
        
        # 预测年份
        pred_years = np.arange(2024, end_year + 1)
        n_pred = len(pred_years)
        
        # 1. 灰色模型预测基准值
        baseline_pred = self.grey_model.predict(steps=n_pred)
        
        # 2. 技术渗透率预测
        penetration_rates = self.penetration_model.predict(pred_years)
        
        # 3. 价值重构后的最终需求
        final_demand = []
        for i, year in enumerate(pred_years):
            F_t = self.recomposition_model.compute_final_demand(
                Y_t=baseline_pred[i],
                P_t=penetration_rates[i],
                t=year
            )
            final_demand.append(F_t)
        final_demand = np.array(final_demand)
        
        # 4. 构建结果DataFrame
        results = pd.DataFrame({
            'year': pred_years,
            'baseline_Yt': baseline_pred,
            'penetration_Pt': penetration_rates,
            'new_market_Nt': [self.recomposition_model.new_market_increment(t) for t in pred_years],
            'final_demand_Ft': final_demand
        })
        
        return results
    
    def get_historical_df(self):
        """获取历史数据DataFrame"""
        hist_years = np.arange(self.start_year, 2024)
        return pd.DataFrame({
            'year': hist_years,
            'employment': self.historical_data
        })


# ============================================================
# 第六部分：可视化模块
# ============================================================

class CareerVisualization:
    """职业演化预测可视化类"""
    
    def __init__(self, model: AICareerEvolutionModel, save_dir='./figures'):
        """
        :param model: AICareerEvolutionModel实例
        :param save_dir: 图表保存目录
        """
        self.model = model
        self.saver = FigureSaver(save_dir)
        self.career_name = model.career_name  # 保存职业名称用于文件名
    
    def _get_filename(self, base_name):
        """
        生成包含职业名称的文件名（参考task1_1.py的方式）
        
        :param base_name: 基础文件名
        :return: 带职业名称的文件名
        """
        # 参考task1_1.py: f"{occupation_english.replace(' ', '_').lower()}_evolution_complete"
        career_filename = self.career_name.replace(' ', '_').lower()
        return f"{career_filename}_{base_name}"
    
    def plot_complete_evolution(self, end_year=2035, figsize=(16, 10)):
        """
        绘制完整的职业演化预测图（主图）
        
        包含：
        1. 历史数据
        2. 基准预测（GM模型）
        3. AI渗透率曲线
        4. 最终需求预测
        """
        # 获取数据
        hist_df = self.model.get_historical_df()
        pred_df = self.model.predict(end_year=end_year)
        
        # 创建图形和子图
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3, 
                             left=0.08, right=0.95, top=0.95, bottom=0.08)
        
        # 主图：就业人数演化（左上，占2列）
        ax_main = fig.add_subplot(gs[0:2, :])
        
        # 绘制历史数据
        ax_main.plot(hist_df['year'], hist_df['employment'], 
                    'o-', color=PlotStyleConfig.COLORS['primary'], 
                    linewidth=2.5, markersize=8, label='Historical Data', zorder=3)
        
        # 绘制基准预测（GM模型）
        pred_years = pred_df['year'].values
        ax_main.plot(pred_years, pred_df['baseline_Yt'], 
                    '--', color=PlotStyleConfig.COLORS['neutral'], 
                    linewidth=2, alpha=0.7, label='Baseline Prediction (GM(1,1))', zorder=2)
        
        # 绘制最终需求预测
        ax_main.plot(pred_years, pred_df['final_demand_Ft'], 
                    '-', color=PlotStyleConfig.COLORS['accent'], 
                    linewidth=3, label='Final Demand (AI-Adjusted)', zorder=3)
        
        # 填充区域
        ax_main.fill_between(pred_years, pred_df['baseline_Yt'], pred_df['final_demand_Ft'],
                           where=(pred_df['final_demand_Ft'] >= pred_df['baseline_Yt']),
                           alpha=0.2, color=PlotStyleConfig.COLORS['success'], 
                           label='AI Enhancement Zone')
        ax_main.fill_between(pred_years, pred_df['baseline_Yt'], pred_df['final_demand_Ft'],
                           where=(pred_df['final_demand_Ft'] < pred_df['baseline_Yt']),
                           alpha=0.2, color=PlotStyleConfig.COLORS['danger'], 
                           label='AI Displacement Zone')
        
        # 添加2024年分界线
        ax_main.axvline(x=2024, color='red', linestyle=':', linewidth=2, 
                       alpha=0.6, label='AI Era Start (2024)', zorder=1)
        
        ax_main.set_xlabel('Year', fontweight='bold', fontsize=13)
        ax_main.set_ylabel('Employment (万人)', fontweight='bold', fontsize=13)
        ax_main.set_title(f'{self.model.career_name.replace("_", " ").title()} - Career Evolution Prediction', 
                         fontweight='bold', fontsize=16, pad=15)
        ax_main.legend(loc='best', fontsize=10, framealpha=0.9)
        ax_main.grid(True, alpha=0.3, linestyle='--')
        ax_main.set_xlim(hist_df['year'].min() - 1, end_year + 1)
        
        # 子图1：AI渗透率曲线（左下）
        ax1 = fig.add_subplot(gs[2, 0])
        ax1.plot(pred_years, pred_df['penetration_Pt'] * 100, 
                '-', color=PlotStyleConfig.COLORS['secondary'], 
                linewidth=2.5, marker='o', markersize=4)
        ax1.fill_between(pred_years, 0, pred_df['penetration_Pt'] * 100,
                         alpha=0.3, color=PlotStyleConfig.COLORS['secondary'])
        ax1.set_xlabel('Year', fontweight='bold', fontsize=11)
        ax1.set_ylabel('AI Penetration Rate (%)', fontweight='bold', fontsize=11)
        ax1.set_title('Technology Penetration (Logistic S-Curve)', fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(0, 100)
        
        # 添加阶段标注
        ax1.axvspan(2024, 2026, alpha=0.1, color='green', label='Phase 1: Initial')
        ax1.axvspan(2027, 2030, alpha=0.1, color='orange', label='Phase 2: Acceleration')
        ax1.axvspan(2031, 2035, alpha=0.1, color='red', label='Phase 3: Saturation')
        
        # 子图2：新市场增量（右下）
        ax2 = fig.add_subplot(gs[2, 1])
        ax2.plot(pred_years, pred_df['new_market_Nt'] * 100, 
                '-', color=PlotStyleConfig.COLORS['success'], 
                linewidth=2.5, marker='s', markersize=4)
        ax2.fill_between(pred_years, 0, pred_df['new_market_Nt'] * 100,
                         alpha=0.3, color=PlotStyleConfig.COLORS['success'])
        ax2.set_xlabel('Year', fontweight='bold', fontsize=11)
        ax2.set_ylabel('New Market Increment (%)', fontweight='bold', fontsize=11)
        ax2.set_title('Market Elasticity Effect (D3)', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle('AI Career Evolution Prediction Model - Complete Analysis', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        filename = self._get_filename('evolution_complete')
        paths = self.saver.save(fig, filename, formats=['png', 'pdf'])
        print(f"    💾 Complete evolution plot saved: {paths[0]}")
        plt.show()
        
        return fig
    
    def plot_model_components(self, end_year=2035, figsize=(14, 8)):
        """
        绘制三个子模型的详细分解图
        """
        hist_df = self.model.get_historical_df()
        pred_df = self.model.predict(end_year=end_year)
        pred_years = pred_df['year'].values
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Components Breakdown', fontsize=16, fontweight='bold', y=0.98)
        
        # 图1：GM(1,1)基准预测
        ax1 = axes[0, 0]
        ax1.plot(hist_df['year'], hist_df['employment'], 
                'o-', color=PlotStyleConfig.COLORS['primary'], 
                linewidth=2, markersize=6, label='Historical')
        ax1.plot(pred_years, pred_df['baseline_Yt'], 
                '--', color=PlotStyleConfig.COLORS['neutral'], 
                linewidth=2.5, label='GM(1,1) Prediction')
        ax1.axvline(x=2024, color='red', linestyle=':', linewidth=1.5, alpha=0.6)
        ax1.set_xlabel('Year', fontweight='bold')
        ax1.set_ylabel('Employment (万人)', fontweight='bold')
        ax1.set_title('Model 1: Grey Prediction (Baseline Trend)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2：Logistic渗透曲线
        ax2 = axes[0, 1]
        ax2.plot(pred_years, pred_df['penetration_Pt'] * 100, 
                '-', color=PlotStyleConfig.COLORS['secondary'], 
                linewidth=2.5, marker='o', markersize=5)
        ax2.fill_between(pred_years, 0, pred_df['penetration_Pt'] * 100,
                         alpha=0.3, color=PlotStyleConfig.COLORS['secondary'])
        ax2.set_xlabel('Year', fontweight='bold')
        ax2.set_ylabel('Penetration Rate (%)', fontweight='bold')
        ax2.set_title('Model 2: Technology Penetration (Logistic S-Curve)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # 添加维度标注
        D1 = self.model.dimensions['D1']
        D2 = self.model.dimensions['D2']
        ax2.text(0.05, 0.95, f'D1 (Automation Potential) = {D1:.2f}\nD2 (Skill Evolution) = {D2:.2f}',
                transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 图3：价值重构分解
        ax3 = axes[1, 0]
        human_core = (1 - pred_df['penetration_Pt']) * (1 - self.model.dimensions['D4'])
        ai_enhanced = pred_df['penetration_Pt'] * self.model.recomposition_model.A
        new_market = pred_df['new_market_Nt']
        
        ax3.plot(pred_years, human_core * 100, '-', label='Human Core Defense', 
                color=PlotStyleConfig.COLORS['primary'], linewidth=2)
        ax3.plot(pred_years, ai_enhanced * 100, '-', label='AI-Enhanced Output', 
                color=PlotStyleConfig.COLORS['accent'], linewidth=2)
        ax3.plot(pred_years, new_market * 100, '-', label='New Market Increment', 
                color=PlotStyleConfig.COLORS['success'], linewidth=2)
        ax3.set_xlabel('Year', fontweight='bold')
        ax3.set_ylabel('Component Contribution (%)', fontweight='bold')
        ax3.set_title('Model 3: Value Recomposition Components', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 图4：最终对比
        ax4 = axes[1, 1]
        ax4.plot(hist_df['year'], hist_df['employment'], 
                'o-', color=PlotStyleConfig.COLORS['primary'], 
                linewidth=2, markersize=6, label='Historical', zorder=3)
        ax4.plot(pred_years, pred_df['baseline_Yt'], 
                '--', color=PlotStyleConfig.COLORS['neutral'], 
                linewidth=2, label='Baseline (No AI)', alpha=0.7, zorder=2)
        ax4.plot(pred_years, pred_df['final_demand_Ft'], 
                '-', color=PlotStyleConfig.COLORS['accent'], 
                linewidth=3, label='Final (AI-Adjusted)', zorder=3)
        ax4.axvline(x=2024, color='red', linestyle=':', linewidth=2, alpha=0.6)
        ax4.set_xlabel('Year', fontweight='bold')
        ax4.set_ylabel('Employment (万人)', fontweight='bold')
        ax4.set_title('Final Comparison: Baseline vs AI-Adjusted', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = self._get_filename('model_components')
        paths = self.saver.save(fig, filename, formats=['png', 'pdf'])
        print(f"    💾 Model components plot saved: {paths[0]}")
        plt.show()
        
        return fig
    
    def plot_dimension_sensitivity(self, figsize=(14, 10)):
        """
        绘制四个维度参数的敏感性分析
        """
        hist_df = self.model.get_historical_df()
        base_dims = self.model.dimensions.copy()
        
        # 测试每个维度变化±30%的影响
        variations = [-0.3, -0.15, 0, 0.15, 0.3]
        dim_names = ['D1', 'D2', 'D3', 'D4']
        dim_labels = ['Automation Potential', 'Skill Evolution', 'Market Elasticity', 'Human Constraints']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Dimension Sensitivity Analysis (±30% Variation)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        for idx, (dim, label) in enumerate(zip(dim_names, dim_labels)):
            ax = axes[idx // 2, idx % 2]
            pred_years = np.arange(2024, 2036)
            
            for var in variations:
                # 修改维度值
                test_dims = base_dims.copy()
                test_dims[dim] = base_dims[dim] * (1 + var)
                
                # 创建临时模型
                temp_model = AICareerEvolutionModel(
                    self.model.career_name,
                    self.model.historical_data,
                    test_dims,
                    self.model.start_year
                )
                temp_pred = temp_model.predict(end_year=2035)
                
                # 绘制
                color_intensity = 0.3 + abs(var) * 0.7 / 0.3
                alpha = 0.4 + abs(var) * 0.4 / 0.3
                linestyle = '-' if var == 0 else '--' if var < 0 else '-.'
                
                ax.plot(temp_pred['year'], temp_pred['final_demand_Ft'],
                       linestyle=linestyle, linewidth=2 if var == 0 else 1.5,
                       alpha=alpha, 
                       label=f'{var*100:+.0f}%' if var != 0 else 'Baseline',
                       color=PlotStyleConfig.COLORS['primary'] if var <= 0 
                       else PlotStyleConfig.COLORS['accent'])
            
            ax.set_xlabel('Year', fontweight='bold')
            ax.set_ylabel('Final Demand (万人)', fontweight='bold')
            ax.set_title(f'{dim}: {label}', fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = self._get_filename('sensitivity_analysis')
        paths = self.saver.save(fig, filename, formats=['png', 'pdf'])
        print(f"    💾 Sensitivity analysis plot saved: {paths[0]}")
        plt.show()
        
        return fig
    
    def plot_phase_analysis(self, end_year=2035, figsize=(14, 6)):
        """
        绘制三个阶段的分析图
        """
        pred_df = self.model.predict(end_year=end_year)
        pred_years = pred_df['year'].values
        
        # 定义三个阶段
        phase1 = (pred_years >= 2024) & (pred_years <= 2026)
        phase2 = (pred_years >= 2027) & (pred_years <= 2030)
        phase3 = (pred_years >= 2031) & (pred_years <= 2035)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Three-Phase Evolution Analysis', fontsize=16, fontweight='bold', y=1.02)
        
        phases = [
            (phase1, 'Phase 1: Initial (2024-2026)', PlotStyleConfig.COLORS['success']),
            (phase2, 'Phase 2: Acceleration (2027-2030)', PlotStyleConfig.COLORS['accent']),
            (phase3, 'Phase 3: Saturation (2031-2035)', PlotStyleConfig.COLORS['secondary'])
        ]
        
        for idx, (phase_mask, title, color) in enumerate(phases):
            ax = axes[idx]
            phase_years = pred_years[phase_mask]
            phase_demand = pred_df['final_demand_Ft'].values[phase_mask]
            phase_baseline = pred_df['baseline_Yt'].values[phase_mask]
            phase_penetration = pred_df['penetration_Pt'].values[phase_mask] * 100
            
            # 绘制需求曲线
            ax2 = ax.twinx()
            ax.plot(phase_years, phase_demand, '-', color=color, linewidth=3, 
                   marker='o', markersize=6, label='Final Demand', zorder=3)
            ax.plot(phase_years, phase_baseline, '--', color=PlotStyleConfig.COLORS['neutral'], 
                   linewidth=2, alpha=0.7, label='Baseline', zorder=2)
            ax2.plot(phase_years, phase_penetration, '-', color=PlotStyleConfig.COLORS['secondary'], 
                    linewidth=2, marker='s', markersize=4, label='AI Penetration (%)', alpha=0.7)
            
            ax.fill_between(phase_years, phase_baseline, phase_demand,
                           where=(phase_demand >= phase_baseline),
                           alpha=0.2, color=PlotStyleConfig.COLORS['success'])
            ax.fill_between(phase_years, phase_baseline, phase_demand,
                           where=(phase_demand < phase_baseline),
                           alpha=0.2, color=PlotStyleConfig.COLORS['danger'])
            
            ax.set_xlabel('Year', fontweight='bold')
            ax.set_ylabel('Employment (万人)', fontweight='bold', color=color)
            ax2.set_ylabel('AI Penetration (%)', fontweight='bold', 
                           color=PlotStyleConfig.COLORS['secondary'])
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', labelcolor=color)
            ax2.tick_params(axis='y', labelcolor=PlotStyleConfig.COLORS['secondary'])
            
            # 添加统计信息
            change_pct = ((phase_demand[-1] - phase_demand[0]) / phase_demand[0] * 100) if len(phase_demand) > 0 else 0
            ax.text(0.05, 0.95, f'Change: {change_pct:+.1f}%',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        filename = self._get_filename('phase_analysis')
        paths = self.saver.save(fig, filename, formats=['png', 'pdf'])
        print(f"    💾 Phase analysis plot saved: {paths[0]}")
        plt.show()
        
        return fig
    
    def plot_dimension_radar(self, figsize=(10, 10)):
        """
        绘制四个维度的雷达图
        """
        dims = self.model.dimensions
        dim_names = ['D1\nAutomation\nPotential', 'D2\nSkill\nEvolution', 
                    'D3\nMarket\nElasticity', 'D4\nHuman\nConstraints']
        values = [dims['D1'], dims['D2'], dims['D3'], dims['D4']]
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(dim_names), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2.5, color=PlotStyleConfig.COLORS['primary'])
        ax.fill(angles, values, alpha=0.25, color=PlotStyleConfig.COLORS['primary'])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dim_names, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{self.model.career_name.replace("_", " ").title()} - Dimension Profile',
                    fontsize=14, fontweight='bold', pad=20)
        
        # 添加数值标注
        for angle, value, name in zip(angles[:-1], values[:-1], dim_names):
            ax.text(angle, value + 0.05, f'{value:.2f}', 
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        filename = self._get_filename('dimension_radar')
        paths = self.saver.save(fig, filename, formats=['png', 'pdf'])
        print(f"    💾 Dimension radar plot saved: {paths[0]}")
        plt.show()
        
        return fig


# ============================================================
# 第七部分：主工作流
# ============================================================

def run_complete_workflow(career_name='software_engineer', data_file='就业人数.csv', 
                         end_year=2035, save_dir='./figures'):
    """
    运行完整的AI职业演化预测工作流
    
    :param career_name: 职业名称
    :param data_file: 历史数据文件路径
    :param end_year: 预测结束年份
    :param save_dir: 图表保存目录
    :return: 模型和可视化对象
    """
    print("\n" + "="*70)
    print("AI 职业演化预测模型 - 完整工作流")
    print("AI Career Evolution Prediction Model - Complete Workflow")
    print("="*70 + "\n")
    
    # Step 1: 加载数据
    print("【Step 1】加载历史数据...")
    df = load_historical_data(data_file)
    
    # 提取指定职业的数据
    if career_name not in df['career'].values:
        print(f"⚠️  职业 '{career_name}' 不在数据文件中，使用第一个职业")
        career_name = df['career'].iloc[0]
    
    historical_data, start_year = extract_career_data(df, career_name)
    
    print(f"  职业: {career_name}")
    print(f"  数据年份: {start_year} - {start_year + len(historical_data) - 1}")
    print(f"  数据点数: {len(historical_data)}")
    
    # Step 2: 获取维度参数
    print("\n【Step 2】获取职业维度参数...")
    dimensions = get_career_dimensions(career_name)
    print(f"  D1 (自动化潜力) = {dimensions['D1']:.2f}")
    print(f"  D2 (技能演进) = {dimensions['D2']:.2f}")
    print(f"  D3 (市场需求弹性) = {dimensions['D3']:.2f}")
    print(f"  D4 (人本约束) = {dimensions['D4']:.2f}")
    
    # Step 3: 创建模型
    print("\n【Step 3】创建预测模型...")
    model = AICareerEvolutionModel(
        career_name=career_name,
        historical_data=historical_data,
        dimensions=dimensions,
        start_year=start_year
    )
    print(f"  ✅ 模型创建成功")
    print(f"  基准趋势: {model.grey_model.get_trend()}")
    growth_rate = model.grey_model.get_growth_rate()
    print(f"  年增长率: {growth_rate*100:.2f}% (r = {growth_rate:.4f})")
    print(f"  发展系数: a = {model.grey_model.a:.4f}")
    
    # Step 4: 执行预测
    print(f"\n【Step 4】执行预测 (至 {end_year} 年)...")
    pred_df = model.predict(end_year=end_year)
    print(f"  ✅ 预测完成，共 {len(pred_df)} 个时间点")
    
    # 打印关键预测值
    print("\n【关键预测值】")
    key_years = [2025, 2030, 2035]
    for year in key_years:
        if year in pred_df['year'].values:
            row = pred_df[pred_df['year'] == year].iloc[0]
            print(f"  {year}年:")
            print(f"    基准预测: {row['baseline_Yt']:.2f} 万人")
            print(f"    AI渗透率: {row['penetration_Pt']*100:.1f}%")
            print(f"    最终需求: {row['final_demand_Ft']:.2f} 万人")
            change = (row['final_demand_Ft'] - row['baseline_Yt']) / row['baseline_Yt'] * 100
            print(f"    相对变化: {change:+.1f}%")
    
    # Step 5: 生成可视化
    print("\n【Step 5】生成可视化图表...")
    viz = CareerVisualization(model, save_dir=save_dir)
    
    print("\n  📊 生成完整演化预测图...")
    viz.plot_complete_evolution(end_year=end_year)
    
    print("\n  📊 生成模型组件分解图...")
    viz.plot_model_components(end_year=end_year)
    
    print("\n  📊 生成维度敏感性分析图...")
    viz.plot_dimension_sensitivity()
    
    print("\n  📊 生成阶段分析图...")
    viz.plot_phase_analysis(end_year=end_year)
    
    print("\n  📊 生成维度雷达图...")
    viz.plot_dimension_radar()
    
    # Step 6: 保存预测结果
    print("\n【Step 6】保存预测结果...")
    output_file = os.path.join(save_dir, 'ai_career_predictions.csv')
    pred_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  📁 预测结果已保存: {output_file}")
    
    print("\n" + "="*70)
    print("✅ 工作流执行完成!")
    print("="*70 + "\n")
    
    return model, viz, pred_df


def run_multi_career_workflow(career_names=None, data_file='就业人数.csv', 
                              end_year=2035, save_dir='./figures'):
    """
    批量处理多个职业的预测工作流
    
    :param career_names: 职业名称列表，如果为None则处理所有职业
    :param data_file: 历史数据文件路径
    :param end_year: 预测结束年份
    :param save_dir: 图表保存目录
    :return: dict，包含每个职业的模型、可视化和预测结果
    """
    print("\n" + "="*70)
    print("AI 职业演化预测模型 - 多职业批量处理")
    print("AI Career Evolution Prediction Model - Multi-Career Batch Processing")
    print("="*70 + "\n")
    
    # Step 1: 加载数据
    print("【Step 1】加载历史数据...")
    df = load_historical_data(data_file)
    
    # 确定要处理的职业列表
    if career_names is None:
        career_names = get_all_careers(df)
        print(f"  自动检测到 {len(career_names)} 个职业: {career_names}")
    else:
        # 验证职业是否在数据中
        available_careers = get_all_careers(df)
        career_names = [c for c in career_names if c in available_careers]
        if len(career_names) == 0:
            raise ValueError(f"没有找到有效的职业，可用职业: {available_careers}")
        print(f"  处理 {len(career_names)} 个职业: {career_names}")
    
    # Step 2: 为每个职业创建模型并预测
    results = {}
    all_predictions = []
    failed_careers = []
    
    for idx, career_name in enumerate(career_names, 1):
        try:
            print(f"\n{'='*70}")
            print(f"【职业 {idx}/{len(career_names)}】{career_name}")
            print('='*70)
            
            # 提取数据
            historical_data, start_year = extract_career_data(df, career_name)
            print(f"  ✅ 数据提取成功: {len(historical_data)} 个数据点，起始年份 {start_year}")
            
            # 获取维度参数
            dimensions = get_career_dimensions(career_name)
            print(f"  ✅ 维度参数获取成功")
            
            # 创建模型
            model = AICareerEvolutionModel(
                career_name=career_name,
                historical_data=historical_data,
                dimensions=dimensions,
                start_year=start_year
            )
            print(f"  ✅ 模型创建成功")
            
            # 执行预测
            pred_df = model.predict(end_year=end_year)
            pred_df['career'] = career_name  # 添加职业列
            print(f"  ✅ 预测完成: {len(pred_df)} 个时间点")
            
            # 创建可视化对象（使用单独目录）
            career_save_dir = os.path.join(save_dir, career_name)
            print(f"  📁 保存目录: {career_save_dir}")
            os.makedirs(career_save_dir, exist_ok=True)  # 确保目录存在
            viz = CareerVisualization(model, save_dir=career_save_dir)
            
            # 为每个职业生成所有可视化图表
            print(f"\n  📊 为 {career_name} 生成可视化图表...")
            try:
                viz.plot_complete_evolution(end_year=end_year)
                print(f"    ✅ 完整演化图已生成")
            except Exception as e:
                print(f"    ⚠️  完整演化图生成失败: {e}")
            
            try:
                viz.plot_model_components(end_year=end_year)
                print(f"    ✅ 模型组件图已生成")
            except Exception as e:
                print(f"    ⚠️  模型组件图生成失败: {e}")
            
            try:
                viz.plot_dimension_sensitivity()
                print(f"    ✅ 敏感性分析图已生成")
            except Exception as e:
                print(f"    ⚠️  敏感性分析图生成失败: {e}")
            
            try:
                viz.plot_phase_analysis(end_year=end_year)
                print(f"    ✅ 阶段分析图已生成")
            except Exception as e:
                print(f"    ⚠️  阶段分析图生成失败: {e}")
            
            try:
                viz.plot_dimension_radar()
                print(f"    ✅ 维度雷达图已生成")
            except Exception as e:
                print(f"    ⚠️  维度雷达图生成失败: {e}")
            
            # 保存该职业的预测结果
            career_output_file = os.path.join(career_save_dir, f'{career_name}_predictions.csv')
            pred_df.to_csv(career_output_file, index=False, encoding='utf-8-sig')
            print(f"  📁 {career_name} 预测结果已保存: {career_output_file}")
            
            # 保存结果
            results[career_name] = {
                'model': model,
                'viz': viz,
                'predictions': pred_df,
                'dimensions': dimensions
            }
            
            all_predictions.append(pred_df)
            
            # 打印关键信息
            growth_rate = model.grey_model.get_growth_rate()
            print(f"\n  📊 关键信息:")
            print(f"    基准趋势: {model.grey_model.get_trend()}")
            print(f"    年增长率: {growth_rate*100:.2f}%")
            print(f"    维度参数: D1={dimensions['D1']:.2f}, D2={dimensions['D2']:.2f}, "
                  f"D3={dimensions['D3']:.2f}, D4={dimensions['D4']:.2f}")
            
            # 2035年预测
            if 2035 in pred_df['year'].values:
                row_2035 = pred_df[pred_df['year'] == 2035].iloc[0]
                print(f"    2035年预测: {row_2035['final_demand_Ft']:.2f} 万人 "
                      f"(基准: {row_2035['baseline_Yt']:.2f} 万人)")
            
            print(f"\n  ✅ {career_name} 处理完成!")
            
        except Exception as e:
            print(f"\n  ❌ 处理 {career_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            failed_careers.append(career_name)
            continue
    
    # Step 3: 处理结果总结
    print(f"\n{'='*70}")
    print("【处理结果总结】")
    print('='*70)
    print(f"  成功处理: {len(results)} 个职业")
    for career_name in results.keys():
        print(f"    ✅ {career_name}")
    
    if failed_careers:
        print(f"\n  处理失败: {len(failed_careers)} 个职业")
        for career_name in failed_careers:
            print(f"    ❌ {career_name}")
    
    if len(results) == 0:
        print("\n  ⚠️  没有成功处理任何职业，无法继续!")
        return {}, pd.DataFrame()
    
    # Step 4: 合并所有预测结果
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # Step 5: 生成多职业对比图
    print(f"\n{'='*70}")
    print("【多职业对比分析】")
    print('='*70)
    try:
        plot_multi_career_comparison(results, end_year=end_year, save_dir=save_dir)
        print("  ✅ 多职业对比图生成成功")
    except Exception as e:
        print(f"  ⚠️  多职业对比图生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 6: 保存合并结果
    print("\n【保存结果】")
    output_file = os.path.join(save_dir, 'all_careers_predictions.csv')
    combined_predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  📁 所有职业预测结果已保存: {output_file}")
    
    print("\n" + "="*70)
    print("✅ 多职业批量处理完成!")
    if failed_careers:
        print(f"⚠️  注意: {len(failed_careers)} 个职业处理失败")
    print("="*70 + "\n")
    
    return results, combined_predictions


def plot_multi_career_comparison(results, end_year=2035, save_dir='./figures'):
    """
    绘制多职业对比图
    
    :param results: 包含所有职业结果的字典
    :param end_year: 预测结束年份
    :param save_dir: 图表保存目录
    """
    saver = FigureSaver(save_dir)
    colors = PlotStyleConfig.get_palette(len(results))
    
    # 图1: 最终需求对比
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Multi-Career Comparison Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    # 子图1: 最终需求演化对比
    ax1 = axes1[0, 0]
    for idx, (career_name, result) in enumerate(results.items()):
        pred_df = result['predictions']
        ax1.plot(pred_df['year'], pred_df['final_demand_Ft'], 
                '-', linewidth=2.5, label=career_name.replace('_', ' ').title(),
                color=colors[idx], marker='o', markersize=4)
    ax1.set_xlabel('Year', fontweight='bold')
    ax1.set_ylabel('Final Demand (万人)', fontweight='bold')
    ax1.set_title('Final Demand Evolution Comparison', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=2024, color='red', linestyle=':', linewidth=1.5, alpha=0.6)
    
    # 子图2: 基准预测对比
    ax2 = axes1[0, 1]
    for idx, (career_name, result) in enumerate(results.items()):
        pred_df = result['predictions']
        ax2.plot(pred_df['year'], pred_df['baseline_Yt'], 
                '--', linewidth=2, label=career_name.replace('_', ' ').title(),
                color=colors[idx], alpha=0.7)
    ax2.set_xlabel('Year', fontweight='bold')
    ax2.set_ylabel('Baseline Prediction (万人)', fontweight='bold')
    ax2.set_title('Baseline Prediction Comparison', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=2024, color='red', linestyle=':', linewidth=1.5, alpha=0.6)
    
    # 子图3: AI渗透率对比
    ax3 = axes1[1, 0]
    for idx, (career_name, result) in enumerate(results.items()):
        pred_df = result['predictions']
        ax3.plot(pred_df['year'], pred_df['penetration_Pt'] * 100, 
                '-', linewidth=2.5, label=career_name.replace('_', ' ').title(),
                color=colors[idx], marker='s', markersize=4)
    ax3.set_xlabel('Year', fontweight='bold')
    ax3.set_ylabel('AI Penetration Rate (%)', fontweight='bold')
    ax3.set_title('AI Penetration Rate Comparison', fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # 子图4: 维度参数对比（雷达图）
    ax4 = axes1[1, 1]
    dim_names = ['D1\nAutomation', 'D2\nSkill\nEvolution', 
                'D3\nMarket\nElasticity', 'D4\nHuman\nConstraints']
    angles = np.linspace(0, 2 * np.pi, len(dim_names), endpoint=False).tolist()
    angles += angles[:1]
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    for idx, (career_name, result) in enumerate(results.items()):
        dims = result['dimensions']
        values = [dims['D1'], dims['D2'], dims['D3'], dims['D4']]
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, 
                label=career_name.replace('_', ' ').title(), color=colors[idx])
        ax4.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(dim_names, fontsize=10, fontweight='bold')
    ax4.set_ylim(0, 1)
    ax4.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax4.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Dimension Profile Comparison', fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    saver.save(fig1, 'multi_career_comparison', formats=['png', 'pdf'])
    plt.show()
    
    # 图2: 关键年份对比柱状图
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('Key Year Comparison (2025, 2030, 2035)', fontsize=16, fontweight='bold', y=1.02)
    
    key_years = [2025, 2030, 2035]
    for ax_idx, year in enumerate(key_years):
        ax = axes2[ax_idx]
        careers = []
        demands = []
        baselines = []
        
        for career_name, result in results.items():
            pred_df = result['predictions']
            if year in pred_df['year'].values:
                row = pred_df[pred_df['year'] == year].iloc[0]
                careers.append(career_name.replace('_', ' ').title())
                demands.append(row['final_demand_Ft'])
                baselines.append(row['baseline_Yt'])
        
        x = np.arange(len(careers))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baselines, width, label='Baseline', 
                      color=PlotStyleConfig.COLORS['neutral'], alpha=0.7)
        bars2 = ax.bar(x + width/2, demands, width, label='Final Demand', 
                      color=PlotStyleConfig.COLORS['accent'], alpha=0.7)
        
        ax.set_xlabel('Career', fontweight='bold')
        ax.set_ylabel('Employment (万人)', fontweight='bold')
        ax.set_title(f'{year} Year Prediction', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(careers, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标注
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    saver.save(fig2, 'multi_career_key_years', formats=['png', 'pdf'])
    plt.show()
    
    return fig1, fig2


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    # ============================================================
    # ★★★ 使用方式1: 批量处理所有职业（推荐） ★★★
    # ============================================================
    results, all_predictions = run_multi_career_workflow(
        career_names=None,              # None表示处理所有职业，或指定列表如 ['software_engineer', 'chef']
        data_file='就业人数.csv',        # 数据文件路径
        end_year=2035,                  # 预测结束年份
        save_dir='./figures'            # 图表保存目录
    )
    
    # 查看所有职业的预测结果
    print("\n【所有职业预测结果预览】")
    print(all_predictions.head(15).to_string())
    
    # ============================================================
    # ★★★ 使用方式2: 单独处理某个职业（可选） ★★★
    # ============================================================
    # 取消下面的注释来单独处理某个职业：
    # 
    # model, viz, predictions = run_complete_workflow(
    #     career_name='software_engineer',  # 职业名称
    #     data_file='就业人数.csv',          # 数据文件路径
    #     end_year=2035,                    # 预测结束年份
    #     save_dir='./figures'              # 图表保存目录
    # )
    # 
    # print("\n【预测结果预览】")
    # print(predictions.head(10).to_string())
    # 
    # print("\n【2030年详细预测】")
    # if 2030 in predictions['year'].values:
    #     row_2030 = predictions[predictions['year'] == 2030].iloc[0]
    #     print(f"  基准预测: {row_2030['baseline_Yt']:.2f} 万人")
    #     print(f"  AI渗透率: {row_2030['penetration_Pt']*100:.1f}%")
    #     print(f"  新市场增量: {row_2030['new_market_Nt']*100:.1f}%")
    #     print(f"  最终需求: {row_2030['final_demand_Ft']:.2f} 万人")
        