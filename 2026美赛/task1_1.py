"""
============================================================
AI 职业演化预测模型 - 完整工作流
(AI Career Evolution Prediction Model - Complete Workflow)
============================================================
功能：预测AI对职业的影响（基准趋势、技术渗透、价值重构）
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================

模型框架：
1. 灰色预测基准模型（GM(1,1)）：预测自然增长趋势
2. 技术渗透速度模型（Logistic S-Curve）：模拟AI扩散
3. 价值重构叠加模型（Task-Based Recomposition）：计算最终劳动力需求
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 图表配置（内联版本，避免导入问题）
# ============================================================

class PlotStyleConfig:
    """图表美化配置类"""

    COLORS = {
        'primary': '#1f77b4',  # 深蓝 - 历史/基准
        'secondary': '#ff7f0e',  # 橙色 - AI影响/预测
        'accent': '#2ca02c',  # 绿色 - 成功/突出
        'danger': '#d62728',  # 红色 - 危险/起始点
        'neutral': '#7f7f7f',  # 灰色 - 中性
        'background': '#f8f9fa',  # 极浅灰背景
        'grid': '#e9ecef'  # 浅灰网格
    }

    PALETTE = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896']

    @staticmethod
    def setup_style(style='academic'):
        """设置学术风格图表"""
        plt.style.use('default')  # 使用默认风格作为基础

        # 设置英文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # 学术风格参数
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.labelsize': 12,
            'axes.titlesize': 16,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 18,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.facecolor': PlotStyleConfig.COLORS['background'],
            'figure.facecolor': 'white',
            'grid.color': PlotStyleConfig.COLORS['grid']
        })

    @staticmethod
    def get_palette(n=None):
        """获取调色板"""
        if n is None:
            return PlotStyleConfig.PALETTE
        return PlotStyleConfig.PALETTE[:n] if n <= len(PlotStyleConfig.PALETTE) else PlotStyleConfig.PALETTE


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
            plt.tight_layout()
        paths = []
        for fmt in formats:
            path = os.path.join(self.save_dir, f"{filename}.{fmt}")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            paths.append(path)
        return paths


# 设置绘图风格
PlotStyleConfig.setup_style('academic')


# ============================================================
# 第一部分：模型参数配置 (Model Parameters Configuration)
# ============================================================

class AICareerParams:
    """
    AI职业演化预测模型参数配置类

    ★★★ 需要调整的参数在这里修改 ★★★

    数据占位符：请替换为你的实际数据
    """

    # 职业参数配置字典
    CAREER_PARAMS = {
        'software_engineer': {
            'D1': 0.85,  # 高自动化潜力
            'D2': 0.8,   # 快技能演进
            'D3': 0.15,   # 高市场需求弹性
            'D4': 0.28,   # 低人本约束
            'A': 1.3,    # 高AI增强系数
            'cost_reduction': 0.41  # 成本降低幅度（示例值，表示15%）
        },
        'chef': {
            'D1': 0.10,   # 低自动化潜力（烹饪需要创意和感官）
            'D2': 0.1,   # 慢技能演进
            'D3': 0.07,   # 中等市场需求弹性
            'D4': 0.45,   # 高人本约束（物理操作）
            'A': 0.5,    # 中等AI增强系数
            'cost_reduction': 0.05  # 成本降低幅度（示例值，表示5%）
        },
        'graphic_designer': {
            'D1': 0.6,   # 中等自动化潜力
            'D2': 0.4,   # 中等技能演进
            'D3': 0.02,   # 高市场需求弹性（创意产业）
            'D4': 0.29,   # 低人本约束
            'A': 0.6,    # 高AI增强系数（设计工具）
            'cost_reduction': 0.10  # 成本降低幅度（示例值，表示10%）
        }
    }

    def __init__(self, occupation_name=None, csv_path='2026美赛\就业人数.csv'):
        # ============ 职业基本信息 ============
        self.occupation_name = occupation_name or "software_engineer"  # 职业名称（英文）

        # ============ 从CSV读取历史数据 ============
        self.csv_path = csv_path
        self._load_data_from_csv()

        # ============ 预测参数 ============
        self.forecast_years = 10  # 预测未来10年
        self.start_year = 2024    # 预测起始年

        # ============ GM(1,1) 灰色预测参数 ============
        # 通常自动计算，无需手动设置

        # ============ Logistic S-Curve 参数 ============
        self.t0 = 2024     # S曲线起始点（年）

        # ============ Task-Based Recomposition 参数 ============
        self.A = 1.5       # AI增强系数，AI使用后效率提升倍数

        # ============ 根据职业设置D参数 ============
        self._set_career_params()

        # ============ 参数范围（用于敏感性分析） ============
        self.D1_range = (max(0, self.D1 - 0.1), min(1, self.D1 + 0.1))
        self.D2_range = (max(0, self.D2 - 0.1), min(1, self.D2 + 0.1))
        self.D3_range = (max(0, self.D3 - 0.1), min(1, self.D3 + 0.1))
        self.D4_range = (max(0, self.D4 - 0.1), min(1, self.D4 + 0.1))

        # 灵敏度分析步长（可由用户在 params 上修改）
        # 例如 0.1 表示按 0.1 的步长生成情景
        self.sensitivity_step = 0.05
        # 最少生成的灵敏度点数（避免某些参数因范围太小只生成很少线）
        self.sensitivity_min_points = 5

    def get_param_values(self, param_name, step=None, num=3):
        """
        返回用于灵敏度分析的一组参数值。
        - 如果提供 step，则使用等步长的 np.arange
        - 否则返回 num 个等间隔值（含端点）
        """
        if step is None:
            step = self.sensitivity_step

        if param_name == 'D1':
            low, high = self.D1_range
        elif param_name == 'D2':
            low, high = self.D2_range
        elif param_name == 'D3':
            low, high = self.D3_range
        elif param_name == 'D4':
            low, high = self.D4_range
        else:
            # 未知参数返回当前值
            val = getattr(self, param_name, None)
            return [val] if val is not None else []

        # 如果 step 非空且在合理范围内，优先使用 arange
        try:
            if step and (high - low) / (step if step > 0 else 1) <= 50:
                vals = np.arange(low, high + step / 2, step)
                # 如果生成点少于最小要求，改为使用均分的最小点数
                if vals.size < getattr(self, 'sensitivity_min_points', num):
                    vals = np.linspace(low, high, getattr(self, 'sensitivity_min_points', num))
                # 保证至少包含端点
                if vals.size == 0:
                    vals = np.linspace(low, high, num)
                return np.round(vals, 4).tolist()
        except Exception:
            pass

        # 回退到等分 num 个点
        vals = np.linspace(low, high, max(num, getattr(self, 'sensitivity_min_points', num)))
        return np.round(vals, 4).tolist()

    def _load_data_from_csv(self):
        """从CSV文件加载数据"""
        try:
            df = pd.read_csv(self.csv_path)
            if self.occupation_name not in df['career'].values:
                raise ValueError(f"职业 '{self.occupation_name}' 在CSV中未找到")

            # 获取该职业的数据
            career_data = df[df['career'] == self.occupation_name].iloc[0]
            years = [col for col in df.columns if col != 'career']
            self.historical_years = np.array([int(year) for year in years])
            self.historical_data = np.array([career_data[year] for year in years])

        except FileNotFoundError:
            print(f"警告: CSV文件 '{self.csv_path}' 未找到，使用默认数据")
            # 默认数据
            self.historical_data = np.array([125.62, 131.00, 136.55, 146.92, 184.79, 162.22, 179.53, 189.71])
            self.historical_years = np.arange(2016, 2016 + len(self.historical_data))
        except Exception as e:
            print(f"警告: 读取CSV失败: {e}，使用默认数据")
            self.historical_data = np.array([125.62, 131.00, 136.55, 146.92, 184.79, 162.22, 179.53, 189.71])
            self.historical_years = np.arange(2016, 2016 + len(self.historical_data))

    def _set_career_params(self):
        """根据职业设置D参数"""
        if self.occupation_name in self.CAREER_PARAMS:
            params = self.CAREER_PARAMS[self.occupation_name]
            self.D1 = params['D1']
            self.D2 = params['D2']
            self.D3 = params['D3']
            self.D4 = params['D4']
            self.A = params['A']
            # 成本降低幅度（x），用于 new_market = P_t * D3 * x
            self.cost_reduction = params.get('cost_reduction', 0.0)
        else:
            # 默认参数
            self.D1 = 0.85
            self.D2 = 0.8
            self.D3 = 0.6
            self.D4 = 0.1
            self.A = 1.5
            self.cost_reduction = 0.0

    def compute_L_from_tasks(self):
        """
        根据职业对应的 tasks CSV 计算 L：
        L = sum(Importance * Exposure_Score) / sum(Importance)

        exposure score 的值等于本职业的 D1。
        任务文件按职业映射到工作表，例如：
        - software_engineer -> tasks_15-1252-00.csv
        - graphic_designer -> tasks_27-1024-00.csv
        - chef -> tasks_35-1011-00.csv
        如果无法读取文件或格式不匹配，则回退为 self.D1
        """
        # 基础目录：优先使用 csv_path 的目录，否则当前目录
        try:
            base_dir = os.path.dirname(self.csv_path) if self.csv_path else '.'
        except Exception:
            base_dir = '.'

        mapping = {
            'software_engineer': 'tasks_15-1252-00.csv',
            'graphic_designer': 'tasks_27-1024-00.csv',
            'chef': 'tasks_35-1011-00.csv'
        }

        fname = mapping.get(self.occupation_name)
        if not fname:
            return self.D1

        path = os.path.join(base_dir, fname)
        try:
            df = pd.read_csv(path)
            if 'Importance' not in df.columns:
                return self.D1
            importance = pd.to_numeric(df['Importance'], errors='coerce').fillna(0.0)
            # exposure score 等于 D1（常数）
            exposure = float(self.D1)
            numerator = (importance * exposure).sum()
            denom = importance.sum()
            if denom == 0:
                return self.D1
            return float(numerator / denom)
        except Exception:
            return self.D1

    def summary(self):
        """打印参数摘要"""
        print("\n" + "="*70)
        print("📋 AI Career Evolution Prediction Model Parameters Configuration")
        print("="*70)

        career_english = self.occupation_name
        print(f"\n【Occupation】: {career_english}")

        print("\n【Historical Data】")
        for year, value in zip(self.historical_years, self.historical_data):
            print(f"  {year}: {value:.1f} (10,000 people)")

        print("\n【Forecast Settings】")
        print(f"  Forecast Years: {self.forecast_years} years")
        print(f"  Start Year: {self.start_year}")

        print("\n【Key Dimension Parameters】")
        print(f"  D1 (Automation Potential): {self.D1}")
        print(f"  D2 (Skill Evolution Speed): {self.D2}")
        print(f"  D3 (Market Demand Elasticity): {self.D3}")
        print(f"  D4 (Human Constraint): {self.D4}")
        print(f"  A (AI Enhancement Coefficient): {self.A}")

        print("="*70 + "\n")


# ============================================================
# 第二部分：AI职业演化模型核心计算 (Core Model Calculations)
# ============================================================

class AICareerModel:
    """
    AI职业演化预测模型核心类

    实现三个子模型的计算
    """

    def __init__(self, params: AICareerParams = None):
        """
        初始化模型

        :param params: 参数配置对象
        """
        self.params = params if params else AICareerParams()

    def gm11_predict(self, data, forecast_steps):
        """
        灰色预测基准模型（GM(1,1)）- 预测自然增长趋势

        :param data: 历史数据序列
        :param forecast_steps: 预测步数
        :return: 预测序列
        """
        n = len(data)
        x0 = data

        # 一次累加生成 (AGO)
        x1 = np.cumsum(x0)

        # 构建数据矩阵
        B = np.column_stack([-0.5 * (x1[:-1] + x1[1:]), np.ones(n-1)])
        Y = x0[1:]

        # 最小二乘法求参数 a, b
        try:
            coef = np.linalg.lstsq(B, Y, rcond=None)[0]
            a, b = coef
        except:
            # 如果最小二乘失败，使用简单线性回归
            from scipy.stats import linregress
            x_vals = np.arange(n-1)
            slope, intercept, _, _, _ = linregress(x_vals, Y)
            a = -slope / (x0[0] - intercept/slope) if x0[0] != intercept/slope else 0.01
            b = slope

        # 计算增长率 r = e^{-a} - 1
        r = np.exp(-a) - 1

        # 自然趋势公式：Y_t = x^{(0)}(n) * (1+r)^{t-n}
        pred_values = []
        for k in range(1, forecast_steps + 1):
            Y_t = x0[-1] * (1 + r)**k
            pred_values.append(Y_t)

        return np.array(pred_values)

    def logistic_curve(self, t, L, k, t0):
        """
        Logistic S-Curve 函数

        :param t: 时间
        :param L: 饱和上限
        :param k: 增长率
        :param t0: 起始时间
        :return: 渗透率
        """
        return L / (1 + np.exp(-k * (t - t0)))

    def fit_logistic_params(self):
        """
        根据D1, D2拟合Logistic参数

        :return: L, k, t0
        """
        p = self.params
        # 使用任务重要性与曝光分数（exposure score = D1）计算 L
        L = p.compute_L_from_tasks()
        k = p.D2 * 0.8 + 0.1  # 增长率正比于D2，调整系数
        t0 = p.t0
        return L, k, t0

    def task_recomposition(self, Y_t, P_t):
        """
        Task-Based Recomposition 价值重构

        :param Y_t: 基准值
        :param P_t: 渗透率
        :return: 修正后的劳动力需求
        """
        p = self.params

        # 人类核心防御区
        defense = 1 - P_t * (1 - p.D4)

        # AI增强产出
        enhancement = P_t * p.A

        # 新市场增量（简化为弹性相关的增长）
        # 使用职业特定的成本降低幅度 x（在 AICareerParams 中以 cost_reduction 保存）
        # new_market = P_t * D3 * x
        new_market = P_t * p.D3 * getattr(p, 'cost_reduction', 0.0)

        # 最终需求
        F_t = Y_t * (defense + enhancement + new_market)

        return F_t, defense, enhancement, new_market

    def predict_evolution(self, verbose=True):
        """
        执行完整预测流程

        :param verbose: 是否打印详细信息
        :return: 预测结果字典
        """
        if verbose:
            print("🔍 开始AI职业演化预测...")

        p = self.params

        # 步骤1: 基准预测
        if verbose:
            print("  📈 步骤1: GM(1,1) 基准预测模型")

        baseline_predictions = self.gm11_predict(p.historical_data, p.forecast_years)

        # 构造时间序列
        future_years = np.arange(p.start_year, p.start_year + p.forecast_years)
        historical_years = p.historical_years

        # 步骤2: Logistic S-Curve 技术渗透
        if verbose:
            print("  🚀 步骤2: Logistic S-Curve 技术渗透模型")
        L, k, t0 = self.fit_logistic_params()
        penetration_rates = self.logistic_curve(future_years, L, k, t0)

        # 步骤3: Task-Based Recomposition 价值重构
        if verbose:
            print("  🔄 步骤3: Task-Based Recomposition 价值重构")

        final_demands = []
        defense_parts = []
        enhancement_parts = []
        new_market_parts = []

        for i, (Y_t, P_t) in enumerate(zip(baseline_predictions, penetration_rates)):
            F_t, defense, enhancement, new_market = self.task_recomposition(Y_t, P_t)
            final_demands.append(F_t)
            defense_parts.append(defense)
            enhancement_parts.append(enhancement)
            new_market_parts.append(new_market)

        # 组织结果
        results = {
            'years': np.concatenate([historical_years, future_years]),
            'historical_data': p.historical_data,
            'baseline_predictions': baseline_predictions,
            'penetration_rates': penetration_rates,
            'final_demands': np.array(final_demands),
            'defense_parts': np.array(defense_parts),
            'enhancement_parts': np.array(enhancement_parts),
            'new_market_parts': np.array(new_market_parts),
            'future_years': future_years,
            'historical_years': historical_years
        }

        if verbose:
            print("  ✅ 预测完成!")

        return results


# ============================================================
# 第三部分：可视化模块 (Visualization Module)
# ============================================================

class AICareerVisualization:
    """
    AI职业演化预测可视化类
    """

    def __init__(self, model: AICareerModel, results: dict, save_dir='./figures'):
        """
        初始化可视化

        :param model: AICareerModel实例
        :param results: 预测结果字典
        :param save_dir: 保存目录
        """
        self.model = model
        self.results = results
        self.saver = FigureSaver(save_dir)

    def plot_complete_evolution(self, figsize=(14, 10)):
        """
        绘制完整演化预测图
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        # 主标题加粗，添加副标题
        occupation_english = self.model.params.occupation_name
        fig.suptitle(f'{occupation_english} - AI Career Evolution Prediction',
                    fontsize=18, fontweight='bold')
        fig.text(0.5, 0.95, 'Comprehensive Analysis of Baseline Trends, Technology Penetration, and Value Recomposition',
                ha='center', fontsize=12, style='italic')

        r = self.results
        colors = PlotStyleConfig.get_palette()

        # 子图1: 历史数据 + 基准预测
        ax1 = axes[0, 0]
        ax1.plot(r['historical_years'], r['historical_data'],
                'o-', color=colors[0], label='Historical Data', linewidth=2.5, markersize=6)
        ax1.plot(r['future_years'], r['baseline_predictions'],
                '--', color=colors[1], label='GM(1,1) Baseline Prediction', linewidth=1.5)
        ax1.set_title('Baseline Trend Prediction', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Employment (10,000 people)')
        ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax1.grid(True, alpha=0.3)

        # 子图2: 技术渗透率
        ax2 = axes[0, 1]
        ax2.plot(r['future_years'], r['penetration_rates'] * 100,
                's-', color=colors[2], label='AI Penetration Rate', linewidth=2.5, markersize=6)
        ax2.set_title('Technology Penetration Speed', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Penetration Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax2.grid(True, alpha=0.3)

        # 子图3: 价值重构分解
        ax3 = axes[1, 0]
        ax3.plot(r['future_years'], r['defense_parts'] * 100,
                '^-', color=colors[3], label='Human Core Defense Zone', linewidth=2, markersize=6)
        ax3.plot(r['future_years'], r['enhancement_parts'] * 100,
                'D-', color=colors[4], label='AI Enhancement Output', linewidth=2, markersize=6)
        ax3.plot(r['future_years'], r['new_market_parts'] * 100,
                'v-', color=colors[5], label='New Market Increment', linewidth=2, markersize=6)
        ax3.set_title('Value Recomposition Breakdown', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Contribution Ratio (%)')
        ax3.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax3.grid(True, alpha=0.3)

        # 子图4: 最终预测结果
        ax4 = axes[1, 1]
        ax4.plot(r['historical_years'], r['historical_data'],
                'o-', color=colors[0], label='Historical Data', linewidth=2.5, markersize=6)
        ax4.plot(r['future_years'], r['final_demands'],
                '*-', color=colors[6], label='Final Predicted Demand', linewidth=3, markersize=8)
        # AI影响起始点：加粗虚线，添加箭头
        ax4.axvline(x=self.model.params.start_year, color=PlotStyleConfig.COLORS['danger'], linestyle='--',
                   linewidth=2.5, alpha=0.8, label='AI Impact Start Point')
        ax4.annotate('AI Impact\nStarts', xy=(self.model.params.start_year, ax4.get_ylim()[1]*0.9),
                    xytext=(self.model.params.start_year+0.5, ax4.get_ylim()[1]*0.85),
                    arrowprops=dict(arrowstyle='->', color=PlotStyleConfig.COLORS['danger'], lw=1.5),
                    fontsize=10, ha='left', va='center')
        # 高亮预测区域
        ax4.axvspan(self.model.params.start_year, r['future_years'][-1], alpha=0.1, color=colors[2])
        ax4.set_title('Final Labor Demand Prediction', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Employment (10,000 people)')
        ax4.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax4.grid(True, alpha=0.3)

        # 移除x轴标签重复，只在底部显示
        for ax in axes.flat:
            if ax != axes[1, 0] and ax != axes[1, 1]:
                ax.set_xlabel('')
        for ax in axes.flat:
            if ax != axes[0, 0] and ax != axes[1, 0]:
                ax.set_ylabel('')

        plt.tight_layout(rect=[0, 0, 1, 0.93])  # 留空间给副标题

        # 保存图片
        career_filename = f"{occupation_english.replace(' ', '_').lower()}_evolution_complete"
        paths = self.saver.save(fig, career_filename)
        print(f"  💾 Complete evolution plot saved: {paths[0]}")

        return fig

    def plot_comparison_scenarios(self, figsize=(12, 8)):
        """
        绘制不同情景对比
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        occupation_english = self.model.params.occupation_name
        fig.suptitle(f'{occupation_english} - Parameter Sensitivity Analysis',
                    fontsize=16, fontweight='bold')
        fig.text(0.5, 0.95, 'Impact of Key Dimensions on Employment Predictions',
                ha='center', fontsize=12, style='italic')

        r = self.results
        colors = PlotStyleConfig.get_palette()

        # 情景1: D1变化 (自动化潜力) — 使用 params 中的范围和步长
        ax1 = axes[0, 0]
        d1_values = self.model.params.get_param_values('D1')
        for i, d1 in enumerate(d1_values):
            temp_params = AICareerParams(occupation_name=self.model.params.occupation_name,
                        csv_path=self.model.params.csv_path)
            temp_params.D1 = d1
            temp_model = AICareerModel(temp_params)
            temp_results = temp_model.predict_evolution(verbose=False)
            ax1.plot(temp_results['future_years'], temp_results['final_demands'],
                label=f'Automation Potential (D1)={d1}', linewidth=2, color=colors[(i+2) % len(colors)])
        ax1.plot(r['future_years'], r['final_demands'],
                '--', label='Baseline', linewidth=3, color='black')
        ax1.set_title('Automation Potential Sensitivity (D1)', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Employment (10,000 people)')
        ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax1.grid(True, alpha=0.3)

        # 情景2: D2变化 (技能演进速度)
        ax2 = axes[0, 1]
        d2_values = self.model.params.get_param_values('D2')
        for i, d2 in enumerate(d2_values):
            temp_params = AICareerParams(occupation_name=self.model.params.occupation_name,
                        csv_path=self.model.params.csv_path)
            temp_params.D2 = d2
            temp_model = AICareerModel(temp_params)
            temp_results = temp_model.predict_evolution(verbose=False)
            ax2.plot(temp_results['future_years'], temp_results['final_demands'],
                label=f'Skill Evolution Speed (D2)={d2}', linewidth=2, color=colors[(i+5) % len(colors)])
        ax2.plot(r['future_years'], r['final_demands'],
                '--', label='Baseline', linewidth=3, color='black')
        ax2.set_title('Skill Evolution Speed Sensitivity (D2)', fontweight='bold', fontsize=14)
        ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax2.grid(True, alpha=0.3)

        # 情景3: D3变化 (市场需求弹性)
        ax3 = axes[1, 0]
        d3_values = self.model.params.get_param_values('D3')
        for i, d3 in enumerate(d3_values):
            temp_params = AICareerParams(occupation_name=self.model.params.occupation_name,
                        csv_path=self.model.params.csv_path)
            temp_params.D3 = d3
            temp_model = AICareerModel(temp_params)
            temp_results = temp_model.predict_evolution(verbose=False)
            ax3.plot(temp_results['future_years'], temp_results['final_demands'],
                label=f'Market Demand Elasticity (D3)={d3}', linewidth=2, color=colors[i % len(colors)])
        ax3.plot(r['future_years'], r['final_demands'],
                '--', label='Baseline', linewidth=3, color='black')
        ax3.set_title('Market Demand Elasticity Sensitivity (D3)', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Employment (10,000 people)')
        ax3.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax3.grid(True, alpha=0.3)

        # 情景4: D4变化 (人本约束)
        ax4 = axes[1, 1]
        d4_values = self.model.params.get_param_values('D4')
        for i, d4 in enumerate(d4_values):
            temp_params = AICareerParams(occupation_name=self.model.params.occupation_name,
                        csv_path=self.model.params.csv_path)
            temp_params.D4 = d4
            temp_model = AICareerModel(temp_params)
            temp_results = temp_model.predict_evolution(verbose=False)
            ax4.plot(temp_results['future_years'], temp_results['final_demands'],
                label=f'Human Constraint (D4)={d4}', linewidth=2, color=colors[(i+3) % len(colors)])
        ax4.plot(r['future_years'], r['final_demands'],
                '--', label='Baseline', linewidth=3, color='black')
        ax4.set_title('Human Constraint Sensitivity (D4)', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Year')
        ax4.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax4.grid(True, alpha=0.3)

        # 移除重复标签
        for ax in axes.flat:
            if ax != axes[1, 0] and ax != axes[1, 1]:
                ax.set_xlabel('')
        for ax in axes.flat:
            if ax != axes[0, 0] and ax != axes[1, 0]:
                ax.set_ylabel('')

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        # 保存图片
        career_filename = f"{occupation_english.replace(' ', '_').lower()}_sensitivity_analysis"
        paths = self.saver.save(fig, career_filename)
        print(f"  💾 Parameter sensitivity analysis plot saved: {paths[0]}")

        return fig

    def plot_model_components(self, figsize=(14, 8)):
        """
        绘制模型组件分解图
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        occupation_english = self.model.params.occupation_name
        fig.suptitle(f'{occupation_english} - Model Components Breakdown',
                    fontsize=16, fontweight='bold')
        fig.text(0.5, 0.95, 'Detailed Analysis of Each Model Component',
                ha='center', fontsize=12, style='italic')

        r = self.results
        colors = PlotStyleConfig.get_palette()

        # 子图1: 基准预测 vs 历史数据
        ax1 = axes[0, 0]
        ax1.plot(r['historical_years'], r['historical_data'],
                'o-', color=colors[0], label='Historical Data', linewidth=2.5, markersize=6)
        ax1.plot(r['future_years'], r['baseline_predictions'],
                '--', color=colors[1], label='GM(1,1) Baseline', linewidth=2)
        ax1.set_title('Baseline Prediction (GM Model)', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Employment (10,000 people)')
        ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax1.grid(True, alpha=0.3)

        # 子图2: AI渗透率时间序列
        ax2 = axes[0, 1]
        ax2.plot(r['future_years'], r['penetration_rates'] * 100,
                's-', color=colors[2], label='AI Penetration Rate', linewidth=2.5, markersize=6)
        ax2.fill_between(r['future_years'], r['penetration_rates'] * 100, alpha=0.3, color=colors[2])
        ax2.set_title('Technology Penetration (Logistic Model)', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Penetration Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax2.grid(True, alpha=0.3)

        # 子图3: 价值重构分解
        ax3 = axes[1, 0]
        ax3.plot(r['future_years'], r['defense_parts'] * 100,
                '^-', color=colors[3], label='Human Core Defense', linewidth=2, markersize=6)
        ax3.plot(r['future_years'], r['enhancement_parts'] * 100,
                'D-', color=colors[4], label='AI Enhancement', linewidth=2, markersize=6)
        ax3.plot(r['future_years'], r['new_market_parts'] * 100,
                'v-', color=colors[5], label='New Market Increment', linewidth=2, markersize=6)
        ax3.set_title('Value Recomposition (Task-Based Model)', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Contribution (%)')
        ax3.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax3.grid(True, alpha=0.3)

        # 子图4: 最终需求 vs 基准预测
        ax4 = axes[1, 1]
        ax4.plot(r['future_years'], r['baseline_predictions'],
                '--', color="#0033FF", label='Baseline Prediction', linewidth=2)
        ax4.plot(r['future_years'], r['final_demands'],
                '*-', color=colors[6], label='Final Demand', linewidth=3, markersize=8)
        ax4.fill_between(r['future_years'], r['baseline_predictions'], r['final_demands'],
                        where=(r['final_demands'] > r['baseline_predictions']),
                        alpha=0.3, color='green', label='AI Impact (+)')
        ax4.fill_between(r['future_years'], r['baseline_predictions'], r['final_demands'],
                        where=(r['final_demands'] < r['baseline_predictions']),
                        alpha=0.3, color='red', label='AI Impact (-)')
        ax4.set_title('Final vs Baseline Comparison', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Employment (10,000 people)')
        ax4.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax4.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        # 保存图片
        career_filename = f"{occupation_english.replace(' ', '_').lower()}_model_components"
        paths = self.saver.save(fig, career_filename)
        print(f"  💾 Model components breakdown plot saved: {paths[0]}")

        return fig

    def plot_dimension_sensitivity(self, figsize=(14, 10)):
        """
        绘制维度敏感性分析图
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        occupation_english = self.model.params.occupation_name
        fig.suptitle(f'{occupation_english} - Dimension Sensitivity Analysis',
                    fontsize=16, fontweight='bold')
        fig.text(0.5, 0.95, 'Impact of Each Dimension Parameter on Final Predictions',
                ha='center', fontsize=12, style='italic')

        r = self.results
        colors = PlotStyleConfig.get_palette()

        # 情景分析函数
        def run_sensitivity_analysis(param_name, param_values=None):
            """
            基于给定参数名运行灵敏度分析。若 param_values 未提供，
            使用模型参数中的范围和步长生成值。
            返回 shape=(len(param_values), len(future_years)) 的数组。
            """
            results = []
            if param_values is None:
                param_values = self.model.params.get_param_values(param_name)
            for val in param_values:
                temp_params = AICareerParams(occupation_name=self.model.params.occupation_name,
                                            csv_path=self.model.params.csv_path)
                setattr(temp_params, param_name, val)
                temp_model = AICareerModel(temp_params)
                temp_results = temp_model.predict_evolution(verbose=False)
                results.append(temp_results['final_demands'])
            return np.array(results), param_values

        # 子图1: D1敏感性
        ax1 = axes[0, 0]
        d1_results, d1_values = run_sensitivity_analysis('D1')
        for i, (val, demands) in enumerate(zip(d1_values, d1_results)):
            ax1.plot(r['future_years'], demands, label=f'D1={val}', linewidth=2, color=colors[i % len(colors)])
        ax1.plot(r['future_years'], r['final_demands'], 'k--', label='Baseline', linewidth=3)
        ax1.set_title('Automation Potential (D1) Sensitivity', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Employment (10,000 people)')
        ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax1.grid(True, alpha=0.3)

        # 子图2: D2敏感性
        ax2 = axes[0, 1]
        d2_results, d2_values = run_sensitivity_analysis('D2')
        for i, (val, demands) in enumerate(zip(d2_values, d2_results)):
            ax2.plot(r['future_years'], demands, label=f'D2={val}', linewidth=2, color=colors[(i+2) % len(colors)])
        ax2.plot(r['future_years'], r['final_demands'], 'k--', label='Baseline', linewidth=3)
        ax2.set_title('Skill Evolution (D2) Sensitivity', fontweight='bold', fontsize=14)
        ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax2.grid(True, alpha=0.3)

        # 子图3: D3敏感性
        ax3 = axes[1, 0]
        d3_results, d3_values = run_sensitivity_analysis('D3')
        for i, (val, demands) in enumerate(zip(d3_values, d3_results)):
            ax3.plot(r['future_years'], demands, label=f'D3={val}', linewidth=2, color=colors[(i+4) % len(colors)])
        ax3.plot(r['future_years'], r['final_demands'], 'k--', label='Baseline', linewidth=3)
        ax3.set_title('Market Elasticity (D3) Sensitivity', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Employment (10,000 people)')
        ax3.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax3.grid(True, alpha=0.3)

        # 子图4: D4敏感性
        ax4 = axes[1, 1]
        d4_results, d4_values = run_sensitivity_analysis('D4')
        for i, (val, demands) in enumerate(zip(d4_values, d4_results)):
            ax4.plot(r['future_years'], demands, label=f'D4={val}', linewidth=2, color=colors[(i+6) % len(colors)])
        ax4.plot(r['future_years'], r['final_demands'], 'k--', label='Baseline', linewidth=3)
        ax4.set_title('Human Constraints (D4) Sensitivity', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Year')
        ax4.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax4.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        # 保存图片
        career_filename = f"{occupation_english.replace(' ', '_').lower()}_dimension_sensitivity"
        paths = self.saver.save(fig, career_filename)
        print(f"  💾 Dimension sensitivity analysis plot saved: {paths[0]}")

        return fig

    def plot_phase_analysis(self, figsize=(14, 6)):
        """
        绘制阶段分析图
        """
        # 简洁阶段分析：左图展示AI渗透率与就业增长率，右图展示最终需求与基准预测对比
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)
        occupation_english = self.model.params.occupation_name
        fig.suptitle(f'{occupation_english} - AI Impact Phase Analysis', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.95, 'Evolution of AI Impact Over Time', ha='center', fontsize=12, style='italic')

        r = self.results
        colors = PlotStyleConfig.get_palette()

        # 左图: 渗透率与增长率
        ax_left.plot(r['future_years'], r['penetration_rates'] * 100, 's-', color=colors[2], label='AI Penetration (%)', linewidth=2.5)
        ax_left.set_ylabel('AI Penetration (%)', color=colors[2])
        ax_left.set_xlabel('Year')
        ax_left_twin = ax_left.twinx()
        growth = np.gradient(r['final_demands']) / (r['final_demands'] + 1e-9) * 100
        ax_left_twin.plot(r['future_years'], growth, 'D--', color=colors[6], label='Employment Growth Rate (%)', linewidth=2)
        ax_left_twin.set_ylabel('Employment Growth Rate (%)', color=colors[6])
        ax_left.set_title('AI Penetration vs Employment Growth', fontweight='bold')
        ax_left.grid(True, alpha=0.3)

        # 右图: 基准预测 vs 最终需求
        ax_right.plot(r['future_years'], r['baseline_predictions'], '--', color=colors[1], label='Baseline Prediction', linewidth=2)
        ax_right.plot(r['future_years'], r['final_demands'], '*-', color=colors[6], label='Final Demand (with AI)', linewidth=2)
        ax_right.set_title('Baseline vs Final Demand', fontweight='bold')
        ax_right.set_xlabel('Year')
        ax_right.set_ylabel('Employment (10,000 people)')
        ax_right.legend()
        ax_right.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        career_filename = f"{occupation_english.replace(' ', '_').lower()}_phase_analysis"
        paths = self.saver.save(fig, career_filename)
        print(f"  💾 Phase analysis plot saved: {paths[0]}")

        return fig

    def plot_dimension_radar(self, figsize=(10, 10)):
        """替代实现：等距彩色环围绕雷达主体，雷达主体缩小"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)

        occupation_english = self.model.params.occupation_name
        fig.suptitle(f'{occupation_english} - Dimension Profile Radar', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.95, 'Four Key Dimensions Analysis', ha='center', fontsize=12, style='italic')

        categories = ['Automation\nPotential (D1)', 'Skill\nEvolution (D2)',
                      'Market\nElasticity (D3)', 'Human\nConstraints (D4)']
        raw_values = [self.model.params.D1, self.model.params.D2, self.model.params.D3, self.model.params.D4]

        # 极坐标角度设置
        n = len(categories)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

        # 为了闭合雷达线
        plot_angles = angles + angles[:1]

        # 颜色设置
        dimension_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        # 缩放雷达主体 (假设原始数据是 0-1)
        max_radar_radius = 0.6
        scaled_values = [v * max_radar_radius for v in raw_values]
        plot_values = scaled_values + scaled_values[:1]

        # --- 核心修改：绘制外环 ---
        ring_bottom = max_radar_radius * 1.1  # 环的内径
        ring_height = 0.2                     # 环的厚度

        # 使用 bar 绘制色块环
        bars = ax.bar(angles, [ring_height] * n, width=2 * np.pi / n, bottom=ring_bottom,
                      color=dimension_colors, alpha=0.8, edgecolor='none', zorder=1)

        # --- 绘制雷达主体 ---
        primary_color = '#2C3E50'
        ax.plot(plot_angles, plot_values, 'o-', linewidth=3, color=primary_color, markersize=7, zorder=4)
        ax.fill(plot_angles, plot_values, alpha=0.22, color=primary_color, zorder=3)

        # --- 标签与刻度 ---
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, fontsize=12, fontweight='bold')

        # 设置显示范围，留出外环空间
        ax.set_ylim(0, ring_bottom + ring_height + 0.1)

        # 设置刻度（仅显示在雷达主体内）
        yticks = [0.2 * max_radar_radius, 0.4 * max_radar_radius, 0.6 * max_radar_radius]
        ax.set_yticks(yticks)
        ax.set_yticklabels(['0.2', '0.4', '0.6'], fontsize=10, color='#2C3E50', fontweight='bold')

        # 修饰网格
        ax.grid(True, color='#2C3E50', alpha=0.3, linewidth=1.2)

        # --- 数值标签 ---
        for i, (angle, val) in enumerate(zip(angles, scaled_values)):
            ax.text(angle, val + 0.04, f'{raw_values[i]:.2f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor=dimension_colors[i], alpha=0.95), zorder=6)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        # 保存逻辑
        career_filename = f"{occupation_english.replace(' ', '_').lower()}_dimension_radar"
        paths = self.saver.save(fig, career_filename)
        return fig

def plot_career_comparison(all_results, save_dir='./figures'):
    """
    绘制多职业对比图

    :param all_results: 包含所有职业结果的字典
    :param save_dir: 保存目录
    """
    if not all_results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Career AI Impact Comparative Analysis', fontsize=18, fontweight='bold')
    fig.text(0.5, 0.95, 'Historical Trends, Future Predictions, and Technology Penetration Across Professions',
            ha='center', fontsize=12, style='italic')

    careers = list(all_results.keys())
    colors = PlotStyleConfig.get_palette(len(careers))

    # 子图1: 历史数据对比
    ax1 = axes[0, 0]
    for i, career in enumerate(careers):
        r = all_results[career]
        career_english = career
        ax1.plot(r['historical_years'], r['historical_data'],
                'o-', color=colors[i], label=career_english, linewidth=2.5, markersize=6)
    ax1.set_title('Historical Employment Data Comparison', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Employment (10,000 people)')
    ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    ax1.grid(True, alpha=0.3)

    # 子图2: 最终预测对比
    ax2 = axes[0, 1]
    for i, career in enumerate(careers):
        r = all_results[career]
        career_english = career
        ax2.plot(r['years'], np.concatenate([r['historical_data'], r['final_demands']]),
                '*-', color=colors[i], label=career_english, linewidth=3, markersize=8)
    ax2.axvline(x=2024, color=PlotStyleConfig.COLORS['danger'], linestyle='--', linewidth=2.5, alpha=0.8, label='AI Impact Start')
    ax2.annotate('AI Impact\nStarts', xy=(2024, ax2.get_ylim()[1]*0.9),
                xytext=(2024.5, ax2.get_ylim()[1]*0.85),
                arrowprops=dict(arrowstyle='->', color=PlotStyleConfig.COLORS['danger'], lw=1.5),
                fontsize=10, ha='left', va='center')
    # 高亮预测区域
    ax2.axvspan(2024, max([r['future_years'][-1] for r in all_results.values()]), alpha=0.1, color=colors[2])
    ax2.set_title('Final Labor Demand Prediction Comparison', fontweight='bold', fontsize=14)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    ax2.grid(True, alpha=0.3)

    # 子图3: AI渗透率对比
    ax3 = axes[1, 0]
    for i, career in enumerate(careers):
        r = all_results[career]
        career_english = career
        ax3.plot(r['future_years'], r['penetration_rates'] * 100,
                's-', color=colors[i], label=career_english, linewidth=2.5, markersize=6)
    ax3.set_title('AI Technology Penetration Rate Comparison', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Penetration Rate (%)')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    ax3.grid(True, alpha=0.3)

    # 子图4: 就业增长率对比 (2030年相对2023年)
    ax4 = axes[1, 1]
    growth_rates = []
    career_names = []
    for career in careers:
        r = all_results[career]
        hist_2023 = r['historical_data'][-1]  # 2023年数据
        pred_2030 = r['final_demands'][6]     # 2030年预测 (2024+6=2030)
        growth = (pred_2030 - hist_2023) / hist_2023 * 100
        growth_rates.append(growth)
        career_english = career
        career_names.append(career_english)

    bars = ax4.bar(career_names, growth_rates, color=colors[:len(career_names)])
    ax4.set_title('Predicted Employment Growth Rate Comparison (2030 vs 2023)', fontweight='bold', fontsize=14)
    ax4.set_ylabel('Growth Rate (%)')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, rate in zip(bars, growth_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{rate:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')

    # 移除重复标签
    for ax in axes.flat:
        if ax != axes[1, 0] and ax != axes[1, 1]:
            ax.set_xlabel('')
    for ax in axes.flat:
        if ax != axes[0, 0] and ax != axes[1, 0]:
            ax.set_ylabel('')

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # 保存图片
    saver = FigureSaver(save_dir)
    paths = saver.save(fig, 'career_comparison_analysis')
    print(f"  💾 Career comparison analysis plot saved: {paths[0]}")


# ============================================================
# 第四部分：主工作流 (Main Workflow)
# ============================================================

def run_multi_career_workflow(csv_path='./就业人数.csv'):
    """
    运行多个职业的AI职业演化预测工作流

    包括：从CSV读取数据 → 为每个职业配置参数 → 模型预测 → 可视化 → 结果保存
    """
    print("\n" + "█"*70)
    print("█" + " "*18 + "多职业AI演化预测模型" + " "*19 + "█")
    print("█" + " "*13 + "Multi-Career AI Evolution Prediction" + " "*14 + "█")
    print("█"*70 + "\n")

    # ========== Step 1: 从CSV读取职业列表 ==========
    print("【Step 1】从CSV读取职业数据...")
    try:
        df = pd.read_csv(csv_path)
        careers = df['career'].tolist()
        print(f"  📁 发现 {len(careers)} 个职业: {', '.join(careers)}")
    except Exception as e:
        print(f"  ❌ 读取CSV失败: {e}")
        return None

    # 创建figures目录
    figures_dir = './2026美赛/figures'
    os.makedirs(figures_dir, exist_ok=True)

    # 存储所有结果
    all_results = {}
    all_params = {}

    # ========== Step 2-5: 循环处理每个职业 ==========
    for i, career in enumerate(careers, 1):
        print(f"\n{'='*50}")
        print(f"【处理职业 {i}/{len(careers)}】: {career}")
        print('='*50)

        # Step 2: 参数配置
        print("  【Step 2】初始化模型参数...")
        params = AICareerParams(occupation_name=career, csv_path=csv_path)
        params.summary()

        # Step 3: 创建模型
        print("  【Step 3】创建预测模型...")
        model = AICareerModel(params)

        # Step 4: 执行预测
        print("  【Step 4】执行AI影响预测...")
        results = model.predict_evolution(verbose=True)

        # Step 5: 生成可视化
        print("\n  【Step 5】生成可视化图表...")
        viz = AICareerVisualization(model, results, save_dir=figures_dir)

        # 图1: 完整演化预测
        print("    🎨 绘制完整演化预测图...")
        viz.plot_complete_evolution()

        # 图2: 参数敏感性分析
        print("    🎨 绘制参数敏感性分析图...")
        viz.plot_comparison_scenarios()

        # 图3: 模型组件分解
        print("    🎨 绘制模型组件分解图...")
        viz.plot_model_components()

        # 图4: 维度敏感性分析
        print("    🎨 绘制维度敏感性分析图...")
        viz.plot_dimension_sensitivity()

        # 图5: 阶段分析
        print("    🎨 绘制阶段分析图...")
        viz.plot_phase_analysis()

        # 图6: 维度雷达图
        print("    🎨 绘制维度雷达图...")
        viz.plot_dimension_radar()

        # 保存结果
        all_results[career] = results
        all_params[career] = params

        # 保存CSV结果
        result_df = pd.DataFrame({
            '年份': results['years'],
            '历史数据': np.concatenate([results['historical_data'],
                                       np.full(len(results['future_years']), np.nan)]),
            '基准预测': np.concatenate([np.full(len(results['historical_years']), np.nan),
                                       results['baseline_predictions']]),
            'AI渗透率': np.concatenate([np.full(len(results['historical_years']), np.nan),
                                       results['penetration_rates']]),
            '最终需求预测': np.concatenate([np.full(len(results['historical_years']), np.nan),
                                           results['final_demands']])
        })

        csv_filename = f'{figures_dir}/{career}_predictions.csv'
        result_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"    📁 {career}预测结果已保存: {csv_filename}")

    # ========== Step 6: 生成综合对比图 ==========
    print(f"\n{'='*50}")
    print("【Step 6】生成职业对比分析...")
    print('='*50)

    plot_career_comparison(all_results, save_dir=figures_dir)

    print("\n" + "█"*70)
    print("█" + " "*23 + "多职业工作流执行完成!" + " "*24 + "█")
    print("█"*70 + "\n")

    return all_params, all_results


def run_ai_career_workflow():
    """
    运行单个职业的AI职业演化预测工作流（向后兼容）

    包括：参数配置 → 模型预测 → 可视化 → 结果保存
    """
    print("\n" + "█"*70)
    print("█" + " "*20 + "AI职业演化预测模型" + " "*21 + "█")
    print("█" + " "*15 + "AI Career Evolution Prediction" + " "*16 + "█")
    print("█"*70 + "\n")

    # ========== Step 1: 参数配置 ==========
    print("【Step 1】初始化模型参数...")
    params = AICareerParams()

    # ★★★ 在这里修改你的参数和数据 ★★★
    # params.occupation_name = "你的职业名称"
    # params.historical_data = np.array([你的历史数据])
    # params.historical_years = np.arange(起始年, 起始年 + len(params.historical_data))
    # params.D1 = 你的D1值
    # params.D2 = 你的D2值
    # params.D3 = 你的D3值
    # params.D4 = 你的D4值

    params.summary()

    # ========== Step 2: 创建模型 ==========
    print("【Step 2】创建预测模型...")
    model = AICareerModel(params)

    # ========== Step 3: 执行预测 ==========
    print("【Step 3】执行AI影响预测...")
    results = model.predict_evolution(verbose=True)

    # ========== Step 4: 生成可视化 ==========
    print("\n【Step 4】生成可视化图表...")
    print("-"*70)

    # 创建figures目录
    os.makedirs('./figures', exist_ok=True)

    viz = AICareerVisualization(model, results, save_dir='./figures')

    # 图1: 完整演化预测
    print("\n  🎨 绘制完整演化预测图...")
    viz.plot_complete_evolution()

    # 图2: 参数敏感性分析
    print("\n  🎨 绘制参数敏感性分析图...")
    viz.plot_comparison_scenarios()

    # ========== Step 5: 保存结果 ==========
    print("\n【Step 5】保存预测结果...")
    print("-"*70)

    # 保存为CSV
    result_df = pd.DataFrame({
        '年份': results['years'],
        '历史数据': np.concatenate([results['historical_data'],
                                   np.full(len(results['future_years']), np.nan)]),
        '基准预测': np.concatenate([np.full(len(results['historical_years']), np.nan),
                                   results['baseline_predictions']]),
        'AI渗透率': np.concatenate([np.full(len(results['historical_years']), np.nan),
                                   results['penetration_rates']]),
        '最终需求预测': np.concatenate([np.full(len(results['historical_years']), np.nan),
                                       results['final_demands']])
    })

    result_df.to_csv('./figures/ai_career_predictions.csv', index=False, encoding='utf-8-sig')
    print("  📁 预测结果已保存: ./figures/ai_career_predictions.csv")

    print("\n" + "█"*70)
    print("█" + " "*25 + "工作流执行完成!" + " "*26 + "█")
    print("█"*70 + "\n")

    return params, model, results, viz


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # ★★★ 使用示例：运行多职业完整工作流 ★★★
    # ============================================================
    all_params, all_results = run_multi_career_workflow(csv_path=r'd:\competition\美国大学生数学建模大赛\2026美赛\就业人数.csv')

    # ============================================================
    # ★★★ 自定义分析示例 ★★★
    # ============================================================

    # 1. 查看特定职业的预测结果
    # if all_results:
    #     career = 'software_engineer'
    #     if career in all_results:
    #         results = all_results[career]
    #         future_idx = 5  # 第6年（2030年）
    #         print(f"\n{career} 2030年预测:")
    #         print(f"  基准预测: {results['baseline_predictions'][future_idx]:.1f} 万人")
    #         print(f"  AI渗透率: {results['penetration_rates'][future_idx]*100:.1f}%")
    #         print(f"  最终需求: {results['final_demands'][future_idx]:.1f} 万人")

    # 2. 比较不同职业的增长率
    # if all_results:
    #     print("\n职业增长率对比 (2030年相对2023年):")
    #     for career, results in all_results.items():
    #         hist_2023 = results['historical_data'][-1]
    #         pred_2030 = results['final_demands'][6]
    #         growth = (pred_2030 - hist_2023) / hist_2023 * 100
    #         print(f"  {career}: {growth:.1f}%")

    # 3. 导出所有职业的综合数据
    # if all_results:
    #     summary_df = pd.DataFrame()
    #     for career, results in all_results.items():
    #         temp_df = pd.DataFrame({
    #             '职业': career,
    #             '年份': results['years'],
    #             '就业人数': np.concatenate([results['historical_data'], results['final_demands']]),
    #             '数据类型': ['历史'] * len(results['historical_years']) + ['预测'] * len(results['future_years'])
    #         })
    #         summary_df = pd.concat([summary_df, temp_df], ignore_index=True)
    #     summary_df.to_csv('./figures/all_careers_summary.csv', index=False, encoding='utf-8-sig')