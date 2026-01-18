"""
============================================================
模块串联使用指南
How to Connect: Preprocessing → Model → Visualization
============================================================

本文件展示如何将分散的三个模块串联起来使用：
- data_preprocessing/  → 数据清洗、标准化、降维
- models/              → 各类数学模型
- visualization/       → 统一的图表样式

核心思路：每个模块都是独立的积木，你可以自由组合！

============================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目根路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 第一部分：理解模块关系
# ============================================================
"""
你的项目结构像一个工具箱：

modelCode/
├── data_preprocessing/     # 🔧 数据处理工具
│   ├── DataCleaner         - 缺失值、重复值处理
│   ├── DataScaler          - 标准化、归一化
│   ├── OutlierDetector     - 异常值检测
│   └── PCAReducer          - 降维
│
├── models/                 # 🧮 数学模型
│   ├── optimization/       - 优化模型
│   ├── prediction/         - 预测模型
│   ├── classification/     - 分类模型
│   └── ...
│
└── visualization/          # 📊 可视化工具
    ├── PlotStyleConfig     - 论文级样式配置
    ├── FigureSaver         - 图表保存
    └── PlotTemplates       - 常用图表模板


串联方式：

    ┌─────────────────┐
    │     原始数据     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐     使用 data_preprocessing/
    │   数据预处理     │     DataCleaner, DataScaler, ...
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐     使用 models/
    │   模型计算       │     Solver, Predictor, ...
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐     使用 visualization/
    │   结果可视化     │     PlotStyleConfig, PlotTemplates
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   保存结果       │     图片 + 数据文件
    └─────────────────┘
"""


# ============================================================
# 第二部分：实战示例 - 投资组合优化完整流程
# ============================================================

def example_full_workflow():
    """
    完整示例：从原始数据到最终结果
    
    场景：投资组合优化
    - 原始数据有缺失值和异常值
    - 需要清洗后建立优化模型
    - 最后生成论文级图表
    """
    
    print("\n" + "="*70)
    print("   完整工作流示例：投资组合优化")
    print("="*70)
    
    # ==========================================
    # 第1步：导入各模块
    # ==========================================
    print("\n【步骤1】导入模块")
    
    # 数据预处理模块
    from data_preprocessing.preprocessing_tools import DataCleaner, DataScaler, OutlierDetector
    
    # 优化模型
    from models.optimization.advanced_nonlinear_programming import NonlinearProgrammingSolver
    
    # 可视化模块
    from visualization.plot_config import PlotStyleConfig, FigureSaver
    
    print("  ✓ 模块导入完成")
    
    
    # ==========================================
    # 第2步：准备原始数据（模拟真实场景）
    # ==========================================
    print("\n【步骤2】准备原始数据")
    
    # 模拟4种资产的历史收益率数据（含缺失值和异常值）
    np.random.seed(42)
    n_days = 252  # 一年交易日
    
    raw_data = pd.DataFrame({
        '科技股': np.random.normal(0.0012, 0.02, n_days),
        '消费股': np.random.normal(0.0008, 0.015, n_days),
        '债券': np.random.normal(0.0005, 0.008, n_days),
        '黄金': np.random.normal(0.0006, 0.012, n_days)
    })
    
    # 人为添加一些问题数据
    raw_data.iloc[10, 0] = np.nan      # 缺失值
    raw_data.iloc[50, 1] = np.nan
    raw_data.iloc[100, 0] = 0.5        # 异常值（50%日收益率不太可能）
    raw_data.iloc[150, 2] = -0.3
    
    print(f"  原始数据形状: {raw_data.shape}")
    print(f"  缺失值数量: {raw_data.isnull().sum().sum()}")
    
    
    # ==========================================
    # 第3步：数据预处理（使用 data_preprocessing）
    # ==========================================
    print("\n【步骤3】数据预处理")
    
    # 3.1 数据质量检查
    cleaner = DataCleaner(verbose=True)
    cleaner.check_quality(raw_data)
    
    # 3.2 填充缺失值
    clean_data = cleaner.fill_missing(raw_data, method='median')
    print(f"  ✓ 缺失值已填充 (使用中位数)")
    
    # 3.3 异常值检测与处理
    outlier_detector = OutlierDetector(verbose=True)
    clean_data, outlier_info = outlier_detector.detect_zscore(
        clean_data, 
        threshold=3.0,
        handle='clip'  # 将异常值裁剪到边界
    )
    print(f"  ✓ 异常值已处理")
    
    # 3.4 计算模型需要的参数（从清洗后的数据）
    expected_returns = clean_data.mean().values * 252  # 年化收益率
    cov_matrix = clean_data.cov().values * 252          # 年化协方差矩阵
    
    print(f"\n  年化预期收益率:")
    for i, col in enumerate(clean_data.columns):
        print(f"    {col}: {expected_returns[i]*100:.2f}%")
    
    
    # ==========================================
    # 第4步：建模求解（使用 models）
    # ==========================================
    print("\n【步骤4】建模求解")
    
    # 定义目标函数：最小化风险
    def portfolio_risk(weights):
        return np.sqrt(weights @ cov_matrix @ weights)
    
    # 约束条件
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
        {'type': 'ineq', 'fun': lambda w: np.dot(expected_returns, w) - 0.08}  # 收益>=8%
    ]
    
    # 变量边界
    bounds = [(0, 1)] * 4
    
    # 求解
    solver = NonlinearProgrammingSolver(verbose=True)
    result = solver.multistart_solve(
        objective=portfolio_risk,
        bounds=bounds,
        n_starts=10,
        constraints=constraints
    )
    
    optimal_weights = result['x']
    optimal_risk = result['fun']
    optimal_return = np.dot(expected_returns, optimal_weights)
    
    
    # ==========================================
    # 第5步：结果可视化（使用 visualization）
    # ==========================================
    print("\n【步骤5】结果可视化")
    
    # 5.1 设置论文级样式
    PlotStyleConfig.setup_style('academic')
    colors = PlotStyleConfig.get_palette(4)
    
    # 5.2 创建图表保存器
    saver = FigureSaver(save_dir='./figures', format='png')
    
    # 5.3 绘制资产配置饼图
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    
    wedges, texts, autotexts = ax1.pie(
        optimal_weights, 
        labels=clean_data.columns,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.02]*4,
        shadow=True
    )
    ax1.set_title('最优投资组合配置', fontsize=16, fontweight='bold', pad=20)
    
    # 添加说明文字
    info_text = f"预期年收益: {optimal_return*100:.2f}%\n风险(标准差): {optimal_risk*100:.2f}%"
    ax1.text(0, -1.3, info_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('./figures/portfolio_allocation.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ 饼图已保存")
    
    # 5.4 绘制风险-收益对比柱状图
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 各资产收益率
    bars1 = axes[0].bar(clean_data.columns, expected_returns * 100, color=colors, edgecolor='white')
    axes[0].set_ylabel('年化收益率 (%)', fontweight='bold')
    axes[0].set_title('各资产预期收益率', fontsize=14, fontweight='bold')
    axes[0].axhline(y=optimal_return*100, color='red', linestyle='--', label=f'组合收益: {optimal_return*100:.1f}%')
    axes[0].legend()
    for bar, val in zip(bars1, expected_returns * 100):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                    f'{val:.1f}%', ha='center', fontsize=10)
    
    # 各资产风险
    individual_risks = np.sqrt(np.diag(cov_matrix)) * 100
    bars2 = axes[1].bar(clean_data.columns, individual_risks, color=colors, edgecolor='white')
    axes[1].set_ylabel('年化风险 (%)', fontweight='bold')
    axes[1].set_title('各资产风险 vs 组合风险', fontsize=14, fontweight='bold')
    axes[1].axhline(y=optimal_risk*100, color='red', linestyle='--', label=f'组合风险: {optimal_risk*100:.1f}%')
    axes[1].legend()
    for bar, val in zip(bars2, individual_risks):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                    f'{val:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('./figures/risk_return_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✓ 对比图已保存")
    
    
    # ==========================================
    # 第6步：生成结果报告
    # ==========================================
    print("\n【步骤6】结果汇总")
    print("="*50)
    print("📊 投资组合优化结果")
    print("="*50)
    print("\n最优资产配置:")
    for i, col in enumerate(clean_data.columns):
        print(f"  {col}: {optimal_weights[i]*100:.2f}%")
    print(f"\n预期年化收益: {optimal_return*100:.2f}%")
    print(f"年化风险(标准差): {optimal_risk*100:.2f}%")
    print(f"夏普比率(假设无风险利率2%): {(optimal_return-0.02)/optimal_risk:.2f}")
    print("="*50)
    
    return result


# ============================================================
# 第三部分：模块串联的通用模式
# ============================================================

def create_reusable_pipeline():
    """
    创建一个可复用的分析流水线类
    
    这展示了如何将三个模块封装成一个可复用的工具
    """
    
    class AnalysisPipeline:
        """
        通用分析流水线
        
        将数据预处理、建模、可视化串联起来
        """
        
        def __init__(self, save_dir='./figures', verbose=True):
            self.save_dir = save_dir
            self.verbose = verbose
            
            # 延迟导入，避免循环依赖
            self.cleaner = None
            self.scaler = None
            self.saver = None
            
            # 存储中间结果
            self.raw_data = None
            self.clean_data = None
            self.model_result = None
            
        def _init_components(self):
            """初始化各组件"""
            from data_preprocessing.preprocessing_tools import DataCleaner, DataScaler
            from visualization.plot_config import PlotStyleConfig, FigureSaver
            
            self.cleaner = DataCleaner(verbose=self.verbose)
            self.scaler = DataScaler()
            self.saver = FigureSaver(save_dir=self.save_dir)
            PlotStyleConfig.setup_style('academic')
            
        def load_data(self, data):
            """加载数据"""
            if isinstance(data, str):
                # 如果是文件路径
                if data.endswith('.csv'):
                    self.raw_data = pd.read_csv(data)
                elif data.endswith('.xlsx'):
                    self.raw_data = pd.read_excel(data)
            else:
                self.raw_data = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            if self.verbose:
                print(f"✓ 数据已加载: {self.raw_data.shape}")
            return self
        
        def preprocess(self, fill_method='auto', remove_outliers=True, scale_method=None):
            """
            数据预处理
            
            :param fill_method: 缺失值填充方法
            :param remove_outliers: 是否处理异常值
            :param scale_method: 标准化方法 (None/'standard'/'minmax')
            """
            self._init_components()
            
            # 填充缺失值
            self.clean_data = self.cleaner.fill_missing(self.raw_data, method=fill_method)
            
            # 处理异常值
            if remove_outliers:
                from data_preprocessing.preprocessing_tools import OutlierDetector
                detector = OutlierDetector(verbose=self.verbose)
                self.clean_data, _ = detector.detect_zscore(self.clean_data, handle='clip')
            
            # 标准化
            if scale_method:
                self.clean_data = pd.DataFrame(
                    self.scaler.fit_transform(self.clean_data, method=scale_method),
                    columns=self.clean_data.columns
                )
            
            if self.verbose:
                print(f"✓ 预处理完成: {self.clean_data.shape}")
            return self
        
        def run_model(self, model_func, **kwargs):
            """
            运行模型
            
            :param model_func: 模型函数，接收数据返回结果
            :param kwargs: 传递给模型的参数
            """
            self.model_result = model_func(self.clean_data, **kwargs)
            
            if self.verbose:
                print(f"✓ 模型运行完成")
            return self
        
        def visualize(self, plot_func, filename=None, **kwargs):
            """
            可视化
            
            :param plot_func: 绘图函数
            :param filename: 保存的文件名
            """
            fig = plot_func(self.model_result, **kwargs)
            
            if filename and self.saver:
                self.saver.save(fig, filename)
            
            plt.show()
            return self
        
        def get_result(self):
            """获取最终结果"""
            return {
                'raw_data': self.raw_data,
                'clean_data': self.clean_data,
                'model_result': self.model_result
            }
    
    return AnalysisPipeline


# ============================================================
# 第四部分：快速串联技巧
# ============================================================
"""
【技巧1：链式调用】

pipeline = AnalysisPipeline()
result = (pipeline
          .load_data('data.csv')
          .preprocess(fill_method='median')
          .run_model(my_model)
          .visualize(my_plot)
          .get_result())


【技巧2：函数组合】

def full_analysis(raw_data):
    # 预处理
    from data_preprocessing import DataCleaner
    clean_data = DataCleaner().fill_missing(raw_data)
    
    # 建模
    from models.optimization import NonlinearProgrammingSolver
    result = NonlinearProgrammingSolver().solve(...)
    
    # 可视化
    from visualization import PlotStyleConfig
    PlotStyleConfig.setup_style()
    plt.plot(...)
    
    return result


【技巧3：配置驱动】

config = {
    'preprocessing': {'fill_method': 'median', 'remove_outliers': True},
    'model': {'method': 'SLSQP', 'multistart': True},
    'visualization': {'style': 'academic', 'save_format': 'png'}
}

# 根据配置自动选择处理方式


【技巧4：使用上下文管理器】

class AnalysisContext:
    def __init__(self, style='academic'):
        self.style = style
        
    def __enter__(self):
        from visualization import PlotStyleConfig
        PlotStyleConfig.setup_style(self.style)
        return self
        
    def __exit__(self, *args):
        plt.close('all')

# 使用
with AnalysisContext('academic'):
    # 所有图表自动使用论文样式
    plt.plot(...)
"""


# ============================================================
# 运行示例
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("   模块串联使用指南 - 演示")
    print("="*70)
    
    # 运行完整示例
    example_full_workflow()
    
    print("\n" + "="*70)
    print("   演示完成！")
    print("="*70)
    print("""
    
总结：串联三个模块的方法

1. 【直接导入】分别导入需要的类，按顺序调用
   from data_preprocessing import DataCleaner
   from models.optimization import Solver
   from visualization import PlotStyleConfig

2. 【封装Pipeline】创建一个Pipeline类，内部串联各组件
   pipeline.load_data() → preprocess() → run_model() → visualize()

3. 【函数组合】写一个函数，内部依次调用各模块

关键点：
- 每个模块保持独立，通过数据传递连接
- 预处理输出 → 模型输入
- 模型输出 → 可视化输入
- 使用统一的数据格式（推荐 DataFrame 或 ndarray）

""")
