"""
============================================================
图表配置与美化工具 (Plot Configuration & Styling)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：统一的图表样式配置、配色方案、学术论文级可视化
作者：MCM/ICM Team
日期：2026年1月
============================================================
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


class PlotStyleConfig:
    """
    图表美化配置类 - 符合学术论文标准
    
    使用方法：
        from visualization.plot_config import PlotStyleConfig
        PlotStyleConfig.setup_style()
    """
    
    # MCM/ICM 推荐配色方案
    COLORS = {
        'primary': '#2E86AB',      # 主色调-深蓝
        'secondary': '#A23B72',    # 辅助色-玫红
        'accent': '#F18F01',       # 强调色-橙色
        'success': '#27AE60',      # 成功-绿色
        'danger': '#C73E1D',       # 危险/最优-红色
        'warning': '#F39C12',      # 警告-黄色
        'neutral': '#3B3B3B',      # 中性色-深灰
        'background': '#FAFAFA',   # 背景色
        'grid': '#E0E0E0',         # 网格色
        'text': '#2C3E50',         # 文本色
    }
    
    # 学术配色板（离散）
    PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4C9A', '#1B998B', '#E94F37', '#44AF69']
    
    # 渐变配色板
    SEQUENTIAL = ['#DEEBF7', '#9ECAE1', '#4292C6', '#2171B5', '#084594']
    DIVERGING = ['#D73027', '#F46D43', '#FDAE61', '#FEE090', '#FFFFBF', '#E0F3F8', '#ABD9E9', '#74ADD1', '#4575B4']
    
    # 传染病/生态模型专用颜色
    EPIDEMIC_COLORS = {
        'S': '#2E86AB',  # 易感者 - 蓝色
        'E': '#F18F01',  # 暴露者 - 橙色
        'I': '#C73E1D',  # 感染者 - 红色
        'R': '#27AE60',  # 康复者 - 绿色
        'D': '#3B3B3B'   # 死亡者 - 黑色
    }
    
    # 算法配色
    ALGO_COLORS = {
        'PSO': '#2E86AB',
        'GA': '#A23B72',
        'ACO': '#F18F01',
        'SA': '#C73E1D',
        'DE': '#6B4C9A',
        'NSGA': '#1B998B'
    }
    
    @staticmethod
    def setup_style(style='academic'):
        """
        设置全局绘图风格
        
        :param style: 'academic'(学术论文) / 'presentation'(演示) / 'dark'(深色)
        """
        if style == 'academic':
            plt.style.use('seaborn-v0_8-whitegrid')
            rcParams['figure.figsize'] = (10, 6)
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
            rcParams['axes.linewidth'] = 1.2
            rcParams['lines.linewidth'] = 2
            rcParams['lines.markersize'] = 6
        elif style == 'presentation':
            plt.style.use('seaborn-v0_8-whitegrid')
            rcParams['figure.figsize'] = (14, 8)
            rcParams['figure.dpi'] = 100
            rcParams['savefig.dpi'] = 150
            rcParams['font.size'] = 14
            rcParams['axes.titlesize'] = 18
            rcParams['axes.labelsize'] = 16
            rcParams['legend.fontsize'] = 14
        elif style == 'dark':
            plt.style.use('dark_background')
            rcParams['figure.figsize'] = (10, 6)
            
        # 支持中文
        rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        rcParams['axes.unicode_minus'] = False
        
    @staticmethod
    def get_color(name):
        """获取单个颜色"""
        return PlotStyleConfig.COLORS.get(name, '#2E86AB')
    
    @staticmethod
    def get_palette(n=None):
        """获取配色板"""
        palette = PlotStyleConfig.PALETTE
        if n is not None:
            if n <= len(palette):
                return palette[:n]
            else:
                # 循环使用颜色
                return [palette[i % len(palette)] for i in range(n)]
        return palette
    
    @staticmethod
    def create_colormap(name='blues'):
        """创建渐变色图"""
        from matplotlib.colors import LinearSegmentedColormap
        if name == 'blues':
            colors = PlotStyleConfig.SEQUENTIAL
        elif name == 'diverging':
            colors = PlotStyleConfig.DIVERGING
        else:
            colors = PlotStyleConfig.PALETTE[:5]
        return LinearSegmentedColormap.from_list(name, colors)


class FigureSaver:
    """图表保存工具类"""
    
    def __init__(self, save_dir='./figures', format='png'):
        """
        :param save_dir: 保存目录
        :param format: 默认格式 'png', 'pdf', 'svg'
        """
        import os
        self.save_dir = save_dir
        self.format = format
        os.makedirs(save_dir, exist_ok=True)
        
    def save(self, fig, filename, formats=None, tight=True):
        """
        保存图表
        
        :param fig: matplotlib figure对象
        :param filename: 文件名（不含扩展名）
        :param formats: 保存格式列表，如 ['png', 'pdf']
        :param tight: 是否使用tight_layout
        """
        import os
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


class PlotTemplates:
    """常用图表模板"""
    
    @staticmethod
    def comparison_bar(data, labels, title='对比分析', xlabel='类别', ylabel='数值',
                      colors=None, show_values=True, figsize=(10, 6)):
        """
        对比柱状图模板
        
        :param data: 数据列表
        :param labels: 标签列表
        :param title: 标题
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if colors is None:
            colors = PlotStyleConfig.get_palette(len(data))
            
        x = np.arange(len(data))
        bars = ax.bar(x, data, color=colors, edgecolor='white', linewidth=2)
        
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        
        if show_values:
            for bar, val in zip(bars, data):
                ax.annotate(f'{val:.2f}', 
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 5), textcoords='offset points',
                           ha='center', va='bottom', fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return fig, ax
    
    @staticmethod
    def trend_line(x, y, title='趋势分析', xlabel='时间', ylabel='数值',
                  color=None, show_points=True, figsize=(12, 6)):
        """
        趋势线图模板
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if color is None:
            color = PlotStyleConfig.COLORS['primary']
            
        ax.plot(x, y, color=color, linewidth=2.5, label='趋势')
        
        if show_points:
            ax.scatter(x, y, color=color, s=50, zorder=5, edgecolors='white', linewidth=2)
            
        ax.fill_between(x, y, alpha=0.2, color=color)
        
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return fig, ax
    
    @staticmethod
    def heatmap(data, row_labels, col_labels, title='热力图',
               cmap='Blues', annotate=True, figsize=(10, 8)):
        """
        热力图模板
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        if annotate:
            for i in range(len(row_labels)):
                for j in range(len(col_labels)):
                    text = ax.text(j, i, f'{data[i, j]:.2f}',
                                  ha='center', va='center', color='black', fontsize=9)
        
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        return fig, ax
    
    @staticmethod
    def multi_line(x, y_dict, title='多线对比', xlabel='X', ylabel='Y',
                  figsize=(12, 6)):
        """
        多线对比图
        
        :param y_dict: {'线条名': y数据, ...}
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = PlotStyleConfig.get_palette(len(y_dict))
        
        for (name, y), color in zip(y_dict.items(), colors):
            ax.plot(x, y, color=color, linewidth=2.5, label=name, marker='o', markersize=4)
            
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='best', framealpha=0.9)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return fig, ax
    
    @staticmethod
    def distribution(data, title='分布分析', xlabel='数值', ylabel='频率',
                    color=None, bins=30, kde=True, figsize=(10, 6)):
        """
        分布直方图（可选KDE）
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if color is None:
            color = PlotStyleConfig.COLORS['primary']
            
        ax.hist(data, bins=bins, density=True, color=color, alpha=0.7, 
               edgecolor='white', linewidth=1.5, label='频率分布')
        
        if kde:
            from scipy import stats
            x_range = np.linspace(min(data), max(data), 200)
            kde_func = stats.gaussian_kde(data)
            ax.plot(x_range, kde_func(x_range), color=PlotStyleConfig.COLORS['danger'],
                   linewidth=2.5, label='KDE估计')
                   
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='best')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return fig, ax


# 初始化默认样式
PlotStyleConfig.setup_style()


if __name__ == "__main__":
    # 演示
    print("="*60)
    print("📊 图表配置工具演示")
    print("="*60)
    
    # 1. 对比柱状图
    data = [85, 72, 90, 65, 78]
    labels = ['方案A', '方案B', '方案C', '方案D', '方案E']
    fig1, ax1 = PlotTemplates.comparison_bar(data, labels, title='方案评分对比')
    plt.show()
    
    # 2. 趋势图
    x = np.arange(0, 10, 0.5)
    y = np.sin(x) * np.exp(-x/5) + 1
    fig2, ax2 = PlotTemplates.trend_line(x, y, title='趋势变化分析')
    plt.show()
    
    # 3. 分布图
    data = np.random.normal(50, 10, 500)
    fig3, ax3 = PlotTemplates.distribution(data, title='数据分布分析')
    plt.show()
    
    print("\n✅ 演示完成!")
