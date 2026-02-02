"""
============================================================
AI 驱动的教育决策模型 - 拆分可视化模块
(Separated Visualization Module - Enhanced & Beautified)
============================================================
功能：将所有可视化图表拆分为单独的图片，每个图片独立保存
美化：采用专业学术风格，高对比度配色，精细化标注
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 图表配置 - 专业学术风格
# ============================================================

class PlotStyleConfig:
    """图表美化配置类 - 专业学术风格"""

    # 高对比度专业配色方案
    COLORS = {
        'primary': '#2E86AB',     # 深海蓝 - 主色调
        'secondary': '#E94F37',   # 珊瑚红 - 强调色
        'accent': '#1B998B',      # 翡翠绿 - 成功/正面
        'danger': '#C73E1D',      # 砖红 - 警告/危险
        'neutral': '#5C6B73',     # 石墨灰 - 中性
        'background': '#FAFBFC',  # 纯净白背景
        'grid': '#E1E5E8',        # 柔和网格
        'gold': '#F2A541',        # 金色 - 突出
        'purple': '#7B68EE',      # 紫色 - 额外强调
        'dark': '#2C3E50'         # 深色文字
    }

    # 高对比度专业调色板
    PALETTE = [
        '#2E86AB',  # 深海蓝
        '#E94F37',  # 珊瑚红
        '#1B998B',  # 翡翠绿
        '#F2A541',  # 金色
        '#7B68EE',  # 紫色
        '#20BF55',  # 鲜绿
        '#FF6B6B',  # 粉红
        '#4ECDC4',  # 青色
        '#45B7D1',  # 天蓝
        '#96CEB4'   # 薄荷绿
    ]
    
    # 学校专属颜色 - 高辨识度
    SCHOOL_COLORS = {
        'CMU': '#C41E3A',   # 卡内基红
        'CCAD': '#FF6B35',  # 橙红
        'CIA': '#1E3A5F'    # 深蓝
    }
    
    # 课程类型颜色 - 柔和淡雅
    COURSE_COLORS = {
        'x_base': '#5B9BEF',    # 蓝色 - Base
        'x_AI': '#F69D62',      # 橙色 - AI
        'x_ethics': '#80EF6A',  # 绿色 - Ethics
        'x_proj': '#EA9DE1'     # 紫色 - Project
    }

    @staticmethod
    def setup_style():
        """设置全局绘图风格"""
        plt.style.use('seaborn-v0_8-whitegrid')
        rcParams['font.family'] = 'DejaVu Sans'
        rcParams['font.size'] = 11
        rcParams['axes.titlesize'] = 14
        rcParams['axes.labelsize'] = 12
        rcParams['xtick.labelsize'] = 10
        rcParams['ytick.labelsize'] = 10
        rcParams['legend.fontsize'] = 10
        rcParams['figure.titlesize'] = 16
        rcParams['axes.spines.top'] = False
        rcParams['axes.spines.right'] = False
        rcParams['axes.facecolor'] = '#FAFBFC'
        rcParams['figure.facecolor'] = 'white'
        rcParams['axes.edgecolor'] = '#CCCCCC'
        rcParams['grid.alpha'] = 0.4
        rcParams['grid.linestyle'] = '--'

    @staticmethod
    def get_school_color(school_name):
        return PlotStyleConfig.SCHOOL_COLORS.get(school_name, '#7f7f7f')

    @staticmethod
    def get_palette(n=None):
        if n is None:
            return PlotStyleConfig.PALETTE
        return PlotStyleConfig.PALETTE[:n]


class FigureSaver:
    """图表保存工具类"""

    def __init__(self, save_dir='./figures/task2_2_split', prefix=''):
        self.save_dir = save_dir
        self.prefix = prefix
        os.makedirs(save_dir, exist_ok=True)

    def save(self, fig, filename, formats=None, dpi=300):
        if formats is None:
            formats = ['png', 'pdf']
        plt.tight_layout()
        paths = []
        full_filename = f"{self.prefix}_{filename}" if self.prefix else filename
        for fmt in formats:
            path = os.path.join(self.save_dir, f"{full_filename}.{fmt}")
            fig.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            paths.append(path)
        plt.close(fig)
        return paths


# 初始化绘图风格
PlotStyleConfig.setup_style()


# ============================================================
# 拆分后的可视化类
# ============================================================

class SplitVisualization:
    """
    拆分可视化类 - 每个图表独立保存
    """

    def __init__(self, model, results, save_dir='./figures/task2_2_split'):
        self.model = model
        self.results = results
        self.school = model.params.school_name
        self.saver = FigureSaver(save_dir, prefix=self.school)

    # ========================================
    # 1. 招生响应相关图表
    # ========================================
    
    def plot_enrollment_bar_chart(self, figsize=(10, 7)):
        """
        招生响应 - 柱状图：供需对比
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        r = self.results['enrollment_response']
        colors = [PlotStyleConfig.COLORS['primary'], 
                  PlotStyleConfig.COLORS['accent'], 
                  PlotStyleConfig.COLORS['secondary']]

        values = [self.model.params.current_graduates, 
                  r['recommended_graduates'], 
                  self.model.params.demand_2030]
        labels = ['Current Supply\n(S_t)', 'Optimized Plan\n(A_t)', 'Market Demand\n(D_t)']
        
        bars = ax.bar(labels, values, color=colors, alpha=0.9, width=0.55,
                     edgecolor='white', linewidth=2)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + max(values)*0.02,
                   f'{val:.0f}', ha='center', va='bottom', 
                   fontsize=14, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
            
            # 变化率标注
            if i > 0:
                change = (val - values[0]) / values[0] * 100
                symbol = '▲' if change > 0 else '▼'
                color = PlotStyleConfig.COLORS['accent'] if change > 0 else PlotStyleConfig.COLORS['danger']
                ax.text(bar.get_x() + bar.get_width()/2, height - height*0.12,
                       f'{symbol} {abs(change):.1f}%', ha='center', va='center', 
                       fontsize=11, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9))

        ax.set_ylabel('Number of Graduates', fontsize=13, fontweight='bold')
        ax.set_title(f'{self.school} - Supply vs Demand Analysis', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, max(values) * 1.2)
        
        # 添加压力指数信息框
        info_text = (f"Pressure Index (Γ) = {r['pressure_index']:.3f}\n"
                    f"Adjustment (ΔA) = {r['adjustment']:+.1f}\n"
                    f"Admin Capacity (λ) = {self.model.params.lambda_admin:.3f}")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11, 
               verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor=PlotStyleConfig.COLORS['primary'], linewidth=2))

        paths = self.saver.save(fig, 'enrollment_bar_chart')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_enrollment_flow_diagram(self, figsize=(12, 6)):
        """
        招生响应 - 流程图：招生调整流程
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis('off')
        
        r = self.results['enrollment_response']
        p = self.model.params
        
        # 绘制流程框
        box_style = dict(boxstyle='round,pad=0.5', facecolor=PlotStyleConfig.COLORS['background'],
                        edgecolor=PlotStyleConfig.COLORS['primary'], linewidth=2)
        arrow_style = dict(arrowstyle='->', color=PlotStyleConfig.COLORS['gold'], lw=3)
        
        # 节点位置
        positions = {
            'current': (1.5, 2),
            'demand': (8.5, 2),
            'pressure': (5, 3.2),
            'result': (5, 0.8)
        }
        
        # 绘制节点
        ax.text(*positions['current'], f"Current Supply\nS_t = {p.current_graduates:.0f}", 
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#E3F2FD', 
                        edgecolor=PlotStyleConfig.COLORS['primary'], linewidth=2))
        
        ax.text(*positions['demand'], f"Market Demand\nD_t = {p.demand_2030:.0f}", 
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFEBEE', 
                        edgecolor=PlotStyleConfig.COLORS['secondary'], linewidth=2))
        
        ax.text(*positions['pressure'], f"Pressure Index\nΓ = (D-S)/S = {r['pressure_index']:.3f}", 
               ha='center', va='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0', 
                        edgecolor=PlotStyleConfig.COLORS['gold'], linewidth=2))
        
        ax.text(*positions['result'], 
               f"Adjustment Formula:\nΔA = S × λ × tanh(Γ)\n= {r['adjustment']:+.1f} students", 
               ha='center', va='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#E8F5E9', 
                        edgecolor=PlotStyleConfig.COLORS['accent'], linewidth=2))
        
        # 绘制箭头
        ax.annotate('', xy=(4, 3.2), xytext=(2.5, 2.5), arrowprops=arrow_style)
        ax.annotate('', xy=(6, 3.2), xytext=(7.5, 2.5), arrowprops=arrow_style)
        ax.annotate('', xy=(5, 1.5), xytext=(5, 2.7), arrowprops=arrow_style)
        
        ax.set_title(f'{self.school} - Enrollment Adjustment Flow', 
                    fontsize=16, fontweight='bold', pad=10)

        paths = self.saver.save(fig, 'enrollment_flow_diagram')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    # ========================================
    # 2. 课程优化相关图表
    # ========================================

    def plot_curriculum_comparison_bar(self, figsize=(10, 7)):
        """
        课程优化 - 柱状图：优化前后对比
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        r = self.results['curriculum_optimization']
        keys = ['x_base', 'x_AI', 'x_ethics', 'x_proj']
        labels = ['Base\n(Traditional)', 'AI\n(Tech-Enhanced)', 
                  'Ethics\n(Responsibility)', 'Project\n(Hands-on)']
        
        current = [self.model.params.current_curriculum.get(k, 0) for k in keys]
        optimal = [r['optimal_curriculum'].get(k, 0) for k in keys]
        
        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, current, width, label='Initial Curriculum', 
                      color=PlotStyleConfig.COLORS['neutral'], alpha=0.7, 
                      edgecolor='white', linewidth=1.5)
        bars2 = ax.bar(x + width/2, optimal, width, label='Optimized Curriculum', 
                      color=[PlotStyleConfig.COURSE_COLORS[k] for k in keys], 
                      alpha=0.9, edgecolor='white', linewidth=1.5)
        
        # 添加数值标签和变化率
        for i, (b1, b2, cur, opt) in enumerate(zip(bars1, bars2, current, optimal)):
            ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 1,
                   f'{cur:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 1,
                   f'{opt:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 变化量标注
            diff = opt - cur
            if abs(diff) > 0.5:
                color = PlotStyleConfig.COLORS['accent'] if diff > 0 else PlotStyleConfig.COLORS['danger']
                mid_x = (b1.get_x() + b2.get_x() + b2.get_width()) / 2
                ax.annotate(f'{diff:+.0f}', xy=(mid_x, max(cur, opt) + 3),
                           ha='center', fontsize=9, fontweight='bold', color=color)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Credit Hours', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.school} - Curriculum Structure Optimization', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.legend(loc='upper right', frameon=True, fancybox=True)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, max(max(current), max(optimal)) * 1.25)

        paths = self.saver.save(fig, 'curriculum_comparison_bar')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_curriculum_pie_chart(self, figsize=(10, 8)):
        """
        课程优化 - 饼图：优化后的学分分布
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        r = self.results['curriculum_optimization']
        keys = ['x_base', 'x_AI', 'x_ethics', 'x_proj']
        labels = ['Core', 'AI', 'Human', 'Cross']
        values = [r['optimal_curriculum'].get(k, 0) for k in keys]
        colors = [PlotStyleConfig.COURSE_COLORS[k] for k in keys]
        
        # 过滤掉零值
        non_zero_mask = [v > 0 for v in values]
        values = [v for v, m in zip(values, non_zero_mask) if m]
        labels = [l for l, m in zip(labels, non_zero_mask) if m]
        colors = [c for c, m in zip(colors, non_zero_mask) if m]
        
        # 突出AI部分
        explode = [0.05 if l == 'AI' else 0 for l in labels]
        
        wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90, 
                                          pctdistance=0.75, explode=explode,
                                          wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
                                          textprops=dict(fontsize=12, fontweight='bold'))
        
        # 中心标注总学分
        total = sum(values)
        ax.text(0, 0, f'{total:.0f}\nCredits', ha='center', va='center', 
               fontsize=16, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        
        ax.set_title(f'{self.school} - Optimized Credit Distribution', 
                    fontsize=16, fontweight='bold', pad=15)

        paths = self.saver.save(fig, 'curriculum_pie_chart')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_utility_breakdown(self, figsize=(10, 7)):
        """
        课程优化 - 柱状图：效用分解
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        r = self.results['curriculum_optimization']
        
        # 获取效用权重
        p = self.model.params
        if p.school_name == 'CMU':
            base_w = {'x_base': 0.45, 'x_AI': 0.35, 'x_ethics': 0.15, 'x_proj': 0.05}
        elif p.school_name == 'CCAD':
            base_w = {'x_base': 0.25, 'x_AI': 0.25, 'x_proj': 0.45, 'x_ethics': 0.05}
        elif p.school_name == 'CIA':
            base_w = {'x_base': 0.30, 'x_AI': 0.10, 'x_proj': 0.60, 'x_ethics': 0.0}
        else:
            base_w = {'x_base': 0.3, 'x_AI': 0.3, 'x_proj': 0.3, 'x_ethics': 0.1}
        
        keys = ['x_base', 'x_AI', 'x_ethics', 'x_proj']
        labels = ['Core', 'AI', 'Human', 'Cross']
        colors = [PlotStyleConfig.COURSE_COLORS[k] for k in keys]
        
        # 计算各部分效用贡献
        contributions = []
        for k in keys:
            credits = r['optimal_curriculum'].get(k, 0)
            weight = base_w.get(k, 0)
            utility = weight * np.sqrt(credits) if credits > 0 else 0
            contributions.append(utility)
        
        bars = ax.bar(labels, contributions, color=colors, alpha=0.9, 
                     edgecolor='white', linewidth=2, width=0.6)
        
        # 添加数值标签
        for bar, val in zip(bars, contributions):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{val:.2f}', ha='center', va='bottom', 
                       fontsize=12, fontweight='bold')
        
        total_utility = sum(contributions)
        ax.axhline(y=total_utility/len([c for c in contributions if c > 0]), 
                  color=PlotStyleConfig.COLORS['gold'], linestyle='--', 
                  linewidth=2, label=f'Average: {total_utility/len([c for c in contributions if c > 0]):.2f}')
        
        ax.set_ylabel('Utility Contribution (w × √credits)', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.school} - Utility Breakdown by Course Type\n(Total Utility = {total_utility:.2f})', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend()

        paths = self.saver.save(fig, 'utility_breakdown')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_ai_marginal_utility_curve(self, figsize=(10, 7)):
        """
        课程优化 - 曲线图：AI学分边际效用
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        r = self.results['curriculum_optimization']
        
        # 获取AI权重
        p = self.model.params
        if p.school_name == 'CMU':
            ai_weight = 0.35
        elif p.school_name == 'CCAD':
            ai_weight = 0.25
        elif p.school_name == 'CIA':
            ai_weight = 0.10
        else:
            ai_weight = 0.3
        
        # 生成效用曲线
        x_range = np.linspace(1, 60, 100)
        utility_curve = ai_weight * np.sqrt(x_range)
        marginal_curve = ai_weight * 0.5 / np.sqrt(x_range)  # 导数
        
        # 绘制效用曲线
        ax.plot(x_range, utility_curve, color=PlotStyleConfig.COLORS['primary'], 
               linewidth=3, label='Total Utility: w × √x')
        
        # 绘制边际效用曲线
        ax_twin = ax.twinx()
        ax_twin.plot(x_range, marginal_curve, color=PlotStyleConfig.COLORS['secondary'], 
                    linewidth=2.5, linestyle='--', label='Marginal Utility: w/(2√x)')
        
        # 标记最优点
        opt_ai = r['optimal_curriculum'].get('x_AI', 0)
        opt_util = ai_weight * np.sqrt(opt_ai) if opt_ai > 0 else 0
        opt_marginal = ai_weight * 0.5 / np.sqrt(opt_ai) if opt_ai > 0 else 0
        
        ax.scatter([opt_ai], [opt_util], s=150, color=PlotStyleConfig.COLORS['accent'], 
                  zorder=5, edgecolors='white', linewidth=2, marker='o')
        ax.axvline(x=opt_ai, color=PlotStyleConfig.COLORS['accent'], linestyle=':', 
                  linewidth=2, alpha=0.7)
        
        ax.annotate(f'Optimal\n({opt_ai:.0f} credits, U={opt_util:.2f})', 
                   xy=(opt_ai, opt_util), xytext=(opt_ai+8, opt_util+0.2),
                   fontsize=11, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=PlotStyleConfig.COLORS['accent']),
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor=PlotStyleConfig.COLORS['accent']))
        
        ax.set_xlabel('AI Course Credits', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Utility', fontsize=12, fontweight='bold', 
                     color=PlotStyleConfig.COLORS['primary'])
        ax_twin.set_ylabel('Marginal Utility', fontsize=12, fontweight='bold', 
                          color=PlotStyleConfig.COLORS['secondary'])
        
        ax.tick_params(axis='y', labelcolor=PlotStyleConfig.COLORS['primary'])
        ax_twin.tick_params(axis='y', labelcolor=PlotStyleConfig.COLORS['secondary'])
        
        ax.set_title(f'{self.school} - AI Credits Utility Analysis (Diminishing Returns)', 
                    fontsize=16, fontweight='bold', pad=15)
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 60)

        paths = self.saver.save(fig, 'ai_marginal_utility_curve')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    # ========================================
    # 3. SA收敛过程
    # ========================================

    def plot_sa_convergence(self, figsize=(11, 7)):
        """
        SA收敛 - 曲线图：优化收敛过程
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        history = self.results['curriculum_optimization']['iteration_history']
        iterations = np.arange(len(history))
        
        # 计算范围
        y_min, y_max = min(history), max(history)
        y_range = y_max - y_min if y_max != y_min else 1
        
        # 渐变填充
        ax.fill_between(iterations, y_min - 0.05*y_range, history, 
                       alpha=0.3, color=PlotStyleConfig.COLORS['primary'])
        ax.plot(iterations, history, color=PlotStyleConfig.COLORS['primary'], 
               linewidth=2.5, label='Best Score')
        
        # 标记起点和终点
        ax.scatter([0], [history[0]], s=150, color=PlotStyleConfig.COLORS['danger'], 
                  zorder=5, edgecolors='white', linewidths=2, label=f'Start: {history[0]:.3f}')
        ax.scatter([len(history)-1], [history[-1]], s=200, color=PlotStyleConfig.COLORS['accent'], 
                  zorder=5, edgecolors='white', linewidths=2, marker='*', label=f'Final: {history[-1]:.3f}')
        
        # 最优线
        ax.axhline(y=history[-1], color=PlotStyleConfig.COLORS['accent'], 
                  linestyle='--', linewidth=2, alpha=0.7)
        
        # 改进率标注
        improvement = (history[-1] - history[0]) / abs(history[0]) * 100 if history[0] != 0 else 0
        ax.annotate(f'Improvement: {improvement:+.1f}%', 
                   xy=(len(history)*0.7, history[-1]), 
                   xytext=(len(history)*0.7, history[-1] + y_range*0.1),
                   fontsize=13, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=PlotStyleConfig.COLORS['gold']),
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=PlotStyleConfig.COLORS['gold'], alpha=0.9))
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Objective Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.school} - Simulated Annealing Convergence', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.4)
        ax.set_ylim(y_min - 0.1*y_range, y_max + 0.15*y_range)

        paths = self.saver.save(fig, 'sa_convergence')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    # ========================================
    # 4. 职业弹性分析
    # ========================================

    def plot_career_elasticity_bar(self, figsize=(12, 7)):
        """
        职业弹性 - 水平柱状图：相似度排名
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        r = self.results['career_elasticity']
        careers = list(r['similarities'].keys())
        similarities = list(r['similarities'].values())
        display_careers = [self.model.params.CAREER_DISPLAY_NAMES.get(c, c) for c in careers]
        
        # 按相似度排序
        sorted_indices = np.argsort(similarities)
        display_careers = [display_careers[i] for i in sorted_indices]
        similarities = [similarities[i] for i in sorted_indices]
        colors = [PlotStyleConfig.PALETTE[i % len(PlotStyleConfig.PALETTE)] for i in range(len(similarities))]
        colors = [colors[i] for i in sorted_indices]
        
        y_pos = np.arange(len(display_careers))
        bars = ax.barh(y_pos, similarities, color=colors, alpha=0.85, 
                      edgecolor='white', linewidth=1.5, height=0.7)
        
        # 添加数值标签
        for bar, sim in zip(bars, similarities):
            width = bar.get_width()
            label_x = width + 0.02 if width < 0.85 else width - 0.08
            color = 'black' if width < 0.85 else 'white'
            ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{sim:.3f}',
                   ha='left' if width < 0.85 else 'right', va='center', 
                   fontsize=11, fontweight='bold', color=color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_careers, fontsize=11)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel('Cosine Similarity Score', fontsize=12, fontweight='bold')
        
        # 阈值线
        ax.axvline(x=0.9, color=PlotStyleConfig.COLORS['accent'], linestyle='--', 
                  linewidth=2, alpha=0.8, label='High (>0.9)')
        ax.axvline(x=0.7, color=PlotStyleConfig.COLORS['gold'], linestyle='--', 
                  linewidth=2, alpha=0.8, label='Medium (>0.7)')
        ax.axvline(x=0.5, color=PlotStyleConfig.COLORS['danger'], linestyle='--', 
                  linewidth=2, alpha=0.8, label='Low (<0.5)')
        
        # 背景区域
        ax.axvspan(0.9, 1.1, alpha=0.1, color=PlotStyleConfig.COLORS['accent'])
        ax.axvspan(0.7, 0.9, alpha=0.08, color=PlotStyleConfig.COLORS['gold'])
        ax.axvspan(0, 0.5, alpha=0.08, color=PlotStyleConfig.COLORS['danger'])
        
        ax.legend(loc='lower right', fontsize=9)
        ax.set_title(f'{self.school} - Career Path Elasticity Analysis\n(Similarity to Origin Career)', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, axis='x', alpha=0.3)

        paths = self.saver.save(fig, 'career_elasticity_bar')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_single_skill_radar(self, target_career, figsize=(8, 8)):
        """
        职业弹性 - 单个技能雷达图
        """
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # 获取当前职业
        if self.school == 'CMU':
            career = 'software_engineer'
        elif self.school == 'CCAD':
            career = 'graphic_designer'
        else:
            career = 'chef'
        
        origin_vec = np.array(self.model.params.CAREER_VECTORS[career])
        target_vec = np.array(self.model.params.CAREER_VECTORS[target_career])
        features = ['Analytical', 'Creative', 'Technical', 'Interpersonal', 'Physical']
        
        # 计算角度
        num_features = len(features)
        angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
        angles += angles[:1]
        
        origin_plot = origin_vec.tolist() + origin_vec.tolist()[:1]
        target_plot = target_vec.tolist() + target_vec.tolist()[:1]
        
        # 绘制
        ax.fill(angles, origin_plot, alpha=0.25, color=PlotStyleConfig.COLORS['primary'])
        ax.plot(angles, origin_plot, 'o-', linewidth=2.5, color=PlotStyleConfig.COLORS['primary'], 
               markersize=8, markerfacecolor='white', markeredgewidth=2, 
               label=f'Origin: {self.model.params.CAREER_DISPLAY_NAMES.get(career, career)}')
        
        ax.fill(angles, target_plot, alpha=0.25, color=PlotStyleConfig.COLORS['secondary'])
        ax.plot(angles, target_plot, 's-', linewidth=2.5, color=PlotStyleConfig.COLORS['secondary'], 
               markersize=8, markerfacecolor='white', markeredgewidth=2,
               label=f'Target: {self.model.params.CAREER_DISPLAY_NAMES.get(target_career, target_career)}')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.5)
        
        similarity = self.results['career_elasticity']['similarities'].get(target_career, 0)
        ax.set_title(f'Skill Comparison\nSimilarity: {similarity:.3f}', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

        paths = self.saver.save(fig, f'skill_radar_{target_career}')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    # ========================================
    # 5. 帕累托前沿
    # ========================================

    def plot_pareto_frontier(self, figsize=(11, 8)):
        """
        资源竞争 - 帕累托前沿图
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        p = self.model.params
        if p.school_name == 'CMU':
            base_w = {'x_base': 0.38, 'x_AI': 0.35}
        elif p.school_name == 'CCAD':
            base_w = {'x_base': 0.20, 'x_AI': 0.25}
        elif p.school_name == 'CIA':
            base_w = {'x_base': 0.25, 'x_AI': 0.08}
        else:
            base_w = {'x_base': 0.3, 'x_AI': 0.3}

        # 生成样本点
        current_ethics = p.current_curriculum.get('x_ethics', 0)
        current_proj = p.current_curriculum.get('x_proj', 0)
        fixed_credits = current_ethics + current_proj
        
        points = []
        for ai_credits in np.linspace(5, 80, 50):
            base_credits = 120 - ai_credits - fixed_credits
            if base_credits >= 10:
                ai_utility = base_w.get('x_AI', 0) * np.sqrt(ai_credits)
                base_utility = base_w.get('x_base', 0) * np.sqrt(base_credits)
                points.append((ai_utility, base_utility, ai_credits))

        points = np.array(points)
        ai_utilities = points[:, 0]
        base_utilities = points[:, 1]
        
        # 绘制所有点
        scatter = ax.scatter(ai_utilities, base_utilities, c=points[:, 2], 
                            cmap='viridis', alpha=0.7, s=60, edgecolors='k', linewidth=0.5)
        
        # 计算帕累托前沿
        def is_dominated(p1, p2):
            return p1[0] <= p2[0] and p1[1] <= p2[1] and (p1[0] < p2[0] or p1[1] < p2[1])
        
        pareto_front = []
        for i, p1 in enumerate(points[:, :2]):
            dominated = False
            for j, p2 in enumerate(points[:, :2]):
                if i != j and is_dominated(p1, p2):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(p1)
        
        pareto_front = np.array(sorted(pareto_front, key=lambda x: x[0]))
        
        if len(pareto_front) > 1:
            ax.plot(pareto_front[:, 0], pareto_front[:, 1], 'r-', linewidth=3, alpha=0.8, label='Pareto Front')
            ax.fill_between(pareto_front[:, 0], pareto_front[:, 1], alpha=0.1, color='red')
        
        # 标记最优点
        r = self.results['curriculum_optimization']
        opt_ai = r['optimal_curriculum'].get('x_AI', 0)
        opt_base = r['optimal_curriculum'].get('x_base', 0)
        opt_ai_utility = base_w.get('x_AI', 0) * np.sqrt(opt_ai)
        opt_base_utility = base_w.get('x_base', 0) * np.sqrt(opt_base)
        
        ax.scatter(opt_ai_utility, opt_base_utility, color=PlotStyleConfig.COLORS['gold'], 
                  s=200, marker='*', edgecolors='black', linewidth=2, label='Optimal', zorder=10)
        ax.annotate(f'Optimal\n({opt_ai:.0f} AI, {opt_base:.0f} Base)', 
                   (opt_ai_utility, opt_base_utility), xytext=(20, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=PlotStyleConfig.COLORS['gold'], alpha=0.9),
                   fontsize=10, ha='center',
                   arrowprops=dict(arrowstyle='->', color=PlotStyleConfig.COLORS['dark']))

        ax.set_xlabel('AI Skill Utility', fontsize=12, fontweight='bold')
        ax.set_ylabel('Base Skill Utility', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.school} - Resource Competition: AI vs Base Trade-off\n(Pareto Frontier Analysis)', 
                    fontsize=16, fontweight='bold', pad=15)
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('AI Credits', fontsize=11)
        
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        paths = self.saver.save(fig, 'pareto_frontier')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    # ========================================
    # 6. 灵敏度分析（拆分）
    # ========================================

    def plot_lambda_sensitivity(self, figsize=(10, 7)):
        """
        灵敏度分析 - Lambda对招生调整的影响
        """
        if 'sensitivity_analysis' not in self.results:
            print("    ⚠️ No sensitivity analysis results found.")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        
        data = self.results['sensitivity_analysis']['lambda_sensitivity']
        x = data['range']
        y = data['adjustments']
        
        ax.plot(x, y, color=PlotStyleConfig.COLORS['primary'], linewidth=2.5, 
               marker='o', markersize=5, label='Adjustment Amount')
        ax.fill_between(x, 0, y, alpha=0.2, color=PlotStyleConfig.COLORS['primary'])
        
        # 标记当前Lambda
        current_lambda = self.model.params.lambda_admin
        current_adj = self.results['enrollment_response']['adjustment']
        ax.scatter([current_lambda], [current_adj], s=150, marker='*', 
                  color=PlotStyleConfig.COLORS['gold'], zorder=10,
                  edgecolors='black', linewidth=1.5, label=f'Current λ={current_lambda:.3f}')
        ax.axvline(current_lambda, color=PlotStyleConfig.COLORS['gold'], linestyle='--', 
                  linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Administrative Coefficient (λ)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Enrollment Adjustment (ΔA)', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.school} - Lambda Sensitivity Analysis\n(Macro Decision)', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        paths = self.saver.save(fig, 'sensitivity_lambda')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_weight_sensitivity(self, figsize=(10, 7)):
        """
        灵敏度分析 - AI权重对学分分配的影响
        """
        if 'sensitivity_analysis' not in self.results:
            print("    ⚠️ No sensitivity analysis results found.")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        
        data = self.results['sensitivity_analysis']['weight_sensitivity']
        x = data['range']
        y_ai = data['ai_credits']
        y_base = data['base_credits']
        
        ax.plot(x, y_ai, color=PlotStyleConfig.COLORS['secondary'], linewidth=2.5, 
               marker='s', markersize=5, label='AI Credits')
        ax.plot(x, y_base, color=PlotStyleConfig.COLORS['neutral'], linewidth=2, 
               linestyle='--', marker='o', markersize=4, label='Base Credits')
        
        ax.fill_between(x, y_ai, alpha=0.15, color=PlotStyleConfig.COLORS['secondary'])
        ax.fill_between(x, y_base, alpha=0.1, color=PlotStyleConfig.COLORS['neutral'])
        
        ax.set_xlabel('Weight of AI Skill (w_AI)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Optimized Credits', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.school} - AI Weight Sensitivity Analysis\n(Micro Decision)', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        paths = self.saver.save(fig, 'sensitivity_weight')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    # ========================================
    # 7. 约束相关图表
    # ========================================

    def plot_constraint_satisfaction_radar(self, figsize=(9, 8)):
        """
        约束满足 - 雷达图
        """
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        r = self.results['curriculum_optimization']
        constraint_details = r.get('constraint_details', {})
        c_params = self.model.params.constraint_params
        
        categories = ['Equity\n(E_max)', 'Green\n(β_env)', 'Safety\n(γ)']
        
        # 计算满足度
        equity_info = constraint_details.get('equity', {})
        green_info = constraint_details.get('green', {})
        safety_info = constraint_details.get('safety', {})
        
        equity_sat = 1.0 if equity_info.get('satisfied', True) else max(0, 0.5)
        green_sat = 1.0 if green_info.get('satisfied', True) else max(0, 0.5)
        safety_sat = 1.0 if safety_info.get('satisfied', True) else max(0, 0.5)
        
        values = [equity_sat, green_sat, safety_sat]
        values += values[:1]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        # 绘制满足度
        ax.fill(angles, values, alpha=0.25, color=PlotStyleConfig.COLORS['accent'])
        ax.plot(angles, values, 'o-', linewidth=3, color=PlotStyleConfig.COLORS['accent'],
               markersize=10, markerfacecolor='white', markeredgewidth=2)
        
        # 参考线（100%满足）
        full_values = [1.0] * 4
        ax.plot(angles, full_values, '--', linewidth=2, color=PlotStyleConfig.COLORS['neutral'], 
               alpha=0.5, label='100% Satisfaction')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9)
        
        ax.set_title(f'{self.school} - Constraint Satisfaction Status', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # 状态指示
        all_satisfied = all([equity_info.get('satisfied', True), 
                            green_info.get('satisfied', True), 
                            safety_info.get('satisfied', True)])
        status_text = "✓ ALL SATISFIED" if all_satisfied else "✗ VIOLATION"
        status_color = PlotStyleConfig.COLORS['accent'] if all_satisfied else PlotStyleConfig.COLORS['danger']
        ax.text(0.5, -0.1, status_text, transform=ax.transAxes, ha='center', 
               fontsize=14, fontweight='bold', color=status_color)

        paths = self.saver.save(fig, 'constraint_satisfaction_radar')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_constraint_e_max_sensitivity(self, figsize=(10, 7)):
        """
        约束灵敏度 - E_max (公平性)
        """
        from task2_2 import EducationDecisionParams, EducationDecisionModel
        
        fig, ax = plt.subplots(figsize=figsize)
        
        p = self.model.params
        original_params = p.constraint_params.copy()
        
        e_max_range = np.linspace(0.15, 0.70, 15)
        ai_credits_list = []
        scores_list = []
        
        for e_max in e_max_range:
            p.constraint_params['E_max'] = e_max
            from task2_2 import EducationDecisionModel
            model_temp = EducationDecisionModel(p)
            result = model_temp.curriculum_optimization_sa()
            ai_credits_list.append(result['optimal_curriculum']['x_AI'])
            scores_list.append(result['optimal_score'])
        
        # 恢复
        p.constraint_params = original_params
        
        ax.plot(e_max_range * 100, ai_credits_list, 'o-', color=PlotStyleConfig.COLORS['primary'],
               linewidth=2.5, markersize=6, label='AI Credits')
        ax.set_xlabel('E_max (Equity Threshold) %', fontsize=12, fontweight='bold')
        ax.set_ylabel('Optimal AI Credits', fontsize=12, fontweight='bold', 
                     color=PlotStyleConfig.COLORS['primary'])
        ax.tick_params(axis='y', labelcolor=PlotStyleConfig.COLORS['primary'])
        
        ax_twin = ax.twinx()
        ax_twin.plot(e_max_range * 100, scores_list, 's--', color=PlotStyleConfig.COLORS['secondary'],
                    linewidth=2, markersize=5, label='Net Score')
        ax_twin.set_ylabel('Net Score', fontsize=12, fontweight='bold', 
                          color=PlotStyleConfig.COLORS['secondary'])
        ax_twin.tick_params(axis='y', labelcolor=PlotStyleConfig.COLORS['secondary'])
        
        ax.axvline(original_params['E_max'] * 100, color='gray', linestyle=':', 
                  linewidth=2, alpha=0.7, label=f'Current ({original_params["E_max"]*100:.0f}%)')
        
        ax.legend(loc='upper left')
        ax.set_title(f'{self.school} - Equity Constraint (E_max) Sensitivity', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)

        paths = self.saver.save(fig, 'constraint_emax_sensitivity')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_constraint_beta_sensitivity(self, figsize=(10, 7)):
        """
        约束灵敏度 - β_env (环境)
        """
        from task2_2 import EducationDecisionParams, EducationDecisionModel
        
        fig, ax = plt.subplots(figsize=figsize)
        
        p = self.model.params
        original_params = p.constraint_params.copy()
        
        beta_range = np.linspace(0.10, 0.50, 15)
        ai_credits_list = []
        scores_list = []
        
        for beta in beta_range:
            p.constraint_params['beta_env'] = beta
            from task2_2 import EducationDecisionModel
            model_temp = EducationDecisionModel(p)
            result = model_temp.curriculum_optimization_sa()
            ai_credits_list.append(result['optimal_curriculum']['x_AI'])
            scores_list.append(result['optimal_score'])
        
        p.constraint_params = original_params
        
        ax.plot(beta_range * 100, ai_credits_list, 'o-', color=PlotStyleConfig.COLORS['accent'],
               linewidth=2.5, markersize=6, label='AI Credits')
        ax.set_xlabel('β_env (Green Cap) %', fontsize=12, fontweight='bold')
        ax.set_ylabel('Optimal AI Credits', fontsize=12, fontweight='bold', 
                     color=PlotStyleConfig.COLORS['accent'])
        ax.tick_params(axis='y', labelcolor=PlotStyleConfig.COLORS['accent'])
        
        ax_twin = ax.twinx()
        ax_twin.plot(beta_range * 100, scores_list, 's--', color=PlotStyleConfig.COLORS['secondary'],
                    linewidth=2, markersize=5, label='Net Score')
        ax_twin.set_ylabel('Net Score', fontsize=12, fontweight='bold', 
                          color=PlotStyleConfig.COLORS['secondary'])
        ax_twin.tick_params(axis='y', labelcolor=PlotStyleConfig.COLORS['secondary'])
        
        ax.axvline(original_params['beta_env'] * 100, color='gray', linestyle=':', 
                  linewidth=2, alpha=0.7, label=f'Current ({original_params["beta_env"]*100:.0f}%)')
        
        ax.legend(loc='upper left')
        ax.set_title(f'{self.school} - Green Cap (β_env) Sensitivity', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)

        paths = self.saver.save(fig, 'constraint_beta_sensitivity')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_constraint_gamma_sensitivity(self, figsize=(10, 7)):
        """
        约束灵敏度 - γ (安全)
        """
        from task2_2 import EducationDecisionParams, EducationDecisionModel
        
        fig, ax = plt.subplots(figsize=figsize)
        
        p = self.model.params
        original_params = p.constraint_params.copy()
        
        gamma_range = np.linspace(0.05, 0.80, 15)
        ethics_credits_list = []
        scores_list = []
        
        for gamma in gamma_range:
            p.constraint_params['gamma_safety'] = gamma
            from task2_2 import EducationDecisionModel
            model_temp = EducationDecisionModel(p)
            result = model_temp.curriculum_optimization_sa()
            ethics_credits_list.append(result['optimal_curriculum']['x_ethics'])
            scores_list.append(result['optimal_score'])
        
        p.constraint_params = original_params
        
        ax.plot(gamma_range, ethics_credits_list, 'o-', color=PlotStyleConfig.COLORS['gold'],
               linewidth=2.5, markersize=6, label='Ethics Credits')
        ax.set_xlabel('γ (Safety Ratio)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Optimal Ethics Credits', fontsize=12, fontweight='bold', 
                     color=PlotStyleConfig.COLORS['gold'])
        ax.tick_params(axis='y', labelcolor=PlotStyleConfig.COLORS['gold'])
        
        ax_twin = ax.twinx()
        ax_twin.plot(gamma_range, scores_list, 's--', color=PlotStyleConfig.COLORS['secondary'],
                    linewidth=2, markersize=5, label='Net Score')
        ax_twin.set_ylabel('Net Score', fontsize=12, fontweight='bold', 
                          color=PlotStyleConfig.COLORS['secondary'])
        ax_twin.tick_params(axis='y', labelcolor=PlotStyleConfig.COLORS['secondary'])
        
        ax.axvline(original_params['gamma_safety'], color='gray', linestyle=':', 
                  linewidth=2, alpha=0.7, label=f'Current ({original_params["gamma_safety"]:.2f})')
        
        ax.legend(loc='upper left')
        ax.set_title(f'{self.school} - Safety Ratio (γ) Sensitivity', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)

        paths = self.saver.save(fig, 'constraint_gamma_sensitivity')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    # ========================================
    # 8. 模型对比图表
    # ========================================

    def plot_baseline_vs_constrained_bar(self, baseline_results, constrained_results, figsize=(12, 7)):
        """
        模型对比 - 课程分配柱状图
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        keys = ['x_base', 'x_AI', 'x_ethics', 'x_proj']
        labels = ['Core', 'AI', 'Human', 'Cross']
        
        baseline_vals = [baseline_results['curriculum_optimization']['optimal_curriculum'][k] for k in keys]
        constrained_vals = [constrained_results['curriculum_optimization']['optimal_curriculum'][k] for k in keys]
        
        x = np.arange(len(keys))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (Utility-Max)',
                      color=PlotStyleConfig.COLORS['secondary'], edgecolor='black', alpha=0.8)
        bars2 = ax.bar(x + width/2, constrained_vals, width, label='Red-Line (Constrained)',
                      color=PlotStyleConfig.COLORS['accent'], edgecolor='black', alpha=0.8)
        
        # 添加标签
        for bar in bars1:
            ax.annotate(f'{bar.get_height():.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        for bar in bars2:
            ax.annotate(f'{bar.get_height():.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Credits', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.school} - Model Comparison: Baseline vs Constrained', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        paths = self.saver.save(fig, 'model_comparison_bar')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_utility_comparison_bar(self, baseline_results, constrained_results, figsize=(10, 7)):
        """
        模型对比 - 效用对比
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        models = ['Baseline\n(Utility-Max)', 'Red-Line\n(Constrained)']
        scores = [baseline_results['curriculum_optimization']['optimal_score'],
                  constrained_results['curriculum_optimization']['optimal_score']]
        colors = [PlotStyleConfig.COLORS['secondary'], PlotStyleConfig.COLORS['accent']]
        
        bars = ax.bar(models, scores, color=colors, edgecolor='black', width=0.5, alpha=0.9)
        
        for bar, score in zip(bars, scores):
            ax.annotate(f'{score:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # 差异标注
        diff = scores[1] - scores[0]
        diff_pct = (diff / abs(scores[0])) * 100 if scores[0] != 0 else 0
        ax.annotate(f'Δ = {diff:.3f} ({diff_pct:+.1f}%)', 
                   xy=(0.5, max(scores) + 0.1), ha='center', fontsize=12,
                   color=PlotStyleConfig.COLORS['danger'] if diff < 0 else PlotStyleConfig.COLORS['accent'],
                   fontweight='bold')
        
        ax.set_ylabel('Net Objective Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.school} - Utility Comparison', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(scores) * 1.2)

        paths = self.saver.save(fig, 'utility_comparison_bar')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    # ========================================
    # 9. 综合图表 (跨学校)
    # ========================================

    def plot_schools_pressure_index(self, all_results, figsize=(10, 7)):
        """
        跨学校 - 压力指数对比
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        schools = list(all_results.keys())
        school_colors = [PlotStyleConfig.get_school_color(s) for s in schools]
        pressure_indices = [all_results[s]['enrollment_response']['pressure_index'] for s in schools]
        
        bars = ax.bar(schools, pressure_indices, color=school_colors, alpha=0.85, 
                     edgecolor='white', linewidth=2, width=0.6)
        
        for bar, val in zip(bars, pressure_indices):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.axhline(0, color='gray', linewidth=1)
        ax.set_ylabel('Pressure Index (Γ)', fontsize=12, fontweight='bold')
        ax.set_title('University Comparison - Enrollment Pressure Index', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)

        saver = FigureSaver('./figures/task2_2_split')
        paths = saver.save(fig, 'comparison_pressure_index')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_schools_adjustment(self, all_results, figsize=(10, 7)):
        """
        跨学校 - 招生调整量对比
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        schools = list(all_results.keys())
        school_colors = [PlotStyleConfig.get_school_color(s) for s in schools]
        adjustments = [all_results[s]['enrollment_response']['adjustment'] for s in schools]
        
        bars = ax.bar(schools, adjustments, color=school_colors, alpha=0.85, 
                     edgecolor='white', linewidth=2, width=0.6)
        
        for bar, val in zip(bars, adjustments):
            color = PlotStyleConfig.COLORS['accent'] if val > 0 else PlotStyleConfig.COLORS['danger']
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:+.0f}', 
                   ha='center', va='bottom' if val > 0 else 'top', 
                   fontsize=12, fontweight='bold', color=color)
        
        ax.axhline(0, color='gray', linewidth=1)
        ax.set_ylabel('Enrollment Adjustment (ΔA)', fontsize=12, fontweight='bold')
        ax.set_title('University Comparison - Recommended Adjustment', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)

        saver = FigureSaver('./figures/task2_2_split')
        paths = saver.save(fig, 'comparison_adjustment')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_schools_ai_integration(self, all_results, figsize=(10, 7)):
        """
        跨学校 - AI课程占比对比
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        schools = list(all_results.keys())
        school_colors = [PlotStyleConfig.get_school_color(s) for s in schools]
        ai_credits = [all_results[s]['curriculum_optimization']['optimal_curriculum']['x_AI'] for s in schools]
        percentages = [a / 120 * 100 for a in ai_credits]
        
        bars = ax.bar(schools, percentages, color=school_colors, alpha=0.85, 
                     edgecolor='white', linewidth=2, width=0.6)
        
        for bar, val in zip(bars, percentages):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%', 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('AI Curriculum (%)', fontsize=12, fontweight='bold')
        ax.set_title('University Comparison - AI Integration Level', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(percentages) * 1.2)

        saver = FigureSaver('./figures/task2_2_split')
        paths = saver.save(fig, 'comparison_ai_integration')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_schools_optimization_score(self, all_results, figsize=(10, 7)):
        """
        跨学校 - 优化得分对比
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        schools = list(all_results.keys())
        school_colors = [PlotStyleConfig.get_school_color(s) for s in schools]
        scores = [all_results[s]['curriculum_optimization']['optimal_score'] for s in schools]
        
        bars = ax.bar(schools, scores, color=school_colors, alpha=0.85, 
                     edgecolor='white', linewidth=2, width=0.6)
        
        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}', 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Optimization Score', fontsize=12, fontweight='bold')
        ax.set_title('University Comparison - Optimization Objective Score', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)

        saver = FigureSaver('./figures/task2_2_split')
        paths = saver.save(fig, 'comparison_optimization_score')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_stacked_curriculum(self, all_results, figsize=(14, 8)):
        """
        跨学校 - 堆积柱状图：优化前后课程结构
        """
        from task2_2 import EducationDecisionParams
        
        fig, ax = plt.subplots(figsize=figsize)
        
        schools = ['CMU', 'CCAD', 'CIA']
        course_types = ['x_base', 'x_AI', 'x_ethics', 'x_proj']
        display_names = ['Core', 'AI', 'Human', 'Cross']
        colors = [PlotStyleConfig.COURSE_COLORS[k] for k in course_types]
        
        # 准备数据
        x_positions = []
        x_labels = []
        bar_width = 0.28
        gap = 0.12
        group_spacing = 0.4
        current_x = 0
        
        plot_data = {ctype: [] for ctype in course_types}
        
        for school in schools:
            if school not in all_results:
                continue
            
            init_params = EducationDecisionParams(school_name=school)
            init_curr = init_params.current_curriculum
            init_total = init_params.total_credits
            
            opt_curr = all_results[school]['curriculum_optimization']['optimal_curriculum']
            opt_total = sum(opt_curr.values())
            
            x_positions.extend([current_x, current_x + bar_width + gap])
            x_labels.extend([f'{school}\nInitial', f'{school}\nOptimized'])
            
            for ctype in course_types:
                plot_data[ctype].append(init_curr[ctype] / init_total * 100)
                plot_data[ctype].append(opt_curr[ctype] / opt_total * 100)
            
            current_x += (2 * bar_width + gap + group_spacing)

        # 绘制堆积图
        bottoms = [0] * len(x_positions)
        
        for i, ctype in enumerate(course_types):
            values = plot_data[ctype]
            bars = ax.bar(x_positions, values, bottom=bottoms, width=bar_width, 
                         label=display_names[i], color=colors[i], 
                         edgecolor='white', linewidth=1, alpha=0.9)
            
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val >= 8:
                    h = bar.get_height()
                    cx = bar.get_x() + bar.get_width()/2
                    cy = bar.get_y() + h/2
                    ax.text(cx, cy, f'{val:.0f}%', ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='#333333')
            
            bottoms = [b + v for b, v in zip(bottoms, values)]

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold')
        ax.set_ylabel('Percentage of Total Credits (%)', fontweight='bold', fontsize=12)
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}%'))
        
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, 
                 frameon=False, fontsize=11)
        ax.set_title('Curriculum Structure: Initial vs Optimized (All Schools)', 
                    fontsize=16, fontweight='bold', pad=30)

        saver = FigureSaver('./figures/task2_2_split')
        paths = saver.save(fig, 'comparison_stacked_curriculum')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    # ========================================
    # 10. AHP分析图表
    # ========================================

    def plot_ahp_radar(self, figsize=(11, 9)):
        """
        AHP分析 - 雷达图
        """
        from task2_2 import get_ahp_calculator
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        ahp = get_ahp_calculator()
        radar_data = ahp.get_radar_data()
        
        criteria = ['Strategic\nScalability\n(W=0.4)', 
                   'Physical\nIndependence\n(W=0.4)', 
                   'Service\nElasticity\n(W=0.2)']
        
        num_criteria = len(criteria)
        angles = np.linspace(0, 2 * np.pi, num_criteria, endpoint=False).tolist()
        angles += angles[:1]
        
        school_styles = {
            'CMU': {'color': '#C41E3A', 'marker': 'o', 'linestyle': '-'},
            'CCAD': {'color': '#FF6B35', 'marker': 's', 'linestyle': '--'},
            'CIA': {'color': '#1E3A5F', 'marker': '^', 'linestyle': '-.'}
        }
        
        for school, scores in radar_data.items():
            values = scores + scores[:1]
            style = school_styles.get(school, {'color': '#7f7f7f', 'marker': 'o', 'linestyle': '-'})
            
            ax.fill(angles, values, alpha=0.2, color=style['color'])
            ax.plot(angles, values, style['linestyle'], linewidth=3, 
                   color=style['color'], markersize=12, marker=style['marker'],
                   markerfacecolor='white', markeredgewidth=2.5,
                   label=f'{school} (λ={ahp.final_lambdas[school]:.3f})')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 0.85)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=10)
        ax.grid(True, alpha=0.6)
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=11)
        ax.set_title('AHP Analysis: λ Derivation across Criteria', 
                    fontsize=16, fontweight='bold', pad=30)

        saver = FigureSaver('./figures/task2_2_split')
        paths = saver.save(fig, 'ahp_radar')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_ahp_bar_chart(self, figsize=(12, 7)):
        """
        AHP分析 - λ值柱状图
        """
        from task2_2 import get_ahp_calculator
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ahp = get_ahp_calculator()
        schools = list(ahp.final_lambdas.keys())
        lambdas = list(ahp.final_lambdas.values())
        school_colors = [PlotStyleConfig.get_school_color(s) for s in schools]
        
        bars = ax.bar(schools, lambdas, color=school_colors, alpha=0.85, 
                     edgecolor='white', linewidth=2, width=0.6)
        
        for bar, val in zip(bars, lambdas):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, 
                   f'{val:.4f}\n({val*100:.1f}%)', 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Administrative Coefficient (λ)', fontsize=12, fontweight='bold')
        ax.set_title('AHP Result: Administrative Adjustment Capacity by University', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(lambdas) * 1.3)
        
        # 添加解释文字
        ax.text(0.02, 0.98, 
               "Higher λ = Greater capacity\nfor enrollment adjustment",
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.9))

        saver = FigureSaver('./figures/task2_2_split')
        paths = saver.save(fig, 'ahp_lambda_bar')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    def plot_career_similarity_matrix(self, figsize=(11, 9)):
        """
        职业相似度矩阵热力图
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        careers = list(self.model.params.CAREER_VECTORS.keys())
        display_careers = [self.model.params.CAREER_DISPLAY_NAMES.get(c, c) for c in careers]
        
        # 计算相似度矩阵
        similarity_matrix = np.zeros((len(careers), len(careers)))
        for i, origin in enumerate(careers):
            origin_vec = np.array(self.model.params.CAREER_VECTORS[origin])
            for j, target in enumerate(careers):
                target_vec = np.array(self.model.params.CAREER_VECTORS[target])
                if np.linalg.norm(origin_vec) == 0 or np.linalg.norm(target_vec) == 0:
                    similarity_matrix[i, j] = 0
                else:
                    similarity_matrix[i, j] = np.dot(origin_vec, target_vec) / (
                        np.linalg.norm(origin_vec) * np.linalg.norm(target_vec))

        im = ax.imshow(similarity_matrix, cmap='YlGnBu', aspect='auto', interpolation='nearest')
        
        for i in range(len(careers)):
            for j in range(len(careers)):
                val = similarity_matrix[i, j]
                text_color = "white" if val > 0.6 else "black"
                text_weight = "bold" if val > 0.8 else "normal"
                ax.text(j, i, f'{val:.2f}', ha="center", va="center", 
                       color=text_color, fontweight=text_weight, fontsize=10)

        ax.set_xticks(np.arange(len(careers)))
        ax.set_yticks(np.arange(len(careers)))
        ax.set_xticklabels(display_careers, rotation=35, ha='right', fontsize=10)
        ax.set_yticklabels(display_careers, fontsize=10)
        
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(len(careers)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(careers)+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Cosine Similarity', fontweight='bold')
        
        ax.set_title('Career Skill Similarity Matrix', 
                    fontsize=16, fontweight='bold', pad=15)

        saver = FigureSaver('./figures/task2_2_split')
        paths = saver.save(fig, 'career_similarity_matrix')
        print(f"    💾 Saved: {paths[0]}")
        return paths

    # ========================================
    # 主运行函数
    # ========================================

    def generate_all_individual_plots(self, baseline_results=None, all_results=None):
        """
        生成所有单独的图表
        """
        print(f"\n{'='*60}")
        print(f"  Generating Individual Plots for {self.school}")
        print(f"{'='*60}")
        
        # 1. 招生响应
        print("\n[1] Enrollment Response Plots:")
        self.plot_enrollment_bar_chart()
        self.plot_enrollment_flow_diagram()
        
        # 2. 课程优化
        print("\n[2] Curriculum Optimization Plots:")
        self.plot_curriculum_comparison_bar()
        self.plot_curriculum_pie_chart()
        self.plot_utility_breakdown()
        self.plot_ai_marginal_utility_curve()
        
        # 3. SA收敛
        print("\n[3] SA Convergence Plot:")
        self.plot_sa_convergence()
        
        # 4. 职业弹性
        print("\n[4] Career Elasticity Plots:")
        self.plot_career_elasticity_bar()
        
        # 生成技能雷达图（针对前3个目标职业）
        target_careers = list(self.results['career_elasticity']['similarities'].keys())[:3]
        for target in target_careers:
            self.plot_single_skill_radar(target)
        
        # 5. 帕累托前沿
        print("\n[5] Pareto Frontier Plot:")
        self.plot_pareto_frontier()
        
        # 6. 灵敏度分析
        print("\n[6] Sensitivity Analysis Plots:")
        self.plot_lambda_sensitivity()
        self.plot_weight_sensitivity()
        
        # 7. 约束分析
        print("\n[7] Constraint Analysis Plots:")
        self.plot_constraint_satisfaction_radar()
        
        # 8. 模型对比
        if baseline_results:
            print("\n[8] Model Comparison Plots:")
            self.plot_baseline_vs_constrained_bar(baseline_results, self.results)
            self.plot_utility_comparison_bar(baseline_results, self.results)
        
        print(f"\n{'='*60}")
        print(f"  ✓ All individual plots for {self.school} completed!")
        print(f"{'='*60}")


def generate_comparison_plots(all_results):
    """
    生成跨学校对比图表
    """
    print("\n" + "="*60)
    print("  Generating Cross-University Comparison Plots")
    print("="*60)
    
    from task2_2 import EducationDecisionParams, EducationDecisionModel
    
    # 创建一个临时的viz实例用于生成跨学校图表
    temp_params = EducationDecisionParams(school_name='CMU', enable_constraints=True)
    temp_model = EducationDecisionModel(temp_params)
    temp_results = temp_model.run_full_analysis(verbose=False)
    
    viz = SplitVisualization(temp_model, temp_results)
    
    print("\n[Comparison Plots]:")
    viz.plot_schools_pressure_index(all_results)
    viz.plot_schools_adjustment(all_results)
    viz.plot_schools_ai_integration(all_results)
    viz.plot_schools_optimization_score(all_results)
    viz.plot_stacked_curriculum(all_results)
    
    print("\n[AHP Analysis Plots]:")
    viz.plot_ahp_radar()
    viz.plot_ahp_bar_chart()
    
    print("\n[Career Analysis Plots]:")
    viz.plot_career_similarity_matrix()
    
    print("\n" + "="*60)
    print("  ✓ All comparison plots completed!")
    print("="*60)


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    from task2_2 import EducationDecisionParams, EducationDecisionModel
    
    print("\n" + "█"*70)
    print("█" + " "*10 + "AI教育决策模型 - 拆分可视化生成器" + " "*10 + "█")
    print("█" + " "*8 + "Generating Individual High-Quality Plots" + " "*9 + "█")
    print("█"*70 + "\n")
    
    # 创建输出目录
    os.makedirs('./figures/task2_2_split', exist_ok=True)
    
    schools = ['CMU', 'CCAD', 'CIA']
    all_results_baseline = {}
    all_results_constrained = {}
    
    for school in schools:
        print(f"\n{'='*70}")
        print(f"Processing {school}...")
        print(f"{'='*70}")
        
        # 基线模型
        params_baseline = EducationDecisionParams(school_name=school, enable_constraints=False)
        model_baseline = EducationDecisionModel(params_baseline)
        results_baseline = model_baseline.run_full_analysis(verbose=False)
        all_results_baseline[school] = results_baseline
        
        # 约束模型
        params_constrained = EducationDecisionParams(school_name=school, enable_constraints=True)
        model_constrained = EducationDecisionModel(params_constrained)
        results_constrained = model_constrained.run_full_analysis(verbose=False)
        all_results_constrained[school] = results_constrained
        
        # 生成该学校的所有单独图表
        viz = SplitVisualization(model_constrained, results_constrained)
        viz.generate_all_individual_plots(baseline_results=results_baseline)
    
    # 生成跨学校对比图表
    generate_comparison_plots(all_results_constrained)
    
    print("\n" + "█"*70)
    print("█" + " "*15 + "所有图表生成完成!" + " "*16 + "█")
    print("█" + " "*10 + "Saved to: ./figures/task2_2_split/" + " "*10 + "█")
    print("█"*70 + "\n")
