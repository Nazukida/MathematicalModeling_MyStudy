"""
Task 2.1 拆分可视化模块 - Split Visualization Module
=======================================================

将原有的多子图可视化拆分为独立的单张图片，并进行专业美化。
基于 task2_1.py 的 EducationDecisionVisualization 类拆分。

输出目录: ./figures/task2_1_split/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 从 task2_1 导入必要的类和函数
from task2_1 import (
    PlotStyleConfig,
    FigureSaver,
    EducationDecisionParams,
    EducationDecisionModel,
    AHPLambdaCalculator,
    get_ahp_calculator,
    get_ahp_lambdas
)


class SplitVisualization:
    """
    拆分可视化类 - 每个子图独立保存为单张图片
    美化风格统一，专业学术论文级别
    """

    def __init__(self, model: EducationDecisionModel, results: dict, 
                 save_dir='./figures/task2_1_split'):
        """
        初始化拆分可视化器
        
        Args:
            model: 教育决策模型实例
            results: 模型分析结果字典
            save_dir: 图片保存目录
        """
        self.model = model
        self.results = results
        self.school = model.params.school_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图风格
        PlotStyleConfig.setup_style('academic')
    
    def _save_figure(self, fig, filename, formats=['png', 'pdf']):
        """保存图片到指定目录"""
        paths = []
        for fmt in formats:
            path = os.path.join(self.save_dir, f"{self.school}_{filename}.{fmt}")
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            paths.append(path)
        plt.close(fig)
        return paths

    # ========== 招生响应分析 ==========
    def plot_enrollment_bar_chart(self, figsize=(10, 7)):
        """
        招生响应柱状图 - 独立图片
        展示当前供给、优化计划、市场需求对比
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        fig.suptitle(f'{self.school} - Enrollment Response Analysis',
                    fontsize=18, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        ax.set_title('Supply vs Demand Adjustment Model (Sub-model 1)', 
                    fontsize=12, style='italic', pad=10)

        r = self.results['enrollment_response']
        colors = [PlotStyleConfig.COLORS['primary'], 
                  PlotStyleConfig.COLORS['accent'], 
                  PlotStyleConfig.COLORS['secondary']]

        values = [self.model.params.current_graduates, 
                  r['recommended_graduates'], 
                  self.model.params.demand_2030]
        labels = ['Current Supply\n(S_t)', 'Optimized Plan\n(A_t)', 'Market Demand\n(D_t)']
        
        bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5, zorder=3,
                     edgecolor='white', linewidth=2)
        
        ax.set_ylabel('Number of Graduates', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, zorder=0, linestyle='--')

        # 添加数值标签和变化率
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=13, 
                   fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
            
            if i > 0:
                change = (val - values[0]) / values[0] * 100
                symbol = '▲' if change > 0 else '▼'
                ax.text(bar.get_x() + bar.get_width()/2, height - (height*0.1),
                       f'{symbol} {abs(change):.1f}%', ha='center', va='center', 
                       fontsize=11, fontweight='bold', color='white')

        # 添加连接箭头
        start_x = bars[0].get_x() + bars[0].get_width()/2
        end_x = bars[1].get_x() + bars[1].get_width()/2
        arrow_color = PlotStyleConfig.COLORS['gold']
        ax.annotate('', xy=(end_x, values[1]), xytext=(start_x, values[0]),
                   arrowprops=dict(arrowstyle="->", color=arrow_color, lw=3, 
                                   connectionstyle="arc3,rad=-0.2"))
        
        # 信息文本框
        info_text = (f"Pressure Index (P) = {r['pressure_index']:.3f}\n"
                    f"Adjustment (ΔA) = {r['adjustment']:+.1f}\n"
                    f"Admin Capacity (λ) = {self.model.params.lambda_admin:.3f}")
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11, 
               verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor=PlotStyleConfig.COLORS['primary'], linewidth=2, alpha=0.9))

        plt.tight_layout()
        paths = self._save_figure(fig, 'enrollment_bar_chart')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    def plot_enrollment_flow_diagram(self, figsize=(12, 6)):
        """
        招生流程图 - 使用箭头展示调整流向
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        r = self.results['enrollment_response']
        
        # 三个节点位置
        positions = {'Current': 0.2, 'Optimized': 0.5, 'Target': 0.8}
        values = {
            'Current': self.model.params.current_graduates,
            'Optimized': r['recommended_graduates'],
            'Target': self.model.params.demand_2030
        }
        colors = {
            'Current': PlotStyleConfig.COLORS['neutral'],
            'Optimized': PlotStyleConfig.COLORS['accent'],
            'Target': PlotStyleConfig.COLORS['primary']
        }
        
        # 绘制圆形节点
        for name, x_pos in positions.items():
            circle = plt.Circle((x_pos, 0.5), 0.12, color=colors[name], 
                               alpha=0.8, zorder=3)
            ax.add_patch(circle)
            ax.text(x_pos, 0.5, f'{values[name]:.0f}', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white', zorder=4)
            ax.text(x_pos, 0.25, name, ha='center', fontsize=12, fontweight='bold')
        
        # 绘制箭头
        ax.annotate('', xy=(0.35, 0.5), xytext=(0.32, 0.5),
                   arrowprops=dict(arrowstyle='->', color=PlotStyleConfig.COLORS['gold'], 
                                   lw=4, mutation_scale=20))
        ax.annotate('', xy=(0.65, 0.5), xytext=(0.62, 0.5),
                   arrowprops=dict(arrowstyle='->', color=PlotStyleConfig.COLORS['gold'], 
                                   lw=4, mutation_scale=20))
        
        # 标注调整量
        adjustment = r['adjustment']
        ax.text(0.35, 0.7, f'Δ = {adjustment:+.0f}', ha='center', fontsize=11,
               fontweight='bold', color=PlotStyleConfig.COLORS['gold'])
        
        gap = values['Target'] - values['Optimized']
        ax.text(0.65, 0.7, f'Gap = {gap:.0f}', ha='center', fontsize=11,
               fontweight='bold', color=PlotStyleConfig.COLORS['danger'])
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        fig.suptitle(f'{self.school} - Enrollment Adjustment Flow',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        paths = self._save_figure(fig, 'enrollment_flow_diagram')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    # ========== 课程优化分析 - 拆分4张图 ==========
    def plot_curriculum_comparison_bar(self, figsize=(10, 7)):
        """
        课程结构优化对比柱状图 - 当前 vs 优化后
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        r = self.results['curriculum_optimization']
        keys = ['x_base', 'x_AI', 'x_ethics', 'x_proj']
        current = [self.model.params.current_curriculum.get(k, 0) for k in keys]
        optimal = [r['optimal_curriculum'].get(k, 0) for k in keys]
        labels = ['Base', 'AI', 'Ethics', 'Project']
        
        x = np.arange(len(labels))
        width = 0.35

        bar1 = ax.bar(x - width/2, current, width, label='Current', 
                      color=PlotStyleConfig.COLORS['neutral'], alpha=0.7, 
                      edgecolor='white', linewidth=1)
        bar2 = ax.bar(x + width/2, optimal, width, label='Optimized', 
                      color=PlotStyleConfig.COLORS['primary'], alpha=0.9, 
                      edgecolor='white', linewidth=1)
        
        ax.set_title(f'{self.school} - Curriculum Structure Optimization', 
                    fontweight='bold', fontsize=14, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Credits Allocation', fontweight='bold', fontsize=12)
        ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=11)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # 标注AI学分变化
        ai_diff = optimal[1] - current[1]
        ax.annotate(f'{ai_diff:+.1f} Cr', 
                    xy=(x[1] + width/2, optimal[1]), 
                    xytext=(x[1] + width/2, optimal[1]+5),
                    ha='center', fontsize=11, fontweight='bold', 
                    color=PlotStyleConfig.COLORS['danger'],
                    arrowprops=dict(arrowstyle='->', 
                                   color=PlotStyleConfig.COLORS['danger']))
        
        # 添加数值标签
        for bars in [bar1, bar2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        paths = self._save_figure(fig, 'curriculum_comparison_bar')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    def plot_curriculum_pie_chart(self, figsize=(10, 8)):
        """
        优化后课程效用贡献饼图 (环形图)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        r = self.results['curriculum_optimization']
        p = self.model.params
        
        # 获取权重
        if p.school_name == 'CMU':
            base_w = {'x_base': 0.40, 'x_AI': 0.25, 'x_ethics': 0.10, 'x_proj': 0.25}
        elif p.school_name == 'CCAD':
            base_w = {'x_base': 0.35, 'x_AI': 0.15, 'x_proj': 0.40, 'x_ethics': 0.10}
        elif p.school_name == 'CIA':
            base_w = {'x_base': 0.45, 'x_AI': 0.10, 'x_proj': 0.35, 'x_ethics': 0.10}
        else:
            base_w = {'x_base': 0.3, 'x_AI': 0.3, 'x_proj': 0.3, 'x_ethics': 0.1}

        keys = ['x_base', 'x_AI', 'x_ethics', 'x_proj']
        sizes = [base_w.get(k, 0) * np.sqrt(r['optimal_curriculum'].get(k, 0)) * 10 for k in keys]
        labels = ['Base', 'AI', 'Ethics', 'Project']
        colors = PlotStyleConfig.get_palette(4)
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90, 
                                          pctdistance=0.80,
                                          wedgeprops=dict(width=0.4, edgecolor='white', 
                                                         linewidth=2))
        
        # 中心文字
        ax.text(0, 0, f"Score\n{r['optimal_score']:.2f}", ha='center', va='center', 
                fontsize=16, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        
        ax.set_title(f'{self.school} - Utility Contribution Breakdown', 
                    fontweight='bold', fontsize=14, pad=20)

        plt.tight_layout()
        paths = self._save_figure(fig, 'curriculum_pie_chart')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    def plot_utility_breakdown(self, figsize=(10, 7)):
        """
        效用分解条形图 - 各课程类型对效用的贡献
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        r = self.results['curriculum_optimization']
        p = self.model.params
        
        # 获取权重
        if p.school_name == 'CMU':
            base_w = {'x_base': 0.40, 'x_AI': 0.25, 'x_ethics': 0.10, 'x_proj': 0.25}
        elif p.school_name == 'CCAD':
            base_w = {'x_base': 0.35, 'x_AI': 0.15, 'x_proj': 0.40, 'x_ethics': 0.10}
        elif p.school_name == 'CIA':
            base_w = {'x_base': 0.45, 'x_AI': 0.10, 'x_proj': 0.35, 'x_ethics': 0.10}
        else:
            base_w = {'x_base': 0.3, 'x_AI': 0.3, 'x_proj': 0.3, 'x_ethics': 0.1}

        keys = ['x_base', 'x_AI', 'x_ethics', 'x_proj']
        labels = ['Base', 'AI', 'Ethics', 'Project']
        utilities = [base_w.get(k, 0) * np.sqrt(r['optimal_curriculum'].get(k, 0)) for k in keys]
        colors = PlotStyleConfig.get_palette(4)
        
        bars = ax.barh(labels, utilities, color=colors, alpha=0.85, 
                       edgecolor='white', linewidth=1.5)
        
        # 添加数值标签
        for bar, val in zip(bars, utilities):
            ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Utility Contribution', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.school} - Course Type Utility Breakdown', 
                    fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # 总效用标注
        total_utility = sum(utilities)
        ax.text(0.95, 0.05, f'Total Utility: {total_utility:.3f}', 
               transform=ax.transAxes, ha='right', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=PlotStyleConfig.COLORS['gold'], 
                        alpha=0.8))

        plt.tight_layout()
        paths = self._save_figure(fig, 'utility_breakdown')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    def plot_ai_marginal_utility_curve(self, figsize=(10, 7)):
        """
        AI学分边际效用曲线
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        r = self.results['curriculum_optimization']
        p = self.model.params
        
        # 获取AI权重
        if p.school_name == 'CMU':
            ai_weight = 0.25
        elif p.school_name == 'CCAD':
            ai_weight = 0.15
        elif p.school_name == 'CIA':
            ai_weight = 0.10
        else:
            ai_weight = 0.3
        
        x_AI_range = np.linspace(0, 50, 100)
        # 边际效用 = w * 1/(2*sqrt(x)) (sqrt的导数)
        marginal_utility = ai_weight * 1 / (2 * np.sqrt(x_AI_range + 0.1))
        total_utility = ai_weight * np.sqrt(x_AI_range)
        
        # 双Y轴
        ax2 = ax.twinx()
        
        line1, = ax.plot(x_AI_range, marginal_utility, 
                        color=PlotStyleConfig.COLORS['danger'], 
                        linewidth=2.5, label='Marginal Utility')
        line2, = ax2.plot(x_AI_range, total_utility, 
                         color=PlotStyleConfig.COLORS['primary'], 
                         linewidth=2.5, linestyle='--', label='Total Utility')
        
        # 标记最优点
        opt_ai = r['optimal_curriculum'].get('x_AI', 0)
        opt_marginal = ai_weight * 1 / (2 * np.sqrt(opt_ai + 0.1))
        opt_total = ai_weight * np.sqrt(opt_ai)
        
        ax.axvline(x=opt_ai, color=PlotStyleConfig.COLORS['gold'], 
                   linestyle=':', alpha=0.8, linewidth=2)
        ax.scatter([opt_ai], [opt_marginal], s=120, 
                  color=PlotStyleConfig.COLORS['danger'], 
                  zorder=5, edgecolors='white', linewidth=2)
        ax2.scatter([opt_ai], [opt_total], s=120, 
                   color=PlotStyleConfig.COLORS['primary'], 
                   zorder=5, edgecolors='white', linewidth=2)
        
        ax.set_xlabel('AI Credits', fontsize=12, fontweight='bold')
        ax.set_ylabel('Marginal Utility', fontsize=12, fontweight='bold', 
                     color=PlotStyleConfig.COLORS['danger'])
        ax2.set_ylabel('Total Utility', fontsize=12, fontweight='bold', 
                      color=PlotStyleConfig.COLORS['primary'])
        
        ax.set_title(f'{self.school} - AI Credits Utility Analysis\n(Diminishing Returns)', 
                    fontsize=14, fontweight='bold', pad=10)
        
        # 合并图例
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=10)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 标注最优点
        ax.annotate(f'Optimal: {opt_ai:.1f} Cr', 
                   xy=(opt_ai, opt_marginal),
                   xytext=(opt_ai + 5, opt_marginal + 0.02),
                   fontsize=11, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='gray'))

        plt.tight_layout()
        paths = self._save_figure(fig, 'ai_marginal_utility_curve')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    # ========== SA收敛过程 ==========
    def plot_sa_convergence(self, figsize=(12, 7)):
        """
        模拟退火收敛过程图
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        history = self.results['curriculum_optimization']['iteration_history']
        iterations = np.arange(len(history))
        
        # 主曲线
        ax.fill_between(iterations, min(history)*0.95, history, 
                       alpha=0.3, color=PlotStyleConfig.COLORS['primary'])
        ax.plot(iterations, history, color=PlotStyleConfig.COLORS['primary'], 
               linewidth=2.5, label='Best Score', zorder=3)
        
        # 标记起点和终点
        ax.scatter([0], [history[0]], s=150, color=PlotStyleConfig.COLORS['danger'], 
                  zorder=5, edgecolors='white', linewidths=2, 
                  label=f'Start: {history[0]:.3f}')
        ax.scatter([len(history)-1], [history[-1]], s=200, 
                  color=PlotStyleConfig.COLORS['accent'], zorder=5,
                  edgecolors='white', linewidths=2, 
                  label=f'Final: {history[-1]:.3f}', marker='*')
        
        # 添加最终最优线
        ax.axhline(y=history[-1], color=PlotStyleConfig.COLORS['accent'], 
                  linestyle='--', linewidth=2, alpha=0.7)
        
        # 标注改进率
        improvement = (history[-1] - history[0]) / abs(history[0]) * 100 if history[0] != 0 else 0
        ax.annotate(f'Improvement: {improvement:+.1f}%', 
                   xy=(len(history)*0.7, history[-1]), 
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor=PlotStyleConfig.COLORS['gold'], alpha=0.8))
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Objective Score', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.4, linestyle='--')
        
        # 调整Y轴范围
        y_min, y_max = min(history), max(history)
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        
        fig.suptitle(f'{self.school} - Simulated Annealing Optimization',
                    fontsize=16, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        ax.set_title('Convergence Process of Curriculum Optimization', 
                    fontsize=12, style='italic', pad=10)

        plt.tight_layout()
        paths = self._save_figure(fig, 'sa_convergence')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    # ========== 职业弹性分析 ==========
    def plot_career_elasticity_bar(self, figsize=(12, 7)):
        """
        职业弹性水平柱状图
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        r = self.results['career_elasticity']
        careers = list(r['similarities'].keys())
        similarities = list(r['similarities'].values())
        display_careers = [self.model.params.CAREER_DISPLAY_NAMES.get(c, c) for c in careers]
        
        # 渐变颜色
        n = len(careers)
        colors = PlotStyleConfig.get_palette(n)
        
        # 水平条形图
        y_pos = np.arange(len(display_careers))
        bars = ax.barh(y_pos, similarities, color=colors, alpha=0.85, 
                       edgecolor='white', linewidth=1.5, height=0.7)
        
        # 数值标签
        for bar, sim in zip(bars, similarities):
            width = bar.get_width()
            label_x = width + 0.02 if width < 0.8 else width - 0.08
            color = 'black' if width < 0.8 else 'white'
            ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{sim:.3f}',
                   ha='left' if width < 0.8 else 'right', va='center', 
                   fontsize=11, fontweight='bold', color=color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_careers, fontsize=11)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel('Cosine Similarity Score', fontsize=12, fontweight='bold')
        
        # 阈值线
        ax.axvline(x=0.9, color=PlotStyleConfig.COLORS['accent'], 
                   linestyle='--', linewidth=2, alpha=0.8, label='High (>0.9)')
        ax.axvline(x=0.7, color=PlotStyleConfig.COLORS['gold'], 
                   linestyle='--', linewidth=2, alpha=0.8, label='Medium (>0.7)')
        ax.axvline(x=0.5, color=PlotStyleConfig.COLORS['danger'], 
                   linestyle='--', linewidth=2, alpha=0.8, label='Low (<0.5)')
        
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        
        # 背景区域
        ax.axvspan(0.9, 1.1, alpha=0.1, color=PlotStyleConfig.COLORS['accent'])
        ax.axvspan(0.7, 0.9, alpha=0.1, color=PlotStyleConfig.COLORS['gold'])
        ax.axvspan(0, 0.5, alpha=0.1, color=PlotStyleConfig.COLORS['danger'])
        
        fig.suptitle(f'{self.school} - Career Path Elasticity Analysis',
                    fontsize=16, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        ax.set_title('Similarity to Origin Career (Higher = Easier Transition)', 
                    fontsize=12, style='italic', pad=10)
        
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()

        paths = self._save_figure(fig, 'career_elasticity_bar')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    # ========== 技能雷达图 - 拆分为多张独立图 ==========
    def plot_single_skill_radar(self, target_career, figsize=(9, 8)):
        """
        单个技能雷达图 - 原始职业 vs 目标职业
        """
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#FAFBFC')
        
        # 获取当前职业
        career = ('software_engineer' if self.model.params.school_name == 'CMU' 
                  else ('graphic_designer' if self.model.params.school_name == 'CCAD' 
                        else 'chef'))
        origin_vec = np.array(self.model.params.CAREER_VECTORS[career])
        target_vec = np.array(self.model.params.CAREER_VECTORS[target_career])
        
        features = ['Analytical', 'Creative', 'Technical', 'Interpersonal', 'Physical']
        
        # 计算角度
        num_features = len(features)
        angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
        angles += angles[:1]
        
        # 准备数据
        origin_plot = origin_vec.tolist() + origin_vec.tolist()[:1]
        target_plot = target_vec.tolist() + target_vec.tolist()[:1]
        
        # 颜色
        origin_color = PlotStyleConfig.COLORS['primary']
        target_color = PlotStyleConfig.COLORS['secondary']
        
        display_career = self.model.params.CAREER_DISPLAY_NAMES.get(career, career)
        display_target = self.model.params.CAREER_DISPLAY_NAMES.get(target_career, target_career)
        
        # 绘制原始职业
        ax.fill(angles, origin_plot, alpha=0.25, color=origin_color, zorder=2)
        ax.plot(angles, origin_plot, 'o-', linewidth=2.5, color=origin_color, 
               markersize=8, markerfacecolor='white', markeredgewidth=2, 
               label=f'Origin: {display_career}', zorder=3)
        
        # 绘制目标职业
        ax.fill(angles, target_plot, alpha=0.25, color=target_color, zorder=2)
        ax.plot(angles, target_plot, 's-', linewidth=2.5, color=target_color, 
               markersize=8, markerfacecolor='white', markeredgewidth=2,
               label=f'Target: {display_target}', zorder=3)
        
        # 设置刻度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, fontsize=11, fontweight='bold', 
                          color=PlotStyleConfig.COLORS['dark'])
        
        # 设置径向范围
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, 
                          color=PlotStyleConfig.COLORS['neutral'])
        
        # 网格
        ax.grid(True, color=PlotStyleConfig.COLORS['grid'], alpha=0.6, linewidth=1)
        
        # 相似度
        similarity = self.results['career_elasticity']['similarities'].get(target_career, 0)
        
        fig.suptitle(f'{self.school} - Skill Fingerprint Comparison',
                    fontsize=14, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        ax.set_title(f'{display_target}\nSimilarity: {similarity:.3f}', 
                    fontsize=12, fontweight='bold', pad=15)
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9, framealpha=0.9)

        plt.tight_layout()
        
        # 文件名使用目标职业
        safe_name = target_career.replace(' ', '_').lower()
        paths = self._save_figure(fig, f'skill_radar_{safe_name}')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    def plot_all_skill_radars(self):
        """
        生成所有技能雷达图
        """
        target_careers = list(self.results['career_elasticity']['similarities'].keys())[:5]
        for target in target_careers:
            self.plot_single_skill_radar(target)

    # ========== 帕累托前沿 ==========
    def plot_pareto_frontier(self, figsize=(12, 8)):
        """
        帕累托前沿图 - AI vs Base 资源竞争
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        p = self.model.params
        
        # 获取权重
        if p.school_name == 'CMU':
            base_w = {'x_base': 0.40, 'x_AI': 0.25}
        elif p.school_name == 'CCAD':
            base_w = {'x_base': 0.35, 'x_AI': 0.15}
        elif p.school_name == 'CIA':
            base_w = {'x_base': 0.45, 'x_AI': 0.10}
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
                points.append((ai_utility, base_utility))

        points = np.array(points)
        ai_utilities = points[:, 0]
        base_utilities = points[:, 1]

        # 散点图
        scatter = ax.scatter(ai_utilities, base_utilities, c=ai_utilities, 
                            cmap='viridis', alpha=0.7, s=50, 
                            edgecolors='k', linewidth=0.5)
        
        # 计算帕累托前沿
        def is_dominated(p1, p2):
            return p1[0] <= p2[0] and p1[1] <= p2[1] and (p1[0] < p2[0] or p1[1] < p2[1])
        
        pareto_front = []
        for i, p1 in enumerate(points):
            dominated = False
            for j, p2 in enumerate(points):
                if i != j and is_dominated(p1, p2):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(p1)
        
        pareto_front = np.array(sorted(pareto_front, key=lambda x: x[0]))
        
        # 绘制帕累托前沿
        if len(pareto_front) > 1:
            ax.plot(pareto_front[:, 0], pareto_front[:, 1], 'r-', 
                   linewidth=3, alpha=0.8, label='Pareto Front')
            ax.fill_between(pareto_front[:, 0], pareto_front[:, 1], 
                           alpha=0.1, color='red')
        
        # 标记最优点
        r = self.results['curriculum_optimization']
        opt_ai = r['optimal_curriculum'].get('x_AI', 0)
        opt_base = r['optimal_curriculum'].get('x_base', 0)
        opt_ai_utility = base_w.get('x_AI', 0) * np.sqrt(opt_ai)
        opt_base_utility = base_w.get('x_base', 0) * np.sqrt(opt_base)
        
        ax.scatter(opt_ai_utility, opt_base_utility, 
                  color=PlotStyleConfig.COLORS['gold'], s=200, marker='*', 
                  edgecolors='black', linewidth=2, label='Optimal Solution', zorder=10)
        ax.annotate(f'Optimal\n({opt_ai:.0f} AI, {opt_base:.0f} Base)', 
                   (opt_ai_utility, opt_base_utility), 
                   xytext=(20, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor=PlotStyleConfig.COLORS['gold'], alpha=0.8),
                   fontsize=10, ha='center')

        ax.set_xlabel('AI Skill Utility (Benefit)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Base Skill Utility (Benefit)', fontsize=14, fontweight='bold')
        
        fig.suptitle(f'{self.school} - Resource Competition Analysis',
                    fontsize=16, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        ax.set_title('AI vs Base Skills Trade-off (Pareto Frontier)', 
                    fontsize=12, style='italic', pad=10)
        
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('AI Utility Intensity', fontsize=12)
        
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        paths = self._save_figure(fig, 'pareto_frontier')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    # ========== 灵敏度分析 - 拆分2张图 ==========
    def plot_sensitivity_lambda(self, figsize=(10, 7)):
        """
        Lambda灵敏度分析图
        """
        if 'sensitivity_analysis' not in self.results:
            print("    ⚠️ No sensitivity analysis results found.")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        data = self.results['sensitivity_analysis']['lambda_sensitivity']
        x = data['range']
        y = data['adjustments']
        
        ax.plot(x, y, color=PlotStyleConfig.COLORS['primary'], linewidth=2.5, 
               marker='o', markersize=4, label='Adjustment Amount')
        
        # 标记当前Lambda
        current_lambda = self.model.params.lambda_admin
        current_adj = self.results['enrollment_response']['adjustment']
        ax.scatter(current_lambda, current_adj, marker='*', s=300, 
                  color=PlotStyleConfig.COLORS['gold'], 
                  label=f'Current λ={current_lambda:.3f}', zorder=10,
                  edgecolors='black', linewidths=1)
        
        ax.axvline(x=current_lambda, color=PlotStyleConfig.COLORS['gold'], 
                   linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Administrative Coefficient (λ)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Enrollment Adjustment (ΔE)', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=10)
        
        fig.suptitle(f'{self.school} - Lambda Sensitivity Analysis',
                    fontsize=14, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        ax.set_title('Macro Sensitivity: Enrollment Adjustment vs λ', 
                    fontsize=12, style='italic', pad=10)

        plt.tight_layout()
        paths = self._save_figure(fig, 'sensitivity_lambda')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    def plot_sensitivity_weight(self, figsize=(10, 7)):
        """
        AI权重灵敏度分析图
        """
        if 'sensitivity_analysis' not in self.results:
            print("    ⚠️ No sensitivity analysis results found.")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        data = self.results['sensitivity_analysis']['weight_sensitivity']
        x = data['range']
        y_ai = data['ai_credits']
        y_base = data['base_credits']
        
        ax.plot(x, y_ai, color=PlotStyleConfig.COLORS['secondary'], linewidth=2.5, 
               marker='s', markersize=4, label='AI Credits')
        ax.plot(x, y_base, color=PlotStyleConfig.COLORS['neutral'], linewidth=2, 
               linestyle='--', marker='o', markersize=3, label='Base Credits')
        
        ax.fill_between(x, y_ai, alpha=0.2, color=PlotStyleConfig.COLORS['secondary'])
        
        ax.set_xlabel('Weight of AI Skill (w_AI)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Optimized Credits', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=10)
        
        fig.suptitle(f'{self.school} - AI Weight Sensitivity Analysis',
                    fontsize=14, fontweight='bold', color=PlotStyleConfig.COLORS['dark'])
        ax.set_title('Micro Sensitivity: Credit Allocation vs AI Weight', 
                    fontsize=12, style='italic', pad=10)

        plt.tight_layout()
        paths = self._save_figure(fig, 'sensitivity_weight')
        print(f"    ✅ Saved: {paths[0]}")
        return paths


# ========== 跨学校比较图 - 独立类 ==========
class CrossSchoolVisualization:
    """
    跨学校比较可视化类
    """
    
    def __init__(self, all_results, save_dir='./figures/task2_1_split'):
        self.all_results = all_results
        self.save_dir = save_dir
        self.schools = list(all_results.keys())
        os.makedirs(save_dir, exist_ok=True)
        PlotStyleConfig.setup_style('academic')
    
    def _save_figure(self, fig, filename, formats=['png', 'pdf']):
        """保存图片"""
        paths = []
        for fmt in formats:
            path = os.path.join(self.save_dir, f"comparison_{filename}.{fmt}")
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            paths.append(path)
        plt.close(fig)
        return paths

    def plot_pressure_index_comparison(self, figsize=(10, 6)):
        """
        压力指数对比图
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        school_colors = [PlotStyleConfig.get_school_color(s) for s in self.schools]
        pressure_indices = [self.all_results[s]['enrollment_response']['pressure_index'] 
                           for s in self.schools]
        
        bars = ax.bar(self.schools, pressure_indices, color=school_colors, 
                     alpha=0.85, edgecolor='white', linewidth=2)
        
        for bar, val in zip(bars, pressure_indices):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.axhline(0, color='gray', linewidth=1)
        ax.set_ylabel('Pressure Index (P)', fontsize=12, fontweight='bold')
        ax.set_title('Enrollment Pressure Index Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        paths = self._save_figure(fig, 'pressure_index')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    def plot_adjustment_comparison(self, figsize=(10, 6)):
        """
        调整量对比图
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        school_colors = [PlotStyleConfig.get_school_color(s) for s in self.schools]
        adjustments = [self.all_results[s]['enrollment_response']['adjustment'] 
                      for s in self.schools]
        
        bars = ax.bar(self.schools, adjustments, color=school_colors, 
                     alpha=0.85, edgecolor='white', linewidth=2)
        
        for bar, val in zip(bars, adjustments):
            color = PlotStyleConfig.COLORS['accent'] if val > 0 else PlotStyleConfig.COLORS['danger']
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:+.0f}', 
                   ha='center', va='bottom' if val > 0 else 'top', 
                   fontsize=12, fontweight='bold', color=color)
        
        ax.axhline(0, color='gray', linewidth=1)
        ax.set_ylabel('Enrollment Adjustment (ΔA)', fontsize=12, fontweight='bold')
        ax.set_title('Recommended Enrollment Adjustment Comparison', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        paths = self._save_figure(fig, 'adjustment')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    def plot_ai_integration_comparison(self, figsize=(10, 6)):
        """
        AI课程集成度对比
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        school_colors = [PlotStyleConfig.get_school_color(s) for s in self.schools]
        ai_credits = [self.all_results[s]['curriculum_optimization']['optimal_curriculum']['x_AI'] 
                     for s in self.schools]
        percentages = [a/120*100 for a in ai_credits]
        
        bars = ax.bar(self.schools, percentages, color=school_colors, 
                     alpha=0.85, edgecolor='white', linewidth=2)
        
        for bar, val, cred in zip(bars, percentages, ai_credits):
            ax.text(bar.get_x() + bar.get_width()/2, val + 1, 
                   f'{val:.1f}%\n({cred:.0f} Cr)', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('AI Credits Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('AI Curriculum Integration Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(0, max(percentages)*1.3)

        plt.tight_layout()
        paths = self._save_figure(fig, 'ai_integration')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    def plot_optimization_score_comparison(self, figsize=(10, 6)):
        """
        优化得分对比
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        school_colors = [PlotStyleConfig.get_school_color(s) for s in self.schools]
        scores = [self.all_results[s]['curriculum_optimization']['optimal_score'] 
                 for s in self.schools]
        
        bars = ax.bar(self.schools, scores, color=school_colors, 
                     alpha=0.85, edgecolor='white', linewidth=2)
        
        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.2f}', 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Total Utility Score', fontsize=12, fontweight='bold')
        ax.set_title('Optimization Objective Score Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        paths = self._save_figure(fig, 'optimization_score')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    def plot_stacked_curriculum_comparison(self, figsize=(14, 8)):
        """
        堆积柱状图 - 课程结构优化前后对比
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')

        schools = ['CMU', 'CCAD', 'CIA']
        course_types = ['x_base', 'x_AI', 'x_ethics', 'x_proj']
        display_names = ['Base', 'AI', 'Ethics', 'Project']
        colors = ["#5B9BEF", "#F69D62", "#80EF6A", "#EA9DE1"]
        
        # 准备数据
        x_positions = []
        x_labels = []
        bar_width = 0.25
        gap = 0.1
        group_spacing = 0.3
        current_x = 0
        
        plot_data = {ctype: [] for ctype in course_types}
        
        for school in schools:
            if school not in self.all_results:
                continue
                
            init_params = EducationDecisionParams(school_name=school)
            init_curr = init_params.current_curriculum
            init_total = init_params.total_credits
            
            opt_curr = self.all_results[school]['curriculum_optimization']['optimal_curriculum']
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
                if val >= 5:
                    h = bar.get_height()
                    cx = bar.get_x() + bar.get_width()/2
                    cy = bar.get_y() + h/2
                    ax.text(cx, cy, f'{val:.0f}%', ha='center', va='center', 
                           fontsize=9, fontweight='bold', color='#444444')
            
            bottoms = [b + v for b, v in zip(bottoms, values)]

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold')
        ax.set_ylabel('Percentage of Total Credits (%)', fontweight='bold', fontsize=12)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}%'))
        
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=4, 
                 frameon=False, fontsize=11)

        fig.suptitle('Curriculum Structure: Before vs After Optimization',
                    fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()
        paths = self._save_figure(fig, 'stacked_curriculum')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    def plot_career_similarity_matrix(self, figsize=(11, 9)):
        """
        职业相似度矩阵热力图
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('white')
        
        # 使用第一个学校的职业向量
        first_school = self.schools[0]
        params = EducationDecisionParams(school_name=first_school)
        careers = list(params.CAREER_VECTORS.keys())
        display_careers = [params.CAREER_DISPLAY_NAMES.get(c, c) for c in careers]
        
        # 计算相似度矩阵
        similarity_matrix = np.zeros((len(careers), len(careers)))
        for i, origin in enumerate(careers):
            origin_vec = np.array(params.CAREER_VECTORS[origin])
            for j, target in enumerate(careers):
                target_vec = np.array(params.CAREER_VECTORS[target])
                if np.linalg.norm(origin_vec) == 0 or np.linalg.norm(target_vec) == 0:
                    similarity_matrix[i, j] = 0
                else:
                    similarity_matrix[i, j] = (np.dot(origin_vec, target_vec) / 
                                               (np.linalg.norm(origin_vec) * np.linalg.norm(target_vec)))

        im = ax.imshow(similarity_matrix, cmap='YlGnBu', aspect='auto', interpolation='nearest')
        
        # 添加数值标签
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
        cbar.set_label('Skill Overlap (Cosine Similarity)', fontweight='bold')
        cbar.outline.set_visible(False)

        fig.suptitle('Career Ecosystem Connectivity Analysis',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()
        paths = self._save_figure(fig, 'career_similarity_matrix')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    def plot_ahp_radar(self, figsize=(12, 10)):
        """
        AHP分析雷达图
        """
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#FAFBFC')

        ahp = get_ahp_calculator()
        radar_data = ahp.get_radar_data()
        
        criteria = ['Strategic\nScalability\n(C1: W=0.4)', 
                   'Physical\nIndependence\n(C2: W=0.4)', 
                   'Service\nElasticity\n(C3: W=0.2)']
        
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
            
            ax.fill(angles, values, alpha=0.2, color=style['color'], zorder=2)
            ax.plot(angles, values, style['linestyle'], linewidth=3, 
                   color=style['color'], markersize=12, marker=style['marker'],
                   markerfacecolor='white', markeredgewidth=2.5,
                   label=f'{school} (λ={ahp.final_lambdas[school]:.3f})', zorder=3)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 0.85)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=10)
        ax.grid(True, alpha=0.7, linewidth=1.2)
        
        legend = ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=12, framealpha=0.95)
        legend.get_frame().set_linewidth(1.5)

        fig.suptitle('AHP Analysis: Administrative Capacity (λ) Derivation',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        paths = self._save_figure(fig, 'ahp_radar')
        print(f"    ✅ Saved: {paths[0]}")
        return paths

    def plot_ahp_lambda_bar(self, figsize=(10, 6)):
        """
        AHP Lambda值柱状图
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#FAFBFC')
        
        ahp = get_ahp_calculator()
        lambdas = ahp.final_lambdas
        
        school_colors = [PlotStyleConfig.get_school_color(s) for s in self.schools]
        lambda_values = [lambdas[s] for s in self.schools]
        
        bars = ax.bar(self.schools, lambda_values, color=school_colors, 
                     alpha=0.85, edgecolor='white', linewidth=2)
        
        for bar, val in zip(bars, lambda_values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, 
                   f'{val:.3f}\n({val*100:.1f}%)', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Administrative Coefficient (λ)', fontsize=12, fontweight='bold')
        ax.set_title('AHP-Derived Lambda Values by University', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        paths = self._save_figure(fig, 'ahp_lambda_bar')
        print(f"    ✅ Saved: {paths[0]}")
        return paths


# ========== 主函数 ==========
def generate_all_split_visualizations():
    """
    生成所有拆分后的可视化图片
    """
    print("\n" + "="*70)
    print("  Task 2.1 拆分可视化生成器 - Split Visualization Generator")
    print("="*70)
    
    # 设置输出目录
    save_dir = './figures/task2_1_split'
    os.makedirs(save_dir, exist_ok=True)
    
    schools = ['CMU', 'CCAD', 'CIA']
    all_results = {}
    
    # 为每个学校生成分析结果和可视化
    for school in schools:
        print(f"\n{'─'*60}")
        print(f"  📊 Processing: {school}")
        print(f"{'─'*60}")
        
        # 创建模型并运行分析
        params = EducationDecisionParams(school_name=school)
        model = EducationDecisionModel(params)
        results = model.run_full_analysis(verbose=False)
        all_results[school] = results
        
        # 创建拆分可视化器
        viz = SplitVisualization(model, results, save_dir=save_dir)
        
        # 生成所有单独图片
        print(f"\n  📈 Generating individual plots for {school}...")
        
        # 招生响应
        viz.plot_enrollment_bar_chart()
        viz.plot_enrollment_flow_diagram()
        
        # 课程优化
        viz.plot_curriculum_comparison_bar()
        viz.plot_curriculum_pie_chart()
        viz.plot_utility_breakdown()
        viz.plot_ai_marginal_utility_curve()
        
        # SA收敛
        viz.plot_sa_convergence()
        
        # 职业弹性
        viz.plot_career_elasticity_bar()
        
        # 技能雷达图
        viz.plot_all_skill_radars()
        
        # 帕累托前沿
        viz.plot_pareto_frontier()
        
        # 灵敏度分析
        viz.plot_sensitivity_lambda()
        viz.plot_sensitivity_weight()
    
    # 生成跨学校比较图
    print(f"\n{'─'*60}")
    print(f"  📊 Generating Cross-School Comparison Plots")
    print(f"{'─'*60}")
    
    cross_viz = CrossSchoolVisualization(all_results, save_dir=save_dir)
    
    cross_viz.plot_pressure_index_comparison()
    cross_viz.plot_adjustment_comparison()
    cross_viz.plot_ai_integration_comparison()
    cross_viz.plot_optimization_score_comparison()
    cross_viz.plot_stacked_curriculum_comparison()
    cross_viz.plot_career_similarity_matrix()
    cross_viz.plot_ahp_radar()
    cross_viz.plot_ahp_lambda_bar()
    
    # 统计生成的文件
    files = [f for f in os.listdir(save_dir) if f.endswith('.png')]
    
    print(f"\n{'='*70}")
    print(f"  ✅ Visualization Generation Complete!")
    print(f"  📁 Output Directory: {os.path.abspath(save_dir)}")
    print(f"  📊 Total Plots Generated: {len(files)} PNG files")
    print(f"{'='*70}\n")
    
    return all_results


if __name__ == "__main__":
    generate_all_split_visualizations()
