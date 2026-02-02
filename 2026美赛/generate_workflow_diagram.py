"""
============================================================
模型工作流程图生成器
(Model Workflow Diagram Generator)
============================================================
功能：绘制Task1-4的完整模型流程图
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ModelWorkflowDiagram:
    """模型工作流程图生成器"""
    
    # 配色方案
    COLORS = {
        'task1': '#3498DB',      # 蓝色 - Task 1
        'task2': '#E74C3C',      # 红色 - Task 2
        'task3': '#2ECC71',      # 绿色 - Task 3
        'task4': '#9B59B6',      # 紫色 - Task 4
        'data': '#F39C12',       # 橙色 - 数据源
        'output': '#1ABC9C',     # 青色 - 输出
        'arrow': '#2C3E50',      # 深色 - 箭头
        'background': '#FAFBFC', # 背景
        'text_dark': '#2C3E50',  # 深色文字
        'text_light': '#FFFFFF', # 浅色文字
        'submodel': '#ECF0F1',   # 子模型背景
    }
    
    def __init__(self, save_dir='./figures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def draw_rounded_box(self, ax, x, y, width, height, color, text, 
                         text_color='white', fontsize=10, alpha=0.9, 
                         box_style='round,pad=0.02', linewidth=2):
        """绘制圆角矩形框"""
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle=box_style,
            facecolor=color,
            edgecolor='white',
            linewidth=linewidth,
            alpha=alpha
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=fontsize, color=text_color, fontweight='bold',
                wrap=True)
        return box
    
    def draw_arrow(self, ax, start, end, color='#2C3E50', style='->',
                   connectionstyle='arc3,rad=0', linewidth=2):
        """绘制箭头"""
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle=style,
            connectionstyle=connectionstyle,
            color=color,
            linewidth=linewidth,
            mutation_scale=15
        )
        ax.add_patch(arrow)
        return arrow
    
    def draw_main_workflow(self):
        """绘制主工作流程图"""
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 16)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor(self.COLORS['background'])
        fig.patch.set_facecolor(self.COLORS['background'])
        
        # ========== 标题 ==========
        ax.text(10, 15.5, 'AI-Driven Education Decision Model', 
                ha='center', va='center', fontsize=20, fontweight='bold',
                color=self.COLORS['text_dark'])
        ax.text(10, 15.0, 'Complete Workflow Diagram', 
                ha='center', va='center', fontsize=14, 
                color=self.COLORS['text_dark'], style='italic')
        
        # ========== 数据源层 ==========
        data_y = 14
        data_sources = [
            ('BLS Data\n就业统计', 4),
            ('O*NET Data\n技能数据库', 8),
            ('School Data\n学校信息', 12),
            ('UNESCO\n伦理指南', 16)
        ]
        for text, x in data_sources:
            self.draw_rounded_box(ax, x, data_y, 2.5, 0.9, 
                                  self.COLORS['data'], text,
                                  fontsize=8)
        
        # 数据源到Task1的箭头
        for _, x in data_sources:
            self.draw_arrow(ax, (x, data_y - 0.5), (10, 12.7),
                           color=self.COLORS['data'])
        
        # ========== Task 1 ==========
        task1_y = 11.5
        # 主框
        task1_box = FancyBboxPatch(
            (2, task1_y - 1.5), 16, 2.5,
            boxstyle='round,pad=0.03',
            facecolor=self.COLORS['task1'],
            edgecolor='white',
            linewidth=3,
            alpha=0.15
        )
        ax.add_patch(task1_box)
        
        # Task 1 标题
        ax.text(2.5, task1_y + 0.7, 'TASK 1', fontsize=12, fontweight='bold',
                color=self.COLORS['task1'])
        ax.text(2.5, task1_y + 0.3, 'AI Career Evolution Prediction', fontsize=9,
                color=self.COLORS['task1'])
        
        # Task 1 子模型
        self.draw_rounded_box(ax, 5, task1_y, 2.2, 1.2, 
                              self.COLORS['task1'], 'GM(1,1)\n基准预测',
                              fontsize=9)
        self.draw_rounded_box(ax, 9, task1_y, 2.2, 1.2, 
                              self.COLORS['task1'], 'S-Curve\n技术渗透',
                              fontsize=9)
        self.draw_rounded_box(ax, 13, task1_y, 2.2, 1.2, 
                              self.COLORS['task1'], 'Value\nRecompose',
                              fontsize=9)
        self.draw_rounded_box(ax, 17, task1_y, 1.8, 1.0, 
                              self.COLORS['output'], 'D_future',
                              fontsize=9)
        
        # Task 1 内部箭头
        self.draw_arrow(ax, (6.2, task1_y), (7.8, task1_y), 
                       color=self.COLORS['task1'])
        self.draw_arrow(ax, (10.2, task1_y), (11.8, task1_y), 
                       color=self.COLORS['task1'])
        self.draw_arrow(ax, (14.2, task1_y), (16, task1_y), 
                       color=self.COLORS['task1'])
        
        # ========== Task 1 到 Task 2 的箭头 ==========
        ax.annotate('', xy=(10, 8.8), xytext=(10, 10),
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['arrow'],
                                  lw=3, mutation_scale=20))
        ax.text(10.3, 9.4, 'D_future\nAI冲击指数', fontsize=8, 
                color=self.COLORS['task1'], ha='left')
        
        # ========== Task 2 ==========
        task2_y = 7.5
        # 主框
        task2_box = FancyBboxPatch(
            (2, task2_y - 1.8), 16, 3.0,
            boxstyle='round,pad=0.03',
            facecolor=self.COLORS['task2'],
            edgecolor='white',
            linewidth=3,
            alpha=0.15
        )
        ax.add_patch(task2_box)
        
        # Task 2 标题
        ax.text(2.5, task2_y + 0.9, 'TASK 2', fontsize=12, fontweight='bold',
                color=self.COLORS['task2'])
        ax.text(2.5, task2_y + 0.5, 'Education Decision Optimization', fontsize=9,
                color=self.COLORS['task2'])
        
        # Task 2 子模型 - 第一行
        self.draw_rounded_box(ax, 5, task2_y + 0.2, 2.4, 1.0, 
                              self.COLORS['task2'], '招生响应\nAHP→λ',
                              fontsize=8)
        self.draw_rounded_box(ax, 9, task2_y + 0.2, 2.4, 1.0, 
                              self.COLORS['task2'], '课程优化\nSA算法',
                              fontsize=8)
        self.draw_rounded_box(ax, 13, task2_y + 0.2, 2.4, 1.0, 
                              self.COLORS['task2'], '职业弹性\n余弦相似度',
                              fontsize=8)
        
        # Task 2 约束框
        self.draw_rounded_box(ax, 9, task2_y - 0.9, 6, 0.7, 
                              '#C0392B', '三重约束: 公平性 | 环境 | 安全',
                              fontsize=8, alpha=0.8)
        
        # Task 2 输出
        self.draw_rounded_box(ax, 17, task2_y, 1.8, 1.0, 
                              self.COLORS['output'], 'E_new, X*',
                              fontsize=9)
        
        # Task 2 内部箭头
        self.draw_arrow(ax, (6.3, task2_y + 0.2), (7.7, task2_y + 0.2), 
                       color=self.COLORS['task2'])
        self.draw_arrow(ax, (10.3, task2_y + 0.2), (11.7, task2_y + 0.2), 
                       color=self.COLORS['task2'])
        self.draw_arrow(ax, (14.3, task2_y + 0.2), (16, task2_y + 0.2), 
                       color=self.COLORS['task2'])
        
        # ========== Task 2 到 Task 3 的箭头 ==========
        ax.annotate('', xy=(10, 4.8), xytext=(10, 5.8),
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['arrow'],
                                  lw=3, mutation_scale=20))
        ax.text(10.3, 5.3, 'Strategy A vs B\n课程配比对比', fontsize=8, 
                color=self.COLORS['task2'], ha='left')
        
        # ========== Task 3 ==========
        task3_y = 3.5
        # 主框
        task3_box = FancyBboxPatch(
            (2, task3_y - 1.5), 16, 2.5,
            boxstyle='round,pad=0.03',
            facecolor=self.COLORS['task3'],
            edgecolor='white',
            linewidth=3,
            alpha=0.15
        )
        ax.add_patch(task3_box)
        
        # Task 3 标题
        ax.text(2.5, task3_y + 0.7, 'TASK 3', fontsize=12, fontweight='bold',
                color=self.COLORS['task3'])
        ax.text(2.5, task3_y + 0.3, 'AHP-TOPSIS Evaluation', fontsize=9,
                color=self.COLORS['task3'])
        
        # Task 3 子模型
        self.draw_rounded_box(ax, 5, task3_y, 2.2, 1.2, 
                              self.COLORS['task3'], 'AHP\n准则权重',
                              fontsize=9)
        self.draw_rounded_box(ax, 9, task3_y, 2.2, 1.2, 
                              self.COLORS['task3'], 'AHP\n方案评估',
                              fontsize=9)
        self.draw_rounded_box(ax, 13, task3_y, 2.2, 1.2, 
                              self.COLORS['task3'], 'TOPSIS\n综合排序',
                              fontsize=9)
        self.draw_rounded_box(ax, 17, task3_y, 1.8, 1.0, 
                              self.COLORS['output'], 'S_A, S_B\nB🏆',
                              fontsize=9)
        
        # Task 3 内部箭头
        self.draw_arrow(ax, (6.2, task3_y), (7.8, task3_y), 
                       color=self.COLORS['task3'])
        self.draw_arrow(ax, (10.2, task3_y), (11.8, task3_y), 
                       color=self.COLORS['task3'])
        self.draw_arrow(ax, (14.2, task3_y), (16, task3_y), 
                       color=self.COLORS['task3'])
        
        # ========== 右侧垂直箭头到Task 4 ==========
        # Task1 -> Task4
        ax.annotate('', xy=(19, 1.2), xytext=(17.9, 11.5),
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['task1'],
                                  lw=2, connectionstyle='arc3,rad=-0.1'))
        ax.text(19.2, 8, 'X轴\nAI冲击', fontsize=7, color=self.COLORS['task1'], rotation=-90)
        
        # Task2 -> Task4
        ax.annotate('', xy=(18.5, 1.2), xytext=(17.9, 7.5),
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['task2'],
                                  lw=2, connectionstyle='arc3,rad=-0.1'))
        ax.text(18.7, 4.5, 'Y轴\n资源弹性', fontsize=7, color=self.COLORS['task2'], rotation=-90)
        
        # Task3 -> Task4
        ax.annotate('', xy=(18, 1.2), xytext=(17.9, 3.5),
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['task3'],
                                  lw=2, connectionstyle='arc3,rad=-0.1'))
        ax.text(18.2, 2.3, 'Z轴\n安全系数', fontsize=7, color=self.COLORS['task3'], rotation=-90)
        
        # ========== Task 4 ==========
        task4_y = 0.6
        # 主框 - 横跨底部
        task4_box = FancyBboxPatch(
            (2, task4_y - 0.5), 16, 1.4,
            boxstyle='round,pad=0.03',
            facecolor=self.COLORS['task4'],
            edgecolor='white',
            linewidth=3,
            alpha=0.15
        )
        ax.add_patch(task4_box)
        
        # Task 4 标题
        ax.text(2.5, task4_y + 0.3, 'TASK 4', fontsize=12, fontweight='bold',
                color=self.COLORS['task4'])
        ax.text(2.5, task4_y - 0.1, 'Global Strategy Framework', fontsize=9,
                color=self.COLORS['task4'])
        
        # Task 4 子模型
        self.draw_rounded_box(ax, 6.5, task4_y, 2.2, 0.8, 
                              self.COLORS['task4'], 'Monte Carlo\nN=1000',
                              fontsize=8)
        self.draw_rounded_box(ax, 10.5, task4_y, 2.2, 0.8, 
                              self.COLORS['task4'], 'K-Means\nK=4聚类',
                              fontsize=8)
        self.draw_rounded_box(ax, 14.5, task4_y, 2.8, 0.8, 
                              self.COLORS['output'], '四类策略\n全球推广',
                              fontsize=8)
        
        # Task 4 内部箭头
        self.draw_arrow(ax, (7.7, task4_y), (9.3, task4_y), 
                       color=self.COLORS['task4'])
        self.draw_arrow(ax, (11.7, task4_y), (13, task4_y), 
                       color=self.COLORS['task4'])
        
        # ========== 图例 ==========
        legend_elements = [
            mpatches.Patch(facecolor=self.COLORS['task1'], label='Task 1: AI职业演化预测'),
            mpatches.Patch(facecolor=self.COLORS['task2'], label='Task 2: 教育决策优化'),
            mpatches.Patch(facecolor=self.COLORS['task3'], label='Task 3: AHP-TOPSIS评价'),
            mpatches.Patch(facecolor=self.COLORS['task4'], label='Task 4: 全球战略框架'),
            mpatches.Patch(facecolor=self.COLORS['data'], label='外部数据源'),
            mpatches.Patch(facecolor=self.COLORS['output'], label='模型输出'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', 
                 fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.save_dir, 'model_workflow_diagram.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=self.COLORS['background'])
        print(f"✓ 主流程图已保存至: {save_path}")
        
        # 同时保存PDF版本
        save_path_pdf = os.path.join(self.save_dir, 'model_workflow_diagram.pdf')
        plt.savefig(save_path_pdf, bbox_inches='tight', 
                   facecolor=self.COLORS['background'])
        print(f"✓ PDF版本已保存至: {save_path_pdf}")
        
        plt.close()
    
    def draw_detailed_task_flow(self):
        """绘制详细的任务流程图（垂直布局）"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 24))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 24)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor(self.COLORS['background'])
        fig.patch.set_facecolor(self.COLORS['background'])
        
        # ========== 标题 ==========
        ax.text(8, 23.5, 'Model Workflow - Detailed View', 
                ha='center', va='center', fontsize=18, fontweight='bold',
                color=self.COLORS['text_dark'])
        
        # ========== Task 1 区块 ==========
        y_start = 22
        
        # Task 1 标题框
        self.draw_rounded_box(ax, 8, y_start, 14, 1.0, 
                              self.COLORS['task1'], 
                              'TASK 1: AI Career Evolution Prediction Model',
                              fontsize=12)
        
        # 子模型
        y_sub = y_start - 1.5
        models1 = [
            ('Input:\nBLS历史数据', 2.5, '#95A5A6'),
            ('GM(1,1)\n灰色预测\n基准趋势', 5.5, self.COLORS['task1']),
            ('Logistic S-Curve\n技术渗透速度\nP(t)', 8.5, self.COLORS['task1']),
            ('Value Overlay\n价值重构\nF(t)', 11.5, self.COLORS['task1']),
            ('Output:\nD_future', 14, '#1ABC9C'),
        ]
        
        for text, x, color in models1:
            self.draw_rounded_box(ax, x, y_sub, 2.4, 1.3, color, text, fontsize=8)
        
        # 箭头
        for i in range(len(models1) - 1):
            self.draw_arrow(ax, (models1[i][1] + 1.3, y_sub), 
                           (models1[i+1][1] - 1.3, y_sub),
                           color=self.COLORS['arrow'])
        
        # 公式注释
        ax.text(5.5, y_sub - 1.2, r'$\hat{x}^{(1)}(k) = [x^{(0)}(1) - \frac{b}{a}]e^{-ak} + \frac{b}{a}$',
                fontsize=9, ha='center', color=self.COLORS['task1'])
        ax.text(8.5, y_sub - 1.2, r'$P(t) = \frac{L}{1+e^{-k(t-t_0)}}$',
                fontsize=9, ha='center', color=self.COLORS['task1'])
        ax.text(11.5, y_sub - 1.2, r'$F(t) = Y_t \times [(1-P)(1-D4) + PA + N]$',
                fontsize=9, ha='center', color=self.COLORS['task1'])
        
        # ========== 连接箭头 ==========
        ax.annotate('', xy=(8, 17.5), xytext=(8, 18.5),
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['arrow'],
                                  lw=3, mutation_scale=20))
        ax.text(8.3, 18, 'D_future → E_current对比', fontsize=9, ha='left')
        
        # ========== Task 2 区块 ==========
        y_start = 17
        
        # Task 2 标题框
        self.draw_rounded_box(ax, 8, y_start, 14, 1.0, 
                              self.COLORS['task2'], 
                              'TASK 2: Education Decision Optimization Model',
                              fontsize=12)
        
        # Layer 1: 招生响应
        y_layer1 = y_start - 1.3
        ax.text(1, y_layer1, 'Layer 1:', fontsize=9, fontweight='bold', 
                color=self.COLORS['task2'])
        
        self.draw_rounded_box(ax, 4, y_layer1, 3, 0.8, 
                              self.COLORS['task2'], 'Γ = (D-E)/E', fontsize=9)
        self.draw_rounded_box(ax, 8, y_layer1, 3.5, 0.8, 
                              self.COLORS['task2'], 'ΔE = E×λ×tanh(Γ)', fontsize=9)
        self.draw_rounded_box(ax, 12.5, y_layer1, 3, 0.8, 
                              self.COLORS['task2'], 'E_new = E + ΔE', fontsize=9)
        
        self.draw_arrow(ax, (5.6, y_layer1), (6.2, y_layer1))
        self.draw_arrow(ax, (9.8, y_layer1), (10.9, y_layer1))
        
        # Layer 2: 课程优化
        y_layer2 = y_layer1 - 1.3
        ax.text(1, y_layer2, 'Layer 2:', fontsize=9, fontweight='bold', 
                color=self.COLORS['task2'])
        
        self.draw_rounded_box(ax, 4.5, y_layer2, 4, 0.8, 
                              self.COLORS['task2'], 'max J(X) = U(X) - C(X)', fontsize=9)
        self.draw_rounded_box(ax, 9.5, y_layer2, 4, 0.8, 
                              self.COLORS['task2'], 'SA: T₀=200, α=0.98', fontsize=9)
        self.draw_rounded_box(ax, 14, y_layer2, 2.5, 0.8, 
                              '#1ABC9C', 'X* optimal', fontsize=9)
        
        self.draw_arrow(ax, (6.6, y_layer2), (7.4, y_layer2))
        self.draw_arrow(ax, (11.6, y_layer2), (12.7, y_layer2))
        
        # 约束条件
        y_const = y_layer2 - 1.0
        ax.text(4, y_const, '约束:', fontsize=9, fontweight='bold', color='#C0392B')
        constraints = [
            '① 公平性: Σ(e·x)/S ≤ E_max',
            '② 环境: Σx_high/S ≤ β_env', 
            '③ 安全: x_ethics ≥ γ·x_AI'
        ]
        for i, c in enumerate(constraints):
            ax.text(4 + i*4.5, y_const - 0.5, c, fontsize=8, color='#C0392B')
        
        # ========== 连接箭头 ==========
        ax.annotate('', xy=(8, 11.5), xytext=(8, 12.5),
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['arrow'],
                                  lw=3, mutation_scale=20))
        ax.text(8.3, 12, 'Strategy A vs B', fontsize=9, ha='left')
        
        # ========== Task 3 区块 ==========
        y_start = 11
        
        # Task 3 标题框
        self.draw_rounded_box(ax, 8, y_start, 14, 1.0, 
                              self.COLORS['task3'], 
                              'TASK 3: AHP-TOPSIS Dual Evaluation Framework',
                              fontsize=12)
        
        # Phase 1: AHP
        y_p1 = y_start - 1.3
        ax.text(1, y_p1, 'Phase 1:', fontsize=9, fontweight='bold', 
                color=self.COLORS['task3'])
        
        self.draw_rounded_box(ax, 4.5, y_p1, 3.5, 0.8, 
                              self.COLORS['task3'], 'AHP 判断矩阵', fontsize=9)
        self.draw_rounded_box(ax, 9, y_p1, 3.5, 0.8, 
                              self.COLORS['task3'], '特征向量法', fontsize=9)
        self.draw_rounded_box(ax, 13.5, y_p1, 3, 0.8, 
                              self.COLORS['task3'], 'w=[.36,.12,.33,.19]', fontsize=8)
        
        self.draw_arrow(ax, (6.3, y_p1), (7.2, y_p1))
        self.draw_arrow(ax, (10.8, y_p1), (11.9, y_p1))
        
        # Phase 2: TOPSIS
        y_p2 = y_p1 - 1.2
        ax.text(1, y_p2, 'Phase 2:', fontsize=9, fontweight='bold', 
                color=self.COLORS['task3'])
        
        self.draw_rounded_box(ax, 4.5, y_p2, 3.5, 0.8, 
                              self.COLORS['task3'], '归一化决策矩阵', fontsize=9)
        self.draw_rounded_box(ax, 9, y_p2, 3.5, 0.8, 
                              self.COLORS['task3'], '理想解 V⁺, V⁻', fontsize=9)
        self.draw_rounded_box(ax, 13.5, y_p2, 3, 0.8, 
                              '#1ABC9C', 'S_B=0.58 🏆', fontsize=9)
        
        self.draw_arrow(ax, (6.3, y_p2), (7.2, y_p2))
        self.draw_arrow(ax, (10.8, y_p2), (11.9, y_p2))
        
        # ========== 连接箭头 ==========
        ax.annotate('', xy=(8, 6.0), xytext=(8, 7.0),
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['arrow'],
                                  lw=3, mutation_scale=20))
        ax.text(8.3, 6.5, '(X, Y, Z) 三维参数', fontsize=9, ha='left')
        
        # ========== Task 4 区块 ==========
        y_start = 5.5
        
        # Task 4 标题框
        self.draw_rounded_box(ax, 8, y_start, 14, 1.0, 
                              self.COLORS['task4'], 
                              'TASK 4: Global Education Strategy Framework',
                              fontsize=12)
        
        # 子流程
        y_sub4 = y_start - 1.5
        models4 = [
            ('3D Space\n(X,Y,Z)', 3, self.COLORS['task4']),
            ('Monte Carlo\nN=1000', 6.5, self.COLORS['task4']),
            ('K-Means\nK=4', 10, self.COLORS['task4']),
            ('Strategy\nMatrix', 13.5, '#1ABC9C'),
        ]
        
        for text, x, color in models4:
            self.draw_rounded_box(ax, x, y_sub4, 2.5, 1.0, color, text, fontsize=9)
        
        for i in range(len(models4) - 1):
            self.draw_arrow(ax, (models4[i][1] + 1.35, y_sub4), 
                           (models4[i+1][1] - 1.35, y_sub4),
                           color=self.COLORS['arrow'])
        
        # 聚类结果
        y_cluster = y_sub4 - 1.5
        clusters = [
            ('Cluster 0:\nTech Pioneers', 2.5, '#3498DB'),
            ('Cluster 1:\nAdaptive Balancers', 6, '#2ECC71'),
            ('Cluster 2:\nTraditional Defenders', 10, '#F39C12'),
            ('Cluster 3:\nCautious Observers', 13.5, '#9B59B6'),
        ]
        
        for text, x, color in clusters:
            self.draw_rounded_box(ax, x, y_cluster, 3, 0.9, color, text, fontsize=8)
        
        # ========== 最终输出 ==========
        y_output = 0.8
        self.draw_rounded_box(ax, 8, y_output, 12, 1.0, 
                              '#1ABC9C', 
                              'FINAL OUTPUT: 全球任意教育机构的定制化策略建议',
                              fontsize=11)
        
        ax.annotate('', xy=(8, 1.4), xytext=(8, 2.0),
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['arrow'],
                                  lw=3, mutation_scale=20))
        
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.save_dir, 'model_workflow_detailed.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=self.COLORS['background'])
        print(f"✓ 详细流程图已保存至: {save_path}")
        
        plt.close()
    
    def draw_data_flow_diagram(self):
        """绘制数据流向图"""
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor(self.COLORS['background'])
        fig.patch.set_facecolor(self.COLORS['background'])
        
        # 标题
        ax.text(9, 11.5, 'Model Data Flow Diagram', 
                ha='center', va='center', fontsize=18, fontweight='bold',
                color=self.COLORS['text_dark'])
        
        # 四个Task框
        tasks = [
            ('TASK 1\nAI Evolution', 3, 8, self.COLORS['task1']),
            ('TASK 2\nDecision Opt.', 9, 8, self.COLORS['task2']),
            ('TASK 3\nEvaluation', 9, 4, self.COLORS['task3']),
            ('TASK 4\nGlobal Strategy', 15, 6, self.COLORS['task4']),
        ]
        
        for text, x, y, color in tasks:
            self.draw_rounded_box(ax, x, y, 4, 2, color, text, fontsize=11)
        
        # 数据流箭头和标注
        # Task 1 -> Task 2
        ax.annotate('', xy=(6.9, 8), xytext=(5.1, 8),
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['arrow'],
                                  lw=3, mutation_scale=20))
        ax.text(6, 8.4, 'D_future', fontsize=10, ha='center', 
                color=self.COLORS['task1'], fontweight='bold')
        
        # Task 2 -> Task 3
        ax.annotate('', xy=(9, 5.1), xytext=(9, 6.9),
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['arrow'],
                                  lw=3, mutation_scale=20))
        ax.text(9.3, 6, 'Strategy\nA vs B', fontsize=9, ha='left', 
                color=self.COLORS['task2'], fontweight='bold')
        
        # Task 1 -> Task 4 (X轴)
        ax.annotate('', xy=(13, 7), xytext=(5, 8),
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['task1'],
                                  lw=2, connectionstyle='arc3,rad=0.3'))
        ax.text(8, 9.5, 'X: AI冲击指数', fontsize=9, 
                color=self.COLORS['task1'], fontweight='bold')
        
        # Task 2 -> Task 4 (Y轴)
        ax.annotate('', xy=(13, 6.5), xytext=(11, 7.5),
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['task2'],
                                  lw=2, connectionstyle='arc3,rad=0.2'))
        ax.text(12, 7.8, 'Y: 资源弹性(λ)', fontsize=9, 
                color=self.COLORS['task2'], fontweight='bold')
        
        # Task 3 -> Task 4 (Z轴)
        ax.annotate('', xy=(13, 5.5), xytext=(11, 4.5),
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['task3'],
                                  lw=2, connectionstyle='arc3,rad=-0.2'))
        ax.text(12, 4.2, 'Z: 安全系数', fontsize=9, 
                color=self.COLORS['task3'], fontweight='bold')
        
        # 输出框
        self.draw_rounded_box(ax, 9, 1.5, 12, 1.5, 
                              '#1ABC9C', 
                              'OUTPUT: 四类策略聚类 + 定制化决策建议矩阵',
                              fontsize=11)
        
        ax.annotate('', xy=(13, 2.3), xytext=(15, 5),
                   arrowprops=dict(arrowstyle='->', color=self.COLORS['arrow'],
                                  lw=3, mutation_scale=20))
        
        # 图例
        legend_elements = [
            mpatches.Patch(facecolor=self.COLORS['task1'], label='Task 1: AI职业演化预测'),
            mpatches.Patch(facecolor=self.COLORS['task2'], label='Task 2: 教育决策优化'),
            mpatches.Patch(facecolor=self.COLORS['task3'], label='Task 3: AHP-TOPSIS评价'),
            mpatches.Patch(facecolor=self.COLORS['task4'], label='Task 4: 全球战略框架'),
        ]
        ax.legend(handles=legend_elements, loc='lower left', 
                 fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.save_dir, 'model_data_flow.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=self.COLORS['background'])
        print(f"✓ 数据流向图已保存至: {save_path}")
        
        plt.close()
    
    def generate_all_diagrams(self):
        """生成所有图表"""
        print("="*60)
        print("📊 模型工作流程图生成器")
        print("="*60)
        
        print("\n[1/3] 生成主工作流程图...")
        self.draw_main_workflow()
        
        print("\n[2/3] 生成详细任务流程图...")
        self.draw_detailed_task_flow()
        
        print("\n[3/3] 生成数据流向图...")
        self.draw_data_flow_diagram()
        
        print("\n" + "="*60)
        print("✅ 所有图表生成完成！")
        print(f"📁 保存目录: {os.path.abspath(self.save_dir)}")
        print("="*60)


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    # 创建图表生成器
    diagram_generator = ModelWorkflowDiagram(
        save_dir='./figures'
    )
    
    # 生成所有图表
    diagram_generator.generate_all_diagrams()
