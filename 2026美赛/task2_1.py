"""
============================================================
AI 驱动的教育决策模型 - 完整工作流
(AI-Driven Education Decision Model - Complete Workflow)
============================================================
功能：基于AI影响预测的教育决策模型
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================

模型框架：
1. 宏观决策 —— 动态招生响应模型 (Sub-model 1)
2. 核心求解 —— 课程优化与多准则约束 (SA Algorithm + Refined Model 4)
3. 安全网 —— 职业路径弹性 (Career Path Elasticity)
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import os
import warnings
from math import tanh, sqrt

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
        plt.style.use('default')
        rcParams['font.family'] = 'DejaVu Sans'
        rcParams['font.size'] = 12
        rcParams['axes.labelsize'] = 12
        rcParams['axes.titlesize'] = 14
        rcParams['xtick.labelsize'] = 10
        rcParams['ytick.labelsize'] = 10
        rcParams['legend.fontsize'] = 10
        rcParams['figure.titlesize'] = 16
        rcParams['axes.grid'] = True
        rcParams['grid.alpha'] = 0.3
        rcParams['axes.facecolor'] = PlotStyleConfig.COLORS['background']

    @staticmethod
    def get_palette(n=None):
        if n is None:
            return PlotStyleConfig.PALETTE
        return PlotStyleConfig.PALETTE[:n] if n <= len(PlotStyleConfig.PALETTE) else PlotStyleConfig.PALETTE * (n // len(PlotStyleConfig.PALETTE)) + PlotStyleConfig.PALETTE[:n % len(PlotStyleConfig.PALETTE)]


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
            fig.savefig(path, dpi=300, bbox_inches='tight')
            paths.append(path)
        return paths


# 设置绘图风格
PlotStyleConfig.setup_style('academic')


# ============================================================
# 第一部分：模型参数配置 (Model Parameters Configuration)
# ============================================================

class EducationDecisionParams:
    """
    AI驱动的教育决策模型参数配置类

    ★★★ 需要调整的参数在这里修改 ★★★

    数据占位符：请替换为你的实际数据
    """

    # 学校参数配置字典
    SCHOOL_PARAMS = {
        'CMU': {
            'lambda': 0.15,  # 行政调整上限
            'current_graduates': 500,  # 当前毕业生人数（占位符）
            'E_cost': 0.8,  # 能源惩罚
            'R_risk': 0.4,  # 风险惩罚
            'current_curriculum': {'x_base': 80, 'x_AI': 5, 'x_ethics': 15, 'x_proj': 20}  # 当前课表
        },
        'CIA': {
            'lambda': 0.05,
            'current_graduates': 200,
            'E_cost': 0.5,
            'R_risk': 0.9,
            'current_curriculum': {'x_base': 85, 'x_AI': 3, 'x_ethics': 20, 'x_proj': 12}
        },
        'RISD': {
            'lambda': 0.05,
            'current_graduates': 150,
            'E_cost': 0.1,
            'R_risk': 0.0,
            'current_curriculum': {'x_base': 90, 'x_AI': 2, 'x_ethics': 10, 'x_proj': 18}
        }
    }

    # 职业技能向量（占位符，基于O*NET数据）
    CAREER_VECTORS = {
        'software_engineer': [0.9, 0.8, 0.7, 0.6, 0.5],
        'graphic_designer': [0.6, 0.9, 0.8, 0.4, 0.3],
        'chef': [0.2, 0.3, 0.9, 0.8, 0.7],
        'web_developer': [0.8, 0.7, 0.6, 0.5, 0.4],
        'fine_artist': [0.3, 0.8, 0.9, 0.7, 0.6],
        'interactive_media': [0.7, 0.8, 0.6, 0.5, 0.4]
    }

    def __init__(self, school_name=None, demand_2030=None):
        # ============ 学校基本信息 ============
        self.school_name = school_name or "CMU"  # 学校名称

        # ============ 预测需求数据 ============
        self.demand_2030 = demand_2030 or 600  # 2030年预测需求（占位符）

        # ============ 模拟退火参数 ============
        self.total_credits = 120  # 总学分
        self.gamma = 0.5  # 惩罚权重（降低惩罚）
        self.alpha = 0.3  # 能源惩罚系数
        self.beta = 0.3   # 风险惩罚系数
        self.sa_iterations = 1000  # SA迭代次数
        self.sa_temp = 100  # 初始温度
        self.sa_cooling = 0.99  # 冷却率

        # ============ 技能权重（O*NET权重） ============
        self.skill_weights = {'x_base': 0.3, 'x_AI': 0.4, 'x_ethics': 0.2, 'x_proj': 0.1}

        # ============ 根据学校设置参数 ============
        self._set_school_params()

    def _set_school_params(self):
        """根据学校设置参数"""
        if self.school_name in self.SCHOOL_PARAMS:
            params = self.SCHOOL_PARAMS[self.school_name]
            self.lambda_admin = params['lambda']
            self.current_graduates = params['current_graduates']
            self.E_cost = params['E_cost']
            self.R_risk = params['R_risk']
            self.current_curriculum = params['current_curriculum']

    def summary(self):
        """打印参数摘要"""
        print("\n" + "="*70)
        print("📋 AI-Driven Education Decision Model Parameters Configuration")
        print("="*70)

        print(f"\n【School】: {self.school_name}")
        print(f"【2030 Demand】: {self.demand_2030}")
        print(f"【Current Graduates】: {self.current_graduates}")
        print(f"【Admin Adjustment Limit (λ)】: {self.lambda_admin}")
        print(f"【Energy Cost (E_cost)】: {self.E_cost}")
        print(f"【Risk Cost (R_risk)】: {self.R_risk}")

        print("\n【Current Curriculum】")
        for k, v in self.current_curriculum.items():
            print(f"  {k}: {v} credits")

        print("\n【SA Parameters】")
        print(f"  Total Credits: {self.total_credits}")
        print(f"  Gamma: {self.gamma}, Alpha: {self.alpha}, Beta: {self.beta}")
        print(f"  Iterations: {self.sa_iterations}")

        print("="*70 + "\n")


# ============================================================
# 第二部分：AI教育决策模型核心计算 (Core Model Calculations)
# ============================================================

class EducationDecisionModel:
    """
    AI教育决策模型核心类

    实现三个子模型的计算
    """

    def __init__(self, params: EducationDecisionParams = None):
        self.params = params if params else EducationDecisionParams()

    def enrollment_response(self):
        """
        宏观决策 —— 动态招生响应模型

        计算压力指数和调整幅度
        """
        p = self.params
        F_t = p.demand_2030
        E_current = p.current_graduates
        lambda_admin = p.lambda_admin

        # 压力指数
        Gamma_t = (F_t - E_current) / E_current

        # 调整幅度
        Delta_E = E_current * lambda_admin * tanh(Gamma_t)

        return {
            'pressure_index': Gamma_t,
            'adjustment': Delta_E,
            'recommended_graduates': E_current + Delta_E
        }

    def curriculum_optimization_sa(self):
        """
        核心求解 —— 课程优化与多准则约束 (SA算法)

        使用模拟退火优化课程学分分配
        """
        p = self.params

        def objective_function(X):
            """目标函数 J(X)"""
            x_base, x_AI, x_ethics, x_proj = X
            skill_utility = sum(p.skill_weights[k] * v for k, v in zip(['x_base', 'x_AI', 'x_ethics', 'x_proj'], X))
            penalty = p.gamma * x_AI * (p.alpha * p.E_cost + p.beta * p.R_risk)
            return skill_utility - penalty

        def constraint(X):
            """约束：总学分=120，且各学分>=0"""
            return p.total_credits - sum(X)

        # 初始化
        current_X = np.array([p.current_curriculum['x_base'], p.current_curriculum['x_AI'],
                             p.current_curriculum['x_ethics'], p.current_curriculum['x_proj']])
        current_J = objective_function(current_X)

        best_X = current_X.copy()
        best_J = current_J

        temp = p.sa_temp

        # SA过程
        for i in range(p.sa_iterations):
            # 扰动：随机调整学分
            new_X = current_X.copy()
            idx1, idx2 = np.random.choice(4, 2, replace=False)
            transfer = np.random.randint(1, 6)  # 转移1-5学分
            new_X[idx1] -= transfer
            new_X[idx2] += transfer

            # 确保非负
            if np.any(new_X < 0):
                continue

            # 确保总学分不变
            if abs(sum(new_X) - p.total_credits) > 1e-6:
                continue

            new_J = objective_function(new_X)

            # 接受准则
            if new_J > current_J or np.random.rand() < np.exp((new_J - current_J) / temp):
                current_X = new_X
                current_J = new_J

                if new_J > best_J:
                    best_X = new_X
                    best_J = new_J

            # 冷却
            temp *= p.sa_cooling

        return {
            'optimal_curriculum': {'x_base': best_X[0], 'x_AI': best_X[1], 'x_ethics': best_X[2], 'x_proj': best_X[3]},
            'optimal_score': best_J,
            'skill_utility': sum(p.skill_weights[k] * v for k, v in zip(['x_base', 'x_AI', 'x_ethics', 'x_proj'], best_X)),
            'penalty': p.gamma * best_X[1] * (p.alpha * p.E_cost + p.beta * p.R_risk)
        }

    def career_elasticity(self, origin_career, target_careers=None):
        """
        安全网 —— 职业路径弹性

        计算余弦相似度
        """
        if target_careers is None:
            target_careers = list(self.params.CAREER_VECTORS.keys())
            target_careers.remove(origin_career)

        origin_vec = np.array(self.params.CAREER_VECTORS[origin_career])

        similarities = {}
        for target in target_careers:
            target_vec = np.array(self.params.CAREER_VECTORS[target])
            dot_product = np.dot(origin_vec, target_vec)
            norm_origin = np.linalg.norm(origin_vec)
            norm_target = np.linalg.norm(target_vec)
            cos_sim = dot_product / (norm_origin * norm_target)
            similarities[target] = cos_sim

        return similarities

    def run_full_analysis(self, verbose=True):
        """
        执行完整分析流程

        :param verbose: 是否打印详细信息
        :return: 分析结果字典
        """
        if verbose:
            print("🔍 Running full education decision analysis...")

        # 子模型1: 招生响应
        enrollment_results = self.enrollment_response()

        # 子模型2: 课程优化
        curriculum_results = self.curriculum_optimization_sa()

        # 子模型3: 职业弹性（针对当前职业）
        career = 'software_engineer' if self.params.school_name == 'CMU' else ('graphic_designer' if self.params.school_name == 'RISD' else 'chef')
        elasticity_results = self.career_elasticity(career)

        results = {
            'enrollment_response': enrollment_results,
            'curriculum_optimization': curriculum_results,
            'career_elasticity': elasticity_results
        }

        if verbose:
            print("✅ Analysis completed.")

        return results


# ============================================================
# 第三部分：可视化模块 (Visualization Module)
# ============================================================

class EducationDecisionVisualization:
    """
    AI教育决策可视化类
    """

    def __init__(self, model: EducationDecisionModel, results: dict, save_dir='./figures'):
        self.model = model
        self.results = results
        self.saver = FigureSaver(save_dir)

    def plot_enrollment_response(self, figsize=(12, 8)):
        """
        绘制招生响应分析图
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{self.model.params.school_name} - Enrollment Response Analysis',
                    fontsize=18, fontweight='bold')

        r = self.results['enrollment_response']
        colors = PlotStyleConfig.get_palette()

        # 子图1: 供需对比
        ax1 = axes[0, 0]
        ax1.bar(['Current Graduates', '2030 Demand'], [self.model.params.current_graduates, self.model.params.demand_2030],
                color=[colors[0], colors[1]], alpha=0.7)
        ax1.set_title('Supply vs Demand Comparison', fontweight='bold')
        ax1.set_ylabel('Number of Graduates')

        # 子图2: 压力指数
        ax2 = axes[0, 1]
        ax2.bar(['Pressure Index'], [r['pressure_index']], color=colors[2])
        ax2.set_title('Pressure Index (Γ_t)', fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 子图3: 调整幅度
        ax3 = axes[1, 0]
        ax3.bar(['Adjustment (ΔE)'], [r['adjustment']], color=colors[3])
        ax3.set_title('Enrollment Adjustment', fontweight='bold')
        ax3.set_ylabel('Change in Graduates')

        # 子图4: 推荐招生
        ax4 = axes[1, 1]
        ax4.bar(['Recommended Graduates'], [r['recommended_graduates']], color=colors[4])
        ax4.set_title('Recommended Enrollment', fontweight='bold')
        ax4.set_ylabel('Number of Graduates')

        plt.tight_layout()
        paths = self.saver.save(fig, 'enrollment_response_analysis')
        print(f"  💾 Enrollment response plot saved: {paths[0]}")

    def plot_curriculum_optimization(self, figsize=(14, 8)):
        """
        绘制课程优化分析图
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f'{self.model.params.school_name} - Curriculum Optimization Analysis',
                    fontsize=18, fontweight='bold')

        r = self.results['curriculum_optimization']
        colors = PlotStyleConfig.get_palette()

        # 子图1: 当前vs最优课表对比
        ax1 = axes[0]
        current = list(self.model.params.current_curriculum.values())
        optimal = list(r['optimal_curriculum'].values())
        labels = list(self.model.params.current_curriculum.keys())

        x = np.arange(len(labels))
        width = 0.35

        ax1.bar(x - width/2, current, width, label='Current', color=colors[0], alpha=0.7)
        ax1.bar(x + width/2, optimal, width, label='Optimal', color=colors[1], alpha=0.7)
        ax1.set_title('Current vs Optimal Curriculum', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('Credits')
        ax1.legend()

        # 子图2: 目标函数分解
        ax2 = axes[1]
        ax2.bar(['Skill Utility', 'Penalty'], [r['skill_utility'], r['penalty']],
                color=[colors[2], colors[3]], alpha=0.7)
        ax2.set_title('Objective Function Breakdown', fontweight='bold')
        ax2.set_ylabel('Score')

        # 子图3: AI学分与惩罚关系
        ax3 = axes[2]
        x_AI_range = np.linspace(0, 30, 100)
        penalty_range = self.model.params.gamma * x_AI_range * (self.model.params.alpha * self.model.params.E_cost + self.model.params.beta * self.model.params.R_risk)
        utility_range = self.model.params.skill_weights['x_AI'] * x_AI_range

        ax3.plot(x_AI_range, utility_range, label='Skill Utility', color=colors[4])
        ax3.plot(x_AI_range, penalty_range, label='Penalty', color=colors[5])
        ax3.plot(x_AI_range, utility_range - penalty_range, label='Net Benefit', color=colors[6])
        ax3.axvline(x=r['optimal_curriculum']['x_AI'], color='red', linestyle='--', label='Optimal AI Credits')
        ax3.set_title('AI Credits vs Costs/Benefits', fontweight='bold')
        ax3.set_xlabel('AI Credits')
        ax3.set_ylabel('Score')
        ax3.legend()

        plt.tight_layout()
        paths = self.saver.save(fig, 'curriculum_optimization_analysis')
        print(f"  💾 Curriculum optimization plot saved: {paths[0]}")

    def plot_career_elasticity(self, figsize=(10, 6)):
        """
        绘制职业路径弹性分析图
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(f'{self.model.params.school_name} - Career Path Elasticity Analysis',
                    fontsize=18, fontweight='bold')

        r = self.results['career_elasticity']
        careers = list(r.keys())
        similarities = list(r.values())
        colors = PlotStyleConfig.get_palette(len(careers))

        bars = ax.bar(careers, similarities, color=colors, alpha=0.7)
        ax.set_title('Cosine Similarity to Origin Career', fontweight='bold')
        ax.set_ylabel('Similarity Score')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Medium Elasticity Threshold')

        # 添加数值标签
        for bar, sim in zip(bars, similarities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{sim:.2f}', ha='center', va='bottom', fontsize=10)

        ax.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        paths = self.saver.save(fig, 'career_elasticity_analysis')
        print(f"  💾 Career elasticity plot saved: {paths[0]}")


# ============================================================
# 第四部分：主工作流 (Main Workflow)
# ============================================================

def run_education_decision_workflow():
    """
    运行AI教育决策工作流

    包括：参数配置 → 模型分析 → 可视化 → 结果保存
    """
    print("\n" + "█"*70)
    print("█" + " "*18 + "AI教育决策模型" + " "*21 + "█")
    print("█" + " "*13 + "AI-Driven Education Decision" + " "*14 + "█")
    print("█"*70 + "\n")

    # ========== Step 1: 参数配置 ==========
    print("【Step 1】初始化模型参数...")
    params = EducationDecisionParams()

    # ★★★ 在这里修改你的参数和数据 ★★★
    # params.school_name = "你的学校名称"
    # params.demand_2030 = 你的2030年需求预测
    # params.current_graduates = 你的当前毕业生人数
    # params.lambda_admin = 你的行政调整上限
    # params.E_cost = 你的能源惩罚系数
    # params.R_risk = 你的风险惩罚系数

    params.summary()

    # ========== Step 2: 创建模型 ==========
    print("【Step 2】创建决策模型...")
    model = EducationDecisionModel(params)

    # ========== Step 3: 执行分析 ==========
    print("【Step 3】执行教育决策分析...")
    results = model.run_full_analysis(verbose=True)

    # ========== Step 4: 生成可视化 ==========
    print("\n【Step 4】生成可视化图表...")
    print("-"*70)

    # 创建figures目录
    os.makedirs('./figures', exist_ok=True)

    viz = EducationDecisionVisualization(model, results, save_dir='./figures')

    # 图1: 招生响应分析
    print("\n  🎨 绘制招生响应分析图...")
    viz.plot_enrollment_response()

    # 图2: 课程优化分析
    print("\n  🎨 绘制课程优化分析图...")
    viz.plot_curriculum_optimization()

    # 图3: 职业弹性分析
    print("\n  🎨 绘制职业路径弹性分析图...")
    viz.plot_career_elasticity()

    # ========== Step 5: 保存结果 ==========
    print("\n【Step 5】保存分析结果...")
    print("-"*70)

    # 保存为CSV（用print输出结果）
    print("\n分析结果:")
    print(f"Pressure Index: {results['enrollment_response']['pressure_index']:.3f}")
    print(f"Adjustment: {results['enrollment_response']['adjustment']:.1f}")
    print(f"Recommended Graduates: {results['enrollment_response']['recommended_graduates']:.1f}")
    print(f"Optimal AI Credits: {results['curriculum_optimization']['optimal_curriculum']['x_AI']:.1f}")
    print(f"Optimal Score: {results['curriculum_optimization']['optimal_score']:.3f}")
    print("  📁 分析结果已打印（CSV保存功能已禁用以避免依赖问题）")

    # result_df = pd.DataFrame({
    #     'Metric': ['Pressure Index', 'Adjustment', 'Recommended Graduates', 'Optimal AI Credits', 'Optimal Score'],
    #     'Value': [results['enrollment_response']['pressure_index'],
    #              results['enrollment_response']['adjustment'],
    #              results['enrollment_response']['recommended_graduates'],
    #              results['curriculum_optimization']['optimal_curriculum']['x_AI'],
    #              results['curriculum_optimization']['optimal_score']]
    # })
    # result_df.to_csv('./figures/education_decision_results.csv', index=False, encoding='utf-8-sig')
    # print("  📁 分析结果已保存: ./figures/education_decision_results.csv")

    print("\n" + "█"*70)
    print("█" + " "*25 + "工作流执行完成!" + " "*26 + "█")
    print("█"*70 + "\n")

    return results


# ============================================================
# 主程序入口 (Main Entry Point)
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # ★★★ 运行教育决策工作流 ★★★
    # ============================================================
    results = run_education_decision_workflow()

    # ============================================================
    # ★★★ 自定义分析示例 ★★★
    # ============================================================

    # 1. 查看招生响应结果
    # print(f"\n招生响应结果:")
    # print(f"  压力指数: {results['enrollment_response']['pressure_index']:.3f}")
    # print(f"  调整幅度: {results['enrollment_response']['adjustment']:.1f}")
    # print(f"  推荐毕业生数: {results['enrollment_response']['recommended_graduates']:.1f}")

    # 2. 查看课程优化结果
    # print(f"\n课程优化结果:")
    # for k, v in results['curriculum_optimization']['optimal_curriculum'].items():
    #     print(f"  {k}: {v:.1f} credits")
    # print(f"  最优得分: {results['curriculum_optimization']['optimal_score']:.3f}")

    # 3. 查看职业弹性结果
    # print(f"\n职业路径弹性:")
    # for career, sim in results['career_elasticity'].items():
    #     print(f"  {career}: {sim:.3f}")