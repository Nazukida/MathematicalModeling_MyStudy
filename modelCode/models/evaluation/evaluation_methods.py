"""
============================================================
综合评价方法工具包 (Comprehensive Evaluation Methods)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
包含模型：AHP层次分析法、DEA数据包络法、模糊综合评价、灰色关联分析、秩和比法
作者：MCM/ICM Team
日期：2026年1月
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')

# 导入统一的绘图配置
try:
    import sys
    sys.path.append('..')
    from visualization.plot_config import PlotStyleConfig
    PlotStyleConfig.setup_style('academic')
except:
    pass


class AHP:
    """
    层次分析法 (Analytic Hierarchy Process)
    
    用于多准则决策问题，通过构建判断矩阵确定各指标权重
    
    使用方法：
        ahp = AHP(n_criteria=3)
        ahp.set_matrix(comparison_matrix)
        weights = ahp.calculate_weights()
        ahp.plot_weights()
    """
    
    # 平均随机一致性指标RI（1-15阶）
    RI_TABLE = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.54, 1.56, 1.57, 1.59]
    
    def __init__(self, n_criteria=None, criteria_names=None):
        """
        初始化AHP模型
        
        :param n_criteria: 评价指标数量
        :param criteria_names: 指标名称列表
        """
        self.n = n_criteria
        self.criteria_names = criteria_names or [f'指标{i+1}' for i in range(n_criteria)] if n_criteria else None
        self.matrix = None
        self.weights = None
        self.max_eigenvalue = None
        self.CI = None
        self.CR = None
        self.is_consistent = None
        
    def set_matrix(self, matrix):
        """
        设置判断矩阵
        
        :param matrix: n×n的判断矩阵，使用1-9标度法
        """
        self.matrix = np.array(matrix, dtype=float)
        self.n = self.matrix.shape[0]
        if self.criteria_names is None:
            self.criteria_names = [f'指标{i+1}' for i in range(self.n)]
        return self
    
    def calculate_weights(self, method='eigenvalue'):
        """
        计算权重向量
        
        :param method: 计算方法 'eigenvalue'(特征值法) / 'geometric'(几何平均法) / 'arithmetic'(算术平均法)
        :return: 归一化的权重向量
        """
        if method == 'eigenvalue':
            eigenvalues, eigenvectors = np.linalg.eig(self.matrix)
            self.max_eigenvalue = np.max(eigenvalues).real
            idx = np.argmax(eigenvalues)
            self.weights = eigenvectors[:, idx].real
            self.weights = np.abs(self.weights) / np.sum(np.abs(self.weights))
            
        elif method == 'geometric':
            # 几何平均法
            row_product = np.prod(self.matrix, axis=1)
            self.weights = np.power(row_product, 1/self.n)
            self.weights = self.weights / np.sum(self.weights)
            self.max_eigenvalue = np.sum(np.dot(self.matrix, self.weights) / self.weights) / self.n
            
        elif method == 'arithmetic':
            # 算术平均法（归一化列求和）
            col_sum = np.sum(self.matrix, axis=0)
            normalized = self.matrix / col_sum
            self.weights = np.mean(normalized, axis=1)
            self.max_eigenvalue = np.sum(np.dot(self.matrix, self.weights) / self.weights) / self.n
        
        # 一致性检验
        self.CI = (self.max_eigenvalue - self.n) / (self.n - 1)
        if self.n < len(self.RI_TABLE) and self.RI_TABLE[self.n] != 0:
            self.CR = self.CI / self.RI_TABLE[self.n]
        else:
            self.CR = 0
        self.is_consistent = self.CR < 0.1
        
        return self.weights
    
    def get_report(self):
        """生成一致性检验报告"""
        report = {
            '矩阵阶数': self.n,
            '最大特征值': round(self.max_eigenvalue.real, 4),
            '一致性指标CI': round(self.CI.real, 4),
            '一致性比率CR': round(self.CR.real, 4),
            '是否通过一致性检验': '是' if self.is_consistent else '否',
            '权重向量': dict(zip(self.criteria_names, self.weights.round(4)))
        }
        return report
    
    def plot_weights(self, figsize=(10, 6), save_path=None):
        """
        可视化权重分布
        
        :param figsize: 图形大小
        :param save_path: 保存路径（可选）
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 条形图
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, self.n))
        bars = ax1.barh(self.criteria_names, self.weights, color=colors, edgecolor='navy', alpha=0.8)
        ax1.set_xlabel('权重值', fontweight='bold')
        ax1.set_title('AHP权重分布 (条形图)', fontweight='bold')
        ax1.set_xlim(0, max(self.weights) * 1.2)
        for bar, w in zip(bars, self.weights):
            ax1.text(w + 0.01, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', fontsize=10)
        
        # 饼图
        explode = [0.05] * self.n
        ax2.pie(self.weights, labels=self.criteria_names, autopct='%1.1f%%',
                explode=explode, colors=colors, startangle=90, shadow=True)
        ax2.set_title('AHP权重分布 (饼图)', fontweight='bold')
        
        # 添加一致性检验结果
        status = '✓ 通过' if self.is_consistent else '✗ 未通过'
        fig.suptitle(f'层次分析法 (AHP) | CR = {self.CR:.4f} | 一致性检验: {status}', 
                     fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig


class DEA:
    """
    数据包络分析 (Data Envelopment Analysis) - CCR模型
    
    用于评价多投入多产出系统的相对效率
    
    使用方法：
        dea = DEA()
        dea.fit(inputs, outputs)
        dea.plot_efficiency()
    """
    
    def __init__(self, orientation='input'):
        """
        初始化DEA模型
        
        :param orientation: 导向类型 'input'(投入导向) / 'output'(产出导向)
        """
        self.orientation = orientation
        self.n_dmu = None
        self.n_inputs = None
        self.n_outputs = None
        self.efficiency = None
        self.lambda_ = None
        self.slack_input = None
        self.slack_output = None
        self.is_efficient = None
        self.dmu_names = None
        
    def fit(self, inputs, outputs, dmu_names=None):
        """
        拟合DEA-CCR模型
        
        :param inputs: 投入矩阵 (n_dmu × n_inputs)
        :param outputs: 产出矩阵 (n_dmu × n_outputs)
        :param dmu_names: 决策单元名称列表
        :return: self
        """
        X = np.array(inputs)
        Y = np.array(outputs)
        
        self.n_dmu, self.n_inputs = X.shape
        self.n_outputs = Y.shape[1]
        self.dmu_names = dmu_names or [f'DMU{i+1}' for i in range(self.n_dmu)]
        
        # 初始化结果
        self.efficiency = np.zeros(self.n_dmu)
        self.lambda_ = np.zeros((self.n_dmu, self.n_dmu))
        self.slack_input = np.zeros((self.n_dmu, self.n_inputs))
        self.slack_output = np.zeros((self.n_dmu, self.n_outputs))
        
        epsilon = 1e-6  # 非阿基米德无穷小
        
        for j in range(self.n_dmu):
            # 变量数量: 1(θ) + n_dmu(λ) + n_inputs(s⁻) + n_outputs(s⁺)
            n_vars = 1 + self.n_dmu + self.n_inputs + self.n_outputs
            
            # 目标函数: min θ - ε*(Σs⁻ + Σs⁺)
            c = np.zeros(n_vars)
            c[0] = 1  # θ系数
            c[1+self.n_dmu:] = -epsilon  # 松弛变量系数
            
            # 约束矩阵
            # 投入约束: Σ(λᵢXᵢ) + s⁻ = θXⱼ  =>  -Σ(λᵢXᵢ) - s⁻ + θXⱼ = 0
            A_eq = np.zeros((self.n_inputs + self.n_outputs, n_vars))
            b_eq = np.zeros(self.n_inputs + self.n_outputs)
            
            for m in range(self.n_inputs):
                A_eq[m, 0] = X[j, m]  # θ系数
                A_eq[m, 1:1+self.n_dmu] = -X[:, m]  # λ系数
                A_eq[m, 1+self.n_dmu+m] = -1  # s⁻系数
                b_eq[m] = 0
            
            # 产出约束: Σ(λᵢYᵢ) - s⁺ = Yⱼ
            for s in range(self.n_outputs):
                A_eq[self.n_inputs+s, 1:1+self.n_dmu] = Y[:, s]
                A_eq[self.n_inputs+s, 1+self.n_dmu+self.n_inputs+s] = -1
                b_eq[self.n_inputs+s] = Y[j, s]
            
            # 变量边界
            bounds = [(None, None)]  # θ无限制
            bounds += [(0, None)] * (self.n_dmu + self.n_inputs + self.n_outputs)  # λ, s⁻, s⁺ ≥ 0
            
            # 求解
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if result.success:
                self.efficiency[j] = result.x[0]
                self.lambda_[j, :] = result.x[1:1+self.n_dmu]
                self.slack_input[j, :] = result.x[1+self.n_dmu:1+self.n_dmu+self.n_inputs]
                self.slack_output[j, :] = result.x[1+self.n_dmu+self.n_inputs:]
        
        # 判断DEA有效性
        self.is_efficient = (np.abs(self.efficiency - 1) < 1e-4) & \
                           (np.sum(self.slack_input, axis=1) < 1e-4) & \
                           (np.sum(self.slack_output, axis=1) < 1e-4)
        
        return self
    
    def get_results(self):
        """获取结果DataFrame"""
        results = pd.DataFrame({
            '决策单元': self.dmu_names,
            '效率值': self.efficiency.round(4),
            'DEA有效': ['是' if e else '否' for e in self.is_efficient]
        })
        return results
    
    def plot_efficiency(self, figsize=(12, 5), save_path=None):
        """
        可视化效率分析结果
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 效率条形图
        colors = ['#27AE60' if e else '#C73E1D' for e in self.is_efficient]
        bars = ax1.bar(self.dmu_names, self.efficiency, color=colors, edgecolor='black', alpha=0.8)
        ax1.axhline(y=1, color='navy', linestyle='--', linewidth=2, label='效率前沿面')
        ax1.set_xlabel('决策单元', fontweight='bold')
        ax1.set_ylabel('效率值', fontweight='bold')
        ax1.set_title('DEA效率分析', fontweight='bold')
        ax1.set_ylim(0, 1.2)
        ax1.legend()
        
        for bar, eff in zip(bars, self.efficiency):
            ax1.text(bar.get_x() + bar.get_width()/2, eff + 0.02, f'{eff:.3f}', 
                    ha='center', va='bottom', fontsize=9)
        
        # 效率排名
        sorted_idx = np.argsort(-self.efficiency)
        sorted_names = [self.dmu_names[i] for i in sorted_idx]
        sorted_eff = self.efficiency[sorted_idx]
        sorted_colors = [colors[i] for i in sorted_idx]
        
        ax2.barh(sorted_names[::-1], sorted_eff[::-1], color=sorted_colors[::-1], 
                edgecolor='black', alpha=0.8)
        ax2.axvline(x=1, color='navy', linestyle='--', linewidth=2)
        ax2.set_xlabel('效率值', fontweight='bold')
        ax2.set_title('效率排名', fontweight='bold')
        ax2.set_xlim(0, 1.2)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#27AE60', label='DEA有效'),
                          Patch(facecolor='#C73E1D', label='DEA无效')]
        ax2.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig


class FuzzyComprehensiveEvaluation:
    """
    模糊综合评价法 (Fuzzy Comprehensive Evaluation)
    
    通过隶属度函数将定性问题定量化
    
    使用方法：
        fce = FuzzyComprehensiveEvaluation(n_levels=3)
        fce.fit(data, weights, criteria)
        fce.plot_evaluation()
    """
    
    def __init__(self, n_levels=3, level_names=None):
        """
        初始化模糊综合评价模型
        
        :param n_levels: 评价等级数
        :param level_names: 等级名称列表（从高到低）
        """
        self.n_levels = n_levels
        self.level_names = level_names or ['优', '良', '中', '及格', '差'][:n_levels]
        self.membership = None
        self.evaluation = None
        self.final_scores = None
        self.ranking = None
        
    def fit(self, data, weights, criteria=None, sample_names=None):
        """
        计算模糊综合评价
        
        :param data: 评价数据矩阵 (n_samples × n_criteria)
        :param weights: 指标权重向量
        :param criteria: 评价等级标准矩阵 (n_criteria × n_levels)，从高到低
        :param sample_names: 样本名称列表
        :return: self
        """
        data = np.array(data)
        weights = np.array(weights)
        
        n_samples, n_criteria = data.shape
        self.sample_names = sample_names or [f'样本{i+1}' for i in range(n_samples)]
        
        # 如果未提供评价标准，自动生成
        if criteria is None:
            criteria = np.zeros((n_criteria, self.n_levels))
            for j in range(n_criteria):
                max_val, min_val = data[:, j].max(), data[:, j].min()
                step = (max_val - min_val) / self.n_levels
                for l in range(self.n_levels):
                    criteria[j, l] = max_val - step * l
        
        criteria = np.array(criteria)
        
        # 计算隶属度矩阵
        self.membership = np.zeros((n_samples, self.n_levels, n_criteria))
        
        for i in range(n_samples):
            for j in range(n_criteria):
                x = data[i, j]
                s = criteria[j, :]
                
                for l in range(self.n_levels):
                    if l == 0:  # 最高等级
                        if x >= s[0]:
                            self.membership[i, l, j] = 1
                        elif x < s[1] if self.n_levels > 1 else x < s[0]:
                            self.membership[i, l, j] = 0
                        else:
                            self.membership[i, l, j] = (x - s[1]) / (s[0] - s[1])
                    elif l == self.n_levels - 1:  # 最低等级
                        if x <= s[l]:
                            self.membership[i, l, j] = 1
                        elif x > s[l-1]:
                            self.membership[i, l, j] = 0
                        else:
                            self.membership[i, l, j] = (s[l-1] - x) / (s[l-1] - s[l])
                    else:  # 中间等级
                        if x >= s[l-1] or x <= s[l+1]:
                            self.membership[i, l, j] = 0
                        elif x >= s[l]:
                            self.membership[i, l, j] = (s[l-1] - x) / (s[l-1] - s[l])
                        else:
                            self.membership[i, l, j] = (x - s[l+1]) / (s[l] - s[l+1])
        
        # 加权综合评价
        self.evaluation = np.zeros((n_samples, self.n_levels))
        for i in range(n_samples):
            for l in range(self.n_levels):
                self.evaluation[i, l] = np.sum(weights * self.membership[i, l, :])
        
        # 计算综合得分
        level_scores = np.arange(self.n_levels, 0, -1)
        self.final_scores = self.evaluation @ level_scores
        
        # 排名
        self.ranking = np.argsort(-self.final_scores)
        
        return self
    
    def get_results(self):
        """获取结果DataFrame"""
        results = pd.DataFrame({
            '样本': self.sample_names,
            '综合得分': self.final_scores.round(4),
            '排名': [np.where(self.ranking == i)[0][0] + 1 for i in range(len(self.sample_names))],
            '最优等级': [self.level_names[np.argmax(self.evaluation[i])] for i in range(len(self.sample_names))]
        })
        return results.sort_values('排名')
    
    def plot_evaluation(self, figsize=(14, 5), save_path=None):
        """可视化模糊综合评价结果"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # 评价矩阵热力图
        im = ax1.imshow(self.evaluation, cmap='YlOrRd', aspect='auto')
        ax1.set_xticks(range(self.n_levels))
        ax1.set_xticklabels(self.level_names)
        ax1.set_yticks(range(len(self.sample_names)))
        ax1.set_yticklabels(self.sample_names)
        ax1.set_xlabel('评价等级', fontweight='bold')
        ax1.set_ylabel('样本', fontweight='bold')
        ax1.set_title('模糊评价矩阵', fontweight='bold')
        plt.colorbar(im, ax=ax1, label='隶属度')
        
        # 综合得分条形图
        sorted_idx = self.ranking
        sorted_names = [self.sample_names[i] for i in sorted_idx]
        sorted_scores = self.final_scores[sorted_idx]
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.3, len(sorted_names)))
        
        ax2.barh(sorted_names[::-1], sorted_scores[::-1], color=colors[::-1], edgecolor='black')
        ax2.set_xlabel('综合得分', fontweight='bold')
        ax2.set_title('综合得分排名', fontweight='bold')
        
        # 雷达图/堆叠条形图
        x = np.arange(len(self.sample_names))
        width = 0.6
        bottom = np.zeros(len(self.sample_names))
        colors = plt.cm.Set2(np.linspace(0, 1, self.n_levels))
        
        for l in range(self.n_levels):
            ax3.bar(x, self.evaluation[:, l], width, label=self.level_names[l],
                   bottom=bottom, color=colors[l], edgecolor='white')
            bottom += self.evaluation[:, l]
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.sample_names, rotation=45, ha='right')
        ax3.set_ylabel('隶属度', fontweight='bold')
        ax3.set_title('等级分布', fontweight='bold')
        ax3.legend(loc='upper right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig


class GreyRelationalAnalysis:
    """
    灰色关联分析 (Grey Relational Analysis)
    
    用于多因素统计分析，分析因素间关联程度
    
    使用方法：
        gra = GreyRelationalAnalysis()
        gra.fit(data, reference)
        gra.plot_analysis()
    """
    
    def __init__(self, rho=0.5):
        """
        初始化灰色关联分析
        
        :param rho: 分辨系数，通常取0.5
        """
        self.rho = rho
        self.relational_coefficients = None
        self.relational_degrees = None
        self.ranking = None
        
    def fit(self, data, reference=None, factor_names=None, sample_names=None):
        """
        计算灰色关联度
        
        :param data: 比较序列矩阵 (n_samples × n_factors)
        :param reference: 参考序列，若为None则取最优值构建理想序列
        :param factor_names: 因素名称列表
        :param sample_names: 样本名称列表
        :return: self
        """
        data = np.array(data)
        n_samples, n_factors = data.shape
        
        self.factor_names = factor_names or [f'因素{i+1}' for i in range(n_factors)]
        self.sample_names = sample_names or [f'样本{i+1}' for i in range(n_samples)]
        
        # 构建参考序列（取每个因素的最优值）
        if reference is None:
            reference = np.max(data, axis=0)
        reference = np.array(reference)
        
        # 数据标准化（均值化）
        data_norm = data / np.mean(data, axis=0)
        ref_norm = reference / np.mean(data, axis=0)
        
        # 计算差序列
        delta = np.abs(data_norm - ref_norm)
        
        # 计算最小差和最大差
        delta_min = np.min(delta)
        delta_max = np.max(delta)
        
        # 计算关联系数
        self.relational_coefficients = (delta_min + self.rho * delta_max) / (delta + self.rho * delta_max)
        
        # 计算关联度（关联系数的均值）
        self.relational_degrees = np.mean(self.relational_coefficients, axis=1)
        
        # 排名
        self.ranking = np.argsort(-self.relational_degrees)
        
        return self
    
    def get_results(self):
        """获取结果DataFrame"""
        results = pd.DataFrame({
            '样本': self.sample_names,
            '灰色关联度': self.relational_degrees.round(4),
            '排名': [np.where(self.ranking == i)[0][0] + 1 for i in range(len(self.sample_names))]
        })
        return results.sort_values('排名')
    
    def plot_analysis(self, figsize=(12, 5), save_path=None):
        """可视化灰色关联分析结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 关联系数热力图
        im = ax1.imshow(self.relational_coefficients, cmap='Blues', aspect='auto')
        ax1.set_xticks(range(len(self.factor_names)))
        ax1.set_xticklabels(self.factor_names, rotation=45, ha='right')
        ax1.set_yticks(range(len(self.sample_names)))
        ax1.set_yticklabels(self.sample_names)
        ax1.set_xlabel('因素', fontweight='bold')
        ax1.set_ylabel('样本', fontweight='bold')
        ax1.set_title('灰色关联系数矩阵', fontweight='bold')
        plt.colorbar(im, ax=ax1, label='关联系数')
        
        # 关联度排名
        sorted_idx = self.ranking
        sorted_names = [self.sample_names[i] for i in sorted_idx]
        sorted_degrees = self.relational_degrees[sorted_idx]
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_names)))
        
        bars = ax2.barh(sorted_names[::-1], sorted_degrees[::-1], 
                       color=colors[::-1], edgecolor='navy')
        ax2.set_xlabel('灰色关联度', fontweight='bold')
        ax2.set_title('灰色关联度排名', fontweight='bold')
        ax2.set_xlim(0, 1)
        
        for bar, deg in zip(bars, sorted_degrees[::-1]):
            ax2.text(deg + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{deg:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig


# =============================================================================
# 演示示例
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("综合评价方法工具包演示")
    print("=" * 60)
    
    # 1. AHP 层次分析法演示
    print("\n【1. AHP层次分析法】")
    ahp = AHP(criteria_names=['成本', '质量', '服务'])
    matrix = [
        [1, 2, 3],
        [1/2, 1, 2],
        [1/3, 1/2, 1]
    ]
    ahp.set_matrix(matrix)
    weights = ahp.calculate_weights()
    report = ahp.get_report()
    for key, value in report.items():
        print(f"  {key}: {value}")
    ahp.plot_weights()
    
    # 2. DEA 数据包络分析演示
    print("\n【2. DEA数据包络分析】")
    inputs = np.array([
        [500, 80],
        [600, 90],
        [400, 70],
        [700, 100],
        [300, 60]
    ])
    outputs = np.array([
        [1200, 300],
        [1350, 320],
        [1000, 280],
        [1500, 350],
        [800, 220]
    ])
    dea = DEA()
    dea.fit(inputs, outputs, dmu_names=['企业A', '企业B', '企业C', '企业D', '企业E'])
    print(dea.get_results())
    dea.plot_efficiency()
    
    # 3. 模糊综合评价演示
    print("\n【3. 模糊综合评价】")
    data = np.array([
        [85, 90, 88],
        [92, 85, 90],
        [78, 92, 86],
        [88, 88, 94]
    ])
    weights = np.array([0.3, 0.4, 0.3])
    fce = FuzzyComprehensiveEvaluation(n_levels=3)
    fce.fit(data, weights, sample_names=['员工A', '员工B', '员工C', '员工D'])
    print(fce.get_results())
    fce.plot_evaluation()
    
    # 4. 灰色关联分析演示
    print("\n【4. 灰色关联分析】")
    gra = GreyRelationalAnalysis(rho=0.5)
    gra.fit(data, sample_names=['方案A', '方案B', '方案C', '方案D'],
           factor_names=['指标1', '指标2', '指标3'])
    print(gra.get_results())
    gra.plot_analysis()
    
    print("\n" + "=" * 60)
    print("演示完成！")
