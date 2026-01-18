"""
============================================================
高级整数规划模型 (Advanced Integer Programming)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：整数规划、0-1规划、混合整数规划、完整可视化
特点：完备的数据预处理 + 模型求解 + 结果可视化三位一体

使用场景：
- 选址问题（设施选址、仓库布局）
- 投资决策（项目选择）
- 背包问题（资源分配）
- 排班调度
- 路径选择

作者：MCM/ICM Team
日期：2026年1月
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Callable, List, Dict, Tuple, Optional, Union
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# 尝试导入pulp，如果没有则提示安装
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("警告: 未安装pulp库，请运行 'pip install pulp' 安装")


# ============================================================
# 第一部分：图表配置
# ============================================================

class IPPlotConfig:
    """整数规划可视化配置"""
    
    COLORS = {
        'selected': '#27AE60',      # 选中项颜色
        'not_selected': '#E0E0E0',  # 未选中项颜色
        'constraint': '#E94F37',    # 约束相关颜色
        'budget': '#2E86AB',        # 预算相关颜色
        'value': '#F18F01',         # 价值相关颜色
        'grid': '#E0E0E0'
    }
    
    @staticmethod
    def setup():
        plt.style.use('seaborn-v0_8-whitegrid')
        rcParams['figure.figsize'] = (12, 8)
        rcParams['figure.dpi'] = 100
        rcParams['savefig.dpi'] = 300
        rcParams['font.size'] = 11
        rcParams['axes.titlesize'] = 14
        rcParams['axes.labelsize'] = 12
        rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False

IPPlotConfig.setup()


# ============================================================
# 第二部分：数据预处理模块
# ============================================================

class IPDataPreprocessor:
    """
    整数规划数据预处理器
    
    功能：
    1. 数据格式转换
    2. 数据验证
    3. 问题规模分析
    4. 数据汇总统计
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.processing_log = []
    
    def _log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        if self.verbose:
            print(log_entry)
    
    def load_from_dataframe(self, df: pd.DataFrame,
                            value_col: str,
                            cost_col: str,
                            name_col: Optional[str] = None) -> Dict:
        """
        从DataFrame加载问题数据
        
        :param df: 数据框
        :param value_col: 价值/收益列名
        :param cost_col: 成本/重量列名
        :param name_col: 项目名称列名
        :return: 标准化的问题数据字典
        """
        self._log("从DataFrame加载数据...")
        
        values = df[value_col].values.astype(float)
        costs = df[cost_col].values.astype(float)
        
        if name_col and name_col in df.columns:
            names = df[name_col].values.tolist()
        else:
            names = [f"项目{i+1}" for i in range(len(values))]
        
        data = {
            'n_items': len(values),
            'values': values,
            'costs': costs,
            'names': names
        }
        
        self._log(f"  加载 {data['n_items']} 个项目")
        self._log(f"  价值范围: [{values.min():.2f}, {values.max():.2f}]")
        self._log(f"  成本范围: [{costs.min():.2f}, {costs.max():.2f}]")
        
        return data
    
    def load_from_dict(self, data_dict: Dict[str, List]) -> Dict:
        """
        从字典加载问题数据
        
        :param data_dict: 格式 {'项目名': [成本, 价值], ...}
        :return: 标准化的问题数据字典
        """
        self._log("从字典加载数据...")
        
        names = list(data_dict.keys())
        costs = np.array([data_dict[n][0] for n in names])
        values = np.array([data_dict[n][1] for n in names])
        
        data = {
            'n_items': len(names),
            'values': values,
            'costs': costs,
            'names': names
        }
        
        self._log(f"  加载 {data['n_items']} 个项目")
        
        return data
    
    def validate_data(self, data: Dict) -> bool:
        """验证数据有效性"""
        self._log("验证数据...")
        
        valid = True
        
        # 检查必需字段
        required = ['values', 'costs', 'n_items']
        for field in required:
            if field not in data:
                self._log(f"  ❌ 缺少必需字段: {field}")
                valid = False
        
        if not valid:
            return False
        
        # 检查数据一致性
        if len(data['values']) != data['n_items']:
            self._log(f"  ❌ 价值数组长度不匹配")
            valid = False
        
        if len(data['costs']) != data['n_items']:
            self._log(f"  ❌ 成本数组长度不匹配")
            valid = False
        
        # 检查非负性
        if np.any(data['values'] < 0):
            self._log(f"  ⚠️ 警告: 存在负价值")
        
        if np.any(data['costs'] < 0):
            self._log(f"  ⚠️ 警告: 存在负成本")
        
        if valid:
            self._log("  ✅ 数据验证通过")
        
        return valid
    
    def summarize(self, data: Dict, budget: float) -> pd.DataFrame:
        """
        生成数据摘要
        
        :return: 摘要DataFrame
        """
        self._log("生成数据摘要...")
        
        efficiency = data['values'] / (data['costs'] + 1e-10)
        
        summary = pd.DataFrame({
            '项目': data.get('names', [f"项目{i+1}" for i in range(data['n_items'])]),
            '成本': data['costs'],
            '价值': data['values'],
            '效率(价值/成本)': efficiency,
            '占预算比例(%)': data['costs'] / budget * 100
        })
        
        summary = summary.sort_values('效率(价值/成本)', ascending=False)
        
        print("\n" + "="*60)
        print("📊 数据摘要")
        print("="*60)
        print(summary.to_string(index=False))
        print(f"\n预算总额: {budget}")
        print(f"项目总成本: {data['costs'].sum():.2f}")
        print(f"项目总价值: {data['values'].sum():.2f}")
        print("="*60)
        
        return summary
    
    def compute_upper_bound(self, data: Dict, budget: float) -> float:
        """
        计算松弛问题的上界（贪心法）
        
        用于评估求解质量
        """
        efficiency = data['values'] / (data['costs'] + 1e-10)
        sorted_idx = np.argsort(-efficiency)
        
        total_value = 0
        remaining_budget = budget
        
        for idx in sorted_idx:
            if data['costs'][idx] <= remaining_budget:
                total_value += data['values'][idx]
                remaining_budget -= data['costs'][idx]
            else:
                # 分数背包的上界
                fraction = remaining_budget / data['costs'][idx]
                total_value += fraction * data['values'][idx]
                break
        
        self._log(f"松弛上界: {total_value:.2f}")
        return total_value


# ============================================================
# 第三部分：整数规划求解器
# ============================================================

class IntegerProgrammingSolver:
    """
    整数规划求解器
    
    支持：
    1. 0-1背包问题
    2. 选址问题
    3. 自定义整数规划
    4. 混合整数规划
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.model = None
        self.result = None
    
    def solve_knapsack(self, 
                       values: np.ndarray,
                       costs: np.ndarray,
                       budget: float,
                       names: Optional[List[str]] = None,
                       item_limits: Optional[Dict[int, Tuple[int, int]]] = None) -> Dict:
        """
        求解0-1背包问题
        
        :param values: 各项目价值
        :param costs: 各项目成本
        :param budget: 总预算
        :param names: 项目名称
        :param item_limits: 项目数量限制 {项目索引: (最小数量, 最大数量)}
        :return: 求解结果
        """
        if not PULP_AVAILABLE:
            return self._solve_knapsack_dp(values, costs, budget, names)
        
        n = len(values)
        if names is None:
            names = [f"项目{i+1}" for i in range(n)]
        
        if self.verbose:
            print("\n" + "="*60)
            print("   0-1背包问题求解")
            print("="*60)
            print(f"  项目数量: {n}")
            print(f"  预算限制: {budget}")
        
        # 创建问题
        prob = pulp.LpProblem("Knapsack_Problem", pulp.LpMaximize)
        
        # 决策变量
        x = pulp.LpVariable.dicts("选择", range(n), cat=pulp.LpBinary)
        
        # 目标函数：最大化总价值
        prob += pulp.lpSum([values[i] * x[i] for i in range(n)]), "总价值"
        
        # 约束条件：总成本不超过预算
        prob += pulp.lpSum([costs[i] * x[i] for i in range(n)]) <= budget, "预算约束"
        
        # 项目数量限制
        if item_limits:
            for idx, (min_qty, max_qty) in item_limits.items():
                if min_qty > 0:
                    prob += x[idx] >= min_qty, f"项目{idx}最小选择"
                if max_qty < 1:
                    prob += x[idx] <= max_qty, f"项目{idx}最大选择"
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 整理结果
        selected = []
        selected_indices = []
        total_cost = 0
        total_value = 0
        
        for i in range(n):
            if pulp.value(x[i]) == 1:
                selected.append(names[i])
                selected_indices.append(i)
                total_cost += costs[i]
                total_value += values[i]
        
        self.result = {
            'status': pulp.LpStatus[prob.status],
            'success': prob.status == pulp.LpStatusOptimal,
            'selected_items': selected,
            'selected_indices': selected_indices,
            'total_value': total_value,
            'total_cost': total_cost,
            'remaining_budget': budget - total_cost,
            'solution_vector': [pulp.value(x[i]) for i in range(n)],
            'names': names,
            'values': values,
            'costs': costs,
            'budget': budget
        }
        
        if self.verbose:
            self._print_knapsack_result()
        
        return self.result
    
    def _solve_knapsack_dp(self, values, costs, budget, names):
        """使用动态规划求解（当pulp不可用时）"""
        n = len(values)
        if names is None:
            names = [f"项目{i+1}" for i in range(n)]
        
        # 将预算转换为整数（乘以精度因子）
        precision = 100
        W = int(budget * precision)
        weights = (costs * precision).astype(int)
        
        # DP表
        dp = np.zeros((n + 1, W + 1))
        
        for i in range(1, n + 1):
            for w in range(W + 1):
                if weights[i-1] <= w:
                    dp[i, w] = max(dp[i-1, w], 
                                   dp[i-1, w - weights[i-1]] + values[i-1])
                else:
                    dp[i, w] = dp[i-1, w]
        
        # 回溯找出选择的项目
        selected_indices = []
        w = W
        for i in range(n, 0, -1):
            if dp[i, w] != dp[i-1, w]:
                selected_indices.append(i-1)
                w -= weights[i-1]
        
        selected_indices.reverse()
        selected = [names[i] for i in selected_indices]
        total_cost = sum(costs[i] for i in selected_indices)
        total_value = sum(values[i] for i in selected_indices)
        
        self.result = {
            'status': 'Optimal',
            'success': True,
            'selected_items': selected,
            'selected_indices': selected_indices,
            'total_value': total_value,
            'total_cost': total_cost,
            'remaining_budget': budget - total_cost,
            'solution_vector': [1 if i in selected_indices else 0 for i in range(n)],
            'names': names,
            'values': values,
            'costs': costs,
            'budget': budget
        }
        
        if self.verbose:
            self._print_knapsack_result()
        
        return self.result
    
    def solve_location(self,
                       fixed_costs: np.ndarray,
                       capacities: np.ndarray,
                       demands: np.ndarray,
                       transport_costs: np.ndarray,
                       budget: Optional[float] = None,
                       max_facilities: Optional[int] = None,
                       facility_names: Optional[List[str]] = None,
                       customer_names: Optional[List[str]] = None) -> Dict:
        """
        求解设施选址问题
        
        :param fixed_costs: 各设施的固定建设成本 (n_facilities,)
        :param capacities: 各设施的容量 (n_facilities,)
        :param demands: 各客户的需求 (n_customers,)
        :param transport_costs: 运输成本矩阵 (n_facilities, n_customers)
        :param budget: 预算限制
        :param max_facilities: 最大设施数量限制
        :return: 求解结果
        """
        if not PULP_AVAILABLE:
            raise ImportError("设施选址问题需要安装pulp库: pip install pulp")
        
        n_facilities = len(fixed_costs)
        n_customers = len(demands)
        
        if facility_names is None:
            facility_names = [f"设施{i+1}" for i in range(n_facilities)]
        if customer_names is None:
            customer_names = [f"客户{j+1}" for j in range(n_customers)]
        
        if self.verbose:
            print("\n" + "="*60)
            print("   设施选址问题求解")
            print("="*60)
            print(f"  候选设施数: {n_facilities}")
            print(f"  客户数: {n_customers}")
        
        # 创建问题
        prob = pulp.LpProblem("Facility_Location", pulp.LpMinimize)
        
        # 决策变量
        y = pulp.LpVariable.dicts("开设设施", range(n_facilities), cat=pulp.LpBinary)
        x = pulp.LpVariable.dicts("分配", 
                                  ((i, j) for i in range(n_facilities) for j in range(n_customers)),
                                  lowBound=0, upBound=1, cat=pulp.LpContinuous)
        
        # 目标函数：最小化总成本
        prob += (pulp.lpSum([fixed_costs[i] * y[i] for i in range(n_facilities)]) +
                 pulp.lpSum([transport_costs[i][j] * demands[j] * x[(i, j)] 
                            for i in range(n_facilities) for j in range(n_customers)])), "总成本"
        
        # 约束条件
        # 每个客户必须被满足
        for j in range(n_customers):
            prob += pulp.lpSum([x[(i, j)] for i in range(n_facilities)]) == 1, f"需求满足_{j}"
        
        # 只能从已开设的设施供应
        for i in range(n_facilities):
            for j in range(n_customers):
                prob += x[(i, j)] <= y[i], f"开设限制_{i}_{j}"
        
        # 容量约束
        for i in range(n_facilities):
            prob += (pulp.lpSum([demands[j] * x[(i, j)] for j in range(n_customers)]) 
                    <= capacities[i] * y[i]), f"容量约束_{i}"
        
        # 预算约束
        if budget is not None:
            prob += pulp.lpSum([fixed_costs[i] * y[i] for i in range(n_facilities)]) <= budget, "预算约束"
        
        # 最大设施数量约束
        if max_facilities is not None:
            prob += pulp.lpSum([y[i] for i in range(n_facilities)]) <= max_facilities, "最大设施数"
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 整理结果
        selected_facilities = []
        assignments = {}
        total_fixed_cost = 0
        total_transport_cost = 0
        
        for i in range(n_facilities):
            if pulp.value(y[i]) == 1:
                selected_facilities.append(facility_names[i])
                total_fixed_cost += fixed_costs[i]
                assignments[facility_names[i]] = []
                for j in range(n_customers):
                    if pulp.value(x[(i, j)]) > 0.5:
                        assignments[facility_names[i]].append(customer_names[j])
                        total_transport_cost += transport_costs[i][j] * demands[j]
        
        self.result = {
            'status': pulp.LpStatus[prob.status],
            'success': prob.status == pulp.LpStatusOptimal,
            'selected_facilities': selected_facilities,
            'n_selected': len(selected_facilities),
            'assignments': assignments,
            'total_cost': pulp.value(prob.objective),
            'fixed_cost': total_fixed_cost,
            'transport_cost': total_transport_cost,
            'facility_names': facility_names,
            'customer_names': customer_names
        }
        
        if self.verbose:
            self._print_location_result()
        
        return self.result
    
    def solve_custom(self,
                     sense: str,
                     objective_coeffs: np.ndarray,
                     constraint_matrix: np.ndarray,
                     constraint_rhs: np.ndarray,
                     constraint_types: List[str],
                     var_types: List[str],
                     var_bounds: Optional[List[Tuple]] = None,
                     var_names: Optional[List[str]] = None) -> Dict:
        """
        求解自定义整数规划问题
        
        :param sense: 'max' 或 'min'
        :param objective_coeffs: 目标函数系数
        :param constraint_matrix: 约束矩阵 A
        :param constraint_rhs: 约束右端项 b
        :param constraint_types: 约束类型 ['<=', '>=', '==', ...]
        :param var_types: 变量类型 ['Binary', 'Integer', 'Continuous', ...]
        :param var_bounds: 变量边界
        :return: 求解结果
        """
        if not PULP_AVAILABLE:
            raise ImportError("自定义整数规划需要安装pulp库: pip install pulp")
        
        n_vars = len(objective_coeffs)
        n_cons = len(constraint_rhs)
        
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(n_vars)]
        
        if var_bounds is None:
            var_bounds = [(0, None) for _ in range(n_vars)]
        
        if self.verbose:
            print("\n" + "="*60)
            print("   自定义整数规划求解")
            print("="*60)
            print(f"  变量数: {n_vars}")
            print(f"  约束数: {n_cons}")
            print(f"  目标: {sense.upper()}")
        
        # 创建问题
        lp_sense = pulp.LpMaximize if sense.lower() == 'max' else pulp.LpMinimize
        prob = pulp.LpProblem("Custom_IP", lp_sense)
        
        # 创建变量
        x = {}
        for i in range(n_vars):
            lb, ub = var_bounds[i]
            vtype = var_types[i] if i < len(var_types) else 'Continuous'
            
            if vtype == 'Binary':
                x[i] = pulp.LpVariable(var_names[i], cat=pulp.LpBinary)
            elif vtype == 'Integer':
                x[i] = pulp.LpVariable(var_names[i], lowBound=lb, upBound=ub, cat=pulp.LpInteger)
            else:
                x[i] = pulp.LpVariable(var_names[i], lowBound=lb, upBound=ub, cat=pulp.LpContinuous)
        
        # 目标函数
        prob += pulp.lpSum([objective_coeffs[i] * x[i] for i in range(n_vars)]), "目标函数"
        
        # 约束条件
        for j in range(n_cons):
            lhs = pulp.lpSum([constraint_matrix[j][i] * x[i] for i in range(n_vars)])
            ctype = constraint_types[j] if j < len(constraint_types) else '<='
            
            if ctype == '<=':
                prob += lhs <= constraint_rhs[j], f"约束{j+1}"
            elif ctype == '>=':
                prob += lhs >= constraint_rhs[j], f"约束{j+1}"
            else:  # ==
                prob += lhs == constraint_rhs[j], f"约束{j+1}"
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 整理结果
        solution = {var_names[i]: pulp.value(x[i]) for i in range(n_vars)}
        
        self.result = {
            'status': pulp.LpStatus[prob.status],
            'success': prob.status == pulp.LpStatusOptimal,
            'objective_value': pulp.value(prob.objective),
            'solution': solution,
            'solution_vector': [pulp.value(x[i]) for i in range(n_vars)],
            'var_names': var_names
        }
        
        if self.verbose:
            self._print_custom_result()
        
        return self.result
    
    def _print_knapsack_result(self):
        """打印背包问题结果"""
        r = self.result
        print("\n" + "-"*50)
        print("📦 背包问题求解结果")
        print("-"*50)
        print(f"  状态: {'✅ 成功' if r['success'] else '❌ 失败'} ({r['status']})")
        print(f"\n  选中项目 ({len(r['selected_items'])}个):")
        for item in r['selected_items']:
            print(f"    - {item}")
        print(f"\n  总价值: {r['total_value']:.2f}")
        print(f"  总成本: {r['total_cost']:.2f}")
        print(f"  剩余预算: {r['remaining_budget']:.2f}")
        print("-"*50)
    
    def _print_location_result(self):
        """打印选址问题结果"""
        r = self.result
        print("\n" + "-"*50)
        print("📍 设施选址求解结果")
        print("-"*50)
        print(f"  状态: {'✅ 成功' if r['success'] else '❌ 失败'} ({r['status']})")
        print(f"\n  开设设施 ({r['n_selected']}个):")
        for fac, custs in r['assignments'].items():
            print(f"    - {fac}: 服务 {', '.join(custs)}")
        print(f"\n  总成本: {r['total_cost']:.2f}")
        print(f"    - 固定成本: {r['fixed_cost']:.2f}")
        print(f"    - 运输成本: {r['transport_cost']:.2f}")
        print("-"*50)
    
    def _print_custom_result(self):
        """打印自定义问题结果"""
        r = self.result
        print("\n" + "-"*50)
        print("🔢 整数规划求解结果")
        print("-"*50)
        print(f"  状态: {'✅ 成功' if r['success'] else '❌ 失败'} ({r['status']})")
        print(f"  目标函数值: {r['objective_value']:.4f}")
        print(f"\n  最优解:")
        for name, val in r['solution'].items():
            print(f"    {name} = {val}")
        print("-"*50)


# ============================================================
# 第四部分：可视化模块
# ============================================================

class IPVisualizer:
    """
    整数规划可视化器
    
    功能：
    1. 项目选择柱状图
    2. 资源利用饼图
    3. 选址地图
    4. 结果汇总图
    """
    
    def __init__(self, save_dir: str = './figures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_knapsack_selection(self, result: Dict, save_name: Optional[str] = None):
        """绘制背包问题选择结果"""
        names = result['names']
        values = result['values']
        costs = result['costs']
        selected = result['solution_vector']
        
        n = len(names)
        colors = [IPPlotConfig.COLORS['selected'] if s == 1 
                  else IPPlotConfig.COLORS['not_selected'] for s in selected]
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # 选择状态
        bars = axes[0].barh(range(n), [1]*n, color=colors, edgecolor='white')
        axes[0].set_yticks(range(n))
        axes[0].set_yticklabels(names)
        axes[0].set_xlabel('选择状态')
        axes[0].set_title('项目选择状态', fontweight='bold')
        axes[0].set_xlim(0, 1.2)
        for i, s in enumerate(selected):
            axes[0].text(0.5, i, '✓ 选中' if s == 1 else '✗ 未选', 
                        ha='center', va='center', fontweight='bold',
                        color='white' if s == 1 else 'gray')
        
        # 价值对比
        axes[1].barh(range(n), values, color=[colors[i] for i in range(n)], 
                    edgecolor='white', alpha=0.8)
        axes[1].set_yticks(range(n))
        axes[1].set_yticklabels(names)
        axes[1].set_xlabel('价值')
        axes[1].set_title('各项目价值', fontweight='bold')
        for i, v in enumerate(values):
            axes[1].text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=9)
        
        # 成本对比
        axes[2].barh(range(n), costs, color=[colors[i] for i in range(n)],
                    edgecolor='white', alpha=0.8)
        axes[2].set_yticks(range(n))
        axes[2].set_yticklabels(names)
        axes[2].set_xlabel('成本')
        axes[2].set_title('各项目成本', fontweight='bold')
        axes[2].axvline(x=result['budget'], color='red', linestyle='--', 
                       linewidth=2, label=f"预算上限: {result['budget']}")
        axes[2].legend()
        for i, c in enumerate(costs):
            axes[2].text(c + 0.5, i, f'{c:.1f}', va='center', fontsize=9)
        
        plt.tight_layout()
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_resource_usage(self, result: Dict, save_name: Optional[str] = None):
        """绘制资源使用饼图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 预算使用
        used = result['total_cost']
        remaining = result['remaining_budget']
        
        axes[0].pie([used, remaining], 
                   labels=[f'已使用\n{used:.1f}', f'剩余\n{remaining:.1f}'],
                   colors=[IPPlotConfig.COLORS['budget'], IPPlotConfig.COLORS['not_selected']],
                   autopct='%1.1f%%', startangle=90, explode=[0.05, 0])
        axes[0].set_title('预算使用情况', fontweight='bold', fontsize=14)
        
        # 价值获取（相对于全选）
        obtained = result['total_value']
        total_possible = sum(result['values'])
        not_obtained = total_possible - obtained
        
        axes[1].pie([obtained, not_obtained],
                   labels=[f'已获取\n{obtained:.1f}', f'未获取\n{not_obtained:.1f}'],
                   colors=[IPPlotConfig.COLORS['value'], IPPlotConfig.COLORS['not_selected']],
                   autopct='%1.1f%%', startangle=90, explode=[0.05, 0])
        axes[1].set_title('价值获取情况', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_efficiency_analysis(self, result: Dict, save_name: Optional[str] = None):
        """绘制效率分析图"""
        names = result['names']
        values = np.array(result['values'])
        costs = np.array(result['costs'])
        selected = result['solution_vector']
        
        efficiency = values / (costs + 1e-10)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 按效率排序
        sorted_idx = np.argsort(-efficiency)
        
        colors = [IPPlotConfig.COLORS['selected'] if selected[i] == 1 
                  else IPPlotConfig.COLORS['not_selected'] for i in sorted_idx]
        
        bars = ax.bar(range(len(names)), efficiency[sorted_idx], color=colors, edgecolor='white')
        
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([names[i] for i in sorted_idx], rotation=45, ha='right')
        ax.set_ylabel('效率 (价值/成本)', fontweight='bold')
        ax.set_title('项目效率分析（按效率降序）', fontweight='bold', fontsize=14)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=IPPlotConfig.COLORS['selected'], label='已选择'),
            Patch(facecolor=IPPlotConfig.COLORS['not_selected'], label='未选择')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_solution_summary(self, result: Dict, save_name: Optional[str] = None):
        """绘制结果汇总"""
        fig = plt.figure(figsize=(14, 8))
        
        # 创建不规则网格布局
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 左上：项目选择
        ax1 = fig.add_subplot(gs[0, :2])
        names = result['names']
        selected = result['solution_vector']
        colors = [IPPlotConfig.COLORS['selected'] if s == 1 
                  else IPPlotConfig.COLORS['not_selected'] for s in selected]
        ax1.barh(range(len(names)), result['values'], color=colors, edgecolor='white')
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names)
        ax1.set_xlabel('价值')
        ax1.set_title('项目选择与价值', fontweight='bold')
        
        # 右上：预算饼图
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.pie([result['total_cost'], result['remaining_budget']],
               labels=['已用', '剩余'],
               colors=[IPPlotConfig.COLORS['budget'], IPPlotConfig.COLORS['not_selected']],
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('预算使用', fontweight='bold')
        
        # 下方：统计信息
        ax3 = fig.add_subplot(gs[1, :])
        info_text = f"""
┌─────────────────────────────────────────────────────────────┐
│                    🎯 整数规划求解结果汇总                     │
├─────────────────────────────────────────────────────────────┤
│  求解状态: {'✅ 最优解' if result['success'] else '❌ 未找到最优解'}
│  
│  选中项目: {len(result['selected_items'])} / {len(names)} 个
│  选中项目列表: {', '.join(result['selected_items'][:5])}{'...' if len(result['selected_items']) > 5 else ''}
│  
│  总价值: {result['total_value']:.2f}
│  总成本: {result['total_cost']:.2f}
│  预算余额: {result['remaining_budget']:.2f}
│  预算利用率: {result['total_cost']/result['budget']*100:.1f}%
└─────────────────────────────────────────────────────────────┘
        """
        ax3.text(0.5, 0.5, info_text, fontsize=11, family='monospace',
                verticalalignment='center', horizontalalignment='center',
                transform=ax3.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax3.axis('off')
        
        plt.tight_layout()
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 第五部分：完整工作流
# ============================================================

class IntegerProgrammingPipeline:
    """
    整数规划完整工作流
    
    集成数据预处理、模型求解、结果可视化
    """
    
    def __init__(self, verbose: bool = True, save_dir: str = './figures'):
        self.preprocessor = IPDataPreprocessor(verbose)
        self.solver = IntegerProgrammingSolver(verbose)
        self.visualizer = IPVisualizer(save_dir)
        self.verbose = verbose
    
    def run_knapsack(self,
                     data: Union[Dict, pd.DataFrame],
                     budget: float,
                     value_col: str = 'value',
                     cost_col: str = 'cost',
                     name_col: Optional[str] = None,
                     plot_selection: bool = True,
                     plot_usage: bool = True,
                     plot_efficiency: bool = True,
                     plot_summary: bool = True) -> Dict:
        """
        执行完整的背包问题求解流程
        """
        if self.verbose:
            print("\n" + "="*60)
            print("   整数规划（背包问题）完整工作流")
            print("="*60)
        
        # 数据预处理
        if isinstance(data, pd.DataFrame):
            problem_data = self.preprocessor.load_from_dataframe(data, value_col, cost_col, name_col)
        elif isinstance(data, dict):
            if 'values' in data and 'costs' in data:
                problem_data = data
            else:
                problem_data = self.preprocessor.load_from_dict(data)
        else:
            raise ValueError("数据格式不支持，请使用DataFrame或Dict")
        
        # 数据验证
        self.preprocessor.validate_data(problem_data)
        
        # 数据摘要
        self.preprocessor.summarize(problem_data, budget)
        
        # 计算上界
        upper_bound = self.preprocessor.compute_upper_bound(problem_data, budget)
        
        # 求解
        result = self.solver.solve_knapsack(
            problem_data['values'],
            problem_data['costs'],
            budget,
            problem_data.get('names')
        )
        
        # 可视化
        if plot_selection:
            self.visualizer.plot_knapsack_selection(result)
        if plot_usage:
            self.visualizer.plot_resource_usage(result)
        if plot_efficiency:
            self.visualizer.plot_efficiency_analysis(result)
        if plot_summary:
            self.visualizer.plot_solution_summary(result)
        
        # 添加额外信息
        result['upper_bound'] = upper_bound
        result['gap'] = (upper_bound - result['total_value']) / upper_bound * 100 if upper_bound > 0 else 0
        
        return result


# ============================================================
# 示例：投资项目选择问题
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("   示例：投资项目选择问题")
    print("="*70)
    
    # 项目数据
    projects = pd.DataFrame({
        '项目名称': ['AI研发', '市场拓展', '设备升级', '人才培训', '品牌建设', '供应链优化'],
        '投资成本(万元)': [150, 80, 120, 50, 90, 110],
        '预期收益(万元)': [200, 100, 160, 70, 130, 140]
    })
    
    budget = 300  # 总预算300万元
    
    # 创建工作流
    pipeline = IntegerProgrammingPipeline(verbose=True)
    
    # 求解
    result = pipeline.run_knapsack(
        data=projects,
        budget=budget,
        value_col='预期收益(万元)',
        cost_col='投资成本(万元)',
        name_col='项目名称'
    )
    
    print("\n" + "="*50)
    print("📊 最终决策")
    print("="*50)
    print(f"在预算 {budget} 万元的限制下：")
    print(f"选择项目: {', '.join(result['selected_items'])}")
    print(f"总投资: {result['total_cost']:.0f} 万元")
    print(f"预期收益: {result['total_value']:.0f} 万元")
    print(f"投资回报率: {(result['total_value']/result['total_cost']-1)*100:.1f}%")
    print("="*50)
