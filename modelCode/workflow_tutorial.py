"""
============================================================
工作流使用指南 - 如何串联各模块
============================================================
这份指南告诉你：
1. 什么时候用工作流，什么时候直接用模块
2. 如何一步步构建你自己的工作流
3. 如何把你现有的模型接入工作流
4. 常见问题的解决模板

核心思路：
    原始数据 → [预处理] → 干净数据 → [模型] → 结果 → [可视化] → 图表
============================================================
"""

# =================================================================
# 第一部分：你真的需要工作流吗？
# =================================================================
"""
【何时使用工作流】
✅ 数据需要多步预处理（缺失值、异常值、标准化）
✅ 想要自动生成可视化结果
✅ 需要对比不同参数下的模型效果
✅ 想要保存完整的处理历史

【何时直接用模块】
❌ 数据已经很干净，不需要预处理
❌ 只是快速测试一个想法
❌ 模型逻辑非常简单

简单来说：如果你的代码超过 50 行，考虑用工作流；否则直接写就行。
"""


# =================================================================
# 第二部分：工作流的核心概念（只有4个）
# =================================================================
"""
1. PipelineData（数据容器）
   - 包装你的数据，让它能在各模块间传递
   - 自动记录处理历史
   
2. PreprocessingStep（预处理步骤）
   - 对数据做某种处理
   - 可以链式调用多个步骤
   
3. ModelAdapter（模型适配器）
   - 把你的模型包装成统一接口
   - 关键方法：run(pipeline_data) -> pipeline_data
   
4. VisualizationStep（可视化步骤）
   - 根据模型结果画图
   - 关键方法：plot(pipeline_data) -> fig
"""


# =================================================================
# 第三部分：实际使用步骤
# =================================================================

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -------------------- 步骤1：导入你需要的组件 --------------------

from workflow.model_validation_pipeline import (
    # 核心类
    ModelValidationPipeline,  # 主工作流
    PipelineData,             # 数据容器
    
    # 预处理步骤（按需选择）
    MissingValueStep,         # 缺失值处理
    OutlierRemovalStep,       # 异常值处理
    NormalizationStep,        # 标准化
    
    # 模型适配器（按需选择）
    ModelAdapter,             # 基类，用于自定义
    DynamicProgrammingAdapter,# 动态规划
    OptimizationAdapter,      # 优化算法
    
    # 可视化（按需选择）
    VisualizationStep,        # 基类，用于自定义
    DPTableVisualization,
    ConvergenceVisualization,
    DataComparisonVisualization,
)


# -------------------- 步骤2：准备你的数据 --------------------

# 你的数据可以是以下任意格式：
my_data_list = [[1, 2], [3, 4], [5, 6]]           # 列表
my_data_array = np.array([[1, 2], [3, 4]])        # numpy数组
my_data_dict = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}  # 字典
my_data_df = pd.DataFrame({'重量': [2, 3, 4], '价值': [10, 20, 30]})  # DataFrame（推荐）


# -------------------- 步骤3：根据你的需求选择组件 --------------------

"""
【预处理步骤选择指南】

数据有缺失值？
    → MissingValueStep('mean')      # 用均值填充
    → MissingValueStep('median')    # 用中位数填充
    → MissingValueStep('knn')       # 用KNN插补
    → MissingValueStep('drop')      # 删除缺失行

数据有异常值？
    → OutlierRemovalStep('iqr', 1.5)    # IQR方法，1.5倍四分位距
    → OutlierRemovalStep('zscore', 3)   # Z-score方法，3个标准差

需要标准化？
    → NormalizationStep('zscore')   # Z-score标准化
    → NormalizationStep('minmax')   # Min-Max归一化到[0,1]
    → NormalizationStep('robust')   # 稳健标准化（对异常值不敏感）
"""


# -------------------- 步骤4：构建并运行工作流 --------------------

def example_basic_usage():
    """基础用法示例"""
    
    # 1. 创建工作流
    pipeline = ModelValidationPipeline("我的分析任务")
    
    # 2. 加载数据
    data = pd.DataFrame({
        '重量': [2, 2, 6, 5, 4],
        '价值': [6, 3, 5, 4, 6]
    })
    pipeline.load_data(data, "物品数据")
    
    # 3. 添加预处理（可选，可以不加）
    # pipeline.add_preprocessing(MissingValueStep('mean'))
    
    # 4. 设置模型
    pipeline.set_model(DynamicProgrammingAdapter())
    pipeline.configure_model(capacity=10)  # 设置参数
    
    # 5. 添加可视化（可选）
    pipeline.add_visualization(DPTableVisualization())
    
    # 6. 运行
    pipeline.run()
    
    # 7. 获取结果
    result = pipeline.get_model_result()
    print(f"最大价值: {result['max_value']}")
    print(f"选择的物品: {result['selected_items']}")
    
    return result


# =================================================================
# 第四部分：如何接入你自己的模型（最重要！）
# =================================================================

"""
你有一个现成的模型代码，想接入工作流？只需要3步：

1. 继承 ModelAdapter
2. 在 __init__ 中定义参数
3. 在 run() 中实现你的逻辑
"""

class MyCustomModelAdapter(ModelAdapter):
    """
    模板：把你的模型接入工作流
    
    复制这个类，修改成你自己的模型
    """
    
    def __init__(self):
        super().__init__("我的模型名称")  # 给模型起个名字
        
        # 定义你的模型需要的参数
        self.params = {
            'param1': 100,
            'param2': 0.5,
            'param3': 'default'
        }
    
    def run(self, pipeline_data):
        """
        这里写你的模型逻辑
        
        输入：pipeline_data（数据容器）
        输出：pipeline_data（带有结果的数据容器）
        """
        
        # ===== 获取数据（选择你需要的格式）=====
        data_df = pipeline_data.get_dataframe()    # pandas DataFrame
        data_np = pipeline_data.get_array()        # numpy array
        data_list = pipeline_data.get_list()       # Python list
        
        # ===== 获取参数 =====
        param1 = self.params['param1']
        param2 = self.params['param2']
        
        # ===== 你的模型逻辑 =====
        # 
        # 把你原来的模型代码放在这里
        # 例如：
        #   result = your_original_model(data_np, param1, param2)
        #
        # 这里用一个简单示例：
        result_value = np.sum(data_np) * param2
        
        # ===== 保存结果 =====
        self.result = {
            'output_value': result_value,
            'input_shape': data_np.shape,
            # 添加你想保存的任何结果...
        }
        
        # ===== 设置输出（重要！）=====
        pipeline_data.set_model_output(self.result, "my_model_type")
        pipeline_data._log(self.name, f"计算完成，结果: {result_value}")
        
        return pipeline_data


# 【使用你的自定义模型】
def example_custom_model():
    """使用自定义模型的示例"""
    
    pipeline = ModelValidationPipeline("自定义模型测试")
    pipeline.load_data([[1, 2], [3, 4], [5, 6]], "测试数据")
    
    # 使用你的自定义模型
    pipeline.set_model(MyCustomModelAdapter())
    pipeline.configure_model(param1=200, param2=0.8)  # 修改参数
    
    pipeline.run()
    
    result = pipeline.get_model_result()
    print(f"自定义模型结果: {result}")
    
    return result


# =================================================================
# 第五部分：如何添加自定义可视化
# =================================================================

import matplotlib.pyplot as plt

class MyCustomVisualization(VisualizationStep):
    """
    模板：自定义可视化
    """
    
    def __init__(self):
        super().__init__("我的图表")
    
    def plot(self, pipeline_data):
        """
        这里写你的绑图代码
        """
        
        # 获取数据或模型结果
        data = pipeline_data.get_dataframe()
        model_output = pipeline_data.model_output  # 可能是 None
        
        # 创建图表
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # 你的绑图逻辑
        if data is not None and len(data.columns) >= 2:
            self.ax.scatter(data.iloc[:, 0], data.iloc[:, 1], 
                          s=100, c='#2E86AB', edgecolors='white')
            self.ax.set_xlabel(data.columns[0])
            self.ax.set_ylabel(data.columns[1])
        
        self.ax.set_title('我的自定义图表', fontweight='bold')
        
        return self.fig


# =================================================================
# 第六部分：常见场景模板
# =================================================================

def template_data_cleaning_only():
    """
    场景1：只做数据清洗，不需要模型
    """
    pipeline = ModelValidationPipeline("数据清洗")
    
    # 假设你的原始数据有问题
    dirty_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 100],  # 有缺失和异常
        'B': [10, np.nan, 30, 40, 50]
    })
    
    pipeline.load_data(dirty_data, "脏数据")
    pipeline.add_preprocessing(MissingValueStep('mean'))
    pipeline.add_preprocessing(OutlierRemovalStep('iqr', 1.5))
    
    # 不设置模型，直接运行
    pipeline.run()
    
    # 获取清洗后的数据
    clean_data = pipeline.get_processed_data()
    print("清洗后的数据:")
    print(clean_data)
    
    return clean_data


def template_optimization_problem():
    """
    场景2：优化问题（求函数最小值）
    """
    # 定义你的目标函数
    def my_objective(x):
        """你要最小化的函数"""
        return x[0]**2 + x[1]**2 + 10*np.sin(x[0])
    
    pipeline = ModelValidationPipeline("优化问题")
    pipeline.pipeline_data = PipelineData(name="优化")  # 优化可能不需要外部数据
    
    model = OptimizationAdapter('pso')
    model.set_objective(my_objective)
    model.set_params(
        bounds=(-10, 10),  # 搜索范围
        n_dims=2,          # 变量维度
        max_iter=100,      # 迭代次数
        n_particles=30     # 粒子数
    )
    
    pipeline.set_model(model)
    pipeline.add_visualization(ConvergenceVisualization())
    pipeline.run()
    
    result = pipeline.get_model_result()
    print(f"最优解: {result['best_position']}")
    print(f"最优值: {result['best_value']}")
    
    return result


def template_compare_preprocessing():
    """
    场景3：对比不同预处理方法的效果
    """
    data = pd.DataFrame({
        'X': [1, 2, 3, 100, 5],  # 有异常值
        'Y': [10, 20, 30, 40, 50]
    })
    
    results = {}
    
    for method in ['iqr', 'zscore']:
        pipeline = ModelValidationPipeline(f"对比-{method}")
        pipeline.load_data(data.copy(), "数据")
        pipeline.add_preprocessing(OutlierRemovalStep(method, threshold=1.5 if method=='iqr' else 2))
        pipeline.run()
        
        results[method] = pipeline.get_processed_data()
        print(f"\n{method} 方法结果:")
        print(results[method])
    
    return results


# =================================================================
# 第七部分：快速问答
# =================================================================

"""
Q: 我不想用预处理，可以跳过吗？
A: 可以，直接不调用 add_preprocessing() 即可。

Q: 我不想可视化，可以跳过吗？
A: 可以，直接不调用 add_visualization() 即可。

Q: 我的模型需要多个输入怎么办？
A: 在 ModelAdapter.run() 中，你可以从 pipeline_data 获取所有数据，
   然后在你的模型逻辑中自己拆分使用。

Q: 我想保存中间结果怎么办？
A: pipeline_data.metadata['my_key'] = my_value  # 随时保存
   later = pipeline_data.metadata['my_key']      # 随时读取

Q: 如何在工作流外单独使用预处理？
A: 
   step = MissingValueStep('mean')
   data = PipelineData(your_data)
   data = step.apply(data)
   clean_data = data.get_dataframe()

Q: 如何只获取处理后的数据，不运行模型？
A:
   pipeline = ModelValidationPipeline("清洗")
   pipeline.load_data(data)
   pipeline.add_preprocessing(...)
   pipeline.run()  # 没设置模型也能运行
   clean = pipeline.get_processed_data()
"""


# =================================================================
# 运行示例
# =================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("工作流使用指南 - 运行示例")
    print("="*60)
    
    print("\n【示例1: 基础用法】")
    example_basic_usage()
    
    print("\n【示例2: 自定义模型】")
    example_custom_model()
    
    print("\n【示例3: 只做数据清洗】")
    template_data_cleaning_only()
    
    print("\n【示例4: 优化问题】")
    template_optimization_problem()
    
    print("\n" + "="*60)
    print("✅ 所有示例运行完成!")
    print("="*60)
    
    plt.show()

