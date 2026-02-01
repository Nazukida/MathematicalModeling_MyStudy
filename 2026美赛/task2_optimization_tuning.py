
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from task2_1 import EducationDecisionParams, EducationDecisionModel

def run_parameter_tuning():
    print("============================================================")
    print("🚀 模拟退火算法参数调优 (SA Hyperparameter Tuning)")
    print("============================================================")
    print("目标：寻找最大化目标函数 J 的最佳 SA 参数组合")

    # 定义参数范围
    param_grid = {
        'sa_temp': [100, 200, 500],      # 初始温度
        'sa_cooling': [0.95, 0.98, 0.99], # 冷却率
        'sa_iterations': [3000, 5000]    # 迭代次数
    }

    # 要测试的学校 (CMU是最复杂的案例，因为有Synergy Bonus)
    target_schools = ['CMU', 'CCAD', 'CIA']
    
    results = []

    # 生成所有参数组合
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"\n即将测试 {len(combinations)} 种参数组合，每种组合针对 3 所学校运行 3 次取平均值...")

    for params_dict in tqdm(combinations, desc="Grid Search Progress"):
        temp = params_dict['sa_temp']
        cooling = params_dict['sa_cooling']
        iterations = params_dict['sa_iterations']

        for school in target_schools:
            scores = []
            ai_credits = []
            
            # 每个组合运行 3 次以减少随机性影响
            for _ in range(3):
                # 初始化参数
                p = EducationDecisionParams(school_name=school)
                p.sa_temp = temp
                p.sa_cooling = cooling
                p.sa_iterations = iterations
                
                # 创建并运行模型
                model = EducationDecisionModel(p)
                res = model.curriculum_optimization_sa()
                
                scores.append(res['optimal_score'])
                ai_credits.append(res['optimal_curriculum']['x_AI'])

            avg_score = np.mean(scores)
            max_score = np.max(scores)
            avg_ai = np.mean(ai_credits)

            results.append({
                'school': school,
                'temp': temp,
                'cooling': cooling,
                'iterations': iterations,
                'avg_score': avg_score,
                'max_score': max_score,
                'avg_ai_credits': avg_ai
            })

    # 转换为 DataFrame
    df = pd.DataFrame(results)

    # 分析结果
    print("\n" + "="*70)
    print("🏆 调优结果分析")
    print("="*70)

    for school in target_schools:
        print(f"\n🏫 学校: {school}")
        school_df = df[df['school'] == school]
        
        # 找到最大化平均分数的配置
        best_config = school_df.loc[school_df['avg_score'].idxmax()]
        
        print(f"最佳参数配置:")
        print(f"  Init Temp: {best_config['temp']}")
        print(f"  Cooling Rate: {best_config['cooling']}")
        print(f"  Iterations: {best_config['iterations']}")
        print(f"  --> Max J Score: {best_config['max_score']:.4f}")
        print(f"  --> Avg J Score: {best_config['avg_score']:.4f}")
        print(f"  --> Avg AI Credits: {best_config['avg_ai_credits']:.1f}")

    # 保存具体结果到CSV
    df.sort_values(by=['school', 'avg_score'], ascending=[True, False]).to_csv('sa_tuning_results.csv', index=False)
    print("\n详细结果已保存至 'sa_tuning_results.csv'")
    
    # 简单的可视化：热力图（如果参数是二维的比较好画，这里简单打印一下Top 5）
    print("\nAttempting to visualize impact of Temperature vs Cooling (for CMU)...")
    try:
        cmu_df = df[df['school'] == 'CMU']
        # 聚合 iterations (取平均)
        pivot = cmu_df.groupby(['temp', 'cooling'])['avg_score'].mean().unstack()
        print("\nAvg Score Matrix (Temp vs Cooling) for CMU:")
        print(pivot)
    except Exception as e:
        print(f"Visualization skip: {e}")

if __name__ == "__main__":
    run_parameter_tuning()
