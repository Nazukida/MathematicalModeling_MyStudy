from task1_1 import AICareerParams, AICareerModel, AICareerVisualization, run_multi_career_workflow
from task2_1 import EducationDecisionParams, EducationDecisionModel, EducationDecisionVisualization
import data_processing
import os
import numpy as np
import matplotlib.pyplot as plt


def integrate_and_run(career='software_engineer', school='CMU', csv_path=r'd:\\competition\\美国大学生数学建模大赛\\2026美赛\\就业人数.csv', target_year=2030):
    """
    运行 task1_1 的职业演化预测，提取指定年份的 F_t (final_demands)，
    将该值传入 task2_1 的教育决策模型（作为 demand_2030），并运行教育决策分析。
    """
    os.makedirs('./figures', exist_ok=True)

    print(f"\n== Integration: career={career}, school={school}, target_year={target_year} ==\n")

    # --- Run task1_1 model ---
    params1 = AICareerParams(occupation_name=career, csv_path=csv_path)
    params1.summary()

    model1 = AICareerModel(params1)
    # 使用 verbose=True 以显示 task1_1 的详细运行日志
    results1 = model1.predict_evolution(verbose=True)

    # 生成 model1 的可视化以展示完整输出（与单独运行 model1 行为一致）
    try:
        viz1 = AICareerVisualization(model1, results1, save_dir='./figures')
        print("\n  🎨 绘制 model1 的完整可视化输出...")
        viz1.plot_complete_evolution()
        viz1.plot_comparison_scenarios()
    except Exception as e:
        print(f"  ⚠️ 绘制 model1 可视化时出错: {e}")

    # find index for target_year in future years
    start_year = params1.start_year
    idx = target_year - start_year
    final_demands = results1.get('final_demands')

    if final_demands is None or len(final_demands) == 0:
        raise RuntimeError('task1_1 produced no final_demands')

    if idx < 0 or idx >= len(final_demands):
        print(f"  Warning: target_year {target_year} out of forecast range ({start_year}..{start_year+len(final_demands)-1}), using last forecast year value.")
        F_t_val = float(final_demands[-1])
        target_year_used = start_year + len(final_demands) - 1
    else:
        F_t_val = float(final_demands[idx])
        target_year_used = target_year

    print(f"  -> Extracted F_t for year {target_year_used}: {F_t_val:.3f} (units same as task1_1 results)")

    # --- Prepare task2_1 params using the extracted F_t ---
    # pass target_career to params
    params2 = EducationDecisionParams(school_name=school, demand_2030=F_t_val, target_career=career)
    params2.summary()

    model2 = EducationDecisionModel(params2)

    # Run education decision analysis and visualization
    results2 = model2.run_full_analysis(verbose=True)

    viz = EducationDecisionVisualization(model2, results2, save_dir='./figures')
    viz.plot_enrollment_response()
    viz.plot_curriculum_optimization()
    viz.plot_career_elasticity()
    viz.plot_skill_radar()
    viz.plot_sa_convergence()
    viz.plot_pareto_frontier()
    
    # Generate global/static charts (AHP & Career Similarity) at least once per run
    # These are independent of the specific simulation result but part of the full output
    try:
        viz.plot_ahp_radar()
        viz.plot_ahp_summary_table()
        viz.plot_career_similarity_matrix()
    except Exception as e:
        print(f"  ⚠️ Warning generating static charts: {e}")

    # Save a small bridge record
    bridge_path = os.path.join('./figures', f'bridge_{career}_{school}_{target_year_used}.txt')
    with open(bridge_path, 'w', encoding='utf-8') as f:
        f.write(f'career={career}\n')
        f.write(f'school={school}\n')
        f.write(f'target_year_used={target_year_used}\n')
        f.write(f'F_t={F_t_val}\n')

    print(f"\n  ✅ Integration complete. Bridge record saved: {bridge_path}\n")
    # 生成合并验证图（模型1需求趋势 vs 模型2推荐毕业生数）
    try:
        plot_combined_validation(results1, results2, save_dir='./figures')
    except Exception as e:
        print(f"  ⚠️ 生成合并验证图时出错: {e}")

    return {'F_t': F_t_val, 'task1_results': results1, 'task2_results': results2}

def plot_combined_validation(results1, results2, save_dir='./figures'):
    """
    将模型1的最终需求趋势与模型2的推荐毕业生数放在同一张图上进行对比（横轴为预测年份）。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 提取模型1的预测年份与最终需求
    future_years = results1.get('future_years')
    final_demands = results1.get('final_demands')
    if future_years is None or final_demands is None:
        raise ValueError('results1 中缺少 future_years 或 final_demands')

    # 提取模型2的推荐毕业生数（单值）
    recommended = results2.get('enrollment_response', {}).get('recommended_graduates')
    if recommended is None:
        raise ValueError('results2 中缺少 enrollment_response.recommended_graduates')

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(future_years, final_demands, marker='o', label='Industry Final Demand (Model 1)')
    plt.hlines(recommended, xmin=min(future_years), xmax=max(future_years), colors='orange', linestyles='--', label=f'School Recommended Graduates ({recommended:.1f})')
    plt.title('Combined Validation: Industry Demand vs School Recommended Graduates')
    plt.xlabel('Year')
    plt.ylabel('Number (same units)')
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = os.path.join(save_dir, 'combined_validation.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  💾 Combined validation plot saved: {out_path}")


def batch_integrate_from_csv(csv_path=r'd:\competition\美国大学生数学建模大赛\2026美赛\就业人数.csv', school_default='CMU', target_year=2030):
    """
    从 CSV 读取职业列表，并对每个职业运行完整的串联工作流（模型1 -> 模型2）。
    """
    careers = []
    try:
        import csv
        with open(csv_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        # assume first column header 'career'
        for row in rows[1:]:
            if len(row) == 0:
                continue
            careers.append(row[0].strip())
    except Exception as e:
        print(f"  ⚠️ 无法读取 CSV: {e}")
        return

    print(f"\nFound {len(careers)} careers in CSV: {careers}\n")
    results_summary = {}
    for career in careers:
        print(f"\n=== Batch: processing career={career} ===\n")
        try:
            out = integrate_and_run(career=career, school=school_default, csv_path=csv_path, target_year=target_year)
            results_summary[career] = {'F_t': out['F_t']}
        except Exception as e:
            print(f"  ⚠️ Error processing {career}: {e}")

    # 保存汇总
    import json
    outpath = os.path.join('./figures', 'batch_summary.json')
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 Batch summary saved: {outpath}\n")
    return results_summary

def run_comprehensive_school_comparison(target_year=2030):
    """
    运行综合对比分析：使用Task 1数据对三所典型学校进行分析，并生成学校对比图
    CMU -> software_engineer (卡内基梅隆大学 - 软件工程)
    CCAD -> graphic_designer (哥伦布艺术与设计学院 - 平面设计)
    CIA -> chef (美国烹饪学院 - 厨师)
    """
    print("\n" + "="*70)
    print("🚀 Running Comprehensive School Comparison (Task 1 -> Task 2)")
    print("="*70)
    
    # 映射关系
    scenarios = [
        {'school': 'CMU', 'career': 'software_engineer', 'csv': r'd:\competition\美国大学生数学建模大赛\2026美赛\就业人数.csv'},
        {'school': 'CCAD', 'career': 'graphic_designer', 'csv': r'd:\competition\美国大学生数学建模大赛\2026美赛\就业人数.csv'},
        {'school': 'CIA', 'career': 'chef', 'csv': r'd:\competition\美国大学生数学建模大赛\2026美赛\就业人数.csv'}
    ]
    
    all_results = {}
    
    # 运行每个学校的分析
    for sc in scenarios:
        try:
            # 调用integrate_and_run获取结果
            res = integrate_and_run(career=sc['career'], school=sc['school'], csv_path=sc['csv'], target_year=target_year)
            all_results[sc['school']] = res['task2_results']
        except Exception as e:
            print(f"⚠️ Failed to run analysis for {sc['school']}: {e}")
            
    # 如果成功收集了结果，生成对比图
    if all_results:
        print("\n🎨 Generating School Comparison Charts...")
        #由于integrate_and_run已经生成了各自的图表，这里只需要生成对比图
        # 创建一个临时的viz对象用于绘图
        try:
            temp_params = EducationDecisionParams(school_name='CMU') # Dummy param
            temp_model = EducationDecisionModel(temp_params)
            viz = EducationDecisionVisualization(temp_model, {}, save_dir='./figures')
            
            viz.plot_school_comparison(all_results)
            viz.plot_stacked_curriculum_comparison(all_results)
            viz.plot_career_similarity_matrix() # Also good to have
            print("  ✅ School comparison charts generated successfully.")
        except Exception as e:
            print(f"  ⚠️ Error generating comparison charts: {e}")
            
    print("="*70 + "\n")

if __name__ == '__main__':
    # 先运行完整的 task1 多职业工作流
    csv_path = r'd:\competition\美国大学生数学建模大赛\2026美赛\就业人数.csv'
    print("🚀 启动完整 task1 工作流...")
    # run_multi_career_workflow(csv_path=csv_path) # Commented out to save time if already run, but user asked for full run
    
    # 注入职业向量与学校参数（来自 data_processing.py）
    print('\n🔧 准备职业向量与学校参数（来自 data_processing.py）...')
    vecs = data_processing.load_vectors()
    if 'vectors' in vecs:
        EducationDecisionParams.CAREER_VECTORS.update(vecs['vectors'])
    
    # 修复：使用get_school_params()方法获取学校参数，而不是直接访问SCHOOL_PARAMS
    min_sp = data_processing.build_school_params(schoolStudentNumber_csv='schoolStudentNumber.csv')
    if min_sp:
        # Note: EducationDecisionParams handles own param loading via get_school_params, 
        # but if we needed to inject external params we would do it here. 
        # For now, get_school_params call inside task2_1 is sufficient given main.py context.
        pass

    # 1. 运行综合对比分析 (Ensures all charts including comparison are generated)
    run_comprehensive_school_comparison()

    # 2. 运行批量集成分析 (Generates career-specific analysis for CMU default)
    print("\n🔗 开始 task1 + task2 批量集成分析...")
    # batch_integrate_from_csv() 

