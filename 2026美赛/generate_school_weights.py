import pandas as pd
import numpy as np

# 读取CIA课程统计数据
df_stats = pd.read_csv("cia_course_stats.csv")

# 定义学校
schools = ['CMU', 'CIA', 'CCAD']

# 定义课程类型权重变化 (AI影响前后的权重变化)
weight_changes = {
    'Core Technical Foundation': {'pre_ai': 0.4, 'post_ai': 0.35},  # 技术基础略微减少
    'AI-Augmented & Applied': {'pre_ai': 0.05, 'post_ai': 0.25},    # AI相关大幅增加
    'High-Order Human Value': {'pre_ai': 0.25, 'post_ai': 0.25},    # 保持稳定
    'Cross-disciplinary & Elasticity': {'pre_ai': 0.3, 'post_ai': 0.15}  # 跨界课程减少
}

# 为每个学校生成模拟数据
all_school_data = []

for school in schools:
    school_data = []
    for _, row in df_stats.iterrows():
        category = row['Category']
        total_credits = row['Total Credits']

        # 基于学校特点调整权重
        if school == 'CMU':  # 技术导向 (Carnegie Mellon)
            base_multiplier = 1.2 if category == 'Core Technical Foundation' else 0.8
        elif school == 'CIA':  # 烹饪艺术 (Culinary Institute of America)
            base_multiplier = 1.3 if category == 'High-Order Human Value' else 0.9
        elif school == 'CCAD':  # 艺术设计 (Columbus College of Art & Design)
            base_multiplier = 1.1 if category in ['High-Order Human Value', 'Cross-disciplinary & Elasticity'] else 0.85

        # 生成AI前后的权重
        pre_ai_weight = weight_changes[category]['pre_ai'] * base_multiplier
        post_ai_weight = weight_changes[category]['post_ai'] * base_multiplier

        # 归一化权重
        total_pre = sum(weight_changes[cat]['pre_ai'] * (1.2 if cat == 'Core Technical Foundation' and school == 'CMU' else
                                                        1.3 if cat == 'High-Order Human Value' and school == 'CIA' else
                                                        1.1 if cat in ['High-Order Human Value', 'Cross-disciplinary & Elasticity'] and school == 'CCAD' else 0.8)
                        for cat in weight_changes.keys())
        total_post = sum(weight_changes[cat]['post_ai'] * (1.2 if cat == 'Core Technical Foundation' and school == 'CMU' else
                                                          1.3 if cat == 'High-Order Human Value' and school == 'CIA' else
                                                          1.1 if cat in ['High-Order Human Value', 'Cross-disciplinary & Elasticity'] and school == 'CCAD' else 0.8)
                         for cat in weight_changes.keys())

        pre_ai_weight /= total_pre
        post_ai_weight /= total_post

        school_data.append({
            'School': school,
            'Category': category,
            'Total Credits': total_credits,
            'Pre-AI Weight': round(pre_ai_weight, 3),
            'Post-AI Weight': round(post_ai_weight, 3),
            'Weight Change': round(post_ai_weight - pre_ai_weight, 3)
        })

    all_school_data.extend(school_data)

# 保存结果
df_schools = pd.DataFrame(all_school_data)
df_schools.to_csv("school_course_weights.csv", index=False, encoding='utf-8-sig')

print("学校课程权重分布生成完成！")
print("结果保存至: school_course_weights.csv")

# 显示结果摘要
print("\n各学校课程权重分布摘要:")
for school in schools:
    school_df = df_schools[df_schools['School'] == school]
    print(f"\n{school}:")
    for _, row in school_df.iterrows():
        print(f"  {row['Category']}: {row['Pre-AI Weight']} → {row['Post-AI Weight']} (变化: {row['Weight Change']:+.3f})")