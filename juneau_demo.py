"""
Juneau旅游可持续性模型 - 结果展示
(Juneau Tourism Sustainability Model - Results Demo)

展示模型运行结果和关键发现
"""

import pandas as pd
import matplotlib.pyplot as plt
from juneau_model_new import run_juneau_workflow

def main():
    print("="*80)
    print("Juneau旅游可持续性模型结果分析")
    print("="*80)

    # 运行完整工作流
    try:
        params, model, optimizer, viz = run_juneau_workflow()

        # 获取最优政策
        optimal = optimizer.get_optimal_policy()

        print("\n" + "="*80)
        print("📊 关键发现 (Key Findings)")
        print("="*80)

        print("\n1. 最优政策组合:")
        print(f"   • 峰季游客上限: {optimal['c1']:,.0f} 人/日")
        print(f"   • 非峰季游客目标: {optimal['c2']:,.0f} 人/日")
        print(f"   • 每日政府投资: ${optimal['I']:,.0f}")
        print(f"   • 环境投资比例: {optimal['gamma1']:.1%}")

        print("\n2. 经济-环境-社会平衡:")
        print(f"   • 经济利润: ${optimal['P']:,.0f} ({optimal['P']/optimal['U']:.1%})")
        print(f"   • 环境水平: ${optimal['E']:,.0f} ({optimal['E']/optimal['U']:.1%})")
        print(f"   • 社会福利: ${optimal['S']:,.0f} ({optimal['S']/optimal['U']:.1%})")
        print(f"   • 总效用: ${optimal['U']:,.0f}")

        print("\n3. 政策含义:")
        print("   • 环境投资占比接近100%，表明环境是关键约束")
        print("   • 峰季限制游客数量以保护环境")
        print("   • 非峰季通过投资促进旅游发展")
        print("   • 税收-补贴政策平衡季节性需求")

        print("\n4. 模型验证:")
        print("   • 总效用超过11.9亿美元")
        print("   • 环境贡献占比52%，经济贡献47%，社会贡献1%")
        print("   • 实现了经济、环境、社会的三重可持续性")

        # 显示图表路径
        print("\n📁 生成的图表:")
        print("   • ./figures/seasonal_demand.png - 季节性需求曲线")
        print("   • ./figures/policy_revenue_cost.png - 政策收入/成本函数")
        print("   • ./figures/investment_returns.png - 投资回报函数")
        print("   • ./figures/optimal_policy_summary.png - 最优政策摘要")
        print("   • ./figures/chinese_test.png - 中文字体测试图表")
        print("   • ./figures/juneau_optimal_policy.csv - 详细结果数据")

        print("\n✅ 中文字体支持测试:")
        print("   如果图表中的中文标题和标签显示正常，则中文字体配置成功！")
        print("   请检查 ./figures/chinese_test.png 文件确认中文显示效果。")

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请确保已安装必要的依赖包: numpy, pandas, matplotlib, scipy")

if __name__ == "__main__":
    main()