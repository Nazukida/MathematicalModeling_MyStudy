"""
可视化模块 (Visualization Module)
=================================

包含统一的图表配置和可视化模板。

模块结构：
- plot_config.py: 图表样式配置、配色方案、保存工具

使用方法：
    from visualization.plot_config import PlotStyleConfig, FigureSaver, PlotTemplates
    
    PlotStyleConfig.setup_style()  # 初始化样式
    colors = PlotStyleConfig.get_palette(5)  # 获取配色
"""

from .plot_config import PlotStyleConfig, FigureSaver, PlotTemplates

__all__ = [
    'PlotStyleConfig',
    'FigureSaver',
    'PlotTemplates'
]
