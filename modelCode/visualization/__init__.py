"""
可视化模块 (Visualization Module)
=================================

包含统一的图表配置和可视化模板。

模块结构：
- plot_config.py: 图表样式配置、配色方案、保存工具
- geo_visualization.py: 地理可视化工具（地图、热力图、路径图等）

使用方法：
    # 基础图表配置
    from visualization.plot_config import PlotStyleConfig, FigureSaver, PlotTemplates
    
    PlotStyleConfig.setup_style()  # 初始化样式
    colors = PlotStyleConfig.get_palette(5)  # 获取配色
    
    # 地图可视化
    from visualization.geo_visualization import (
        FoliumMapVisualizer,      # 交互式地图（Folium）
        StaticMapVisualizer,      # 静态地图（Matplotlib）
        PlotlyMapVisualizer,      # 交互式地图（Plotly）
        quick_marker_map,         # 快速标记点地图
        quick_heatmap,            # 快速热力图
        quick_route_map           # 快速路径地图
    )
"""

from .plot_config import PlotStyleConfig, FigureSaver, PlotTemplates

# 地图可视化模块（延迟导入，避免依赖问题）
try:
    from .geo_visualization import (
        FoliumMapVisualizer,
        StaticMapVisualizer,
        PlotlyMapVisualizer,
        quick_marker_map,
        quick_heatmap,
        quick_route_map
    )
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False

__all__ = [
    # 基础配置
    'PlotStyleConfig',
    'FigureSaver',
    'PlotTemplates',
    # 地图可视化
    'FoliumMapVisualizer',
    'StaticMapVisualizer',
    'PlotlyMapVisualizer',
    'quick_marker_map',
    'quick_heatmap',
    'quick_route_map'
]
