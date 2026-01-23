"""
============================================================
地理可视化工具 (Geographic Visualization Tools)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：地图绑定数据可视化、热力图、路径规划、区域填充等
作者：MCM/ICM Team
日期：2026年1月
============================================================

【依赖库安装说明】
基础地图可视化：
    pip install folium           # 交互式地图
    pip install geopandas        # 地理数据处理
    pip install shapely          # 几何操作

高级可视化（可选）：
    pip install plotly           # 交互式图表
    pip install cartopy          # 地图投影
    pip install contextily       # 底图瓦片
    pip install pyproj           # 坐标转换

============================================================
【模块功能概览】

1. FoliumMapVisualizer - 基于Folium的交互式地图
   - create_marker_map()      : 标记点地图（显示多个地点位置）
   - create_heatmap()         : 热力图（密度可视化）
   - create_choropleth()      : 分级统计图（区域着色）
   - create_route_map()       : 路径/轨迹地图
   - create_cluster_map()     : 聚类标记地图（大量点位）

2. StaticMapVisualizer - 基于Matplotlib的静态地图
   - plot_scatter_map()       : 散点地图
   - plot_bubble_map()        : 气泡地图（大小表示数值）
   - plot_connection_map()    : 连线地图（OD流向图）
   - plot_choropleth_static() : 静态分级统计图

3. PlotlyMapVisualizer - 基于Plotly的交互式地图
   - create_scatter_mapbox()  : Mapbox散点图
   - create_density_mapbox()  : Mapbox密度图
   - create_choropleth_mapbox(): Mapbox分级图

============================================================
【快速使用示例】

示例1: 创建标记点地图
>>> from visualization.geo_visualization import FoliumMapVisualizer
>>> viz = FoliumMapVisualizer()
>>> locations = [(39.9042, 116.4074, '北京'), (31.2304, 121.4737, '上海')]
>>> map_obj = viz.create_marker_map(locations, center=[35, 105], zoom=4)
>>> map_obj.save('china_cities.html')

示例2: 创建热力图
>>> data_points = [(lat1, lon1, weight1), (lat2, lon2, weight2), ...]
>>> heatmap = viz.create_heatmap(data_points, center=[35, 105])
>>> heatmap.save('heatmap.html')

示例3: 创建路径地图
>>> route = [(lat1, lon1), (lat2, lon2), (lat3, lon3)]
>>> route_map = viz.create_route_map([route], center=[35, 105])
>>> route_map.save('route.html')

============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings

# 导入本模块配色
try:
    from .plot_config import PlotStyleConfig
except ImportError:
    from plot_config import PlotStyleConfig


class FoliumMapVisualizer:
    """
    基于Folium的交互式地图可视化类
    
    Folium特点：
    - 生成交互式HTML地图
    - 支持多种底图（OpenStreetMap, Stamen, CartoDB等）
    - 支持标记、热力图、分级统计图等
    - 可嵌入Jupyter Notebook
    
    使用方法：
        viz = FoliumMapVisualizer()
        map_obj = viz.create_marker_map(locations)
        map_obj.save('map.html')  # 保存为HTML
        map_obj  # 在Jupyter中直接显示
    """
    
    # 预设底图样式
    TILE_PROVIDERS = {
        'default': 'OpenStreetMap',
        'satellite': 'Esri.WorldImagery',
        'terrain': 'Stamen Terrain',
        'toner': 'Stamen Toner',
        'watercolor': 'Stamen Watercolor',
        'cartodb_light': 'CartoDB positron',
        'cartodb_dark': 'CartoDB dark_matter'
    }
    
    # 标记颜色
    MARKER_COLORS = ['blue', 'red', 'green', 'purple', 'orange', 
                     'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
                     'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue',
                     'lightgreen', 'gray', 'black', 'lightgray']
    
    def __init__(self, default_location=[39.9, 116.4], default_zoom=10):
        """
        初始化Folium地图可视化器
        
        参数:
            default_location: 默认地图中心 [纬度, 经度]
            default_zoom: 默认缩放级别 (1-18, 数值越大越详细)
        
        缩放级别参考：
            - 1-4: 洲/国家级
            - 5-7: 省/大区域
            - 8-10: 城市级
            - 11-14: 街区级
            - 15-18: 街道/建筑级
        """
        self.default_location = default_location
        self.default_zoom = default_zoom
        
    def _import_folium(self):
        """延迟导入folium"""
        try:
            import folium
            from folium import plugins
            return folium, plugins
        except ImportError:
            raise ImportError(
                "folium库未安装，请运行: pip install folium\n"
                "如需热力图功能，还需安装: pip install folium[plugins]"
            )
    
    def create_marker_map(self, locations, center=None, zoom=None, 
                          tile='default', popup_on_click=True,
                          cluster=False, custom_icons=None):
        """
        创建标记点地图
        
        【适用场景】
        - 显示多个地点位置（如商店、学校、医院分布）
        - 标注感兴趣的位置点
        - POI（兴趣点）可视化
        
        参数:
            locations: 位置数据列表，每个元素为:
                       - (lat, lon) 仅坐标
                       - (lat, lon, label) 坐标+标签
                       - (lat, lon, label, popup) 坐标+标签+弹窗内容
            center: 地图中心 [lat, lon]，默认自动计算
            zoom: 缩放级别
            tile: 底图样式 ('default', 'satellite', 'terrain', 等)
            popup_on_click: 点击时显示信息弹窗
            cluster: 是否启用标记聚合（大量点位时推荐）
            custom_icons: 自定义图标配置字典
        
        返回:
            folium.Map 对象
        
        示例:
            >>> viz = FoliumMapVisualizer()
            >>> # 简单标记
            >>> locs = [(39.9, 116.4), (31.2, 121.5)]
            >>> m = viz.create_marker_map(locs)
            >>> 
            >>> # 带标签的标记
            >>> locs = [
            ...     (39.9042, 116.4074, '北京'),
            ...     (31.2304, 121.4737, '上海'),
            ...     (23.1291, 113.2644, '广州')
            ... ]
            >>> m = viz.create_marker_map(locs, center=[35, 110], zoom=4)
            >>> m.save('cities.html')
        """
        folium, plugins = self._import_folium()
        
        # 解析位置数据
        parsed_locs = []
        for loc in locations:
            if len(loc) == 2:
                parsed_locs.append({'lat': loc[0], 'lon': loc[1], 
                                   'label': '', 'popup': ''})
            elif len(loc) == 3:
                parsed_locs.append({'lat': loc[0], 'lon': loc[1], 
                                   'label': str(loc[2]), 'popup': str(loc[2])})
            else:
                parsed_locs.append({'lat': loc[0], 'lon': loc[1], 
                                   'label': str(loc[2]), 'popup': str(loc[3])})
        
        # 计算中心点
        if center is None:
            center = [
                np.mean([loc['lat'] for loc in parsed_locs]),
                np.mean([loc['lon'] for loc in parsed_locs])
            ]
        
        # 获取底图
        tile_name = self.TILE_PROVIDERS.get(tile, 'OpenStreetMap')
        
        # 创建地图
        m = folium.Map(
            location=center,
            zoom_start=zoom or self.default_zoom,
            tiles=tile_name
        )
        
        # 添加标记
        if cluster:
            # 使用标记聚合
            marker_cluster = plugins.MarkerCluster()
            for i, loc in enumerate(parsed_locs):
                color = self.MARKER_COLORS[i % len(self.MARKER_COLORS)]
                marker = folium.Marker(
                    location=[loc['lat'], loc['lon']],
                    popup=loc['popup'] if popup_on_click else None,
                    tooltip=loc['label'],
                    icon=folium.Icon(color=color)
                )
                marker_cluster.add_child(marker)
            m.add_child(marker_cluster)
        else:
            for i, loc in enumerate(parsed_locs):
                color = self.MARKER_COLORS[i % len(self.MARKER_COLORS)]
                folium.Marker(
                    location=[loc['lat'], loc['lon']],
                    popup=loc['popup'] if popup_on_click else None,
                    tooltip=loc['label'],
                    icon=folium.Icon(color=color)
                ).add_to(m)
        
        return m
    
    def create_heatmap(self, data_points, center=None, zoom=None,
                      radius=15, blur=10, max_zoom=18,
                      gradient=None, tile='default'):
        """
        创建热力图
        
        【适用场景】
        - 人口/事件密度可视化
        - 犯罪热点分析
        - 疾病传播热点
        - 客流/交通流量密度
        
        参数:
            data_points: 数据点列表，每个元素为:
                        - (lat, lon) 仅坐标，权重默认为1
                        - (lat, lon, weight) 带权重
            center: 地图中心
            zoom: 缩放级别
            radius: 热力点半径 (像素)
            blur: 模糊程度
            max_zoom: 最大缩放级别
            gradient: 渐变色配置，如 {0.4: 'blue', 0.65: 'lime', 1: 'red'}
            tile: 底图样式
        
        返回:
            folium.Map 对象
        
        示例:
            >>> viz = FoliumMapVisualizer()
            >>> # 简单热力图
            >>> points = [(39.9, 116.4), (39.91, 116.41), (39.92, 116.42)]
            >>> heatmap = viz.create_heatmap(points)
            >>>
            >>> # 带权重的热力图
            >>> points = [
            ...     (39.9042, 116.4074, 100),  # 北京，权重100
            ...     (31.2304, 121.4737, 80),   # 上海，权重80
            ...     (23.1291, 113.2644, 60),   # 广州，权重60
            ... ]
            >>> heatmap = viz.create_heatmap(points, center=[35, 110], zoom=4)
            >>> heatmap.save('population_heatmap.html')
        """
        folium, plugins = self._import_folium()
        from folium.plugins import HeatMap
        
        # 解析数据
        heat_data = []
        for point in data_points:
            if len(point) == 2:
                heat_data.append([point[0], point[1], 1])
            else:
                heat_data.append([point[0], point[1], point[2]])
        
        # 计算中心
        if center is None:
            center = [
                np.mean([p[0] for p in heat_data]),
                np.mean([p[1] for p in heat_data])
            ]
        
        # 创建地图
        tile_name = self.TILE_PROVIDERS.get(tile, 'OpenStreetMap')
        m = folium.Map(
            location=center,
            zoom_start=zoom or self.default_zoom,
            tiles=tile_name
        )
        
        # 默认渐变色
        if gradient is None:
            gradient = {0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 
                       0.8: 'yellow', 1: 'red'}
        
        # 添加热力图层
        HeatMap(
            heat_data,
            radius=radius,
            blur=blur,
            max_zoom=max_zoom,
            gradient=gradient
        ).add_to(m)
        
        return m
    
    def create_choropleth(self, geo_json, data, columns, key_on,
                         center=None, zoom=None, 
                         fill_color='YlOrRd', fill_opacity=0.7,
                         line_opacity=0.3, legend_name='数值',
                         tile='default'):
        """
        创建分级统计图（Choropleth Map）
        
        【适用场景】
        - 各省/州/国家数据对比（如GDP、人口、感染率）
        - 选举结果地图
        - 区域销售额分布
        - 任何按行政区划划分的统计数据
        
        参数:
            geo_json: GeoJSON文件路径或GeoJSON对象
                     （可从 https://geojson.io 获取或下载行政区划数据）
            data: pandas DataFrame，包含要可视化的数据
            columns: [key_column, value_column]，如 ['省份', 'GDP']
            key_on: GeoJSON中用于匹配的属性，如 'feature.properties.name'
            center: 地图中心
            zoom: 缩放级别
            fill_color: 填充色方案 
                       ('BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn',
                        'PuRd', 'RdPu', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd')
            fill_opacity: 填充透明度
            line_opacity: 边界线透明度
            legend_name: 图例名称
            tile: 底图样式
        
        返回:
            folium.Map 对象
        
        示例:
            >>> import pandas as pd
            >>> from visualization.geo_visualization import FoliumMapVisualizer
            >>> 
            >>> # 准备数据
            >>> data = pd.DataFrame({
            ...     'state': ['California', 'Texas', 'Florida'],
            ...     'population': [39.5, 29.0, 21.5]
            ... })
            >>> 
            >>> # 创建分级统计图
            >>> viz = FoliumMapVisualizer()
            >>> m = viz.create_choropleth(
            ...     geo_json='us-states.json',  # 美国各州GeoJSON
            ...     data=data,
            ...     columns=['state', 'population'],
            ...     key_on='feature.properties.name',
            ...     legend_name='Population (M)'
            ... )
            >>> m.save('us_population.html')
        
        【获取GeoJSON数据】
        - 美国各州: https://raw.githubusercontent.com/python-visualization/folium/master/tests/us-states.json
        - 中国省份: 搜索 "china province geojson"
        - 世界各国: https://geojson-maps.ash.ms/
        """
        folium, plugins = self._import_folium()
        
        # 创建地图
        tile_name = self.TILE_PROVIDERS.get(tile, 'OpenStreetMap')
        m = folium.Map(
            location=center or self.default_location,
            zoom_start=zoom or self.default_zoom,
            tiles=tile_name
        )
        
        # 添加Choropleth层
        folium.Choropleth(
            geo_data=geo_json,
            data=data,
            columns=columns,
            key_on=key_on,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            line_opacity=line_opacity,
            legend_name=legend_name
        ).add_to(m)
        
        return m
    
    def create_route_map(self, routes, center=None, zoom=None,
                        colors=None, weights=None, opacity=0.8,
                        add_markers=True, tile='default'):
        """
        创建路径/轨迹地图
        
        【适用场景】
        - 物流配送路线规划
        - 旅行轨迹可视化
        - 车辆GPS轨迹
        - 台风/飓风路径
        - 航线/航班路线
        
        参数:
            routes: 路径列表，每条路径为坐标点列表
                   [[(lat1, lon1), (lat2, lon2), ...], [...], ...]
            center: 地图中心
            zoom: 缩放级别
            colors: 每条路径的颜色列表
            weights: 线宽列表
            opacity: 透明度
            add_markers: 是否添加起点和终点标记
            tile: 底图样式
        
        返回:
            folium.Map 对象
        
        示例:
            >>> viz = FoliumMapVisualizer()
            >>> # 单条路径
            >>> route1 = [
            ...     (39.9042, 116.4074),  # 北京
            ...     (34.3416, 108.9398),  # 西安
            ...     (30.5728, 104.0668),  # 成都
            ... ]
            >>> m = viz.create_route_map([route1], center=[35, 110], zoom=5)
            >>>
            >>> # 多条路径
            >>> route2 = [(39.9, 116.4), (31.2, 121.5)]  # 北京-上海
            >>> m = viz.create_route_map([route1, route2], 
            ...                          colors=['blue', 'red'],
            ...                          weights=[5, 3])
            >>> m.save('routes.html')
        """
        folium, plugins = self._import_folium()
        
        # 计算所有点的中心
        all_points = [point for route in routes for point in route]
        if center is None:
            center = [
                np.mean([p[0] for p in all_points]),
                np.mean([p[1] for p in all_points])
            ]
        
        # 创建地图
        tile_name = self.TILE_PROVIDERS.get(tile, 'OpenStreetMap')
        m = folium.Map(
            location=center,
            zoom_start=zoom or self.default_zoom,
            tiles=tile_name
        )
        
        # 默认颜色
        if colors is None:
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#27AE60']
        
        # 默认线宽
        if weights is None:
            weights = [4] * len(routes)
        
        # 绘制路径
        for i, route in enumerate(routes):
            color = colors[i % len(colors)]
            weight = weights[i] if i < len(weights) else 4
            
            # 绘制折线
            folium.PolyLine(
                locations=route,
                color=color,
                weight=weight,
                opacity=opacity
            ).add_to(m)
            
            # 添加起点终点标记
            if add_markers and len(route) >= 2:
                # 起点 - 绿色
                folium.Marker(
                    location=route[0],
                    popup=f'路径{i+1} 起点',
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(m)
                # 终点 - 红色
                folium.Marker(
                    location=route[-1],
                    popup=f'路径{i+1} 终点',
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(m)
        
        return m
    
    def create_cluster_map(self, locations, center=None, zoom=None,
                          tile='default'):
        """
        创建聚类标记地图
        
        【适用场景】
        - 大量POI点展示（如全国门店分布）
        - 避免标记重叠
        - 支持缩放时自动聚合/展开
        
        参数:
            locations: 位置列表 [(lat, lon), ...] 或 [(lat, lon, popup), ...]
            center: 地图中心
            zoom: 缩放级别
            tile: 底图样式
        
        返回:
            folium.Map 对象
        
        示例:
            >>> viz = FoliumMapVisualizer()
            >>> # 生成大量随机点
            >>> import numpy as np
            >>> locs = [(39.9 + np.random.randn()*0.1, 
            ...          116.4 + np.random.randn()*0.1) for _ in range(100)]
            >>> m = viz.create_cluster_map(locs)
            >>> m.save('cluster.html')
        """
        folium, plugins = self._import_folium()
        from folium.plugins import MarkerCluster
        
        # 计算中心
        if center is None:
            center = [
                np.mean([loc[0] for loc in locations]),
                np.mean([loc[1] for loc in locations])
            ]
        
        # 创建地图
        tile_name = self.TILE_PROVIDERS.get(tile, 'OpenStreetMap')
        m = folium.Map(
            location=center,
            zoom_start=zoom or self.default_zoom,
            tiles=tile_name
        )
        
        # 创建聚类层
        marker_cluster = MarkerCluster()
        
        for loc in locations:
            if len(loc) >= 3:
                popup = str(loc[2])
            else:
                popup = f'{loc[0]:.4f}, {loc[1]:.4f}'
            
            folium.Marker(
                location=[loc[0], loc[1]],
                popup=popup
            ).add_to(marker_cluster)
        
        marker_cluster.add_to(m)
        
        return m


class StaticMapVisualizer:
    """
    基于Matplotlib的静态地图可视化类
    
    特点：
    - 生成静态图片（PNG/PDF/SVG）
    - 适合论文插图
    - 可高度自定义
    - 需要GeoPandas和Cartopy支持完整功能
    
    使用方法：
        viz = StaticMapVisualizer()
        fig = viz.plot_scatter_map(gdf, value_col='population')
        fig.savefig('map.png', dpi=300)
    """
    
    def __init__(self):
        """初始化静态地图可视化器"""
        pass
    
    def _check_geopandas(self):
        """检查GeoPandas是否可用"""
        try:
            import geopandas as gpd
            return gpd
        except ImportError:
            raise ImportError(
                "GeoPandas未安装，请运行:\n"
                "pip install geopandas\n"
                "conda install -c conda-forge geopandas  # 或使用conda"
            )
    
    def plot_scatter_map(self, gdf, value_col=None, 
                        color_col=None, cmap='viridis',
                        size=50, alpha=0.7, 
                        title='散点地图', figsize=(12, 8),
                        add_basemap=False, show_colorbar=True):
        """
        绑定在地图上的散点图
        
        【适用场景】
        - 点状数据的地理分布（如地震震中、气象站点）
        - 配合颜色/大小表示属性值
        
        参数:
            gdf: GeoDataFrame，包含geometry列（Point类型）
            value_col: 用于着色的数值列名
            color_col: 用于分类着色的列名（与value_col二选一）
            cmap: 颜色映射
            size: 点大小（可以是数值或列名）
            alpha: 透明度
            title: 标题
            figsize: 图片大小
            add_basemap: 是否添加底图（需要contextily）
            show_colorbar: 是否显示颜色条
        
        返回:
            (fig, ax) matplotlib对象
        
        示例:
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> 
            >>> # 创建测试数据
            >>> data = {
            ...     'city': ['北京', '上海', '广州'],
            ...     'population': [21.5, 24.2, 15.3],
            ...     'geometry': [Point(116.4, 39.9), Point(121.5, 31.2), Point(113.3, 23.1)]
            ... }
            >>> gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
            >>> 
            >>> viz = StaticMapVisualizer()
            >>> fig, ax = viz.plot_scatter_map(gdf, value_col='population',
            ...                                 title='中国主要城市人口分布')
            >>> fig.savefig('cities.png', dpi=300)
        """
        gpd = self._check_geopandas()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绑定散点
        if value_col is not None:
            gdf.plot(ax=ax, column=value_col, cmap=cmap, 
                    markersize=size, alpha=alpha, legend=show_colorbar)
        elif color_col is not None:
            gdf.plot(ax=ax, column=color_col, categorical=True,
                    markersize=size, alpha=alpha, legend=True)
        else:
            gdf.plot(ax=ax, color=PlotStyleConfig.COLORS['primary'],
                    markersize=size, alpha=alpha)
        
        # 添加底图
        if add_basemap:
            try:
                import contextily as ctx
                ctx.add_basemap(ax, crs=gdf.crs.to_string(), 
                               source=ctx.providers.OpenStreetMap.Mapnik)
            except ImportError:
                warnings.warn("contextily未安装，跳过底图")
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_bubble_map(self, gdf, size_col, color_col=None,
                       cmap='YlOrRd', scale_factor=1000,
                       alpha=0.6, title='气泡地图', figsize=(12, 8),
                       add_basemap=False, legend_title='数值'):
        """
        气泡地图（大小表示数值）
        
        【适用场景】
        - 城市人口规模对比
        - 销售额地理分布
        - 任何需要用大小表示数量的场景
        
        参数:
            gdf: GeoDataFrame
            size_col: 控制气泡大小的列名
            color_col: 控制颜色的列名（可选）
            scale_factor: 气泡大小缩放因子
            alpha: 透明度
            title: 标题
            figsize: 图片大小
            add_basemap: 是否添加底图
            legend_title: 图例标题
        
        返回:
            (fig, ax) matplotlib对象
        
        示例:
            >>> viz = StaticMapVisualizer()
            >>> fig, ax = viz.plot_bubble_map(gdf, size_col='gdp',
            ...                                color_col='growth_rate',
            ...                                title='各省GDP及增长率')
        """
        gpd = self._check_geopandas()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算气泡大小
        sizes = gdf[size_col] / gdf[size_col].max() * scale_factor
        
        # 获取坐标
        x = gdf.geometry.x
        y = gdf.geometry.y
        
        # 绑定颜色
        if color_col is not None:
            scatter = ax.scatter(x, y, s=sizes, c=gdf[color_col], 
                                cmap=cmap, alpha=alpha, edgecolors='white', linewidth=1)
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label(legend_title)
        else:
            ax.scatter(x, y, s=sizes, color=PlotStyleConfig.COLORS['primary'],
                      alpha=alpha, edgecolors='white', linewidth=1)
        
        # 添加底图
        if add_basemap:
            try:
                import contextily as ctx
                ctx.add_basemap(ax, crs=gdf.crs.to_string())
            except ImportError:
                warnings.warn("contextily未安装，跳过底图")
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_connection_map(self, origins, destinations, weights=None,
                           base_gdf=None, cmap='Blues', 
                           linewidth_range=(0.5, 5),
                           alpha=0.5, title='OD连线图', figsize=(12, 8),
                           arrow=False):
        """
        OD连线图（起点-终点流向图）
        
        【适用场景】
        - 人口迁移流向
        - 货物运输流向
        - 航班/铁路连接
        - 贸易往来关系
        
        参数:
            origins: 起点坐标列表 [(lon1, lat1), ...]
            destinations: 终点坐标列表 [(lon2, lat2), ...]
            weights: 流量权重列表（控制线宽）
            base_gdf: 底图GeoDataFrame（如省份边界）
            cmap: 颜色映射
            linewidth_range: 线宽范围 (min, max)
            alpha: 透明度
            title: 标题
            figsize: 图片大小
            arrow: 是否显示箭头
        
        返回:
            (fig, ax) matplotlib对象
        
        示例:
            >>> viz = StaticMapVisualizer()
            >>> origins = [(116.4, 39.9), (121.5, 31.2)]  # 北京、上海
            >>> dests = [(104.1, 30.7), (113.3, 23.1)]    # 成都、广州
            >>> weights = [1000, 500]  # 流量
            >>> fig, ax = viz.plot_connection_map(origins, dests, weights,
            ...                                    title='城市间人口流动')
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绑定底图
        if base_gdf is not None:
            base_gdf.plot(ax=ax, color='lightgray', edgecolor='white')
        
        # 归一化权重
        if weights is None:
            weights = [1] * len(origins)
        weights = np.array(weights)
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
        linewidths = norm_weights * (linewidth_range[1] - linewidth_range[0]) + linewidth_range[0]
        
        # 获取颜色映射
        cmap_obj = plt.cm.get_cmap(cmap)
        
        # 绘制连线
        for i, (orig, dest) in enumerate(zip(origins, destinations)):
            color = cmap_obj(norm_weights[i])
            lw = linewidths[i]
            
            if arrow:
                ax.annotate('', xy=dest, xytext=orig,
                           arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=alpha))
            else:
                ax.plot([orig[0], dest[0]], [orig[1], dest[1]], 
                       color=color, linewidth=lw, alpha=alpha)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_choropleth_static(self, gdf, value_col, cmap='YlOrRd',
                              edgecolor='white', linewidth=0.5,
                              title='分级统计图', figsize=(12, 8),
                              legend_title='数值', scheme='quantiles',
                              k=5):
        """
        静态分级统计图
        
        【适用场景】
        - 各省/州/国家对比（用于论文插图）
        - 区域差异可视化
        - 无需交互的简洁展示
        
        参数:
            gdf: GeoDataFrame，包含Polygon/MultiPolygon geometry
            value_col: 着色值列名
            cmap: 颜色映射
            edgecolor: 边界颜色
            linewidth: 边界线宽
            title: 标题
            figsize: 图片大小
            legend_title: 图例标题
            scheme: 分级方案 
                    'quantiles' - 分位数
                    'equal_interval' - 等间隔
                    'fisher_jenks' - Fisher-Jenks自然断点
                    'natural_breaks' - 自然断点
            k: 分级数量
        
        返回:
            (fig, ax) matplotlib对象
        
        示例:
            >>> import geopandas as gpd
            >>> # 加载省份边界
            >>> china = gpd.read_file('china_provinces.shp')
            >>> china['gdp'] = [...]  # 添加GDP数据
            >>> 
            >>> viz = StaticMapVisualizer()
            >>> fig, ax = viz.plot_choropleth_static(china, 'gdp',
            ...                                       title='中国各省GDP分布',
            ...                                       scheme='quantiles')
            >>> fig.savefig('china_gdp.png', dpi=300)
        """
        gpd = self._check_geopandas()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绑定分级统计图
        try:
            import mapclassify
            gdf.plot(column=value_col, ax=ax, cmap=cmap,
                    edgecolor=edgecolor, linewidth=linewidth,
                    legend=True, scheme=scheme, k=k,
                    legend_kwds={'title': legend_title, 'loc': 'lower right'})
        except ImportError:
            # 无mapclassify时使用简单分级
            gdf.plot(column=value_col, ax=ax, cmap=cmap,
                    edgecolor=edgecolor, linewidth=linewidth,
                    legend=True, legend_kwds={'label': legend_title})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')  # 隐藏坐标轴（地图通常不需要）
        
        plt.tight_layout()
        return fig, ax


class PlotlyMapVisualizer:
    """
    基于Plotly的交互式地图可视化类
    
    特点：
    - 高度交互（缩放、平移、悬停信息）
    - 支持Mapbox底图（需要token）
    - 可嵌入Web应用
    - 美观的默认样式
    
    使用方法：
        viz = PlotlyMapVisualizer()
        fig = viz.create_scatter_mapbox(df, lat_col='lat', lon_col='lon')
        fig.show()
        fig.write_html('map.html')
    """
    
    def __init__(self, mapbox_token=None):
        """
        初始化Plotly地图可视化器
        
        参数:
            mapbox_token: Mapbox访问令牌
                         免费获取: https://account.mapbox.com/access-tokens/
                         如不提供，使用开源底图
        """
        self.mapbox_token = mapbox_token
        
    def _import_plotly(self):
        """延迟导入plotly"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            return px, go
        except ImportError:
            raise ImportError(
                "Plotly未安装，请运行: pip install plotly"
            )
    
    def create_scatter_mapbox(self, df, lat_col, lon_col, 
                              color_col=None, size_col=None,
                              hover_name=None, hover_data=None,
                              center=None, zoom=3, 
                              mapbox_style='open-street-map',
                              title='散点地图', height=600):
        """
        Mapbox散点图
        
        【适用场景】
        - 交互式点数据展示
        - 支持悬停查看详情
        - Web展示/报告
        
        参数:
            df: pandas DataFrame
            lat_col: 纬度列名
            lon_col: 经度列名
            color_col: 颜色映射列名
            size_col: 大小映射列名
            hover_name: 悬停时显示的名称列
            hover_data: 悬停时显示的额外数据列表
            center: 地图中心 {'lat': ..., 'lon': ...}
            zoom: 缩放级别
            mapbox_style: 底图样式
                         开源: 'open-street-map', 'carto-positron', 'carto-darkmatter'
                         需token: 'basic', 'streets', 'outdoors', 'light', 'dark', 'satellite'
            title: 标题
            height: 图表高度（像素）
        
        返回:
            plotly.graph_objects.Figure
        
        示例:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'city': ['北京', '上海', '广州'],
            ...     'lat': [39.9, 31.2, 23.1],
            ...     'lon': [116.4, 121.5, 113.3],
            ...     'population': [21.5, 24.2, 15.3]
            ... })
            >>> 
            >>> viz = PlotlyMapVisualizer()
            >>> fig = viz.create_scatter_mapbox(df, lat_col='lat', lon_col='lon',
            ...                                  color_col='population',
            ...                                  hover_name='city')
            >>> fig.show()
        """
        px, go = self._import_plotly()
        
        # 计算中心
        if center is None:
            center = {
                'lat': df[lat_col].mean(),
                'lon': df[lon_col].mean()
            }
        
        fig = px.scatter_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            color=color_col,
            size=size_col,
            hover_name=hover_name,
            hover_data=hover_data,
            zoom=zoom,
            center=center,
            mapbox_style=mapbox_style,
            title=title,
            height=height
        )
        
        # 设置Mapbox token（如果有）
        if self.mapbox_token:
            fig.update_layout(mapbox_accesstoken=self.mapbox_token)
        
        return fig
    
    def create_density_mapbox(self, df, lat_col, lon_col, z_col=None,
                              radius=10, center=None, zoom=3,
                              mapbox_style='open-street-map',
                              colorscale='Hot', title='密度图', height=600):
        """
        Mapbox密度图
        
        【适用场景】
        - 点密度可视化
        - 热点分析
        - 聚集程度展示
        
        参数:
            df: pandas DataFrame
            lat_col: 纬度列名
            lon_col: 经度列名
            z_col: 权重列名（可选）
            radius: 密度半径
            center: 地图中心
            zoom: 缩放级别
            mapbox_style: 底图样式
            colorscale: 颜色方案
            title: 标题
            height: 图表高度
        
        返回:
            plotly.graph_objects.Figure
        
        示例:
            >>> viz = PlotlyMapVisualizer()
            >>> fig = viz.create_density_mapbox(df, 'lat', 'lon',
            ...                                  z_col='count',
            ...                                  title='事件密度分布')
            >>> fig.write_html('density.html')
        """
        px, go = self._import_plotly()
        
        if center is None:
            center = {
                'lat': df[lat_col].mean(),
                'lon': df[lon_col].mean()
            }
        
        fig = px.density_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            z=z_col,
            radius=radius,
            center=center,
            zoom=zoom,
            mapbox_style=mapbox_style,
            title=title,
            height=height,
            color_continuous_scale=colorscale
        )
        
        if self.mapbox_token:
            fig.update_layout(mapbox_accesstoken=self.mapbox_token)
        
        return fig
    
    def create_choropleth_mapbox(self, df, geojson, locations_col, 
                                  color_col, featureidkey='properties.name',
                                  center=None, zoom=3,
                                  mapbox_style='carto-positron',
                                  color_scale='Viridis',
                                  title='分级统计图', height=600):
        """
        Mapbox分级统计图
        
        【适用场景】
        - 区域数据可视化
        - 交互式区域对比
        - Web报告展示
        
        参数:
            df: pandas DataFrame
            geojson: GeoJSON对象或文件路径
            locations_col: 地区标识列名（对应GeoJSON中的属性）
            color_col: 颜色值列名
            featureidkey: GeoJSON中匹配的属性键
            center: 地图中心
            zoom: 缩放级别
            mapbox_style: 底图样式
            color_scale: 颜色方案
            title: 标题
            height: 图表高度
        
        返回:
            plotly.graph_objects.Figure
        
        示例:
            >>> import json
            >>> with open('us-states.json') as f:
            ...     geojson = json.load(f)
            >>> 
            >>> viz = PlotlyMapVisualizer()
            >>> fig = viz.create_choropleth_mapbox(
            ...     df, geojson,
            ...     locations_col='state',
            ...     color_col='unemployment',
            ...     title='美国各州失业率'
            ... )
            >>> fig.show()
        """
        px, go = self._import_plotly()
        
        fig = px.choropleth_mapbox(
            df,
            geojson=geojson,
            locations=locations_col,
            color=color_col,
            featureidkey=featureidkey,
            center=center,
            zoom=zoom,
            mapbox_style=mapbox_style,
            color_continuous_scale=color_scale,
            title=title,
            height=height
        )
        
        if self.mapbox_token:
            fig.update_layout(mapbox_accesstoken=self.mapbox_token)
        
        return fig


# ============================================================
# 便捷函数
# ============================================================

def quick_marker_map(locations, save_path=None, **kwargs):
    """
    快速创建标记点地图
    
    参数:
        locations: [(lat, lon), ...] 或 [(lat, lon, label), ...]
        save_path: 保存路径（如 'map.html'）
        **kwargs: 传递给 FoliumMapVisualizer.create_marker_map
    
    返回:
        folium.Map
    
    示例:
        >>> m = quick_marker_map([(39.9, 116.4, '北京'), (31.2, 121.5, '上海')])
        >>> m.save('cities.html')
    """
    viz = FoliumMapVisualizer()
    m = viz.create_marker_map(locations, **kwargs)
    if save_path:
        m.save(save_path)
        print(f"📍 地图已保存: {save_path}")
    return m


def quick_heatmap(data_points, save_path=None, **kwargs):
    """
    快速创建热力图
    
    参数:
        data_points: [(lat, lon), ...] 或 [(lat, lon, weight), ...]
        save_path: 保存路径
        **kwargs: 传递给 FoliumMapVisualizer.create_heatmap
    
    返回:
        folium.Map
    
    示例:
        >>> points = [(39.9 + i*0.01, 116.4 + i*0.01, 10-i) for i in range(10)]
        >>> m = quick_heatmap(points, save_path='heatmap.html')
    """
    viz = FoliumMapVisualizer()
    m = viz.create_heatmap(data_points, **kwargs)
    if save_path:
        m.save(save_path)
        print(f"🔥 热力图已保存: {save_path}")
    return m


def quick_route_map(routes, save_path=None, **kwargs):
    """
    快速创建路径地图
    
    参数:
        routes: [[(lat1, lon1), (lat2, lon2), ...], [...]]
        save_path: 保存路径
        **kwargs: 传递给 FoliumMapVisualizer.create_route_map
    
    返回:
        folium.Map
    
    示例:
        >>> route = [(39.9, 116.4), (34.3, 108.9), (30.6, 104.1)]
        >>> m = quick_route_map([route], save_path='route.html')
    """
    viz = FoliumMapVisualizer()
    m = viz.create_route_map(routes, **kwargs)
    if save_path:
        m.save(save_path)
        print(f"🛤️ 路径图已保存: {save_path}")
    return m


# ============================================================
# 演示代码
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🗺️  地理可视化工具演示")
    print("="*60)
    
    # 检查folium是否安装
    try:
        import folium
        print("✓ folium 已安装")
        
        # 演示1: 标记点地图
        print("\n📍 演示1: 创建标记点地图")
        viz = FoliumMapVisualizer()
        locations = [
            (39.9042, 116.4074, '北京'),
            (31.2304, 121.4737, '上海'),
            (23.1291, 113.2644, '广州'),
            (30.5728, 104.0668, '成都'),
            (34.3416, 108.9398, '西安'),
        ]
        m = viz.create_marker_map(locations, center=[35, 110], zoom=4)
        m.save('./figures/demo_marker_map.html')
        print("  ✓ 保存至 ./figures/demo_marker_map.html")
        
        # 演示2: 热力图
        print("\n🔥 演示2: 创建热力图")
        import random
        heat_points = [
            (39.9 + random.gauss(0, 0.05), 116.4 + random.gauss(0, 0.05), random.random())
            for _ in range(100)
        ]
        heatmap = viz.create_heatmap(heat_points, center=[39.9, 116.4], zoom=11)
        heatmap.save('./figures/demo_heatmap.html')
        print("  ✓ 保存至 ./figures/demo_heatmap.html")
        
        # 演示3: 路径地图
        print("\n🛤️ 演示3: 创建路径地图")
        route1 = [(39.9042, 116.4074), (34.3416, 108.9398), (30.5728, 104.0668)]
        route2 = [(39.9042, 116.4074), (31.2304, 121.4737)]
        route_map = viz.create_route_map([route1, route2], center=[35, 112], zoom=5)
        route_map.save('./figures/demo_route_map.html')
        print("  ✓ 保存至 ./figures/demo_route_map.html")
        
    except ImportError:
        print("✗ folium 未安装，请运行: pip install folium")
    
    # 检查plotly
    try:
        import plotly
        print("\n✓ plotly 已安装")
    except ImportError:
        print("\n✗ plotly 未安装 (可选)，请运行: pip install plotly")
    
    # 检查geopandas
    try:
        import geopandas
        print("✓ geopandas 已安装")
    except ImportError:
        print("✗ geopandas 未安装 (可选)，请运行: pip install geopandas")
    
    print("\n" + "="*60)
    print("✅ 演示完成!")
    print("="*60)
