import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("flow.csv")

fig = plt.figure(figsize=(16, 9), facecolor='#1a1a2e')
ax = plt.axes(projection=ccrs.Robinson())  # Robinson投影更美观
ax.set_global()

# ===== 灰色地图底图 =====
# 海洋颜色
ax.set_facecolor('#2d3436')

# 陆地填充（深灰色）
ax.add_feature(cfeature.LAND, facecolor='#636e72', edgecolor='none')

# 海洋填充
ax.add_feature(cfeature.OCEAN, facecolor='#2d3436')

# 国家边界（浅灰色细线）
ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#b2bec3', alpha=0.5)

# 海岸线
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#dfe6e9')

# ===== 绘制流动线条 =====
# 根据value值计算线宽和颜色
max_value = df['value'].max()
min_value = df['value'].min()

# 颜色映射（从浅蓝到亮红）
cmap = plt.cm.plasma

for _, row in df.iterrows():
    # 归一化value用于颜色和透明度
    norm_value = (row['value'] - min_value) / (max_value - min_value + 1)
    
    # 线宽：根据value动态调整（0.5 到 4）
    linewidth = 0.5 + 3.5 * norm_value
    
    # 透明度：value越大越不透明
    alpha = 0.4 + 0.5 * norm_value
    
    # 颜色：根据value选择
    color = cmap(0.3 + 0.6 * norm_value)
    
    # 绘制大圆弧线条
    ax.plot(
        [row['origin_lon'], row['dest_lon']],
        [row['origin_lat'], row['dest_lat']],
        linewidth=linewidth,
        alpha=alpha,
        color=color,
        transform=ccrs.Geodetic(),  # 大圆弧
        zorder=2
    )
    
    # 绘制起点和终点标记
    ax.scatter(row['origin_lon'], row['origin_lat'], 
               s=20 + 30 * norm_value, c='#00cec9', 
               transform=ccrs.PlateCarree(), zorder=3, 
               edgecolors='white', linewidths=0.5)
    ax.scatter(row['dest_lon'], row['dest_lat'], 
               s=20 + 30 * norm_value, c='#fd79a8', 
               transform=ccrs.PlateCarree(), zorder=3,
               edgecolors='white', linewidths=0.5)

# ===== 添加标题和图例 =====
plt.title('Global Flow Network Visualization', 
          fontsize=16, fontweight='bold', color='white', pad=20)

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_value, vmax=max_value))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
                    fraction=0.03, pad=0.05, shrink=0.5)
cbar.set_label('Flow Value', color='white', fontsize=10)
cbar.ax.xaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')

plt.tight_layout()
plt.savefig('figures/global_flow_map.png', dpi=300, 
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
