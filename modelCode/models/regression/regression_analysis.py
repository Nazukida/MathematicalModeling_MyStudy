#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回归分析预测模型案例说明：
本案例使用多元线性回归和多项式回归模型预测房屋价格，
基于房屋面积、卧室数量和建造年份三个特征。回归分析
是一种统计方法，用于建立自变量和因变量之间的关系模型。
本案例比较线性回归和二次多项式回归的效果，展示如何
处理特征与目标之间的非线性关系。
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. 准备数据
# 生成模拟数据：200套房屋的特征和价格
np.random.seed(42)  # 设置随机种子，确保结果可复现

# 特征数据：[房屋面积(平方米), 卧室数量, 建造年份]
n_samples = 200
area = np.random.uniform(50, 200, n_samples)          # 房屋面积
bedrooms = np.random.randint(1, 5, n_samples)         # 卧室数量
year_built = np.random.uniform(1980, 2020, n_samples) # 建造年份

# 组合特征
X = np.column_stack((area, bedrooms, year_built))

# 目标数据：房屋价格(万元)，与特征呈线性和非线性关系
# 价格公式：基础价格 + 面积*单价 + 卧室数量*加成 + 年份因素(非线性) + 随机噪声
price = 50 + 0.8*area + 10*bedrooms + 0.02*(year_built-2000)**2 + np.random.normal(0, 8, n_samples)
price = np.maximum(price, 80)  # 确保价格不为过低

# 2. 数据预处理：标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(price.reshape(-1, 1)).flatten()

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# 4. 线性回归模型
# 创建并训练模型
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 预测
y_train_pred_linear = linear_model.predict(X_train)
y_test_pred_linear = linear_model.predict(X_test)

# 5. 多项式回归模型（二次）
# 创建多项式特征（二次项和交互项）
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 创建并训练模型
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# 预测
y_train_pred_poly = poly_model.predict(X_train_poly)
y_test_pred_poly = poly_model.predict(X_test_poly)

# 6. 反标准化预测结果
y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_train_pred_linear_actual = scaler_y.inverse_transform(y_train_pred_linear.reshape(-1, 1)).flatten()
y_test_pred_linear_actual = scaler_y.inverse_transform(y_test_pred_linear.reshape(-1, 1)).flatten()
y_train_pred_poly_actual = scaler_y.inverse_transform(y_train_pred_poly.reshape(-1, 1)).flatten()
y_test_pred_poly_actual = scaler_y.inverse_transform(y_test_pred_poly.reshape(-1, 1)).flatten()

# 7. 模型评估
# 计算线性回归性能指标
linear_train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_linear_actual))
linear_test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_linear_actual))
linear_train_r2 = r2_score(y_train_actual, y_train_pred_linear_actual)
linear_test_r2 = r2_score(y_test_actual, y_test_pred_linear_actual)

# 计算多项式回归性能指标
poly_train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_poly_actual))
poly_test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_poly_actual))
poly_train_r2 = r2_score(y_train_actual, y_train_pred_poly_actual)
poly_test_r2 = r2_score(y_test_actual, y_test_pred_poly_actual)

# 8. 预测新数据
# 新房屋特征
new_houses = np.array([
    [100, 2, 2000],  # 100平米，2卧室，2000年建造
    [150, 3, 2010],  # 150平米，3卧室，2010年建造
    [80, 1, 1990]    # 80平米，1卧室，1990年建造
])
new_houses_scaled = scaler_X.transform(new_houses)
new_houses_poly = poly.transform(new_houses_scaled)

# 预测价格
linear_pred = scaler_y.inverse_transform(linear_model.predict(new_houses_scaled).reshape(-1, 1)).flatten()
poly_pred = scaler_y.inverse_transform(poly_model.predict(new_houses_poly).reshape(-1, 1)).flatten()

# 9. 输出结果
print("回归分析预测模型结果：")
print("\n线性回归模型系数：")
feature_names = ['房屋面积', '卧室数量', '建造年份']
for i, name in enumerate(feature_names):
    print(f"{name}系数: {linear_model.coef_[i]:.4f}")
print(f"截距: {linear_model.intercept_:.4f}")

print("\n模型性能比较：")
print(f"线性回归 - 训练集RMSE: {linear_train_rmse:.2f}万元, R²: {linear_train_r2:.4f}")
print(f"线性回归 - 测试集RMSE: {linear_test_rmse:.2f}万元, R²: {linear_test_r2:.4f}")
print(f"二次多项式回归 - 训练集RMSE: {poly_train_rmse:.2f}万元, R²: {poly_train_r2:.4f}")
print(f"二次多项式回归 - 测试集RMSE: {poly_test_rmse:.2f}万元, R²: {poly_test_r2:.4f}")

print("\n新房屋价格预测：")
for i in range(len(new_houses)):
    print(f"{new_houses[i,0]}平米, {new_houses[i,1]}卧室, {new_houses[i,2]:.0f}年建造:")
    print(f"  线性回归预测: {linear_pred[i]:.2f}万元")
    print(f"  多项式回归预测: {poly_pred[i]:.2f}万元")

# 10. 可视化结果（房屋面积与价格的关系）
plt.figure(figsize=(12, 6))
# 选择卧室数量=2，建造年份在2000年左右的样本点进行可视化
mask = (bedrooms == 2) & (year_built > 1995) & (year_built < 2005)
area_subset = area[mask]
price_subset = price[mask]

# 排序用于绘图
sorted_indices = np.argsort(area_subset)
area_subset = area_subset[sorted_indices]
price_subset = price_subset[sorted_indices]

# 创建用于预测的特征
X_plot = np.array([[a, 2, 2000] for a in area_subset])
X_plot_scaled = scaler_X.transform(X_plot)
X_plot_poly = poly.transform(X_plot_scaled)

# 预测
linear_plot_pred = scaler_y.inverse_transform(linear_model.predict(X_plot_scaled).reshape(-1, 1)).flatten()
poly_plot_pred = scaler_y.inverse_transform(poly_model.predict(X_plot_poly).reshape(-1, 1)).flatten()

# 绘制
plt.scatter(area_subset, price_subset, c='blue', alpha=0.6, label='实际数据点')
plt.plot(area_subset, linear_plot_pred, 'r-', linewidth=2, label='线性回归')
plt.plot(area_subset, poly_plot_pred, 'g--', linewidth=2, label='二次多项式回归')
plt.xlabel('房屋面积 (平方米)')
plt.ylabel('房屋价格 (万元)')
plt.title('房屋面积与价格关系（控制变量：2卧室，2000年左右建造）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
