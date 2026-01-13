"""
============================================================
Prophet 时间序列预测
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：多季节性建模、假日效应、趋势突变点检测
原理：可加性模型 y(t) = g(t) + s(t) + h(t) + ε
作者：MCM/ICM Team
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from prophet import Prophet

# 图表美化设置
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class ProphetPredictor:
    """
    Prophet时间序列预测器封装类
    
    核心组件：
    - g(t): 趋势项（线性或分段线性）
    - s(t): 季节性项（年/周/日）
    - h(t): 假日效应
    - ε: 误差项
    
    优点：
    - 自动检测趋势突变点
    - 处理缺失值和异常值
    - 灵活的季节性设置
    - 直观的置信区间
    """
    
    def __init__(self, yearly_seasonality=True, weekly_seasonality=True,
                 daily_seasonality=False, changepoint_prior_scale=0.05,
                 verbose=True):
        """
        参数配置
        
        :param yearly_seasonality: 年季节性
        :param weekly_seasonality: 周季节性
        :param daily_seasonality: 日季节性
        :param changepoint_prior_scale: 趋势变化灵活度（0.001-0.5）
        """
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale
        )
        self.verbose = verbose
        self.forecast = None
        self.train_data = None
        self.metrics = None
    
    def fit(self, data, ds_col='ds', y_col='y'):
        """
        训练模型
        
        :param data: DataFrame，需包含日期列和目标列
        :param ds_col: 日期列名
        :param y_col: 目标列名
        """
        # 格式化数据
        df = data.rename(columns={ds_col: 'ds', y_col: 'y'})[['ds', 'y']]
        df['ds'] = pd.to_datetime(df['ds'])
        self.train_data = df
        
        # 训练
        self.model.fit(df)
        
        if self.verbose:
            print("\n" + "="*50)
            print("📅 Prophet 模型训练完成")
            print("="*50)
            print(f"  训练数据: {len(df)} 条")
            print(f"  时间范围: {df['ds'].min().date()} 至 {df['ds'].max().date()}")
            print(f"  趋势突变点数: {len(self.model.changepoints)}")
            print("="*50)
        
        return self
    
    def predict(self, periods=30, freq='D'):
        """
        预测未来
        
        :param periods: 预测步数
        :param freq: 频率（'D'日/'W'周/'M'月）
        """
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        self.forecast = self.model.predict(future)
        
        if self.verbose:
            self._print_forecast(periods)
        
        return self.forecast
    
    def _print_forecast(self, periods):
        """打印预测结果"""
        future_data = self.forecast.tail(periods)
        
        print("\n" + "="*50)
        print(f"🔮 Prophet 未来{periods}期预测")
        print("="*50)
        print("\n  预测值（采样）:")
        sample_idx = [0, periods//4, periods//2, -1]
        for i in sample_idx:
            row = future_data.iloc[i]
            print(f"    {row['ds'].date()}: {row['yhat']:.1f} "
                  f"[{row['yhat_lower']:.1f}, {row['yhat_upper']:.1f}]")
        print("="*50)
    
    def evaluate(self, test_data=None):
        """模型评估"""
        if self.forecast is None:
            raise ValueError("请先调用predict()")
        
        # 使用训练集评估
        merged = self.train_data.merge(
            self.forecast[['ds', 'yhat']], on='ds', how='left'
        )
        
        mae = np.mean(np.abs(merged['y'] - merged['yhat']))
        mape = np.mean(np.abs((merged['y'] - merged['yhat']) / merged['y'])) * 100
        rmse = np.sqrt(np.mean((merged['y'] - merged['yhat']) ** 2))
        
        self.metrics = {'MAE': mae, 'MAPE': mape, 'RMSE': rmse}
        
        print("\n模型评估指标（训练集）:")
        print(f"  MAE: {mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: {rmse:.2f}")
        
        return self.metrics
    
    def add_holidays(self, holidays_df):
        """
        添加假日效应
        
        :param holidays_df: DataFrame with columns ['ds', 'holiday']
        """
        self.model = Prophet()
        self.model.add_country_holidays(country_name='CN')  # 中国假日
        return self
    
    def plot_forecast(self, save_path=None):
        """可视化预测结果"""
        if self.forecast is None:
            raise ValueError("请先调用predict()")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 历史数据
        ax.plot(self.train_data['ds'], self.train_data['y'], 
               'o', color='#2E86AB', markersize=3, label='历史数据', alpha=0.7)
        
        # 预测值
        ax.plot(self.forecast['ds'], self.forecast['yhat'],
               color='#E94F37', linewidth=2, label='预测值')
        
        # 置信区间
        ax.fill_between(self.forecast['ds'],
                       self.forecast['yhat_lower'],
                       self.forecast['yhat_upper'],
                       color='#E94F37', alpha=0.2, label='95% 置信区间')
        
        # 标记预测区域
        last_train = self.train_data['ds'].max()
        ax.axvline(x=last_train, color='gray', linestyle='--', 
                  linewidth=1.5, label='预测起点')
        
        ax.set_xlabel('日期', fontsize=12, fontweight='bold')
        ax.set_ylabel('数值', fontsize=12, fontweight='bold')
        ax.set_title('Prophet 时间序列预测', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_components(self, save_path=None):
        """可视化分解成分"""
        if self.forecast is None:
            raise ValueError("请先调用predict()")
        
        fig = self.model.plot_components(self.forecast)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("   Prophet 时间序列预测演示 - 客流量预测")
    print("="*60)
    
    # 1. 模拟数据（带周季节性的客流量）
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    n = len(dates)
    trend = np.linspace(1000, 1100, n)  # 年趋势
    weekly_season = np.where(pd.to_datetime(dates).weekday >= 5, 200, 50)  # 周末高峰
    noise = np.random.normal(0, 30, n)
    
    data = pd.DataFrame({
        "ds": dates,
        "y": trend + weekly_season + noise
    })
    
    print("\n数据概览：")
    print(data.describe().round(2))
    
    # 2. 建模与预测
    prophet = ProphetPredictor(
        yearly_seasonality=True,
        weekly_seasonality=True,
        verbose=True
    )
    prophet.fit(data)
    prophet.predict(periods=30)
    
    # 3. 评估
    prophet.evaluate()
    
    # 4. 可视化
    prophet.plot_forecast()
    prophet.plot_components()
    
    # 5. 输出预测结果
    print("\n未来7天详细预测：")
    future_7 = prophet.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
    print(future_7.round(0).to_string(index=False))
