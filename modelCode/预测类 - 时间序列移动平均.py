"""
============================================================
时间序列移动平均预测 (Moving Average Forecasting)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：平滑数据波动、短期趋势预测、异常值检测
方法：SMA/WMA/EMA 三种移动平均
作者：MCM/ICM Team
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 图表美化设置
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class MovingAveragePredictor:
    """
    移动平均预测器
    
    支持三种方法：
    - SMA (Simple Moving Average): 简单移动平均
    - WMA (Weighted Moving Average): 加权移动平均，近期权重大
    - EMA (Exponential Moving Average): 指数移动平均
    
    核心公式：
    - SMA: MA_t = (1/n) * Σ(x_{t-i}), i=0..n-1
    - WMA: MA_t = Σ(w_i * x_{t-i}) / Σ(w_i)
    - EMA: EMA_t = α * x_t + (1-α) * EMA_{t-1}, α = 2/(n+1)
    """
    
    def __init__(self, window=7, method='sma', verbose=True):
        """
        参数配置
        
        :param window: 窗口大小（3-12常用）
        :param method: 'sma'/'wma'/'ema'
        :param verbose: 是否打印过程
        """
        self.window = window
        self.method = method.lower()
        self.verbose = verbose
        self.data = None
        self.predictions = None
        self.mae = None
        self.rmse = None
    
    def fit_predict(self, data, column='value'):
        """
        拟合并预测
        
        :param data: DataFrame或Series
        :param column: 数值列名
        """
        if isinstance(data, pd.Series):
            self.data = data.values
        elif isinstance(data, pd.DataFrame):
            self.data = data[column].values
        else:
            self.data = np.array(data)
        
        n = len(self.data)
        self.predictions = np.full(n, np.nan)
        
        if self.method == 'sma':
            self._compute_sma()
        elif self.method == 'wma':
            self._compute_wma()
        elif self.method == 'ema':
            self._compute_ema()
        
        self._compute_metrics()
        
        if self.verbose:
            self._print_results()
        
        return self.predictions
    
    def _compute_sma(self):
        """简单移动平均"""
        for i in range(self.window, len(self.data)):
            self.predictions[i] = np.mean(self.data[i-self.window:i])
    
    def _compute_wma(self):
        """加权移动平均（线性权重）"""
        weights = np.arange(1, self.window + 1)
        for i in range(self.window, len(self.data)):
            window_data = self.data[i-self.window:i]
            self.predictions[i] = np.sum(weights * window_data) / np.sum(weights)
    
    def _compute_ema(self):
        """指数移动平均"""
        alpha = 2 / (self.window + 1)
        self.predictions[self.window-1] = np.mean(self.data[:self.window])
        for i in range(self.window, len(self.data)):
            self.predictions[i] = alpha * self.data[i-1] + (1 - alpha) * self.predictions[i-1]
    
    def _compute_metrics(self):
        """计算误差指标"""
        valid_idx = ~np.isnan(self.predictions)
        actual = self.data[valid_idx]
        pred = self.predictions[valid_idx]
        
        self.mae = np.mean(np.abs(actual - pred))
        self.rmse = np.sqrt(np.mean((actual - pred) ** 2))
    
    def _print_results(self):
        """打印结果"""
        method_names = {'sma': '简单移动平均', 'wma': '加权移动平均', 'ema': '指数移动平均'}
        print("\n" + "="*50)
        print(f"📈 {method_names[self.method]} 预测结果")
        print("="*50)
        print(f"  窗口大小: {self.window}")
        print(f"  MAE: {self.mae:.4f}")
        print(f"  RMSE: {self.rmse:.4f}")
        print("="*50)
    
    def forecast(self, steps=1):
        """向前预测"""
        last_values = self.data[-self.window:]
        
        forecasts = []
        for _ in range(steps):
            if self.method == 'sma':
                pred = np.mean(last_values)
            elif self.method == 'wma':
                weights = np.arange(1, self.window + 1)
                pred = np.sum(weights * last_values) / np.sum(weights)
            else:  # ema
                alpha = 2 / (self.window + 1)
                pred = alpha * last_values[-1] + (1 - alpha) * np.mean(last_values)
            
            forecasts.append(pred)
            last_values = np.append(last_values[1:], pred)
        
        return np.array(forecasts)
    
    def plot_result(self, time_index=None, save_path=None):
        """可视化预测结果"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if time_index is None:
            time_index = np.arange(len(self.data))
        
        # 原始数据
        ax.plot(time_index, self.data, 'o-', color='#2E86AB', 
               label='原始数据', markersize=4, linewidth=1.5)
        
        # 预测
        ax.plot(time_index, self.predictions, 's--', color='#E94F37',
               label=f'{self.method.upper()} 预测 (窗口={self.window})',
               markersize=3, linewidth=2)
        
        ax.set_xlabel('时间', fontsize=12, fontweight='bold')
        ax.set_ylabel('数值', fontsize=12, fontweight='bold')
        ax.set_title(f'时间序列移动平均预测 (MAE={self.mae:.2f}, RMSE={self.rmse:.2f})',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("   移动平均预测演示 - 客流量数据")
    print("="*60)
    
    # 1. 模拟数据（含趋势和季节性的客流量）
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100)  # 100天数据
    trend = np.linspace(100, 200, 100)  # 增长趋势
    seasonal = 30 * np.sin(np.linspace(0, 10, 100))  # 周期性波动
    data = pd.DataFrame({
        "日期": dates,
        "客流量": trend + seasonal + np.random.normal(0, 10, 100)  # 加噪声
    })
    
    print("\n数据概览：")
    print(data.describe().round(2))
    
    # 2. 简单移动平均
    window = 7
    sma = MovingAveragePredictor(window=window, method='sma')
    sma.fit_predict(data, column='客流量')
    sma.plot_result(time_index=dates)
    
    # 3. 指数移动平均
    ema = MovingAveragePredictor(window=window, method='ema')
    ema.fit_predict(data, column='客流量')
    ema.plot_result(time_index=dates)
    
    # 4. 未来预测
    future = ema.forecast(steps=7)
    print(f"\n未来7天预测: {future.round(2)}")
