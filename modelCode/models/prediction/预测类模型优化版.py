"""
============================================================
预测类模型 (Prediction Models)
包含：时间序列 + 回归模型 + 机器学习预测
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：时间序列预测、回归分析、趋势预测
特点：完整的参数设置、数据预处理、可视化与美化
作者：MCM/ICM Team
日期：2026年1月
============================================================

使用场景：
- 销量/客流量/股价预测
- 趋势分析与外推
- 多变量回归预测
- 时间序列分解与预测
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
from datetime import datetime, timedelta
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

warnings.filterwarnings('ignore')


# ============================================================
# 第一部分：全局配置与美化设置 (Global Configuration)
# ============================================================

class PlotStyleConfig:
    """图表美化配置类 - 符合学术论文标准"""
    
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'success': '#C73E1D',
        'neutral': '#3B3B3B',
        'background': '#FAFAFA',
        'actual': '#2E86AB',
        'predicted': '#C73E1D',
        'confidence': '#F18F01'
    }
    
    PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4C9A', '#1B998B']
    
    @staticmethod
    def setup_style():
        plt.style.use('seaborn-v0_8-whitegrid')
        rcParams['figure.figsize'] = (12, 8)
        rcParams['figure.dpi'] = 100
        rcParams['savefig.dpi'] = 300
        rcParams['font.size'] = 11
        rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False

PlotStyleConfig.setup_style()


# ============================================================
# 第二部分：时间序列数据生成器 (Time Series Generator)
# ============================================================

class TimeSeriesGenerator:
    """时间序列数据生成器 - 用于测试和演示"""
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_trend_seasonal(self, n_periods=365, 
                                 trend_type='linear',
                                 seasonal_period=7,
                                 noise_level=10):
        """
        生成带趋势和季节性的时间序列
        
        :param n_periods: 数据点数量
        :param trend_type: 趋势类型 ('linear', 'quadratic', 'exponential')
        :param seasonal_period: 季节周期
        :param noise_level: 噪声水平
        """
        t = np.arange(n_periods)
        
        # 趋势成分
        if trend_type == 'linear':
            trend = 100 + 0.5 * t
        elif trend_type == 'quadratic':
            trend = 100 + 0.01 * t**2
        elif trend_type == 'exponential':
            trend = 100 * np.exp(0.005 * t)
        
        # 季节性成分
        seasonal = 20 * np.sin(2 * np.pi * t / seasonal_period)
        
        # 噪声
        noise = np.random.normal(0, noise_level, n_periods)
        
        # 合成
        y = trend + seasonal + noise
        
        dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'value': y,
            'trend': trend,
            'seasonal': seasonal,
            'noise': noise
        })
    
    def generate_arima_like(self, n_periods=200, ar_coefs=[0.7], ma_coefs=[0.3]):
        """生成ARIMA风格的时间序列"""
        np.random.seed(self.random_seed)
        
        y = np.zeros(n_periods)
        errors = np.random.normal(0, 1, n_periods)
        
        p = len(ar_coefs)
        q = len(ma_coefs)
        
        for t in range(max(p, q), n_periods):
            ar_term = sum(ar_coefs[i] * y[t-i-1] for i in range(p))
            ma_term = sum(ma_coefs[i] * errors[t-i-1] for i in range(q))
            y[t] = ar_term + ma_term + errors[t]
        
        y = y + 100  # 平移
        dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='D')
        
        return pd.DataFrame({'date': dates, 'value': y})
    
    def generate_multivariate(self, n_samples=200):
        """生成多变量回归数据"""
        np.random.seed(self.random_seed)
        
        # 自变量
        X1 = np.random.uniform(10, 100, n_samples)  # 广告投入
        X2 = np.random.uniform(500, 2000, n_samples)  # 客流量
        X3 = np.random.randint(1, 6, n_samples)  # 促销力度
        
        # 因变量（有噪声的线性关系）
        y = 5 + 0.3 * X1 + 0.01 * X2 + 2 * X3 + np.random.normal(0, 3, n_samples)
        
        return pd.DataFrame({
            '广告投入': X1,
            '客流量': X2,
            '促销力度': X3,
            '销量': y
        })


# ============================================================
# 第三部分：时间序列分析 (Time Series Analysis)
# ============================================================

class TimeSeriesAnalyzer:
    """时间序列分析类"""
    
    def __init__(self, data, date_col='date', value_col='value'):
        """
        初始化
        :param data: DataFrame或Series
        :param date_col: 日期列名
        :param value_col: 值列名
        """
        if isinstance(data, pd.DataFrame):
            self.dates = pd.to_datetime(data[date_col])
            self.values = data[value_col].values
        else:
            self.dates = data.index
            self.values = data.values
        
        self.n = len(self.values)
        self.decomposition = None
    
    def moving_average(self, window=7):
        """
        移动平均
        :param window: 窗口大小
        :return: 移动平均序列
        """
        return pd.Series(self.values).rolling(window=window, center=True).mean().values
    
    def exponential_smoothing(self, alpha=0.3):
        """
        简单指数平滑
        :param alpha: 平滑系数 (0-1)
        :return: 平滑后的序列
        """
        smoothed = np.zeros(self.n)
        smoothed[0] = self.values[0]
        
        for t in range(1, self.n):
            smoothed[t] = alpha * self.values[t] + (1 - alpha) * smoothed[t-1]
        
        return smoothed
    
    def holt_winters(self, alpha=0.3, beta=0.1, gamma=0.1, 
                     seasonal_period=7, n_forecast=30):
        """
        Holt-Winters三次指数平滑（加法模型）
        
        :param alpha: 水平平滑系数
        :param beta: 趋势平滑系数
        :param gamma: 季节平滑系数
        :param seasonal_period: 季节周期
        :param n_forecast: 预测步数
        """
        n = self.n
        m = seasonal_period
        
        # 初始化
        level = np.zeros(n + n_forecast)
        trend = np.zeros(n + n_forecast)
        seasonal = np.zeros(n + n_forecast)
        fitted = np.zeros(n + n_forecast)
        
        # 初始值
        level[0] = np.mean(self.values[:m])
        trend[0] = (np.mean(self.values[m:2*m]) - np.mean(self.values[:m])) / m
        for i in range(m):
            seasonal[i] = self.values[i] - level[0]
        
        # 拟合
        for t in range(1, n):
            level[t] = alpha * (self.values[t] - seasonal[t-m]) + (1 - alpha) * (level[t-1] + trend[t-1])
            trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]
            seasonal[t] = gamma * (self.values[t] - level[t]) + (1 - gamma) * seasonal[t-m]
            fitted[t] = level[t-1] + trend[t-1] + seasonal[t-m]
        
        # 预测
        for t in range(n, n + n_forecast):
            level[t] = level[n-1] + (t - n + 1) * trend[n-1]
            fitted[t] = level[t] + seasonal[t-m]
        
        return {
            'fitted': fitted[:n],
            'forecast': fitted[n:],
            'level': level,
            'trend': trend,
            'seasonal': seasonal
        }
    
    def decompose(self, period=7, model='additive'):
        """
        时间序列分解
        
        :param period: 季节周期
        :param model: 'additive' 或 'multiplicative'
        """
        # 趋势（移动平均）
        trend = self.moving_average(window=period)
        
        if model == 'additive':
            detrended = self.values - trend
        else:
            detrended = self.values / (trend + 1e-10)
        
        # 季节性（按周期平均）
        seasonal = np.zeros(self.n)
        for i in range(period):
            indices = np.arange(i, self.n, period)
            valid_indices = indices[~np.isnan(detrended[indices])]
            if len(valid_indices) > 0:
                seasonal[indices] = np.nanmean(detrended[valid_indices])
        
        # 残差
        if model == 'additive':
            residual = self.values - trend - seasonal
        else:
            residual = self.values / ((trend + 1e-10) * (seasonal + 1e-10))
        
        self.decomposition = {
            'observed': self.values,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'model': model
        }
        
        return self.decomposition
    
    def compute_metrics(self, actual, predicted):
        """计算预测评价指标"""
        mask = ~np.isnan(actual) & ~np.isnan(predicted)
        actual = actual[mask]
        predicted = predicted[mask]
        
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
        r2 = r2_score(actual, predicted)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }


# ============================================================
# 第四部分：移动平均预测 (Moving Average Prediction)
# ============================================================

class MovingAveragePredictor:
    """
    移动平均预测器
    
    方法：
    - 简单移动平均 (SMA)
    - 加权移动平均 (WMA)
    - 指数移动平均 (EMA)
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.history = None
        self.predictions = None
        self.metrics = None
    
    def fit_predict(self, data, window=7, method='simple', n_forecast=7):
        """
        拟合并预测
        
        :param data: 时间序列数据
        :param window: 移动窗口大小
        :param method: 'simple', 'weighted', 'exponential'
        :param n_forecast: 预测步数
        """
        if isinstance(data, pd.DataFrame):
            values = data.iloc[:, -1].values if 'value' not in data.columns else data['value'].values
        else:
            values = np.array(data)
        
        n = len(values)
        fitted = np.zeros(n)
        
        if method == 'simple':
            # 简单移动平均
            for t in range(window, n):
                fitted[t] = np.mean(values[t-window:t])
            fitted[:window] = np.nan
            
            # 预测
            forecast = np.zeros(n_forecast)
            last_values = list(values[-window:])
            for i in range(n_forecast):
                forecast[i] = np.mean(last_values)
                last_values.pop(0)
                last_values.append(forecast[i])
        
        elif method == 'weighted':
            # 加权移动平均
            weights = np.arange(1, window + 1)
            weights = weights / weights.sum()
            
            for t in range(window, n):
                fitted[t] = np.sum(weights * values[t-window:t])
            fitted[:window] = np.nan
            
            forecast = np.zeros(n_forecast)
            last_values = list(values[-window:])
            for i in range(n_forecast):
                forecast[i] = np.sum(weights * last_values)
                last_values.pop(0)
                last_values.append(forecast[i])
        
        elif method == 'exponential':
            # 指数移动平均
            alpha = 2 / (window + 1)
            fitted[0] = values[0]
            for t in range(1, n):
                fitted[t] = alpha * values[t] + (1 - alpha) * fitted[t-1]
            
            forecast = np.zeros(n_forecast)
            last_ema = fitted[-1]
            for i in range(n_forecast):
                forecast[i] = last_ema  # EMA趋于稳定
        
        self.history = values
        self.fitted = fitted
        self.predictions = forecast
        
        # 计算评价指标
        valid_idx = ~np.isnan(fitted)
        self.metrics = {
            'RMSE': np.sqrt(mean_squared_error(values[valid_idx], fitted[valid_idx])),
            'MAE': mean_absolute_error(values[valid_idx], fitted[valid_idx]),
            'MAPE': np.mean(np.abs((values[valid_idx] - fitted[valid_idx]) / 
                                   (values[valid_idx] + 1e-10))) * 100
        }
        
        if self.verbose:
            self._print_results(method, window)
        
        return self
    
    def _print_results(self, method, window):
        """打印结果"""
        print("\n" + "="*60)
        print(f"📊 移动平均预测结果 ({method.upper()}, window={window})")
        print("="*60)
        print(f"  RMSE: {self.metrics['RMSE']:.4f}")
        print(f"  MAE:  {self.metrics['MAE']:.4f}")
        print(f"  MAPE: {self.metrics['MAPE']:.2f}%")
        print(f"  预测值: {self.predictions[:5].round(2)} ...")
        print("="*60)
    
    def get_forecast(self):
        """获取预测结果"""
        return self.predictions


# ============================================================
# 第五部分：回归预测器 (Regression Predictor)
# ============================================================

class RegressionPredictor:
    """
    回归预测器
    
    模型：
    - 线性回归
    - Ridge回归（L2正则化）
    - Lasso回归（L1正则化）
    - 随机森林回归
    - 梯度提升回归
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metrics = None
        self.feature_importance = None
    
    def fit(self, X, y, model_type='random_forest', 
            test_size=0.2, scale=True, **kwargs):
        """
        拟合模型
        
        :param X: 特征矩阵
        :param y: 目标变量
        :param model_type: 模型类型
        :param test_size: 测试集比例
        :param scale: 是否标准化
        :param kwargs: 模型额外参数
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"特征{i+1}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 标准化
        if scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # 选择模型
        if model_type == 'linear':
            self.model = LinearRegression(**kwargs)
        elif model_type == 'ridge':
            self.model = Ridge(alpha=kwargs.get('alpha', 1.0))
        elif model_type == 'lasso':
            self.model = Lasso(alpha=kwargs.get('alpha', 1.0))
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=42
            )
        
        # 训练
        self.model.fit(X_train, y_train)
        
        # 预测
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # 评估
        self.metrics = {
            'train': {
                'R2': r2_score(y_train, y_train_pred),
                'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'MAE': mean_absolute_error(y_train, y_train_pred)
            },
            'test': {
                'R2': r2_score(y_test, y_test_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'MAE': mean_absolute_error(y_test, y_test_pred)
            }
        }
        
        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.Series(
                self.model.feature_importances_, 
                index=self.feature_names
            )
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = pd.Series(
                np.abs(self.model.coef_), 
                index=self.feature_names
            )
        
        if self.verbose:
            self._print_results(model_type)
        
        return self
    
    def _print_results(self, model_type):
        """打印结果"""
        print("\n" + "="*60)
        print(f"📊 回归模型结果 ({model_type})")
        print("="*60)
        print("\n  训练集:")
        print(f"    R²:   {self.metrics['train']['R2']:.4f}")
        print(f"    RMSE: {self.metrics['train']['RMSE']:.4f}")
        print(f"    MAE:  {self.metrics['train']['MAE']:.4f}")
        print("\n  测试集:")
        print(f"    R²:   {self.metrics['test']['R2']:.4f}")
        print(f"    RMSE: {self.metrics['test']['RMSE']:.4f}")
        print(f"    MAE:  {self.metrics['test']['MAE']:.4f}")
        
        if self.feature_importance is not None:
            print("\n  特征重要性:")
            for name, imp in self.feature_importance.sort_values(ascending=False).items():
                print(f"    {name}: {imp:.4f}")
        print("="*60)
    
    def predict(self, X):
        """预测"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def cross_validate(self, X, y, cv=5):
        """交叉验证"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        
        print(f"\n交叉验证 R² (cv={cv}):")
        print(f"  Mean: {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"  Scores: {scores.round(4)}")
        
        return scores


# ============================================================
# 第六部分：集成预测器 (Ensemble Predictor)
# ============================================================

class EnsemblePredictor:
    """
    集成预测器 - 结合多个模型的预测
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.models = {}
        self.weights = None
        self.metrics = {}
    
    def add_model(self, name, model):
        """添加模型"""
        self.models[name] = model
    
    def fit_all(self, X, y, test_size=0.2):
        """训练所有模型"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        predictions = {}
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions[name] = pred
            
            self.metrics[name] = {
                'R2': r2_score(y_test, pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, pred))
            }
        
        # 计算最优权重（基于R2分数）
        r2_scores = np.array([self.metrics[name]['R2'] for name in self.models])
        r2_scores = np.maximum(r2_scores, 0)  # 确保非负
        self.weights = r2_scores / (r2_scores.sum() + 1e-10)
        
        # 集成预测
        ensemble_pred = np.zeros_like(y_test, dtype=float)
        for i, name in enumerate(self.models):
            ensemble_pred += self.weights[i] * predictions[name]
        
        self.metrics['Ensemble'] = {
            'R2': r2_score(y_test, ensemble_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, ensemble_pred))
        }
        
        if self.verbose:
            self._print_comparison()
        
        return self
    
    def _print_comparison(self):
        """打印模型对比"""
        print("\n" + "="*60)
        print("📊 集成模型对比")
        print("="*60)
        print(f"\n  {'模型':<20} {'R²':>10} {'RMSE':>10}")
        print("  " + "-"*40)
        for name, metrics in self.metrics.items():
            print(f"  {name:<20} {metrics['R2']:>10.4f} {metrics['RMSE']:>10.4f}")
        print("\n  模型权重:", dict(zip(self.models.keys(), self.weights.round(4))))
        print("="*60)
    
    def predict(self, X):
        """集成预测"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        pred = np.zeros(X.shape[0])
        for i, (name, model) in enumerate(self.models.items()):
            pred += self.weights[i] * model.predict(X)
        
        return pred


# ============================================================
# 第七部分：可视化模块 (Visualization)
# ============================================================

class PredictionVisualizer:
    """预测模型可视化类"""
    
    def __init__(self):
        self.colors = PlotStyleConfig.COLORS
    
    def plot_time_series(self, dates, actual, predicted=None, 
                         forecast_dates=None, forecast=None,
                         title="时间序列预测", save_path=None):
        """绘制时间序列预测图"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(dates, actual, 'o-', markersize=3, linewidth=1.5,
               color=self.colors['actual'], label='实际值', alpha=0.8)
        
        if predicted is not None:
            ax.plot(dates, predicted, '-', linewidth=2,
                   color=self.colors['predicted'], label='拟合值')
        
        if forecast is not None and forecast_dates is not None:
            ax.plot(forecast_dates, forecast, '--', linewidth=2,
                   color=self.colors['confidence'], label='预测值')
            ax.axvline(x=dates.iloc[-1] if hasattr(dates, 'iloc') else dates[-1],
                      color='gray', linestyle=':', alpha=0.7)
        
        ax.set_xlabel('日期', fontweight='bold')
        ax.set_ylabel('值', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_decomposition(self, decomposition, title="时间序列分解", save_path=None):
        """绘制时间序列分解图"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        components = ['observed', 'trend', 'seasonal', 'residual']
        titles = ['(a) 原始序列', '(b) 趋势成分', '(c) 季节成分', '(d) 残差']
        colors = [self.colors['primary'], self.colors['secondary'], 
                  self.colors['accent'], self.colors['neutral']]
        
        for ax, comp, t, c in zip(axes, components, titles, colors):
            ax.plot(decomposition[comp], color=c, linewidth=1.5)
            ax.set_ylabel(t, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('时间', fontweight='bold')
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_actual_vs_predicted(self, actual, predicted, 
                                  title="实际值 vs 预测值", save_path=None):
        """绘制实际值与预测值散点图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 散点图
        ax1 = axes[0]
        ax1.scatter(actual, predicted, alpha=0.6, 
                   color=self.colors['primary'], edgecolors='white')
        
        # 对角线
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='y=x (完美预测)')
        
        ax1.set_xlabel('实际值', fontweight='bold')
        ax1.set_ylabel('预测值', fontweight='bold')
        ax1.set_title('(a) 预测散点图', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 残差图
        ax2 = axes[1]
        residuals = actual - predicted
        ax2.scatter(predicted, residuals, alpha=0.6,
                   color=self.colors['secondary'], edgecolors='white')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('预测值', fontweight='bold')
        ax2.set_ylabel('残差', fontweight='bold')
        ax2.set_title('(b) 残差图', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, importance, title="特征重要性", save_path=None):
        """绘制特征重要性图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        importance = importance.sort_values(ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance)))
        
        bars = ax.barh(importance.index, importance.values, 
                      color=colors, edgecolor='white', linewidth=2)
        
        ax.set_xlabel('重要性', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, importance.values):
            ax.text(val + max(importance.values)*0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, metrics_dict, title="模型性能对比", save_path=None):
        """绘制模型性能对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        models = list(metrics_dict.keys())
        r2_values = [metrics_dict[m]['R2'] for m in models]
        rmse_values = [metrics_dict[m]['RMSE'] for m in models]
        
        colors = PlotStyleConfig.PALETTE[:len(models)]
        
        # R²对比
        ax1 = axes[0]
        bars1 = ax1.bar(models, r2_values, color=colors, edgecolor='white', linewidth=2)
        ax1.set_ylabel('R²', fontweight='bold')
        ax1.set_title('(a) R² Score', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1)
        for bar, val in zip(bars1, r2_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # RMSE对比
        ax2 = axes[1]
        bars2 = ax2.bar(models, rmse_values, color=colors, edgecolor='white', linewidth=2)
        ax2.set_ylabel('RMSE', fontweight='bold')
        ax2.set_title('(b) RMSE', fontsize=12, fontweight='bold')
        for bar, val in zip(bars2, rmse_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 第八部分：主程序与完整示例 (Main Program)
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("   PREDICTION MODELS FOR MCM/ICM")
    print("   预测类模型 - 时间序列 + 回归分析")
    print("   Extended Version with Visualization")
    print("="*70)
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    📊 预测模型分析流程                            ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║   [时间序列预测]                                                  ║
    ║      ├─ 移动平均 (MA): 简单、加权、指数                           ║
    ║      ├─ 指数平滑: 单指数、双指数、三指数                          ║
    ║      └─ 分解: 趋势 + 季节性 + 残差                               ║
    ║                                                                  ║
    ║   [回归预测]                                                      ║
    ║      ├─ 线性回归: Linear, Ridge, Lasso                           ║
    ║      ├─ 集成方法: RandomForest, GradientBoosting                 ║
    ║      └─ 模型评估: R², RMSE, MAE, MAPE                            ║
    ║                                                                  ║
    ║   [模型选择建议]                                                  ║
    ║      ├─ 趋势明显 → 线性回归、指数平滑                             ║
    ║      ├─ 季节性强 → Holt-Winters、分解法                          ║
    ║      └─ 复杂关系 → RandomForest、GradientBoosting                ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    visualizer = PredictionVisualizer()
    generator = TimeSeriesGenerator(random_seed=2026)
    
    # ================================================================
    # 示例1：时间序列分解与预测
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 1: 时间序列分解与移动平均预测")
    print("="*70)
    
    # 生成数据
    ts_data = generator.generate_trend_seasonal(
        n_periods=180, 
        trend_type='linear',
        seasonal_period=7,
        noise_level=15
    )
    
    print(f"\n数据概览:")
    print(f"  时间范围: {ts_data['date'].min()} 到 {ts_data['date'].max()}")
    print(f"  数据点数: {len(ts_data)}")
    
    # 时间序列分析
    analyzer = TimeSeriesAnalyzer(ts_data, 'date', 'value')
    decomposition = analyzer.decompose(period=7)
    
    # 可视化分解
    visualizer.plot_decomposition(decomposition, title="时间序列分解 (周期=7)")
    
    # 移动平均预测
    print("\n移动平均预测:")
    
    ma_predictor = MovingAveragePredictor(verbose=True)
    
    # 简单移动平均
    ma_predictor.fit_predict(ts_data['value'], window=7, method='simple', n_forecast=14)
    
    # 指数移动平均
    ma_predictor.fit_predict(ts_data['value'], window=7, method='exponential', n_forecast=14)
    
    # 可视化
    forecast_dates = pd.date_range(
        start=ts_data['date'].iloc[-1] + timedelta(days=1),
        periods=14
    )
    
    visualizer.plot_time_series(
        dates=ts_data['date'],
        actual=ts_data['value'].values,
        predicted=ma_predictor.fitted,
        forecast_dates=forecast_dates,
        forecast=ma_predictor.predictions,
        title="移动平均预测结果"
    )
    
    # ================================================================
    # 示例2：Holt-Winters三次指数平滑
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 2: Holt-Winters三次指数平滑")
    print("="*70)
    
    hw_result = analyzer.holt_winters(
        alpha=0.3, beta=0.1, gamma=0.1,
        seasonal_period=7, n_forecast=30
    )
    
    print(f"\n预测未来30天:")
    print(f"  预测均值: {np.mean(hw_result['forecast']):.2f}")
    print(f"  预测范围: [{np.min(hw_result['forecast']):.2f}, {np.max(hw_result['forecast']):.2f}]")
    
    # 计算评价指标
    metrics = analyzer.compute_metrics(ts_data['value'].values, hw_result['fitted'])
    print(f"\n  模型评价:")
    print(f"    RMSE: {metrics['RMSE']:.4f}")
    print(f"    MAPE: {metrics['MAPE']:.2f}%")
    print(f"    R²:   {metrics['R2']:.4f}")
    
    # 可视化
    forecast_dates_hw = pd.date_range(
        start=ts_data['date'].iloc[-1] + timedelta(days=1),
        periods=30
    )
    
    visualizer.plot_time_series(
        dates=ts_data['date'],
        actual=ts_data['value'].values,
        predicted=hw_result['fitted'],
        forecast_dates=forecast_dates_hw,
        forecast=hw_result['forecast'],
        title="Holt-Winters三次指数平滑预测"
    )
    
    # ================================================================
    # 示例3：多变量回归预测
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 3: 多变量回归预测（销量预测）")
    print("="*70)
    
    # 生成多变量数据
    reg_data = generator.generate_multivariate(n_samples=300)
    print(f"\n数据概览:")
    print(reg_data.describe().round(2))
    
    X = reg_data[['广告投入', '客流量', '促销力度']]
    y = reg_data['销量']
    
    # 训练多个模型
    print("\n--- 线性回归 ---")
    linear_predictor = RegressionPredictor(verbose=True)
    linear_predictor.fit(X, y, model_type='linear')
    
    print("\n--- 随机森林回归 ---")
    rf_predictor = RegressionPredictor(verbose=True)
    rf_predictor.fit(X, y, model_type='random_forest', n_estimators=100)
    
    print("\n--- 梯度提升回归 ---")
    gb_predictor = RegressionPredictor(verbose=True)
    gb_predictor.fit(X, y, model_type='gradient_boosting', n_estimators=100)
    
    # 特征重要性可视化
    if rf_predictor.feature_importance is not None:
        visualizer.plot_feature_importance(
            rf_predictor.feature_importance,
            title="随机森林特征重要性"
        )
    
    # 模型对比
    all_metrics = {
        'Linear': linear_predictor.metrics['test'],
        'RandomForest': rf_predictor.metrics['test'],
        'GradientBoosting': gb_predictor.metrics['test']
    }
    visualizer.plot_model_comparison(all_metrics, title="回归模型性能对比")
    
    # ================================================================
    # 示例4：集成学习
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 4: 集成学习预测")
    print("="*70)
    
    ensemble = EnsemblePredictor(verbose=True)
    ensemble.add_model('Linear', LinearRegression())
    ensemble.add_model('Ridge', Ridge(alpha=1.0))
    ensemble.add_model('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42))
    ensemble.add_model('GradientBoosting', GradientBoostingRegressor(n_estimators=100, random_state=42))
    
    ensemble.fit_all(X.values, y.values, test_size=0.2)
    
    visualizer.plot_model_comparison(ensemble.metrics, title="集成模型性能对比")
    
    # ================================================================
    # 示例5：交叉验证
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 5: 交叉验证")
    print("="*70)
    
    print("\n随机森林交叉验证:")
    rf_predictor.cross_validate(X, y, cv=5)
    
    # ================================================================
    # 使用说明
    # ================================================================
    print("\n" + "="*70)
    print("📖 使用说明 (Usage Guide)")
    print("="*70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                     预测模型使用指南                             │
    └─────────────────────────────────────────────────────────────────┘
    
    【时间序列预测】
    
    1️⃣ 移动平均
       predictor = MovingAveragePredictor()
       predictor.fit_predict(data, window=7, method='simple')
    
    2️⃣ Holt-Winters
       analyzer = TimeSeriesAnalyzer(data)
       result = analyzer.holt_winters(alpha=0.3, beta=0.1, gamma=0.1)
    
    【回归预测】
    
    1️⃣ 单模型
       predictor = RegressionPredictor()
       predictor.fit(X, y, model_type='random_forest')
       predictions = predictor.predict(X_new)
    
    2️⃣ 集成模型
       ensemble = EnsemblePredictor()
       ensemble.add_model('RF', RandomForestRegressor())
       ensemble.fit_all(X, y)
    
    【模型选择建议】
    
    - 数据量小(<100): 线性回归、Ridge
    - 数据量中等: 随机森林
    - 数据量大(>1000): 梯度提升、神经网络
    - 有季节性: Holt-Winters
    
    【论文图表建议】
    
    Figure 1: 时间序列分解图
    Figure 2: 预测结果与实际值对比
    Figure 3: 残差分析图
    Figure 4: 特征重要性
    Figure 5: 模型性能对比（R², RMSE柱状图）
    
    Table 1: 模型参数设置
    Table 2: 预测评价指标（RMSE, MAE, MAPE, R²）
    Table 3: 交叉验证结果
    """)
    
    print("\n" + "="*70)
    print("   ✅ All examples completed successfully!")
    print("   💡 Use the above code templates for your MCM/ICM paper")
    print("="*70)
