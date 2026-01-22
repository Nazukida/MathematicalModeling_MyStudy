"""
============================================================
预测模型完整教程 (Comprehensive Prediction Tutorial)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
本教程展示如何将数据预处理、预测模型、可视化完整串联起来

包含内容：
1. 数据预处理模块 (Data Preprocessing)
   - 数据加载与清洗
   - 平稳性检验
   - 数据标准化
2. 预测模型 (Prediction Models)
   - 移动平均法 (Moving Average)
   - 指数平滑法 (Exponential Smoothing)
   - ARIMA时间序列预测
   - 灰色预测 GM(1,1)
   - 回归预测 (随机森林/XGBoost)
3. 可视化模块 (Visualization)
4. 模型评价与对比
5. 完整案例演示

作者：MCM/ICM Team
日期：2026年1月22日
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings('ignore')

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.figsize'] = (12, 6)
rcParams['figure.dpi'] = 100


# ============================================================
# 第一部分：完整工作流程概览
# ============================================================

def print_workflow():
    """打印完整工作流程"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              预测模型完整工作流程 (Prediction Workflow)                   ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                          ║
    ║   ┌─────────────────────────────────────────────────────────────────┐    ║
    ║   │  Step 1: 数据准备 (Data Preparation)                            │    ║
    ║   │  ├─ 加载时间序列或多变量数据                                     │    ║
    ║   │  ├─ 缺失值处理（插值/填充）                                      │    ║
    ║   │  └─ 异常值检测与处理                                             │    ║
    ║   └─────────────────────────────────────────────────────────────────┘    ║
    ║                            ↓                                             ║
    ║   ┌─────────────────────────────────────────────────────────────────┐    ║
    ║   │  Step 2: 数据分析 (Data Analysis)                               │    ║
    ║   │  ├─ 平稳性检验（ADF检验）                                        │    ║
    ║   │  ├─ 自相关分析（ACF/PACF图）                                     │    ║
    ║   │  └─ 趋势与季节性分解                                             │    ║
    ║   └─────────────────────────────────────────────────────────────────┘    ║
    ║                            ↓                                             ║
    ║   ┌─────────────────────────────────────────────────────────────────┐    ║
    ║   │  Step 3: 模型选择与拟合 (Model Selection & Fitting)             │    ║
    ║   │  ├─ 小样本 → 灰色预测 GM(1,1)                                    │    ║
    ║   │  ├─ 时间序列 → ARIMA / 指数平滑                                  │    ║
    ║   │  └─ 多变量 → 回归 / 随机森林 / XGBoost                           │    ║
    ║   └─────────────────────────────────────────────────────────────────┘    ║
    ║                            ↓                                             ║
    ║   ┌─────────────────────────────────────────────────────────────────┐    ║
    ║   │  Step 4: 预测与评价 (Prediction & Evaluation)                   │    ║
    ║   │  ├─ 样本内拟合（In-sample fitting）                              │    ║
    ║   │  ├─ 样本外预测（Out-of-sample forecast）                         │    ║
    ║   │  └─ 评价指标（RMSE, MAE, MAPE, R²）                              │    ║
    ║   └─────────────────────────────────────────────────────────────────┘    ║
    ║                            ↓                                             ║
    ║   ┌─────────────────────────────────────────────────────────────────┐    ║
    ║   │  Step 5: 可视化输出 (Visualization)                             │    ║
    ║   │  ├─ 拟合与预测曲线图                                             │    ║
    ║   │  ├─ 残差分析图                                                   │    ║
    ║   │  ├─ 置信区间图                                                   │    ║
    ║   │  └─ 多模型对比图                                                 │    ║
    ║   └─────────────────────────────────────────────────────────────────┘    ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)


# ============================================================
# 第二部分：数据预处理类
# ============================================================

class PredictionDataPreprocessor:
    """
    预测数据预处理器
    功能：数据加载、缺失值处理、平稳性检验、数据标准化
    """
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.dates = None
        self.values = None
        self.is_stationary = None
        self.preprocessing_log = []
    
    def load_data(self, data, date_col='date', value_col='value'):
        """
        加载时间序列数据
        
        :param data: DataFrame、数组或CSV文件路径
        :param date_col: 日期列名
        :param value_col: 值列名
        :return: self
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
            self.raw_data = df
        elif isinstance(data, pd.DataFrame):
            self.raw_data = data.copy()
        elif isinstance(data, (list, np.ndarray)):
            self.raw_data = pd.DataFrame({
                'date': pd.date_range(start='2024-01-01', periods=len(data), freq='D'),
                'value': data
            })
            date_col, value_col = 'date', 'value'
        
        if date_col in self.raw_data.columns:
            self.dates = pd.to_datetime(self.raw_data[date_col])
        else:
            self.dates = pd.date_range(start='2024-01-01', periods=len(self.raw_data), freq='D')
        
        if value_col in self.raw_data.columns:
            self.values = self.raw_data[value_col].values.astype(float)
        else:
            self.values = self.raw_data.iloc[:, -1].values.astype(float)
        
        self.processed_data = self.values.copy()
        self.preprocessing_log.append("数据加载完成")
        
        print(f"✅ 数据加载成功：{len(self.values)}个数据点")
        return self
    
    def generate_demo_data(self, n_periods=100, pattern='trend_seasonal', noise_level=5):
        """
        生成演示数据
        
        :param n_periods: 数据点数量
        :param pattern: 数据模式 ('trend', 'seasonal', 'trend_seasonal', 'random')
        :param noise_level: 噪声水平
        """
        np.random.seed(42)
        t = np.arange(n_periods)
        
        if pattern == 'trend':
            values = 100 + 0.5 * t + np.random.normal(0, noise_level, n_periods)
        elif pattern == 'seasonal':
            values = 100 + 15 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, noise_level, n_periods)
        elif pattern == 'trend_seasonal':
            values = 100 + 0.3 * t + 15 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, noise_level, n_periods)
        elif pattern == 'random':
            values = 100 + np.cumsum(np.random.normal(0, noise_level, n_periods))
        else:
            values = 100 + np.random.normal(0, noise_level, n_periods)
        
        self.dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='D')
        self.values = values
        self.processed_data = values.copy()
        self.raw_data = pd.DataFrame({'date': self.dates, 'value': values})
        
        print(f"✅ 生成{pattern}模式演示数据：{n_periods}个数据点")
        return self
    
    def handle_missing_values(self, method='interpolate'):
        """
        处理缺失值
        
        :param method: 'interpolate'(插值), 'ffill'(前向填充), 'mean'(均值填充)
        """
        if np.isnan(self.processed_data).any():
            if method == 'interpolate':
                series = pd.Series(self.processed_data)
                self.processed_data = series.interpolate().values
            elif method == 'ffill':
                series = pd.Series(self.processed_data)
                self.processed_data = series.fillna(method='ffill').values
            elif method == 'mean':
                mean_val = np.nanmean(self.processed_data)
                self.processed_data = np.where(np.isnan(self.processed_data), mean_val, self.processed_data)
            
            self.preprocessing_log.append(f"缺失值处理：{method}")
            print(f"✅ 缺失值已使用 {method} 方法处理")
        else:
            print("✅ 无缺失值")
        return self
    
    def adf_test(self, significance=0.05):
        """
        ADF平稳性检验
        
        :param significance: 显著性水平
        :return: 是否平稳
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(self.processed_data)
            
            adf_stat = result[0]
            p_value = result[1]
            critical_values = result[4]
            
            self.is_stationary = p_value < significance
            
            print("\n" + "="*50)
            print("📊 ADF平稳性检验结果")
            print("="*50)
            print(f"  ADF统计量: {adf_stat:.4f}")
            print(f"  p值: {p_value:.4f}")
            print(f"  临界值:")
            for key, val in critical_values.items():
                print(f"    {key}: {val:.4f}")
            print(f"  结论: 序列{'平稳' if self.is_stationary else '不平稳'}")
            print("="*50)
            
            return self.is_stationary
        except ImportError:
            print("⚠️ 需要安装statsmodels库进行ADF检验")
            return None
    
    def difference(self, order=1):
        """
        差分处理
        
        :param order: 差分阶数
        """
        for _ in range(order):
            self.processed_data = np.diff(self.processed_data)
            self.dates = self.dates[1:]
        
        self.preprocessing_log.append(f"差分处理：{order}阶")
        print(f"✅ 已进行{order}阶差分")
        return self
    
    def normalize(self, method='minmax'):
        """
        数据标准化
        
        :param method: 'minmax' / 'zscore'
        """
        if method == 'minmax':
            min_val = self.processed_data.min()
            max_val = self.processed_data.max()
            self.processed_data = (self.processed_data - min_val) / (max_val - min_val + 1e-10)
        elif method == 'zscore':
            mean_val = self.processed_data.mean()
            std_val = self.processed_data.std()
            self.processed_data = (self.processed_data - mean_val) / (std_val + 1e-10)
        
        self.preprocessing_log.append(f"数据标准化：{method}")
        return self
    
    def get_data(self):
        """获取处理后的数据"""
        return pd.DataFrame({
            'date': self.dates[:len(self.processed_data)],
            'value': self.processed_data
        })
    
    def summary(self):
        """打印数据摘要"""
        print("\n" + "="*60)
        print("📊 数据预处理摘要")
        print("="*60)
        print(f"  数据点数量: {len(self.processed_data)}")
        print(f"  时间范围: {self.dates.min()} ~ {self.dates.max()}")
        print(f"  数值范围: [{self.processed_data.min():.2f}, {self.processed_data.max():.2f}]")
        print(f"  均值: {self.processed_data.mean():.2f}")
        print(f"  标准差: {self.processed_data.std():.2f}")
        print(f"  预处理步骤: {self.preprocessing_log}")
        print("="*60)


# ============================================================
# 第三部分：预测模型
# ============================================================

class MovingAverageModel:
    """移动平均预测模型"""
    
    def __init__(self, window=7):
        self.window = window
        self.fitted = None
        self.forecast = None
        self.metrics = None
    
    def fit_predict(self, data, n_forecast=10):
        """拟合并预测"""
        values = np.array(data) if not isinstance(data, np.ndarray) else data
        n = len(values)
        
        # 拟合
        self.fitted = np.zeros(n)
        self.fitted[:self.window] = np.nan
        for t in range(self.window, n):
            self.fitted[t] = np.mean(values[t-self.window:t])
        
        # 预测
        self.forecast = np.zeros(n_forecast)
        last_values = list(values[-self.window:])
        for i in range(n_forecast):
            self.forecast[i] = np.mean(last_values)
            last_values.pop(0)
            last_values.append(self.forecast[i])
        
        # 评价
        valid_idx = ~np.isnan(self.fitted)
        self.metrics = self._compute_metrics(values[valid_idx], self.fitted[valid_idx])
        
        return self
    
    def _compute_metrics(self, actual, predicted):
        """计算评价指标"""
        return {
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
            'MAE': mean_absolute_error(actual, predicted),
            'MAPE': np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100,
            'R2': r2_score(actual, predicted)
        }
    
    def get_results(self):
        """获取结果"""
        return {
            'fitted': self.fitted,
            'forecast': self.forecast,
            'metrics': self.metrics
        }


class ExponentialSmoothingModel:
    """指数平滑预测模型"""
    
    def __init__(self, alpha=0.3, beta=None, gamma=None, seasonal_period=None):
        """
        :param alpha: 水平平滑系数
        :param beta: 趋势平滑系数（Holt方法）
        :param gamma: 季节平滑系数（Holt-Winters方法）
        :param seasonal_period: 季节周期
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonal_period = seasonal_period
        self.fitted = None
        self.forecast = None
        self.metrics = None
        self.method = 'simple'
        
        if beta is not None and gamma is not None:
            self.method = 'holt_winters'
        elif beta is not None:
            self.method = 'holt'
    
    def fit_predict(self, data, n_forecast=10):
        """拟合并预测"""
        values = np.array(data) if not isinstance(data, np.ndarray) else data
        n = len(values)
        
        if self.method == 'simple':
            # 简单指数平滑
            self.fitted = np.zeros(n)
            self.fitted[0] = values[0]
            for t in range(1, n):
                self.fitted[t] = self.alpha * values[t] + (1 - self.alpha) * self.fitted[t-1]
            
            # 预测（简单指数平滑预测为常数）
            self.forecast = np.full(n_forecast, self.fitted[-1])
        
        elif self.method == 'holt':
            # Holt双参数指数平滑
            level = np.zeros(n)
            trend = np.zeros(n)
            self.fitted = np.zeros(n)
            
            level[0] = values[0]
            trend[0] = values[1] - values[0] if n > 1 else 0
            
            for t in range(1, n):
                level[t] = self.alpha * values[t] + (1 - self.alpha) * (level[t-1] + trend[t-1])
                trend[t] = self.beta * (level[t] - level[t-1]) + (1 - self.beta) * trend[t-1]
                self.fitted[t] = level[t-1] + trend[t-1]
            
            self.fitted[0] = values[0]
            
            # 预测
            self.forecast = np.zeros(n_forecast)
            for h in range(n_forecast):
                self.forecast[h] = level[-1] + (h + 1) * trend[-1]
        
        elif self.method == 'holt_winters':
            # Holt-Winters三参数（加法模型）
            m = self.seasonal_period or 12
            
            level = np.zeros(n)
            trend = np.zeros(n)
            seasonal = np.zeros(n + n_forecast)
            self.fitted = np.zeros(n)
            
            # 初始化
            level[0] = np.mean(values[:m])
            trend[0] = (np.mean(values[m:2*m]) - np.mean(values[:m])) / m if n >= 2*m else 0
            for i in range(m):
                seasonal[i] = values[i] - level[0] if i < n else 0
            
            for t in range(1, n):
                if t >= m:
                    level[t] = self.alpha * (values[t] - seasonal[t-m]) + (1 - self.alpha) * (level[t-1] + trend[t-1])
                    trend[t] = self.beta * (level[t] - level[t-1]) + (1 - self.beta) * trend[t-1]
                    seasonal[t] = self.gamma * (values[t] - level[t]) + (1 - self.gamma) * seasonal[t-m]
                    self.fitted[t] = level[t-1] + trend[t-1] + seasonal[t-m]
                else:
                    level[t] = self.alpha * values[t] + (1 - self.alpha) * (level[t-1] + trend[t-1])
                    trend[t] = self.beta * (level[t] - level[t-1]) + (1 - self.beta) * trend[t-1]
                    self.fitted[t] = level[t-1] + trend[t-1]
            
            self.fitted[0] = values[0]
            
            # 预测
            self.forecast = np.zeros(n_forecast)
            for h in range(n_forecast):
                self.forecast[h] = level[-1] + (h + 1) * trend[-1] + seasonal[n - m + (h % m)]
        
        # 评价
        valid_idx = ~np.isnan(self.fitted) & (self.fitted != 0)
        if valid_idx.sum() > 0:
            self.metrics = self._compute_metrics(values[valid_idx], self.fitted[valid_idx])
        
        return self
    
    def _compute_metrics(self, actual, predicted):
        """计算评价指标"""
        return {
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
            'MAE': mean_absolute_error(actual, predicted),
            'MAPE': np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100,
            'R2': r2_score(actual, predicted)
        }
    
    def get_results(self):
        """获取结果"""
        return {
            'fitted': self.fitted,
            'forecast': self.forecast,
            'metrics': self.metrics,
            'method': self.method
        }


class GreyPredictionModel:
    """灰色预测模型 GM(1,1)"""
    
    def __init__(self):
        self.a = None  # 发展系数
        self.b = None  # 灰作用量
        self.fitted = None
        self.forecast = None
        self.metrics = None
        self.C = None  # 后验差比
        self.P = None  # 小误差概率
    
    def fit_predict(self, data, n_forecast=3):
        """
        拟合并预测
        
        :param data: 原始数据（至少4个数据点）
        :param n_forecast: 预测步数
        """
        x0 = np.array(data, dtype=np.float64)
        n = len(x0)
        
        if n < 4:
            print("⚠️ 灰色预测至少需要4个数据点")
            return self
        
        # 1. 累加生成
        x1 = np.cumsum(x0)
        
        # 2. 构造矩阵
        B = np.zeros((n-1, 2))
        Y = np.zeros((n-1, 1))
        
        for i in range(n-1):
            B[i, 0] = -0.5 * (x1[i] + x1[i+1])
            B[i, 1] = 1
            Y[i, 0] = x0[i+1]
        
        # 3. 最小二乘估计参数
        BT = B.T
        params = np.dot(np.dot(np.linalg.inv(np.dot(BT, B)), BT), Y)
        self.a = params[0, 0]
        self.b = params[1, 0]
        
        # 4. 拟合
        self.fitted = np.zeros(n)
        self.fitted[0] = x0[0]
        for k in range(1, n):
            x1_k = (x0[0] - self.b/self.a) * np.exp(-self.a * k) + self.b/self.a
            x1_k_1 = (x0[0] - self.b/self.a) * np.exp(-self.a * (k-1)) + self.b/self.a
            self.fitted[k] = x1_k - x1_k_1
        
        # 5. 预测
        self.forecast = np.zeros(n_forecast)
        for i in range(n_forecast):
            k = n + i
            x1_k = (x0[0] - self.b/self.a) * np.exp(-self.a * k) + self.b/self.a
            x1_k_1 = (x0[0] - self.b/self.a) * np.exp(-self.a * (k-1)) + self.b/self.a
            self.forecast[i] = x1_k - x1_k_1
        
        # 6. 模型检验
        residual = x0 - self.fitted
        s1 = np.std(x0, ddof=1)
        s2 = np.std(residual, ddof=1)
        self.C = s2 / s1 if s1 != 0 else 0
        self.P = np.mean(np.abs(residual - np.mean(residual)) < 0.6745 * s1)
        
        # 7. 评价指标
        self.metrics = self._compute_metrics(x0, self.fitted)
        self.metrics['C'] = self.C
        self.metrics['P'] = self.P
        self.metrics['Grade'] = self._get_grade()
        
        return self
    
    def _compute_metrics(self, actual, predicted):
        """计算评价指标"""
        return {
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
            'MAE': mean_absolute_error(actual, predicted),
            'MAPE': np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
        }
    
    def _get_grade(self):
        """获取模型精度等级"""
        if self.C < 0.35 and self.P > 0.95:
            return "好"
        elif self.C < 0.5 and self.P > 0.8:
            return "合格"
        elif self.C < 0.65 and self.P > 0.7:
            return "勉强"
        else:
            return "不合格"
    
    def get_results(self):
        """获取结果"""
        return {
            'fitted': self.fitted,
            'forecast': self.forecast,
            'metrics': self.metrics,
            'a': self.a,
            'b': self.b
        }


class RegressionPredictionModel:
    """回归预测模型（随机森林/梯度提升）"""
    
    def __init__(self, model_type='random_forest', **kwargs):
        """
        :param model_type: 'random_forest' / 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = None
        self.metrics = None
        self.feature_importance = None
        self.kwargs = kwargs
    
    def fit(self, X, y, test_size=0.2):
        """
        拟合模型
        
        :param X: 特征矩阵
        :param y: 目标变量
        :param test_size: 测试集比例
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
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 选择模型
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=self.kwargs.get('n_estimators', 100),
                max_depth=self.kwargs.get('max_depth', None),
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=self.kwargs.get('n_estimators', 100),
                learning_rate=self.kwargs.get('learning_rate', 0.1),
                random_state=42
            )
        
        # 训练
        self.model.fit(X_train_scaled, y_train)
        
        # 预测
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # 评价
        self.metrics = {
            'train': self._compute_metrics(y_train, y_train_pred),
            'test': self._compute_metrics(y_test, y_test_pred)
        }
        
        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        return self
    
    def predict(self, X):
        """预测"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def _compute_metrics(self, actual, predicted):
        """计算评价指标"""
        return {
            'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
            'MAE': mean_absolute_error(actual, predicted),
            'R2': r2_score(actual, predicted)
        }
    
    def get_results(self):
        """获取结果"""
        return {
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }


# ============================================================
# 第四部分：可视化模块
# ============================================================

class PredictionVisualizer:
    """预测可视化器"""
    
    COLORS = {
        'actual': '#2E86AB',
        'fitted': '#A23B72',
        'forecast': '#F18F01',
        'confidence': '#C73E1D'
    }
    
    @staticmethod
    def plot_forecast(dates, actual, fitted, forecast, 
                      title="预测结果", confidence_interval=None, save_path=None):
        """
        绘制预测结果图
        
        :param dates: 日期序列
        :param actual: 实际值
        :param fitted: 拟合值
        :param forecast: 预测值
        :param title: 标题
        :param confidence_interval: 置信区间 (lower, upper)
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        n = len(actual)
        n_forecast = len(forecast)
        
        # 实际值
        ax.plot(range(n), actual, 'o-', color=PredictionVisualizer.COLORS['actual'],
               label='实际值', linewidth=2, markersize=4)
        
        # 拟合值
        valid_idx = ~np.isnan(fitted)
        ax.plot(np.where(valid_idx)[0], fitted[valid_idx], '--',
               color=PredictionVisualizer.COLORS['fitted'], label='拟合值', linewidth=2)
        
        # 预测值
        forecast_x = range(n, n + n_forecast)
        ax.plot(forecast_x, forecast, 's-', color=PredictionVisualizer.COLORS['forecast'],
               label='预测值', linewidth=2, markersize=6)
        
        # 置信区间
        if confidence_interval is not None:
            lower, upper = confidence_interval
            ax.fill_between(forecast_x, lower, upper, 
                          color=PredictionVisualizer.COLORS['forecast'], alpha=0.2,
                          label='95%置信区间')
        
        ax.axvline(x=n-0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('时间', fontsize=12, fontweight='bold')
        ax.set_ylabel('值', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_residuals(actual, fitted, title="残差分析", save_path=None):
        """绘制残差分析图"""
        valid_idx = ~np.isnan(fitted)
        residuals = actual[valid_idx] - fitted[valid_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 残差时序图
        ax1 = axes[0, 0]
        ax1.plot(residuals, 'o-', color='#2E86AB', markersize=4)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('时间', fontweight='bold')
        ax1.set_ylabel('残差', fontweight='bold')
        ax1.set_title('(a) 残差时序图', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 残差直方图
        ax2 = axes[0, 1]
        ax2.hist(residuals, bins=15, color='#A23B72', edgecolor='white', density=True)
        # 添加正态分布拟合曲线
        mu, std = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label='正态分布')
        ax2.set_xlabel('残差', fontweight='bold')
        ax2.set_ylabel('频率', fontweight='bold')
        ax2.set_title('(b) 残差分布', fontweight='bold')
        ax2.legend()
        
        # Q-Q图
        ax3 = axes[1, 0]
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('(c) Q-Q图', fontweight='bold')
        
        # 残差自相关图
        ax4 = axes[1, 1]
        n = len(residuals)
        lags = min(20, n // 2)
        acf = [np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1] if lag > 0 else 1 
               for lag in range(lags)]
        ax4.bar(range(lags), acf, color='#F18F01', edgecolor='white')
        ax4.axhline(y=1.96/np.sqrt(n), color='red', linestyle='--')
        ax4.axhline(y=-1.96/np.sqrt(n), color='red', linestyle='--')
        ax4.set_xlabel('滞后阶数', fontweight='bold')
        ax4.set_ylabel('自相关系数', fontweight='bold')
        ax4.set_title('(d) 残差自相关图', fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_model_comparison(models_results, model_names, actual, title="模型对比", save_path=None):
        """
        绘制多模型对比图
        
        :param models_results: 各模型结果列表
        :param model_names: 模型名称列表
        :param actual: 实际值
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4C9A', '#1B998B']
        
        # 子图1: 拟合曲线对比
        ax1 = axes[0]
        ax1.plot(actual, 'ko-', label='实际值', linewidth=2, markersize=4)
        
        for i, (result, name) in enumerate(zip(models_results, model_names)):
            fitted = result.get('fitted', result.get('train_pred', None))
            if fitted is not None:
                valid_idx = ~np.isnan(fitted)
                ax1.plot(np.where(valid_idx)[0], fitted[valid_idx], '--',
                        color=colors[i % len(colors)], label=name, linewidth=2)
        
        ax1.set_xlabel('时间', fontweight='bold')
        ax1.set_ylabel('值', fontweight='bold')
        ax1.set_title('(a) 拟合效果对比', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 评价指标对比
        ax2 = axes[1]
        metrics_names = ['RMSE', 'MAE', 'MAPE']
        x = np.arange(len(metrics_names))
        width = 0.8 / len(model_names)
        
        for i, (result, name) in enumerate(zip(models_results, model_names)):
            metrics = result.get('metrics', {})
            values = [metrics.get(m, 0) for m in metrics_names]
            ax2.bar(x + i * width, values, width, label=name, color=colors[i % len(colors)])
        
        ax2.set_xlabel('评价指标', fontweight='bold')
        ax2.set_ylabel('值', fontweight='bold')
        ax2.set_title('(b) 评价指标对比', fontweight='bold')
        ax2.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax2.set_xticklabels(metrics_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_full_report(actual, results_dict, title="预测分析报告", save_path=None):
        """
        生成完整预测报告
        
        :param actual: 实际值
        :param results_dict: 包含fitted, forecast, metrics的字典
        """
        fig = plt.figure(figsize=(16, 12))
        
        fitted = results_dict.get('fitted', np.zeros_like(actual))
        forecast = results_dict.get('forecast', [])
        metrics = results_dict.get('metrics', {})
        
        n = len(actual)
        n_forecast = len(forecast) if forecast is not None else 0
        
        # 子图1: 预测结果
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(range(n), actual, 'o-', color='#2E86AB', label='实际值', markersize=4)
        valid_idx = ~np.isnan(fitted)
        ax1.plot(np.where(valid_idx)[0], fitted[valid_idx], '--', color='#A23B72', label='拟合值')
        if n_forecast > 0:
            ax1.plot(range(n, n + n_forecast), forecast, 's-', color='#F18F01', label='预测值')
            ax1.axvline(x=n-0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('时间', fontweight='bold')
        ax1.set_ylabel('值', fontweight='bold')
        ax1.set_title('(a) 预测结果', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 评价指标
        ax2 = fig.add_subplot(2, 2, 2)
        metric_names = ['RMSE', 'MAE', 'MAPE']
        metric_values = [metrics.get(m, 0) for m in metric_names]
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        bars = ax2.bar(metric_names, metric_values, color=colors, edgecolor='white')
        for bar, val in zip(bars, metric_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax2.set_ylabel('值', fontweight='bold')
        ax2.set_title('(b) 评价指标', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 子图3: 残差分布
        ax3 = fig.add_subplot(2, 2, 3)
        residuals = actual[valid_idx] - fitted[valid_idx]
        ax3.hist(residuals, bins=15, color='#6B4C9A', edgecolor='white', density=True)
        mu, std = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax3.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2)
        ax3.set_xlabel('残差', fontweight='bold')
        ax3.set_ylabel('频率', fontweight='bold')
        ax3.set_title(f'(c) 残差分布 (μ={mu:.2f}, σ={std:.2f})', fontweight='bold')
        
        # 子图4: 拟合vs实际散点图
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.scatter(actual[valid_idx], fitted[valid_idx], c='#1B998B', alpha=0.6, s=50)
        # 添加对角线
        min_val = min(actual[valid_idx].min(), fitted[valid_idx].min())
        max_val = max(actual[valid_idx].max(), fitted[valid_idx].max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想拟合线')
        ax4.set_xlabel('实际值', fontweight='bold')
        ax4.set_ylabel('拟合值', fontweight='bold')
        r2 = metrics.get('R2', r2_score(actual[valid_idx], fitted[valid_idx]))
        ax4.set_title(f'(d) 拟合效果 (R²={r2:.4f})', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 第五部分：完整案例演示
# ============================================================

def run_complete_example():
    """运行完整的预测案例"""
    
    print_workflow()
    
    print("\n" + "="*70)
    print("🎯 预测模型完整案例：时间序列销量预测")
    print("="*70)
    
    # ========================================
    # Step 1: 数据准备
    # ========================================
    print("\n" + "-"*50)
    print("📊 Step 1: 数据准备")
    print("-"*50)
    
    preprocessor = PredictionDataPreprocessor()
    preprocessor.generate_demo_data(n_periods=100, pattern='trend_seasonal', noise_level=5)
    preprocessor.summary()
    
    data = preprocessor.get_data()
    actual = preprocessor.values
    
    # ========================================
    # Step 2: 数据分析
    # ========================================
    print("\n" + "-"*50)
    print("📊 Step 2: 数据分析")
    print("-"*50)
    
    preprocessor.adf_test()
    
    # ========================================
    # Step 3: 模型拟合与预测
    # ========================================
    print("\n" + "-"*50)
    print("📊 Step 3: 模型拟合与预测")
    print("-"*50)
    
    n_forecast = 10
    
    # 3.1 移动平均
    print("\n【1. 移动平均法】")
    ma_model = MovingAverageModel(window=7)
    ma_model.fit_predict(actual, n_forecast=n_forecast)
    ma_results = ma_model.get_results()
    print(f"  RMSE: {ma_results['metrics']['RMSE']:.4f}")
    print(f"  MAE:  {ma_results['metrics']['MAE']:.4f}")
    print(f"  MAPE: {ma_results['metrics']['MAPE']:.2f}%")
    
    # 3.2 指数平滑（Holt方法）
    print("\n【2. 指数平滑法（Holt双参数）】")
    es_model = ExponentialSmoothingModel(alpha=0.3, beta=0.1)
    es_model.fit_predict(actual, n_forecast=n_forecast)
    es_results = es_model.get_results()
    print(f"  RMSE: {es_results['metrics']['RMSE']:.4f}")
    print(f"  MAE:  {es_results['metrics']['MAE']:.4f}")
    print(f"  MAPE: {es_results['metrics']['MAPE']:.2f}%")
    
    # 3.3 灰色预测（使用最后10个数据点）
    print("\n【3. 灰色预测 GM(1,1)】")
    grey_model = GreyPredictionModel()
    grey_model.fit_predict(actual[-10:], n_forecast=3)
    grey_results = grey_model.get_results()
    print(f"  参数: a={grey_results['a']:.4f}, b={grey_results['b']:.4f}")
    print(f"  后验差比C: {grey_results['metrics']['C']:.4f}")
    print(f"  小误差概率P: {grey_results['metrics']['P']:.4f}")
    print(f"  模型精度等级: {grey_results['metrics']['Grade']}")
    
    # ========================================
    # Step 4: 可视化分析
    # ========================================
    print("\n" + "-"*50)
    print("📊 Step 4: 可视化分析")
    print("-"*50)
    
    visualizer = PredictionVisualizer()
    
    # 预测结果图
    visualizer.plot_forecast(
        preprocessor.dates, actual, es_results['fitted'], es_results['forecast'],
        title="指数平滑预测结果 (Holt方法)"
    )
    
    # 残差分析
    visualizer.plot_residuals(actual, es_results['fitted'], title="指数平滑模型残差分析")
    
    # 模型对比
    visualizer.plot_model_comparison(
        [ma_results, es_results],
        ['移动平均', 'Holt指数平滑'],
        actual,
        title="预测模型对比"
    )
    
    # 完整报告
    visualizer.plot_full_report(actual, es_results, title="预测分析完整报告")
    
    # ========================================
    # 结论
    # ========================================
    print("\n" + "="*70)
    print("🏆 预测结论")
    print("="*70)
    
    print(f"\n移动平均法: RMSE={ma_results['metrics']['RMSE']:.4f}")
    print(f"指数平滑法: RMSE={es_results['metrics']['RMSE']:.4f}")
    
    if es_results['metrics']['RMSE'] < ma_results['metrics']['RMSE']:
        print("\n✅ 指数平滑法表现更好，推荐使用")
    else:
        print("\n✅ 移动平均法表现更好，推荐使用")
    
    print(f"\n未来{n_forecast}期预测值:")
    print(f"  {es_results['forecast'].round(2)}")
    
    print("\n" + "="*70)
    print("   ✅ 预测分析完成！")
    print("="*70)
    
    return {
        'actual': actual,
        'ma_results': ma_results,
        'es_results': es_results,
        'grey_results': grey_results
    }


# ============================================================
# 第六部分：使用指南
# ============================================================

def print_usage_guide():
    """打印使用指南"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                        预测模型使用指南                                   ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                          ║
    ║  【快速开始】                                                            ║
    ║                                                                          ║
    ║  from comprehensive_prediction_tutorial import *                         ║
    ║                                                                          ║
    ║  # 1. 准备数据                                                           ║
    ║  preprocessor = PredictionDataPreprocessor()                             ║
    ║  preprocessor.load_data(your_data)  # 或 generate_demo_data()            ║
    ║                                                                          ║
    ║  # 2. 数据分析                                                           ║
    ║  preprocessor.adf_test()  # 平稳性检验                                   ║
    ║                                                                          ║
    ║  # 3. 模型选择与拟合                                                     ║
    ║  model = ExponentialSmoothingModel(alpha=0.3, beta=0.1)                  ║
    ║  model.fit_predict(preprocessor.values, n_forecast=10)                   ║
    ║                                                                          ║
    ║  # 4. 获取结果                                                           ║
    ║  results = model.get_results()                                           ║
    ║  print(results['forecast'])  # 预测值                                    ║
    ║  print(results['metrics'])   # 评价指标                                  ║
    ║                                                                          ║
    ║  # 5. 可视化                                                             ║
    ║  visualizer = PredictionVisualizer()                                     ║
    ║  visualizer.plot_full_report(actual, results)                            ║
    ║                                                                          ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  【模型选择建议】                                                        ║
    ║  - 数据量<10: 灰色预测 GM(1,1)                                           ║
    ║  - 无趋势无季节: 简单指数平滑                                            ║
    ║  - 有趋势无季节: Holt双参数 (alpha, beta)                                ║
    ║  - 有趋势有季节: Holt-Winters (alpha, beta, gamma)                       ║
    ║  - 多变量预测: 回归模型 / 随机森林                                       ║
    ║                                                                          ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  【论文图表建议】                                                        ║
    ║  Figure 1: 原始数据时序图                                                ║
    ║  Figure 2: ACF/PACF分析图（时间序列模型）                                ║
    ║  Figure 3: 拟合与预测结果图                                              ║
    ║  Figure 4: 残差分析图                                                    ║
    ║  Figure 5: 多模型对比图                                                  ║
    ║                                                                          ║
    ║  Table 1: 数据描述性统计                                                 ║
    ║  Table 2: 模型参数                                                       ║
    ║  Table 3: 预测结果对比（RMSE, MAE, MAPE）                                ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    # 运行完整案例
    results = run_complete_example()
    
    # 打印使用指南
    print_usage_guide()
