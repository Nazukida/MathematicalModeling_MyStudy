"""
============================================================
XGBoost 回归预测
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：高精度回归预测、特征重要性分析、非线性建模
原理：梯度提升决策树 (GBDT)
作者：MCM/ICM Team
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 图表美化设置
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class XGBPredictor:
    """
    XGBoost回归预测器封装类
    
    核心原理：
    - 梯度提升：每棵树拟合前一轮的残差
    - 正则化：L1/L2正则化防止过拟合
    - 并行计算：列采样加速训练
    
    关键参数：
    - learning_rate: 学习率（0.01-0.3）
    - max_depth: 树深度（3-10）
    - n_estimators: 迭代次数（100-1000）
    - subsample: 样本采样比例（0.5-1.0）
    - colsample_bytree: 特征采样比例
    """
    
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=True):
        """
        参数配置
        
        :param n_estimators: 迭代次数
        :param max_depth: 最大深度
        :param learning_rate: 学习率
        :param subsample: 样本采样比例
        :param colsample_bytree: 特征采样比例
        """
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1
        )
        self.verbose = verbose
        self.feature_names = None
        self.feature_importance = None
        self.r2 = None
        self.rmse = None
        self.mae = None
        self.y_test = None
        self.y_pred = None
    
    def fit(self, X, y, test_size=0.2, early_stopping=False):
        """
        训练模型
        
        :param X: 特征DataFrame或数组
        :param y: 标签
        :param test_size: 测试集比例
        :param early_stopping: 是否使用早停
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"特征{i+1}" for i in range(X.shape[1])]
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 训练
        if early_stopping:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        # 预测与评估
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)
        self.r2 = r2_score(y_test, self.y_pred)
        self.rmse = np.sqrt(mean_squared_error(y_test, self.y_pred))
        self.mae = mean_absolute_error(y_test, self.y_pred)
        
        # 特征重要性
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        
        if self.verbose:
            self._print_results()
        
        return self
    
    def _print_results(self):
        """打印结果"""
        print("\n" + "="*50)
        print("🚀 XGBoost 回归结果")
        print("="*50)
        print(f"\n  R² 得分: {self.r2:.4f}")
        print(f"  RMSE: {self.rmse:.4f}")
        print(f"  MAE: {self.mae:.4f}")
        print(f"\n  特征重要性:")
        for name, imp in self.feature_importance.items():
            bar = "█" * int(imp * 30)
            print(f"    {name}: {imp:.4f} {bar}")
        print("="*50)
    
    def cross_validate(self, X, y, cv=5):
        """交叉验证"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        print(f"\n交叉验证 (cv={cv}):")
        print(f"  R² 得分: {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"  各折得分: {scores.round(4)}")
        return scores
    
    def predict(self, X):
        """预测"""
        return self.model.predict(X)
    
    def plot_feature_importance(self, save_path=None):
        """可视化特征重要性"""
        if self.feature_importance is None:
            raise ValueError("请先调用fit()训练模型")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        importance = self.feature_importance.sort_values(ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance)))
        
        bars = ax.barh(importance.index, importance.values, color=colors,
                      edgecolor='white', linewidth=2)
        
        ax.set_xlabel('重要性', fontsize=12, fontweight='bold')
        ax.set_title('XGBoost 特征重要性', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, importance.values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction(self, save_path=None):
        """可视化预测 vs 真实"""
        if self.y_test is None:
            raise ValueError("请先调用fit()训练模型")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：散点图
        ax1 = axes[0]
        ax1.scatter(self.y_test, self.y_pred, alpha=0.6, 
                   color='#2E86AB', edgecolor='white', s=60)
        
        lims = [min(self.y_test.min(), self.y_pred.min()),
                max(self.y_test.max(), self.y_pred.max())]
        ax1.plot(lims, lims, 'r--', linewidth=2, label='理想预测线')
        
        ax1.set_xlabel('真实值', fontsize=12, fontweight='bold')
        ax1.set_ylabel('预测值', fontsize=12, fontweight='bold')
        ax1.set_title(f'预测 vs 真实 (R²={self.r2:.4f})', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：残差分布
        ax2 = axes[1]
        residuals = self.y_test - self.y_pred
        ax2.hist(residuals, bins=30, color='#E94F37', alpha=0.7,
                edgecolor='white', linewidth=1)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax2.set_xlabel('残差', fontsize=12, fontweight='bold')
        ax2.set_ylabel('频数', fontsize=12, fontweight='bold')
        ax2.set_title('残差分布', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("   XGBoost 回归演示 - 销量预测")
    print("="*60)
    
    # 1. 模拟数据（销量与特征的关系）
    np.random.seed(42)
    n = 200
    ad_spend = np.random.uniform(10, 100, n)
    promotion = np.random.randint(1, 6, n)
    traffic = np.random.uniform(500, 2000, n)
    # 非线性关系
    sales = (5 + 0.3*ad_spend + 2*promotion + 0.01*traffic + 
             0.001*ad_spend*promotion + np.random.normal(0, 2, n))
    
    data = pd.DataFrame({
        "广告投入": ad_spend,
        "促销力度": promotion,
        "客流量": traffic,
        "销量": sales
    })
    
    print("\n数据概览：")
    print(data.describe().round(2))
    
    # 2. 训练模型
    X = data[["广告投入", "促销力度", "客流量"]]
    y = data["销量"]
    
    xgb = XGBPredictor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        verbose=True
    )
    xgb.fit(X, y, test_size=0.2)
    
    # 3. 交叉验证
    xgb.cross_validate(X, y, cv=5)
    
    # 4. 可视化
    xgb.plot_feature_importance()
    xgb.plot_prediction()
    
    # 5. 新样本预测
    new_data = pd.DataFrame({
        "广告投入": [50, 80],
        "促销力度": [3, 5],
        "客流量": [1000, 1500]
    })
    predictions = xgb.predict(new_data)
    print(f"\n新样本预测销量: {predictions.round(2)}")
