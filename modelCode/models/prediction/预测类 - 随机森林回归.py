"""
============================================================
随机森林回归 (Random Forest Regression)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：多变量回归预测、特征重要性分析、非线性关系建模
原理：集成多棵回归树，取平均值作为预测
作者：MCM/ICM Team
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 图表美化设置
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class RFRegressor:
    """
    随机森林回归器封装类
    
    优点：
    - 捕捉非线性关系
    - 自动处理特征交互
    - 输出特征重要性
    - 抗过拟合能力强
    
    参数说明：
    - n_estimators: 决策树数量（100-500）
    - max_depth: 最大深度（None不限制，5-20常用）
    - min_samples_split: 分裂最小样本数
    """
    
    def __init__(self, n_estimators=100, max_depth=None, 
                 min_samples_split=2, random_state=42, verbose=True):
        """
        参数配置
        
        :param n_estimators: 决策树数量
        :param max_depth: 最大深度
        :param min_samples_split: 最小分裂样本数
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
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
    
    def fit(self, X, y, test_size=0.2):
        """
        训练模型
        
        :param X: 特征DataFrame或数组
        :param y: 标签
        :param test_size: 测试集比例
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
        print("🌲 随机森林回归结果")
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
        ax.set_title('随机森林特征重要性（回归）', fontsize=14, fontweight='bold')
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
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(self.y_test, self.y_pred, alpha=0.6, 
                  color='#2E86AB', edgecolor='white', s=60)
        
        # 理想线
        lims = [min(self.y_test.min(), self.y_pred.min()),
                max(self.y_test.max(), self.y_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='理想预测线')
        
        ax.set_xlabel('真实值', fontsize=12, fontweight='bold')
        ax.set_ylabel('预测值', fontsize=12, fontweight='bold')
        ax.set_title(f'预测 vs 真实 (R²={self.r2:.4f})', fontsize=14, fontweight='bold')
        ax.legend()
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
    print("   随机森林回归演示 - 销量预测")
    print("="*60)
    
    # 1. 模拟数据（销量与广告投入、客流量的关系）
    np.random.seed(42)
    n_samples = 200
    ad_spend = np.random.uniform(10, 100, n_samples)
    traffic = np.random.uniform(500, 2000, n_samples)
    # 非线性关系
    sales = 5 + 0.3*ad_spend + 0.01*traffic + 0.001*ad_spend*traffic/10 + np.random.normal(0, 2, n_samples)
    
    data = pd.DataFrame({
        "广告投入": ad_spend,
        "客流量": traffic,
        "销量": sales
    })
    
    print("\n数据概览：")
    print(data.describe().round(2))
    
    # 2. 训练模型
    X = data[["广告投入", "客流量"]]
    y = data["销量"]
    
    rf = RFRegressor(n_estimators=100, max_depth=10, verbose=True)
    rf.fit(X, y, test_size=0.2)
    
    # 3. 交叉验证
    rf.cross_validate(X, y, cv=5)
    
    # 4. 可视化
    rf.plot_feature_importance()
    rf.plot_prediction()
    
    # 5. 新样本预测
    new_data = pd.DataFrame({
        "广告投入": [50, 80],
        "客流量": [1000, 1500]
    })
    predictions = rf.predict(new_data)
    print(f"\n新样本预测销量: {predictions.round(2)}")
