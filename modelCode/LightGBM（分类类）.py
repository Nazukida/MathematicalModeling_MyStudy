"""
============================================================
LightGBM 分类模型
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：高效梯度提升分类、大规模数据处理、特征重要性分析
原理：基于直方图的梯度提升决策树
作者：MCM/ICM Team
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# 图表美化设置
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class LightGBMClassifier:
    """
    LightGBM分类器封装类
    
    优点：
    - 训练速度快（直方图算法）
    - 内存占用低
    - 支持类别特征
    - 高精度
    
    核心参数：
    - learning_rate: 学习率（0.01-0.1）
    - max_depth: 树深度（-1不限制，3-10常用）
    - num_leaves: 叶子数（20-300）
    - n_estimators: 迭代次数（100-1000）
    """
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 num_leaves=31, random_state=42, verbose=True):
        """
        参数配置
        
        :param n_estimators: 迭代次数
        :param max_depth: 最大深度
        :param learning_rate: 学习率
        :param num_leaves: 最大叶子数
        """
        self.params = {
            "objective": "binary",
            "metric": "auc",
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "seed": random_state,
            "verbosity": -1
        }
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.accuracy = None
        self.auc = None
        self.confusion_mat = None
    
    def fit(self, X, y, test_size=0.2, early_stopping=True):
        """
        训练模型
        
        :param X: 特征DataFrame或数组
        :param y: 标签（0/1）
        :param test_size: 测试集比例
        :param early_stopping: 是否使用早停
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"特征{i+1}" for i in range(X.shape[1])]
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # 训练
        callbacks = []
        if early_stopping:
            callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False))
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[valid_data],
            callbacks=callbacks
        )
        
        # 预测与评估
        y_prob = self.model.predict(X_test)
        y_pred = (y_prob > 0.5).astype(int)
        
        self.accuracy = accuracy_score(y_test, y_pred)
        self.auc = roc_auc_score(y_test, y_prob)
        self.confusion_mat = confusion_matrix(y_test, y_pred)
        
        # 特征重要性
        self.feature_importance = pd.Series(
            self.model.feature_importance(importance_type='gain'),
            index=self.feature_names
        ).sort_values(ascending=False)
        
        if self.verbose:
            self._print_results(y_test, y_pred)
        
        return self
    
    def _print_results(self, y_test, y_pred):
        """打印结果"""
        print("\n" + "="*50)
        print("⚡ LightGBM 分类结果")
        print("="*50)
        print(f"\n  准确率: {self.accuracy:.4f}")
        print(f"  AUC: {self.auc:.4f}")
        print(f"\n  混淆矩阵:")
        print(self.confusion_mat)
        print(f"\n  特征重要性 (Gain):")
        for name, imp in self.feature_importance.head(10).items():
            bar = "█" * int(imp / self.feature_importance.max() * 20)
            print(f"    {name}: {imp:.0f} {bar}")
        print("="*50)
    
    def predict(self, X):
        """预测类别"""
        prob = self.model.predict(X)
        return (prob > 0.5).astype(int)
    
    def predict_proba(self, X):
        """预测概率"""
        return self.model.predict(X)
    
    def plot_feature_importance(self, top_n=10, save_path=None):
        """可视化特征重要性"""
        if self.feature_importance is None:
            raise ValueError("请先调用fit()训练模型")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        importance = self.feature_importance.head(top_n).sort_values(ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance)))
        
        bars = ax.barh(importance.index, importance.values, color=colors,
                      edgecolor='white', linewidth=2)
        
        ax.set_xlabel('重要性 (Gain)', fontsize=12, fontweight='bold')
        ax.set_title('LightGBM 特征重要性', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, importance.values):
            ax.text(val + importance.max()*0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.0f}', va='center', fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, class_names=None, save_path=None):
        """可视化混淆矩阵"""
        if self.confusion_mat is None:
            raise ValueError("请先调用fit()训练模型")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(self.confusion_mat, cmap='Blues')
        
        if class_names is None:
            class_names = [f"类别{i}" for i in range(len(self.confusion_mat))]
        
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        thresh = self.confusion_mat.max() / 2
        for i in range(len(self.confusion_mat)):
            for j in range(len(self.confusion_mat)):
                ax.text(j, i, self.confusion_mat[i, j],
                       ha='center', va='center',
                       color='white' if self.confusion_mat[i, j] > thresh else 'black',
                       fontsize=14, fontweight='bold')
        
        ax.set_xlabel('预测标签', fontsize=12, fontweight='bold')
        ax.set_ylabel('真实标签', fontsize=12, fontweight='bold')
        ax.set_title(f'混淆矩阵 (准确率={self.accuracy:.4f})', fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("   LightGBM 分类演示 - 用户购买预测")
    print("="*60)
    
    # 1. 模拟数据（用户购买行为）
    np.random.seed(42)
    n = 500
    age = np.random.randint(18, 60, n)
    income = np.random.randint(1, 5, n)
    browse_time = np.random.uniform(0.5, 4, n)
    # 购买标签（收入高且浏览时间长更可能购买）
    purchase = np.where((income >= 3) & (browse_time >= 2), 1, 0)
    
    data = pd.DataFrame({
        "年龄": age,
        "消费能力": income,
        "浏览时长": browse_time,
        "是否购买": purchase
    })
    
    print("\n数据概览：")
    print(data.describe().round(2))
    print(f"\n购买率: {data['是否购买'].mean()*100:.1f}%")
    
    # 2. 训练模型
    X = data.drop("是否购买", axis=1)
    y = data["是否购买"]
    
    lgbm = LightGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
    lgbm.fit(X, y, test_size=0.2)
    
    # 3. 可视化
    lgbm.plot_feature_importance()
    lgbm.plot_confusion_matrix(class_names=["未购买", "购买"])
    
    # 4. 新样本预测
    new_users = pd.DataFrame({
        "年龄": [25, 45],
        "消费能力": [4, 2],
        "浏览时长": [3.5, 1.0]
    })
    probs = lgbm.predict_proba(new_users)
    print(f"\n新用户购买概率: {probs.round(4)}")
