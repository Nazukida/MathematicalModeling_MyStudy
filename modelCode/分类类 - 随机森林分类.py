"""
============================================================
随机森林分类 (Random Forest Classification)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：二分类/多分类、故障诊断、模式识别
原理：集成多棵决策树，通过投票得出分类结果
作者：MCM/ICM Team
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 图表美化设置
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class RFClassifier:
    """
    随机森林分类器封装类
    
    优点：
    - 抗过拟合能力强
    - 可处理高维数据
    - 自动输出特征重要性
    - 对缺失值和异常值不敏感
    
    参数说明：
    - n_estimators: 决策树数量（100-500）
    - max_depth: 树的最大深度（防止过拟合）
    - class_weight: 'balanced' 处理类别不均衡
    """
    
    def __init__(self, n_estimators=100, max_depth=None, 
                 class_weight=None, random_state=42, verbose=True):
        """
        参数配置
        
        :param n_estimators: 决策树数量
        :param max_depth: 最大深度（None不限制）
        :param class_weight: 类别权重（'balanced'自动平衡）
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1
        )
        self.verbose = verbose
        self.feature_names = None
        self.feature_importance = None
        self.confusion_mat = None
        self.accuracy = None
    
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
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 训练
        self.model.fit(X_train, y_train)
        
        # 预测与评估
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.confusion_mat = confusion_matrix(y_test, y_pred)
        
        # 特征重要性
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        
        if self.verbose:
            self._print_results(y_test, y_pred)
        
        return self
    
    def _print_results(self, y_test, y_pred):
        """打印结果"""
        print("\n" + "="*50)
        print("🌲 随机森林分类结果")
        print("="*50)
        print(f"\n  准确率: {self.accuracy:.4f}")
        print(f"\n  混淆矩阵:")
        print(self.confusion_mat)
        print(f"\n  特征重要性:")
        for name, imp in self.feature_importance.items():
            bar = "█" * int(imp * 30)
            print(f"    {name}: {imp:.4f} {bar}")
        print("="*50)
    
    def cross_validate(self, X, y, cv=5):
        """交叉验证"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        print(f"\n交叉验证 (cv={cv}):")
        print(f"  准确率: {scores.mean():.4f} ± {scores.std():.4f}")
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
        ax.set_title('随机森林特征重要性', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, importance.values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=10)
        
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
        
        # 添加数值
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
    print("   随机森林分类演示 - 设备故障诊断")
    print("="*60)
    
    # 1. 模拟数据（设备状态：正常/故障）
    np.random.seed(42)
    # 正常设备
    normal = pd.DataFrame({
        "温度": np.random.normal(50, 5, 80),
        "振动": np.random.normal(0.5, 0.1, 80),
        "压力": np.random.normal(100, 10, 80),
        "故障类型": 0
    })
    # 温度故障
    fault = pd.DataFrame({
        "温度": np.random.normal(80, 8, 80),
        "振动": np.random.normal(0.8, 0.2, 80),
        "压力": np.random.normal(120, 15, 80),
        "故障类型": 1
    })
    data = pd.concat([normal, fault], ignore_index=True)
    
    print("\n数据概览：")
    print(data.describe().round(2))
    
    # 2. 训练模型
    X = data[["温度", "振动", "压力"]]
    y = data["故障类型"]
    
    rf = RFClassifier(n_estimators=100, max_depth=5, verbose=True)
    rf.fit(X, y, test_size=0.2)
    
    # 3. 交叉验证
    rf.cross_validate(X, y, cv=5)
    
    # 4. 可视化
    rf.plot_feature_importance()
    rf.plot_confusion_matrix(class_names=["正常", "故障"])
    
    # 5. 新样本预测
    new_data = pd.DataFrame({
        "温度": [55, 85],
        "振动": [0.5, 1.0],
        "压力": [105, 130]
    })
    predictions = rf.predict(new_data)
    print(f"\n新样本预测结果: {['正常' if p==0 else '故障' for p in predictions]}")
