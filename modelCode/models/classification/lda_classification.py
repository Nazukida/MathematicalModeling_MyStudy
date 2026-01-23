"""
============================================================
LDA 线性判别分析 (Linear Discriminant Analysis)
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：有监督降维、分类、特征提取
原理：最大化类间距离/类内距离比值
与PCA区别：LDA利用标签信息，实现有监督的降维
作者：MCM/ICM Team
日期：2026年1月
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_curve, auc)
from sklearn.decomposition import PCA
import warnings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from visualization.plot_config import PlotStyleConfig, FigureSaver

PlotStyleConfig.setup_style()


class LDAClassifier:
    """
    LDA线性判别分析封装类
    
    核心原理：
    Fisher准则：最大化 J(w) = w^T S_B w / w^T S_W w
    - S_B: 类间散度矩阵 (Between-class scatter)
    - S_W: 类内散度矩阵 (Within-class scatter)
    
    应用场景：
    - 有监督降维（保留类别区分信息）
    - 多分类问题
    - 人脸识别（Fisherfaces）
    - 医学诊断、信用评分
    
    与PCA的对比：
    | 维度 | PCA | LDA |
    |------|-----|-----|
    | 监督类型 | 无监督 | 有监督 |
    | 目标 | 最大化方差 | 最大化类间/类内比 |
    | 最大维度 | min(n, p) | min(c-1, p) |
    | 适用 | 数据压缩、可视化 | 分类、特征提取 |
    
    假设条件：
    1. 各类数据服从多元正态分布
    2. 各类协方差矩阵相等
    3. 类别间存在线性决策边界
    """
    
    def __init__(self, n_components=None, solver='svd', verbose=True):
        """
        参数配置
        
        :param n_components: 保留的判别方向数（None自动为类别数-1）
        :param solver: 求解器 ('svd', 'lsqr', 'eigen')
        :param verbose: 是否打印详细信息
        """
        self.n_components = n_components
        self.solver = solver
        self.verbose = verbose
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.explained_variance_ratio_ = None
        self.scalings_ = None
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.feature_names_ = None
        self.metrics_ = {}
        
    def fit(self, X, y, feature_names=None):
        """
        拟合LDA模型
        
        :param X: 特征矩阵 (n_samples, n_features)
        :param y: 标签向量 (n_samples,)
        :param feature_names: 特征名称列表
        """
        # 处理DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_array = X.values
        else:
            X_array = X.copy()
            self.feature_names_ = feature_names or [f'X{i+1}' for i in range(X.shape[1])]
        
        # 编码标签
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y.copy()
        
        # 如果标签是字符串，进行编码
        if y_array.dtype == object or isinstance(y_array[0], str):
            y_encoded = self.label_encoder.fit_transform(y_array)
            self.classes_ = self.label_encoder.classes_
        else:
            y_encoded = y_array
            self.classes_ = np.unique(y_array)
        
        self.n_classes_ = len(self.classes_)
        
        # 标准化
        X_std = self.scaler.fit_transform(X_array)
        
        # 确定成分数
        if self.n_components is None:
            self.n_components = min(self.n_classes_ - 1, X_std.shape[1])
        
        # 创建并拟合LDA模型
        self.model = LinearDiscriminantAnalysis(
            n_components=self.n_components,
            solver=self.solver
        )
        self.model.fit(X_std, y_encoded)
        
        # 提取模型参数
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_
        self.scalings_ = self.model.scalings_
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        # 计算类间和类内散度矩阵
        self._compute_scatter_matrices(X_std, y_encoded)
        
        if self.verbose:
            self._print_results(X_array, y_encoded)
        
        return self
    
    def _compute_scatter_matrices(self, X, y):
        """
        计算散度矩阵
        """
        n_samples, n_features = X.shape
        
        # 总体均值
        overall_mean = np.mean(X, axis=0)
        
        # 类内散度矩阵 S_W
        S_W = np.zeros((n_features, n_features))
        
        # 类间散度矩阵 S_B
        S_B = np.zeros((n_features, n_features))
        
        for c in np.unique(y):
            X_c = X[y == c]
            n_c = X_c.shape[0]
            mean_c = np.mean(X_c, axis=0)
            
            # 类内散度
            S_W += (X_c - mean_c).T @ (X_c - mean_c)
            
            # 类间散度
            mean_diff = (mean_c - overall_mean).reshape(-1, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)
        
        self.within_class_scatter_ = S_W
        self.between_class_scatter_ = S_B
        
        # 计算Fisher比率
        try:
            S_W_inv = np.linalg.inv(S_W)
            fisher_matrix = S_W_inv @ S_B
            eigenvalues = np.linalg.eigvalsh(fisher_matrix)
            self.fisher_ratio_ = np.sum(eigenvalues[:self.n_components])
        except:
            self.fisher_ratio_ = None
    
    def _print_results(self, X, y):
        """打印分析结果"""
        print("\n" + "=" * 60)
        print("LDA 线性判别分析结果")
        print("=" * 60)
        
        print(f"\n【基本信息】")
        print(f"  - 样本数: {X.shape[0]}")
        print(f"  - 特征数: {X.shape[1]}")
        print(f"  - 类别数: {self.n_classes_}")
        print(f"  - 类别: {list(self.classes_)}")
        print(f"  - 保留判别方向数: {self.n_components}")
        
        # 各类样本分布
        print(f"\n【各类样本分布】")
        for c in np.unique(y):
            count = np.sum(y == c)
            class_name = self.classes_[c] if c < len(self.classes_) else c
            print(f"  - 类别 {class_name}: {count} 个 ({100*count/len(y):.1f}%)")
        
        # 方差解释
        if self.explained_variance_ratio_ is not None:
            print(f"\n【判别方向方差解释】")
            cum_var = 0
            for i, var in enumerate(self.explained_variance_ratio_):
                cum_var += var
                print(f"  - LD{i+1}: {var*100:.2f}% (累计: {cum_var*100:.2f}%)")
        
        # Fisher比率
        if self.fisher_ratio_ is not None:
            print(f"\n【Fisher判别比率】")
            print(f"  - J(w) = {self.fisher_ratio_:.4f}")
            print(f"    (类间距离/类内距离，越大越好)")
        
        # 判别系数（二分类时）
        if self.n_classes_ == 2:
            print(f"\n【判别函数系数】")
            for i, (name, coef) in enumerate(zip(self.feature_names_, self.coef_[0])):
                print(f"  - {name}: {coef:.4f}")
            print(f"  - 截距: {self.intercept_[0]:.4f}")
        
        print("=" * 60)
    
    def transform(self, X):
        """
        将数据投影到判别空间
        
        :param X: 输入数据
        :return: 降维后的数据
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X.copy()
        
        X_std = self.scaler.transform(X_array)
        return self.model.transform(X_std)
    
    def predict(self, X):
        """
        预测类别
        
        :param X: 输入数据
        :return: 预测标签
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X.copy()
        
        X_std = self.scaler.transform(X_array)
        y_pred = self.model.predict(X_std)
        
        # 转换回原始标签
        return self.classes_[y_pred]
    
    def predict_proba(self, X):
        """
        预测各类的概率
        
        :param X: 输入数据
        :return: 概率矩阵
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X.copy()
        
        X_std = self.scaler.transform(X_array)
        return self.model.predict_proba(X_std)
    
    def evaluate(self, X, y_true):
        """
        评估模型性能
        
        :param X: 测试特征
        :param y_true: 真实标签
        :return: 评估指标字典
        """
        y_pred = self.predict(X)
        
        # 处理标签
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        
        # 计算指标
        self.metrics_['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 混淆矩阵
        self.metrics_['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # 分类报告
        self.metrics_['classification_report'] = classification_report(
            y_true, y_pred, target_names=[str(c) for c in self.classes_]
        )
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("模型评估结果")
            print("=" * 60)
            print(f"\n准确率: {self.metrics_['accuracy']*100:.2f}%")
            print(f"\n【分类报告】")
            print(self.metrics_['classification_report'])
        
        return self.metrics_
    
    def cross_validate(self, X, y, cv=5):
        """
        交叉验证评估
        
        :param X: 特征矩阵
        :param y: 标签
        :param cv: 折数
        :return: 交叉验证得分
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X.copy()
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y.copy()
        
        X_std = self.scaler.fit_transform(X_array)
        
        # 编码标签
        if y_array.dtype == object or isinstance(y_array[0], str):
            y_encoded = self.label_encoder.fit_transform(y_array)
        else:
            y_encoded = y_array
        
        # 交叉验证
        lda = LinearDiscriminantAnalysis(
            n_components=self.n_components,
            solver=self.solver
        )
        scores = cross_val_score(lda, X_std, y_encoded, cv=cv)
        
        if self.verbose:
            print(f"\n【{cv}折交叉验证结果】")
            print(f"  - 各折准确率: {scores}")
            print(f"  - 平均准确率: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")
        
        return scores
    
    def plot_projection_2d(self, X, y, save_path=None):
        """
        绘制2D判别投影图
        
        :param X: 特征矩阵
        :param y: 标签
        :param save_path: 保存路径
        """
        # 获取投影数据
        X_lda = self.transform(X)
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y.copy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 为每个类别绘制散点
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_classes_))
        
        for i, class_label in enumerate(self.classes_):
            mask = y_array == class_label
            
            if X_lda.shape[1] >= 2:
                ax.scatter(X_lda[mask, 0], X_lda[mask, 1],
                          c=[colors[i]], s=50, alpha=0.7,
                          label=f'{class_label}')
            else:
                # 只有1个判别方向时，y轴用0
                ax.scatter(X_lda[mask, 0], np.zeros(mask.sum()),
                          c=[colors[i]], s=50, alpha=0.7,
                          label=f'{class_label}')
        
        # 标签
        if X_lda.shape[1] >= 2:
            var1 = self.explained_variance_ratio_[0] * 100
            var2 = self.explained_variance_ratio_[1] * 100
            ax.set_xlabel(f'LD1 ({var1:.1f}%)', fontsize=12)
            ax.set_ylabel(f'LD2 ({var2:.1f}%)', fontsize=12)
        else:
            var1 = self.explained_variance_ratio_[0] * 100
            ax.set_xlabel(f'LD1 ({var1:.1f}%)', fontsize=12)
            ax.set_ylabel('', fontsize=12)
        
        ax.set_title('LDA 判别投影', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            FigureSaver.save(fig, save_path)
        plt.show()
        
        return fig
    
    def plot_confusion_matrix(self, X, y, save_path=None):
        """
        绘制混淆矩阵
        
        :param X: 测试特征
        :param y: 真实标签
        :param save_path: 保存路径
        """
        y_pred = self.predict(X)
        
        if isinstance(y, pd.Series):
            y_true = y.values
        else:
            y_true = y
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, cmap='Blues')
        
        # 设置刻度
        ax.set_xticks(range(self.n_classes_))
        ax.set_yticks(range(self.n_classes_))
        ax.set_xticklabels([str(c) for c in self.classes_])
        ax.set_yticklabels([str(c) for c in self.classes_])
        
        # 添加数值
        for i in range(self.n_classes_):
            for j in range(self.n_classes_):
                value = cm[i, j]
                color = 'white' if value > cm.max() / 2 else 'black'
                ax.text(j, i, str(value), ha='center', va='center', 
                       color=color, fontsize=14)
        
        # 颜色条
        plt.colorbar(im, ax=ax)
        
        ax.set_xlabel('预测类别', fontsize=12)
        ax.set_ylabel('真实类别', fontsize=12)
        ax.set_title('混淆矩阵', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            FigureSaver.save(fig, save_path)
        plt.show()
        
        return fig
    
    def plot_roc_curve(self, X, y, save_path=None):
        """
        绘制ROC曲线（二分类或多分类的一对多）
        
        :param X: 测试特征
        :param y: 真实标签
        :param save_path: 保存路径
        """
        y_prob = self.predict_proba(X)
        
        if isinstance(y, pd.Series):
            y_true = y.values
        else:
            y_true = y.copy()
        
        # 编码标签
        if y_true.dtype == object or isinstance(y_true[0], str):
            y_encoded = self.label_encoder.transform(y_true)
        else:
            y_encoded = y_true
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_classes_))
        
        for i, class_label in enumerate(self.classes_):
            # 二值化标签（一对多）
            y_binary = (y_encoded == i).astype(int)
            
            fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                   label=f'{class_label} (AUC = {roc_auc:.3f})')
        
        # 对角线
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测')
        
        ax.set_xlabel('假阳性率 (FPR)', fontsize=12)
        ax.set_ylabel('真阳性率 (TPR)', fontsize=12)
        ax.set_title('ROC曲线', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            FigureSaver.save(fig, save_path)
        plt.show()
        
        return fig
    
    def plot_feature_importance(self, save_path=None):
        """
        绘制特征重要性（基于判别系数绝对值）
        
        :param save_path: 保存路径
        """
        if self.coef_ is None:
            print("请先拟合模型")
            return
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(self.feature_names_) * 0.4)))
        
        # 计算特征重要性（系数绝对值的均值）
        if self.n_classes_ == 2:
            importance = np.abs(self.coef_[0])
        else:
            importance = np.mean(np.abs(self.coef_), axis=0)
        
        # 排序
        sorted_idx = np.argsort(importance)
        
        # 绘制条形图
        y_pos = range(len(self.feature_names_))
        ax.barh(y_pos, importance[sorted_idx], color='steelblue', alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names_[i] for i in sorted_idx])
        ax.set_xlabel('重要性（判别系数绝对值）', fontsize=12)
        ax.set_title('特征重要性排序', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            FigureSaver.save(fig, save_path)
        plt.show()
        
        return fig
    
    def plot_decision_boundary_2d(self, X, y, resolution=200, save_path=None):
        """
        绘制决策边界（仅限2D特征或2D投影）
        
        :param X: 特征矩阵
        :param y: 标签
        :param resolution: 网格分辨率
        :param save_path: 保存路径
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X.copy()
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y.copy()
        
        # 如果特征多于2个，先用LDA降维到2D
        if X_array.shape[1] > 2:
            X_2d = self.transform(X_array)[:, :2]
            x_label = 'LD1'
            y_label = 'LD2'
        else:
            X_2d = self.scaler.transform(X_array)
            x_label = self.feature_names_[0]
            y_label = self.feature_names_[1]
        
        # 创建网格
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        
        # 预测网格点（需要在原始空间预测）
        # 这里简化处理，直接在降维空间用最近邻
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=1)
        
        # 编码标签
        if y_array.dtype == object or isinstance(y_array[0], str):
            y_encoded = self.label_encoder.transform(y_array)
        else:
            y_encoded = y_array
        
        knn.fit(X_2d, y_encoded)
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制决策区域
        colors = plt.cm.Pastel1(np.linspace(0, 1, self.n_classes_))
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Pastel1)
        ax.contour(xx, yy, Z, colors='k', linewidths=0.5)
        
        # 绘制数据点
        colors_dark = plt.cm.tab10(np.linspace(0, 1, self.n_classes_))
        for i, class_label in enumerate(self.classes_):
            mask = y_array == class_label
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=[colors_dark[i]], s=50, alpha=0.8,
                      edgecolors='white', linewidths=0.5,
                      label=f'{class_label}')
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title('LDA 决策边界', fontsize=14)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            FigureSaver.save(fig, save_path)
        plt.show()
        
        return fig


def compare_pca_lda(X, y, feature_names=None, save_path=None):
    """
    对比PCA和LDA的降维效果
    
    :param X: 特征矩阵
    :param y: 标签
    :param feature_names: 特征名称
    :param save_path: 保存路径
    """
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        if feature_names is None:
            feature_names = list(X.columns)
    else:
        X_array = X.copy()
        if feature_names is None:
            feature_names = [f'X{i+1}' for i in range(X.shape[1])]
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y.copy()
    
    # 标准化
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_array)
    
    # PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    
    # LDA降维
    n_classes = len(np.unique(y_array))
    n_components = min(2, n_classes - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X_std, y_array)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    unique_classes = np.unique(y_array)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    
    # PCA图
    ax1 = axes[0]
    for i, class_label in enumerate(unique_classes):
        mask = y_array == class_label
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors[i]], s=50, alpha=0.7, label=f'{class_label}')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax1.set_title('PCA 降维结果（无监督）', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # LDA图
    ax2 = axes[1]
    for i, class_label in enumerate(unique_classes):
        mask = y_array == class_label
        if X_lda.shape[1] >= 2:
            ax2.scatter(X_lda[mask, 0], X_lda[mask, 1],
                       c=[colors[i]], s=50, alpha=0.7, label=f'{class_label}')
        else:
            ax2.scatter(X_lda[mask, 0], np.zeros(mask.sum()),
                       c=[colors[i]], s=50, alpha=0.7, label=f'{class_label}')
    
    if X_lda.shape[1] >= 2:
        ax2.set_xlabel(f'LD1 ({lda.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax2.set_ylabel(f'LD2 ({lda.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    else:
        ax2.set_xlabel(f'LD1 ({lda.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax2.set_ylabel('', fontsize=12)
    ax2.set_title('LDA 降维结果（有监督）', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        FigureSaver.save(fig, save_path)
    plt.show()
    
    print("\n【PCA vs LDA 对比说明】")
    print("  - PCA（左图）：无监督方法，仅最大化数据方差")
    print("  - LDA（右图）：有监督方法，利用标签信息最大化类间分离度")
    print("  - 当类别信息重要时，LDA通常能获得更好的分类边界")
    
    return fig


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("LDA 线性判别分析 - 使用示例")
    print("=" * 60)
    
    # 加载示例数据（鸢尾花数据集）
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target_names[iris.target], name='species')
    
    print(f"\n数据集信息：")
    print(f"  - 样本数: {X.shape[0]}")
    print(f"  - 特征数: {X.shape[1]}")
    print(f"  - 类别: {list(np.unique(y))}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 创建LDA分类器
    print("\n" + "-" * 40)
    lda = LDAClassifier(n_components=2, verbose=True)
    
    # 拟合模型
    lda.fit(X_train, y_train)
    
    # 评估模型
    print("\n" + "-" * 40)
    lda.evaluate(X_test, y_test)
    
    # 交叉验证
    print("\n" + "-" * 40)
    cv_scores = lda.cross_validate(X, y, cv=5)
    
    # 绘制投影图
    print("\n绘制LDA投影图...")
    lda.plot_projection_2d(X, y)
    
    # 绘制混淆矩阵
    print("\n绘制混淆矩阵...")
    lda.plot_confusion_matrix(X_test, y_test)
    
    # 绘制ROC曲线
    print("\n绘制ROC曲线...")
    lda.plot_roc_curve(X_test, y_test)
    
    # 绘制特征重要性
    print("\n绘制特征重要性...")
    lda.plot_feature_importance()
    
    # 绘制决策边界
    print("\n绘制决策边界...")
    lda.plot_decision_boundary_2d(X, y)
    
    # PCA vs LDA对比
    print("\n" + "-" * 40)
    print("【PCA vs LDA 对比】")
    compare_pca_lda(X, y)
    
    print("\n" + "=" * 60)
    print("LDA分析完成！")
    print("=" * 60)
