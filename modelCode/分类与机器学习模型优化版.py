"""
============================================================
分类与机器学习模型 (Classification & ML Models)
包含：随机森林分类 + XGBoost + LightGBM + 多目标优化
适用于美国大学生数学建模竞赛 (MCM/ICM)
============================================================
功能：分类预测、特征工程、模型评估
特点：完整的参数设置、数据预处理、可视化与美化
作者：MCM/ICM Team
日期：2026年1月
============================================================

使用场景：
- 二分类/多分类问题
- 故障诊断、疾病预测
- 用户行为分类
- 多目标优化决策
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')


# ============================================================
# 第一部分：全局配置与美化设置 (Global Configuration)
# ============================================================

class PlotStyleConfig:
    """图表美化配置类"""
    
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'success': '#27AE60',
        'danger': '#C73E1D',
        'neutral': '#3B3B3B'
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
# 第二部分：分类数据生成器 (Classification Data Generator)
# ============================================================

class ClassificationDataGenerator:
    """分类数据生成器 - 用于测试和演示"""
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_binary_classification(self, n_samples=500, scenario='equipment'):
        """
        生成二分类数据
        
        :param n_samples: 样本数量
        :param scenario: 场景
            - 'equipment': 设备故障检测
            - 'customer': 客户流失预测
            - 'medical': 疾病诊断
        """
        if scenario == 'equipment':
            # 设备故障检测
            n_normal = n_samples // 2
            n_fault = n_samples - n_normal
            
            normal_data = pd.DataFrame({
                '温度': np.random.normal(50, 5, n_normal),
                '振动': np.random.normal(0.5, 0.1, n_normal),
                '压力': np.random.normal(100, 10, n_normal),
                '运行时间': np.random.uniform(100, 1000, n_normal),
                '标签': 0
            })
            
            fault_data = pd.DataFrame({
                '温度': np.random.normal(80, 8, n_fault),
                '振动': np.random.normal(1.2, 0.3, n_fault),
                '压力': np.random.normal(130, 15, n_fault),
                '运行时间': np.random.uniform(800, 2000, n_fault),
                '标签': 1
            })
            
            data = pd.concat([normal_data, fault_data], ignore_index=True)
            feature_names = ['温度', '振动', '压力', '运行时间']
            target_names = ['正常', '故障']
            
        elif scenario == 'customer':
            # 客户流失预测
            n_retain = n_samples // 2
            n_churn = n_samples - n_retain
            
            retain_data = pd.DataFrame({
                '消费金额': np.random.uniform(500, 5000, n_retain),
                '购买频次': np.random.randint(5, 30, n_retain),
                '会员时长': np.random.uniform(12, 60, n_retain),
                '投诉次数': np.random.randint(0, 3, n_retain),
                '标签': 0
            })
            
            churn_data = pd.DataFrame({
                '消费金额': np.random.uniform(100, 1000, n_churn),
                '购买频次': np.random.randint(1, 10, n_churn),
                '会员时长': np.random.uniform(1, 24, n_churn),
                '投诉次数': np.random.randint(2, 10, n_churn),
                '标签': 1
            })
            
            data = pd.concat([retain_data, churn_data], ignore_index=True)
            feature_names = ['消费金额', '购买频次', '会员时长', '投诉次数']
            target_names = ['留存', '流失']
            
        elif scenario == 'medical':
            # 疾病诊断
            n_healthy = n_samples // 2
            n_sick = n_samples - n_healthy
            
            healthy_data = pd.DataFrame({
                '血压': np.random.normal(120, 10, n_healthy),
                '血糖': np.random.normal(95, 10, n_healthy),
                '胆固醇': np.random.normal(180, 20, n_healthy),
                '年龄': np.random.randint(20, 60, n_healthy),
                '标签': 0
            })
            
            sick_data = pd.DataFrame({
                '血压': np.random.normal(150, 15, n_sick),
                '血糖': np.random.normal(130, 20, n_sick),
                '胆固醇': np.random.normal(240, 30, n_sick),
                '年龄': np.random.randint(40, 80, n_sick),
                '标签': 1
            })
            
            data = pd.concat([healthy_data, sick_data], ignore_index=True)
            feature_names = ['血压', '血糖', '胆固醇', '年龄']
            target_names = ['健康', '患病']
        
        # 打乱数据
        data = data.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        return {
            'data': data,
            'feature_names': feature_names,
            'target_names': target_names,
            'X': data[feature_names],
            'y': data['标签']
        }
    
    def generate_multiclass(self, n_samples=600, n_classes=3):
        """生成多分类数据"""
        samples_per_class = n_samples // n_classes
        
        data_list = []
        for i in range(n_classes):
            class_data = pd.DataFrame({
                '特征1': np.random.normal(i * 10, 3, samples_per_class),
                '特征2': np.random.normal(i * 5, 2, samples_per_class),
                '特征3': np.random.uniform(i * 2, i * 2 + 5, samples_per_class),
                '标签': i
            })
            data_list.append(class_data)
        
        data = pd.concat(data_list, ignore_index=True)
        data = data.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        return {
            'data': data,
            'feature_names': ['特征1', '特征2', '特征3'],
            'target_names': [f'类别{i+1}' for i in range(n_classes)],
            'X': data[['特征1', '特征2', '特征3']],
            'y': data['标签']
        }


# ============================================================
# 第三部分：分类器基类 (Base Classifier)
# ============================================================

class BaseClassifier:
    """分类器基类"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.target_names = None
        self.metrics = None
        self.feature_importance = None
        self.is_fitted = False
    
    def _scale_data(self, X_train, X_test=None):
        """标准化数据"""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        return X_train_scaled
    
    def _compute_metrics(self, y_true, y_pred, y_prob=None):
        """计算评估指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if y_prob is not None and len(np.unique(y_true)) == 2:
            metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        
        return metrics
    
    def predict(self, X):
        """预测"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测概率"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)


# ============================================================
# 第四部分：随机森林分类器 (Random Forest Classifier)
# ============================================================

class RandomForestModel(BaseClassifier):
    """
    随机森林分类器
    
    原理：
    基于决策树的集成学习方法，通过Bootstrap采样和特征随机选择
    构建多棵决策树，最终通过投票决定分类结果。
    
    优点：
    - 抗过拟合能力强
    - 可处理高维数据
    - 可输出特征重要性
    - 对缺失值和异常值不敏感
    """
    
    def __init__(self, n_estimators=100, max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1,
                 class_weight=None, verbose=True):
        """
        参数配置说明
        
        :param n_estimators: 决策树数量
            - 建议：100-500，越多越稳定但计算量增加
            
        :param max_depth: 树的最大深度
            - None：不限制
            - 建议：5-20，防止过拟合
            
        :param min_samples_split: 分裂所需最小样本数
            - 默认：2，增大可防止过拟合
            
        :param min_samples_leaf: 叶节点最小样本数
            - 默认：1，增大可防止过拟合
            
        :param class_weight: 类别权重
            - 'balanced': 自动平衡不均衡数据
        """
        super().__init__(verbose)
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
    
    def fit(self, X, y, test_size=0.2, scale=True):
        """
        训练模型
        
        :param X: 特征矩阵
        :param y: 标签
        :param test_size: 测试集比例
        :param scale: 是否标准化
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
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 标准化
        if scale:
            X_train, X_test = self._scale_data(X_train, X_test)
        
        # 训练
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # 预测
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_prob = self.model.predict_proba(X_test)
        
        # 评估
        self.metrics = {
            'train': self._compute_metrics(y_train, y_train_pred),
            'test': self._compute_metrics(y_test, y_test_pred, y_test_prob)
        }
        
        # 混淆矩阵
        self.confusion_matrix = confusion_matrix(y_test, y_test_pred)
        
        # 特征重要性
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        )
        
        # 保存测试数据用于可视化
        self._y_test = y_test
        self._y_test_pred = y_test_pred
        self._y_test_prob = y_test_prob
        
        if self.verbose:
            self._print_results()
        
        return self
    
    def _print_results(self):
        """打印结果"""
        print("\n" + "="*60)
        print("📊 随机森林分类结果 (Random Forest)")
        print("="*60)
        
        print("\n  训练集指标:")
        for k, v in self.metrics['train'].items():
            print(f"    {k}: {v:.4f}")
        
        print("\n  测试集指标:")
        for k, v in self.metrics['test'].items():
            print(f"    {k}: {v:.4f}")
        
        print("\n  混淆矩阵:")
        print(self.confusion_matrix)
        
        print("\n  特征重要性:")
        for name, imp in self.feature_importance.sort_values(ascending=False).items():
            print(f"    {name}: {imp:.4f}")
        
        print("="*60)
    
    def cross_validate(self, X, y, cv=5):
        """交叉验证"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        print(f"\n交叉验证准确率 (cv={cv}):")
        print(f"  Mean: {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"  Scores: {scores.round(4)}")
        
        return scores


# ============================================================
# 第五部分：集成分类器 (Ensemble Classifier)
# ============================================================

class EnsembleClassifier(BaseClassifier):
    """
    集成分类器 - 结合多个模型
    
    方法：
    - 投票法 (Voting)
    - 加权平均法 (Weighted Average)
    """
    
    def __init__(self, verbose=True):
        super().__init__(verbose)
        self.models = {}
        self.weights = None
        self.individual_metrics = {}
    
    def add_model(self, name, model):
        """添加模型"""
        self.models[name] = model
    
    def add_default_models(self):
        """添加默认模型组合"""
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
    
    def fit(self, X, y, test_size=0.2, scale=True):
        """训练所有模型"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"特征{i+1}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 标准化
        if scale:
            X_train, X_test = self._scale_data(X_train, X_test)
        
        predictions = {}
        probabilities = {}
        
        # 训练每个模型
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)
            
            predictions[name] = pred
            probabilities[name] = prob
            
            self.individual_metrics[name] = self._compute_metrics(y_test, pred, prob)
        
        # 计算权重（基于准确率）
        accuracies = np.array([self.individual_metrics[name]['accuracy'] for name in self.models])
        self.weights = accuracies / accuracies.sum()
        
        # 加权平均预测
        ensemble_prob = np.zeros_like(list(probabilities.values())[0])
        for i, (name, prob) in enumerate(probabilities.items()):
            ensemble_prob += self.weights[i] * prob
        
        ensemble_pred = np.argmax(ensemble_prob, axis=1)
        
        self.individual_metrics['Ensemble'] = self._compute_metrics(y_test, ensemble_pred, ensemble_prob)
        self.confusion_matrix = confusion_matrix(y_test, ensemble_pred)
        
        self._y_test = y_test
        self._ensemble_pred = ensemble_pred
        self._ensemble_prob = ensemble_prob
        
        self.is_fitted = True
        
        if self.verbose:
            self._print_comparison()
        
        return self
    
    def _print_comparison(self):
        """打印模型对比"""
        print("\n" + "="*70)
        print("📊 集成分类器对比 (Ensemble Comparison)")
        print("="*70)
        
        print(f"\n  {'模型':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("  " + "-"*60)
        
        for name, metrics in self.individual_metrics.items():
            print(f"  {name:<20} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
                  f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f}")
        
        print(f"\n  模型权重:")
        for name, weight in zip(self.models.keys(), self.weights):
            print(f"    {name}: {weight:.4f}")
        
        print("="*70)
    
    def predict(self, X):
        """集成预测"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        probabilities = []
        for i, model in enumerate(self.models.values()):
            prob = model.predict_proba(X)
            probabilities.append(self.weights[i] * prob)
        
        ensemble_prob = np.sum(probabilities, axis=0)
        return np.argmax(ensemble_prob, axis=1)


# ============================================================
# 第六部分：多目标优化 - NSGA-II (Multi-objective Optimization)
# ============================================================

class NSGAII:
    """
    NSGA-II 多目标优化算法
    (Non-dominated Sorting Genetic Algorithm II)
    
    适用于：多目标决策、帕累托最优
    
    特点：
    - 非支配排序
    - 拥挤度距离
    - 精英保留策略
    """
    
    def __init__(self, objectives, bounds, n_dims,
                 pop_size=50, max_iter=100,
                 crossover_rate=0.8, mutation_rate=0.1,
                 random_seed=42, verbose=True):
        """
        参数配置
        
        :param objectives: 目标函数列表 [func1, func2, ...]
        :param bounds: 变量范围 [(min1,max1), ...]
        :param n_dims: 变量维度
        :param pop_size: 种群大小
        :param max_iter: 最大迭代次数
        """
        self.objectives = objectives
        self.n_objectives = len(objectives)
        self.bounds = np.array(bounds)
        self.n_dims = n_dims
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_seed = random_seed
        self.verbose = verbose
        
        np.random.seed(random_seed)
        
        self.population = None
        self.objective_values = None
        self.pareto_front = None
        self.pareto_solutions = None
        self.history = {'hypervolume': []}
    
    def _evaluate(self, population):
        """评估种群"""
        n = len(population)
        obj_values = np.zeros((n, self.n_objectives))
        
        for i, ind in enumerate(population):
            for j, obj_func in enumerate(self.objectives):
                obj_values[i, j] = obj_func(ind)
        
        return obj_values
    
    def _dominates(self, obj1, obj2):
        """判断obj1是否支配obj2（最小化）"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def _non_dominated_sort(self, obj_values):
        """非支配排序"""
        n = len(obj_values)
        ranks = np.zeros(n, dtype=int)
        dominated_by = [[] for _ in range(n)]
        domination_count = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i+1, n):
                if self._dominates(obj_values[i], obj_values[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(obj_values[j], obj_values[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1
        
        current_rank = 0
        remaining = set(range(n))
        
        while remaining:
            current_front = [i for i in remaining if domination_count[i] == 0]
            if not current_front:
                break
            
            for i in current_front:
                ranks[i] = current_rank
                remaining.discard(i)
                for j in dominated_by[i]:
                    domination_count[j] -= 1
            
            current_rank += 1
        
        return ranks
    
    def _crowding_distance(self, obj_values, indices):
        """计算拥挤度距离"""
        n = len(indices)
        if n <= 2:
            return {i: float('inf') for i in indices}
        
        distances = {i: 0.0 for i in indices}
        
        for m in range(self.n_objectives):
            sorted_indices = sorted(indices, key=lambda x: obj_values[x, m])
            
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            obj_range = obj_values[sorted_indices[-1], m] - obj_values[sorted_indices[0], m]
            if obj_range == 0:
                continue
            
            for i in range(1, n-1):
                distances[sorted_indices[i]] += (
                    obj_values[sorted_indices[i+1], m] - obj_values[sorted_indices[i-1], m]
                ) / obj_range
        
        return distances
    
    def _crossover(self, parent1, parent2):
        """交叉操作"""
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.rand(self.n_dims)
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def _mutate(self, individual):
        """变异操作"""
        mutated = individual.copy()
        for i in range(self.n_dims):
            if np.random.rand() < self.mutation_rate:
                range_i = self.bounds[i, 1] - self.bounds[i, 0]
                mutated[i] += np.random.normal(0, 0.1 * range_i)
                mutated[i] = np.clip(mutated[i], self.bounds[i, 0], self.bounds[i, 1])
        return mutated
    
    def optimize(self):
        """执行NSGA-II优化"""
        # 初始化
        self.population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            (self.pop_size, self.n_dims)
        )
        
        if self.verbose:
            print("\n" + "="*60)
            print("🎯 NSGA-II多目标优化开始...")
            print("="*60)
            print(f"  目标函数数量: {self.n_objectives}")
            print(f"  决策变量维度: {self.n_dims}")
            print(f"  种群大小: {self.pop_size}")
            print("-"*60)
        
        for generation in range(self.max_iter):
            # 评估
            self.objective_values = self._evaluate(self.population)
            
            # 非支配排序
            ranks = self._non_dominated_sort(self.objective_values)
            
            # 生成子代
            offspring = []
            while len(offspring) < self.pop_size:
                # 锦标赛选择
                candidates = np.random.choice(self.pop_size, 4, replace=False)
                parent1 = self.population[min(candidates[:2], key=lambda x: ranks[x])]
                parent2 = self.population[min(candidates[2:], key=lambda x: ranks[x])]
                
                child1, child2 = self._crossover(parent1, parent2)
                offspring.extend([self._mutate(child1), self._mutate(child2)])
            
            offspring = np.array(offspring[:self.pop_size])
            
            # 合并种群
            combined = np.vstack([self.population, offspring])
            combined_obj = self._evaluate(combined)
            combined_ranks = self._non_dominated_sort(combined_obj)
            
            # 选择下一代
            new_pop = []
            current_rank = 0
            
            while len(new_pop) < self.pop_size:
                front_indices = np.where(combined_ranks == current_rank)[0]
                
                if len(new_pop) + len(front_indices) <= self.pop_size:
                    new_pop.extend(front_indices)
                else:
                    # 使用拥挤度距离
                    remaining_spots = self.pop_size - len(new_pop)
                    distances = self._crowding_distance(combined_obj, front_indices)
                    sorted_front = sorted(front_indices, key=lambda x: -distances[x])
                    new_pop.extend(sorted_front[:remaining_spots])
                
                current_rank += 1
            
            self.population = combined[new_pop]
            
            if self.verbose and (generation + 1) % 20 == 0:
                pareto_count = np.sum(combined_ranks[new_pop] == 0)
                print(f"  Generation {generation+1}: Pareto front size = {pareto_count}")
        
        # 获取Pareto最优解
        final_obj = self._evaluate(self.population)
        final_ranks = self._non_dominated_sort(final_obj)
        pareto_indices = np.where(final_ranks == 0)[0]
        
        self.pareto_solutions = self.population[pareto_indices]
        self.pareto_front = final_obj[pareto_indices]
        
        if self.verbose:
            self._print_results()
        
        return self.pareto_solutions, self.pareto_front
    
    def _print_results(self):
        """打印结果"""
        print("\n" + "="*60)
        print("📊 NSGA-II优化完成")
        print("="*60)
        print(f"  Pareto最优解数量: {len(self.pareto_solutions)}")
        print(f"\n  Pareto前沿范围:")
        for i in range(self.n_objectives):
            print(f"    目标{i+1}: [{self.pareto_front[:, i].min():.4f}, "
                  f"{self.pareto_front[:, i].max():.4f}]")
        print("="*60)


# ============================================================
# 第七部分：可视化模块 (Visualization)
# ============================================================

class ClassificationVisualizer:
    """分类模型可视化类"""
    
    def __init__(self):
        self.colors = PlotStyleConfig.PALETTE
    
    def plot_confusion_matrix(self, cm, class_names=None, 
                              title="混淆矩阵", save_path=None):
        """绘制混淆矩阵"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        if class_names is None:
            class_names = [f"类别{i}" for i in range(len(cm))]
        
        ax.set(xticks=np.arange(len(cm)),
               yticks=np.arange(len(cm)),
               xticklabels=class_names,
               yticklabels=class_names,
               xlabel='预测标签',
               ylabel='真实标签')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值
        thresh = cm.max() / 2.
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=14)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_prob, title="ROC曲线", save_path=None):
        """绘制ROC曲线"""
        if y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color=self.colors[0], linewidth=2,
               label=f'ROC曲线 (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测')
        ax.fill_between(fpr, tpr, alpha=0.2, color=self.colors[0])
        
        ax.set_xlabel('假正例率 (FPR)', fontweight='bold')
        ax.set_ylabel('真正例率 (TPR)', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, importance, title="特征重要性", save_path=None):
        """绘制特征重要性"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        importance = importance.sort_values(ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance)))
        
        bars = ax.barh(importance.index, importance.values,
                      color=colors, edgecolor='white', linewidth=2)
        
        ax.set_xlabel('重要性', fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, importance.values):
            ax.text(val + max(importance.values)*0.02, 
                   bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, metrics_dict, title="模型性能对比", save_path=None):
        """绘制模型对比"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = list(metrics_dict.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1']
        titles = ['(a) Accuracy', '(b) Precision', '(c) Recall', '(d) F1 Score']
        
        for ax, metric, t in zip(axes.flatten(), metrics_names, titles):
            values = [metrics_dict[m].get(metric, 0) for m in models]
            bars = ax.bar(models, values, color=self.colors[:len(models)],
                         edgecolor='white', linewidth=2)
            ax.set_ylabel(metric.capitalize(), fontweight='bold')
            ax.set_title(t, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            ax.tick_params(axis='x', rotation=30)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pareto_front(self, pareto_front, obj_names=None, 
                          title="Pareto前沿", save_path=None):
        """绘制Pareto前沿（2目标）"""
        if pareto_front.shape[1] != 2:
            print("Pareto前沿可视化仅支持2个目标")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 排序以便连线
        sorted_idx = np.argsort(pareto_front[:, 0])
        sorted_front = pareto_front[sorted_idx]
        
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1],
                  s=100, c=self.colors[0], edgecolors='white',
                  linewidths=2, zorder=5, label='Pareto最优解')
        ax.plot(sorted_front[:, 0], sorted_front[:, 1],
               '--', color=self.colors[1], alpha=0.7, linewidth=2)
        
        if obj_names is None:
            obj_names = ['目标1', '目标2']
        
        ax.set_xlabel(obj_names[0], fontweight='bold')
        ax.set_ylabel(obj_names[1], fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================
# 第八部分：主程序与完整示例 (Main Program)
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("   CLASSIFICATION & ML MODELS FOR MCM/ICM")
    print("   分类与机器学习模型")
    print("   Extended Version with Visualization")
    print("="*70)
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    📊 分类模型分析流程                            ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║   [分类算法]                                                      ║
    ║      ├─ 随机森林: 集成学习，可解释性强                            ║
    ║      ├─ 梯度提升: 高精度，适合复杂问题                            ║
    ║      ├─ 逻辑回归: 简单高效，可解释性好                            ║
    ║      └─ 支持向量机: 高维数据效果好                               ║
    ║                                                                  ║
    ║   [评估指标]                                                      ║
    ║      ├─ Accuracy: 整体准确率                                     ║
    ║      ├─ Precision: 精确率（查准率）                              ║
    ║      ├─ Recall: 召回率（查全率）                                 ║
    ║      ├─ F1 Score: 精确率和召回率的调和平均                        ║
    ║      └─ AUC: ROC曲线下面积                                       ║
    ║                                                                  ║
    ║   [多目标优化]                                                    ║
    ║      └─ NSGA-II: Pareto最优解集                                  ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    generator = ClassificationDataGenerator(random_seed=2026)
    visualizer = ClassificationVisualizer()
    
    # ================================================================
    # 示例1：设备故障检测
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 1: 设备故障检测 (Equipment Fault Detection)")
    print("="*70)
    
    data = generator.generate_binary_classification(n_samples=500, scenario='equipment')
    print(f"\n数据概览:")
    print(f"  样本数量: {len(data['data'])}")
    print(f"  特征: {data['feature_names']}")
    print(f"  类别分布: {data['y'].value_counts().to_dict()}")
    
    # 随机森林分类
    rf_model = RandomForestModel(n_estimators=100, max_depth=10, verbose=True)
    rf_model.fit(data['X'], data['y'], test_size=0.2)
    
    # 可视化
    visualizer.plot_confusion_matrix(
        rf_model.confusion_matrix, 
        class_names=data['target_names'],
        title="设备故障检测混淆矩阵"
    )
    
    visualizer.plot_roc_curve(
        rf_model._y_test, rf_model._y_test_prob,
        title="设备故障检测ROC曲线"
    )
    
    visualizer.plot_feature_importance(
        rf_model.feature_importance,
        title="故障检测特征重要性"
    )
    
    # 交叉验证
    rf_model.cross_validate(data['X'], data['y'], cv=5)
    
    # ================================================================
    # 示例2：集成分类器对比
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 2: 集成分类器对比")
    print("="*70)
    
    ensemble = EnsembleClassifier(verbose=True)
    ensemble.add_default_models()
    ensemble.fit(data['X'], data['y'], test_size=0.2)
    
    visualizer.plot_model_comparison(
        ensemble.individual_metrics,
        title="多模型性能对比"
    )
    
    # ================================================================
    # 示例3：客户流失预测
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 3: 客户流失预测 (Customer Churn Prediction)")
    print("="*70)
    
    churn_data = generator.generate_binary_classification(n_samples=600, scenario='customer')
    
    rf_churn = RandomForestModel(n_estimators=150, class_weight='balanced', verbose=True)
    rf_churn.fit(churn_data['X'], churn_data['y'])
    
    visualizer.plot_feature_importance(
        rf_churn.feature_importance,
        title="客户流失预测特征重要性"
    )
    
    # ================================================================
    # 示例4：NSGA-II多目标优化
    # ================================================================
    print("\n" + "="*70)
    print("📍 EXAMPLE 4: NSGA-II多目标优化")
    print("="*70)
    
    # 定义两个目标函数
    def objective1(x):
        """成本最小化"""
        return x[0]**2 + x[1]**2
    
    def objective2(x):
        """效率最大化（转为最小化）"""
        return (x[0] - 2)**2 + (x[1] - 2)**2
    
    nsga = NSGAII(
        objectives=[objective1, objective2],
        bounds=[(0, 5), (0, 5)],
        n_dims=2,
        pop_size=50,
        max_iter=100,
        verbose=True
    )
    
    pareto_solutions, pareto_front = nsga.optimize()
    
    visualizer.plot_pareto_front(
        pareto_front,
        obj_names=['成本 (最小化)', '效率损失 (最小化)'],
        title="Pareto最优前沿"
    )
    
    # ================================================================
    # 使用说明
    # ================================================================
    print("\n" + "="*70)
    print("📖 使用说明 (Usage Guide)")
    print("="*70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                     分类模型使用指南                             │
    └─────────────────────────────────────────────────────────────────┘
    
    【基本使用】
    
    1️⃣ 随机森林
       model = RandomForestModel(n_estimators=100)
       model.fit(X, y, test_size=0.2)
       predictions = model.predict(X_new)
    
    2️⃣ 集成分类器
       ensemble = EnsembleClassifier()
       ensemble.add_default_models()
       ensemble.fit(X, y)
    
    3️⃣ 多目标优化
       nsga = NSGAII(objectives=[obj1, obj2], bounds=[(0,5),(0,5)], n_dims=2)
       pareto_solutions, pareto_front = nsga.optimize()
    
    【不均衡数据处理】
    
    model = RandomForestModel(class_weight='balanced')
    
    【模型选择建议】
    
    - 小样本: 逻辑回归、SVM
    - 中等样本: 随机森林、梯度提升
    - 大样本: 深度学习
    - 高维稀疏: SVM、Lasso
    
    【论文图表建议】
    
    Figure 1: 混淆矩阵
    Figure 2: ROC曲线
    Figure 3: 特征重要性
    Figure 4: 模型对比（柱状图）
    Figure 5: Pareto前沿（多目标）
    
    Table 1: 数据集描述
    Table 2: 模型参数设置
    Table 3: 评估指标对比
    """)
    
    print("\n" + "="*70)
    print("   ✅ All examples completed successfully!")
    print("   💡 Use the above code templates for your MCM/ICM paper")
    print("="*70)
