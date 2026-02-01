# Task 2 模型详细梳理：AI驱动的教育决策模型

本模型旨在解决高等教育机构在面对AI变革时的决策问题，主要包含三个相互关联的子模型：**宏观招生响应模型**、**微观课程优化模型**以及作为安全网的**职业路径弹性模型**。

## 1. 符号说明 (Nomenclature)

| 符号 | 含义 | 单位/备注 |
| :--- | :--- | :--- |
| $t$ | 时间年份 | Year |
| $E_t$ | 第 $t$ 年的毕业生供给量 (Enrollment/Graduates) | 人数 |
| $D_t$ | 第 $t$ 年的市场需求量 (Demand) | 人数 |
| $\Gamma_t$ | 供需压力指数 (Pressure Index) | 无量纲 |
| $\lambda$ | 行政调整系数 (Administrative Inertia) | $0 < \lambda < 1$ |
| $\mathbf{X}$ | 课程学分分配向量 | $[x_{base}, x_{AI}, x_{ethics}, x_{proj}]$ |
| $U(\mathbf{X})$ | 技能总效用函数 | 分值 |
| $w_i$ | 各类课程的技能权重 | 权重系数 |
| $C_{total}$ | 总学分约束 | 120学分 |
| $\vec{V}_c$ | 职业 $c$ 的技能特征向量 | 5维向量 |
| $S(\vec{u}, \vec{v})$ | 职业间的相似度 (Elasticity) | $[0, 1]$ |
| $A_{C_k}$ | 准则 $C_k$ 下的判断矩阵 (AHP) | $n \times n$ 矩阵 |
| $W_C$ | 准则权重向量 (AHP) | $[0.4, 0.4, 0.2]$ |
| $Z_i$ | 学校 $i$ 的综合得分 (AHP) | 无量纲 |
| $CI$ | 一致性指标 (Consistency Index) | 无量纲 |
| $CR$ | 一致性比率 (Consistency Ratio) | $CR < 0.1$ 为合格 |
| $\lambda_{max}$ | 判断矩阵最大特征值 | 无量纲 |

---

## 1.5 数据预处理与清洗 (Data Preprocessing)

### 1.5.1 数据源说明

本模型使用的数据来源于以下文件：

| 文件名 | 描述 | 关键字段 |
| :--- | :--- | :--- |
| `schoolStudentNumber.csv` | 各学校毕业生人数 | `school_name`, `graduate_number` |
| `career_vectors.json` | 职业技能特征向量 | 5维技能向量 |
| `*_skills.csv` / `*_abilities.csv` | O*NET职业技能数据 | 技能名称、重要性评分 |

### 1.5.2 学校名称标准化

**问题发现：** 数据集与代码中的学校名称需保持一致。

| 数据源标识 | 学校全称 | 代码中使用 |
| :--- | :--- | :--- |
| `CMU` | Carnegie Mellon University (卡内基梅隆大学) | `CMU` |
| `CCAD` | Columbus College of Art & Design (哥伦布艺术与设计学院) | `CCAD` |
| `CIA` | Culinary Institute of America (美国烹饪学院) | `CIA` |

**学校-职业映射关系：**
```python
SCHOOL_CAREER_MAPPING = {
    'CMU': 'software_engineer',     # 软件工程师
    'CCAD': 'graphic_designer',     # 平面设计师  
    'CIA': 'chef'                   # 厨师
}
```

### 1.5.3 CSV数据清洗

**问题描述：** `schoolStudentNumber.csv` 文件存在列名空格问题。

**原始格式：**
```csv
school_name, graduate_number
CMU, 1073
CCAD, 900
CIA, 3011
```

**问题：** 列名包含前导/尾随空格（如 `" graduate_number"`），导致 `KeyError`。

**解决方案：**
```python
# 在读取CSV时处理空格
df = pd.read_csv('schoolStudentNumber.csv', skipinitialspace=True)
df.columns = df.columns.str.strip()  # 移除列名空格
```

### 1.5.4 毕业生数据统计

| 学校 | 毕业生人数 | 对应职业 |
| :--- | :--- | :--- |
| CMU | 1,073 | Software Engineer |
| CCAD | 900 | Graphic Designer |
| CIA | 3,011 | Chef |

> **注意：** 这些数值直接用于动态招生响应模型的 $E_{current}$ 参数。

---

## 2. 子模型 I：动态招生响应模型 (Dynamic Enrollment Response Model)

### 2.1 模型逻辑
该子模型用于决定学校在宏观层面应如何调整招生规模。主要思想是建立一个负反馈控制系统，使毕业生供给 $E_t$ 逐步逼近市场预测需求 $D_t$，但受限于学校行政能力的惯性（Physical constraints and Administrative inertia）。

### 2.2 数学表达

**Step 1: 定义供需压力指数 (Pressure Index)**
衡量当前供给与未来需求的不匹配程度：
$$ \Gamma_t = \frac{D_{future} - E_{current}}{E_{current}} $$

**Step 2: 定义调整响应函数 (Response Function)**
学校的调整能力不是线性的，而是存在饱和效应。我们引入双曲正切函数 $\tanh$ 来模拟这种饱和特性，并使用 $\lambda$ 作为调整上限系数。

$$ \Delta E = E_{current} \cdot \lambda \cdot \tanh(\Gamma_t) $$

其中：
*   $\lambda$ (Lambda)：行政调整系数。例如 $\lambda=0.15$ 表示学校一年最多能调整 15% 的招生规模。CMU这类综合性大学 $\lambda$ 较高，而CCAD这类艺术设计院校 $\lambda$ 较低。
*   $\tanh(\cdot)$：将压力指数映射到 $(-1, 1)$ 区间，防止因预测偏差过大导致招生人数剧烈波动。

**Step 3: 状态更新**
$$ E_{new} = E_{current} + \Delta E $$

### 2.3 为什么这样设定？
*   ** $\tanh$ 的使用**：模拟了现实中的资源瓶颈。即使市场需求翻倍（$\Gamma$ 很大），学校也不可能瞬间扩招一倍，调整量会被 $\tanh$ 限制在 $\lambda$ 附近。
*   ** $\lambda$ 的差异化**：不同类型的学校（综合性 vs 艺术类）对市场变化的敏感度和调整能力不同。

---

## 3. 子模型 II：课程结构优化模型 (Curriculum Optimization Model)

### 3.1 模型逻辑
在确定了招生规模后，需决定具体的培养方案。这是一个资源分配优化问题：在总学分有限的情况下，如何分配不同类型课程的学分，以最大化学生毕业后的综合竞争力（技能效用），同时避免课程体系剧烈变动带来的混乱。

### 3.2 自变量与决策空间
自变量为课程学分分配向量 $\mathbf{X}$：
$$ \mathbf{X} = [x_{base}, x_{AI}, x_{ethics}, x_{proj}]^T $$
分别代表：基础学科、AI技能、伦理/社会科学、项目实践。

### 3.3 目标函数 (Objective Function)
我们要最大化净效用 $J(\mathbf{X})$：
$$ \text{Maximize } J(\mathbf{X}) = U_{skill}(\mathbf{X}) - C_{trans}(\mathbf{X}) $$

**1. 技能总效用 $U_{skill}$：**
$$ U_{skill}(\mathbf{X}) = \sum_{i \in \{base, AI, ethics, proj\}} w_i \cdot x_i $$
*   $w_i$：各类课程的边际技能贡献率（权重）。例如 AI 课程的权重 $w_{AI}$ 设为 0.4（未来需求高），而基础课程 $w_{base}$ 设为 0.3。

**2. 平滑过渡成本 $C_{trans}$ (Transition Cost)：**
为了惩罚过于激进的课程改革，引入变革成本：
$$ C_{trans}(\mathbf{X}) = \gamma \cdot \sum_{i} \mathbb{I}\left( \left| \frac{x_i - x_i^{old}}{x_i^{old}} \right| > \delta \right) \cdot \left| \frac{x_i - x_i^{old}}{x_i^{old}} \right| $$
*   逻辑：只有当某类课程的变动幅度超过阈值 $\delta$ (如 30%) 时，才计算惩罚。
*   代码实现中 $\gamma=0.05, \delta=0.3$。

### 3.4 约束条件 (Constraints)
$$ \begin{cases} \sum_{i} x_i = 120 & (\text{Total Credits Constraint}) \\ x_{base} \ge 40 & (\text{Foundation Requirement}) \\ x_{AI} \ge 5 & (\text{Minimum AI Literacy}) \\ x_i \ge 0 & (\text{Non-negativity}) \end{cases} $$

### 3.5 求解算法：模拟退火 (Simulated Annealing)
由于目标函数可能非凸且存在整数约束（学分通常取整），采用模拟退火算法求解全局最优。
*   **扰动机制**：随机选择两类课程，一增一减 $k \in [1, 5]$ 学分，保持总和不变。
*   **接受准则**：Metropolis 准则，以概率 $P = \exp(\frac{\Delta J}{T})$ 接受劣解，避免陷入局部最优。

---

## 4. 子模型 III：职业路径弹性模型 (Career Path Elasticity Model)

### 4.1 模型逻辑
考虑到AI时代的职业不确定性，单一的最优课程是不够的。该模型评估当前培养方案下的学生在面临失业风险时，转型到其他职业的容易程度（Elasticity）。

### 4.2 数学表达

**1. 职业向量空间**
将每个职业 $c$ 定义为5维技能空间中的一个向量：
$$ \vec{V}_c = [v_{analytical}, v_{creative}, v_{technical}, v_{interpersonal}, v_{physical}] $$
数据来源于 O*NET 数据库。

**2. 弹性度量 (Elasticity Metric)**
定义从源职业 $c_{origin}$ 到目标职业 $c_{target}$ 的弹性为两个向量的余弦相似度：
$$ S(c_{origin}, c_{target}) = \cos(\theta) = \frac{\vec{V}_{origin} \cdot \vec{V}_{target}}{\|\vec{V}_{origin}\| \|\vec{V}_{target}\|} $$

*   $S \to 1$：技能高度重合，转型容易（高弹性）。
*   $S \to 0$：技能完全正交，转型困难（低弹性）。

**3. 转移差距 (Transfer Gap)**
识别转型的最大瓶颈维度 $k^*$：
$$ k^* = \arg\max_{k} |v_{origin}^{(k)} - v_{target}^{(k)}| $$
这为学校提供了具体的改进建议（例如：需加强 Creative 维度以增加向 Designer 转型的可能性）。

---

## 5. 模型参数设定依据

### 5.1 行政调整系数 λ 的AHP推导 (AHP-based λ Parameter Estimation)

行政调整系数 $\lambda$ 是动态招生响应模型的核心参数，决定了学校对市场变化的响应能力。我们采用**层次分析法（Analytic Hierarchy Process, AHP）**进行科学的参数估计。

#### 5.1.1 评价体系构建 (Hierarchy Structure)

**目标层 (Goal)**：评估机构扩招潜力 ($\lambda$)

**准则层 (Criteria)**：
| 准则 | 符号 | 权重 | 含义 |
| :--- | :--- | :--- | :--- |
| 战略灵活性 | $C_1$ | 0.4 | 课程数字化程度，是否可远程/在线扩展 |
| 硬件独立性 | $C_2$ | 0.4 | 对物理设施（实验室、工作室、厨房）的依赖程度 |
| 服务弹性 | $C_3$ | 0.2 | 教学服务的可扩展性（助教、班级规模、安全限制） |

**方案层 (Alternatives)**：CMU (卡内基梅隆大学), CCAD (哥伦布艺术与设计学院), CIA (美国烹饪学院)

#### 5.1.2 判断矩阵构造 (Pairwise Comparison Matrices)

使用 Saaty 1-9标度法构造判断矩阵，基于以下定性分析：
- **CMU**：课程高度数字化，主要依赖计算机，可雇用大量助教，灵活性最高
- **CCAD**：需要画室和工作室，有一定物理限制，小班制教学
- **CIA**：必须使用厨房设备，受食品安全法规和设备工位限制，灵活性最低

**矩阵 1：关于 $C_1$ 战略灵活性**
$$A_{C_1} = \begin{bmatrix} 1 & 3 & 7 \\ 1/3 & 1 & 3 \\ 1/7 & 1/3 & 1 \end{bmatrix}$$
- CMU vs CCAD = 3（稍微重要）：CMU课程可完全在线化
- CMU vs CIA = 7（非常重要）：软件课程vs厨艺课程

**矩阵 2：关于 $C_2$ 硬件独立性**
$$A_{C_2} = \begin{bmatrix} 1 & 5 & 9 \\ 1/5 & 1 & 3 \\ 1/9 & 1/3 & 1 \end{bmatrix}$$
- CMU vs CCAD = 5（明显重要）：电脑vs工作室
- CMU vs CIA = 9（极端重要）：软件vs商业厨房

**矩阵 3：关于 $C_3$ 服务弹性**
$$A_{C_3} = \begin{bmatrix} 1 & 3 & 5 \\ 1/3 & 1 & 2 \\ 1/5 & 1/2 & 1 \end{bmatrix}$$
- CMU可大量雇用TA，CCAD小班制，CIA受安全法规限制

#### 5.1.3 一致性检验 (Consistency Check)

对每个判断矩阵计算一致性比率 $CR$：
$$ CR = \frac{CI}{RI}, \quad CI = \frac{\lambda_{max} - n}{n - 1} $$

其中 $RI$ 为随机一致性指标（$n=3$ 时 $RI=0.58$）。

| 矩阵 | $\lambda_{max}$ | $CI$ | $CR$ | 判定 |
| :--- | :--- | :--- | :--- | :--- |
| $A_{C_1}$ | 3.004 | 0.002 | 0.006 | ✅ 通过 ($<0.1$) |
| $A_{C_2}$ | 3.029 | 0.015 | 0.025 | ✅ 通过 ($<0.1$) |
| $A_{C_3}$ | 3.002 | 0.001 | 0.003 | ✅ 通过 ($<0.1$) |

#### 5.1.4 综合计算与 λ 映射 (Synthesis & Mapping)

**优先权重向量** (Priority Vectors)：
| 学校 | $C_1$ 得分 | $C_2$ 得分 | $C_3$ 得分 |
| :--- | :--- | :--- | :--- |
| CMU | 0.669 | 0.751 | 0.648 |
| CCAD | 0.243 | 0.178 | 0.230 |
| CIA | 0.088 | 0.070 | 0.122 |

**综合得分** (Composite Score)：
$$ Z_i = 0.4 \times S_{C_1} + 0.4 \times S_{C_2} + 0.2 \times S_{C_3} $$

**λ 值映射**（映射区间 $[\lambda_{min}, \lambda_{max}] = [0.02, 0.18]$）：
$$ \lambda = 0.02 + 0.16 \times Z $$

**最终结果**：
| 学校 | 综合得分 $Z$ | $\lambda$ 值 | 含义 |
| :--- | :--- | :--- | :--- |
| CMU | 0.698 | **0.132** (13.2%) | 高弹性，可快速响应市场 |
| CCAD | 0.214 | **0.054** (5.4%) | 中等弹性，受工作室限制 |
| CIA | 0.088 | **0.034** (3.4%) | 低弹性，受厨房设施限制 |

#### 5.1.5 结论解释
> "Through the AHP analysis, we derived objective capacity constraints. CMU ($\lambda=0.132$) demonstrates high elasticity due to its digital nature. In contrast, CIA ($\lambda=0.034$) is heavily constrained by physical infrastructure, limiting its annual growth potential to roughly 3% regardless of market demand."

---

### 5.2 其他参数设定

| 参数 | 代码变量 | 设定值 | 设定依据 |
| :--- | :--- | :--- | :--- |
| **学分总数** | `total_credits` | 120 | 美国本科标准毕业学分要求。 |
| **基础课底线** | `constraint` | 40 | 保证通识教育和核心学科基础，约占总学分的 1/3。 |
| **AI课底线** | `constraint` | 5 | 确保所有毕业生具备最低限度的数字素养。 |
| **技能权重** | `skill_weights` | AI: 0.4 <br> Base: 0.3 <br> Ethics: 0.2 <br> Proj: 0.1 | 基于对未来AI时代需求的假设：Technical技能溢价最高，其次是坚实的Base。 |
| **过渡阈值** | `change_ratio` | 0.3 | 只有超过30%的课程变动才会被视为剧烈变动，触发阻力成本。 |

---

## 6. 模型工作流程 (Model Workflow)

完整的AI教育决策模型工作流程如下：

```
┌─────────────────────────────────────────────────────────────┐
│                    Stage 0: AHP参数估计                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │ C1矩阵  │    │ C2矩阵  │    │ C3矩阵  │                 │
│  │战略灵活性│    │硬件独立性│    │服务弹性 │                 │
│  └────┬────┘    └────┬────┘    └────┬────┘                 │
│       │              │              │                        │
│       └──────────────┼──────────────┘                        │
│                      ↓                                       │
│              ┌──────────────┐                               │
│              │ 综合得分 Z   │                               │
│              │ → λ 值映射  │                               │
│              └──────────────┘                               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Stage 1: 动态招生响应模型                        │
│  Input: E_current, D_future, λ (from AHP)                   │
│  Output: Γ (压力指数), ΔE (调整幅度), E_new (推荐招生数)      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Stage 2: 课程结构优化模型 (SA)                   │
│  Input: 当前课表, 技能权重, 约束条件                          │
│  Output: 最优学分分配 X* = [x_base, x_AI, x_ethics, x_proj]  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Stage 3: 职业路径弹性评估                        │
│  Input: 职业技能向量 (O*NET)                                 │
│  Output: 余弦相似度矩阵, 转型瓶颈分析                         │
└─────────────────────────────────────────────────────────────┘
```

## 7. 输出结果与图表解读指南 (Interpretation Guide)

###  1. AHP雷达图 (ahp_radar_analysis.png)

**图表说明：**
*   **三角形雷达图**展示3所学校在3个评估维度的表现
*   **C1 (Strategic Scalability):** 战略可扩展性 - 学校扩招的战略灵活性
*   **C2 (Physical Independence):** 物理独立性 - 硬件设施不受限于学生数量的程度
*   **C3 (Service Elasticity):** 服务弹性 - 师生配比调整能力

**解读方法：**
| 学校 | λ值 | 含义 |
| :--- | :--- | :--- |
| **CMU** | 0.132 (13.2%) | 最高扩容能力，可承受最大招生调整 |
| **CCAD** | 0.054 (5.4%) | 中等扩容能力 |
| **CIA** | 0.034 (3.4%) | 最小扩容能力，需谨慎调整招生 |

###  2. AHP汇总表 (ahp_summary_table.png)

**关键列解读：**
*   **Composite Score (Z):** 综合得分 =  \times C1 + W_2 \times C2 + W_3 \times C3$
*   **Final λ:** 归一化后的行政调整系数，直接用于招生决策公式

###  3. 招生响应图 (enrollment_response_*.png)

**图表元素：**
*   **X轴:** 时间 (2025-2035年)
*   **Y轴:** 学生人数
*   **蓝线:** 市场需求预测 $(t)$
*   **橙线:** 当前供给趋势 $(t)$
*   **绿线:** 优化后的招生计划 $(t)$

**解读要点：**
*   **压力指数 P(t):**
    *    > 0 $\rightarrow$ 市场需求超过供给，需要扩招
    *    < 0 $\rightarrow$ 供给过剩，需要缩减
*   **调整量 ΔA:** 实际招生调整 = $\lambda \times P(t) \times$ 当前学生数

###  4. 技能雷达图 (skill_radar_*.png)

**图表说明：**
*   展示目标职业所需技能与学校课程培养能力的匹配度
*   每个维度代表一项核心技能

**解读方法：**
*   **重叠面积越大** $\rightarrow$ 课程与职业需求匹配度越高
*   **Similarity Score:** 余弦相似度 (0-1)，$>0.7$ 表示良好匹配

###  5. 职业弹性图 (career_elasticity_*.png)

**图表说明：**
*   水平条形图展示各职业的匹配度
*   颜色渐变：绿色=高匹配，黄色=中等，红色=低匹配

**解读阈值：**

| 相似度 | 评估 | 含义 |
| :--- | :--- | :--- |
| > 0.7 |  高弹性 | 学生可轻松转型到该职业 |
| 0.5-0.7 |  中等 | 需要额外培训 |
| < 0.5 |  低弹性 | 需要大量再培训 |

###  6. 模拟退火收敛图 (sa_convergence_*.png)

**图表说明：**
*   展示课程优化算法的收敛过程
*   X轴：迭代次数，Y轴：目标函数值

**解读要点：**
*   **曲线下降** $\rightarrow$ 算法正在找到更优解
*   **趋于平稳** $\rightarrow$ 达到最优或接近最优
*   **改进百分比:** 显示优化效果

###  7. 帕累托前沿图 (pareto_frontier_*.png)

**图表说明：**
*   展示多目标优化的权衡关系
*   每个点代表一个可行的课程方案

**解读方法：**
*   **前沿上的点** $\rightarrow$ 帕累托最优解（不能在不损害其他目标的情况下改进）
*   **选择策略:** 根据学校偏好在前沿上选择平衡点

