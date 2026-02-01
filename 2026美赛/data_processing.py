import os
import json
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

# Mapping: career -> ability/skill file codes present in workspace
CAREER_FILE_MAP = {
    'software_engineer': {'abilities': 'abilities_15-1252-00.csv', 'skills': 'skills_15-1252-00.csv'},
    'software_neighbor': {'abilities': 'abilities_15-1243-00.csv', 'skills': 'skills_15-1243-00.csv'},
    'graphic_designer': {'abilities': 'abilities_27-1024-00.csv', 'skills': 'skills_27-1024-00.csv'},
    'graphic_neighbor': {'abilities': 'abilities_27-1011-00.csv', 'skills': 'skills_27-1011-00.csv'},
    'chef': {'abilities': 'abilities_35-1011-00.csv', 'skills': 'skills_35-1011-00.csv'},
    'chef_neighbor': {'abilities': 'abilities_11-9051-00.csv', 'skills': 'skills_11-9051-00.csv'}
}

# Feature mapping: target feature -> list of candidate terms (prioritized)
# Adjusted for different careers to avoid all-zero vectors
FEATURE_MAP = {
    'Analytical': ['Deductive Reasoning', 'Complex Problem Solving', 'Mathematical Reasoning', 'Inductive Reasoning'],
    'Creative': ['Originality', 'Visualization', 'Fluency of Ideas'],
    'Technical': ['Programming', 'Technology Design', 'Computer Programming', 'Equipment Maintenance', 'Operation and Control'],  # Added for chef
    'Interpersonal': ['Social Perceptiveness', 'Speaking', 'Oral Comprehension'],
    'Physical': ['Finger Dexterity', 'Manual Dexterity', 'Arm-Hand Steadiness']
}


def _read_csv_if_exists(fn):
    if not fn or not os.path.exists(fn):
        return None
    try:
        return pd.read_csv(fn)
    except Exception:
        return None


def extract_vector_for_files(ability_file, skill_file):
    """从两个 CSV 中提取 5 维向量（0-1 归一化）"""
    values = {k: [] for k in FEATURE_MAP.keys()}

    df_a = _read_csv_if_exists(ability_file)
    df_s = _read_csv_if_exists(skill_file)

    # helper to search a dataframe for a term and return its Importance if present
    def find_importance(df, term):
        if df is None:
            return None
        # 自动识别列名：第一列通常是分数，第二列通常是名字
        val_col = df.columns[0] 
        name_col = df.columns[1]
        
        # 模糊匹配：去除空格、转小写
        mask = df[name_col].str.strip().str.lower() == term.strip().lower()
        if mask.any():
            return float(df.loc[mask, val_col].values[0])
        return None

    # search each feature's candidate terms in abilities then skills
    for feat, terms in FEATURE_MAP.items():
        found = []
        for t in terms:
            imp = None
            if df_a is not None:
                imp = find_importance(df_a, t)
            if imp is None and df_s is not None:
                imp = find_importance(df_s, t)
            if imp is not None:
                found.append(imp)
        # aggregate: mean of found, or 0 if none
        values[feat] = float(np.mean(found)) if len(found) > 0 else 0.0

    # Normalize to 0-1 using Min-Max scaling instead of /100
    vec = np.array([values['Analytical'], values['Creative'], values['Technical'], values['Interpersonal'], values['Physical']], dtype=float)
    if vec.max() > 0:
        vec = (vec - vec.min()) / (vec.max() - vec.min())
    return vec.tolist()


def compute_all_vectors(out_json='career_vectors.json'):
    results = {}
    for key, files in CAREER_FILE_MAP.items():
        vec = extract_vector_for_files(files.get('abilities'), files.get('skills'))
        results[key] = vec

    # compute cosine similarities for origin vs neighbor pairs
    pairs = [('software_engineer', 'software_neighbor'), ('graphic_designer', 'graphic_neighbor'), ('chef', 'chef_neighbor')]
    cpe = {}
    for a, b in pairs:
        va = np.array(results.get(a, [0]*5))
        vb = np.array(results.get(b, [0]*5))
        denom = (np.linalg.norm(va) * np.linalg.norm(vb))
        sim = float(np.dot(va, vb) / denom) if denom > 0 else 0.0
        cpe[f'{a}_to_{b}'] = sim

    out = {'vectors': results, 'cpe': cpe}
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out


def load_vectors(json_path='career_vectors.json'):
    if not os.path.exists(json_path):
        return compute_all_vectors(json_path)
    with open(json_path, encoding='utf-8') as f:
        return json.load(f)


def build_school_params(schoolStudentNumber_csv='schoolStudentNumber.csv'):
    """
    构建学校参数字典
    
    学校数据来源：
    - CMU (Carnegie Mellon University): 软件工程方向
    - CCAD (Columbus College of Art & Design): 艺术设计方向  
    - CIA (Culinary Institute of America): 烹饪艺术方向
    """
    default = {
        'CMU': {'lambda': 0.15, 'E_cost': 0.8, 'R_risk': 0.4},
        'CCAD': {'lambda': 0.08, 'E_cost': 0.5, 'R_risk': 0.7},
        'CIA': {'lambda': 0.05, 'E_cost': 0.2, 'R_risk': 0.9}
    }

    # read current graduates from schoolStudentNumber.csv if available
    grads = {}
    if os.path.exists(schoolStudentNumber_csv):
        df = pd.read_csv(schoolStudentNumber_csv, skipinitialspace=True)
        # 去除列名的空格
        df.columns = df.columns.str.strip()
        for _, row in df.iterrows():
            name = str(row['school_name']).strip().upper()
            try:
                grads[name] = float(row['graduate_number'])
            except Exception:
                pass

    out = {}
    for s, v in default.items():
        current = grads.get(s, 300.0)  # 直接从grads字典获取

        # 区分化的课程初始结构
        if s == 'CMU':
            # 技术密集型: 重 Base 和 Proj
            curr = {'x_base': 60, 'x_AI': 5, 'x_ethics': 15, 'x_proj': 40}
        elif s == 'CCAD':
            # 创意密集型: 重 Project, 中等 Base
            curr = {'x_base': 40, 'x_AI': 5, 'x_ethics': 15, 'x_proj': 60}
        elif s == 'CIA':
            # 实践密集型: 重 Project (厨房实操), 较少 Base (理论)
            curr = {'x_base': 30, 'x_AI': 2, 'x_ethics': 8, 'x_proj': 80}
        else:
            curr = {'x_base': 50, 'x_AI': 5, 'x_ethics': 15, 'x_proj': 50}

        out[s] = {
            'lambda': v['lambda'],
            'current_graduates': current,
            'E_cost': v['E_cost'],
            'R_risk': v['R_risk'],
            'current_curriculum': curr
        }

    return out


if __name__ == '__main__':
    print('Computing career vectors from local CSVs...')
    out = compute_all_vectors()
    print(json.dumps(out, indent=2, ensure_ascii=False))
