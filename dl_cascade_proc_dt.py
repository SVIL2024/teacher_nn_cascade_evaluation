import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def parse_excel_hierarchy(file_path):
    """
    解析Excel文件中的层级结构，支持多sheet和多级
    """
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names
    hierarchy = {}
    for sheet in sheets:
        df = pd.read_excel(xls, sheet_name=sheet, header=None)
        first_row = df.iloc[0].tolist()
        second_row = df.iloc[1].tolist()
        current_second_level = None
        sheet_hierarchy = {}
        for col_idx, (sec, thr) in enumerate(zip(first_row, second_row)):
            if pd.notna(sec):
                current_second_level = sec
                sheet_hierarchy[current_second_level] = []
            if pd.notna(thr) and current_second_level is not None:
                sheet_hierarchy[current_second_level].append(f"{current_second_level}_{thr}")
        # 构建 raw_columns 字段
        raw_columns = []
        current_sec = None
        for sec, thr in zip(first_row, second_row):
            if pd.notna(sec):
                current_sec = sec
            if pd.notna(thr):
                raw_columns.append(f"{current_sec}_{thr}")
            else:
                raw_columns.append(current_sec)
        sheet_hierarchy['raw_columns'] = raw_columns
        hierarchy[sheet] = sheet_hierarchy
    return hierarchy

def load_data(file_path):
    """
    加载Excel文件并修复列名
    """
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names
    data = {}
    hierarchy_all = parse_excel_hierarchy(file_path)
    for sheet in sheets:
        hierarchy = hierarchy_all[sheet]
        raw_columns = hierarchy.get('raw_columns', [])
        df_full = pd.read_excel(xls, sheet_name=sheet, header=None, skiprows=2)
        if len(df_full.columns) == len(raw_columns):
            df_full.columns = raw_columns
        else:
            raise ValueError(f"列数不匹配：{sheet} 中检测到 {len(df_full.columns)} 列，但期望 {len(raw_columns)} 列")
        data[sheet] = {
            'df': df_full,
            'hierarchy': hierarchy
        }
    return data

def preprocess_data_normNum(df, hierarchy, target_column='score', one_hot_cols=None):
    """
    预处理数据，并根据层级结构组织输入，返回每个二级指标的特征名列表，便于SHAP解释。
    """
    import torch
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # 分离特征和目标变量
    X = df.drop(columns=[target_column], errors='ignore')
    y = df[target_column] if target_column in df.columns else df.iloc[:, -1]

    # 自动检测分类特征（字符串类型）
    if one_hot_cols is None:
        one_hot_cols = []
        for col in X.columns:
            if any(isinstance(x, str) and not pd.isna(x) for x in X[col]):
                one_hot_cols.append(col)
    numerical_cols = [col for col in X.columns if col not in one_hot_cols]

    # 标准化数值型列
    scaler = MinMaxScaler()
    if numerical_cols:
        X_num_scaled = scaler.fit_transform(X[numerical_cols])
        X_num_scaled = pd.DataFrame(X_num_scaled, columns=numerical_cols, index=X.index)
    else:
        X_num_scaled = pd.DataFrame(index=X.index)

    # One-Hot 编码分类列
    if len(one_hot_cols) > 0:
        X_cat_encoded = pd.get_dummies(X[one_hot_cols], drop_first=False)
    else:
        X_cat_encoded = pd.DataFrame(index=X.index)

    # 合并所有列
    X_processed = pd.concat([X_num_scaled, X_cat_encoded], axis=1)

    
    feature_names = X_processed.columns.tolist()
    X_tensor = torch.tensor(X_processed.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)

    # 划分训练集和测试集（主输入）
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X_tensor, y_tensor, np.arange(len(X_tensor)), test_size=0.2, random_state=42
    )

    # 构建三级指标到编码后特征列索引的映射（用完整列名！）
    third_level_to_indices = {}
    for col in X_cat_encoded.columns:
        third_level_to_indices[col] = [feature_names.index(col)]
    for col in numerical_cols:
        third_level_to_indices[col] = [feature_names.index(col)]

    # 构建按二级指标划分的输入张量和特征名（用主划分索引切片）
    X_train_dict = {}
    X_test_dict = {}
    X_train_dict_names = {}
    X_test_dict_names = {}
    for second_level, third_levels in hierarchy.items():
        if second_level == 'raw_columns':
            continue
        indices = []
        names = []
        for thr in third_levels:
            # 先尝试直接匹配
            if thr in third_level_to_indices:
                indices.extend(third_level_to_indices[thr])
                names.extend([thr])
            else:
                # 匹配所有以thr为前缀的列
                matched = [(name, idx) for name, idxs in third_level_to_indices.items() if name.startswith(thr) for idx in idxs]
                indices.extend([idx for name, idx in matched])
                names.extend([name for name, idx in matched])
        indices = sorted(set(indices))
        names = [feature_names[i] for i in indices]
        if indices:
            X_train_dict[second_level] = X_train[:, indices]
            X_test_dict[second_level] = X_test[:, indices]
            X_train_dict_names[second_level] = names
            X_test_dict_names[second_level] = names

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_dict': X_train_dict,
        'X_test_dict': X_test_dict,
        'X_train_dict_names': X_train_dict_names,
        'X_test_dict_names': X_test_dict_names,
        'feature_names': feature_names,
        'all_features': X.columns.tolist(),
        'preprocessor': {'scaler': scaler}
    }
def preprocess_data(df, hierarchy, target_column='score', one_hot_cols=None):
    """
    预处理数据，并根据层级结构组织输入，返回每个二级指标的特征名列表，便于SHAP解释。
    """
    import torch
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import pandas as pd

    # 分离特征和目标变量
    y = df[target_column] if target_column in df.columns else df.iloc[:, -1]
    X = df.drop(columns=[target_column], errors='ignore')
    # 去掉—_score属性
    X = X.drop(columns=df.columns[-1])
    # 自动检测分类特征（字符串类型）
    if one_hot_cols is None:
        one_hot_cols = []
        for col in X.columns:
            if any(isinstance(x, str) and not pd.isna(x) for x in X[col]):
                one_hot_cols.append(col)
    numerical_cols = [col for col in X.columns if col not in one_hot_cols]

    # One-Hot 编码分类列
    if len(one_hot_cols) > 0:
        X_cat_encoded = pd.get_dummies(X[one_hot_cols], drop_first=False)
    else:
        X_cat_encoded = pd.DataFrame(index=X.index)

    # 合并数值型列和One-Hot编码后的分类列
    X_processed = pd.concat([X[numerical_cols], X_cat_encoded], axis=1)

    # 归一化所有特征
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_processed)
    X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)

    feature_names = X_scaled.columns.tolist()
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)

    # 划分训练集和测试集（主输入）
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X_tensor, y_tensor, np.arange(len(X_tensor)), test_size=0.2, random_state=42
    )

    # 构建三级指标到编码后特征列索引的映射（用完整列名！）
    third_level_to_indices = {}
    for col in X_scaled.columns:
        third_level_to_indices[col] = [feature_names.index(col)]

    # 构建按二级指标划分的输入张量和特征名（用主划分索引切片）
    X_train_dict = {}
    X_test_dict = {}
    X_train_dict_names = {}
    X_test_dict_names = {}
    for second_level, third_levels in hierarchy.items():
        if second_level == 'raw_columns':
            continue
        indices = []
        names = []
        for thr in third_levels:
            # 先尝试直接匹配
            if thr in third_level_to_indices:
                indices.extend(third_level_to_indices[thr])
                names.extend([thr])
            else:
                # 匹配所有以thr为前缀的列
                matched = [(name, idx) for name, idxs in third_level_to_indices.items() if name.startswith(thr) for idx in idxs]
                indices.extend([idx for name, idx in matched])
                names.extend([name for name, idx in matched])
        indices = sorted(set(indices))
        names = [feature_names[i] for i in indices]
        if indices:
            X_train_dict[second_level] = X_train[:, indices]
            X_test_dict[second_level] = X_test[:, indices]
            X_train_dict_names[second_level] = names
            X_test_dict_names[second_level] = names

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_dict': X_train_dict,
        'X_test_dict': X_test_dict,
        'X_train_dict_names': X_train_dict_names,
        'X_test_dict_names': X_test_dict_names,
        'feature_names': feature_names,
        'all_features': X.columns.tolist(),
        'preprocessor': {'scaler': scaler}
    }
def load_and_preprocess_all_sheets(
    file_path,
    target_column='score',
    one_hot_cols=None
):
    """
    加载Excel所有sheet，合并数据并预处理，返回processed_data
    """
    data_by_sheet = load_data(file_path)
    # 合并所有sheet的数据
    all_dfs = []
    all_hierarchies = []
    for sheet in data_by_sheet:
        all_dfs.append(data_by_sheet[sheet]['df'])
        all_hierarchies.append(data_by_sheet[sheet]['hierarchy'])
    # 合并所有sheet的数据（假设列名一致）
    df_all = pd.concat(all_dfs, ignore_index=True)
    # 合并所有sheet的层级结构（以第一个为主，或根据需要自定义合并方式）
    hierarchy = all_hierarchies[0]
    # 预处理数据
    processed_data = preprocess_data(df_all, hierarchy, target_column, one_hot_cols)
    return processed_data

# 用法示例
if __name__ == "__main__":
    file_path = "dl_cascade/teacher_info_2025_name-7-3-english.xlsx"
    processed_data = load_and_preprocess_all_sheets(file_path, target_column='score')
    print("X_train shape:", processed_data['X_train'].shape)
    print("X_train_dict keys:", processed_data['X_train_dict'].keys())