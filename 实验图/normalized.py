import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('results_12000_8_100.csv')

# 提取需要归一化的三列
cols_to_normalize = ['H', 'accurate', 'performance']
data = df[cols_to_normalize].copy()

# 处理performance列中的缺失值（用0填充）
data['performance'] = data['performance'].fillna(0)

# 最值归一化函数
def min_max_normalize(series):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val)

# 应用归一化
normalized_data = data.apply(min_max_normalize)

# 重命名列以明确含义
normalized_data.columns = ['H', 'Completeness', 'Quality Factor']

# 保存结果到新CSV文件（只保留归一化后的三列）
normalized_data.to_csv('normalized_results.csv', index=False)

print("归一化完成并已保存到 normalized_results.csv")
print("结果包含以下三列：")
print(normalized_data.columns.tolist())