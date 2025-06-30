import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

# 设置中文字体支持（如果标签需要中文）
plt.rcParams['font.family'] = 'serif'  # 使用衬线字体族
plt.rcParams['font.serif'] = ['Times New Roman']  # 默认英文字体[6,11](@ref)
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取Excel文件
file_path = 'normalized_results.csv'
df = pd.read_csv(file_path)

# 数据清洗
df.fillna(0, inplace=True)  # 用0填充缺失值

# 转换数值类型
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 数据重塑为长格式
df_long = pd.melt(
    df,
    # id_vars=['Accession'],
    value_vars=df.columns[0:],
    var_name='Sample',
    value_name='Expression'
)

# 重命名样本标签为英文
# 注意：根据你的样本数量需要调整此处映射关系
# 这里假设只有两个样本，分别命名为'H'和'Completeness and Quality Factor'
sample_rename_mapping = {
    df.columns[0]: 'H',
    df.columns[1]: 'Completeness',
    df.columns[2]: 'Quality Factor'
}
df_long['Sample'] = df_long['Sample'].map(sample_rename_mapping)
print(df_long)

# 创建小提琴箱线图
plt.figure(figsize=(14, 8))
ax = plt.subplot(111)

# 绘制小提琴图
sns.violinplot(
    x='Sample',
    y='Expression',
    data=df_long,
    inner=None,
    palette='Set3',
    cut=0,
    ax=ax
)

# 叠加箱线图
sns.boxplot(
    x='Sample',
    y='Expression',
    data=df_long,
    width=0.15,
    boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1.5},
    whiskerprops={'color': 'black', 'linewidth': 1.5},
    medianprops={'color': 'red', 'linewidth': 2},
    flierprops={'marker': 'x', 'markersize': 4, 'markeredgecolor': 'black'},
    ax=ax
)

# 添加统计信息
sample_means = df_long.groupby('Sample',sort=False)['Expression'].mean().values
print(sample_means)
for i, mean_val in enumerate(sample_means):

    ax.text(i, mean_val, f'{mean_val:.2f}',
            horizontalalignment='center',
            color='blue',
            weight='bold',
            fontsize=16)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.6, which='major', axis='y')

plt.xticks(fontsize=16)  # X轴刻度标签
plt.yticks(fontsize=16)  # Y轴刻度标签

# 设置图表标题和标签（英文）
plt.title('Box plot of H-Completeness-Quality distribution', fontsize=18, pad=20)
plt.xlabel('Parameters', fontsize=18)  # 横坐标统一标题
plt.ylabel('Normalized Expression', fontsize=18)  # 纵坐标标签
plt.xticks(rotation=15)

# 调整布局
plt.tight_layout()

# 保存高质量图片
plt.savefig('protein_expression_violin_boxplot.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()