from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib as mpl

# ========== 关键字体设置 ========== [1,3,6,11](@ref)
plt.rcParams['font.family'] = 'serif'  # 使用衬线字体族
plt.rcParams['font.serif'] = ['Times New Roman']  # 默认英文字体[6,11](@ref)
plt.rcParams['font.size'] = 20  # 全局增大基础字号
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题[1,3](@ref)

# 添加中文字体支持（备用）
try:
    plt.rcParams['font.serif'].insert(0, 'SimSun')  # 宋体作为中文字体[11](@ref)
except:
    pass  # 忽略字体缺失错误

# ========== 数据加载与处理 ==========
df = pd.read_csv('results_4000_8_150.csv', encoding='utf-8')

# ========== 三维可视化 ==========
fig = plt.figure(figsize=(12, 9), dpi=120)
ax = fig.add_subplot(111, projection='3d')

# 创建带颜色映射的三维散点图
sc = ax.scatter(
    df['H'], 
    df['pitch'], 
    df['performance'],
    c=df['performance'], 
    cmap='viridis', 
    s=50,
    alpha=0.7,
    edgecolor='w',
    linewidth=0.3
)

# ========== 英文标签设置 ==========
ax.set_xlabel('Flight Height (m)', fontsize=18, labelpad=12)
ax.set_ylabel('Pitch Angle (°)', fontsize=18, labelpad=12)
ax.set_zlabel('Q', fontsize=18, labelpad=12)
ax.set_title('3D Relationship: Flight and Pitch Parameters vs Q', 
             fontsize=14, pad=20)

# 添加颜色条并设置标签
cbar = fig.colorbar(sc, ax=ax, shrink=0.7, aspect=20, pad=0.1)
cbar.set_label('Q', fontsize=11)

# ========== 视角优化 ==========
ax.view_init(elev=25, azim=-45)  # 最佳观察角度
ax.dist = 10.5  # 调整相机距离

# ========== 网格和刻度优化 ==========
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, linestyle=':', alpha=0.6)

# ========== 保存高质量图像 ==========
plt.tight_layout(pad=2.5)
plt.savefig('flight_performance_3d.png', dpi=300, bbox_inches='tight')
plt.show()