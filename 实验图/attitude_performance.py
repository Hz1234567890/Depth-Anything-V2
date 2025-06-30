import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ========== Key Settings ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # For Chinese characters on Windows
plt.rcParams['axes.unicode_minus'] = False

# ========== Data Processing ==========
df = pd.read_csv('results_4000_8_150.csv', encoding='utf-8')
# Filter records where pitch is greater than -89 degrees
# df = df[df['pitch'] > -89.0]
attitude_df = df[(df['H'] >= 60.0) & (df['H'] <= 64.0)]

# ========== Create Scatter Plot ==========
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    attitude_df['pitch'], 
    attitude_df['performance'],
    c='b',             # Set point color to blue
    s=35,              # Point size
    alpha=0.7,         # Transparency
    edgecolors='w'     # White point borders
)

# ========== Label Settings (English) ==========
plt.title('Impact of Pitch Angle on Performance')
plt.xlabel('Pitch Angle (degrees)')
plt.ylabel('Performance Score')

# ========== Styling ==========
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(range(-90, -50, 5))

# # Optional: Add trendline
# plt.plot(
#     attitude_df['pitch'],
#     pd.Series(attitude_df['performance']).rolling(10, center=True).mean(),
#     'r-',
#     linewidth=2,
#     alpha=0.8,
#     label='Trend Line'
# )
# plt.legend()

plt.tight_layout()
plt.savefig('scatter_pitch_performance.png', dpi=300, bbox_inches='tight')
plt.show()