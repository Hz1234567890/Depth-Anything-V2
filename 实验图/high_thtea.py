import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 定义参数范围
H = np.linspace(5, 100, 100)  # 高度范围：500米到30000米
theta_deg = np.linspace(45, 89, 100)  # 角度范围：1度到89度（避免0度除零错误）

# 创建网格
H_grid, theta_grid = np.meshgrid(H, theta_deg)

# 转换为弧度
theta_rad = np.abs(theta_grid) * np.pi / 180
sin_term = np.sin(theta_rad)

# 计算base_pixels
base_pixels = (4.217e7 / H_grid) * (sin_term ** (-1.463)) - 145.3 * theta_grid

# 1. 创建3D图
fig = plt.figure(figsize=(16, 12))

# 1.1 3D表面图
ax1 = fig.add_subplot(221, projection='3d')
surf = ax1.plot_surface(H_grid, theta_grid, base_pixels, 
                       cmap=cm.viridis, alpha=0.8, rstride=3, cstride=3)
ax1.set_xlabel('Height (m)')
ax1.set_ylabel('Angle (degrees)')
ax1.set_zlabel('Base Pixels')
ax1.set_title('3D Surface Plot of Base Pixels')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

# 1.2 3D等高线图
ax2 = fig.add_subplot(222, projection='3d')
contour = ax2.plot_surface(H_grid, theta_grid, base_pixels, 
                          cmap=cm.viridis, alpha=0.8, rstride=3, cstride=3)
ax2.contour(H_grid, theta_grid, base_pixels, zdir='z', offset=np.min(base_pixels), 
           cmap=cm.viridis)
ax2.set_xlabel('Height (m)')
ax2.set_ylabel('Angle (degrees)')
ax2.set_zlabel('Base Pixels')
ax2.set_title('3D Contour Plot of Base Pixels')
fig.colorbar(contour, ax=ax2, shrink=0.5, aspect=5)

# 2. 创建二维投影图
# 2.1 固定角度，变化高度
ax3 = fig.add_subplot(223)
fixed_angles = [45, 65, 75, 80, 85, 90]  # 固定角度值
for angle in fixed_angles:
    idx = np.argmin(np.abs(theta_deg - angle))
    ax3.plot(H, base_pixels[idx, :], label=f'{angle}°')
ax3.set_xlabel('Height (m)')
ax3.set_ylabel('Base Pixels')
ax3.set_title('Fixed Angle - Varying Height')
ax3.legend()
ax3.grid(True)

# 2.2 固定高度，变化角度
ax4 = fig.add_subplot(224)
fixed_heights = [10, 20, 30, 40, 60, 80]  # 固定高度值
for height in fixed_heights:
    idx = np.argmin(np.abs(H - height))
    ax4.plot(theta_deg, base_pixels[:, idx], label=f'{height}m')
ax4.set_xlabel('Angle (degrees)')
ax4.set_ylabel('Base Pixels')
ax4.set_title('Fixed Height - Varying Angle')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()