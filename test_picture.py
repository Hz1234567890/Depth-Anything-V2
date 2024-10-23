import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# # 创建网格
# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# x, y = np.meshgrid(x, y)

# # 定义地形的高度（z坐标），这里使用一个简单的函数模拟
# z = np.sin(np.sqrt(x**2 + y**2))
def threeD_picture(rows, columns,z):
    x = np.linspace(0, columns, columns)
    y = np.linspace(0, rows, rows)
    x, y = np.meshgrid(x, y)
    # 创建三维图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维地形图
    ax.plot_surface(x, y, z, cmap='terrain')

    # 设置标题和轴标签
    ax.set_title('3D Terrain Map')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Height (Z axis)')

    plt.show()
