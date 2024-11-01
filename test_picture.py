import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
# # 创建网格
# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# x, y = np.meshgrid(x, y)

# # 定义地形的高度（z坐标），这里使用一个简单的函数模拟
# z = np.sin(np.sqrt(x**2 + y**2))
def threeD_picture(rows, columns,depth,filepath):

    #对深度矩阵进行处理，得到三维点
    # 获取图像的高度和宽度
    print("depth矩阵的大小：",depth.shape)
    #x对应width对应columns
    #y对应height对应rows

    height, width = rows, columns

    # 初始化三维点列表
    points = []

    # 假设图像中心为坐标原点
    center_x = width / 2
    center_y = height / 2

    for v in tqdm(range(height - 1), desc="Processing rows"):
        for u in tqdm(range(width - 1), desc="Processing columns", leave=False):
            z = depth[v, u]  # 深度值作为z坐标
            x = u - center_x  # x坐标
            y = v - center_y  # y坐标
            points.append([x, y, z])

    # 转换为NumPy数组
    points = np.array(points)
    # # 按照 z 值从小到大排序
    # start_time = time.time()
    # sorted_points = sorted(points, key=lambda point: point[2])
    # end_time = time.time()
    # print(f"总用时：{end_time-start_time}")
    # print(sorted_points)

    # 输出生成的三维点
    print("生成的三维点：")
    print(points)
    input("请输入任意字符……")
    # 拆分点集
    X = points[:, 0]  # x坐标
    Y = points[:, 1]  # y坐标
    Z = points[:, 2]  # z坐标

    # 构建矩阵X，包含x, y, 以及常数1
    A_matrix = np.column_stack((X, Y, np.ones_like(X)))

    # 使用最小二乘法拟合
    P, residuals, rank, s = np.linalg.lstsq(A_matrix, Z, rcond=None)

    # 得到平面方程的参数A, B, C
    A_fitted, B_fitted, C_fitted = P

    # 计算误差函数
    Z_fitted = A_fitted * X + B_fitted * Y + C_fitted
    error = np.sum((Z - Z_fitted) ** 2)

    # 计算法向量
    plane_normal = np.array([A_fitted, B_fitted, -1])
    horizontal_plane_normal = np.array([0, 0, 1])

    # 计算夹角
    cos_theta = np.dot(plane_normal, horizontal_plane_normal) / (np.linalg.norm(plane_normal) * np.linalg.norm(horizontal_plane_normal))
    angle = np.arccos(cos_theta) * 180 / np.pi

    # 输出结果
    print(f"误差函数值: {error:.2f}")
    print(f"拟合平面的法向量: {plane_normal}")
    print(f"水平面法向量: {horizontal_plane_normal}")
    print(f"夹角: {angle:.2f} 度")


    x = np.linspace(0, columns, columns)
    y = np.linspace(0, rows, rows)
    x, y = np.meshgrid(x, y)
    # 创建三维图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维地形图
    ax.plot_surface(x, y, depth, cmap='terrain')

    # 设置标题和轴标签
    ax.set_title('3D Terrain Map')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Height (Z axis)')
    
    #保存图像为文件
    plt.savefig(filepath)
    # plt.show()
    
