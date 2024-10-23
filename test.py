import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设一个更大的深度矩阵 (示例 10x10)
depth_matrix = np.random.rand(100, 100) + 1.0  # 随机生成深度矩阵, 增加1确保正值

# 1. 使用 Sobel 算子计算深度图的梯度
# 计算 x 方向梯度
grad_x = cv2.Sobel(depth_matrix, cv2.CV_64F, 1, 0, ksize=3)
# 计算 y 方向梯度
grad_y = cv2.Sobel(depth_matrix, cv2.CV_64F, 0, 1, ksize=3)

# 2. 计算法向量
def compute_normals(grad_x, grad_y):
    h, w = grad_x.shape
    normals = np.zeros((h, w, 3))  # 存储法向量 (Nx, Ny, Nz)
    
    for i in range(h):
        for j in range(w):
            # 每个像素的法向量 Nx, Ny, Nz
            nx = -grad_x[i, j]
            ny = -grad_y[i, j]
            nz = 1.0  # Z方向的默认分量
            
            # 正则化法向量
            norm = np.sqrt(nx**2 + ny**2 + nz**2)
            normals[i, j, 0] = nx / norm
            normals[i, j, 1] = ny / norm
            normals[i, j, 2] = nz / norm
    return normals

# 计算得到法向量矩阵
normals = compute_normals(grad_x, grad_y)

# 3. 基于 RANSAC 进行平面拟合，并处理多个平面
def ransac_multiple_planes(normals, depth_matrix, threshold=0.02, max_iterations=1000, min_inliers=10):
    h, w, _ = normals.shape
    remaining_points = np.ones((h, w), dtype=bool)  # 记录未处理的点
    planes = []
    
    while True:
        best_plane = None
        best_inliers = []
        num_points = np.sum(remaining_points)  # 剩余点数量

        if num_points < min_inliers:
            print(f"停止迭代，剩余点数: {num_points}")
            break  # 如果剩余点少于最小内点数量，停止检测

        # RANSAC 核心算法
        for _ in range(max_iterations):
            # 随机选取三个点
            valid_indices = np.argwhere(remaining_points)  # 获取所有未处理点的索引
            if len(valid_indices) < 3:
                break  # 剩余点不足以选取三个点

            i1, j1 = valid_indices[np.random.randint(0, len(valid_indices))]
            i2, j2 = valid_indices[np.random.randint(0, len(valid_indices))]
            i3, j3 = valid_indices[np.random.randint(0, len(valid_indices))]

            # 确保这三点不同
            if (i1 == i2 and j1 == j2) or (i1 == i3 and j1 == j3) or (i2 == i3 and j2 == j3):
                continue

            # 获取法向量
            p1 = normals[i1, j1]
            p2 = normals[i2, j2]
            p3 = normals[i3, j3]

            # 使用这三个法向量计算平面法向量 (n = p2 - p1 x p3 - p1)
            plane_normal = np.cross(p2 - p1, p3 - p1)

            # 检查法向量的模是否为零，如果为零则跳过本次循环
            norm_plane_normal = np.linalg.norm(plane_normal)
            if norm_plane_normal == 0:
                continue  # 跳过本次迭代

            plane_normal = plane_normal / norm_plane_normal  # 正则化

            # 计算平面参数 (a, b, c, d)
            a, b, c = plane_normal
            d = -(a * p1[0] + b * p1[1] + c * p1[2])

            # 统计符合平面的内点
            inliers = []
            for i in range(h):
                for j in range(w):
                    if remaining_points[i, j]:  # 只考虑剩余的未处理点
                        nx, ny, nz = normals[i, j]
                        z = depth_matrix[i, j]
                        distance = abs(a * nx + b * ny + c * nz + d)
                        if distance < threshold:
                            inliers.append((i, j))

            # 如果当前平面的内点数量超过之前的最佳结果，则更新
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (a, b, c, d)

        # 输出调试信息
        print(f"本次迭代找到内点数量: {len(best_inliers)}")

        # 如果最佳平面找到的内点数小于阈值，停止
        if len(best_inliers) < min_inliers:
            break

        # 记录当前找到的平面
        planes.append((best_plane, best_inliers))

        # 将内点标记为已处理
        for (i, j) in best_inliers:
            remaining_points[i, j] = False

    return planes

# 通过 RANSAC 找出多个平面
planes = ransac_multiple_planes(normals, depth_matrix)

# 4. 三维可视化 - 完整地形 + 平面标注
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制原始地形
x, y = np.meshgrid(np.arange(depth_matrix.shape[1]), np.arange(depth_matrix.shape[0]))
z = depth_matrix
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.6, rstride=1, cstride=1, edgecolor='none')

# 不同颜色的平面
colors = ['r', 'g', 'b', 'y', 'c', 'm']  # 预定义颜色

# 绘制检测到的平面
for idx, (plane, inliers) in enumerate(planes):
    a, b, c, d = plane
    inlier_points = np.array(inliers)
    
    # 使用内点绘制平面上的点，并为不同平面使用不同的颜色
    inlier_x = inlier_points[:, 1]
    inlier_y = inlier_points[:, 0]
    inlier_z = depth_matrix[inlier_y, inlier_x]

    ax.scatter(inlier_x, inlier_y, inlier_z, color=colors[idx % len(colors)], label=f'平面 {idx + 1}', s=50)

# 设置相同的比例尺
max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0

mid_x = (x.max() + x.min()) * 0.5
mid_y = (y.max() + y.min()) * 0.5
mid_z = (z.max() + z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# 设置标签
ax.set_xlabel('X 轴')
ax.set_ylabel('Y 轴')
ax.set_zlabel('深度 (Z 轴)')
plt.legend()

# 显示图形
plt.show()

