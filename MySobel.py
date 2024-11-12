import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter

# 假设depth_matrix是深度矩阵
def compute_normal_vectors(depth_matrix):
    # 使用Sobel算子计算x和y方向的梯度
    sobel_x = cv2.Sobel(depth_matrix, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_matrix, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度的模长
    grad_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    
    # 计算法向量，假设z方向的梯度为1
    normal_x = -sobel_x / (grad_magnitude + 1e-8)  # 避免除零
    normal_y = -sobel_y / (grad_magnitude + 1e-8)
    normal_z = np.ones_like(depth_matrix)
    
    # 归一化法向量
    normal_length = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
    normal_x /= normal_length
    normal_y /= normal_length
    normal_z /= normal_length

    # 计算与z轴的夹角（即与水平面夹角的余角）
    theta = np.arccos(normal_z)
    
    # 计算与水平面的夹角
    angle_with_horizontal = np.degrees(theta)
    
    return normal_x, normal_y, normal_z,angle_with_horizontal

# 可视化法向量场
def visualize_normals(normal_x, normal_y, normal_z,filename,plane_filename):
    plt.figure(figsize=(10, 10))
    
    # 计算平面平整度（例如，可以通过法向量的变化率评估）
    normal_variance = np.sqrt((np.gradient(normal_x)[0]**2) + (np.gradient(normal_y)[0]**2) + (np.gradient(normal_z)[0]**2))
    print("平面平整度:",normal_variance)
    # 显示深度梯度和法向量
    # plt.subplot(1, 2, 1)
    plt.plot()
    plt.title('Normal vectors variance (smoothness)')
    plt.imshow(normal_variance, cmap='viridis')
    plt.colorbar()
    
    # # 使用箭头可视化法向量
    # plt.subplot(1, 2, 2)
    # plt.title('Normal Vectors')
    # x, y = np.meshgrid(np.arange(normal_x.shape[1]), np.arange(normal_x.shape[0]))
    # plt.quiver(x, y, normal_x, normal_y, normal_z, scale=50, color='blue')
    
    # 保存图像
    plt.savefig(filename)  # 保存为文件 normal_vectors_visualization.png
    # plt.show()

    # 使用 3x3 邻域检查每个点与邻域的平整度是否相似
    plane_mask = generic_filter(normal_variance, plane_similarity, size=(5, 5))

    # 可视化同一平面的点
    plt.figure(figsize=(8, 6))
    plt.imshow(plane_mask, cmap='gray')  # 使用灰度图显示平面区域
    plt.title('Detected Planes Based on Smoothness Difference')
    plt.savefig(plane_filename)
    # plt.show()
"""
下面这部分代码，通过设定阈值的方式判定平面
"""
# 定义平整度差异阈值
threshold = 0.05  # 根据实际情况调整阈值

# 函数用于计算当前点与邻域的平整度差异
def plane_similarity(values):
    center_value = values[len(values) // 2]  # 中心点的平整度
    diff = np.abs(values - center_value)  # 计算与邻域内其他点的平整度差
    return np.all(diff < threshold)  # 如果所有邻域内点与中心点的平整度差在阈值内，返回True

    

# # 假设depth_matrix是我们给定的深度矩阵
# depth_matrix = np.random.rand(100, 100) * 100  # 示例深度矩阵
# normal_x, normal_y, normal_z = compute_normal_vectors(depth_matrix)
# visualize_normals(normal_x, normal_y, normal_z)
