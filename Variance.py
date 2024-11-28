import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import cv2
from Myglobal import *

import cv2
import numpy as np

def compute_pitch_matrix(pitch_angle):
    """
    计算仰角旋转矩阵
    :param pitch_angle: 仰角（角度）
    :return: 旋转矩阵 (3x3)
    """
    pitch_angle = 90 + pitch_angle
    phi = np.deg2rad(pitch_angle)  # 角度转弧度
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])
    return R_pitch

def apply_pitch_transform(image, pitch_angle, K):
    """
    对图像应用仰角透视变换并居中调整
    :param image: 输入图像
    :param pitch_angle: 仰角（角度）
    :param K: 相机内参矩阵
    :return: 居中调整后的图像
    """
    h, w = image.shape[:2]
    R_pitch = compute_pitch_matrix(pitch_angle)

    # 计算透视变换矩阵 H
    K_inv = np.linalg.inv(K)
    H = K @ R_pitch @ K_inv

    # 计算变换后的图像四个角点位置
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]])
    transformed_corners = H @ corners.T
    transformed_corners /= transformed_corners[2, :]  # 归一化
    transformed_corners = transformed_corners[:2, :].T

    # 计算有效区域的包围盒
    min_x = max(0, int(np.min(transformed_corners[:, 0])))
    max_x = min(w, int(np.max(transformed_corners[:, 0])))
    min_y = max(0, int(np.min(transformed_corners[:, 1])))
    max_y = min(h, int(np.max(transformed_corners[:, 1])))

    # # 应用透视变换
    # transformed_image = cv2.warpPerspective(image, H, (w, h))

    # # 裁剪有效区域
    # cropped = transformed_image[min_y:max_y, min_x:max_x]

    # # 计算变换后图像的尺寸
    # new_w = max_x - min_x
    # new_h = max_y - min_y

    # 计算变换后图像的尺寸
    new_w = max_x - min_x
    new_h = max_y - min_y

    # 应用透视变换，得到变换后的图像
    transformed_image = cv2.warpPerspective(image, H, (new_w, new_h))

    # # 将裁剪图像居中到原始尺寸
    # centered_image = np.zeros_like(image)  # 创建空白图像
    # start_y = (h - (max_y - min_y)) // 2
    # start_x = (w - (max_x - min_x)) // 2
    # centered_image[start_y:start_y + cropped.shape[0], start_x:start_x + cropped.shape[1]] = cropped
    # centered_image = cv2.resize(centered_image, (w, h), interpolation=cv2.INTER_LINEAR)
    return transformed_image

def get_perspective_matrix(yaw, pitch, roll):
    # 角度转弧度
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)

    # 偏航矩阵
    Ryaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw), 0],
        [0, 1, 0, 0],
        [-np.sin(yaw), 0, np.cos(yaw), 0],
        [0, 0, 0, 1]
    ])

    # 俯仰矩阵
    Rpitch = np.array([
        [1, 0, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch), 0],
        [0, np.sin(pitch), np.cos(pitch), 0],
        [0, 0, 0, 1]
    ])

    # 滚动矩阵
    Rroll = np.array([
        [np.cos(roll), -np.sin(roll), 0, 0],
        [np.sin(roll), np.cos(roll), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # 综合矩阵
    R = Rroll @ Rpitch @ Ryaw

    # 取 3x3 的部分
    return R[:3, :3]

def apply_rotation(image, R, K):
    """
    使用相机旋转矩阵对图像进行透视变换
    :param image: 输入图像
    :param R: 相机外参旋转矩阵 (3x3)
    :param K: 相机内参矩阵 (3x3)
    :return: 透视变换后的图像
    """
    h, w = image.shape[:2]

    # 计算 K * R * K^-1
    K_inv = np.linalg.inv(K)
    H = K @ R @ K_inv

    # 应用透视变换
    transformed_image = cv2.warpPerspective(image, H, (w, h))
    return transformed_image



# # 读取图像并应用变换
# image = cv2.imread("example.jpg")
# transformed_image = apply_rotation(image, R, K)

# # 显示结果
# cv2.imshow("Original Image", image)
# cv2.imshow("Transformed Image", transformed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def variance_plane(depth_matrix,window_size,step_size,Gif_file_path):
    # 获取图像中心坐标
    center_x, center_y = depth_matrix.shape[1] // 2, depth_matrix.shape[0] // 2

    # 定义一个列表来存储每个窗口的方差及其位置
    variance_list = []

    # # 初始化绘图
    # fig, ax = plt.subplots()
    # ax.imshow(depth_matrix, cmap='gray')
    # rect = patches.Rectangle((0, 0), window_size, window_size, linewidth=2, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)

    # 遍历滑动窗口并更新矩形位置，计算方差
    frames = []
    for i in range(0, depth_matrix.shape[0] - window_size + 1, step_size):  # 每step_size步滑动一次
        for j in range(0, depth_matrix.shape[1] - window_size + 1, step_size):
            # 提取当前窗口
            window = depth_matrix[i:i + window_size, j:j + window_size]
            # 计算方差
            variance = np.var(window)
            # 将方差和窗口位置存储起来
            variance_list.append((variance, i, j))
            frames.append((j, i))  # 将每帧的窗口位置存储

    # 按方差排序，选择最小的前15个
    variance_list.sort(key=lambda x: x[0])
    quarter_index = len(variance_list) // 10  # 计算一半的索引
    top_quarter_windows = variance_list[:quarter_index]  # 取出排名前一半的元素

    # # 动画更新函数
    # def update(frame):
    #     x, y = frame
    #     rect.set_xy((x, y))
    #     return rect,

    # # 创建动图
    # ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=50)

    # # 保存滑动过程的动图
    # ani.save(Gif_file_path, writer=PillowWriter(fps=10))
    # plt.close()

    return top_quarter_windows