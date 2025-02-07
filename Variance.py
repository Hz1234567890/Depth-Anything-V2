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

def compute_pitch_matrix(pitch_angle, roll_angle):
    """
    计算仰角（pitch）和滚转（roll）旋转矩阵
    :param pitch_angle: 仰角（角度）
    :param roll_angle: 滚转角（角度）
    :return: 仰角和滚转角的旋转矩阵 (3x3)
    """
    # 将角度转换为弧度
    pitch_angle = 90 + pitch_angle  # 仰角的定义方式
    roll_angle = roll_angle  # 滚转角

    # 计算仰角（pitch）的旋转矩阵
    phi = np.deg2rad(pitch_angle)  # 仰角转弧度
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])
    
    # 计算滚转（roll）的旋转矩阵
    theta = np.deg2rad(roll_angle)  # 滚转角转弧度
    R_roll = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # 返回仰角和滚转角的组合旋转矩阵（先仰角旋转，再滚转）
    return np.dot(R_roll, R_pitch)


def apply_pitch_transform(image, pitch_angle,roll_angle ,K):
    """
    对图像应用仰角透视变换并居中调整
    :param image: 输入图像
    :param pitch_angle: 仰角（角度）
    :param K: 相机内参矩阵
    :return: 居中调整后的图像
    """
    h, w = image.shape[:2]
    R_pitch = compute_pitch_matrix(pitch_angle,roll_angle)

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

    # 计算变换后图像的尺寸
    new_w = max_x - min_x
    new_h = max_y - min_y
    # 计算平移变换，使图像居中
    translation_matrix = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])

    # 应用透视变换，得到变换后的图像
    transformed_image = cv2.warpPerspective(image, H, (new_w, new_h))
    # transformed_image = cv2.warpPerspective(image, H, (w, h))


    return transformed_image

def inverse_pitch_transform(image, pitch_angle, roll_angle, K):
    """
    使用逆透视变换将经过 apply_pitch_transform 变换后的图像恢复。
    :param image: 已经过透视变换后的图像
    :param pitch_angle: 原先对图像应用的仰角（角度）
    :param roll_angle: 原先对图像应用的滚转角（角度）
    :param K: 相机内参矩阵
    :return: 恢复后的图像
    """
    h, w = image.shape[:2]
    
    # 1. 先获取原先对图像进行透视变换时的单应矩阵 H
    #    也就是 apply_pitch_transform 里的 H = K @ R @ K_inv
    R_pitch = compute_pitch_matrix(pitch_angle, roll_angle)
    
    K_inv = np.linalg.inv(K)
    H = K @ R_pitch @ K_inv
    
    # 2. 计算逆矩阵 H_inv
    H_inv = np.linalg.inv(H)
    
    # 3. 使用 H_inv 对图像进行逆透视变换
    restored_image = cv2.warpPerspective(image, H_inv, (w, h))
    
    return restored_image



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
    """
    # quarter_index = len(variance_list) // 10  # 计算一半的索引
    # top_quarter_windows = variance_list[:quarter_index]  # 取出排名前一半的元素
    """
    top_quarter_windows = variance_list[0]

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