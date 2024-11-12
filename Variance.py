import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import cv2
from Myglobal import *


def variance_plane(depth_matrix,window_size,Gif_file_path):
    # 获取图像中心坐标
    center_x, center_y = depth_matrix.shape[1] // 2, depth_matrix.shape[0] // 2

    # 定义一个列表来存储每个窗口的方差及其位置
    variance_list = []

    # 初始化绘图
    fig, ax = plt.subplots()
    ax.imshow(depth_matrix, cmap='gray')
    rect = patches.Rectangle((0, 0), window_size, window_size, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # 遍历滑动窗口并更新矩形位置，计算方差
    frames = []
    for i in range(0, depth_matrix.shape[0] - window_size + 1, step_size):  # 每40步滑动一次
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
    quarter_index = len(variance_list) // 4  # 计算一半的索引
    top_quarter_windows = variance_list[:quarter_index]  # 取出排名前一半的元素

    # 动画更新函数
    def update(frame):
        x, y = frame
        rect.set_xy((x, y))
        return rect,

    # 创建动图
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=50)

    # 保存滑动过程的动图
    ani.save(Gif_file_path, writer=PillowWriter(fps=10))
    plt.close()

    return top_quarter_windows


def distortion_correction():
    # 提取旋转矩阵 R 的第三列，代表相机坐标系的 z 轴在世界坐标系中的方向
    z_axis = R[:, 2]

    # 计算 z 轴与水平面的夹角
    # 水平夹角即为 z 轴向量与垂直方向（世界坐标系 z 轴）的夹角
    # xita = 相机光轴向量在z轴上的分量与z轴的夹角
    angle_with_horizontal = np.arccos(z_axis[2] / np.linalg.norm(z_axis)) * (180 / np.pi)  # 转换为角度
    
    print(f"开始计算夹角为{angle_with_horizontal}度平面的理论方差")
    for row in range(window_size):
        for column in range(window_size):
            variance = 
    print("理论方差为")
    return angle_with_horizontal

if __name__ == '__main__':
    result = distortion_correction()
    result = round(result, 6)
    print(f"夹角为：{result}度")