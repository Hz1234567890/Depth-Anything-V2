import numpy as np
import cv2
from scipy.ndimage import binary_dilation

def dynamic_threshold(d, j, alpha=0.009, tau=3, lambda_=1, kappa=20, H=480, W=640):
    """
    计算动态阈值 T(d, j) ，用于控制平面生长范围
    :param d: 当前像素深度值
    :param j: 当前生长迭代次数
    :param alpha, tau, lambda_, kappa: 控制阈值变化的参数
    :param H, W: 深度矩阵尺寸
    :return: 计算得到的动态阈值
    """
    if j <= (H * W) / (kappa**2):
        return (tau * (1 - np.exp(-j / lambda_)))**2
    else:
        return alpha * d**2 * (tau * (1 - np.exp(-j / lambda_)))**2

def region_growing(depth, x, y, w, h, max_iters=5000):
    """
    在 2D 深度矩阵上进行平面生长（不转换 3D）
    :param depth: 深度矩阵 (H, W)
    :param x, y, w, h: 种子区域 (左上角坐标 + 宽高)
    :param max_iters: 最大迭代次数，防止死循环
    :return: 平面区域的二值掩码 (H, W)
    """
    H, W = depth.shape
    mask = np.zeros((H, W), dtype=np.uint8)  # 结果掩码
    seed_region = depth[y:y+h, x:x+w]
    seed_mean = np.mean(seed_region)  # 种子区域深度均值
    seed_var = np.var(seed_region)    # 种子区域深度方差

    # 初始化生长队列（从种子区域开始）
    queue = [(i, j) for i in range(y, y+h) for j in range(x, x+w)]
    processed = set(queue)  # 记录已处理的点
    mask[y:y+h, x:x+w] = 1  # 标记种子区域

    iters = 0
    while queue and iters < max_iters:
        i, j = queue.pop(0)
        iters += 1
        d = depth[i, j]

        # 计算动态阈值 T(d, j)
        T_dj = dynamic_threshold(d, iters)

        # 遍历 8 邻域像素
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and (ni, nj) not in processed:
                d_new = depth[ni, nj]
                if abs(d_new - seed_mean) < T_dj:
                    mask[ni, nj] = 1
                    queue.append((ni, nj))
                    processed.add((ni, nj))

    return mask

# 示例：生成一个模拟深度矩阵 (480, 640)，假设深度范围在 800mm~1200mm
np.random.seed(50)
depth_matrix = np.random.randint(900, 1000, (480, 640))

# 设定种子区域 (100, 150) 作为起点，大小为 (10, 10)
seed_x, seed_y, seed_w, seed_h = 100, 150, 10, 10

# 运行平面生长算法
plane_mask = region_growing(depth_matrix, seed_x, seed_y, seed_w, seed_h)

# 显示结果
cv2.imshow("Detected Plane", plane_mask * 255)  # 让白色代表检测到的平面
cv2.waitKey(0)
cv2.destroyAllWindows()
