import numpy as np
import cv2
from collections import deque
from outline import outline

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
"""
def region_growing(depth, x, y, w, h, step, max_iters=10000):
    
    在 2D 深度矩阵上进行平面生长（不转换 3D），使用广度优先搜索（BFS）。
    :param depth: 深度矩阵 (H, W)
    :param x, y, w, h: 种子区域 (左上角坐标 + 宽高)
    :param step: 每次在水平/竖直方向以及对角方向跳跃的步长
    :param max_iters: 最大迭代次数，防止死循环
    :return: 平面区域的二值掩码 (H, W)
    
    H, W = depth.shape
    mask = np.zeros((H, W), dtype=np.uint8)  # 结果掩码

    # 种子区域的信息
    seed_region = depth[y:y+h, x:x+w]
    seed_mean = np.mean(seed_region)  # 种子区域深度均值

    # 初始化队列，使用 deque 来实现 BFS
    queue = deque()
    # 将种子区域内的点加入队列
    for i in range(y, y+h, step):
        for j in range(x, x+w, step):
            queue.append((i, j))

    processed = set(queue)     # 记录已进入队列的点，防止重复
    mask[y:y+h, x:x+w] = 1     # 标记种子区域

    iters = 0
    while queue and iters < max_iters:
        i, j = queue.popleft()  # BFS：从队列左端弹出
        iters += 1

        # 计算当前像素附近的平均深度（避免像素中有少量噪声时直接用 depth[i,j]）
        row_start = max(0, i - step)
        row_end   = min(H, i + step)
        col_start = max(0, j - step)
        col_end   = min(W, j + step)
        d = np.nanmean(depth[row_start:row_end, col_start:col_end])

        # 计算动态阈值
        T_dj = dynamic_threshold(d, iters)

        # 遍历 8 邻域像素（同样以 step 为步长）
        for di, dj in [(-step, 0), (step, 0), (0, -step), (0, step),
                       (-step, -step), (-step, step), (step, -step), (step, step)]:
            ni, nj = i + di, j + dj
            # 判断是否在图像范围内
            if 0 <= ni < H and 0 <= nj < W and (ni, nj) not in processed:
                # 计算邻域平均深度
                row_start_ngb = max(0, ni - step)
                row_end_ngb   = min(H, ni + step)
                col_start_ngb = max(0, nj - step)
                col_end_ngb   = min(W, nj + step)
                d_new = np.nanmean(depth[row_start_ngb:row_end_ngb, col_start_ngb:col_end_ngb])

                diff = abs(d_new - seed_mean)
                if np.isnan(diff):
                    diff = 0.0  # 若出现 NaN，视为 0
                # print(f"diff={diff} T_dj={T_dj}")
                # 判断是否满足生长条件
                if diff < T_dj:
                    mask[row_start_ngb:row_end_ngb, col_start_ngb:col_end_ngb] = 1
                    # print(f"add:{np.sum(mask)}")
                    queue.append((ni, nj))     # BFS：新的像素加到队列右端
                    processed.add((ni, nj))    # 标记已处理

    return mask

"""

def region_growing(white_mask, depth,step, max_iters,old_seeds):
    """
    使用第一次生长生成的二值 white_mask（0/1）作为种子平面进行二次生长，
    二次生长中步长固定为1，采用广度优先搜索（BFS）扩展区域。

    :param white_mask: 第一次生长生成的二值掩码（0/1），其中1代表种子区域
    :param depth: 深度矩阵 (H, W)
    :param max_iters: 最大迭代次数，防止死循环
    :return: 二次生长后的区域二值掩码 (H, W)
    """
    H, W = depth.shape

    # 提取种子区域：直接选择 white_mask 中值为1的像素
    seeds = np.argwhere(white_mask == 1)
    if seeds.size == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # 使用加权平均计算种子区域的深度均值，注意 white_mask 为非矩形区域
    total = np.sum(white_mask)  # 种子像素总数
    seed_mean = np.sum(depth * white_mask) / total

    from collections import deque
    queue = deque()
    processed = set()
    # 将所有种子点加入队列
    # print("1111111")
    for i, j in seeds:
        # print("append")
        queue.append((i, j))
        processed.add((i, j))
    # print("2222222")
    # 初始化二次生长的掩码，新区域中种子部分标记为1
    new_mask = np.zeros((H, W), dtype=np.uint8)
    new_mask[white_mask == 1] = 1

    iters = 0

    while queue and iters < max_iters:
        i, j = queue.popleft()
        iters += 1
        if iters % 1000==0:
            print(f"ing:{iters}/{max_iters}")
        # 计算当前像素局部区域（以当前像素为中心，窗口大小为 (2*step+1)）的平均深度
        row_start = max(0, i - step)
        row_end   = min(H, i + step)
        col_start = max(0, j - step)
        col_end   = min(W, j + step)
        d = np.nanmean(depth[row_start:row_end, col_start:col_end])

        # 根据当前深度和迭代次数计算动态阈值
        T_dj = dynamic_threshold(d, iters)

        # 遍历当前像素的8邻域（步长为1）
        for di, dj in [(-step, 0), (step, 0), (0, -step), (0, step),
                       (-step, -step), (-step, step), (step, -step), (step, step)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and (ni, nj) not in processed:
                # 计算邻域局部区域平均深度
                row_start_ngb = max(0, ni - step)
                row_end_ngb   = min(H, ni + step)
                col_start_ngb = max(0, nj - step)
                col_end_ngb   = min(W, nj + step)
                d_new = np.nanmean(depth[row_start_ngb:row_end_ngb, col_start_ngb:col_end_ngb])

                diff = abs(d_new - seed_mean)
                if np.isnan(diff):
                    diff = 0.0

                # print(f"diff={diff},T_dj={T_dj}")
                # 若深度差小于动态阈值，则扩展该像素到区域中
                if diff < T_dj:
                    
                    new_mask[row_start_ngb:row_end_ngb, col_start_ngb:col_end_ngb] = 1
                    # print(f"add:{np.sum(new_mask)}")
                    # new_mask[ni, nj] = 1
                    queue.append((ni, nj))
                    processed.add((ni, nj))

    return new_mask,seeds

def grow(rotation_image, depth_matrix, top_window, window_size, path):
    
    """
    外层调用函数，根据给定的种子窗口进行生长，并可视化。
    :param rotation_image: 原始旋转后的图像 (H, W, 3)
    :param depth_matrix: 深度矩阵 (H, W)
    :param top_window: (var, y, x)，其中 y, x 为窗口左上角
    :param window_size: 种子窗口大小
    :param path: 结果可视化后保存的路径
    """
    var, y, x = top_window
    seed_x = x
    seed_y = y
    seed_w = window_size
    seed_h = window_size

    h, w = depth_matrix.shape

    plane_mask = np.zeros((h, w), dtype=np.uint8)  # 结果掩码
    plane_mask[seed_y:seed_y+seed_h, seed_x:seed_x+seed_w] = 1     # 标记种子区域

    # 根据 window_size 设定步长
    step = max(1, int(window_size / 25))

    # 运行平面生长算法 (BFS)x, y, w, h
    epcho=0
    max_iters=100000
    seeds = np.empty((0, 2), dtype=int)

    while step>0:
        epcho=epcho+1
        print(f"开始{epcho}次生长:step={step};max_iters={max_iters}")
        plane_mask,seeds = region_growing(plane_mask, depth_matrix,step,max_iters,seeds)
        count_ones = np.sum(plane_mask)
        print(f"window_size={window_size}  count_ones={count_ones}")

        print("******************************************************")
        step=step//2
        max_iters=max_iters*2

    # 复制原始图像，避免修改原数据
    overlay = rotation_image.copy()

    # 创建一个 3 通道的白色掩码（即 mask 处为白色）
    white_mask = np.zeros_like(rotation_image, dtype=np.uint8)
    white_mask[plane_mask > 0] = [255, 255, 255]  # 将 mask 区域设置为白色

    # #test outline.py
    # outline(white_mask)

    # 叠加 plane_mask 到原始图像上（50% 透明度）
    alpha = 0.5  # 透明度参数
    blended = cv2.addWeighted(overlay, 1 - alpha, white_mask, alpha, 0)

    # 绘制种子区域红色框
    cv2.rectangle(blended, 
                  (seed_x, seed_y),
                  (seed_x + window_size, seed_y + window_size), 
                  (0, 0, 255), 2)  # 红色边框

    # 添加整体边框（3像素宽，白色）
    border_thickness = 3
    cv2.rectangle(blended, 
                  (border_thickness, border_thickness), 
                  (w - border_thickness, h - border_thickness), 
                  (255, 255, 255), border_thickness)
    
    

    # 保存结果
    cv2.imwrite(path, blended)

