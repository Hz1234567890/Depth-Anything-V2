import numpy as np
import cv2
from collections import deque
from outline import outline

def dynamic_threshold(d, j, alpha=0.009, tau=3.5, lambda_=1, kappa=20, H=480, W=640):
    """
    计算动态阈值 T(d, j) ，用于控制平面生长范围
    """
    if j <= (H * W) / (kappa**2):
        return (tau * (1 - np.exp(-j / lambda_)))**2
    else:
        return alpha * d**2 * (tau * (1 - np.exp(-j / lambda_)))**2

def region_growing(white_mask, depth, step, max_iters, old_seeds):
    """
    使用第一次生长生成的二值 white_mask（0/1）作为种子平面进行二次生长，
    使用广度优先搜索（BFS）扩展区域。
    """
    H, W = depth.shape
    seeds = np.argwhere(white_mask == 1)
    # print(seeds)
    
    if seeds.size == 0:
        return np.zeros((H, W), dtype=np.uint8), old_seeds

    total = np.sum(white_mask)  # 种子像素总数
    seed_mean = np.sum(depth * white_mask) / total

    # 将 old_seeds 和 seeds 中的点加入已处理集，避免重复添加到队列
    processed = set(tuple(seed) for seed in old_seeds)

    # 初始化队列，只加入不在 processed 中的点
    queue = deque()
    count=0
    # for seed in seeds:
    #     if tuple(seed) not in processed:
    #         queue.append(tuple(seed))
    #     # processed.add(tuple(seed))
    #     count+=1
    #     if(count%10000==0):
    #         print(f"{count}/{seeds.size}")
    # 计算外接矩形范围
    min_y, min_x = np.min(seeds, axis=0)
    max_y, max_x = np.max(seeds, axis=0)

    # 生成x方向的网格交点坐标
    x_coords = np.arange(min_x, max_x + step, step)
    x_coords = x_coords[x_coords <= max_x]  # 过滤超出外接矩形的点

    # 生成y方向的网格交点坐标
    y_coords = np.arange(min_y, max_y + step, step)
    y_coords = y_coords[y_coords <= max_y]

    # 生成所有网格交点的坐标组合
    grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')
    grid_points = np.column_stack((grid_y.ravel(), grid_x.ravel()))

    # 将网格交点加入队列
    queue.extend(grid_points.tolist())
    print("queue的大小 =", len(queue))


        
    # 初始化二次生长的掩码，新区域中种子部分标记为1
    new_mask = np.zeros((H, W), dtype=np.uint8)
    new_mask[white_mask == 1] = 1

    iters = 0
    while queue and iters < max_iters:
        i, j = queue.popleft()
        # if (i,j) in processed:
        #     # print("continue!!!")
        #     continue
        processed.add((i,j))
        iters += 1
        if iters % 10000 == 0:
            print(f"ing:{iters}/{max_iters}")

        # 计算当前像素局部区域的平均深度
        row_start = max(0, i - step//2)
        row_end = min(H, i + step//2)
        col_start = max(0, j - step//2)
        col_end = min(W, j + step//2)
        d = np.nanmean(depth[row_start:row_end, col_start:col_end])

        # 计算动态阈值
        T_dj = dynamic_threshold(d, iters)

        # 遍历当前像素的8邻域（步长为1）
        for di, dj in [(-step, 0), (step, 0), (0, -step), (0, step),
                       (-step, -step), (-step, step), (step, -step), (step, step)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and (ni, nj) not in processed:
                queue.append((ni, nj))
                processed.add((ni, nj))
                # 计算邻域局部区域的平均深度
                row_start_ngb = max(0, ni - step)
                row_end_ngb = min(H, ni + step)
                col_start_ngb = max(0, nj - step)
                col_end_ngb = min(W, nj + step)

                new_window =  depth[row_start_ngb:row_end_ngb, col_start_ngb:col_end_ngb]
                d_new = np.nanmean(new_window)


                new_variancen = np.var(new_window)

                diff = abs(d_new - seed_mean)
                if np.isnan(diff):
                    diff = 0.0

                # 若深度差小于动态阈值，则扩展该像素到区域中
                if diff < T_dj and new_variancen<0.001:
                # if diff < T_dj:
                    new_mask[row_start_ngb:row_end_ngb, col_start_ngb:col_end_ngb] = 1
                    
                    queue.append((ni, nj))
                    processed.add((ni, nj))

                    

    return new_mask, seeds

def process_mask(plane_mask, kernel_size):
    # 生成椭圆核（与原文相同）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    # Step 1: 先腐蚀（消除小物体）
    eroded = cv2.erode(plane_mask, kernel)

    # Step 3: 找最大连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded)
    
    # 排除背景（假设背景是0）
    if num_labels <= 1:
        return np.zeros_like(plane_mask)  # 没有找到连通区域
    
    # 找到面积最大的区域（跳过背景0）
    max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    max_mask = (labels == max_label).astype(np.uint8) * 255

    final_mask = cv2.morphologyEx(max_mask, cv2.MORPH_OPEN, kernel)
    
    
    
    return final_mask

def grow(rotation_image, depth_matrix, top_window, window_size, path):
    """
    外层调用函数，根据给定的种子窗口进行生长，并可视化。
    """
    var, y, x = top_window
    seed_x = x
    seed_y = y
    seed_w = window_size
    seed_h = window_size

    h, w = depth_matrix.shape

    plane_mask = np.zeros((h, w), dtype=np.uint8)  # 结果掩码
    plane_mask[seed_y:seed_y + seed_h, seed_x:seed_x + seed_w] = 1  # 标记种子区域

    step = max(1, int(window_size / 5))

    epcho = 0
    max_iters = 8000
    max_epcho = 10
    seeds = np.empty((0, 2), dtype=int)
    old_count=0
    # while step > 1:
    while step >= 1 and epcho<max_epcho:
        epcho += 1
        print(f"开始{epcho}次生长:step={step};max_iters={max_iters}")
        plane_mask, seeds = region_growing(plane_mask, depth_matrix, step, max_iters, seeds)
        count_ones = np.sum(plane_mask)
        print(f"window_size={window_size}  count_ones={count_ones}")
        print("******************************************************")
        if count_ones==old_count:
            break
        old_count=count_ones
        step = int(step // 2)
        max_iters = int(max_iters * 2)

    # # nuclear = max(1, int(window_size / 80))
    # # 假设 plane_mask 是二值矩阵（0/1）
    # kernel_size = (100, 100)  # 根据凸起大小调整核尺寸
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    # # 开运算：先腐蚀后膨胀
    # plane_mask = cv2.morphologyEx(plane_mask, cv2.MORPH_OPEN, kernel)
    kernel_size = (window_size//2, window_size//2)
    plane_mask = process_mask (plane_mask,kernel_size)


    overlay = rotation_image.copy()
    white_mask = np.zeros_like(rotation_image, dtype=np.uint8)
    white_mask[plane_mask > 0] = [255, 255, 255]

    alpha = 0.5  # 透明度参数
    blended = cv2.addWeighted(overlay, 1 - alpha, white_mask, alpha, 0)
    
    cv2.rectangle(blended, 
                  (seed_x, seed_y),
                  (seed_x + window_size, seed_y + window_size), 
                  (0, 0, 255), 2)  # 红色边框

    border_thickness = 3
    cv2.rectangle(blended, 
                  (border_thickness, border_thickness), 
                  (w - border_thickness, h - border_thickness), 
                  (255, 255, 255), border_thickness)

    cv2.imwrite(path, blended)

    return plane_mask