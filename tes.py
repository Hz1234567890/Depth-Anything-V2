import json
import numpy as np
import cv2

def Test(jsonname, rotation_image, result_mask, compare_filepath):
    # 统一创建所有需要的掩码
    safe_mask, unsafe_mask, combined_mask = create_masks(jsonname, rotation_image)
    
    # 复用掩码进行计算
    accurate = compute_accurate(result_mask, safe_mask)
    correct = compute_correct(result_mask, unsafe_mask)
    performance = compute_performance(result_mask, safe_mask, unsafe_mask, correct)
    
    # 可视化并保存结果
    visualize_results(rotation_image, result_mask, combined_mask, compare_filepath)
    
    print(f"accurate:{accurate:.4f}")
    print(f"correct:{correct}")
    print(f"performance:{performance:.4f}")

def create_masks(json_file, image):
    """创建安全区域、危险区域和组合掩码"""
    safe_points, unsafe_points = parse_json(json_file)
    
    # 创建基础掩码
    safe_mask = np.zeros_like(image, dtype=np.uint8)
    unsafe_mask = np.zeros_like(image, dtype=np.uint8)
    
    # 填充安全区域（绿色）
    if safe_points:
        cv2.fillPoly(safe_mask, [np.array(safe_points, dtype=np.int32)], (0, 255, 0))
    
    # 填充危险区域（红色）
    for points in unsafe_points:
        cv2.fillPoly(unsafe_mask, [np.array(points, dtype=np.int32)], (0, 0, 255))
    
    # 创建组合掩码（安全区域+危险区域）
    combined_mask = cv2.addWeighted(safe_mask, 1.0, unsafe_mask, 1.0, 0)
    
    return safe_mask, unsafe_mask, combined_mask

def parse_json(json_file):
    """解析JSON文件获取安全点和危险点"""
    with open(json_file, "r") as f:
        data = json.load(f)
    
    safe_points = []
    unsafe_points = []
    
    for shape in data["shapes"]:
        if shape["label"] == "unsafe-area":
            unsafe_points.append(shape["points"])
        elif shape["label"] == "safe-area":
            safe_points = shape["points"]
    
    return safe_points, unsafe_points

def binarize_mask(mask):
    """将掩码转换为二值形式"""
    if mask.ndim == 3:  # 彩色图像
        # 分别处理绿色安全区和红色危险区
        green_mask = np.all(mask == [0, 255, 0], axis=-1)
        red_mask = np.all(mask == [0, 0, 255], axis=-1)
        return (green_mask | red_mask).astype(np.uint8)
    else:  # 灰度图像
        return (mask >= 128).astype(np.uint8)

def compute_accurate(result_mask, ground_truth_mask):
    """计算准确率: 检测区域在标注范围内的比例"""
    result_bin = binarize_mask(result_mask)
    gt_bin = binarize_mask(ground_truth_mask)
    
    intersection = np.logical_and(result_bin, gt_bin)
    union = gt_bin
    
    if np.sum(union) == 0:
        return 0.0
    
    accurate = np.sum(intersection) / np.sum(union)
    return accurate

def compute_correct(result_mask, unsafe_mask):
    """检查结果是否包含危险区域"""
    result_bin = binarize_mask(result_mask)
    unsafe_bin = binarize_mask(unsafe_mask)
    
    # 如果存在交集则不安全
    if np.any(np.logical_and(result_bin, unsafe_bin)):
        return 0
    return 1

def compute_performance(result_mask, safe_mask, unsafe_mask, is_correct):
    """计算性能指标"""
    result_bin = binarize_mask(result_mask)
    safe_bin = binarize_mask(safe_mask)
    
    tp = np.logical_and(result_bin, safe_bin)
    tp_count = np.sum(tp)
    tp_fn_count = np.sum(safe_bin)
    
    # 如果不正确，添加危险区域惩罚
    penalty = np.sum(binarize_mask(unsafe_mask)) if not is_correct else 0
    
    denominator = tp_fn_count + penalty
    if denominator == 0:
        return 0.0
    
    return tp_count / denominator

def visualize_results(image, result_mask, combined_mask, output_path):
    """可视化并保存结果"""
    # 创建结果可视化
    result_vis = np.zeros_like(image, dtype=np.uint8)
    result_vis[result_mask > 0] = [0, 255, 0]  # 绿色表示结果
    
    # 叠加原图
    blended = cv2.addWeighted(combined_mask, 0.7, result_vis, 0.3, 0)
    
    # 保存结果
    cv2.imwrite(output_path, blended)