import json
import numpy as np
import cv2


def Test(jsonname,rotation_image,result_mask,compare_filepath):
    Pare(jsonname,rotation_image,result_mask,compare_filepath)
    accurate=Accurate(jsonname,rotation_image,result_mask)
    correct=Correct(jsonname,rotation_image,result_mask)
    performance=Performance(jsonname,rotation_image,result_mask,correct)
    print(f"accurate:{accurate:.4f}")
    print(f"correct:{correct}")
    print(f"performance:{performance:.4f}")

def compute_accurate(mask1,mask2):
    # 处理第一个掩码（灰度或彩色）
    if mask1.ndim == 3:  # 彩色图像 (3通道)
        # 注意：OpenCV 使用 BGR 格式，绿色对应 [0, 255, 0]
        mask1_binary = np.all(mask1 == [0, 255, 0], axis=-1).astype(np.uint8)
    else:  # 灰度图像 (1通道)
        mask1_binary = (mask1 >= 128).astype(np.uint8)  # 阈值设为128
    
    # 处理第二个掩码（同样支持灰度和彩色）
    if mask2.ndim == 3:
        mask2_binary = np.all(mask2 == [0, 255, 0], axis=-1).astype(np.uint8)
    else:
        mask2_binary = (mask2 >= 128).astype(np.uint8)
    
    # 计算交集
    intersection = np.logical_and(mask1_binary, mask2_binary)
    intersection_count = np.sum(intersection)
    
    # 计算标注安全区
    all_count = np.sum(mask2_binary)
    
    # 避免除零错误
    if all_count == 0:
        return 0.0
    else:
        accurate = intersection_count / all_count
        # print(f"iou:{accurate:.4f}")
        return accurate

def compute_correct(mask1,mask2):
    # 处理第一个掩码（灰度或彩色）
    if mask1.ndim == 3:  # 彩色图像 (3通道)
        # 注意：OpenCV 使用 BGR 格式，绿色对应 [0, 255, 0]
        mask1_binary = np.all(mask1 == [0, 0, 255], axis=-1).astype(np.uint8)
    else:  # 灰度图像 (1通道)
        mask1_binary = (mask1 >= 128).astype(np.uint8)  # 阈值设为128
    
    # 处理第二个掩码（同样支持灰度和彩色）
    if mask2.ndim == 3:
        mask2_binary = np.all(mask2 > [0, 0, 0], axis=-1).astype(np.uint8)
    else:
        mask2_binary = (mask2 >= 128).astype(np.uint8)
    
    # 计算交集
    # print(np.sum(mask1_binary))
    intersection = np.logical_and(mask1_binary, mask2_binary)
    intersection_count = np.sum(intersection)
    # print("intersection_count:",intersection_count)
    if intersection_count == 0:#交集为空说明安全
        return 1
    else:
        return 0

def compute_performance(result_mask,right_mask,unsafe_mask,ifCorrect):
    # 处理第一个掩码（灰度或彩色）
    if result_mask.ndim == 3:  # 彩色图像 (3通道)
        # 注意：OpenCV 使用 BGR 格式，绿色对应 [0, 255, 0]
        result_mask_binary = np.all(result_mask == [0, 255, 0], axis=-1).astype(np.uint8)
    else:  # 灰度图像 (1通道)
        result_mask_binary = (result_mask >= 128).astype(np.uint8)  # 阈值设为128

    # 处理第一个掩码（灰度或彩色）
    if unsafe_mask.ndim == 3:  # 彩色图像 (3通道)
        # 注意：OpenCV 使用 BGR 格式，绿色对应 [0, 255, 0]
        unsafe_mask_binary = np.all(unsafe_mask == [0, 0, 255], axis=-1).astype(np.uint8)
    else:  # 灰度图像 (1通道)
        unsafe_mask_binary = (unsafe_mask >= 128).astype(np.uint8)  # 阈值设为128
    
    # 处理第二个掩码（同样支持灰度和彩色）
    if right_mask.ndim == 3:
        right_mask_binary = np.all(right_mask == [0, 255, 0], axis=-1).astype(np.uint8)
    else:
        right_mask_binary = (right_mask >= 128).astype(np.uint8)
    tp = np.logical_and(result_mask_binary, right_mask_binary)
    tp_count = np.sum(tp)
    tp_fn_count = np.sum(right_mask_binary)

    

    if ifCorrect == 0:
        unsafe_count=np.sum(unsafe_mask_binary)
    else:
        unsafe_count=0
    # print(f"tp_count/(tp_fn_count+unsafe_count={tp_count}/({tp_fn_count}+{unsafe_count})")
    Q = tp_count/(tp_fn_count+unsafe_count)

    return Q

def Accurate(json_file,image,result_mask):
    """
    精确性 检测区域在标注范围内的像素数/标注范围内的像素数
    """
    # image = cv2.imread(image_file)
    safe_points,unsafe_points=test_json(json_file,image)
    safe_mask = np.zeros_like(image, dtype=np.uint8)
    safe_polygon = np.array(safe_points, dtype=np.int32)

    # 填充安全区域（白色）
    # safe_polygon = np.array(safe_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(safe_mask, [safe_polygon], (0, 255, 0))
    # print(safe_mask)
    accurate=compute_accurate(result_mask,safe_mask)
    return accurate

def Correct(json_file,image,result_mask):
    """
    正确性 检测区域中是否有unsafe区域，若有则不安全，返回0;若无则安全，返回1
    """
    safe_points,unsafe_points=test_json(json_file,image)
    print(unsafe_points)
    unsafe_mask = np.zeros_like(image, dtype=np.uint8)

    print("非安全区域个数：",len(unsafe_points))
    # print(unsafe_mask)

    if(len(unsafe_points)==0):
        #如果没有不安全区域，则记为正确，返回1
        return 1
      

    for i in range(len(unsafe_points)):
        unsafe_polygon = np.array(unsafe_points[i], dtype=np.int32)
        # unsafe_polygon = np.array(unsafe_points[i], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(unsafe_mask, [unsafe_polygon], (0, 0, 255))
        if(compute_correct(unsafe_mask,result_mask)==0):
            return 0

    return 1

def Performance(json_file,image,result_mask,ifCorrect):
    safe_points,unsafe_points=test_json(json_file,image)
    safe_mask = np.zeros_like(image, dtype=np.uint8)
    unsafe_mask = np.zeros_like(image, dtype=np.uint8)
    safe_polygon = np.array(safe_points, dtype=np.int32)

    # 填充安全区域（白色）
    # safe_polygon = np.array(safe_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(safe_mask, [safe_polygon], (0, 255, 0))    

    combined_mask = safe_mask.copy()
    for i in range(len(unsafe_points)):
        unsafe_polygon = np.array(unsafe_points[i], dtype=np.int32)
        cv2.fillPoly(unsafe_mask, [unsafe_polygon], (0, 0, 255))
        # 合并两个掩码（危险区域会覆盖安全区域）
        combined_mask = cv2.addWeighted(combined_mask, 1.0, unsafe_mask, 1.0, 0)
        # print(f"{i}/{len(unsafe_points)}unsafe_polygon:",unsafe_polygon)


    performance = compute_performance(result_mask,safe_mask,combined_mask,ifCorrect)
    return performance

def Pare(json_file,image,result_mask,path):
    safe_points,unsafe_points=test_json(json_file,image)

    safe_mask = np.zeros_like(image, dtype=np.uint8)
    unsafe_mask = np.zeros_like(image, dtype=np.uint8)

    # 填充安全区域（绿色）
    safe_polygon = np.array(safe_points, dtype=np.int32)
    cv2.fillPoly(safe_mask, [safe_polygon], (255, 255, 0))
    # print("safe_polygon:",safe_polygon)

    combined_mask = safe_mask.copy()
    for i in range(len(unsafe_points)):
        unsafe_polygon = np.array(unsafe_points[i], dtype=np.int32)
        cv2.fillPoly(unsafe_mask, [unsafe_polygon], (0, 0, 255))
        # 合并两个掩码（危险区域会覆盖安全区域）
        combined_mask = cv2.addWeighted(combined_mask, 1.0, unsafe_mask, 1.0, 0)
        # print(f"{i}/{len(unsafe_points)}unsafe_polygon:",unsafe_polygon)

    # 保存合并后的掩码
    cv2.imwrite(path, combined_mask)
    overlay = image.copy()
    mask_image = cv2.imread(path)
    white_mask = np.zeros_like(image, dtype=np.uint8)
    white_mask[result_mask > 0] = [0, 255, 0]
    alpha = 0.8  # 透明度参数
    
    blended = cv2.addWeighted(white_mask, alpha, mask_image, 1-alpha, 0)
    
    result_blended = cv2.addWeighted(blended, 0.5,overlay, 0.5, 0)

    # border_thickness = 3
    cv2.imwrite(path, result_blended)

def test_json(json_file,image):
    # read json file
    with open(json_file, "r") as f:
        data = f.read()
    
    # convert str to json objs
    data = json.loads(data)
    
    safe_points = []
    unsafe_points = []

    for i in range(len(data["shapes"])):

        
        if(data["shapes"][i]["label"]=="unsafe-area"):
            # # print(data["shapes"][i]["points"])
            # print("______________________________________________")
            # print((data["shapes"][i]["points"])[0])
            # print("______________________________________________")
            # print((data["shapes"][i]["points"]))
            # print("______________________________________________")
            # print("end")
            unsafe_points.append((data["shapes"][i]["points"]))
        elif(data["shapes"][i]["label"]=="safe-area"):
            safe_points.append(data["shapes"][i]["points"])

    # print("safe:",safe_points)
    # print("unsafe:",unsafe_points)
    
    # read image to get shape
    # image = cv2.imread(image_file)
    # safe_mask = np.zeros_like(image, dtype=np.uint8)
    # unsafe_mask = np.zeros_like(image, dtype=np.uint8)
    # safe_points = np.array(safe_points, dtype=np.int32)

    # # 填充安全区域（白色）
    # safe_polygon = np.array(safe_points, dtype=np.int32).reshape((-1, 1, 2))
    # cv2.fillPoly(safe_mask, [safe_polygon], (0, 255, 0))
    # # print("safe_polygon:",safe_polygon)

    # combined_mask = safe_mask.copy()
    # for i in range(len(unsafe_points)):
    #     unsafe_polygon = np.array(unsafe_points[i], dtype=np.int32).reshape((-1, 1, 2))
    #     cv2.fillPoly(unsafe_mask, [unsafe_polygon], (0, 0, 255))
    #     # 合并两个掩码（危险区域会覆盖安全区域）
    #     combined_mask = cv2.addWeighted(combined_mask, 1.0, unsafe_mask, 1.0, 0)
    #     print(f"{i}/{len(unsafe_points)}unsafe_polygon:",unsafe_polygon)

    # # 保存合并后的掩码
    # cv2.imwrite("mask.png", combined_mask)
    return safe_points,unsafe_points

if __name__ == '__main__':
    json_file="/media/hz/新加卷/0mywork/mine/area_1/DJI_20250514142903_0045_V.json"
    image_file="/media/hz/新加卷/0mywork/mine/area_1/DJI_20250514142903_0045_V.JPG"
    test_json(json_file,image_file)