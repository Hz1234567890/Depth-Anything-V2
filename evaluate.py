import json
import numpy as np
import cv2

def read_json(json_file,jpg_file):
    # read json file
    with open(json_file, "r") as f:
        data = f.read()
    
    # convert str to json objs
    data = json.loads(data)
    
    # get the points 
    points = data["shapes"][0]["points"]
    points = np.array(points, dtype=np.int32)   # tips: points location must be int32
    
    # read image to get shape
    image = cv2.imread(jpg_file)
    
    # create a blank image
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # fill the contour with 255
    cv2.fillPoly(mask, [points], (255, 255, 255))
    
    # # save the mask 
    # cv2.imwrite("mask.png", mask)
    return mask

import numpy as np

def binary_iou(mask1, mask2, epsilon=1e-6):
    """
    计算二值掩膜的前景类IoU
    :param mask1: 预测的二值掩膜（0或1）
    :param mask2: 真实的二值掩膜（0或1）
    :param epsilon: 平滑因子，防止分母为0
    :return: 前景类的IoU值
    """
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return intersection / (union + epsilon)


def pixel_accuracy(mask_pred, mask_true, epsilon=1e-6):
    """
    计算二值掩膜的像素精度（PA）
    :param mask_pred: 预测的二值掩膜（0或1）
    :param mask_true: 真实的二值掩膜（0或1）
    :param epsilon: 平滑因子，防止分母为0
    :return: PA值（范围[0,1]）
    """
    correct_pixels = np.sum(mask_pred == mask_true)
    total_pixels = mask_pred.size
    return correct_pixels / (total_pixels + epsilon)


# json_file="/media/hz/新加卷/0mywork/mine/test3/DJI_20250401164312_0027_V.json"
# jpg_file="/media/hz/新加卷/0mywork/mine/test3/DJI_20250401164312_0027_V.JPG"
# mask=read_json(json_file,jpg_file)
# cv2.imwrite("mask.png", mask)