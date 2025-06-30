import numpy as np
import math
# 定义相机内参矩阵K (请替换为实际参数)
"""
540.984175572699	0	2257.27594822434
0	544.656255239778	1498.86282767362
0	0	1
"""
K =  np.array( [[2.84363021e+03 ,0.00000000e+00 ,1.98151896e+03],
                [0.00000000e+00 ,2.84945543e+03 ,1.51398156e+03],
                [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]])# 提供的内参矩阵

# # 像素距离p(m)
# p = 1.7e-6

# # 相机焦距f(m)
# f = 0.026

# # 俯仰角
# pitch_angle = 24

def find_max_x(depth, width):
    # 定义不等式的左边表达式
    def inequality(x):
        return (2 * depth / int(x) - 1) * (2 * width / int(x) - 1)
    
    # 遍历 x 的值，从 1 开始逐步增大，直到不等式不成立
    x = 1
    while True:
        if inequality(x) <= 200:
            # 当不等式不成立时，返回上一个有效的 x 值
            return int(x - 1)
        x += 1
def find_window_size(H, theta_deg, target_width=100, target_height=100):
    """
    预测指定区域对应的像素数
    
    参数：
        H : 海拔高度
        theta_deg : 俯仰角（度）
        target_width : 目标区域宽度（cm），默认200cm
        target_height : 目标区域高度（cm），默认200cm
        
    返回：
        目标区域像素数
    """
    # 原始像素代表的实际区域尺寸（固定值）
    pixel_width = 255  # cm
    pixel_height = 388  # cm
    
    # 计算转换系数
    target_area = target_width * target_height
    pixel_area = pixel_width * pixel_height
    conversion_factor = target_area / pixel_area
    
    # 转换角度为弧度
    theta_rad = np.abs(theta_deg) * np.pi / 180
    
    # 计算基础像素数
    sin_term = np.sin(theta_rad)
    base_pixels = (4.217e7 / H) * (sin_term ** (-1.463)) - 145.3 * theta_deg
    
    # 转换到目标区域
    return int(math.sqrt(conversion_factor * base_pixels))
    
# #预选框大小(px)
# window_size = 300

# #滑动步长(px)
# step_size = 150

"""
    Y (Yaw) - 偏航角
    P (Pitch) - 俯仰角
    R (Roll) - 横滚角
"""
class Gimbal_data:
    def __init__(self,Y,P,R):
        self.Yaw = Y
        self.Pitch = P
        self.Roll = R