import numpy as np

# 定义相机内参矩阵K (请替换为实际参数)
"""
540.984175572699	0	2257.27594822434
0	544.656255239778	1498.86282767362
0	0	1
"""
K =  np.array([[2787.92123871456, 0, 1319.23947109118],
               [0, 2731.92041136203, 1305.90610350384],
               [0, 0, 1]])# 提供的内参矩阵

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