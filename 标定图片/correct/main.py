import cv2
import numpy as np

# 加载倾斜图像
image = cv2.imread('test3.jpg')
# cv2.imshow("Raw Image", image)
# cv2.waitKey(0)
# 定义相机内参矩阵K (请替换为实际参数)
f_x, f_y = 3052.74045205680, 3054.04988708449  # 焦距
c_x, c_y = 2011.14595030742, 1501.75719918596  # 主点坐标
K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

# 定义相机旋转矩阵R (请替换为实际旋转矩阵)
R = np.array([
    [0.920385726773255, 0.0467763042816092, 0.388203672457928],
    [-0.0471108260842716, 0.998852114251377, -0.00866163501957808],
    [-0.388163218270003, -0.0103165504559019, 0.921532899450131]
])

# 计算透视变换矩阵H
H = K @ R @ np.linalg.inv(K)

# 应用透视变换
height, width = image.shape[:2]
corrected_image = cv2.warpPerspective(image, H, (width*2, height*2))

# 显示结果
# cv2.imshow("Corrected Image", corrected_image)
# cv2.waitKey(0)
cv2.imwrite("corrected_image3.jpg", corrected_image)
cv2.destroyAllWindows()
