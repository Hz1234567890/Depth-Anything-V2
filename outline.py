import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

def outline(mask_original):
    # 设置字体为支持中文的字体（例如 SimHei）
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # Noto Sans CJK
    my_font = font_manager.FontProperties(fname=font_path)
    # 假设mask为二值图像，值为0和255
    # 读取或生成mask图像
    # mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
    mask = cv2.cvtColor(mask_original, cv2.COLOR_BGR2GRAY)
    # 找到轮廓（只提取外部轮廓）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 选择面积最大的轮廓（假设该轮廓代表我们的目标mask区域）
    cnt = max(contours, key=cv2.contourArea)

    # 多边形逼近，epsilon设置为轮廓周长的1%
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # 绘制原始轮廓与逼近后的多边形对比
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask_color, [cnt], -1, (0, 255, 0), 2)        # 原始轮廓，绿色
    cv2.drawContours(mask_color, [approx], -1, (0, 0, 255), 2)       # 逼近多边形，红色

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB))
    plt.title("轮廓与逼近多边形",fontproperties=my_font)
    plt.axis("off")
    plt.show()