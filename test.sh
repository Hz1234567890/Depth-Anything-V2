#!/bin/bash

# 运行Python脚本
python run_no.py --encoder vits --img-path /media/hz/新加卷/0mywork/mine/标定图片/angle1 --outdir /media/hz/新加卷/0mywork/mine/标定图片/angle1_out


# 检查是否成功运行
if [ $? -eq 0 ]; then
    echo "Python脚本运行成功！"
else
    echo "Python脚本运行失败。"
fi
