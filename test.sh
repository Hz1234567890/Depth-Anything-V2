#!/bin/bash

# 运行Python脚本
# python run_no.py --encoder vits --img-path /media/hz/新加卷/0mywork/database/平面检测数据集all --outdir /media/hz/新加卷/0mywork/database/平面检测数据集all_out_6 --result-path results_12000_8_no.csv.csv
python run_no.py --encoder vits --img-path /media/hz/新加卷/0mywork/database/lshi --outdir /media/hz/新加卷/0mywork/database/lshi --result-path results.csv
# python run_no.py --encoder vits --img-path /media/hz/新加卷/0mywork/mine/test3 --outdir /media/hz/新加卷/0mywork/mine/test3_out --result-path results.csv
# python run_no.py --encoder vits --img-path /media/hz/新加卷/0mywork/mine/area_1 --outdir /media/hz/新加卷/0mywork/mine/area_1_out

# 检查是否成功运行
if [ $? -eq 0 ]; then
    echo "Python脚本运行成功！"
else
    echo "Python脚本运行失败。"
fi
