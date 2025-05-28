import subprocess
import json
import os
from Variance import *
import pandas as pd
import re

def read_dji_metadata(image_path):
    """
    调用 exiftool 并以 JSON 格式输出所有元数据信息，然后解析返回的 Python dict。
    """
    # -j 表示以 JSON 格式输出
    # -G 表示带上 Group 名称（例如XMP-drone-dji之类的分组）
    cmd = ["exiftool", "-j", "-G", image_path]

    try:
        output = subprocess.check_output(cmd, universal_newlines=True)
        metadata_list = json.loads(output)
        if metadata_list:
            return metadata_list[0]
        else:
            return {}
    except subprocess.CalledProcessError as e:
        print("调用 exiftool 出错:", e)
        return {}
    except json.JSONDecodeError as e:
        print("解析 exiftool 输出的 JSON 出错:", e)
        return {}


def show_metadata_in_terminal(metadata_dict):
    """
    将元数据字典里的信息逐行打印到终端。
    """
    for key, value in metadata_dict.items():
        print(f"{key}: {value}")


def show_emphasized_parameters(metadata_dict):
    result_data=[]
    """
    重点输出指定的六个参数。
    如果在元数据中未找到对应的键，则显示空字符串。
    """
    keys_to_emphasize = [
        "XMP:GimbalRollDegree",
        "XMP:GimbalYawDegree",
        "XMP:GimbalPitchDegree",
        "XMP:FlightRollDegree",
        "XMP:FlightYawDegree",
        "XMP:FlightPitchDegree",
        "EXIF:GPSAltitude"
    ]

    print("\n==== 重点参数 ====")
    for key in keys_to_emphasize:
        # 使用 dict.get() 方法，如果不存在则返回空字符串
        value = metadata_dict.get(key, "")
        print(f"{key}: {value}")
        result_data.append(value)
    return result_data

def get_gimbal_data(image_file):
    dji_metadata = read_dji_metadata(image_file)
    result=show_emphasized_parameters(dji_metadata)
    return result

def quadrilateral_area(points):
    """计算四边形面积（坐标需按顺序排列）"""
    if len(points) != 4:
        raise ValueError("必须是4个点的四边形")
    
    # Shoelace公式
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return abs(
        (x[0]*y[1] + x[1]*y[2] + x[2]*y[3] + x[3]*y[0]) -
        (y[0]*x[1] + y[1]*x[2] + y[2]*x[3] + y[3]*x[0])
    ) / 2

if __name__ == "__main__":
    img_path = "/media/hz/新加卷/0mywork/database/area_b"
    os.makedirs(img_path, exist_ok=True)
    
    Gimbal_Pitch=[]
    height = []
    num=[]

    for filename in [f for f in os.listdir(img_path) if f.endswith('.png')]:
        # 处理图片元数据（原有代码）
        image_path = os.path.join(img_path, filename)
        new_filename = filename.replace("Rotation_", "")#源文件
        new_image_path = os.path.join(img_path,new_filename)
        new_image_path = os.path.splitext(new_image_path)[0] + '.JPG'

        dji_metadata = read_dji_metadata(new_image_path)
        show_emphasized_parameters(dji_metadata)
        # show_metadata_in_terminal(dji_metadata)

        # 处理对应的JSON文件
        json_path = os.path.splitext(image_path)[0] + '.json'
        print(f"json文件：{json_path};源文件{new_image_path}")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                    print(data.keys())
                    area_points = data['shapes'][0]['points']  # 直接获取四边形坐标
                    print(area_points)
                    # area_points = [item['points'] for item in area if 'points' in item]
                    # print(area_points[0][0])

                    # 坐标格式校验
                    if len(area_points) != 4 or any(len(p)!=2 for p in area_points):
                        print(f"坐标格式错误: {filename}")
                        continue
                    
                    # 转换为数值类型
                    points = [[float(p[0]), float(p[1])] for p in area_points]
                    
                    # 计算面积并取整（像素数为整数）
                    pixel_count = int(quadrilateral_area(points))
                    print(f"【{filename}】四边形区域像素数量: {pixel_count}")
                    Gimbal_Pitch.append(dji_metadata.get('XMP:GimbalPitchDegree', ""))
                    h = re.findall(r"-?\d+\.?\d*", dji_metadata.get('EXIF:GPSAltitude', ""))
                    height.append(h[0])
                    # height.append(dji_metadata.get('EXIF:GPSAltitude', ""))
                    num.append(pixel_count)
                except json.JSONDecodeError:
                    print(f"JSON解析失败: {json_path}")
                except KeyError:
                    print(f"缺少area字段: {json_path}")
                except ValueError as e:
                    print(f"数据错误: {e}")
        else:
            print(f"缺失JSON文件: {json_path}")
    
    # 创建DataFrame，按列名匹配数据
    df = pd.DataFrame({
        "俯仰角": Gimbal_Pitch,
        "海拔高度":height,
        "像素数": num
    })

    # 写入Excel文件（默认保存到当前目录）
    df.to_csv("data_output.csv", index=False)

    print("Excel文件已生成：data_output.xlsx")
