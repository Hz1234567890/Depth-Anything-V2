import subprocess
import json


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
        "XMP:FlightPitchDegree"
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


if __name__ == "__main__":

    """
        这个函数没用于完成像素大小-实际大小的映射函数回归
    """
    files = "/media/hz/新加卷/0mywork/area_a"

    filenames = [file for file in files if file.endswith('.JPG')]
    for file in filenames:
        dji_metadata = read_dji_metadata(image_file)

        # # 输出全部元数据
        # show_metadata_in_terminal(dji_metadata)

        # 重点输出六个参数
        result=show_emphasized_parameters(dji_metadata)
        # print(result)

