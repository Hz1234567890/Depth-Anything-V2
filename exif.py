import piexif
from PIL import Image
import re
import os
from Myglobal import *

# 提取特定字段
def extract_gimbal_and_flight_data(exif_data):
    gimbal_pattern = r"GimbalDegree\(Y,P,R\):(-?\d+),(-?\d+),(-?\d+)"
    flight_pattern = r"FlightDegree\(Y,P,R\):(-?\d+),(-?\d+),(-?\d+)"

    gimbal_match = re.search(gimbal_pattern, exif_data)
    flight_match = re.search(flight_pattern, exif_data)

    gimbal_data = None
    flight_data = None

    if gimbal_match:
        gimbal_data = tuple(float(gimbal_match.group(i)) for i in range(1, 4))
    
    if flight_match:
        flight_data = tuple(float(flight_match.group(i)) for i in range(1, 4))

    return gimbal_data, flight_data

def angular_correction(angle):
    # angle = angle%360
    # angle = (angle + 360)%360
    angle = angle / 10
    return angle
def get_gimbal_data(filename):
    img = Image.open(filename)
    exif_dict = piexif.load(img.info['exif'])

    if 'Exif' in exif_dict:
        exif_data = exif_dict['Exif']
        exif_data_str = str(exif_data)
        gimbal, flight = extract_gimbal_and_flight_data(exif_data_str)
        Y,P,R = gimbal
        Y=angular_correction(Y)
        P=angular_correction(P)
        R=angular_correction(R)
        FY,FP,FR = flight
        FY=angular_correction(FY)
        FP=angular_correction(FP)
        FR=angular_correction(FR)
        print("Gimbal Data(Y,P,R):", Y,P,R)
        print("Flight Data(Y,P,R):", FY,FP,FR)
    else:
        print('Exif数据不存在')
    return Y,P,R
if __name__ == '__main__':
    # Read Image
    files_path = "/media/hz/新加卷/0mywork/Depth-Anything-V2/标定图片/angle1"
    files = os.listdir(files_path)
    filenames = [file for file in files if file.endswith('.JPG')]
    print(filenames)
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        filename = os.path.join(files_path,filename)
        img = Image.open(filename)
        exif_dict = piexif.load(img.info['exif'])

        if 'Exif' in exif_dict:
            exif_data = exif_dict['Exif']
            exif_data_str = str(exif_data)
            gimbal, flight = extract_gimbal_and_flight_data(exif_data_str)
            Y,P,R = gimbal
            Y=angular_correction(Y)
            P=angular_correction(P)
            R=angular_correction(R)
            FY,FP,FR = flight
            FY=angular_correction(FY)
            FP=angular_correction(FP)
            FR=angular_correction(FR)
            print("Gimbal Data(Y,P,R):", Y,P,R)
            print("Flight Data(Y,P,R):", FY,FP,FR)
        else:
            print('Exif数据不存在')
        print("######################")

