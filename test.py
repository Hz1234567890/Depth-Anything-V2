import json
import numpy as np
import cv2

def Accurate():
    """
    精确性
    """

    return 0

def Correct():

    return 0

def Performance():
    return 0

def test_json(json_file,image_file):
    # read json file
    with open(json_file, "r") as f:
        data = f.read()
    
    # convert str to json objs
    data = json.loads(data)
    
    safe_points = []
    unsafe_points = []

    for i in range(len(data["shapes"])):

        
        if(data["shapes"][i]["label"]=="unsafe-area"):
            # # print(data["shapes"][i]["points"])
            # print("______________________________________________")
            # print((data["shapes"][i]["points"])[0])
            # print("______________________________________________")
            # print((data["shapes"][i]["points"]))
            # print("______________________________________________")
            # print("end")
            unsafe_points.append((data["shapes"][i]["points"]))
        elif(data["shapes"][i]["label"]=="safe-area"):
            safe_points.append(data["shapes"][i]["points"])

    print("safe:",safe_points)
    print("safe_point:",unsafe_points)
    safe_points = np.array(safe_points, dtype=np.int32)
    unsafe_points = np.array(unsafe_points, dtype=np.int32)

    # get the points 
    points = data["shapes"][0]["points"]
    print("points:",points)
    points = np.array(points, dtype=np.int32)   # tips: points location must be int32

    # read image to get shape
    image = cv2.imread(image_file)
    
    # create a blank image
    mask = np.zeros_like(image, dtype=np.uint8)

    safe_mask = mask.copy()
    unsafe_mask = mask.copy()
    # fill the contour with 255
    cv2.fillPoly(safe_mask, safe_points, (255, 255, 255))

    cv2.fillPoly(unsafe_mask, unsafe_points, (255, 0, 0))
    
    # # save the mask 
    cv2.imwrite("mask.png", safe_mask)
    return mask

if __name__ == '__main__':
    json_file="/media/hz/新加卷/0mywork/mine/area_1/DJI_20250514142903_0045_V.json"
    image_file="/media/hz/新加卷/0mywork/mine/area_1/DJI_20250514142903_0045_V.JPG"
    test_json(json_file,image_file)