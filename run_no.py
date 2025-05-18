import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
# from MySobel import compute_normal_vectors,visualize_normals 
import torch
from test_picture import threeD_picture

from depth_anything_v2.dpt import DepthAnythingV2
from Variance import *
from Myglobal import *
from MyCanny import Canny
from grow import grow
from evaluate import *

# from exif import *
from rxif import get_gimbal_data
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./mine_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', default=False, dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print("cuda是否可用",DEVICE)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    files = os.listdir(args.img_path)
    filenames = [file for file in files if file.endswith('.JPG')]

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        file_raw_name = filename
        file_only_name = os.path.splitext(os.path.basename(filename))[0]
        json_only_name = file_only_name + ".json"
        print("文件名不含拓展名",file_only_name)
        filename = os.path.join(args.img_path,filename)
        jsonname = os.path.join(args.img_path,json_only_name)
        item_output_folder_path = os.path.join(args.outdir,file_only_name)
        print(f"当前输出文件夹路径为{item_output_folder_path}")
        if not os.path.exists(item_output_folder_path):
            os.makedirs(item_output_folder_path, exist_ok=True)
        
        true_mask_file = f"TMask{file_only_name}" + ".json"
        
        true_mask = read_json(jsonname,filename)
        cv2.imwrite("mask.png", mask)
        angles = get_gimbal_data(filename)
        # "XMP:GimbalRollDegree",
        # "XMP:GimbalYawDegree",
        # "XMP:GimbalPitchDegree",
        roll = float(angles[0])
        yaw = float(angles[1])
        pitch = float(angles[2])

        
        # yaw,pitch,roll = get_gimbal_data(filename)

        raw_image = cv2.imread(filename)
        h, w = raw_image.shape[:2]
        original_size=(h,w)
        print(f"original_size={original_size}")
        # 对RGB影像进行云台透视矫正
        Rotation_filename = f"Rotation_{os.path.splitext(os.path.basename(filename))[0]}" + '.png'
        Rotation_filepath = os.path.join(item_output_folder_path,Rotation_filename)
        
        # pitch only
        rotation_image = apply_pitch_transform(raw_image, pitch,roll, K)
        cv2.imwrite(Rotation_filepath, rotation_image)

        # Boundary_filename = f"Boundary_{os.path.splitext(os.path.basename(filename))[0]}" + '.png'
        # Boundary_filepath = os.path.join(item_output_folder_path,Boundary_filename)
        # print("开始边界检测")
        # Canny(rotation_image,Boundary_filepath)
        

        # # all 
        # R = get_perspective_matrix(yaw,pitch,roll)
        # raw_image = apply_rotation(raw_image, R, K)
        # cv2.imwrite(Rotation_filepath, raw_image)


        depth = depth_anything.infer_image(rotation_image, args.input_size)
        rows, columns = depth.shape

        # 根据图像大小初始化window_size和step_size
        window_size = find_max_x(rows,columns)
        step_size = int(window_size/2)
        # print(f"window_size={window_size}   step_size={step_size}")

        # threeD_filename = f"threeD_{os.path.splitext(os.path.basename(filename))[0]}" + '.png'
        # threeD_filepath = os.path.join(item_output_folder_path,threeD_filename)
        # print(f"{file_raw_name}的三维深度图保存于{threeD_filepath}")
        # # input("按下任意键继续...")
        # threeD_picture(rows, columns, depth,threeD_filepath)

        Gif_filename = f"Gif_{os.path.splitext(os.path.basename(filename))[0]}" + '.gif'
        Gif_filepath = os.path.join(item_output_folder_path,Gif_filename)
        result_filename = f"Result_{os.path.splitext(os.path.basename(filename))[0]}" + '.png'
        result_filepath = os.path.join(item_output_folder_path,result_filename)
        # print("现在开始计算depth的方差")
        
        
        top_tenth_windows = variance_plane(depth,window_size,step_size,Gif_filepath)
        print(top_tenth_windows)

        grow_filename = f"Grow_{os.path.splitext(os.path.basename(filename))[0]}" + '.png'
        grow_filepath = os.path.join(item_output_folder_path,grow_filename)
        grow(rotation_image,depth,top_tenth_windows,window_size,grow_filepath)
        # print("原始的深度矩阵",depth)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        # print("归一化之后的深度矩阵",depth)
        depth = depth.astype(np.uint8)


        result_path = os.path.join(item_output_folder_path, os.path.splitext(os.path.basename(filename))[0] + '.png')
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            depth_image = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            depth_image = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        
        if args.pred_only:
            for var, y, x in top_tenth_windows:
                cv2.rectangle(depth_image, (x, y), (x + window_size, y + window_size), (0, 255, 0), 2)
            # depth_image=inverse_pitch_transform(depth_image, pitch,roll, K)
            cv2.imwrite(result_path, depth_image)
            
        else:
            var, y, x = top_tenth_windows
            cv2.rectangle(depth_image, (x, y), (x + window_size, y + window_size), (0, 255, 0), 2)
            cv2.rectangle(rotation_image, (x, y), (x + window_size, y + window_size), (0, 255, 0), 2)
            # for var, y, x in top_tenth_windows:
            #     cv2.rectangle(depth_image, (x, y), (x + window_size, y + window_size), (0, 255, 0), 2)
            #     cv2.rectangle(rotation_image, (x, y), (x + window_size, y + window_size), (0, 255, 0), 2)
            # depth_image=inverse_pitch_transform(depth_image, pitch,roll, K)
            # rotation_image=inverse_pitch_transform(rotation_image, pitch,roll, K)
            split_region = np.ones((rotation_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([rotation_image, split_region, depth_image])
            
            cv2.imwrite(result_path, combined_result)
            
        print("结果已经保存到",result_path)