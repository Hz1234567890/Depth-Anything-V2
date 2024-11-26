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
from Variance import variance_plane,depth_to_ortho_with_rotation,apply_pitch_transform
from Myglobal import *
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./mine_depth')
    
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
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
    filenames = [file for file in files if file.endswith('.jpeg')]

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        file_raw_name = filename
        file_only_name = os.path.splitext(os.path.basename(filename))[0]
        print("文件名不含拓展名",file_only_name)
        filename = os.path.join(args.img_path,filename)
        item_output_folder_path = os.path.join(args.outdir,file_only_name)
        print(f"当前输出文件夹路径为{item_output_folder_path}")
        if not os.path.exists(item_output_folder_path):
            os.makedirs(item_output_folder_path, exist_ok=True)
        
        raw_image = cv2.imread(filename)
        
        # # 对RGB影像进行云台透视矫正
        # Rotation_filename = f"Rotation_{os.path.splitext(os.path.basename(filename))[0]}" + '.png'
        # Rotation_filepath = os.path.join(item_output_folder_path,Rotation_filename)
        # raw_image = apply_pitch_transform(raw_image, pitch_angle, K)
        # cv2.imwrite(Rotation_filepath, raw_image)

        depth = depth_anything.infer_image(raw_image, args.input_size)
        rows, columns = depth.shape

        # #对深度信息进行正则矫正
        # depth = depth_to_ortho_with_rotation(depth, K, R, T, rows, columns)

        threeD_filename = f"threeD_{os.path.splitext(os.path.basename(filename))[0]}" + '.png'
        threeD_filepath = os.path.join(item_output_folder_path,threeD_filename)
        print(f"{file_raw_name}的三维深度图保存于{threeD_filepath}")
        # input("按下任意键继续...")
        threeD_picture(rows, columns, depth,threeD_filepath)

        Gif_filename = f"Gif_{os.path.splitext(os.path.basename(filename))[0]}" + '.gif'
        Gif_filepath = os.path.join(item_output_folder_path,Gif_filename)
        result_filename = f"Result_{os.path.splitext(os.path.basename(filename))[0]}" + '.png'
        result_filepath = os.path.join(item_output_folder_path,result_filename)
        print("现在开始计算depth的方差")
        
        
        top_quarter_windows = variance_plane(depth,window_size,Gif_filepath)
        
        print("原始的深度矩阵",depth)

        # print("开始Sobel算子的计算")
        
        # normals_filename = f"annotated_{os.path.splitext(os.path.basename(filename))[0]}" + '.png'
        # plane_filename = f"plane_{os.path.splitext(os.path.basename(filename))[0]}"
        # normals_path = os.path.join(item_output_folder_path, normals_filename)
        # plane_path = os.path.join(item_output_folder_path, plane_filename)
        # normal_x, normal_y, normal_z ,angle_with_horizontal= compute_normal_vectors(depth)
        # top_quarter_windows = variance_plane(angle_with_horizontal,window_size,Gif_filepath)
        # # #################################################################################################
        # # vectors_results_filepath = os.path.join(item_output_folder_path,"normal_vectors_results.txt")
        # # with open(vectors_results_filepath, 'w') as f:
        # #     # 写入 normal_x 矩阵
        # #     f.write("normal_x:\n")
        # #     np.savetxt(f, normal_x, fmt='%.2f')
            
        # #     # 写入 normal_y 矩阵
        # #     f.write("\nnormal_y:\n")
        # #     np.savetxt(f, normal_y, fmt='%.2f')
            
        # #     # 写入 normal_z 矩阵
        # #     f.write("\nnormal_z:\n")
        # #     np.savetxt(f, normal_z, fmt='%.2f')
            
        # #     # 写入 angle_with_horizontal 矩阵
        # #     f.write("\nangle_with_horizontal:\n")
        # #     np.savetxt(f, angle_with_horizontal, fmt='%.2f')
        # # print("结果已写入 normal_vectors_results.txt")
        # # ####################################################################################################
        # visualize_normals(normal_x, normal_y, normal_z,normals_path,plane_path)
        
        # print("结束Sobel算子的计算")

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        # # depth = depth.astype(np.uint8)
        
        print("归一化之后的深度矩阵",depth)
        depth = depth.astype(np.uint8)


        result_path = os.path.join(item_output_folder_path, os.path.splitext(os.path.basename(filename))[0] + '.png')
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            depth_image = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            depth_image = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        
        if args.pred_only:
            for var, y, x in top_quarter_windows:
                cv2.rectangle(depth_image, (x, y), (x + window_size, y + window_size), (0, 255, 0), 2)
            cv2.imwrite(result_path, depth_image)
            
        else:
            for var, y, x in top_quarter_windows:
                cv2.rectangle(depth_image, (x, y), (x + window_size, y + window_size), (0, 255, 0), 2)
                cv2.rectangle(raw_image, (x, y), (x + window_size, y + window_size), (0, 255, 0), 2)
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth_image])
            
            cv2.imwrite(result_path, combined_result)
            
        print("结果已经保存到",result_path)