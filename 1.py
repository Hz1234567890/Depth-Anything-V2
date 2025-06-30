import os

# 设置要操作的文件夹路径（替换为实际路径）
folder_path = "/media/hz/新加卷/0mywork/database/平面检测数据集非90"  # 示例：r"C:\MyFiles"

def rename_rotation_files():
    # 获取文件夹所有JSON文件
    json_files = [f for f in os.listdir(folder_path) 
                 if f.endswith('.json') and os.path.isfile(os.path.join(folder_path, f))]
    
    renamed_count = 0
    
    for filename in json_files:
        # 检查是否以"Rotation_"开头
        if filename.startswith("Rotation_"):
            # 分割前缀和剩余部分
            parts = filename.split("_", 1)
            if len(parts) == 2 and parts[1]:
                new_name = parts[1]
                # 构造完整文件路径
                src = os.path.join(folder_path, filename)
                dst = os.path.join(folder_path, new_name)
                
                # 执行重命名
                try:
                    os.rename(src, dst)
                    print(f"Renamed: {filename} -> {new_name}")
                    renamed_count += 1
                except Exception as e:
                    print(f"Error renaming {filename}: {str(e)}")
    
    print(f"\nProcess complete. {renamed_count} files renamed.")

# 执行重命名操作
rename_rotation_files()