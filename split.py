import os
import random
from shutil import copy2
from tqdm import tqdm  # 用于显示进度条


def split_data():
    # 源文件路径
    source_path = '/tmp/pycharm_project_36/UCMerced_LandUse/Images'
    # 新文件路径
    dest_path = '/tmp/pycharm_project_36/UCMdatasets82'
    # 划分比例 (train, val, test)
    split_ratio = (0.8, 0.2, 0)  # 修改为8:2:0的比例
    split_names = ['train', 'val', 'test']

    print(f"Starting dataset splitting with ratio {split_ratio[0]}:{split_ratio[1]}:{split_ratio[2]}")

    # 获取所有类别
    class_names = sorted(os.listdir(source_path))
    if not class_names:
        raise ValueError("No classes found in the source directory!")

    # 创建目标目录结构
    os.makedirs(dest_path, exist_ok=True)
    for split_name in split_names:
        split_dir = os.path.join(dest_path, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for class_name in class_names:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

    # 遍历每个类别进行划分
    for class_name in tqdm(class_names, desc="Processing classes"):
        class_path = os.path.join(source_path, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))]
        random.shuffle(images)

        total_images = len(images)
        if total_images == 0:
            print(f"Warning: No images found in class {class_name}")
            continue

        # 计算划分点
        train_end = int(split_ratio[0] * total_images)
        val_end = train_end + int(split_ratio[1] * total_images)

        # 复制文件到对应目录
        for i, img in enumerate(images):
            src = os.path.join(class_path, img)
            if i < train_end:
                dest = os.path.join(dest_path, 'train', class_name, img)
            elif i < val_end:
                dest = os.path.join(dest_path, 'val', class_name, img)
            else:
                dest = os.path.join(dest_path, 'test', class_name, img)
            copy2(src, dest)

        print(
            f"Class {class_name}: {total_images} images | Train: {train_end} | Val: {val_end - train_end} | Test: {total_images - val_end}")

    print("Dataset splitting completed successfully!")


if __name__ == '__main__':
    split_data()
