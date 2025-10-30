# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:02:29 2025

@author: admin
"""


import ants
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def process_folder(folder_path, template_path, prob_path, label_path):
    """
    处理单个文件夹
    """
    # 定义输入和输出文件路径
    print('working:',folder_path)
    ncct_bet_path = os.path.join(folder_path, "NCCT_bet.nii.gz")
    ncct_prob_output = os.path.join(folder_path, "NCCT_Prob.nii.gz")
    left_brain_output = os.path.join(folder_path, "left_brain.nii.gz")
    right_brain_output = os.path.join(folder_path, "right_brain.nii.gz")

    # 检查输出文件是否已存在
    if os.path.exists(left_brain_output) and os.path.exists(right_brain_output):
        print(f"跳过处理文件夹 {folder_path}：输出文件已存在")
        return

    # Step 1: 读取图像数据
    CTn = ants.image_read(ncct_bet_path)  # 目标 NCCT 图像
    MNI = ants.image_read(template_path)  # MNI 模板
    prob_image = ants.image_read(prob_path)  # 概率图
    label_image = ants.image_read(label_path)  # 标签图

    # Step 2: 配准 MNI 到 NCCT 图像
    # print(f"Registering {template_path} to {ncct_bet_path}")
    mytx = ants.registration(fixed=CTn, moving=MNI, type_of_transform="SyN")

    # Step 3: 应用变换到概率图
    print(f"Applying transform to probability image {prob_path}")
    warped_prob_image = ants.apply_transforms(
        fixed=CTn,
        moving=prob_image,
        transformlist=mytx['fwdtransforms']
    )
    # 保存配准后的概率图结果
    ants.image_write(warped_prob_image, ncct_prob_output)

    # Step 4: 应用变换到标签图
    # print(f"Applying transform to label image {label_path}")
    warped_label_image = ants.apply_transforms(
        fixed=CTn,
        moving=label_image,
        transformlist=mytx['fwdtransforms'],
        interpolator="nearestNeighbor"  # 标签需要最近邻插值
    )

    # Step 5: 处理标签图，提取左脑和右脑区域
    warped_label_data = warped_label_image.numpy()  # 转换为 NumPy 数组

    # 提取左脑区域 (label = 1)
    left_brain_data = np.where(warped_label_data == 1, CTn.numpy(), 0)  # 保留强度值
    left_brain_image = ants.from_numpy(left_brain_data, origin=CTn.origin, spacing=CTn.spacing, direction=CTn.direction)
    ants.image_write(left_brain_image, left_brain_output)

    # 提取右脑区域 (label = 2)
    right_brain_data = np.where(warped_label_data == 2, CTn.numpy(), 0)  # 保留强度值
    right_brain_image = ants.from_numpy(right_brain_data, origin=CTn.origin, spacing=CTn.spacing, direction=CTn.direction)
    ants.image_write(right_brain_image, right_brain_output)

    print(f"Finished processing folder: {folder_path}")

def main():
    # 输入路径
    base_dir = "/home/zsun/NCCT_blood_nii/ncct/Radiology2022/data/multicenter_nii_nc"
    template_path = "/home/zsun/NCCT_blood_nii/ncct/Radiology2022/data/MNI152_brain.nii.gz"
    prob_path = "/home/zsun/NCCT_blood_nii/ncct/Radiology2022/data/ProbArterialAtlas_BMM_1_double_prep.nii.gz"
    label_path = "/home/zsun/NCCT_blood_nii/ncct/Radiology2022/data/MNI152_half.nii.gz"

    # 获取所有子文件夹
    folder_paths = [
        os.path.join(base_dir, folder)
        for folder in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, folder))
    ]

    # 并行处理每个文件夹
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(lambda folder: process_folder(folder, template_path, prob_path, label_path), folder_paths)

if __name__ == "__main__":
    main()