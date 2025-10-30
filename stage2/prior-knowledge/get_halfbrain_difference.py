# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:07:51 2025

@author: admin
"""


import os
import numpy as np
import nibabel as nib
import ants
from scipy.ndimage import median_filter
from concurrent.futures import ProcessPoolExecutor

def process_folder(folder):
    try:
        # 构造原始文件路径
        left_brain_path = os.path.join(folder, 'left_brain.nii.gz')
        right_brain_path = os.path.join(folder, 'right_brain.nii.gz')

        # 检查原始文件是否存在
        if os.path.exists(left_brain_path) and os.path.exists(right_brain_path):
            
            # 1. 镜像右脑图像
            img = nib.load(right_brain_path)
            img_data = img.get_fdata()
            flipped_data = np.flip(img_data, axis=0)  # 平面内，左右 镜像翻转

            # 保存镜像后的右脑图像
            flipped_img = nib.Nifti1Image(flipped_data, img.affine, img.header)
            flipped_image_path = os.path.join(folder, 'right_brain_flipped.nii.gz')
            nib.save(flipped_img, flipped_image_path)

            # 2. 使用 ANTsPy 进行配准
            left_brain_ants = ants.image_read(left_brain_path)
            right_brain_flipped_ants = ants.image_read(flipped_image_path)

            # 使用 SyN 方法进行配准
            registered_right_brain = ants.registration(
                fixed=left_brain_ants,
                moving=right_brain_flipped_ants,
                type_of_transform='SyN'
            )

            # 保存注册后的右脑图像
            registered_right_brain_path = os.path.join(folder, 'registered_right_brain.nii.gz')
            ants.image_write(registered_right_brain['warpedmovout'], registered_right_brain_path)

            # 3. 去噪和差异计算
            left_brain_data = nib.load(left_brain_path).get_fdata()
            registered_right_brain_data = nib.load(registered_right_brain_path).get_fdata()

            # 使用中值滤波进行去噪
            left_brain_data = median_filter(left_brain_data, size=3)  # 可以调整size参数
            registered_right_brain_data = median_filter(registered_right_brain_data, size=3)  # 可以调整size参数

            # 计算两个图像的绝对差值
            difference_data = np.abs(left_brain_data - registered_right_brain_data)

            # 将小于15的位置的绝对差值置为0
            mask = (left_brain_data < 20) | (registered_right_brain_data < 20)
            difference_data[mask] = 0

            # 将差值大于70的位置置为0
            difference_data[difference_data > 70] = 0

            # 保存差值图像
            difference_image = nib.Nifti1Image(difference_data, img.affine, img.header)
            difference_image_path = os.path.join(folder, 'difference_left_registered_right.nii.gz')
            nib.save(difference_image, difference_image_path)

            print(f"处理文件夹 {os.path.basename(folder)}: 所有操作成功。")
        else:
            print(f"在文件夹 {os.path.basename(folder)} 中未找到所需文件。")
    except Exception as e:
        print(f"处理文件夹 {folder} 时出错: {e}")

if __name__ == "__main__":
    # 两个输入目录
    input_folders = [
        '/home/zsun/NCCT_blood_nii/ncct/Radiology2022/data/multicenter_nii_nc',
        '/home/zsun/NCCT_blood_nii/ncct/Radiology2022/data/multicenter_nii_close'
    ]

    # 获取所有子文件夹的路径（包括两个目录中的子文件夹）
    folders = []
    for input_folder in input_folders:
        folders.extend([f.path for f in os.scandir(input_folder) if f.is_dir()])

    # 使用并行处理
    with ProcessPoolExecutor(max_workers=20) as executor:
        executor.map(process_folder, folders)