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
        # Original file path
        left_brain_path = os.path.join(folder, 'left_brain.nii.gz')
        right_brain_path = os.path.join(folder, 'right_brain.nii.gz')

        # Check whether the original file exists
        if os.path.exists(left_brain_path) and os.path.exists(right_brain_path):
            
            # 1. Mirror right brain image
            img = nib.load(right_brain_path)
            img_data = img.get_fdata()
            flipped_data = np.flip(img_data, axis=0)  # left and right mirror image flipping

            # Save the mirrored right brain image
            flipped_img = nib.Nifti1Image(flipped_data, img.affine, img.header)
            flipped_image_path = os.path.join(folder, 'right_brain_flipped.nii.gz')
            nib.save(flipped_img, flipped_image_path)

            # 2. Registration using ants
            left_brain_ants = ants.image_read(left_brain_path)
            right_brain_flipped_ants = ants.image_read(flipped_image_path)

            # SyN method
            registered_right_brain = ants.registration(
                fixed=left_brain_ants,
                moving=right_brain_flipped_ants,
                type_of_transform='SyN'
            )

            # Save the registered right brain image
            registered_right_brain_path = os.path.join(folder, 'registered_right_brain.nii.gz')
            ants.image_write(registered_right_brain['warpedmovout'], registered_right_brain_path)

            # 3. Denoising and difference calculation
            left_brain_data = nib.load(left_brain_path).get_fdata()
            registered_right_brain_data = nib.load(registered_right_brain_path).get_fdata()

            # Denoising using median filtering
            left_brain_data = median_filter(left_brain_data, size=3)
            registered_right_brain_data = median_filter(registered_right_brain_data, size=3)

            # Calculate the absolute difference between two images
            difference_data = np.abs(left_brain_data - registered_right_brain_data)

            # Remove the influence of sulcus/cerebrospinal fluid: 
            # the position less than 15 Hu on either side should be set to 0, which cannot be used for calculation.
            mask = (left_brain_data < 15) | (registered_right_brain_data < 15)
            difference_data[mask] = 0

            # If there are extreme conditions such as vascular calcification, it will also lead to a great difference. 
            # It is also necessary to set a threshold
            difference_data[difference_data > 70] = 0

            # save
            difference_image = nib.Nifti1Image(difference_data, img.affine, img.header)
            difference_image_path = os.path.join(folder, 'difference_left_registered_right.nii.gz')
            nib.save(difference_image, difference_image_path)

            print(f"Processing folder {os.path.basename (folder)}: all operations succeeded.")
        else:
            print(f"The required file was not found in the folder {os.path.basename (folder)}.")
    except Exception as e:
        print(f"Error processing folder {folder}: {e}")

if __name__ == "__main__":

    input_folders = [
        '...YOUDIR/data/train_nii_nc',
        '...YOUDIR/data/train_nii_close' # Can also be merged into one folder for processing
    ]

    # Get the path of all subfolders (including subfolders in two directories)
    folders = []
    for input_folder in input_folders:
        folders.extend([f.path for f in os.scandir(input_folder) if f.is_dir()])

    # 
    with ProcessPoolExecutor(max_workers=20) as executor:
        executor.map(process_folder, folders)