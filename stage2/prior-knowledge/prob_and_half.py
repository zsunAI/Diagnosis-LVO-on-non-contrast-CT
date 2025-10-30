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
    Processing individual folders
    """
    # Define input and output file paths
    print('working:',folder_path)
    ncct_bet_path = os.path.join(folder_path, "NCCT_bet.nii.gz")
    ncct_prob_output = os.path.join(folder_path, "NCCT_Prob.nii.gz")
    left_brain_output = os.path.join(folder_path, "left_brain.nii.gz")
    right_brain_output = os.path.join(folder_path, "right_brain.nii.gz")

    # Check whether the output file already exists
    if os.path.exists(left_brain_output) and os.path.exists(right_brain_output):
        print(f"Skip processing folder {folder_path}: output file already exists")
        return

    # Step 1: Read image data
    CTn = ants.image_read(ncct_bet_path)  # Target NCCT image
    MNI = ants.image_read(template_path)  # MNI template
    prob_image = ants.image_read(prob_path)  # Probability map
    label_image = ants.image_read(label_path)  # Label map

    # Step 2: Registration of MNI to NCCT images
    # print(f"Registering {template_path} to {ncct_bet_path}")
    mytx = ants.registration(fixed=CTn, moving=MNI, type_of_transform="SyN")

    # Step 3: Applying transformation matrix to probability map
    print(f"Applying transform to probability image {prob_path}")
    warped_prob_image = ants.apply_transforms(
        fixed=CTn,
        moving=prob_image,
        transformlist=mytx['fwdtransforms']
    )
    # Save probability map results
    ants.image_write(warped_prob_image, ncct_prob_output)

    # Step 4: Apply transform to label images
    # print(f"Applying transform to label image {label_path}")
    warped_label_image = ants.apply_transforms(
        fixed=CTn,
        moving=label_image,
        transformlist=mytx['fwdtransforms'],
        interpolator="nearestNeighbor"  # Labels need nearest neighbor interpolation
    )

    # Step 5: Processing label maps, extracting left and right brain regions
    warped_label_data = warped_label_image.numpy() 

    # Extract the left brain region (label=1)
    left_brain_data = np.where(warped_label_data == 1, CTn.numpy(), 0)
    left_brain_image = ants.from_numpy(left_brain_data, origin=CTn.origin, spacing=CTn.spacing, direction=CTn.direction)
    ants.image_write(left_brain_image, left_brain_output)

    # Extract the right brain region (label=1)
    right_brain_data = np.where(warped_label_data == 2, CTn.numpy(), 0)
    right_brain_image = ants.from_numpy(right_brain_data, origin=CTn.origin, spacing=CTn.spacing, direction=CTn.direction)
    ants.image_write(right_brain_image, right_brain_output)

    print(f"Finished processing folder: {folder_path}")

def main():
    # 
    base_dir = "...YOUDIR/data/multicenter_nii_nc"
    template_path = "...YOUDIR/data/MNI152_brain.nii.gz"
    prob_path = "...YOUDIR/data/ProbArterialAtlas_BMM_1_double_prep.nii.gz"
    label_path = "...YOUDIR/data/MNI152_half.nii.gz"

    # All subfolders
    folder_paths = [
        os.path.join(base_dir, folder)
        for folder in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, folder))
    ]

    # Process each folder in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(lambda folder: process_folder(folder, template_path, prob_path, label_path), folder_paths)

if __name__ == "__main__":
    main()