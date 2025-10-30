import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
def image_normalization(image, contrast):
    """ Normalize the image based on the type of contrast. """
    if image.size == 0:
        print("Warning: Input image is empty. Returning a zero array.")
        return np.zeros((1, 1))  
    if contrast == 'NCCT':
        # Set cutoff value for brain tissue of NCCT 
        image[image < 0] = 0
        image[image > 70] = 0
        non_zero_mask = image > 0
        if non_zero_mask.sum() == 0:
            print("Warning: No non-zero pixels found in the image. Returning unmodified image.")
            return image
        min_val = np.min(image[non_zero_mask])
        max_val = np.max(image[non_zero_mask])
        if min_val != max_val:
            image[non_zero_mask] = (image[non_zero_mask] - min_val) / (max_val - min_val)
    elif contrast == 'blood':
        if np.all(image == 0):
            print("Warning: Input image for 'blood' is empty or all zeros. Returning unmodified image.")
            return image
        image[image > 0] = 1  # Previously, blood vessel segmentation was multi labeled. Now, labels with values greater than 0 have been changed to 1
    
    elif contrast == 'NCCT_Prob':
        image = image
    elif contrast == 'sub':
        non_zero_mask = image > 0
        min_val = np.min(image[non_zero_mask])
        max_val = np.max(image[non_zero_mask])
        if min_val != max_val:
            image[non_zero_mask] = (image[non_zero_mask] - min_val) / (max_val - min_val)
    return image


def pad_image(image, target_dim):
    # Padding to ensure image size
    padding_image = np.zeros(target_dim)
    
    xpadding = (target_dim[0] - image.shape[0]) // 2
    ypadding = (target_dim[1] - image.shape[1]) // 2
    zpadding = (target_dim[2] - image.shape[2]) // 2
    
    padding_image[xpadding:xpadding + image.shape[0],
                  ypadding:ypadding + image.shape[1],
                  zpadding:zpadding + image.shape[2]] = image
    return padding_image

class NiftiDataset(Dataset):
    def __init__(self, dataframe, positive_dir, negative_dir,target_dim):
        self.dataframe = dataframe
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.target_dim = target_dim
        self.data = []
        self.labels = []
        self.patient_ids = []
        self._load_data(positive_dir, label=1)
        self._load_data(negative_dir, label=0)

    def _load_data(self, directory, label):
        for patient_folder in os.listdir(directory):
            folder_path = os.path.join(directory, patient_folder)
            if os.path.isdir(folder_path):
                patient_id = patient_folder
                if patient_id in self.dataframe['Patient ID'].values:
                    self.data.append(folder_path)  # Or load the actual data
                    self.labels.append(label)
                    self.patient_ids.append(patient_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient_path = self.data[idx]
        patient_id = self.patient_ids[idx]

        # Define the file name to be loaded
        file_names = {
            'NCCT': 'NCCT.nii.gz',
            'blood': 'blood.nii.gz', # a binary mask image of blood vessels
            'prob': 'NCCT_Prob.nii.gz', # probabilistic arterial territory maps
            'sub': 'difference_left_registered_right.nii.gz', # hemispheric difference maps
        }

        images = {
            'NCCT': None,
            'blood': None,
            'prob': None,
            'sub': None,
        }

        for key, file_name in file_names.items():
            file_path = os.path.join(patient_path, file_name)
            if file_name == 'NCCT.nii.gz':
                try:
                    ncct_img = nib.load(file_path).get_fdata()
                    if ncct_img.size == 0:  # Check if NCCT is empty
                        print(f"Warning: NCCT for patient {patient_id} is empty.")
                        ncct_img = np.zeros(self.target_dim)
                except Exception as e:
                    print(f"Error loading NCCT for patient {patient_id}: {e}")
                    ncct_img = np.zeros(self.target_dim)  
                images['NCCT'] = ncct_img

            else:
                try:
                    img_data = nib.load(file_path).get_fdata()
                    if img_data.size == 0:
                        print(f"Warning: {file_name} for patient {patient_id} is empty.")
                        images[key] = images['NCCT']  # Fill in the image of this channel with an NCCT image in the absence of other channel images
                    else:
                        images[key] = img_data
                except Exception as e:
                    print(f"Error loading {file_name} for patient {patient_id}: {e}")
                    images[key] = images['NCCT']  # 


        for key in images.keys():
            # Ensure that the image is not None
            if images[key] is None:
                print(f"Warning: {key} image is None for patient {patient_id}. Returning a zero array.")
                images[key] = np.zeros(self.target_dim)

            images[key] = image_normalization(images[key], key)
            images[key] = pad_image(images[key], self.target_dim)

        # Combine the loaded multi class images into input data, Second channel: grayscale image of blood vessels
        input_data = np.stack([images['NCCT'], images['blood']*images['NCCT'], images['prob'], images['sub']], axis=0)
        return torch.tensor(input_data, dtype=torch.float32), self.labels[idx], patient_id

positive_dir = '...YOURDIR/data/train_nii_close'
negative_dir = '...YOURDIR/data/train_nii_nc'

if __name__ == '__main__':
    dataframe = None  
    target_dim = (512, 512, 200)
    dataset = NiftiDataset(dataframe, positive_dir, negative_dir, target_dim)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Example to iterate through the data
    for batch_data, batch_labels in data_loader:
        print(batch_data.shape, batch_labels)
