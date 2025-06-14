"""
Script:          ws_dataset.py
Purpose:         Defines the dataset class for WS-IMC images
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-10-2025

PyTorch Version: 2.7.1
"""

# == Imports ==========================================================

import torch
from torch.utils.data import Dataset
from utils.config_utils import load_image



# == Class Definitions ================================================

class WSDataset(Dataset):

    def __init__(self, 
                 image_dir, 
                 patch_size = (200, 200), 
                 stride = (200, 200),
                 transforms = None):
        
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.stride = stride
        self.transforms = transforms

        # Precompute patch coordinates (img_id, y, x)
        self.patch_coords = []
        for img_idx, img in enumerate(self.images):
            H, W = img.shae[:2]
            for y in range(0, H - patch_size[0] + 1, stride[0]):
                for x in range(0, W - patch_size[1] + 1, stride[1]):
                    self.patch_coords.append((img_idx, y, x))
    
    def __len__(self):
        return len(self.patch_coords)
    
    def _load_image(self, img):
        return load_image(img)
         
    def __getitem__(self, idx):
        # Fetches the next patch and augmented views for input to the model
        img_idx, y, x, = self.patch_coords[idx]

        img = self._load_image(self.images[img_idx])
        patch = img[y:y + self.patch_size[0], x:x + self.patch_size[1]]

        # Generate two randomly augmented views of the patch
        view1 = self.transform(patch)
        view2 = self.transform(patch)
        
        # Convert from NumPy arrays to tensor objects
        view1 = torch.from_numpy(view1).float
        view2 = torch.from_numpy(view2).float

