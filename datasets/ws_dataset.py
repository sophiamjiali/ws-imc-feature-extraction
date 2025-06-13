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



# == Class Definitions ================================================



class WSDataset(Dataset):

    def __init__(self, image_dir, patch_size = (200, 200), transforms = None):
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.transforms = transforms
    
    def __len__(self):
        


