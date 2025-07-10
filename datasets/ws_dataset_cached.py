"""
Script:          ws_dataset_cached.py
Purpose:         Cached version of ws_dataset.py
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-25-2025

PyTorch Version: 2.7.1
"""

import torch
from torch.utils.data import Dataset
import numpy as np

import time

class WSDatasetCached(Dataset):

    def __init__(self, image_paths = None):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, patch_idx):
        patch_path = self.image_paths[patch_idx]
        patch = np.load(patch_path)
        patch = torch.from_numpy(patch).float()
        return patch, patch
        