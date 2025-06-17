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
from utils.config_utils import load_image, load_image_and_markers
import numpy as np

from src.preprocess import remove_background_stains, normalize, extract_patch, has_sufficient_content

# == Class Definitions ================================================

class WSDataset(Dataset):

    def __init__(self, 
                 image_paths, 
                 patch_size = (200, 200), 
                 stride = (200, 200),
                 transforms = None,
                 panel = None,
                 preproc_cfg = None):
        
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.stride = stride
        self.transforms = transforms
        self.panel = panel
        self.preproc_cfg = preproc_cfg
        self._cache_img_idx = None
        self._cache_img = None

        # Precompute valid patch coordinates (img_id, y, x)
        self.patch_coords = []
        for img_idx, img_path in enumerate(self.image_paths):

            img = self._load_image(img_path)
            H, W = img.shape[-2:]

            for y in range(0, H, self.stride[0]):
                for x in range(0, W, self.stride[1]):

                    # Extract and pad the patch if necessary
                    patch = extract_patch(img, self.patch_size, (y, x), (H, W))
                    
                    # Screen for if the patch has sufficient biological content
                    if has_sufficient_content(patch, self.preproc_cfg.get('bio_content_threshold')):
                        self.patch_coords.append((img_idx, y, x))
    

    def __len__(self):
        return len(self.patch_coords)
    
    def _load_image(self, path):
        return load_image(path)
        
    def __getitem__(self, patch_idx):
        # Fetches the next patch and augmented views for input to the model

        img_idx, y, x, = self.patch_coords[patch_idx]

        # If processing the same whole-slide image, load it; else, cache it
        if self._cache_img_idx != img_idx:

            # Load the image and preprocess it
            img, markers = load_image_and_markers(self.image_paths[img_idx])
            img = remove_background_stains(img, self.panel, markers)
            img  = normalize(img, self.preproc_cfg.get('normalize'))

            # Cache the preprocessed image and update the index
            self._cache_img = img
            self._cache_img_idx = img_idx

        # Extract a patch with sufficient biological content
        patch = self._cache_img[y:y + self.patch_size[0], x:x + self.patch_size[1], ...]

        # Augment the patch and convert to Tensor object
        patch = np.transpose(patch, (2, 0, 1))
        patch = self.transform(patch)
        patch = torch.from_numpy(patch).float
        
        # Input and target are the same for CAE
        return patch, patch

