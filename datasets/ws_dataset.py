"""
Script:          ws_dataset.py
Purpose:         Defines the dataset class for WS-IMC images
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-10-2025

PyTorch Version: 2.7.1
"""

# == Imports ==========================================================

from torch.utils.data import Dataset
from utils.config_utils import load_image, load_image_and_markers

from src.preprocess import preprocess_image, extract_patch
import random

# == Class Definitions ================================================

class WSDataset(Dataset):

    def __init__(self, 
                 image_paths = None,
                 patch_coords = None,
                 patch_size = (200, 200), 
                 stride = (200, 200),
                 transforms = None,
                 panel = None,
                 preproc_cfg = None):
        
        self.image_paths = image_paths
        self.patch_coords = patch_coords
        self.patch_size = patch_size
        self.stride = stride
        self.transforms = transforms
        self.panel = panel
        self.preproc_cfg = preproc_cfg
        self._cache_img_idx = None
        self._cache_img = None

        # Initialize random image and patch order
        self._prepare_epoch_indices()
        
    def _prepare_epoch_indices(self):
        # Randomizes patch order per image, and image order

        self.epoch_coords = []

        img_indices = list(self.patch_coords.keys())
        random.shuffle(img_indices)

        for img_idx in img_indices:
            patches = self.patch_coords[img_idx][:]
            random.shuffle(patches)
            self.epoch_coords.extend([(img_idx, patch) for patch in patches])

    def on_epoch_start(self):
        # Randomizes patch order per image
        self._prepare_epoch_indices()

    def __len__(self):
        return len(self.epoch_coords)
    
    def _load_image(self, path):
        return load_image(path)
        
    def __getitem__(self, patch_idx):
        # Fetches the next patch and augmented views for input to the model

        img_idx, (y, x), = self.epoch_coords[patch_idx]

        # If processing the same whole-slide image, load it; else, cache it
        if self._cache_img_idx != img_idx:

            # Load the image and preprocess it
            img, markers = load_image_and_markers(self.image_paths[img_idx])
            img = preprocess_image(img, markers, self.panel, self.preproc_cfg)

            # Cache the preprocessed image and update the index
            self._cache_img = img
            self._cache_img_idx = img_idx

        # Extract a patch with sufficient biological content
        H, W = self._cache_img.shape[-2:]
        patch = extract_patch(self._cache_img, self.patch_size, (y, x), (H, W))

        # Augment the patch and convert to Tensor object
        patch = self.transforms(patch)
        
        # Input and target are the same for CAE
        return patch, patch
