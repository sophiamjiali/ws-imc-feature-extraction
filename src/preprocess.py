"""
Script:          preprocess.py
Purpose:         Preprocesses raw WS-IMC data
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-08-2025

PyTorch Version: 2.7.1
"""

# == Imports ==========================================================

import numpy as np

from scipy.stats.mstats import winsorize
from scipy.ndimage import median_filter
from sklearn.preprocessing import MinMaxScaler

# == Entry-point Main Function =========================================

def preprocess_image(img, markers, panel, preproc_cfg = None):
    # Wraps preprocesses the provided image

    img = remove_background_stains(img, panel, markers)
    img = normalize(img, preproc_cfg.get('normalize'))

    return img

# == Preprocessing Functions ===========================================

## 2. Remove Background Stains
def remove_background_stains(img, panel, markers):
    # Removes background stains by omittance in markers
    background_idx = [idx for idx, marker in enumerate(markers) if marker not in panel]

    for idx in sorted(background_idx, reverse = True):
        img = np.delete(img, idx, axis = 0)
    return img


## 3. Spillover Correction

    # might not cause an issue, here as placeholder


## 3. Denoising: remove hot pixels, shot noise, etc.
def denoise(img, size = (1, 3, 3)):
    # Applies a median filter to remove hot pixels
    return median_filter(img, size)


## 4. Background Removal: remove non-tissue regions
def remove_background(img, threshold = 4):
    # Removes the background via intensity threshold across all channels
    img[img < threshold] = 0
    return img


## 4. Normalization: channel-wise intensity normalization and scaling
def normalize(img, normalize_cfg):
    # Normalizes and scales the image via winsorization and min-max scaling
    winsorize_cfg = normalize_cfg.get('winsorization', {})

    if winsorize_cfg.get('enabled', True):
        img = winsorize_intensity(img, winsorize_cfg.get('quantile', 0.0))
    if normalize_cfg.get('min_max_scale', True):
        img = min_max_scale(img)
    return img

def winsorize_intensity(img, quantile = 0.01):
    # Winsorizes the top and bottom quantile specified per channel
    channels = img.shape[0]
    img_wins = np.zeros_like(img, dtype = np.float32)

    for c in range(channels):
        img_wins[c] = winsorize(img[c], limits = [quantile, quantile])

    return img_wins

def min_max_scale(img):
    # Scales the pixel intensities to [0, 1] per channel
    channels = img.shape[0]
    img_norm = np.zeros_like(img, dtype = np.float32)

    for c in range(channels):
        scaler = MinMaxScaler()
        channel_data = img[c]
        img_norm[c] = scaler.fit_transform(channel_data)

    return img_norm


## 5. Quality Control: screen for batch effects
def batch_effects():
    return
    # add the batch as a co-variable

    # might not cause an issue, here as placeholder


## 6. Patch Extraction: divide whole-slide images via sliding window approach
def extract_patch(img, 
                  patch_size = (200, 200), 
                  coords = (0, 0),
                  img_shape = (1000, 1000)):
    # Extracts a patch starting at (X,Y), padding if necessary
    
    y_end = min(coords[0] + patch_size[0], img_shape[0])
    x_end = min(coords[1] + patch_size[1], img_shape[1])
    
    patch = img[:, coords[0]:y_end, coords[1]:x_end]

    pad_h = patch_size[0] - (y_end - coords[0])
    pad_w = patch_size[1] - (x_end - coords[1])

    if pad_h > 0 or pad_w > 0:
        patch = np.pad(
            patch,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode = "constant"
        )

    return patch

## 7. Patch Quality Control: evaluate sufficiency of biological content
def has_sufficient_content(patch, threshold):
    # Returns true if the patch contains at least the threshold percentage
    channel_sums = patch.sum(axis = 0)
    bool_mask = channel_sums > 0
    return np.mean(bool_mask) > threshold
