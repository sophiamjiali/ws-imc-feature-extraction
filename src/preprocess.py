"""
Script:          preprocess.py
Purpose:         Preprocesses raw WS-IMC data
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-08-2025

PyTorch Version: 2.7.1
"""

# == Imports ==========================================================

import tifffile as tf
import numpy as np
import pandas as pd
import scanpy as sc
import os
import cv2

from skimage.util import view_as_windows
from scipy.stats.mstats import winsorize
from scipy.ndimage import median_filter
from sklearn.preprocessing import MinMaxScaler

from readimc import MCDFile

# == Preprocessing Functions ===========================================

## 1. Image Conversion: convert raw MCD files to OME-TIFF files and NumPy arrays

def mcd_to_tiff(mcd_dir, tiff_dir):
    # Converts all MCD files in the provided directory to TIFF files

    for file in os.listdir(mcd_dir):

        mcd_file = os.path.join(mcd_dir, file)
        name = [file.split(' ')[-1].split('.')[0] if file.startswith('2024') else file.split('.')][0]

        with MCDFile(mcd_file) as mcd_read:

            for slide in mcd_read.slides:
                for acquisition in slide.acquisitions:

                    try:
                        img = mcd_read.read_acquisition(acquisition)
                        file_name = name + f"_{acquisition.description}.tiff"
                        tf.imwrite(os.path.join(tiff_dir, file_name), img)
                    except:
                        print(f"Error processing {file} - acquisition {acquisition.description}")

def load_image(tiff_path):
    # Loads a TIFF file as a NumPy array

    try:
        img = tf.imread(tiff_path)
    except Exception as e:
        file_name = os.path.basename(tiff_path)
        print(f"Error loading image {file_name}: {e}")
        return None
    return img

## 2. Remove Background Stains

    # load marker panel
    # remove the associated channels

## 3. Spillover Correction

    # might not cause an issue, here as placeholder

## 3. Denoising: remove hot pixels, shot noise, etc.

def denoise(img, size = (1, 3, 3)):
    # Applies a median filter to remove hot pixels
    return median_filter(img, size)

## 4. Background Removal: remove non-tissue regions

def remove_background(img, threshold = 4):
    # Removes the background per channel via intensity threshold
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
    return

def winsorize_intensity(img, quantile = 0.01):
    # Winsorizes the top and bottom quantile specified
    return winsorize(img, [quantile, quantile])

def min_max_scale(img):
    # Scales the pixel intensities to [0, 1]
    scaler = MinMaxScaler()
    return scaler.fit_transform(img)

## 5. Quality Control: screen for batch effects

def batch_effects():
    return
    # add the batch as a co-variable

    # might not cause an issue, here as placeholder


## 6. Patch Extraction: divide whole-slide images via sliding window approach



        # change it so I initialize/save the tissue mask, then do it on the fly in the Dataset class (see both functions below)



def extract_patches(img, patch_size = (200, 200)):
    # Extracts patches with sufficient biological content via sliding window

    # Create a binary tissue mask to separate background from tissue
    tissue_mask = create_tissue_mask(img)


    return

def create_tissue_mask(img):
    # Creates a binary tissue mask to separate background from foreground
    channel_sums = img.sum(axis = 0)
    bool_mask = channel_sums > 0
    return np.where(bool_mask, 1, 0)






## 7. Save Preprocessed Image

def save_image(img, file_name, processed_dir):
    # Saves the preprocessed image to disk
    img = np.moveaxis(img, 0, -1)
    tifffile.imwrite()
    