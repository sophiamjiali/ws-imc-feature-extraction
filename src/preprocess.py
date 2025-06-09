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


## 2. Denoising and Artifact Removal: remove hot pixels, shot noise, etc.

def denoise(img, denoise_cfg):
    # Applies both median filter and Guassian blur in sequence

    filter_cfg = denoise_cfg.get('median_filter', {})
    blur_cfg = denoise_cfg.get('guassian_blur', {})

    if filter_cfg.get('enabled', True):
        img = median_filter_denoise(img, filter_cfg.get('size', (1, 3, 3)))

    if blur_cfg.get('enabled', True):
        img = gaussian_blur_denoise(img, 
                                    blur_cfg.get('k_size', 3), 
                                    blur_cfg.get('sigma', 1))
    return img

def median_filter_denoise(img, size = (1, 3, 3)):
    # Applies a median filter to remove hot pixels
    return median_filter(img, size = size)

def gaussian_blur_denoise(img, k_size = 3, sigma = 1):
    # Applies a channel-wise Gaussian blur to smooth noise

## 3. Background Removal: remove non-tissue regions


## 4. Normalization: channel-wise intensity normalization and scaling
def normalize(img):
    return

## 5. Quality Control: screen for artifacts, low signal, and batch effects

## 6. Patch Extraction: divide whole-slide images via sliding window approach

def extract_patches(img, patch_size = (200, 200)):
    return
