"""
Script:          loggers.py
Purpose:         Extracts and saves patches from raw IMC images
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-25-2025

PyTorch Version: 2.7.1
"""
from multiprocessing import Pool
from src.preprocess import extract_patch, preprocess_image, has_sufficient_content
from utils.config_utils import load_image_and_markers
import os
import numpy as np
import cv2

def extract_and_save_patches(image_paths, panel, preproc_cfg, patch_dir):
    # Parallelization adapter for extracting and saving patches

    print("Beginning patch extraction and saving...")

    args = [(img_path, panel, preproc_cfg, patch_dir) for img_path in image_paths]
    
    with Pool() as pool:
        pool.starmap(_extract_per_img, args)


def _extract_per_img(img_path, panel, preproc_cfg, patch_dir):
    # Extracts and saves processes from raw WS-IMC images

    stride = preproc_cfg.get('stride')
    patch_size = preproc_cfg.get('patch_size')
    content_threshold = preproc_cfg.get('bio_content_threshold')
    resize_size = preproc_cfg.get('resize_size')

    ## Load and Preprocess the raw image
    img, markers = load_image_and_markers(img_path)
    img = preprocess_image(img, markers, panel, preproc_cfg)
    H, W = img.shape[-2:]

    ## Compute and quality control all possible patches
    for y in range(0, H, stride[0]):
        for x in range(0, W, stride[1]):

            # Extract and pad the patch if necessary
            patch = extract_patch(img, patch_size, (y, x), (H, W))

            # Screen for sufficient biological content
            if has_sufficient_content(patch, content_threshold):

                # Resize the (200,200) patch to (224,224)
                patch = resize_patch(patch, resize_size)
                
                # Save the patch, named by the original patch
                img_id = extract_img_id(img_path)
                filename = f"{img_id}_y{y}_x{x}.npy"
                save_path = os.path.join(patch_dir, filename)
                np.save(save_path, patch)


def resize_patch(patch, resize_size):
    # Resizes the patch from (200,200) to (224,224) for model input
    patch = np.transpose(patch, (1, 2, 0)) # (C, H, W) -> (H, W, C)
    patch = cv2.resize(patch, resize_size, interpolation = cv2.INTER_AREA)
    patch = np.transpose(patch, (2, 0, 1)) # (H, W, C) -> (C, H, W)
    return patch


def extract_img_id(img_path):
    # Extract the Barcode/UHNL_ID from the image path
    filename = os.path.basename(img_path)
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    return parts[2]



                
    