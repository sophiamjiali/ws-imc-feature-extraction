"""
Script:          compute_correlations.py
Purpose:         Computes and saves correlations of each WS-IMC patch.
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            07-07-2025

PyTorch Version: 2.7.1
"""
import tifffile 
import pandas as pd
import numpy as np
import os
from utils.config_utils import load_image_and_markers, get_image_paths_from_dir
from utils.extract_patches import extract_img_id
from multiprocessing import Pool
from src.preprocess import preprocess_image


def extract_patch_correlations(ws_dir, corr_dir, common_stains, preproc_cfg):
    # Calculate and save the flattened correlation matrix of each patch per image

    img_paths = get_image_paths_from_dir(ws_dir)
    args = [(img_path, corr_dir, common_stains, preproc_cfg) for img_path in img_paths]
    
    with Pool() as pool:
        pool.starmap(_patch_correlations_per_img, args)


def _patch_correlations_per_img(img_path, corr_dir, common_stains, preproc_cfg):
    # Extract and save correlation matrices for the given WS-IMC image

    img_id = extract_img_id(img_path)

    stride = preproc_cfg.get('stride')
    patch_size = preproc_cfg.get('patch_size')

    ## Load and preprocess the image; filter for common markers
    img, markers = load_image_and_markers(img_path)
    img = preprocess_image(img, markers, common_stains, preproc_cfg)
    H, W = img.shape[-2:]

    ## Collect all patch correlations to save together as a single CSV
    patch_corrs = []
    patch_indices = []

    ## Compute all possible patches
    for y in range(0, H, stride[0]):
        for x in range(0, W, stride[1]):

            # Extract the patch, allowing overlap if on an edge
            y_end = min(y + patch_size[0], H)
            y_start = max(y_end - patch_size[0], 0)
            
            x_end = min(x + patch_size[1], W)
            x_start = max(x_end - patch_size[1], 0)

            patch = img[:, y_start:y_end, x_start:x_end]

            # Compute and save the flattened pairwise correlation matrix
            corr_matrix = _compute_patch_correlations(patch, common_stains)

            # Add the patch and its locations to the running list
            patch_corrs.append(corr_matrix)
            patch_indices.append((y, x))

    ## Concatenate all patch correlations into a single object for saving
    all_corrs = pd.concat(patch_corrs, keys = patch_indices, 
                          names = ['Patch', 'Marker_pair'])
    out_path = os.path.join(corr_dir, f"{img_id}_patch_correlations.csv")
    all_corrs.to_csv(out_path)


def _compute_patch_correlations(patch, common_stains):
    # Compute and flatten the pairwise correlation matrix

    # Flatten the array: (C, H, W) -> (C, H*W)
    flat = patch.reshape(patch.shape[0], -1)
    corr_matrix = np.corrcoef(flat)

    # Fetch the unique correlations (upper triangle)
    triu_indices = np.triu_indices_from(corr_matrix, k = 1)
    flat_unique_corr = corr_matrix[triu_indices]

    # Convert to a Pandas dataframe and label with marker names
    pairs = [(common_stains[i], common_stains[j]) for i, j in zip(*triu_indices)]
    index = pd.MultiIndex.from_tuples(pairs, names = ["Marker 1", "Marker 2"])
    df_flat_corr = pd.DataFrame(flat_unique_corr, index = index, columns = ["Correlation"])
    
    return df_flat_corr


def extract_tma_correlations(tma_dir, tma_corr_dir, common_stains, 
                             ws_mapping, tma_mapping, preproc_cfg):
    # Calculate and save the flattened correlation matrix of each TMA core image

    ## Initialize the mappings and TMA image paths
    caseid_to_barcode = dict(zip(ws_mapping['CaseID'], ws_mapping['Barcode']))
    tiff_to_caseid = dict(zip(tma_mapping['TiffName'], tma_mapping['CaseID']))
    tma_paths = get_image_paths_from_dir(tma_dir, {".tiff"})

    ## Map TMA images to WS-IMC barcodes
    ws_to_tma_paths = {}
    for tma_path in tma_paths:

        # If the TMA image maps to a WS image, track it
        tiff_name = os.path.basename(tma_path)
        
        if tiff_name in tiff_to_caseid:
            case_id = tiff_to_caseid[tiff_name]
            barcode = caseid_to_barcode[case_id]
            ws_to_tma_paths.setdefault(barcode, []).append(tma_path)

    ## Save each TMA correlation grouped by WS-IMC Barcode
    for barcode, tma_img_list in ws_to_tma_paths.items():
        tma_corrs = []
        tma_filenames = []

        for tma_path in tma_img_list:

            ## Load and preprocess each TMA image
            markers_path = os.path.splitext(tma_path)[0] + '.csv'
            img, markers = load_image_and_markers(tma_path, markers_path, ".tiff")
            img = preprocess_image(img, markers, common_stains, preproc_cfg)

            ## Compute the correlation matrix and save it
            corr_matrix = _compute_tma_correlations(img, common_stains)
            tma_corrs.append(corr_matrix)
            tma_filenames.append(os.path.basename(tma_path))

        ## Concatenate all TMA correlations into a single object for saving
        all_corrs = pd.concat(tma_corrs, keys = tma_filenames, names = ['TMA', 'Marker_pair'])
        out_path = os.path.join(tma_corr_dir, f"{barcode}_tma_correlations.csv")
        all_corrs.to_csv(out_path)


def _compute_tma_correlations(img, common_stains):
    # Compute and flatten the pairwise correlation matrix

    # Flatten the array: (C, H, W) -> (C, H*W)
    flat = img.reshape(img.shape[0], -1)
    corr_matrix = np.corrcoef(flat)

    # Fetch the unique correlations (upper triangle)
    triu_indices = np.triu_indices_from(corr_matrix, k = 1)
    flat_unique_corr = corr_matrix[triu_indices]

    # Convert to a Pandas dataframe and label with marker names
    pairs = [(common_stains[i], common_stains[j]) for i, j in zip(*triu_indices)]
    index = pd.MultiIndex.from_tuples(pairs, names = ["Marker 1", "Marker 2"])
    df_flat_corr = pd.DataFrame(flat_unique_corr, index = index, columns = ["Correlation"])
    
    return df_flat_corr
