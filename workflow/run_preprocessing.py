"""
Script:          run_preprocessing.py
Purpose:         Preprocesses raw data and augments for model input
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-08-2025

PyTorch Version: 2.7.1
"""

# == Imports ==========================================================

import yaml
import argparse
import os

from src.preprocess import mcd_to_tiff, load_image, denoise, remove_background, normalize, save_image
from utils.config_utils import parse_args, load_config

# == Preprocess Data ==================================================

def main(config):

    ## Parse and load the configuration file provided
    args = parse_args()
    config = load_config(args.config)

    ## Convert all raw IMC MCD files to TIFF files
    pre_cfg = config.get('preprocessing', {})

    mcd_dir = config['directories'].get('mcd_dir', '')
    tiff_dir = config['directories'].get('tiff_dir', '')
    processed_dir = config['directories'].get('processed_dir', '')

    mcd_to_tiff(mcd_dir, tiff_dir)

    ## Preprocess each TIFF file
    for file_name in os.listdir(tiff_dir):
        if file_name.lower().endswith(('.tiff', 'tif', 'ome.tiff')):

            # A. Load the TIFF image as a NumPy Array
            file_path = os.path.join(tiff_dir, file_name)
            img = load_image(file_path)

            # B. Preprocess the data
            #   Spillover: not implemented yet, might not have an impact
            #   Batch Correction: add as co-variable (?)
            #   Denoise: already implemented, might not have an impact
            img = remove_background(img, pre_cfg.get('background_threshold', 0))
            img = normalize(img, pre_cfg.get('normalize', {}))

            # C. Save the preprocessed data
            save_image(img, file_name, processed_dir)






   

    # C. Extract patches using a sliding window approach

    # D. Augment the patches for SimCLR
    for patch in x:
        
        view1 = data_transforms(patch)
        view2 = data_transforms(patch)

    # E. Create batches of augmented patches (pooled across all IMC images)
    # turn into dataset object (MyDataset(data, labels))
    # then make batches (dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)I s)

    # ... haven't verified below yet

    # Convert NumPy arrays into Tensor objects

    # Create batches

if __name__ == "__main__":
    main()