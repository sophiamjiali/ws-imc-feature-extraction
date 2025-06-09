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

from src.preprocess import mcd_to_tiff, load_image
from simclr.augmentations import Compose, build_augmentation_pipeline, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomTranslation, ChannelWiseGaussianBlur, ChannelWiseGaussianNoise, ChannelWiseIntensityScaling

# == Preprocess Data ==================================================

def main(config):

    ##  Convert all raw IMC MCD files to TIFF files
    mcd_dir = config['directories'].get('mcd_dir', '')
    tiff_dir = config['directories'].get('tiff_dir', '')
    mcd_to_tiff(mcd_dir, tiff_dir)

    ## Initialize the augmentation pipeline as specified by the configurations
    data_transforms = build_augmentation_pipeline(config['augmentation'])

    ## Preprocess each TIFF file
    for file_name in os.listdir(tiff_dir):
        if file_name.lower().endswith(('.tiff', 'tif', 'ome.tiff')):

            # A. Load the TIFF image as a NumPy Array
            file_path = os.path.join(tiff_dir, file_name)
            img = load_image(file_path)

            # B. Preprocess the data
            img = denoise(img, config['preprocessing']['denoise'])

            # C. Extract patches using a sliding window approach

            # D. Augment the patches for SimCLR
            for patch in x:
                
                view1 = data_transforms(patch)
                view2 = data_transforms(patch)

    # E. Create batches of augmented patches (pooled across all IMC images)


    # ... haven't verified below yet

    # Convert NumPy arrays into Tensor objects

    # Create batches

if __name__ == "__main__":

    # parse the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, required = True,
                         help = 'Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)