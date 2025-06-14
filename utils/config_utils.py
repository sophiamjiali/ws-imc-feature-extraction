"""
Script:          config_utils.py
Purpose:         Parses command-line attributes and configuration files
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-12-2025

PyTorch Version: 2.7.1
"""

import random
import argparse
import yaml
import os
import numpy as np
import tifffile as tf

from readimc import MCDFile
from datasets.ws_dataset import WSDataset


def parse_args():
    # Parse command-line arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, required = True,
                         help = 'Path to YAML config file')
    return parser.parse_args()


def load_config(path):
    # Loads in the provided configuration file

    with open(path, 'r') as f:
        return yaml.safe_load(f)
    

def get_image_paths_from_dir(image_dir, extensions = {'.tiff'}):
    # Returns a list of image file paths from the provided image directory

    image_paths = []
    for file_name in os.listdir(image_dir):
        if os.path.splitext(file_name)[1].lower() in extensions:
            image_paths.append(os.path.join(image_dir, file_name))
    return sorted(image_paths)


def build_datasets(image_paths = '', patch_size = (200,200), transforms = None):
    # Builds the train, validation, and test datasets

    random.shuffle(image_paths)
    n = len(image_paths)

    train_paths = image_paths[:int(0.7 * n)]
    val_paths = image_paths[int(0.7 * n):int(0.85 * n)]
    test_paths = image_paths[int(0.85 * n):]

    train_dataset = WSDataset(train_paths, patch_size, transforms)
    val_dataset = WSDataset(val_paths, patch_size, transforms)
    test_dataset = WSDataset(test_paths, patch_size, transforms)

    return train_dataset, val_dataset, test_dataset


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


def save_image(img, file_name, processed_dir):
    # Saves the preprocessed image to disk
    img = np.moveaxis(img, 0, -1)
    tf.imwrite()
    