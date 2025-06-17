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
import re


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


def load_image(mcd_path):
    # Loads an MCD file as a NumPy array
    try:
        with MCDFile(mcd_path) as f:
            img = f.read_acquisition(f.slides[0].acquisitions[0])
    except Exception as e:
        print(f"Error processing file: {mcd_path} - {e}")
    return img
    

def load_image_and_markers(mcd_path):
    # Loads both the MCD File and the marker labels
    try:
        with MCDFile(mcd_path) as f:
            acq = f.slides[0].acquisitions[0]
            img = f.read_acquisition(acq)
            channel_names = acq.channel_names
    except Exception as e:
        print(f"Error processing file: {mcd_path} - {e}")

    return img, channel_names

def load_panel(panel_path):
    # Loads the protein marker panel as a list

    markers = []
    with open(panel_path, newline = '') as csvfile:
        next(csvfile) # Skip the first line

        for row in csvfile:
            values = [v.strip() for v in row.strip().split(',')]
            if all(v == '' for v in values): 
                continue
            curr_marker = values[1]

            # Format the metal label to align with MCD metadata
            curr_marker = curr_marker.replace(' ', '')
            letters = ''.join(re.findall(r'[A-Za-z]', curr_marker))
            numbers = ''.join(re.findall(r'\d', curr_marker))
            
            markers.append(f"{letters}{numbers}")
    return markers