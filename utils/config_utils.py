"""
Script:          config_utils.py
Purpose:         Parses command-line attributes and configuration files
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-12-2025

PyTorch Version: 2.7.1
"""

import argparse
import yaml
import os


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