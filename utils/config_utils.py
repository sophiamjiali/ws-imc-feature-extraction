"""
Script:          config_utils.py
Purpose:         Parses command-line attributes and configuration files
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-12-2025

PyTorch Version: 2.7.1
"""

import numpy as np
import argparse
import yaml
import os
import re
import tifffile as tf
import pandas as pd
import csv

from readimc import MCDFile

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
    

def get_image_paths_from_dir(image_dir, extensions = {'.mcd'}):
    # Returns a list of image file paths from the provided image directory

    image_paths = []
    for file_name in os.listdir(image_dir):
        if os.path.splitext(file_name)[1].lower() in extensions:
            image_paths.append(os.path.join(image_dir, file_name))
    return sorted(image_paths)


def load_image(mcd_path):
    # Loads an MCD file as a NumPy array
    try:
        with MCDFile(mcd_path) as f:
            img = f.read_acquisition(f.slides[0].acquisitions[0])
    except Exception as e:
        print(f"Error processing file: {mcd_path} - {e}")
    return img
    

def load_image_and_markers(img_path, markers_path = None, extensions = ".mcd"):
    # Loads both the MCD File and the marker labels
    
    if extensions == ".mcd":
        try:
            with MCDFile(img_path) as f:
                acq = f.slides[0].acquisitions[0]
                img = f.read_acquisition(acq)
                channel_names = acq.channel_names
        except Exception as e:
            print(f"Error processing file: {img_path} - {e}")
        return img, channel_names
            
    elif extensions == ".tiff":
        img = tf.imread(img_path)
        with open(markers_path, newline='') as f:
            reader = csv.reader(f)
            channel_names = [row[0] for row in reader]

        return img, channel_names
    else:
        return None, None


def load_panel(panel_path):
    # Loads the protein marker panel as a list; returns the metal tags

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

def load_clean_panel(panel_path):
    # Loads the protein marker panel excluding background stains; returns metal tags
    
    markers = []
    with open(panel_path, newline = '') as csvfile:
        next(csvfile) # Skip the first line

        for row in csvfile:
            values = [v.strip() for v in row.strip().split(',')]
            if all(v == '' for v in values): 
                continue
            curr_marker = values[1]

            # If not a background stain, format the label and append
            if int(values[2]) != 0:
                curr_marker = curr_marker.replace(' ', '')
                letters = ''.join(re.findall(r'[A-Za-z]', curr_marker))
                numbers = ''.join(re.findall(r'\d', curr_marker))
                
                markers.append(f"{letters}{numbers}")
                
    return markers

def fetch_marker_indices(markers, panel):
    # Returns a list of corresponding indices for the markers in the panel

    indices = [panel.index(marker) for marker in markers if marker in panel]
    return indices
    

    