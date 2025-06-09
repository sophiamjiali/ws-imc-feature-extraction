"""
Script:          run_preprocessing.py
Purpose:         Preprocesses raw data and augments for model input
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-08-2025

PyTorch Version: 2.7.1
"""

# == Imports ==========================================================



# == Preprocess Data ==================================================

## 1. Convert Raw IMC MCD files to TIFF files

## 2. Extract pixel-level intensities as NumPy Array

## 3. Preprocess the data: normalization, scaling, etc.

## 4. Extract patches using a sliding window approach

## 5. Augment the patches for SimCLR
pipeline = Compose([
    RandomHorizontalFlip(),
    ...
])
aug_img = pipeline(img)

## 6. Create batches of augmented patches

# Convert NumPy arrays into Tensor objects

# Create batcheshow