"""
Script:          augmentations.py
Purpose:         Creates image augmentations for SimCLR training
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-08-2025

PyTorch Version: 2.7.1
"""

# == Imports ==========================================================

import numpy as np
import cv2
from skimage.util import random_noise

# == Composition Class ================================================

# Applies the transformations upon the provided image
class Compose:
    
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

# == Image Augmentations Classes ======================================

## 1. Spatial Augmentation

## a. Random cropping
class RandomCropping:
    

## b. Random flipping
class RandomHorizontalFlip:

    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return np.flip(img, axis = 2)
        return img
    
class RandomVerticalFlip:

    def __init__(self, p = 0.5):
        return
    def __call__(self, img):
        return


## c. Random rotation  

## d. Random translation

## 2. Channel-wise Intensity Augmentation

## a. Guassian blur
class ChannelWiseGuassianBlur:

    def __init__(self, kernal_size = 5, sigma = 1.0):
        self.kernal_size = kernal_size
        self.sigma = sigma
    
    def __call__(self, img):
        blurred = np.zeroes_like(img)
        for channel in range(img.shape[0]):
            blurred[channel] = cv2.GaussianBlur(img[channel], 
                                          (self.kernal_size, self.kernal_size), 
                                          self.sigma)
        return blurred



## b. Noise Injection
class ChannelWiseGuassianNoise:

    def __init__(self, var = 0.01):
        self.var = var
    
    def __call__(self, img):
        noisy = np.empty_like(img)
        for channel in range(img.shape[0]):
            noisy[channel] = random_noise(img[channel], 
                                          mode = 'guassian', 
                                          var = self.var)
        return noisy

## c. Intensity Scaling

## 3. Realistic Artifact Augmentation

## a. Defocus

## b. Over-saturation





hmi_transforms = transforms.Compose([
    # Augments WS-IMC patches for SimCLR

    # Random cropping
    transforms.RandomResizedCrop(size = 224, scale = (0.2, 1.0)),

    # Random flipping
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),

    # Guassian blur


    # guassian blur (kernal size = 5, sigma = 1.0) per channel
    # scaling: enlarged image by factor of 2.0, then chose central crop

    transforms.ToTensor()
]) 
 
