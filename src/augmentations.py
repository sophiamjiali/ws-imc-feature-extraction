"""
Script:          augmentations.py
Purpose:         Creates image augmentations for SimCLR training
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-08-2025

PyTorch Version: 2.7.1
"""

# == Imports ==========================================================

import torch
import numpy as np
import cv2
from skimage.util import random_noise

# == Composition Class ================================================

class Compose:
    # Applies the transformations upon the provided image
    
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
    

def build_augmentation_pipeline(aug_cfg):
    # Builds the augmentation list per the provided configurations

    transforms = []
    if aug_cfg.get('resize', {}).get('enabled', True):
        transforms.append(Resize(aug_cfg['resize']['size']))
    if aug_cfg.get('random_cropping', {}).get('enabled', True):
        transforms.append(RandomCrop(aug_cfg['random_cropping']['crop_size']))
    if aug_cfg.get('random_flip', True):
        transforms.append(RandomHorizontalFlip())
        transforms.append(RandomVerticalFlip())
    if aug_cfg.get('random_rotation', True):
        transforms.append(RandomRotation())
    if aug_cfg.get('random_translation', {}).get('enabled', True):
        transforms.append(RandomTranslation(aug_cfg['random_translation']['max_shift']))
    if aug_cfg.get('gaussian_blur', True):
        transforms.append(ChannelWiseGaussianBlur())
    if aug_cfg.get('gaussian_noise', True):
        transforms.append(ChannelWiseGaussianNoise())
    if aug_cfg.get('intensity_scaling', {}).get('enabled', True):
        transforms.append(ChannelWiseIntensityScaling(aug_cfg['intensity_scaling']['scale_range']))

    transforms.append(ToTensor)

    return Compose(transforms) 


# == Image Augmentations Classes ======================================

## 0. Resize to (224x224)
class Resize:
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        img = np.transpose(img, (2, 0, 1)) # (C, H, W) -> (H, W, C)
        img = cv2.resize(img, self.size, interpolation = cv2.INTER_AREA)
        img = np.transpose(img, (2, 0, 1)) # (H, W, C) -> (C, H, W)
        return img
    
## 1. Tensor Conversion
class ToTensor:
    def __init__(self):
        pass
    def __call__(self, img):
        return torch.from_numpy(img).float()


## 1. Spatial Augmentation

## a. Random cropping
class RandomCrop:

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        C, H, W = img.shape
        ch, cw = self.crop_size
        if H == ch and W == cw:
            return img
        top = np.random.randint(0, H - ch + 1)
        left = np.random.randint(0, W - cw + 1)
        return img[:, top:top + ch, left:left + cw]


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
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return np.flip(img, axis = 1)


## c. Random rotation  
class RandomRotation:

    def __init__(self, p = 1.0):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            angle = np.random.randint(0, 4)
            return np.rot90(img, angle, axes = (1, 2))
        return img

## d. Random translation
class RandomTranslation:

    def __init__(self, max_shift = 20, p = 0.5):
        self.max_shift = max_shift
        self.p = p
    
    def __call__(self, img):
        if np.random.rand() < self.p:
            shift_x = np.random.randint(-self.max_shift, self.max_shift + 1)
            shift_y = np.random.randint(-self.max_shift, self.max_shift + 1)
            img = np.roll(img, shift_x, axis = 2)
            img = np.roll(img, shift_y, axis = 1)

            # pad empty space with zeroes
            if shift_x > 0:
                img[:, :, :shift_x] = 0
            elif shift_x < 0:
                img[:, :, shift_x:] = 0
            if shift_y > 0:
                img[:, :shift_y, :] = 0
            elif shift_y < 0:
                img[:, shift_y:, :] = 0
        return img

## 2. Channel-wise IntenHowsity Augmentation

## a. Guassian blur
class ChannelWiseGaussianBlur:

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
class ChannelWiseGaussianNoise:

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
class ChannelWiseIntensityScaling:

    def __init__(self, scale_range = (0.8, 1.2), p = 0.5):
        self.scale_range = scale_range
        self.p  = p
    
    def __call__(self, img):
        if np.random.rand() < self.p:
            scales = np.random.uniform(self.scale_range[0], 
                                       self.scale_range[1], 
                                       size = (img.shape[0], 1, 1))
            img = img * scales
        return img
 
