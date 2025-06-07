"""
Script:          resnet_backbone.py
Purpose:         Sets up the ResNet backbone for the CNN model
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-07-2025

PyTorch Version: 2.7.1
"""

# == Imports ==========================================================

import torch
import torch.nn as nn
import torchvision.models as models

# == ResNet Backbone Class ============================================

class ResNetBackbone(nn.Module):
    # Initializes a non-pretrained ResNet-18 backbone

    def __init__(self, 
                 base_model: str = 'resnet18', 
                 in_channels: int = 3, 
                 pretrained: bool = False):
        
        super().__init__()

        # Initialize the ResNet backbone using the provided constructor
        resnet_constructor = getattr(models, base_model)
        resnet = resnet_constructor(pretrained = pretrained)

        # Modify the first convolutional layer for the marker channels
        if in_channels != 3:

            original_conv = resnet.conv1

            resnet.conv1 = nn.Conv2d(
                in_channels = in_channels,
                out_channels = original_conv.out_channels,
                kernal_size = original_conv.kernel_size,
                stride = original_conv.stride,
                padding = original_conv.padding,
                bias = original_conv.bias is not None
            )

            # Reinitialize the weights of the new convolutional layer
            nn.init.kaiming_normal_(resnet.conv1.weight, 
                                    mode='fan_out', 
                                    nonlinearity='relu')
            
        # Remove the final fully connecte layer and average pooling (avgpool)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.out_features = resnet.fc.in_features


    def forward(self, x):

        # Forward pass through the ResNet backbone
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x