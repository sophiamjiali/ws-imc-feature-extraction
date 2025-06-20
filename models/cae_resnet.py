"""
Script:          cae_resnet.py
Purpose:         Sets up the Convolutional Autoencoder with a ResNet encoder
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-10-2025

PyTorch Version: 2.7.1
"""

import torch.nn as nn
from torchvision import models


class ConvAutoencoder(nn.Module):
    # Initializes an autoencoder using a ResNet encoder

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ResNetEncoder(nn.Module):
    # Initializes a non-pretrained ResNet-18 encoder
    
    def __init__(
            self,
            base_model = 'resnet18',
            in_channels = 52,
            weights = None):
        
        # Initializes a pre-built ResNet model
        super().__init__()
        resnet_constructor = getattr(models, base_model)
        resnet = resnet_constructor(weights = weights)

        # Modify the first convolutional layer for the marker channels
        if in_channels != 3:

            original_conv = resnet.conv1

            resnet.conv1 = nn.Conv2d(
                in_channels = in_channels,
                out_channels = original_conv.out_channels,
                kernel_size = original_conv.kernel_size,
                stride = original_conv.stride,
                padding = original_conv.padding,
                bias = original_conv.bias is not None
            )
        
        # Remove the fully connected layer and avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.model(x)
    

class Decoder(nn.Module):
    # Initializes a custom decoder

    def __init__(self, in_channels = 512, out_channels = 52):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 3, 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)