"""
Script:          cae.py
Purpose:         Sets up the Convolutional Autoencoder with a ResNet encoder
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-10-2025

PyTorch Version: 2.7.1
"""

import torch.nn as nn
from torchvision import models


class Autoencoder(nn.Module):
    # Initializes an autoencoder using a ResNet encoder

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class ResNetEncoder(nn.Module):
    # Initializes a non-pretrained ResNet-18 encoder
    
    def __init__(self,
                 base_model = 'resnet18',
                 in_channels = 3,
                 pretrained = False):
        
        # Initializes a pre-built ResNet model
        super().__init__()
        resnet_constructor = getattr(models, base_model)
        resnet = resnet_constructor(pretrained = pretrained)

        # Modify the first convolutional layer for the marker channels
        if in_channels != 3:

            original_conv = resnet.conv1

            resnet.conv1 = nn.Conv2d(
                in_channels = in_channels,
                out_channels = original_conv.out_channels,
                kernel_si = original_conv.kernel_size,
                stride = original_conv.stride,
                padding = original_conv.padding,
                bias = original_conv.bias is not None
            )
        
        # Remove the fully connected layer and avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # Modify the output channel
        self.out_channels = (512 if base_model in ['resnet18', 'resnet34'] else 2048)

    def forward(self, x):
        return self.features(x)
    

class Decoder(nn.Module):
    # Initializes a custom decoder

    def __init__(self, in_channels, out_channels):
        
        super().__init__()

        self.decoder = nn.sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)