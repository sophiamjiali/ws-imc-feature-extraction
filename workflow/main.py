"""
Script:          main.py
Purpose:         Entry-point for training the model
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-10-2025

PyTorch Version: 2.7.1
"""

from datasets.ws_dataset import WSDataset
from datasets.ws_datamodule import WSDataModule
#from src.train import ...

from simclr.augmentations import build_augmentation_pipeline

from utils.config_utils import parse_args, load_config


def main():

    ## Parse the provided configuration file
    args = parse_args()
    config = load_config(args.config)

    ## Initialize the augmentation pipeline as specified by the configurations
    data_transforms = build_augmentation_pipeline(config.get['augmentation'])

    ## Build the dataset
    image_paths = ...

    dataset = WSDataset(
        ...
    )

    # datamodule = WSDataModule(...)
    # model = ...

    # Initialize the Autoencoder model (as a function?)
    encoder = ResNetEncoder()
    decoder = Decoder()
    autoencoder = Autoencoder(encoder, decoder)


if __name__ == "__main__":
    main()