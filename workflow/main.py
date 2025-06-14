"""
Script:          main.py
Purpose:         Entry-point for training the model
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-10-2025

PyTorch Version: 2.7.1
"""

from models.cae_resnet import Autoencoder, ResNetEncoder, Decoder
from datasets.ws_datamodule import WSDataModule
from models.cae_lightning_module import CAELightningModule
#from src.train import ...

from simclr.augmentations import build_augmentation_pipeline
from utils.config_utils import parse_args, load_config, get_image_paths_from_dir, build_datasets


def main():

    ## Parse the provided configuration file
    args = parse_args()
    config = load_config(args.config)

    ## Initialize the augmentation pipeline as specified by the configurations
    data_transforms = build_augmentation_pipeline(config.get['augmentation'])

    ## Build the training, validation, and testing datasets
    image_paths = get_image_paths_from_dir(config.get('directories', {})
                                                 .get('processed_dir', ''))
    train_dataset, val_dataset, test_dataset = build_datasets(image_paths, config.get('preprocessing', {}).get('patch_size', 200), data_transforms)

    # datamodule = WSDataModule(...)
    # model = ...

    # Initialize the Autoencoder model (as a function?)
    encoder = ResNetEncoder()
    decoder = Decoder()
    autoencoder = Autoencoder(encoder, decoder)
    model = CAELightningModule(autoencoder)


if __name__ == "__main__":
    main()