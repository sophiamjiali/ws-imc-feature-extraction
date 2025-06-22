"""
Script:          main.py
Purpose:         Entry-point for training the model
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-10-2025

PyTorch Version: 2.7.1
"""


from datamodules.ws_datamodule import WSDataModule
from lightning import seed_everything

from src.train import train_cae
from utils.loggers import get_loggers
from utils.callbacks import get_callbacks

from src.augmentations import build_augmentation_pipeline
from utils.config_utils import parse_args, load_config, get_image_paths_from_dir, load_panel


def main():

    ## Parse the provided configuration file
    args = parse_args()
    config = load_config(args.config)

    dir_cfg = config.get('directories')
    model_cfg = config.get('model')
    train_cfg = config.get('training')
    preproc_cfg = config.get('preprocessing')
    aug_cfg = config.get('augmentation')

    ## Set the seed for all subsequent processes
    seed_everything(train_cfg.get('seed'), workers = True)


    # Initialize loggers and callbacks: TensorBoard, CSV, Checkpoints
    loggers = get_loggers(dir_cfg.get('logs'), model_cfg.get('name'))
    callbacks = get_callbacks()

    ## Initialize the augmentation pipeline and build the datasets
    data_transforms = build_augmentation_pipeline(aug_cfg)
    image_paths = get_image_paths_from_dir(dir_cfg.get('mcd_dir'))
    panel = load_panel(dir_cfg.get('panel'))

    # Datamodule initializes the train/test datasets and loaders
    datamodule = WSDataModule(
        image_paths = image_paths,
        patch_size = preproc_cfg.get('patch_size'),
        stride = preproc_cfg.get('stride'),
        preproc_cfg = preproc_cfg,
        transforms = data_transforms,
        panel = panel,
        batch_size = train_cfg.get('batch_size'),
        num_workers = train_cfg.get('num_workers')
    )
    datamodule.setup()

    #latent_dimensions = [64, 128, 256, 384]
    latent_dimensions = 64

    for latent_dim in latent_dimensions:
        model, result = train_cae(
            latent_dim = latent_dim,
            datamodule = datamodule,
            max_epochs = train_cfg.get('epochs'),
            loggers = loggers,
            callbacks = callbacks,
            strategy = "ddp_notebook"
        )

        # process the result, store the model if best (?)

    
    


if __name__ == "__main__":
    main()