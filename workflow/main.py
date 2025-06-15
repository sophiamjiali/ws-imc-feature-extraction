"""
Script:          main.py
Purpose:         Entry-point for training the model
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-10-2025

PyTorch Version: 2.7.1
"""


from datamodules.ws_datamodule import WSDataModule
from models.cae_lightning_module import CAELightningModule
import pytorch_lightning as pl



from src.train import train_cae
from utils.loggers import get_loggers
from utils.callbacks import get_callbacks

from augmentations import build_augmentation_pipeline
from utils.config_utils import parse_args, load_config, get_image_paths_from_dir, build_datasets


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
    pl.seed_everything(train_cfg.get('seed'), workers = True)

    # Initialize loggers and callbacks: TensorBoard, CSV, Checkpoints
    loggers = get_loggers(dir_cfg.get('logs'), model_cfg.get('name'))
    callbacks = get_callbacks()

    ## Initialize the augmentation pipeline and build the datasets
    data_transforms = build_augmentation_pipeline(config.get['augmentation'])
    image_paths = get_image_paths_from_dir(dir_cfg.get('processed_dir'))

    train_dataset, val_dataset, test_dataset = build_datasets(
        image_paths = image_paths,
        patch_size = preproc_cfg.get('patch_size'),
        transforms = data_transforms
    )

    datamodule = WSDataModule(
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        test_dataset = test_dataset,
        predict_dataset = None,
        batch_size = 32,
        num_workers = 8
    )


    for latent_dim in [64, 128, 256, 384]:
        model, result = train_cae(
            latent_dim = latent_dim,
            datamodule = datamodule
            max_epochs = train_cfg.get('epochs'),
            data_transorms = data_transforms
            loggers = loggers,
            callbacks = callbacks,
            
        )









    

   

    # Evaluate the model
    trainer.test(model, datamodule = datamodule)
    


if __name__ == "__main__":
    main()