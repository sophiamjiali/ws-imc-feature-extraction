"""
Script:          main.py
Purpose:         Trains and logs the CAE given the provided config file
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-25-2025

PyTorch Version: 2.7.1
"""

from lightning import seed_everything
from lightning.pytorch import Trainer
import os
import wandb
import argparse

from utils.extract_patches import extract_and_save_patches
from utils.config_utils import load_config, get_image_paths_from_dir, load_panel, fetch_marker_indices
from utils.callbacks import get_callbacks
from utils.loggers import get_loggers
from utils.extract_patches import extract_and_save_patches

from datamodules.ws_datamodule_cached import WSDataModuleCached

from models.cae_resnet import ResNetEncoder, Decoder, ConvAutoencoder
from models.cae_lightning_module import CAELightningModule

def main():

    ## Parse command line arguments
    parser = argparse.ArgumentParser(description="Train CAE")
    parser.add_argument('--epochs', type = int, default = 500, help = "Number of epochs")
    parser.add_argument('--learning_rate', type = int, default = 0.001, help = 'Batch size')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch size')
    parser.add_argument('--seed', type = int, default = 42, help = 'Seed')
    args = parser.parse_args()

    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    seed = args.seed

    ## Load and parse the configuration file
    config = load_config('config/config.yaml')
    
    dir_cfg = config.get('directories')
    model_cfg = config.get('model')
    train_cfg = config.get('training')
    preproc_cfg = config.get('preprocessing')
    aug_cfg = config.get('augmentation')
    log_cfg = config.get('logging')

    ## Set the seed for all subsequent processes
    seed_everything(seed, workers = True)

    ## Build the datamodule
    patch_paths = get_image_paths_from_dir(dir_cfg.get('patch_dir'), {'.npy'})
    panel = load_panel(dir_cfg.get('panel'))
    
    datamodule = WSDataModuleCached(
        image_paths = patch_paths,
        patch_size = preproc_cfg.get('patch_size'),
        stride = preproc_cfg.get('stride'),
        preproc_cfg = preproc_cfg,
        panel = panel,
        batch_size = batch_size,
        num_workers = train_cfg.get('num_workers')
    )

    ## Initialize loggers and callbacks: TensorBoard, CSV, Checkpoints, wandb
    hyperparams = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed
    }
    loggers = get_loggers(dir_cfg.get('logs'), model_cfg.get('name'), hyperparams)
    callbacks = get_callbacks(
        dir_cfg.get('checkpoints'), 
        (epochs, learning_rate, batch_size, seed)
    )

    ## Initialize patch reconstruction logging
    log_channels = fetch_marker_indices(log_cfg.get('log_channels'), panel)
    log_patches = [os.path.join(dir_cfg.get('patch_dir'), patch) for patch in log_cfg.get('log_patches')]

    ## Initialize the model and trainer
    encoder = ResNetEncoder()
    decoder = Decoder()
    cae = ConvAutoencoder(encoder, decoder)
    model = CAELightningModule(
        autoencoder = cae, 
        lr = learning_rate,
        log_patches = log_patches,
        log_channels = log_channels
    )
    
    trainer = Trainer(
        devices = train_cfg.get('num_devices'),
        max_epochs = epochs,
        accelerator = "auto",
        benchmark = False,
        logger = loggers,
        callbacks = callbacks,
        log_every_n_steps = log_cfg.get('log_step_n'),
        deterministic = True
    )

    ## Fit the data
    trainer.fit(model, datamodule)

    ## Log the most optimal validation loss achieved
    best_val_loss = callbacks[0].best_model_score.item()
    wandb.run.summary["best_val_loss"] = best_val_loss

    ## Finalize logging
    wandb.finish()



if __name__ == "__main__":
    main()