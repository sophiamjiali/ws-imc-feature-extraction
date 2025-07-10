"""
Script:          loggers.py
Purpose:         Provides utilities for PyTorch Lightning Loggers
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-15-2025

PyTorch Version: 2.7.1
"""

from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, WandbLogger
import wandb


def get_loggers(log_dir, name, hyperparams = None):
    # Initializes Tensorboard and CSV loggers for model training and evaluation

    # Initialize and log into weights and biases (wandb)
    wandb.login()

    loggers = [
        TensorBoardLogger(
            save_dir = log_dir.get('tensorboard', ''),
            name = name
        ),
        CSVLogger(
            save_dir = log_dir.get('csv', ''),
            name = name
        ),
        WandbLogger(
            save_dir = log_dir.get('wandb', ''),
            log_model = "best"
        )
    ]

    # Log hyperparameters if provided
    if hyperparams is not None:
        for logger in loggers:
            if hasattr(logger, "log_hyperparams"):
                logger.log_hyperparams(hyperparams)

        # Update wandb config directly
        try:
            wandb.config.update(hyperparams)
        except Exception:
            print("Could not update wandb config for additional hyperparameters")

    return loggers
