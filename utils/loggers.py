"""
Script:          loggers.py
Purpose:         Provides utilities for PyTorch Lightning Loggers
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-15-2025

PyTorch Version: 2.7.1
"""

from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger


def get_loggers(log_dir, name):
    # Initializes Tensorboard and CSV loggers for model training and evaluation

    return [
        TensorBoardLogger(
            save_dir = log_dir.get('tensorboard', ''),
            name = name
        ),
        CSVLogger(
            save_dir = log_dir.get('csv', ''),
            name = name
        )
    ]

