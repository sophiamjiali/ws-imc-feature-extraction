"""
Script:          callbacks.py
Purpose:         Provides utilities for PyTorch Lightning Callbacks
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-15-2025

PyTorch Version: 2.7.1
"""

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, LearningRateFinder
from lightning.pytorch import Callback

# Removed EarlyStopping for now, just want to run vanilla

def get_callbacks(checkpoints_path, args):
    # Wrapper for fetching all defined callbacks

    epochs, learning_rate, batch_size, seed = args
    
    return [
        ModelCheckpoint(
            monitor = "val_loss",
            save_top_k = 1,
            mode = "min",
            dirpath = checkpoints_path,
            filename = f"e{epochs}_lr{learning_rate}_bs{batch_size}_s{seed}-{{epoch}}_{{val_loss:.5f}}"
        ),
        LearningRateMonitor(
            logging_interval = "epoch"
        ),
        PrintCallback()
    ]

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("[PrintCallback] Training is started ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    def on_train_end(self, trainer, pl_module):
        print("[PrintCallback]: Training is done ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")