"""
Script:          callbacks.py
Purpose:         Provides utilities for PyTorch Lightning Callbacks
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-15-2025

PyTorch Version: 2.7.1
"""

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import Callback

# Removed EarlyStopping for now, just want to run vanilla

def get_callbacks():
    # Wrapper for fetching all defined callbacks
    return [
        ModelCheckpoint(
            monitor = "val_loss",
            save_top_k = 1,
            mode = "min"
        ),
        LearningRateMonitor(
            logging_interval = "epoch"
        ),
        PrintCallback()
    ]


def get_checkpoint_callbacks(monitor = "val_loss", save_top_k = 1, mode ="min"):
    # i honestly dont remember what this is for i hope its not important
    return 


class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")