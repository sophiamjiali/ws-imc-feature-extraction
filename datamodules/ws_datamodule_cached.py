"""
Script:          ws_datamodule_cached.py
Purpose:         Datamodule version with pre-saved patches
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-10-2025

PyTorch Version: 2.7.1
"""

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from datasets.ws_dataset_cached import WSDatasetCached
import random

class WSDataModuleCached(LightningDataModule):

    def __init__(self,
                 image_paths = None,
                 patch_size = [200,200],
                 stride = [200,200],
                 preproc_cfg = None,
                 panel = None,
                 batch_size = 32,
                 num_workers = 4
                 ):
        super().__init__()
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.stride = stride
        self.preproc_cfg = preproc_cfg
        self.panel = panel
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):
        # Initializes the dataset splits

        ## Shuffle and divide images by predefined ratios 
        random.shuffle(self.image_paths)
        n = len(self.image_paths)
        train_paths = self.image_paths[:int(0.70 * n)]
        val_paths = self.image_paths[int(0.70 * n):]

        self.train_dataset = WSDatasetCached(train_paths)
        self.val_dataset = WSDatasetCached(val_paths)
        self.test_dataset = None
        self.predict_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            drop_last = True,
            shuffle = True
        )

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size = self.batch_size,
                num_workers = self.num_workers, 
                pin_memory = True,
                drop_last = False,
                shuffle = False
            )
        return None

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size = self.batch_size,
                num_workers = self.num_workers,
                pin_memory = True,
                drop_last = False,
                shuffle = False
            )
        return None

    def predict_dataloader(self):
        if self.predict_dataset is not None:
            return DataLoader(
                self.predict_dataset,
                batch_size = self.batch_size,
                num_workers = self.num_workers,
                worker_init_fn = worker_init_fn,
                pin_memory = True,
                drop_last = False,
                shuffle = False
            )
        return None    


        