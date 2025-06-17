"""
Script:          ws_datamodule.py
Purpose:         Defines the DataModule object for PyTorch Lightning
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-10-2025

PyTorch Version: 2.7.1
"""
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class WSDataModule(LightningDataModule):

    def __init__(
            self, 
            train_dataset, 
            val_dataset = None, 
            test_dataset = None, 
            predict_dataset = None,
            batch_size = 32,
            num_workers = 4,
            pin_memory = True,
            shuffle_train = True
        ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = self.shuffle_train,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            drop_last = True
        )
    
    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers, 
                pin_memory = self.pin_memory,
                drop_last = False
            )
        return None
    
    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers,
                pin_memory = self.pin_memory,
                drop_last = False
            )
        return None
    
    def predict_dataloader(self):
        if self.predict_dataset is not None:
            return DataLoader(
                self.predict_dataset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers,
                pin_memory = self.pin_memory,
                drop_last = False
            )
        return None