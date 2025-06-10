"""
Script:          ws_datamodule.py
Purpose:         Defines the LightningDataModule for WS data
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-10-2025

PyTorch Version: 2.7.1
"""

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets.ws_dataset import WSDataset

class WSDataModule(LightningDataModule):