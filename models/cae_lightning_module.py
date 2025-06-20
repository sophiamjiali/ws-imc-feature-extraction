"""
Script:          cae_lightning_module.py
Purpose:         Defines the model and training logic as a Lightning Module
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-13-2025

PyTorch Version: 2.7.1
"""

import lightning as L
import torch
import torch.nn.functional as F


class CAELightningModule(L.LightningModule):

    def __init__(self, autoencoder, lr = 1e-3):
        super().__init__()
        self.autoencoder = autoencoder
        self.lr = lr
        self.save_hyperparameters(ignore = ['autoencoder'])

    def forward(self, x):
        return self.autoencoder(x)
    
    def _get_reconstruction_loss(self, batch):
        input, target = batch
        output = self(input)
        loss = F.mse_loss(output, target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        return loss

    def predict_step(self, batch, batch_idx):
        # Return the encoder features
        input, _ = batch
        return self.autoencoder.encoder(input)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
