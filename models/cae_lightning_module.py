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
import torch.nn as nn
from torchvision import models


class CAELightningModule(L.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        # from PyTorch wiki

        inputs, target = batch
        output = self.model(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs, target)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
    
    def test_step(self, batch, batch_idx):
        return

    def predict_step(self, batch, batch_idx):
        return
    
    def configure_optimizers():
        return torch.optim.SGD(self.model.parameters(),lr = 0.1)
        