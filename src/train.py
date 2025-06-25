"""
Script:          train.py
Purpose:         Defines the training of the model
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-10-2025

PyTorch Version: 2.7.1
"""

from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from models.cae_resnet import ConvAutoencoder, ResNetEncoder, Decoder
from models.cae_lightning_module import CAELightningModule


def train_cae(
        latent_dim = 64,
        datamodule = None,
        max_epochs = 100,
        loggers = None,
        callbacks = None,
        strategy = "ddp"
    ):

     ## Initialize the Autoencoder model with a default learning rate
    encoder = ResNetEncoder()
    decoder = Decoder()
    cae = ConvAutoencoder(encoder, decoder)
    model = CAELightningModule(cae)

    ## Initialize the Trainer with the provided parameters 
    trainer = Trainer(
        devices = 1,
        max_epochs = 2,
        accelerator = "auto",
        strategy = strategy,
        benchmark = True,        # Fixed input size, speeds up training
        logger = loggers,
        callbacks = callbacks,
        log_every_n_steps = 10
    )

    ## Tune the trainer for the optimal learning rate
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model)
    new_lr = lr_finder.suggestion()
    model.lr = new_lr

    ## Fit the data
    trainer.fit(model, datamodule)

    ## Evaluate the model and return the metrics
    return model, trainer.test(model, datamodule)