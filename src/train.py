"""
Script:          train.py
Purpose:         Defines the training of the model
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-10-2025

PyTorch Version: 2.7.1
"""

from pytorch_lightning import Trainer
from models.cae_resnet import ConvAutoencoder, ResNetEncoder, Decoder


def train_cae(
        latent_dim = 64,
        datamodule = None,
        max_epochs = 100,
        loggers = None,
        callbacks = None
    ):

    ## Initialize the Trainer with the provided parameters 
    trainer = Trainer(
        devices = 2,
        accelerator = "gpu",
        strategy = "ddp",
        benchmark = True,        # Fixed input size, speeds up training
        callbacks = PrintCallback()
    )

    
    ## Estimate the initial learning rate
    lr_finder = trainer.tuner.lr_find(model, datamodule)




     # Initialize the Autoencoder model
    encoder = ResNetEncoder()
    decoder = Decoder()
    cae = ConvAutoencoder(encoder, decoder)
    model = CAELightningModule(cae)

    # Initialize the trainer
    trainer = Trainer(
        max_epochs = 100,
        accelerator = "gpu",
        devices = 1,
        log_every_n_steps = 10,
        default_root_dir = "outputs"
    )

    trainer.fit(model, datamodule)



