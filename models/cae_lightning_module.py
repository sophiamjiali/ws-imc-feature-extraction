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
import os
import numpy as np
import wandb


class CAELightningModule(L.LightningModule):

    def __init__(self, 
                 autoencoder = None, 
                 lr = 1e-3, 
                 log_patches = None, 
                 log_channels = None
                ):
        
        super().__init__()
        self.autoencoder = autoencoder
        self.lr = lr
        self.save_hyperparameters(ignore = ['autoencoder'])
        self.log_patches = log_patches
        self.log_channels = log_channels

    def forward(self, x):
        return self.autoencoder(x)
    
    def _get_reconstruction_loss(self, batch):
        input, target = batch
        output = self(input)
        loss = F.mse_loss(output, target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True, logger = True, sync_dist = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss, on_step = True, on_epoch = True, prog_bar = True, logger = True, sync_dist = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss, on_step = True, on_epoch = True, prog_bar = True, logger = True, sync_dist = True)
        return loss

    def predict_step(self, batch, batch_idx):
        # Return the encoder features
        input, _ = batch
        return self.autoencoder.encoder(input)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def on_train_epoch_end(self):
        #self._log_reconstructions('train')
        pass

    def on_validation_epoch_end(self):
        self._log_reconstructions('validation')

    def _log_reconstructions(self, step):
        # Saves the predefined reconstructed patches to the provided logger
    
        reconstructed_patches = []
        for patch_path in self.log_patches:
    
            # Load and conver tthe patch to a tensor for model input
            patch = np.load(patch_path)
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)
    
            # Move the patch to the same device as the current model
            device = next(self.parameters()).device
            patch_tensor = patch_tensor.to(device)
    
            # Reconstruct the patch
            with torch.no_grad():
                reconstructed = self(patch_tensor)
    
            # Move to CPU and keep only the specified logging channels
            reconstructed = reconstructed.detach().cpu().numpy()
            reconstructed = reconstructed[0, self.log_channels, :, :]
            reconstructed = np.transpose(reconstructed, (1, 2, 0))

            # Scale the image from float to integer for RGB input
            reconstructed = ((reconstructed / reconstructed.max()) * 255).astype(np.uint8)

            # Stack the input and output images
            patch = patch[self.log_channels, :, :]
            patch = np.transpose(patch, (1, 2, 0))
            patch = ((patch / patch.max()) * 255).astype(np.uint8)

            pair_img = np.concatenate([patch, reconstructed], axis = 1)
                                 
            # Append the reconstructed image for logging
            filename = os.path.basename(patch_path)
            reconstructed_patches.append((filename, pair_img))
    
        # Log the reconstructed image to wandb
        images = [wandb.Image(img, caption = f"Patch {filename} - Left: Input, Right: Output") for filename, img in reconstructed_patches]
        wandb.log({f"{step} reconstructions": images})
            
