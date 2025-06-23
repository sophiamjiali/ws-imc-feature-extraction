"""
Script:          ws_datamodule.py
Purpose:         Defines the DataModule object for PyTorch Lightning
Author:          Sophia Li
Affiliation:     Campbell Lab
Date:            06-10-2025

PyTorch Version: 2.7.1
"""
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from utils.config_utils import load_image_and_markers
from src.preprocess import preprocess_image, extract_patch, has_sufficient_content
from datasets.ws_dataset import WSDataset
import random
from multiprocessing import Pool

class WSDataModule(LightningDataModule):

    def __init__(
            self, 
            image_paths = None,
            patch_size = (200, 200), 
            stride = (200, 200),
            preproc_cfg = None,
            transforms = None,
            panel = None,
            batch_size = 32,
            num_workers = 4
        ):
        super().__init__()
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.stride = stride
        self.preproc_cfg = preproc_cfg
        self.transforms = transforms
        self.panel = panel
        self.batch_size = batch_size
        self.num_workers = num_workers
        

    def _precompute_patches(self, args):
        # Precomputes valid patch coordinates for each image

        # Load and preprocess the image
        img_idx, img_path = args
            
        img, markers = load_image_and_markers(img_path)
        img = preprocess_image(img, markers, self.panel, self.preproc_cfg)
        H, W = img.shape[-2:]

        # Compute and quality-control each possible patch of the image
        patch_coords = []
        for y in range(0, H, self.stride[0]):
            for x in range(0, W, self.stride[1]):

                # Extract and pad the patch if necessary
                patch = extract_patch(img, self.patch_size, (y, x), (H, W))
                
                # Screen for if the patch has sufficient biological content
                if has_sufficient_content(patch, self.preproc_cfg.get('bio_content_threshold')):
                    patch_coords.append((y, x))

        return img_idx, patch_coords
    

    def _parallel_precompute_patches(self, image_paths):
        # Parallelization adapter for patch precomputation

        args = list(enumerate(image_paths))
        with Pool() as pool:
            results = pool.map(self._precompute_patches, args)
        return dict(results)


    def setup(self, stage = None):
        # Initializes the train, validation, test, and predict datasets

        random.shuffle(self.image_paths)
        n = len(self.image_paths)
        train_paths = self.image_paths[:int(0.7 * n)]
        val_paths = self.image_paths[int(0.7 * n):int(0.85 * n)]
        test_paths = self.image_paths[int(0.85 * n):]

        self.train_patch_coords = self._parallel_precompute_patches(train_paths)
        self.val_patch_coords = self._parallel_precompute_patches(val_paths)
        self.test_patch_coords = self._parallel_precompute_patches(test_paths)

        self.train_dataset = WSDataset(
            image_paths = train_paths,
            patch_coords = self.train_patch_coords, 
            patch_size = self.patch_size,
            stride = self.stride,
            transforms = self.transforms,
            panel = self.panel,
            preproc_cfg = self.preproc_cfg
        )
        self.val_dataset = WSDataset(
            image_paths = val_paths,
            patch_coords = self.val_patch_coords, 
            patch_size = self.patch_size,
            stride = self.stride,
            transforms = self.transforms,
            panel = self.panel,
            preproc_cfg = self.preproc_cfg
        )
        self.test_dataset = WSDataset(
            image_paths = test_paths,
            patch_coords = self.test_patch_coords, 
            patch_size = self.patch_size,
            stride = self.stride,
            transforms = self.transforms,
            panel = self.panel,
            preproc_cfg = self.preproc_cfg
        )
        self.predict_dataset = None


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            drop_last = True
        )
    
    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size = self.batch_size,
                num_workers = self.num_workers, 
                pin_memory = True,
                drop_last = True
            )
        return None
    
    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size = self.batch_size,
                num_workers = self.num_workers,
                pin_memory = True,
                drop_last = False
            )
        return None
    
    def predict_dataloader(self):
        if self.predict_dataset is not None:
            return DataLoader(
                self.predict_dataset,
                batch_size = self.batch_size,
                num_workers = self.num_workers,
                pin_memory = True,
                drop_last = False
            )
        return None
    
    def on_train_epoch_start(self):
        # Randomizes the patch orders
        self.train_dataset.on_epoch_start()

    def on_validation_epoch_start(self):
        self.val_dataset.on_epoch_start()

    def on_test_epoch_start(self):
        self.test_dataset.on_epoch_start()