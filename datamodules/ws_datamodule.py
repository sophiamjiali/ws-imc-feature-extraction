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
import os
import pickle

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
            num_workers = 8
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

    def _get_patch_coords_path(self, split):
        # Fetch the patch coords path
        return os.path.join("patch_coords", f"{split}_patch_coords.pkl")

    def _load_patch_coords(self, split):
        # Loads the patch coords dictionary
        path = self._get_patch_coords_path(split)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_patch_coords(self, split, coords):
        # Saves a patch coord dictionary
        os.makedirs("patch_coords", exist_ok=True)
        path = self._get_patch_coords_path(split)
        with open(path, "wb") as f:
            pickle.dump(coords, f)

    def setup(self, stage = None):
        # Initializes the train, validation, test, and predict datasets

        print("[DataModule] Beginning DataModule Setup ==================================")

        # Initialize the ratioed image paths for each dataset
        random.shuffle(self.image_paths)
        n = len(self.image_paths)
        train_paths = self.image_paths[:int(0.7 * n)]
        val_paths = self.image_paths[int(0.7 * n):int(0.85 * n)]
        test_paths = self.image_paths[int(0.85 * n):]
        
        splits = {
            "train": train_paths,
            "val": val_paths,
            "test": test_paths
        }

        # Check to see if the patch coords were already computed
        patch_coords = {}
        for split, paths in splits.items():
            coords = self._load_patch_coords(split)
            if coords is None:
                print(f"[DataModule] {split} patch coords is not cached, computing...")
                coords = self._parallel_precompute_patches(paths)
                self._save_patch_coords(split, coords)
            else:
                print(f"[DataModule] {split} patch coords are cached, loading...")
            patch_coords[split] = coords

        self.train_patch_coords = patch_coords["train"]
        self.val_patch_coords = patch_coords["val"]
        self.test_patch_coords = patch_coords["test"]

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

        print("[DataModule] Completed DataModule Setup ==================================")


    def train_dataloader(self):
        print("[DataModule] Initializing train dataloader")
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            drop_last = True,
            shuffle = False
        )
    
    def val_dataloader(self):
        if self.val_dataset is not None:
            print("[DataModule] Initializing evaluate dataloader")
            return DataLoader(
                self.val_dataset,
                batch_size = self.batch_size,
                num_workers = self.num_workers, 
                pin_memory = True,
                drop_last = True,
                shuffle = False
            )
        return None
    
    def test_dataloader(self):
        if self.test_dataset is not None:
            print("[DataModule] Initializing test dataloader")
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
            print("[DataModule] Initializing predict dataloader")
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
    
    def on_train_epoch_start(self):
        # Randomizes the patch orders, maintaining image order (already random)
        self.train_dataset.on_epoch_start()

    def on_validation_epoch_start(self):
        self.val_dataset.on_epoch_start()

    def on_test_epoch_start(self):
        self.test_dataset.on_epoch_start()